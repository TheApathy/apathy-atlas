#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cuda_root_path=${CUDA_ROOT_PATH:-/usr/local/cuda}
nvcc_bin=${NVCC_BIN:-"$cuda_root_path/bin/nvcc"}
cuobjdump_bin=${CUOBJDUMP_BIN:-"$cuda_root_path/bin/cuobjdump"}
nvdisasm_bin=${NVDISASM_BIN:-"$cuda_root_path/bin/nvdisasm"}
probe_dir=$(mktemp -d /tmp/atlas-exl3-shared-v2.XXXXXX)
trap 'rm -rf -- "$probe_dir"' EXIT

for tool in "$nvcc_bin" "$cuobjdump_bin" "$nvdisasm_bin"; do
    if [[ ! -x "$tool" ]]; then
        echo "missing CUDA tool: $tool" >&2
        exit 2
    fi
done

cubin="$probe_dir/w4a16_gemv.cubin"
"$nvcc_bin" -std=c++17 -arch=sm_121a \
    -cubin "$repo_root/kernels/gb10/common/w4a16_gemv.cu" \
    -o "$cubin" -Xptxas=-v 2>"$probe_dir/ptxas.log"
"$cuobjdump_bin" --dump-resource-usage "$cubin" >"$probe_dir/resources.txt"
"$nvdisasm_bin" "$cubin" >"$probe_dir/disasm.txt"

extract_function() {
    local symbol=$1
    local destination=$2
    awk -v marker=".text.${symbol} " '
        /^\/\/--------------------- \.text\./ {
            if (active) exit
            if (index($0, marker)) active = 1
        }
        active { print }
    ' "$probe_dir/disasm.txt" >"$destination"
}

resource_value() {
    local symbol=$1
    local field=$2
    awk -v symbol="$symbol" -v field="$field" '
        $0 == " Function " symbol ":" { getline; line = $0 }
        END {
            count = split(line, parts, " ")
            for (i = 1; i <= count; i++) {
                split(parts[i], pair, ":")
                if (pair[1] == field) { print pair[2]; exit }
            }
        }
    ' "$probe_dir/resources.txt"
}

instruction_count() {
    awk '/^[[:space:]]*\/\*[0-9a-f]+\*\// { count++ } END { print count + 0 }' "$1"
}

mnemonic_count() {
    local file=$1
    local mnemonic=$2
    grep -c "[[:space:]]${mnemonic}[[:space:]]" "$file" || true
}

incumbent=w4a16_gemv_grouped_batchm
v2_m6=w4a16_gemv_grouped_batchm_v2_m6
declare -A register_ceiling=(
    [w4a16_gemv_grouped_batchm]=76
    [w4a16_gemv_grouped_batchm_v2_m4]=80
    [w4a16_gemv_grouped_batchm_v2_m5]=80
    [w4a16_gemv_grouped_batchm_v2_m6]=80
    [w4a16_gemv_grouped_batchm_v2_m8]=90
    [w4a16_gemv_grouped_batchm_v2_m16]=148
)

for symbol in "${!register_ceiling[@]}"; do
    sass="$probe_dir/$symbol.sass"
    extract_function "$symbol" "$sass"
    [[ -s "$sass" ]]
    registers=$(resource_value "$symbol" REG)
    [[ -n "$registers" ]]
    (( registers <= register_ceiling[$symbol] ))
    for field in STACK LOCAL; do
        [[ $(resource_value "$symbol" "$field") == 0 ]]
    done
    grep -A1 "Function properties for $symbol" "$probe_dir/ptxas.log" \
        | grep -q '0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads'
done

incumbent_sass="$probe_dir/$incumbent.sass"
v2_m6_sass="$probe_dir/$v2_m6.sass"
incumbent_instructions=$(instruction_count "$incumbent_sass")
v2_m6_instructions=$(instruction_count "$v2_m6_sass")
(( v2_m6_instructions < incumbent_instructions ))

# The incumbent compiles eight guarded row bodies; m6 compiles six unguarded
# bodies. Equal per-row FFMA/FADD/shuffle counts are the arithmetic-order gate.
for mnemonic in FFMA FADD SHFL.DOWN; do
    incumbent_count=$(mnemonic_count "$incumbent_sass" "$mnemonic")
    v2_m6_count=$(mnemonic_count "$v2_m6_sass" "$mnemonic")
    (( incumbent_count * 6 == v2_m6_count * 8 ))
done
incumbent_branches=$(mnemonic_count "$incumbent_sass" BRA)
v2_m6_branches=$(mnemonic_count "$v2_m6_sass" BRA)
(( v2_m6_branches < incumbent_branches ))

cubin_sha256=$(sha256sum "$cubin" | awk '{print $1}')
echo "shared V2 m6: instructions=$v2_m6_instructions registers=$(resource_value "$v2_m6" REG) branches=$v2_m6_branches spills=0"
echo "runtime-M incumbent: instructions=$incumbent_instructions registers=$(resource_value "$incumbent" REG) branches=$incumbent_branches spills=0"
echo "per-row FFMA/FADD/shuffle counts match; cubin_sha256=$cubin_sha256"
