#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cuda_root_path=${CUDA_ROOT_PATH:-/usr/local/cuda}
nvcc_bin=${NVCC_BIN:-"$cuda_root_path/bin/nvcc"}
cuobjdump_bin=${CUOBJDUMP_BIN:-"$cuda_root_path/bin/cuobjdump"}
nvdisasm_bin=${NVDISASM_BIN:-"$cuda_root_path/bin/nvdisasm"}
probe_dir=$(mktemp -d /tmp/atlas-exl3-dual-pre.XXXXXX)
trap 'rm -rf -- "$probe_dir"' EXIT

for tool in "$nvcc_bin" "$cuobjdump_bin" "$nvdisasm_bin"; do
    if [[ ! -x "$tool" ]]; then
        echo "missing CUDA tool: $tool" >&2
        exit 2
    fi
done

cubin="$probe_dir/exl3_gemv_k2.cubin"
"$nvcc_bin" -std=c++17 -arch=sm_121a \
    -I "$repo_root/kernels/gb10/common" \
    -cubin "$repo_root/kernels/gb10/common/exl3_gemv_k2.cu" \
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

dual_symbol=exl3_h128_pre_dual_rows_h4096
single_symbol=exl3_h128_pre_rows
extract_function "$dual_symbol" "$probe_dir/dual.sass"
extract_function "$single_symbol" "$probe_dir/single.sass"

dual_instructions=$(instruction_count "$probe_dir/dual.sass")
single_instructions=$(instruction_count "$probe_dir/single.sass")
dual_registers=$(resource_value "$dual_symbol" REG)
single_registers=$(resource_value "$single_symbol" REG)

[[ -n "$dual_registers" && -n "$single_registers" ]]
(( dual_registers <= 33 ))
(( single_registers <= 19 ))
(( dual_instructions < 2 * single_instructions ))

for field in STACK SHARED LOCAL; do
    [[ $(resource_value "$dual_symbol" "$field") == 0 ]]
done
grep -q '0 bytes spill stores, 0 bytes spill loads' "$probe_dir/ptxas.log"

declare -A expected_dual=(
    [FFMA]=48
    [SHFL.BFLY]=40
    [FMUL]=12
    [F2FP.BF16.F32.PACK_AB]=4
    [LDG.E.U16.CONSTANT]=4
    [STG.E]=4
    [STG.E.U16]=0
)
declare -A expected_single=(
    [FFMA]=24
    [SHFL.BFLY]=20
    [FMUL]=6
    [F2FP.BF16.F32.PACK_AB]=4
    [LDG.E.U16.CONSTANT]=4
    [STG.E.U16]=4
)
for mnemonic in "${!expected_dual[@]}"; do
    [[ $(mnemonic_count "$probe_dir/dual.sass" "$mnemonic") == "${expected_dual[$mnemonic]}" ]]
done
for mnemonic in "${!expected_single[@]}"; do
    [[ $(mnemonic_count "$probe_dir/single.sass" "$mnemonic") == "${expected_single[$mnemonic]}" ]]
done

cubin_sha256=$(sha256sum "$cubin" | awk '{print $1}')
echo "dual: instructions=$dual_instructions registers=$dual_registers spills=0 stack=0 shared=0 local=0"
echo "single: instructions=$single_instructions registers=$single_registers"
echo "dual input loads=4; packed stores=4; two singles input loads=8 stores=8; cubin_sha256=$cubin_sha256"
