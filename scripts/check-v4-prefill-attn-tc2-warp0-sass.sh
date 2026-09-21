#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cuda_root_path=${CUDA_ROOT_PATH:-/usr/local/cuda}
nvcc_bin=${NVCC_BIN:-"$cuda_root_path/bin/nvcc"}
cuobjdump_bin=${CUOBJDUMP_BIN:-"$cuda_root_path/bin/cuobjdump"}
nvdisasm_bin=${NVDISASM_BIN:-"$cuda_root_path/bin/nvdisasm"}
probe_dir=$(mktemp -d /tmp/atlas-v4-tc2-warp0.XXXXXX)
trap 'rm -rf -- "$probe_dir"' EXIT

for tool in "$nvcc_bin" "$cuobjdump_bin" "$nvdisasm_bin"; do
    [[ -x $tool ]] || { echo "missing CUDA tool: $tool" >&2; exit 2; }
done
for injected_flags in NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS; do
    [[ -z ${!injected_flags:-} ]] || {
        echo "refusing injected CUDA flags from $injected_flags" >&2
        exit 2
    }
done

compile() {
    local label=$1
    local source=$2
    "$nvcc_bin" -std=c++17 -O3 --fmad=false -arch=sm_121a -cubin "$source" \
        -o "$probe_dir/$label.cubin" -Xptxas=-v 2>"$probe_dir/$label.log"
    "$cuobjdump_bin" --dump-resource-usage "$probe_dir/$label.cubin" >"$probe_dir/$label.resources"
    "$nvdisasm_bin" "$probe_dir/$label.cubin" >"$probe_dir/$label.sass"
}

compile baseline "$repo_root/kernels/gb10/deepseek-v4-flash/nvfp4/prefill_attn_compressed.cu"
compile warp0 "$repo_root/kernels/gb10/deepseek-v4-flash/nvfp4/v4_prefill_attn_compressed_tc2_warp0.cu"

resource_value() {
    local file=$1 symbol=$2 field=$3
    awk -v symbol="$symbol" -v field="$field" '
        $0 == " Function " symbol ":" { getline; line = $0 }
        END {
            count = split(line, parts, " ")
            for (i = 1; i <= count; i++) {
                split(parts[i], pair, ":")
                if (pair[1] == field) { print pair[2]; exit }
            }
        }
    ' "$file"
}

extract_function() {
    local input=$1 symbol=$2 output=$3
    awk -v symbol="$symbol" '
        emit && /^\/\/--------------------- \.text\./ { exit }
        $0 == symbol ":" { emit = 1 }
        emit { print }
    ' "$input" >"$output"
}

opcode_count() {
    grep -c "$2" "$1" || true
}

check_kernel() {
    local label=$1 symbol=$2 expected_registers=$3 expected_shared=$4 expected_instructions=$5
    local expected_mufu=$6 expected_lds=$7 expected_sts=$8
    local resources="$probe_dir/$label.resources"
    local function_sass="$probe_dir/$label.function.sass"
    extract_function "$probe_dir/$label.sass" "$symbol" "$function_sass"
    local registers stack local_bytes shared instructions rounded_registers
    registers=$(resource_value "$resources" "$symbol" REG)
    stack=$(resource_value "$resources" "$symbol" STACK)
    local_bytes=$(resource_value "$resources" "$symbol" LOCAL)
    shared=$(resource_value "$resources" "$symbol" SHARED)
    instructions=$(grep -cE '^[[:space:]]*/\*[0-9a-f]+\*/' "$function_sass" || true)
    rounded_registers=$(( (registers * 128 + 255) / 256 * 256 ))
    [[ $registers == "$expected_registers" ]]
    [[ $stack == 0 && $local_bytes == 0 ]]
    [[ $shared == "$expected_shared" && $shared -le 24576 ]]
    [[ $rounded_registers -le 65536 ]]
    [[ $instructions == "$expected_instructions" ]]
    [[ $(opcode_count "$function_sass" 'MUFU.EX2') == "$expected_mufu" ]]
    [[ $(opcode_count "$function_sass" 'LDSM') == 32 ]]
    [[ $(opcode_count "$function_sass" 'HMMA') == 64 ]]
    [[ $(opcode_count "$function_sass" 'BAR.SYNC') == 8 ]]
    [[ $(opcode_count "$function_sass" 'F2FP.BF16') == 32 ]]
    [[ $(opcode_count "$function_sass" 'LDS') == "$expected_lds" ]]
    [[ $(opcode_count "$function_sass" 'STS') == "$expected_sts" ]]
    [[ $(grep -cE '[[:space:]](ATOM|RED|LDL|STL)(\.|[[:space:]])' "$function_sass" || true) == 0 ]]
    grep -A1 "Function properties for $symbol" "$probe_dir/$label.log" \
        | grep -q '0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads'
    if [[ $label == warp0 ]]; then
        # Both macro expansions branch threads 32..127 around their complete
        # softmax/MUFU regions; this is the compiled warp-0 ownership proof.
        [[ $(grep -c 'ISETP.GT.U32.AND .*0x1f,' "$function_sass" || true) == 2 ]]
        [[ $(grep -A2 'ISETP.GT.U32.AND .*0x1f,' "$function_sass" | grep -c '@P0 BRA' || true) == 2 ]]
    fi
    echo "$symbol: instructions=$instructions registers=$registers shared=$shared MUFU.EX2=$expected_mufu LDSM=32 HMMA=64 BAR.SYNC=8 stack=0 local=0 spills=0"
}

check_kernel baseline prefill_attn_compressed_tc2 158 22016 1832 24 48 48
check_kernel warp0 v4_prefill_attn_compressed_tc2_warp0 142 23552 2408 44 54 54

digest=$(sha256sum "$probe_dir"/{baseline,warp0}.cubin | awk '{print $1}' | sha256sum | awk '{print $1}')
echo "TC2 warp-0 production candidate: dynamic tile softmax is lane-broadcast; static opcode census is not a runtime estimate"
echo "cubin_set_sha256=$digest"
