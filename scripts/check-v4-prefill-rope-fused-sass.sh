#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cuda_root_path=${CUDA_ROOT_PATH:-/usr/local/cuda}
nvcc_bin=${NVCC_BIN:-"$cuda_root_path/bin/nvcc"}
cuobjdump_bin=${CUOBJDUMP_BIN:-"$cuda_root_path/bin/cuobjdump"}
nvdisasm_bin=${NVDISASM_BIN:-"$cuda_root_path/bin/nvdisasm"}
source_file="$repo_root/kernels/gb10/experiments/v4_prefill_rope_fused.cu"
probe_dir=$(mktemp -d /tmp/atlas-v4-prefill-rope-fused.XXXXXX)
trap 'rm -rf -- "$probe_dir"' EXIT

for tool in "$nvcc_bin" "$cuobjdump_bin" "$nvdisasm_bin"; do
    if [[ ! -x $tool ]]; then
        echo "missing CUDA tool: $tool" >&2
        exit 2
    fi
done
for injected_flags in NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS; do
    if [[ -n ${!injected_flags:-} ]]; then
        echo "refusing unreceipted nvcc flags from $injected_flags" >&2
        exit 2
    fi
done
if [[ ! -f $source_file ]]; then
    echo "missing isolated V4 fused RoPE source: $source_file" >&2
    exit 1
fi

cubin="$probe_dir/v4-prefill-rope-fused.cubin"
log="$probe_dir/v4-prefill-rope-fused.log"
resources="$probe_dir/v4-prefill-rope-fused.resources"
sass="$probe_dir/v4-prefill-rope-fused.sass"

"$nvcc_bin" -std=c++17 -O3 --fmad=false -arch=sm_121a -cubin "$source_file" \
    -o "$cubin" -Xptxas=-v 2>"$log"
"$cuobjdump_bin" --dump-resource-usage "$cubin" >"$resources"
"$nvdisasm_bin" "$cubin" >"$sass"

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
    ' "$resources"
}

extract_function() {
    local symbol=$1
    local output=$2
    awk -v symbol="$symbol" '
        emit && /^\/\/--------------------- \.text\./ { exit }
        $0 == symbol ":" { emit = 1 }
        emit { print }
    ' "$sass" >"$output"
}

opcode_count() {
    local file=$1
    local pattern=$2
    grep -cE "$pattern" "$file" || true
}

check_function() {
    local symbol=$1
    local expected_instructions=$2
    local expected_registers=$3
    local function_sass="$probe_dir/${symbol}.sass"
    extract_function "$symbol" "$function_sass"

    local registers stack local_bytes shared instructions guard_exit_line first_global_load_line
    registers=$(resource_value "$symbol" REG)
    stack=$(resource_value "$symbol" STACK)
    local_bytes=$(resource_value "$symbol" LOCAL)
    shared=$(resource_value "$symbol" SHARED)
    instructions=$(grep -cE '^[[:space:]]*/\*[0-9a-f]+\*/' "$function_sass" || true)
    guard_exit_line=$(grep -n -m1 -E '@P[0-9]+ EXIT' "$function_sass" | cut -d: -f1)
    first_global_load_line=$(grep -n -m1 -E '[[:space:]]LDG\.E' "$function_sass" | cut -d: -f1)

    [[ $registers == "$expected_registers" ]]
    [[ $stack == 0 ]]
    [[ $local_bytes == 0 ]]
    [[ $shared == 0 ]]
    [[ $instructions == "$expected_instructions" ]]
    [[ -n $guard_exit_line && -n $first_global_load_line ]]
    [[ $guard_exit_line -lt $first_global_load_line ]]
    grep -A1 "Function properties for $symbol" "$log" \
        | grep -q '0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads'
    [[ $(opcode_count "$function_sass" 'F2FP\.BF16\.F32\.PACK_AB') == 1 ]]
    [[ $(opcode_count "$function_sass" '[[:space:]]FFMA([[:space:]]|\.)') == 13 ]]
    [[ $(opcode_count "$function_sass" '[[:space:]]FMUL([[:space:]]|\.)') == 12 ]]
    [[ $(opcode_count "$function_sass" '[[:space:]]FADD([[:space:]]|\.)') == 4 ]]
    [[ $(opcode_count "$function_sass" '[[:space:]]MUFU\.(COS|SIN)') == 0 ]]
    [[ $(opcode_count "$function_sass" '[[:space:]](ATOM|RED|LDL|STL)(\.|[[:space:]])') == 0 ]]
    echo "$symbol: instructions=$instructions registers=$registers shared=$shared bf16_pack=1 guard_before_global=1 stack=0 local=0 spills=0 atomics=0"
}

check_function v4_prefill_rope_fused_forward 338 25
check_function v4_prefill_rope_fused_inverse 328 23

digest=$(sha256sum "$cubin" | awk '{print $1}')
echo "V4 fused RoPE: exact 32-thread pair ownership, precise cosf/sinf, BF16 boundary"
echo "cubin_set_sha256=$digest"
