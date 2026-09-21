#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cuda_root_path=${CUDA_ROOT_PATH:-/usr/local/cuda}
nvcc_bin=${NVCC_BIN:-"$cuda_root_path/bin/nvcc"}
cuobjdump_bin=${CUOBJDUMP_BIN:-"$cuda_root_path/bin/cuobjdump"}
nvdisasm_bin=${NVDISASM_BIN:-"$cuda_root_path/bin/nvdisasm"}
probe_dir=$(mktemp -d /tmp/atlas-exl3-w2a8-n128.XXXXXX)
trap 'rm -rf -- "$probe_dir"' EXIT

for tool in "$nvcc_bin" "$cuobjdump_bin" "$nvdisasm_bin"; do
    if [[ ! -x $tool ]]; then
        echo "missing CUDA tool: $tool" >&2
        exit 2
    fi
done

source_file="$repo_root/kernels/gb10/experiments/exl3_w2a8_grouped_prefill_n128.cu"

resource_value() {
    local file=$1
    local symbol=$2
    local field=$3
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

instruction_count() {
    awk '/^[[:space:]]*\/\*[0-9a-f]+\*\// { count++ } END { print count + 0 }' "$1"
}

compile_and_check() {
    local kind=$1
    local fixed_n=$2
    local fixed_k=$3
    local symbol="exl3_w2a8_grouped_prefill_n128_${kind}"
    local cubin="$probe_dir/${kind}.cubin"
    local log="$probe_dir/${kind}.log"
    local resources="$probe_dir/${kind}.resources"
    local sass="$probe_dir/${kind}.sass"

    "$nvcc_bin" -std=c++17 -O3 --fmad=false -arch=sm_121a \
        -DW2A8_FIXED_N="$fixed_n" -DW2A8_FIXED_K="$fixed_k" \
        -DW2A8_KERNEL_NAME="$symbol" -cubin "$source_file" \
        -o "$cubin" -Xptxas=-v 2>"$log"
    "$cuobjdump_bin" --dump-resource-usage "$cubin" >"$resources"
    "$nvdisasm_bin" "$cubin" >"$sass"

    local registers stack local_bytes shared instructions
    registers=$(resource_value "$resources" "$symbol" REG)
    stack=$(resource_value "$resources" "$symbol" STACK)
    local_bytes=$(resource_value "$resources" "$symbol" LOCAL)
    shared=$(resource_value "$resources" "$symbol" SHARED)
    instructions=$(instruction_count "$sass")
    [[ $registers == 96 ]]
    [[ $stack == 0 ]]
    [[ $local_bytes == 0 ]]
    [[ $shared == 8192 ]]
    [[ $instructions == 1016 ]]
    grep -A1 "Function properties for $symbol" "$log" \
        | grep -q '0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads'
    grep -q 'Used 96 registers, used 1 barriers, 7168 bytes smem' "$log"
    [[ $(grep -c 'QMMA.16832.F32.E4M3.E4M3' "$sass" || true) == 16 ]]
    [[ $(grep -c 'SHFL.IDX' "$sass" || true) == 16 ]]
    [[ $(grep -c 'F2FP.SATFINITE.E4M3' "$sass" || true) == 16 ]]
    [[ $(grep -c 'BAR.SYNC' "$sass" || true) == 3 ]]
    [[ $(grep -c 'HMMA' "$sass" || true) == 0 ]]
    [[ $(grep -cE '[[:space:]](ATOM|RED|LDL|STL)' "$sass" || true) == 0 ]]
    echo "$symbol: instructions=$instructions registers=$registers shared=$shared barriers=1 stack=0 local=0 spills=0"
}

compile_and_check gu 2048 4096
compile_and_check down 4096 2048

expect_compile_failure() {
    local fixed_n=$1
    local fixed_k=$2
    local label=$3
    if "$nvcc_bin" -std=c++17 -O3 --fmad=false -arch=sm_121a \
        -DW2A8_FIXED_N="$fixed_n" -DW2A8_FIXED_K="$fixed_k" \
        -DW2A8_KERNEL_NAME="exl3_w2a8_n128_invalid_${label}" \
        -cubin "$source_file" -o "$probe_dir/invalid-${label}.cubin" \
        >/dev/null 2>&1; then
        echo "invalid W2A8 N128 shape ${fixed_n}x${fixed_k} unexpectedly compiled" >&2
        exit 1
    fi
}

expect_compile_failure 3072 4096 unsupported_n
expect_compile_failure 2048 2048 crossed_gu
expect_compile_failure 4096 4096 crossed_down

digest=$(sha256sum "$probe_dir"/{down,gu}.cubin | awk '{print $1}' | sha256sum | awk '{print $1}')
echo "W2A8 N128 GU/down: exact 256-thread N128 strip, 16 native E4M3 QMMA instructions"
echo "cubin_set_sha256=$digest"
