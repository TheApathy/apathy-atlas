#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cuda_root_path=${CUDA_ROOT_PATH:-/usr/local/cuda}
nvcc_bin=${NVCC_BIN:-"$cuda_root_path/bin/nvcc"}
cuobjdump_bin=${CUOBJDUMP_BIN:-"$cuda_root_path/bin/cuobjdump"}
nvdisasm_bin=${NVDISASM_BIN:-"$cuda_root_path/bin/nvdisasm"}
probe_dir=$(mktemp -d /tmp/atlas-exl3-w2a8-n256.XXXXXX)
trap 'rm -rf -- "$probe_dir"' EXIT

for tool in "$nvcc_bin" "$cuobjdump_bin" "$nvdisasm_bin"; do
    if [[ ! -x $tool ]]; then
        echo "missing CUDA tool: $tool" >&2
        exit 2
    fi
done

source_file="$repo_root/kernels/gb10/experiments/exl3_w2a8_grouped_prefill_n256.cu"
threads=512
register_file_limit=65536
static_shared_limit=$((48 * 1024))

for contract in \
    '#define W2A8_N_TILE 256' \
    '#define W2A8_THREADS 512' \
    'uint4 smem_A[W2A8_M_TILE][5]' \
    'uint4 smem_T[4][W2A8_T_WORDS]' \
    'const unsigned int k_tile = load / W2A8_T_WORDS' \
    'const unsigned int word = load % W2A8_T_WORDS' \
    'routing_index < num_experts' \
    '__ballot_sync(0xffffffffu, routing_invalid)' \
    'const unsigned int col = n_base + warp * 16 + nt * 8 + tid * 2'; do
    grep -Fq "$contract" "$source_file"
done

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

instruction_count() {
    awk '/^[[:space:]]*\/\*[0-9a-f]+\*\// { count++ } END { print count + 0 }' "$1"
}

compile_and_check() {
    local kind=$1 fixed_n=$2 fixed_k=$3
    local symbol="exl3_w2a8_grouped_prefill_n256_${kind}"
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

    local registers stack local_bytes shared instructions register_block allocated_register_block
    registers=$(resource_value "$resources" "$symbol" REG)
    stack=$(resource_value "$resources" "$symbol" STACK)
    local_bytes=$(resource_value "$resources" "$symbol" LOCAL)
    shared=$(resource_value "$resources" "$symbol" SHARED)
    instructions=$(instruction_count "$sass")
    for value in "$registers" "$stack" "$local_bytes" "$shared" "$instructions"; do
        [[ $value =~ ^[0-9]+$ ]]
    done
    register_block=$((registers * threads))
    # SM121 allocates registers to each warp in 256-register quanta. Gate the
    # rounded physical allocation as well as reporting the nominal product.
    allocated_register_block=$(( ((registers * 32 + 255) / 256) * 256 * (threads / 32) ))
    (( allocated_register_block <= register_file_limit ))
    (( shared <= static_shared_limit ))
    [[ $registers == 97 ]]
    [[ $shared == 10240 ]]
    [[ $stack == 0 ]]
    [[ $local_bytes == 0 ]]
    [[ $instructions == 1093 ]]
    grep -A1 "Function properties for $symbol" "$log" \
        | grep -q '0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads'
    grep -q 'Used 97 registers, used 1 barriers, 9216 bytes smem' "$log"
    [[ $(grep -c 'QMMA.16832.F32.E4M3.E4M3' "$sass" || true) == 16 ]]
    [[ $(grep -c 'SHFL.IDX' "$sass" || true) == 16 ]]
    [[ $(grep -c 'F2FP.SATFINITE.E4M3' "$sass" || true) == 16 ]]
    [[ $(grep -c 'BAR.SYNC' "$sass" || true) == 3 ]]
    [[ $(grep -c 'HMMA' "$sass" || true) == 0 ]]
    [[ $(grep -cE '[[:space:]](ATOM|RED|LDL|STL)' "$sass" || true) == 0 ]]
    echo "$symbol: instructions=$instructions registers=$registers registers_per_block=$register_block allocated_registers_per_block=$allocated_register_block shared=$shared barriers=1 stack=0 local=0 spills=0"
}

compile_and_check gu 2048 4096
compile_and_check down 4096 2048

expect_compile_failure() {
    local fixed_n=$1 fixed_k=$2 label=$3
    if "$nvcc_bin" -std=c++17 -O3 --fmad=false -arch=sm_121a \
        -DW2A8_FIXED_N="$fixed_n" -DW2A8_FIXED_K="$fixed_k" \
        -DW2A8_KERNEL_NAME="exl3_w2a8_n256_invalid_${label}" \
        -cubin "$source_file" -o "$probe_dir/invalid-${label}.cubin" \
        >/dev/null 2>&1; then
        echo "invalid W2A8 N256 shape ${fixed_n}x${fixed_k} unexpectedly compiled" >&2
        exit 1
    fi
}

expect_compile_failure 3072 4096 unsupported_n
expect_compile_failure 2048 2048 crossed_gu
expect_compile_failure 4096 4096 crossed_down

digest=$(sha256sum "$probe_dir"/{down,gu}.cubin | awk '{print $1}' | sha256sum | awk '{print $1}')
echo "W2A8 N256 GU/down: exact 512-thread N256 strip, 16 native E4M3 QMMA instructions"
echo "resource ceilings: registers_per_block<=65536 shared<=49152 stack=0 local=0 spills=0"
echo "cubin_set_sha256=$digest"
