#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cuda_root_path=${CUDA_ROOT_PATH:-/usr/local/cuda}
nvcc_bin=${NVCC_BIN:-"$cuda_root_path/bin/nvcc"}
cuobjdump_bin=${CUOBJDUMP_BIN:-"$cuda_root_path/bin/cuobjdump"}
nvdisasm_bin=${NVDISASM_BIN:-"$cuda_root_path/bin/nvdisasm"}
source_file="$repo_root/kernels/gb10/experiments/exl3_w2a8_grouped_prefill_n256.cu"
probe_dir=$(mktemp -d /tmp/atlas-exl3-w2a8-n256-route-guard.XXXXXX)
trap 'rm -rf -- "$probe_dir"' EXIT
static_shared_limit=$((48 * 1024))

if [[ -n ${NVCC_PREPEND_FLAGS:-} || -n ${NVCC_APPEND_FLAGS:-} ]]; then
    echo "NVCC_PREPEND_FLAGS and NVCC_APPEND_FLAGS must be empty" >&2
    exit 2
fi
for tool in "$nvcc_bin" "$cuobjdump_bin" "$nvdisasm_bin"; do
    if [[ ! -x $tool ]]; then
        echo "missing CUDA tool: $tool" >&2
        exit 2
    fi
done

for contract in \
    '#define W2A8_N256_SINGLE_WARP_ROUTE_GUARD_CANDIDATE 0' \
    'static_assert(W2A8_N256_SINGLE_WARP_ROUTE_GUARD_CANDIDATE == 0 ||' \
    'w2a8_route_lane_invalid' \
    '__shared__ unsigned int route_invalid_block' \
    'if (threadIdx.x < 32)' \
    'if (threadIdx.x == 0) route_invalid_block = invalid_mask != 0' \
    '__syncthreads();' \
    'if (route_invalid_block != 0) return;'; do
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

compile_variant() {
    local name=$1 selector=$2
    local symbol="exl3_w2a8_grouped_prefill_n256_k2_down_${name}"
    local cubin="$probe_dir/${name}.cubin"
    local log="$probe_dir/${name}.log"
    local resources="$probe_dir/${name}.resources"
    local sass="$probe_dir/${name}.sass"

    "$nvcc_bin" -std=c++17 -O3 --fmad=false -arch=sm_121a \
        -DW2A8_FIXED_N=4096 -DW2A8_FIXED_K=2048 \
        -DW2A8_PACKED_E4M3_CANDIDATE=1 \
        -DW2A8_N256_SINGLE_WARP_ROUTE_GUARD_CANDIDATE="$selector" \
        -DW2A8_KERNEL_NAME="$symbol" -cubin "$source_file" \
        -o "$cubin" -Xptxas=-v 2>"$log"
    "$cuobjdump_bin" --dump-resource-usage "$cubin" >"$resources"
    "$nvdisasm_bin" "$cubin" >"$sass"

    local registers stack local_bytes shared instructions barriers qmma shfl e4m3
    registers=$(resource_value "$resources" "$symbol" REG)
    stack=$(resource_value "$resources" "$symbol" STACK)
    local_bytes=$(resource_value "$resources" "$symbol" LOCAL)
    shared=$(resource_value "$resources" "$symbol" SHARED)
    instructions=$(instruction_count "$sass")
    barriers=$(grep -c 'BAR.SYNC' "$sass" || true)
    qmma=$(grep -c 'QMMA.16832.F32.E4M3.E4M3' "$sass" || true)
    shfl=$(grep -c 'SHFL.IDX' "$sass" || true)
    e4m3=$(grep -c 'F2FP.SATFINITE.E4M3.F16' "$sass" || true)
    for value in \
        "$registers" "$stack" "$local_bytes" "$shared" "$instructions" \
        "$barriers" "$qmma" "$shfl" "$e4m3"; do
        [[ $value =~ ^[0-9]+$ ]]
    done
    [[ $stack == 0 ]]
    [[ $local_bytes == 0 ]]
    [[ $qmma == 16 ]]
    [[ $shfl == 16 ]]
    [[ $e4m3 == 16 ]]
    grep -A1 "Function properties for $symbol" "$log" \
        | grep -q '0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads'
    [[ $(grep -cE '[[:space:]](ATOM|RED|LDL|STL)' "$sass" || true) == 0 ]]
    printf '%s\n' \
        "registers=$registers" \
        "stack=$stack" \
        "local=$local_bytes" \
        "shared=$shared" \
        "instructions=$instructions" \
        "barriers=$barriers" >"$probe_dir/${name}.metrics"
}

compile_variant incumbent 0
compile_variant candidate 1

metric() {
    local name=$1 field=$2
    awk -F= -v field="$field" '$1 == field { print $2 }' "$probe_dir/${name}.metrics"
}

incumbent_registers=$(metric incumbent registers)
candidate_registers=$(metric candidate registers)
incumbent_shared=$(metric incumbent shared)
candidate_shared=$(metric candidate shared)
incumbent_barriers=$(metric incumbent barriers)
candidate_barriers=$(metric candidate barriers)
incumbent_instructions=$(metric incumbent instructions)
candidate_instructions=$(metric candidate instructions)

[[ $incumbent_registers == 97 ]]
[[ $incumbent_shared == 10240 ]]
[[ $incumbent_instructions == 1045 ]]
[[ $candidate_instructions == 1061 ]]
(( candidate_registers == incumbent_registers ))
(( candidate_shared == incumbent_shared + 16 ))
(( candidate_shared <= static_shared_limit ))
(( candidate_barriers == incumbent_barriers + 1 ))

if "$nvcc_bin" -std=c++17 -O3 --fmad=false -arch=sm_121a \
    -DW2A8_FIXED_N=4096 -DW2A8_FIXED_K=2048 \
    -DW2A8_PACKED_E4M3_CANDIDATE=1 \
    -DW2A8_N256_SINGLE_WARP_ROUTE_GUARD_CANDIDATE=2 \
    -DW2A8_KERNEL_NAME=exl3_w2a8_n256_invalid_route_guard \
    -cubin "$source_file" -o "$probe_dir/invalid.cubin" >/dev/null 2>&1; then
    echo "non-boolean route-guard selector unexpectedly compiled" >&2
    exit 1
fi

digest=$(
    sha256sum "$probe_dir"/{incumbent,candidate}.{cubin,resources,sass} \
        | awk '{print $1}' | sha256sum | awk '{print $1}'
)
echo "N256 down route guard: packed E4M3, one selector, 16 QMMA, exact resource ceiling"
echo "incumbent: instructions=$incumbent_instructions registers=$incumbent_registers shared=$incumbent_shared barriers=$incumbent_barriers"
echo "candidate: instructions=$candidate_instructions registers=$candidate_registers shared=$candidate_shared barriers=$candidate_barriers"
echo "stack=0 local=0 spills=0 atomics=0 cubin_set_sha256=$digest"
