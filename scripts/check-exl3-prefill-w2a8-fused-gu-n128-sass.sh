#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cuda_root_path=${CUDA_ROOT_PATH:-/usr/local/cuda}
nvcc_bin=${NVCC_BIN:-"$cuda_root_path/bin/nvcc"}
cuobjdump_bin=${CUOBJDUMP_BIN:-"$cuda_root_path/bin/cuobjdump"}
nvdisasm_bin=${NVDISASM_BIN:-"$cuda_root_path/bin/nvdisasm"}
source_file="$repo_root/kernels/gb10/experiments/exl3_w2a8_fused_gu_down_emit_n128.cu"
probe_dir=$(mktemp -d /tmp/atlas-exl3-w2a8-fused-gu-n128.XXXXXX)
trap 'rm -rf -- "$probe_dir"' EXIT

for tool in "$nvcc_bin" "$cuobjdump_bin" "$nvdisasm_bin"; do
    if [[ ! -x $tool ]]; then
        echo "missing CUDA tool: $tool" >&2
        exit 2
    fi
done
if [[ ! -f $source_file ]]; then
    echo "missing fused W2A8 N128 component: $source_file" >&2
    exit 1
fi

threads=256
register_file_limit=65536
static_shared_limit=$((48 * 1024))
symbol=exl3_w2a8_fused_gu_down_emit_n128
cubin="$probe_dir/fused-gu.cubin"
log="$probe_dir/fused-gu.log"
resources="$probe_dir/fused-gu.resources"
sass="$probe_dir/fused-gu.sass"

resource_value() {
    local file=$1 symbol_name=$2 field=$3
    awk -v symbol="$symbol_name" -v field="$field" '
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

compile_component() {
    "$nvcc_bin" -std=c++17 -O3 --fmad=false -arch=sm_121a \
        -I "$repo_root/kernels/gb10/common" \
        -DW2A8_FIXED_N=2048 -DW2A8_FIXED_K=4096 \
        -DW2A8_KERNEL_NAME="$symbol" \
        -cubin "$source_file" -o "$cubin" -Xptxas=-v 2>"$log"
}

expect_compile_failure() {
    local label=$1
    shift
    if "$nvcc_bin" -std=c++17 -O3 --fmad=false -arch=sm_121a \
        -I "$repo_root/kernels/gb10/common" "$@" \
        -cubin "$source_file" -o "$probe_dir/invalid-${label}.cubin" \
        >/dev/null 2>&1; then
        echo "invalid fused W2A8 N128 macro set unexpectedly compiled: $label" >&2
        exit 1
    fi
}

# The component must not invent shape or symbol defaults. It is the exact
# DeepSeek gate/up projection (N=2048,K=4096); its emitted down activation has
# K=2048. Crossed and merely-divisible shapes are not compatible fallbacks.
expect_compile_failure missing_all
expect_compile_failure missing_symbol -DW2A8_FIXED_N=2048 -DW2A8_FIXED_K=4096
expect_compile_failure unsupported_n \
    -DW2A8_FIXED_N=3072 -DW2A8_FIXED_K=4096 \
    -DW2A8_KERNEL_NAME=exl3_w2a8_fused_invalid_n
expect_compile_failure crossed_gu \
    -DW2A8_FIXED_N=2048 -DW2A8_FIXED_K=2048 \
    -DW2A8_KERNEL_NAME=exl3_w2a8_fused_invalid_k
expect_compile_failure crossed_down \
    -DW2A8_FIXED_N=4096 -DW2A8_FIXED_K=4096 \
    -DW2A8_KERNEL_NAME=exl3_w2a8_fused_invalid_shape

compile_component
"$cuobjdump_bin" --dump-resource-usage "$cubin" >"$resources"
"$nvdisasm_bin" "$cubin" >"$sass"

registers=$(resource_value "$resources" "$symbol" REG)
stack=$(resource_value "$resources" "$symbol" STACK)
local_bytes=$(resource_value "$resources" "$symbol" LOCAL)
shared=$(resource_value "$resources" "$symbol" SHARED)
instructions=$(instruction_count "$sass")
for value in "$registers" "$stack" "$local_bytes" "$shared" "$instructions"; do
    if [[ ! $value =~ ^[0-9]+$ ]]; then
        echo "missing or malformed resource value for $symbol: $value" >&2
        exit 1
    fi
done

registers_per_block=$((registers * threads))
# SM121 register allocation is rounded per warp to 256-register quanta.
allocated_registers_per_block=$(( ((registers * 32 + 255) / 256) * 256 * (threads / 32) ))
(( allocated_registers_per_block <= register_file_limit ))
(( shared <= static_shared_limit ))
[[ $stack == 0 ]]
[[ $local_bytes == 0 ]]
grep -A1 "Function properties for $symbol" "$log" \
    | grep -q '0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads'

qmmas=$(grep -c 'QMMA.16832.F32.E4M3.E4M3' "$sass" || true)
bf16_rounds=$(grep -cE 'F2F(P)?\.BF16\.F32' "$sass" || true)
fp8_converts=$(grep -c 'F2FP.SATFINITE.E4M3' "$sass" || true)
swiglu_exp=$(grep -c 'MUFU.EX2' "$sass" || true)
fma_ops=$(grep -cE '[[:space:]](FFMA|HFMA2)' "$sass" || true)
shuffle_ops=$(grep -c 'SHFL\.' "$sass" || true)
barriers=$(grep -c 'BAR.SYNC' "$sass" || true)
for value in "$qmmas" "$bf16_rounds" "$fp8_converts" "$swiglu_exp" "$fma_ops" "$shuffle_ops" "$barriers"; do
    [[ $value =~ ^[0-9]+$ ]]
done

# Freeze the arithmetic/topology census. The total instruction count may move
# when fail-closed guards change, but the two projections, BF16 boundaries,
# H128/SwiGLU/FP8 work, shuffles, and synchronization must remain exact.
(( qmmas == 32 ))
(( bf16_rounds == 52 ))
(( fp8_converts == 36 ))
(( swiglu_exp == 4 ))
(( fma_ops == 166 ))
(( shuffle_ops == 193 ))
(( barriers == 7 ))
[[ $(grep -cE '[[:space:]](ATOM|RED|LDL|STL)' "$sass" || true) == 0 ]]

cubin_sha256=$(sha256sum "$cubin" | awk '{print $1}')
source_sha256=$(sha256sum "$source_file" | awk '{print $1}')
echo "$symbol: instructions=$instructions registers=$registers registers_per_block=$registers_per_block allocated_registers_per_block=$allocated_registers_per_block shared=$shared stack=0 local=0 spills=0 atomics=0"
echo "census: e4m3_qmma=$qmmas bf16_rounds=$bf16_rounds fp8_converts=$fp8_converts swiglu_exp=$swiglu_exp h128_fma=$fma_ops shuffles=$shuffle_ops bar_sync=$barriers"
echo "resource_ceiling: allocated_registers_per_block<=65536 shared<=49152 threads=256 arch=sm_121a"
echo "source_sha256=$source_sha256 cubin_sha256=$cubin_sha256"
