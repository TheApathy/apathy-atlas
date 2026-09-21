#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cuda_root_path=${CUDA_ROOT_PATH:-/usr/local/cuda}
nvcc_bin=${NVCC_BIN:-"$cuda_root_path/bin/nvcc"}
cuobjdump_bin=${CUOBJDUMP_BIN:-"$cuda_root_path/bin/cuobjdump"}
source_file="$repo_root/kernels/gb10/deepseek-v4-flash/nvfp4/prefill_attn_compressed.cu"
probe_dir=$(mktemp -d /tmp/atlas-v4-prefill-attn-tc2.XXXXXX)
trap 'rm -rf -- "$probe_dir"' EXIT

for tool in "$nvcc_bin" "$cuobjdump_bin"; do
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
    echo "missing V4 compressed-prefill attention source: $source_file" >&2
    exit 1
fi

cubin="$probe_dir/prefill-attn-compressed.cubin"
ptxas_log="$probe_dir/prefill-attn-compressed.ptxas"
resources="$probe_dir/prefill-attn-compressed.resources"
tc2_sass="$probe_dir/prefill-attn-compressed-tc2.sass"

"$nvcc_bin" -std=c++17 -O3 -arch=sm_121a -diag-suppress 186 \
    -cubin "$source_file" -o "$cubin" -Xptxas=-v 2>"$ptxas_log"
"$cuobjdump_bin" --dump-resource-usage "$cubin" >"$resources"
"$cuobjdump_bin" --dump-sass --function prefill_attn_compressed_tc2 \
    "$cubin" >"$tc2_sass"

resource_line() {
    local symbol=$1
    awk -v symbol="$symbol" '
        $0 == " Function " symbol ":" { getline; print; found = 1; exit }
        END { if (!found) exit 1 }
    ' "$resources" | xargs
}

scalar_resource=$(resource_line prefill_attn_compressed)
tc_resource=$(resource_line prefill_attn_compressed_tc)
tc2_resource=$(resource_line prefill_attn_compressed_tc2)

# CUDA 13.0 SM121a compile contract. cuobjdump includes a 1,024-byte ABI
# reservation in SHARED; the ptxas static-smem values checked below do not.
expected_scalar_resource='REG:166 STACK:0 SHARED:33792 LOCAL:0 CONSTANT[0]:984 TEXTURE:0 SURFACE:0 SAMPLER:0'
expected_tc_resource='REG:150 STACK:0 SHARED:39744 LOCAL:0 CONSTANT[0]:984 TEXTURE:0 SURFACE:0 SAMPLER:0'
expected_tc2_resource='REG:167 STACK:0 SHARED:22016 LOCAL:0 CONSTANT[0]:984 TEXTURE:0 SURFACE:0 SAMPLER:0'
[[ $scalar_resource == "$expected_scalar_resource" ]]
[[ $tc_resource == "$expected_tc_resource" ]]
[[ $tc2_resource == "$expected_tc2_resource" ]]

for contract in \
    'prefill_attn_compressed.*0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads' \
    'prefill_attn_compressed_tc.*0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads' \
    'prefill_attn_compressed_tc2.*0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads'; do
    symbol=${contract%%.*}
    grep -A1 "Function properties for $symbol" "$ptxas_log" \
        | grep -q '0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads'
done
grep -A2 'Function properties for prefill_attn_compressed$' "$ptxas_log" \
    | grep -q 'Used 166 registers, used 1 barriers, 32768 bytes smem'
grep -A2 'Function properties for prefill_attn_compressed_tc$' "$ptxas_log" \
    | grep -q 'Used 150 registers, used 1 barriers, 38720 bytes smem'
grep -A2 'Function properties for prefill_attn_compressed_tc2$' "$ptxas_log" \
    | grep -q 'Used 167 registers, used 1 barriers, 20992 bytes smem'

instruction_count=$(awk '/^[[:space:]]*\/\*[0-9a-f]+\*\// { count++ } END { print count + 0 }' "$tc2_sass")
hmma_count=$(grep -c 'HMMA.16816.F32.BF16' "$tc2_sass" || true)
ldm_m88_count=$(grep -c 'LDSM.16.M88.4' "$tc2_sass" || true)
ldm_mt88_count=$(grep -c 'LDSM.16.MT88.4' "$tc2_sass" || true)
exp_count=$(grep -c 'MUFU.EX2' "$tc2_sass" || true)
barrier_count=$(grep -c 'BAR.SYNC' "$tc2_sass" || true)
[[ $instruction_count == 1776 ]]
[[ $hmma_count == 64 ]]
[[ $ldm_m88_count == 16 ]]
[[ $ldm_mt88_count == 16 ]]
[[ $exp_count == 24 ]]
[[ $barrier_count == 8 ]]
[[ $(grep -cE '[[:space:]](ATOM|RED|LDL|STL)' "$tc2_sass" || true) == 0 ]]

threads=128
registers=167
register_file_limit=65536
allocated_registers_per_block=$(( ((registers * 32 + 255) / 256) * 256 * (threads / 32) ))
[[ $allocated_registers_per_block == 21504 ]]
(( allocated_registers_per_block * 3 <= register_file_limit ))
(( allocated_registers_per_block * 4 > register_file_limit ))

source_sha256=$(sha256sum "$source_file" | awk '{print $1}')
cubin_sha256=$(sha256sum "$cubin" | awk '{print $1}')
echo "prefill_attn_compressed_tc2: instructions=$instruction_count registers=167 allocated_registers_per_block=$allocated_registers_per_block ptxas_shared=20992 cuobjdump_shared=22016 stack=0 local=0 spills=0 atomics=0"
echo "census: hmma_bf16=$hmma_count ldm_m88=$ldm_m88_count ldm_mt88=$ldm_mt88_count exp2=$exp_count bar_sync=$barrier_count"
echo "siblings: scalar={$scalar_resource} tc={$tc_resource}"
echo "source_sha256=$source_sha256 cubin_sha256=$cubin_sha256"
