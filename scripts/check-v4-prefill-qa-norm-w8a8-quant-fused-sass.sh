#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cuda_root_path=${CUDA_ROOT_PATH:-/usr/local/cuda}
nvcc_bin=${NVCC_BIN:-"$cuda_root_path/bin/nvcc"}
cuobjdump_bin=${CUOBJDUMP_BIN:-"$cuda_root_path/bin/cuobjdump"}
source_file="$repo_root/kernels/gb10/experiments/v4_prefill_qa_norm_w8a8_quant_fused.cu"
probe_dir=$(mktemp -d /tmp/atlas-v4-qa-norm-w8a8-fused.XXXXXX)
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
    echo "missing isolated V4 q_a norm/W8A8 source: $source_file" >&2
    exit 1
fi

cubin="$probe_dir/v4-qa-norm-w8a8-fused.cubin"
log="$probe_dir/v4-qa-norm-w8a8-fused.ptxas"
resources="$probe_dir/v4-qa-norm-w8a8-fused.resources"
sass="$probe_dir/v4-qa-norm-w8a8-fused.sass"

"$nvcc_bin" -std=c++17 -O3 --fmad=false -arch=sm_121a -cubin "$source_file" \
    -o "$cubin" -Xptxas=-v 2>"$log"
"$cuobjdump_bin" --dump-resource-usage "$cubin" >"$resources"
"$cuobjdump_bin" --dump-sass --function v4_prefill_qa_norm_w8a8_quant_fused \
    "$cubin" >"$sass"

resource=$(awk '
    $0 == " Function v4_prefill_qa_norm_w8a8_quant_fused:" {
        getline
        print
        found = 1
        exit
    }
    END { if (!found) exit 1 }
' "$resources" | xargs)
expected_resource='REG:22 STACK:0 SHARED:3200 LOCAL:0 CONSTANT[0]:940 TEXTURE:0 SURFACE:0 SAMPLER:0'
[[ $resource == "$expected_resource" ]]
grep -A1 'Function properties for v4_prefill_qa_norm_w8a8_quant_fused' "$log" \
    | grep -q '0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads'
grep -A2 'Function properties for v4_prefill_qa_norm_w8a8_quant_fused' "$log" \
    | grep -q 'Used 22 registers, used 1 barriers, 2176 bytes smem'

opcode_count() {
    local pattern=$1
    grep -cE "$pattern" "$sass" || true
}

instructions=$(opcode_count '^[[:space:]]*/\*[0-9a-f]+\*/')
bf16_pack=$(opcode_count 'F2FP\.BF16\.F32\.PACK_AB')
e4m3=$(opcode_count 'F2FP\.SATFINITE\.E4M3\.F32')
rsqrt=$(opcode_count '[[:space:]]MUFU\.RSQ')
reciprocal=$(opcode_count '[[:space:]]MUFU\.RCP')
xor_shuffle=$(opcode_count '[[:space:]]SHFL\.BFLY')
down_shuffle=$(opcode_count '[[:space:]]SHFL\.DOWN')
barrier=$(opcode_count '[[:space:]]BAR\.SYNC')
shared_load=$(opcode_count '[[:space:]]LDS([[:space:]]|\.)')
shared_store=$(opcode_count '[[:space:]]STS([[:space:]]|\.)')
forbidden=$(opcode_count '[[:space:]](ATOM|RED|LDL|STL)(\.|[[:space:]])')
[[ $instructions == 440 ]]
[[ $bf16_pack == 1 ]]
[[ $e4m3 == 4 ]]
[[ $rsqrt == 2 ]]
[[ $reciprocal == 6 ]]
[[ $xor_shuffle == 15 ]]
[[ $down_shuffle == 10 ]]
[[ $barrier == 5 ]]
[[ $shared_load == 9 ]]
[[ $shared_store == 5 ]]
[[ $forbidden == 0 ]]

source_sha256=$(sha256sum "$source_file" | awk '{print $1}')
cubin_sha256=$(sha256sum "$cubin" | awk '{print $1}')
echo "v4_prefill_qa_norm_w8a8_quant_fused: instructions=$instructions registers=22 ptxas_shared=2176 cuobjdump_shared=3200 stack=0 local=0 spills=0 atomics=0"
echo "census: bf16_pack=$bf16_pack e4m3=$e4m3 rsqrt=$rsqrt reciprocal=$reciprocal xor_shuffle=$xor_shuffle down_shuffle=$down_shuffle bar_sync=$barrier shared_load=$shared_load shared_store=$shared_store"
echo "source_sha256=$source_sha256 cubin_sha256=$cubin_sha256"
