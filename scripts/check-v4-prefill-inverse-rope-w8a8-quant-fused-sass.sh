#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cuda_root_path=${CUDA_ROOT_PATH:-/usr/local/cuda}
nvcc_bin=${NVCC_BIN:-"$cuda_root_path/bin/nvcc"}
cuobjdump_bin=${CUOBJDUMP_BIN:-"$cuda_root_path/bin/cuobjdump"}
source_file="$repo_root/kernels/gb10/experiments/v4_prefill_inverse_rope_w8a8_quant_fused.cu"
probe_dir=$(mktemp -d /tmp/atlas-v4-inverse-rope-w8a8-quant.XXXXXX)
cleanup() {
    local exit_code=$?
    rm -rf -- "$probe_dir"
    trap - EXIT
    exit "$exit_code"
}
trap cleanup EXIT

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
    echo "missing isolated V4 inverse-RoPE/W8A8 source: $source_file" >&2
    exit 1
fi

cubin="$probe_dir/v4-inverse-rope-w8a8-quant.cubin"
log="$probe_dir/v4-inverse-rope-w8a8-quant.ptxas"
resources="$probe_dir/v4-inverse-rope-w8a8-quant.resources"
sass="$probe_dir/v4-inverse-rope-w8a8-quant.sass"
symbol=v4_prefill_inverse_rope_w8a8_quant_fused

"$nvcc_bin" -std=c++17 -O3 --fmad=false -arch=sm_121a -cubin "$source_file" \
    -o "$cubin" -Xptxas=-v 2>"$log"
"$cuobjdump_bin" --dump-resource-usage "$cubin" >"$resources"
"$cuobjdump_bin" --dump-sass --function "$symbol" "$cubin" >"$sass"

resource=$(awk -v symbol="$symbol" '
    $0 == " Function " symbol ":" {
        getline
        print
        found = 1
        exit
    }
    END { if (!found) exit 1 }
' "$resources" | xargs)
expected_resource='REG:34 STACK:0 SHARED:9248 LOCAL:0 CONSTANT[0]:968 TEXTURE:0 SURFACE:0 SAMPLER:0'
[[ $resource == "$expected_resource" ]]
grep -A1 "Function properties for $symbol" "$log" \
    | grep -q '0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads'
grep -A2 "Function properties for $symbol" "$log" \
    | grep -q 'Used 34 registers, used 1 barriers, 8224 bytes smem'

opcode_count() {
    local pattern=$1
    grep -cE "$pattern" "$sass" || true
}

instructions=$(opcode_count '^[[:space:]]*/\*[0-9a-f]+\*/')
bf16_pack=$(opcode_count 'F2FP\.BF16\.F32\.PACK_AB')
e4m3=$(opcode_count 'F2FP\.SATFINITE\.E4M3\.F32')
reciprocal=$(opcode_count '[[:space:]]MUFU\.RCP')
rsqrt=$(opcode_count '[[:space:]]MUFU\.RSQ')
down_shuffle=$(opcode_count '[[:space:]]SHFL\.DOWN')
barrier=$(opcode_count '[[:space:]]BAR\.SYNC')
shared_load=$(opcode_count '[[:space:]]LDS([[:space:]]|\.)')
shared_store=$(opcode_count '[[:space:]]STS([[:space:]]|\.)')
global_load=$(opcode_count '[[:space:]]LDG([[:space:]]|\.)')
global_store=$(opcode_count '[[:space:]]STG([[:space:]]|\.)')
fmnmx=$(opcode_count '[[:space:]]FMNMX([[:space:]]|\.)')
ffma=$(opcode_count '[[:space:]]FFMA([[:space:]]|\.)')
forbidden=$(opcode_count '[[:space:]](ATOM|RED|LDL|STL)(\.|[[:space:]])')

[[ $instructions == 584 ]]
[[ $bf16_pack == 1 ]]
[[ $e4m3 == 1 ]]
[[ $reciprocal == 6 ]]
[[ $rsqrt == 1 ]]
[[ $down_shuffle == 5 ]]
[[ $barrier == 3 ]]
[[ $shared_load == 5 ]]
[[ $shared_store == 3 ]]
[[ $global_load == 8 ]]
[[ $global_store == 2 ]]
[[ $fmnmx == 14 ]]
[[ $ffma == 36 ]]
[[ $forbidden == 0 ]]

source_sha256=$(sha256sum "$source_file" | awk '{print $1}')
cubin_sha256=$(sha256sum "$cubin" | awk '{print $1}')
echo "$symbol: instructions=$instructions registers=34 ptxas_shared=8224 cuobjdump_shared=9248 stack=0 local=0 spills=0 atomics=0"
echo "topology: bf16_pack=$bf16_pack e4m3=$e4m3 reciprocal=$reciprocal rsqrt=$rsqrt down_shuffle=$down_shuffle bar_sync=$barrier shared_load=$shared_load shared_store=$shared_store global_load=$global_load global_store=$global_store fmnmx=$fmnmx ffma=$ffma"
echo "source_sha256=$source_sha256 cubin_sha256=$cubin_sha256"
