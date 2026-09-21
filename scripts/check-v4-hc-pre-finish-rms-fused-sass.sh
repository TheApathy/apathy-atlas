#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cuda_root_path=${CUDA_ROOT_PATH:-/usr/local/cuda}
nvcc_bin=${NVCC_BIN:-"$cuda_root_path/bin/nvcc"}
cuobjdump_bin=${CUOBJDUMP_BIN:-"$cuda_root_path/bin/cuobjdump"}
source_file="$repo_root/kernels/gb10/experiments/v4_hc_pre_finish_rms_fused.cu"
probe_dir=$(mktemp -d /tmp/atlas-v4-hc-finish-rms-fused.XXXXXX)
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
    echo "missing V4 HC finish/RMS implementation: $source_file" >&2
    exit 1
fi

cubin="$probe_dir/v4-hc-finish-rms-fused.cubin"
log="$probe_dir/v4-hc-finish-rms-fused.ptxas"
resources="$probe_dir/v4-hc-finish-rms-fused.resources"
sass="$probe_dir/v4-hc-finish-rms-fused.sass"

"$nvcc_bin" -std=c++17 -O3 --fmad=false -arch=sm_121a -cubin "$source_file" \
    -o "$cubin" -Xptxas=-v 2>"$log"
"$cuobjdump_bin" --dump-resource-usage "$cubin" >"$resources"
"$cuobjdump_bin" --dump-sass --function v4_hc_pre_finish_rms_fused \
    "$cubin" >"$sass"

resource=$(awk '
    $0 == " Function v4_hc_pre_finish_rms_fused:" { getline; print; found = 1; exit }
    END { if (!found) exit 1 }
' "$resources" | xargs)
expected_resource='REG:48 STACK:0 SHARED:1168 LOCAL:0 CONSTANT[0]:996 TEXTURE:0 SURFACE:0 SAMPLER:0'
[[ $resource == "$expected_resource" ]]
grep -A1 'Function properties for v4_hc_pre_finish_rms_fused' "$log" \
    | grep -q '0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads'
grep -A2 'Function properties for v4_hc_pre_finish_rms_fused' "$log" \
    | grep -q 'Used 48 registers, used 1 barriers, 144 bytes smem'

opcode_count() {
    local pattern=$1
    grep -cE "$pattern" "$sass" || true
}

instructions=$(opcode_count '^[[:space:]]*/\*[0-9a-f]+\*/')
bf16=$(opcode_count '[[:space:]]F2F\.BF16\.F32')
exp2=$(opcode_count '[[:space:]]MUFU\.EX2')
reciprocal=$(opcode_count '[[:space:]]MUFU\.RCP')
rsqrt=$(opcode_count '[[:space:]]MUFU\.RSQ')
shuffle=$(opcode_count '[[:space:]]SHFL\.BFLY')
barrier=$(opcode_count '[[:space:]]BAR\.SYNC')
atomics=$(opcode_count '[[:space:]](ATOM|RED)(\.|[[:space:]])')
local_ops=$(opcode_count '[[:space:]](LDL|STL)(\.|[[:space:]])')
[[ $instructions == 2064 ]]
[[ $bf16 == 8 ]]
[[ $exp2 == 24 ]]
[[ $reciprocal == 81 ]]
[[ $rsqrt == 3 ]]
[[ $shuffle == 15 ]]
[[ $barrier == 3 ]]
[[ $atomics == 0 ]]
[[ $local_ops == 0 ]]

source_sha256=$(sha256sum "$source_file" | awk '{print $1}')
cubin_sha256=$(sha256sum "$cubin" | awk '{print $1}')
echo "v4_hc_pre_finish_rms_fused: instructions=$instructions registers=48 ptxas_shared=144 cuobjdump_shared=1168 stack=0 local=0 spills=0 atomics=0"
echo "census: bf16=$bf16 exp2=$exp2 reciprocal=$reciprocal rsqrt=$rsqrt shuffle=$shuffle bar_sync=$barrier"
echo "source_sha256=$source_sha256 cubin_sha256=$cubin_sha256"
