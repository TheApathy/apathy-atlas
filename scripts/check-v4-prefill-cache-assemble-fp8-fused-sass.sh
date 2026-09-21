#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cuda_root_path=${CUDA_ROOT_PATH:-/usr/local/cuda}
nvcc_bin=${NVCC_BIN:-"$cuda_root_path/bin/nvcc"}
cuobjdump_bin=${CUOBJDUMP_BIN:-"$cuda_root_path/bin/cuobjdump"}
nvdisasm_bin=${NVDISASM_BIN:-"$cuda_root_path/bin/nvdisasm"}
source_file="$repo_root/kernels/gb10/experiments/v4_prefill_cache_assemble_fp8_fused.cu"
probe_dir=$(mktemp -d /tmp/atlas-v4-cache-assemble-fp8-fused.XXXXXX)
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
    echo "missing isolated V4 cache-assemble/FP8 source: $source_file" >&2
    exit 1
fi

cubin="$probe_dir/v4-prefill-cache-assemble-fp8-fused.cubin"
log="$probe_dir/v4-prefill-cache-assemble-fp8-fused.log"
resources="$probe_dir/v4-prefill-cache-assemble-fp8-fused.resources"
sass="$probe_dir/v4-prefill-cache-assemble-fp8-fused.sass"

"$nvcc_bin" -std=c++17 -O3 --fmad=false -arch=sm_121a -cubin "$source_file" \
    -o "$cubin" -Xptxas=-v 2>"$log"
"$cuobjdump_bin" --dump-resource-usage "$cubin" >"$resources"
"$nvdisasm_bin" "$cubin" >"$sass"

symbol=v4_prefill_cache_assemble_fp8_fused
resource=$(awk -v symbol="$symbol" '
    $0 == " Function " symbol ":" { getline; print; found = 1; exit }
    END { if (!found) exit 1 }
' "$resources" | xargs)
expected_resource='REG:23 STACK:0 SHARED:0 LOCAL:0 CONSTANT[0]:968 TEXTURE:0 SURFACE:0 SAMPLER:0'
[[ $resource == "$expected_resource" ]]
grep -A1 "Function properties for $symbol" "$log" \
    | grep -q '0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads'
grep -A2 "Function properties for $symbol" "$log" \
    | grep -q 'Used 23 registers, used 0 barriers'

opcode_count() {
    local pattern=$1
    grep -cE "$pattern" "$sass" || true
}

instructions=$(opcode_count '^[[:space:]]*/\*[0-9a-f]+\*/')
fp8_pairs=$(opcode_count 'F2FP\.SATFINITE\.E4M3\.F32\.PACK_AB_MERGE_C')
global_loads=$(opcode_count '[[:space:]]LDG([[:space:]]|\.)')
global_stores=$(opcode_count '[[:space:]]STG([[:space:]]|\.)')
forbidden=$(opcode_count '[[:space:]](ATOM|RED|LDL|STL|BAR)(\.|[[:space:]])')

[[ $instructions == 356 ]]
[[ $fp8_pairs == 2 ]]
[[ $global_loads == 3 ]]
[[ $global_stores == 2 ]]
[[ $forbidden == 0 ]]

source_sha256=$(sha256sum "$source_file" | awk '{print $1}')
cubin_sha256=$(sha256sum "$cubin" | awk '{print $1}')
echo "$symbol: instructions=$instructions registers=23 shared=0 stack=0 local=0 spills=0 barriers=0 atomics=0"
echo "topology: fp8_pair_converts=$fp8_pairs global_loads=$global_loads global_stores=$global_stores"
echo "source_sha256=$source_sha256 cubin_sha256=$cubin_sha256"
