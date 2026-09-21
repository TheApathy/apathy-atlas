#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cuda_root_path=${CUDA_ROOT_PATH:-/usr/local/cuda}
nvcc_bin=${NVCC_BIN:-"$cuda_root_path/bin/nvcc"}
cuobjdump_bin=${CUOBJDUMP_BIN:-"$cuda_root_path/bin/cuobjdump"}
nvdisasm_bin=${NVDISASM_BIN:-"$cuda_root_path/bin/nvdisasm"}
probe_dir=$(mktemp -d /tmp/atlas-mla-head-group.XXXXXX)
trap 'rm -rf -- "$probe_dir"' EXIT

for tool in "$nvcc_bin" "$cuobjdump_bin" "$nvdisasm_bin"; do
    if [[ ! -x $tool ]]; then
        echo "missing CUDA tool: $tool" >&2
        exit 2
    fi
done
for injected_flags in NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS; do
    if [[ -n ${!injected_flags:-} ]]; then
        echo "refusing injected CUDA flags from $injected_flags" >&2
        exit 2
    fi
done

source_file="$repo_root/kernels/gb10/experiments/mla_paged_decode_fp8_heads8.cu"
cubin="$probe_dir/mla-head-group.cubin"
log="$probe_dir/ptxas.log"
resources="$probe_dir/resources.txt"
sass="$probe_dir/kernel.sass"

"$nvcc_bin" -std=c++17 -O3 --fmad=false -arch=sm_121a \
    -cubin "$source_file" -o "$cubin" -Xptxas=-v 2>"$log"
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

check_symbol() {
    local symbol=$1
    local register_limit=$2
    local registers stack local_bytes shared
    registers=$(resource_value "$symbol" REG)
    stack=$(resource_value "$symbol" STACK)
    local_bytes=$(resource_value "$symbol" LOCAL)
    shared=$(resource_value "$symbol" SHARED)
    for value in "$registers" "$stack" "$local_bytes" "$shared"; do
        [[ $value =~ ^[0-9]+$ ]]
    done
    [[ $registers -le $register_limit ]]
    [[ $stack == 0 ]]
    [[ $local_bytes == 0 ]]
    [[ $shared == 9216 ]]
    grep -A1 "Function properties for $symbol" "$log" \
        | grep -q '0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads'
}

check_symbol mla_paged_decode_fp8_heads8 104
check_symbol mla_paged_decode_fp8_heads8_kvalias 92
[[ $(grep -c 'used 1 barriers, 8192 bytes smem' "$log") == 2 ]]
[[ $(grep -c 'SHFL' "$sass" || true) -ge 64 ]]
[[ $(grep -c 'BAR' "$sass" || true) -ge 16 ]]
[[ $(grep -c 'MUFU.EX2' "$sass" || true) -ge 20 ]]
[[ $(grep -c 'F2FP' "$sass" || true) -ge 32 ]]
[[ $(grep -cE '[[:space:]](ATOM|LDL|STL)' "$sass" || true) == 0 ]]

digest=$(sha256sum "$cubin" | awk '{print $1}')
echo "MLA heads8 compile-only: generic<=104 regs, kvalias<=92 regs, 8192 B smem"
echo "zero stack/local/spills/atomics; sm_121a cubin_sha256=$digest"
