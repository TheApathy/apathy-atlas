#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cuda_root_path=${CUDA_ROOT_PATH:-/usr/local/cuda}
nvcc_bin=${NVCC_BIN:-"$cuda_root_path/bin/nvcc"}
cuobjdump_bin=${CUOBJDUMP_BIN:-"$cuda_root_path/bin/cuobjdump"}
nvdisasm_bin=${NVDISASM_BIN:-"$cuda_root_path/bin/nvdisasm"}
probe_dir=$(mktemp -d /tmp/atlas-exl3-prefill-n256.XXXXXX)
trap 'rm -rf -- "$probe_dir"' EXIT

for tool in "$nvcc_bin" "$cuobjdump_bin" "$nvdisasm_bin"; do
    if [[ ! -x "$tool" ]]; then
        echo "missing CUDA tool: $tool" >&2
        exit 2
    fi
done

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

for width in 128 256; do
    for kind in gu down; do
        symbol="exl3_grouped_prefill_k64_n${width}_k2_${kind}"
        source="$repo_root/kernels/gb10/common/${symbol}.cu"
        cubin="$probe_dir/${symbol}.cubin"
        log="$probe_dir/${symbol}.log"
        resources="$probe_dir/${symbol}.resources"
        sass="$probe_dir/${symbol}.sass"
        "$nvcc_bin" -std=c++17 -arch=sm_121a --fmad=false \
            -I "$repo_root/kernels/gb10/common" -cubin "$source" \
            -o "$cubin" -Xptxas=-v 2>"$log"
        "$cuobjdump_bin" --dump-resource-usage "$cubin" >"$resources"
        "$nvdisasm_bin" "$cubin" >"$sass"

        [[ $(resource_value "$resources" "$symbol" REG) -le 62 ]]
        [[ $(resource_value "$resources" "$symbol" STACK) == 0 ]]
        [[ $(resource_value "$resources" "$symbol" LOCAL) == 0 ]]
        grep -A1 "Function properties for $symbol" "$log" \
            | grep -q '0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads'
        [[ $(grep -c 'HMMA' "$sass" || true) == 32 ]]
        [[ $(grep -cE '[[:space:]](ATOM|LDL|STL)' "$sass" || true) == 0 ]]
        if [[ $width == 128 ]]; then
            [[ $(resource_value "$resources" "$symbol" SHARED) -le 11520 ]]
        else
            [[ $(resource_value "$resources" "$symbol" SHARED) -le 13568 ]]
        fi
    done
done

digest=$(
    for cubin in "$probe_dir"/*.cubin; do
        sha256sum "$cubin" | awk '{print $1}'
    done | sha256sum | awk '{print $1}'
)
echo "N128/N256: <=62 registers, 32 HMMAs, zero stack/local/spills/atomics"
echo "cuobjdump shared: N128<=11520 B; N256<=13568 B; cubin_set_sha256=$digest"
