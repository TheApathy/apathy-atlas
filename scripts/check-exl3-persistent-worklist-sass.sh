#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cuda_root_path=${CUDA_ROOT_PATH:-/usr/local/cuda}
nvcc_bin=${NVCC_BIN:-"$cuda_root_path/bin/nvcc"}
cuobjdump_bin=${CUOBJDUMP_BIN:-"$cuda_root_path/bin/cuobjdump"}
nvdisasm_bin=${NVDISASM_BIN:-"$cuda_root_path/bin/nvdisasm"}
probe_dir=$(mktemp -d /tmp/atlas-exl3-persistent-worklist.XXXXXX)
trap 'rm -rf -- "$probe_dir"' EXIT

for tool in "$nvcc_bin" "$cuobjdump_bin" "$nvdisasm_bin"; do
    if [[ ! -x "$tool" ]]; then
        echo "missing CUDA tool: $tool" >&2
        exit 2
    fi
done

cubin="$probe_dir/exl3_gemv_k2.cubin"
"$nvcc_bin" -std=c++17 -arch=sm_121a --fmad=false \
    -I "$repo_root/kernels/gb10/common" \
    -cubin "$repo_root/kernels/gb10/common/exl3_gemv_k2.cu" \
    -o "$cubin" -Xptxas=-v 2>"$probe_dir/ptxas.log"
"$cuobjdump_bin" --dump-resource-usage "$cubin" >"$probe_dir/resources.txt"
"$nvdisasm_bin" "$cubin" >"$probe_dir/disasm.txt"

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
    ' "$probe_dir/resources.txt"
}

extract_function() {
    local symbol=$1
    local destination=$2
    awk -v marker=".text.${symbol} " '
        /^\/\/--------------------- \.text\./ {
            if (active) exit
            if (index($0, marker)) active = 1
        }
        active { print }
    ' "$probe_dir/disasm.txt" >"$destination"
}

declare -A register_ceiling=(
    [exl3_build_m6_worklist]=18
    [exl3_gemv_mrow_persistent_gate_up_m6]=109
    [exl3_gemv_mrow_persistent_down_m6]=104
    [exl3_gemv_mrow_fused_gate_up_m6]=108
    [exl3_gemv_mrow_fused_down_m6]=108
    [exl3_build_m16_worklist]=20
    [exl3_gemv_mrow_persistent_gate_up_m16]=128
    [exl3_gemv_mrow_persistent_down_m16]=128
    [exl3_gemv_mrow_fused_gate_up_m16]=128
    [exl3_gemv_mrow_fused_down_m16]=128
)

declare -A m16_shared_ceiling=(
    [exl3_build_m16_worklist]=0
    [exl3_gemv_mrow_persistent_gate_up_m16]=37972
    [exl3_gemv_mrow_persistent_down_m16]=37972
    [exl3_gemv_mrow_fused_gate_up_m16]=40020
    [exl3_gemv_mrow_fused_down_m16]=40020
)

for symbol in "${!register_ceiling[@]}"; do
    registers=$(resource_value "$symbol" REG)
    [[ -n "$registers" ]]
    (( registers <= register_ceiling[$symbol] ))
    for field in STACK LOCAL; do
        [[ $(resource_value "$symbol" "$field") == 0 ]]
    done
    grep -A1 "Function properties for $symbol" "$probe_dir/ptxas.log" \
        | grep -q '0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads'
done

for symbol in "${!m16_shared_ceiling[@]}"; do
    (( $(resource_value "$symbol" SHARED) <= m16_shared_ceiling[$symbol] ))
done

for rows in 6 16; do
    for kind in gate_up down; do
        persistent="exl3_gemv_mrow_persistent_${kind}_m${rows}"
        incumbent="exl3_gemv_mrow_fused_${kind}_m${rows}"
        (( $(resource_value "$persistent" SHARED) <= $(resource_value "$incumbent" SHARED) ))
        extract_function "$persistent" "$probe_dir/$persistent.sass"
        [[ $(grep -c 'BAR.SYNC' "$probe_dir/$persistent.sass" || true) -ge 1 ]]
    done
done

cubin_sha256=$(sha256sum "$cubin" | awk '{print $1}')
echo "builder: registers=$(resource_value exl3_build_m6_worklist REG) spills=0"
echo "persistent gate/up: registers=$(resource_value exl3_gemv_mrow_persistent_gate_up_m6 REG) shared=$(resource_value exl3_gemv_mrow_persistent_gate_up_m6 SHARED) spills=0"
echo "persistent down: registers=$(resource_value exl3_gemv_mrow_persistent_down_m6 REG) shared=$(resource_value exl3_gemv_mrow_persistent_down_m6 SHARED) spills=0"
echo "M16 builder: registers=$(resource_value exl3_build_m16_worklist REG) shared=$(resource_value exl3_build_m16_worklist SHARED) spills=0"
echo "M16 persistent gate/up: registers=$(resource_value exl3_gemv_mrow_persistent_gate_up_m16 REG) shared=$(resource_value exl3_gemv_mrow_persistent_gate_up_m16 SHARED) spills=0"
echo "M16 persistent down: registers=$(resource_value exl3_gemv_mrow_persistent_down_m16 REG) shared=$(resource_value exl3_gemv_mrow_persistent_down_m16 SHARED) spills=0"
echo "M16 incumbent gate/up: registers=$(resource_value exl3_gemv_mrow_fused_gate_up_m16 REG) shared=$(resource_value exl3_gemv_mrow_fused_gate_up_m16 SHARED) spills=0"
echo "M16 incumbent down: registers=$(resource_value exl3_gemv_mrow_fused_down_m16 REG) shared=$(resource_value exl3_gemv_mrow_fused_down_m16 SHARED) spills=0"
echo "fixed-grid worklist cubin_sha256=$cubin_sha256"
