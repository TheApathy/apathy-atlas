#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cuda_root_path=${CUDA_ROOT_PATH:-/usr/local/cuda}
nvcc_bin=${NVCC_BIN:-"$cuda_root_path/bin/nvcc"}
cuobjdump_bin=${CUOBJDUMP_BIN:-"$cuda_root_path/bin/cuobjdump"}
nvdisasm_bin=${NVDISASM_BIN:-"$cuda_root_path/bin/nvdisasm"}
w2a8_probe_dir=$(mktemp -d /tmp/atlas-exl3-w2a8.XXXXXX)
trap 'rm -rf -- "$w2a8_probe_dir"' EXIT

for tool in "$nvcc_bin" "$cuobjdump_bin" "$nvdisasm_bin"; do
    if [[ ! -x "$tool" ]]; then
        echo "missing CUDA tool: $tool" >&2
        exit 2
    fi
done

source_file="$repo_root/kernels/gb10/experiments/exl3_w2a8_grouped_prefill.cu"
emit_source_file="$repo_root/kernels/gb10/experiments/exl3_w2a8_h128_emit.cu"

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

extract_function() {
    local source=$1
    local symbol=$2
    local destination=$3
    awk -v marker=".text.${symbol} " '
        /^\/\/--------------------- \.text\./ {
            if (active) exit
            if (index($0, marker)) active = 1
        }
        active { print }
    ' "$source" >"$destination"
}

instruction_count() {
    awk '/^[[:space:]]*\/\*[0-9a-f]+\*\// { count++ } END { print count + 0 }' "$1"
}

compile_and_check() {
    local kind=$1
    local fixed_n=$2
    local fixed_k=$3
    local symbol="exl3_w2a8_grouped_prefill_${kind}"
    local cubin="$w2a8_probe_dir/${kind}.cubin"
    local log="$w2a8_probe_dir/${kind}.log"
    local resources="$w2a8_probe_dir/${kind}.resources"
    local sass="$w2a8_probe_dir/${kind}.sass"

    "$nvcc_bin" -std=c++17 -O3 --fmad=false -arch=sm_121a \
        -DW2A8_FIXED_N="$fixed_n" -DW2A8_FIXED_K="$fixed_k" \
        -DW2A8_KERNEL_NAME="$symbol" -cubin "$source_file" \
        -o "$cubin" -Xptxas=-v 2>"$log"
    "$cuobjdump_bin" --dump-resource-usage "$cubin" >"$resources"
    "$nvdisasm_bin" "$cubin" >"$sass"

    local registers stack local_bytes shared
    registers=$(resource_value "$resources" "$symbol" REG)
    stack=$(resource_value "$resources" "$symbol" STACK)
    local_bytes=$(resource_value "$resources" "$symbol" LOCAL)
    shared=$(resource_value "$resources" "$symbol" SHARED)
    for value in "$registers" "$stack" "$local_bytes" "$shared"; do
        [[ $value =~ ^[0-9]+$ ]]
    done
    [[ $registers -le 103 ]]
    [[ $stack == 0 ]]
    [[ $local_bytes == 0 ]]
    [[ $shared == 7168 ]]
    grep -A1 "Function properties for $symbol" "$log" \
        | grep -q '0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads'
    grep -q 'used 1 barriers, 6144 bytes smem' "$log"
    [[ $(grep -c 'QMMA.16832.F32.E4M3.E4M3' "$sass" || true) == 16 ]]
    [[ $(grep -c 'SHFL.IDX' "$sass" || true) == 16 ]]
    [[ $(grep -c 'F2FP.SATFINITE.E4M3' "$sass" || true) == 16 ]]
    [[ $(grep -c 'HMMA' "$sass" || true) == 0 ]]
    [[ $(grep -cE '[[:space:]](ATOM|LDL|STL)' "$sass" || true) == 0 ]]
}

compile_and_check gu 2048 4096
compile_and_check down 4096 2048

expect_compile_failure() {
    local fixed_n=$1
    local fixed_k=$2
    local label=$3
    if "$nvcc_bin" -std=c++17 -O3 --fmad=false -arch=sm_121a \
        -DW2A8_FIXED_N="$fixed_n" -DW2A8_FIXED_K="$fixed_k" \
        -DW2A8_KERNEL_NAME="exl3_w2a8_invalid_${label}" -cubin "$source_file" \
        -o "$w2a8_probe_dir/invalid-${label}.cubin" >/dev/null 2>&1; then
        echo "invalid W2A8 shape ${fixed_n}x${fixed_k} unexpectedly compiled" >&2
        exit 1
    fi
}

expect_compile_failure 3072 4096 unsupported_n
expect_compile_failure 2048 2048 crossed_gu
expect_compile_failure 4096 4096 crossed_down

emit_cubin="$w2a8_probe_dir/h128-emit.cubin"
emit_log="$w2a8_probe_dir/h128-emit.log"
emit_resources="$w2a8_probe_dir/h128-emit.resources"
emit_disasm="$w2a8_probe_dir/h128-emit.disasm"
"$nvcc_bin" -std=c++17 -O3 --fmad=false -arch=sm_121a \
    -I "$repo_root/kernels/gb10/common" -cubin "$emit_source_file" \
    -o "$emit_cubin" -Xptxas=-v 2>"$emit_log"
"$cuobjdump_bin" --dump-resource-usage "$emit_cubin" >"$emit_resources"
"$nvdisasm_bin" "$emit_cubin" >"$emit_disasm"

check_emitter() {
    local symbol=$1
    local max_registers=$2
    local expected_bf16_rounds=$3
    local expected_fp8_converts=$4
    local expected_shuffle_down=$5
    local expected_shuffle_index=$6
    local expected_shared_stores=$7
    local expected_shared_loads=$8
    local expected_global_stores=$9
    local sass="$w2a8_probe_dir/${symbol}.sass"
    extract_function "$emit_disasm" "$symbol" "$sass"
    [[ -s $sass ]]

    local registers stack local_bytes shared instructions
    registers=$(resource_value "$emit_resources" "$symbol" REG)
    stack=$(resource_value "$emit_resources" "$symbol" STACK)
    local_bytes=$(resource_value "$emit_resources" "$symbol" LOCAL)
    shared=$(resource_value "$emit_resources" "$symbol" SHARED)
    instructions=$(instruction_count "$sass")
    for value in "$registers" "$stack" "$local_bytes" "$shared" "$instructions"; do
        [[ $value =~ ^[0-9]+$ ]]
    done
    [[ $registers -le $max_registers ]]
    [[ $stack == 0 ]]
    [[ $local_bytes == 0 ]]
    [[ $shared == 5120 ]]
    grep -A1 "Function properties for $symbol" "$emit_log" \
        | grep -q '0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads'
    grep -A2 "Function properties for $symbol" "$emit_log" \
        | grep -q 'used 0 barriers, 4096 bytes smem'

    [[ $(grep -c 'F2F.BF16.F32' "$sass" || true) == "$expected_bf16_rounds" ]]
    [[ $(grep -c 'F2FP.SATFINITE.E4M3' "$sass" || true) == "$expected_fp8_converts" ]]
    [[ $(grep -c 'SHFL.DOWN' "$sass" || true) == "$expected_shuffle_down" ]]
    [[ $(grep -c 'SHFL.IDX' "$sass" || true) == "$expected_shuffle_index" ]]
    [[ $(grep -c '[[:space:]]STS' "$sass" || true) == "$expected_shared_stores" ]]
    [[ $(grep -c '[[:space:]]LDS' "$sass" || true) == "$expected_shared_loads" ]]
    [[ $(grep -c '[[:space:]]STG.E' "$sass" || true) == "$expected_global_stores" ]]
    [[ $(grep -c 'STG.E.U16' "$sass" || true) == 0 ]]
    [[ $(grep -cE '[[:space:]](ATOM|LDL|STL|BAR\.SYNC|HMMA|QMMA)' "$sass" || true) == 0 ]]
    echo "$symbol: instructions=$instructions registers=$registers shared=$shared stack=0 local=0 spills=0"
}

# Dual emit repeats the exact four-warp quant reduction twice. The down symbol
# has twelve additional BF16 barriers for post gate/up and SwiGLU before its
# four final H128 boundary conversions.
check_emitter exl3_w2a8_h128_pre_dual_emit_h4096 36 8 8 40 2 2 8 4
check_emitter exl3_w2a8_h128_post_silu_pre_emit_h2048 40 16 4 20 1 1 4 2

digest=$(
    for cubin in "$w2a8_probe_dir"/{down,gu,h128-emit}.cubin; do
        sha256sum "$cubin" | awk '{print $1}'
    done | sha256sum | awk '{print $1}'
)
echo "W2A8 GU/down: <=103 registers, 7168 B shared, zero stack/local/spills/atomics"
echo "retained K64-stage body: 16 E4M3 QMMA, 16 SHFL.IDX, 16 native E4M3 conversions"
echo "H128 emitters: exact BF16 boundaries and four-warp K128 reduction topology; zero BF16 global stores"
echo "compile-only cubin_set_sha256=$digest"
