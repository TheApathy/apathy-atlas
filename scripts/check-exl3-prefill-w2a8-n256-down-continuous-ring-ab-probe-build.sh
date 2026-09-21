#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only

set -euo pipefail
export LC_ALL=C

# BEGIN N256 down continuous ring A/B threshold
if [[ $# -ne 1 ]]; then
    echo "usage: $0 <min_n256_down_continuous_ring_speedup>" >&2
    exit 2
fi
requested_min_speedup=$1
decimal_pattern='^(0|[1-9][0-9]*)([.][0-9]*[1-9])?$'
if [[ ! $requested_min_speedup =~ $decimal_pattern ]] ||
    ! min_speedup=$(awk -v min_speedup="$requested_min_speedup" '
        BEGIN {
            if (!(min_speedup >= 1.01 && min_speedup <= 100.0)) exit 1
            printf "%.17g", min_speedup
        }
    '); then
    echo "invalid canonical decimal threshold; require >=1.01 and <=100" >&2
    exit 2
fi
for injected_flags in NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS; do
    if [[ -n ${!injected_flags:-} ]]; then
        echo "refusing unreceipted nvcc flags from $injected_flags" >&2
        exit 2
    fi
done
readonly requested_min_speedup min_speedup
# END N256 down continuous ring A/B threshold

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cuda_root_path=${CUDA_ROOT_PATH:-/usr/local/cuda}
nvcc_bin=${NVCC_BIN:-"$cuda_root_path/bin/nvcc"}
cuobjdump_bin=${CUOBJDUMP_BIN:-"$cuda_root_path/bin/cuobjdump"}
host_cxx_bin=${HOST_CXX_BIN:-/usr/bin/g++}
timeout_bin=${TIMEOUT_BIN:-/usr/bin/timeout}

if [[ -n ${W2A8_N256_DOWN_CONTINUOUS_RING_AB_OUTPUT_DIR:-} ]]; then
    build_dir=$W2A8_N256_DOWN_CONTINUOUS_RING_AB_OUTPUT_DIR
    if [[ -e $build_dir || -L $build_dir ]]; then
        echo "refusing existing W2A8_N256_DOWN_CONTINUOUS_RING_AB_OUTPUT_DIR: $build_dir" >&2
        exit 2
    fi
    mkdir -m 0700 -- "$build_dir"
    persist_probe=1
else
    build_dir=$(mktemp -d /tmp/atlas-exl3-n256-down-continuous-ring-ab.XXXXXX)
    trap 'rm -rf -- "$build_dir"' EXIT
    persist_probe=0
fi

for tool in "$nvcc_bin" "$cuobjdump_bin" "$host_cxx_bin" "$timeout_bin"; do
    if [[ ! -x $tool ]]; then
        echo "missing N256-down continuous-ring A/B tool: $tool" >&2
        exit 2
    fi
done

# BEGIN N256 down continuous ring A/B immutable inputs
harness_file="$repo_root/kernels/gb10/experiments/exl3_w2a8_composed_n128_n256_probe.cu"
emitter_wrapper="$repo_root/kernels/gb10/deepseek-v4-flash/nvfp4/exl3_w2a8_h128_emit.cu"
n64_gu_wrapper="$repo_root/kernels/gb10/deepseek-v4-flash/nvfp4/exl3_w2a8_grouped_prefill_k2_gu.cu"
n64_down_wrapper="$repo_root/kernels/gb10/deepseek-v4-flash/nvfp4/exl3_w2a8_grouped_prefill_k2_down.cu"
fused_wrapper="$repo_root/kernels/gb10/deepseek-v4-flash/nvfp4/exl3_w2a8_fused_gu_down_emit_n128.cu"
n256_down_wrapper="$repo_root/kernels/gb10/deepseek-v4-flash/nvfp4/exl3_w2a8_grouped_prefill_n256_k2_down.cu"
emitter_component="$repo_root/kernels/gb10/experiments/exl3_w2a8_h128_emit.cu"
n64_component="$repo_root/kernels/gb10/experiments/exl3_w2a8_grouped_prefill.cu"
fused_component="$repo_root/kernels/gb10/experiments/exl3_w2a8_fused_gu_down_emit_n128.cu"
n256_component="$repo_root/kernels/gb10/experiments/exl3_w2a8_grouped_prefill_n256.cu"
common_exl3="$repo_root/kernels/gb10/common/exl3_gemv.cu"
common_blend="$repo_root/kernels/gb10/common/moe_batched_blend.cuh"
dispatch_source="$repo_root/crates/spark-model/src/layers/moe/forward_prefill_exl3_w2a8.rs"
prefill_source="$repo_root/crates/spark-model/src/layers/moe/forward_prefill.rs"
prefill_exl3_source="$repo_root/crates/spark-model/src/layers/moe/forward_prefill_exl3.rs"
prefill_tail_source="$repo_root/crates/spark-model/src/layers/moe/forward_prefill_exl3_tail.rs"
prefill_phase_source="$repo_root/crates/spark-model/src/layers/moe/forward_prefill_phase.rs"
state_source="$repo_root/crates/spark-model/src/layers/moe/exl3_decode.rs"
kernel_manifest="$repo_root/kernels/gb10/deepseek-v4-flash/nvfp4/KERNEL.toml"
candidate_source_test="$repo_root/crates/spark-model/tests/exl3_w2a8_n256_down_continuous_ring_model.rs"
candidate_sass_gate="$repo_root/scripts/check-exl3-prefill-w2a8-n256-down-continuous-ring-sass.sh"
build_script="$repo_root/scripts/check-exl3-prefill-w2a8-n256-down-continuous-ring-ab-probe-build.sh"
dependencies=(
    "$harness_file" "$emitter_wrapper" "$n64_gu_wrapper" "$n64_down_wrapper"
    "$fused_wrapper" "$n256_down_wrapper" "$emitter_component" "$n64_component"
    "$fused_component" "$n256_component" "$common_exl3" "$common_blend"
    "$dispatch_source" "$prefill_source" "$prefill_exl3_source"
    "$prefill_tail_source" "$prefill_phase_source" "$state_source"
    "$kernel_manifest" "$candidate_source_test" "$candidate_sass_gate"
    "$build_script"
)
declare -A dependency_hashes
for dependency in "${dependencies[@]}"; do
    if [[ ! -f $dependency ]]; then
        echo "missing N256-down continuous-ring A/B input: $dependency" >&2
        exit 2
    fi
    dependency_hashes[$dependency]=$(sha256sum "$dependency" | awk '{print $1}')
done

nvcc_real_path=$(realpath "$nvcc_bin")
cuobjdump_real_path=$(realpath "$cuobjdump_bin")
host_cxx_real_path=$(realpath "$host_cxx_bin")
timeout_real_path=$(realpath "$timeout_bin")
nvcc_version=$("$nvcc_bin" --version | tail -n 1)
cuobjdump_version=$("$cuobjdump_bin" --version | tail -n 1)
host_cxx_version=$("$host_cxx_bin" --version | head -n 1)
timeout_version=$("$timeout_bin" --version | head -n 1)
nvcc_binary_hash=$(sha256sum "$nvcc_bin" | awk '{print $1}')
cuobjdump_binary_hash=$(sha256sum "$cuobjdump_bin" | awk '{print $1}')
host_cxx_binary_hash=$(sha256sum "$host_cxx_bin" | awk '{print $1}')
timeout_binary_hash=$(sha256sum "$timeout_bin" | awk '{print $1}')
git_commit=$(git -C "$repo_root" rev-parse HEAD)
git_status_hash=$(git -C "$repo_root" status --porcelain=v1 -uall | sha256sum | awk '{print $1}')
# END N256 down continuous ring A/B immutable inputs

# BEGIN generated experiment-local N256 wrapper
n256_experiment_wrapper="$build_dir/n256-down-experiment-wrapper.cu"
cat >"$n256_experiment_wrapper" <<GENERATED_N256_WRAPPER
// SPDX-License-Identifier: AGPL-3.0-only
// Receipt-bound N256 down wrapper. Never installed in the production registry.
#define W2A8_FIXED_N 4096
#define W2A8_FIXED_K 2048
#define W2A8_KERNEL_NAME exl3_w2a8_grouped_prefill_n256_k2_down
#ifndef W2A8_PACKED_E4M3_CANDIDATE
#error "W2A8_PACKED_E4M3_CANDIDATE must be receipt-bound"
#endif
#if W2A8_PACKED_E4M3_CANDIDATE != 1
#error "W2A8_PACKED_E4M3_CANDIDATE must equal 1"
#endif
#ifndef W2A8_N256_SINGLE_WARP_ROUTE_GUARD_CANDIDATE
#error "W2A8_N256_SINGLE_WARP_ROUTE_GUARD_CANDIDATE must be receipt-bound"
#endif
#if W2A8_N256_SINGLE_WARP_ROUTE_GUARD_CANDIDATE != 0
#error "W2A8_N256_SINGLE_WARP_ROUTE_GUARD_CANDIDATE must equal 0"
#endif
#ifndef W2A8_N256_DOWN_DOUBLE_BUFFER_CANDIDATE
#error "W2A8_N256_DOWN_DOUBLE_BUFFER_CANDIDATE must be receipt-bound"
#endif
#if W2A8_N256_DOWN_DOUBLE_BUFFER_CANDIDATE != 1
#error "W2A8_N256_DOWN_DOUBLE_BUFFER_CANDIDATE must equal 1"
#endif
#ifndef W2A8_N256_DOWN_CONTINUOUS_RING_CANDIDATE
#error "W2A8_N256_DOWN_CONTINUOUS_RING_CANDIDATE must be receipt-bound"
#endif
#include "$n256_component"
GENERATED_N256_WRAPPER
generated_wrapper_sha256=$(sha256sum "$n256_experiment_wrapper" | awk '{print $1}')
# END generated experiment-local N256 wrapper

# BEGIN N256 down continuous ring two binary build
manifest="$build_dir/build-manifest.txt"
{
    echo "receipt_format=atlas-w2a8-n256-down-continuous-ring-composed-ab-v1"
    echo "min_n256_down_continuous_ring_speedup=$min_speedup"
    echo "min_n256_down_continuous_ring_speedup_binary64=$min_speedup"
    echo "internal_composed_speed_gate=parity-only"
    echo "internal_composed_speedup_report_reference=1.01"
    echo "geometry=tokens2410_rows14460_topk6_experts256_hidden4096_intermediate2048"
    echo "timing_route=checkpoint-like-synthetic-v1"
    echo "packed_e4m3=enabled_both_arms"
    echo "fixed_n256_route_guard=0"
    echo "fixed_packed_e4m3=1"
    echo "fixed_n256_down_double_buffer=1"
    echo "fixed_fused_gu_n128_double_buffer=1"
    echo "fixed_fused_gu_n128_continuous_ring=1"
    echo "runtime_device_identity=measured_all_four_runs"
    echo "generated_n256_wrapper_sha256=$generated_wrapper_sha256"
    echo "only_ab_compile_factor=W2A8_N256_DOWN_CONTINUOUS_RING_CANDIDATE"
    echo "variant incumbent compile_define=-DW2A8_N256_DOWN_CONTINUOUS_RING_CANDIDATE=0"
    echo "variant candidate compile_define=-DW2A8_N256_DOWN_CONTINUOUS_RING_CANDIDATE=1"
    echo "git_commit=$git_commit"
    echo "git_status_sha256=$git_status_hash"
    for dependency in "${dependencies[@]}"; do
        relative=${dependency#"$repo_root/"}
        echo "source_sha256 $relative ${dependency_hashes[$dependency]}"
    done
    echo "nvcc_path=$nvcc_real_path"
    echo "nvcc_version=$nvcc_version"
    echo "nvcc_binary_sha256=$nvcc_binary_hash"
    echo "cuobjdump_path=$cuobjdump_real_path"
    echo "cuobjdump_version=$cuobjdump_version"
    echo "cuobjdump_binary_sha256=$cuobjdump_binary_hash"
    echo "host_cxx_path=$host_cxx_real_path"
    echo "host_cxx_version=$host_cxx_version"
    echo "host_cxx_binary_sha256=$host_cxx_binary_hash"
    echo "timeout_path=$timeout_real_path"
    echo "timeout_version=$timeout_version"
    echo "timeout_binary_sha256=$timeout_binary_hash"
    echo "n256_experiment_wrapper=fixed_N4096_K2048_symbol_packed1_route_guard0_receipt_bound_selector_direct_component_include"
    echo "compile_command_template=<nvcc> -ccbin <host_cxx> -std=c++17 -O3 --fmad=false -arch=sm_121a -Xptxas=-v -DW2A8_PACKED_E4M3_CANDIDATE=1 -DW2A8_N256_SINGLE_WARP_ROUTE_GUARD_CANDIDATE=0 -DW2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE=1 -DW2A8_FUSED_GU_N128_CONTINUOUS_RING_CANDIDATE=1 -DW2A8_N256_DOWN_DOUBLE_BUFFER_CANDIDATE=1 -DW2A8_N256_DOWN_CONTINUOUS_RING_CANDIDATE=<0|1> -DW2A8_COMPOSED_PROBE_ENFORCE_SPEEDUP=0 <receipt-bound macros> <harness> <five production-shape wrappers including generated N256> -o <binary>"
    echo "sass_command_template=<cuobjdump> --dump-sass --function <target-symbol> <binary>"
    echo "resource_command_template=<cuobjdump> --dump-resource-usage <binary>"
    echo "extract_command_template=<cuobjdump> --extract-elf all <binary>"
} >"$manifest"
pair_id=$(sha256sum "$manifest" | awk '{print $1}')

n128_symbol=exl3_w2a8_fused_gu_down_emit_n128
n256_symbol=exl3_w2a8_grouped_prefill_n256_k2_down
target_symbols=("$n128_symbol" "$n256_symbol")
declare -A compile_commands compile_command_hashes binary_hashes
declare -A registers shared stack local_bytes instructions f16_converts f32_converts
declare -A barriers qmma shfl sass_hashes cubin_hashes cubin_counts
declare -A bf16_rounds fp8_converts swiglu_exp fma_ops
declare -A ldgsts ldgdepbar depbar_waits zfill
declare -A fadd fmul ffma hadd2 hmul2
artifact_files=("${manifest#"$build_dir/"}" "${n256_experiment_wrapper#"$build_dir/"}")

resource_value() {
    local file=$1 symbol=$2 field=$3
    awk -v symbol="$symbol" -v field="$field" '
        $0 == " Function " symbol ":" { getline; line = $0; found = 1 }
        END {
            if (!found) exit 1
            count = split(line, parts, " ")
            for (i = 1; i <= count; i++) {
                split(parts[i], pair, ":")
                if (pair[1] == field) { print pair[2]; exit }
            }
            exit 1
        }
    ' "$file"
}

build_variant() {
    local variant=$1 selector=$2
    local binary="$build_dir/exl3-w2a8-n256-down-continuous-ring-${variant}"
    local ptxas="$build_dir/${variant}.ptxas"
    local resources="$build_dir/${variant}.resources"
    local compile_command
    compile_command="$nvcc_bin -ccbin $host_cxx_bin -std=c++17 -O3 --fmad=false -arch=sm_121a -Xptxas=-v -DW2A8_PACKED_E4M3_CANDIDATE=1 -DW2A8_N256_SINGLE_WARP_ROUTE_GUARD_CANDIDATE=0 -DW2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE=1 -DW2A8_FUSED_GU_N128_CONTINUOUS_RING_CANDIDATE=1 -DW2A8_N256_DOWN_DOUBLE_BUFFER_CANDIDATE=1 -DW2A8_N256_DOWN_CONTINUOUS_RING_CANDIDATE=$selector -DW2A8_COMPOSED_PROBE_ENFORCE_SPEEDUP=0 '-DW2A8_COMPOSED_PROBE_BUILD_ID=\"$pair_id\"' -DW2A8_COMPOSED_PROBE_MIN_SPEEDUP=1.01 '-DW2A8_COMPOSED_PROBE_MIN_SPEEDUP_TEXT=\"1.01\"' $harness_file $emitter_wrapper $n64_gu_wrapper $n64_down_wrapper $fused_wrapper $n256_experiment_wrapper -o $binary"
    compile_commands[$variant]=$compile_command
    compile_command_hashes[$variant]=$(printf '%s' "$compile_command" | sha256sum | awk '{print $1}')
    "$nvcc_bin" -ccbin "$host_cxx_bin" -std=c++17 -O3 --fmad=false \
        -arch=sm_121a -Xptxas=-v -DW2A8_PACKED_E4M3_CANDIDATE=1 \
        -DW2A8_N256_SINGLE_WARP_ROUTE_GUARD_CANDIDATE=0 \
        -DW2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE=1 \
        -DW2A8_FUSED_GU_N128_CONTINUOUS_RING_CANDIDATE=1 \
        -DW2A8_N256_DOWN_DOUBLE_BUFFER_CANDIDATE=1 \
        "-DW2A8_N256_DOWN_CONTINUOUS_RING_CANDIDATE=$selector" \
        -DW2A8_COMPOSED_PROBE_ENFORCE_SPEEDUP=0 \
        "-DW2A8_COMPOSED_PROBE_BUILD_ID=\"$pair_id\"" \
        -DW2A8_COMPOSED_PROBE_MIN_SPEEDUP=1.01 \
        '-DW2A8_COMPOSED_PROBE_MIN_SPEEDUP_TEXT="1.01"' \
        "$harness_file" "$emitter_wrapper" "$n64_gu_wrapper" \
        "$n64_down_wrapper" "$fused_wrapper" "$n256_experiment_wrapper" \
        -o "$binary" 2>"$ptxas"
    binary_hashes[$variant]=$(sha256sum "$binary" | awk '{print $1}')
    "$cuobjdump_bin" --dump-resource-usage "$binary" >"$resources"
    artifact_files+=("${binary#"$build_dir/"}" "${ptxas#"$build_dir/"}" "${resources#"$build_dir/"}")

    local symbol sass key value
    for symbol in "${target_symbols[@]}"; do
        sass="$build_dir/${variant}.${symbol}.sass"
        "$cuobjdump_bin" --dump-sass --function "$symbol" "$binary" >"$sass"
        grep -Fq "$symbol" "$sass"
        key="$variant|$symbol"
        sass_hashes[$key]=$(sha256sum "$sass" | awk '{print $1}')
        registers[$key]=$(resource_value "$resources" "$symbol" REG)
        stack[$key]=$(resource_value "$resources" "$symbol" STACK)
        local_bytes[$key]=$(resource_value "$resources" "$symbol" LOCAL)
        shared[$key]=$(resource_value "$resources" "$symbol" SHARED)
        instructions[$key]=$(grep -cE '^[[:space:]]*/\*[0-9a-f]+\*/' "$sass" || true)
        f16_converts[$key]=$(grep -cE 'F2FP\.SATFINITE\.E4M3\.F16' "$sass" || true)
        f32_converts[$key]=$(grep -cE 'F2FP\.SATFINITE\.E4M3\.F32' "$sass" || true)
        barriers[$key]=$(grep -cE 'BAR\.SYNC' "$sass" || true)
        qmma[$key]=$(grep -cE 'QMMA\.16832\.F32\.E4M3\.E4M3' "$sass" || true)
        shfl[$key]=$(grep -cE 'SHFL\.' "$sass" || true)
        bf16_rounds[$key]=$(grep -cE 'F2F(P)?\.BF16\.F32' "$sass" || true)
        fp8_converts[$key]=$(grep -cE 'F2FP\.SATFINITE\.E4M3' "$sass" || true)
        swiglu_exp[$key]=$(grep -cE 'MUFU\.EX2' "$sass" || true)
        fma_ops[$key]=$(grep -cE '[[:space:]](FFMA|HFMA2)' "$sass" || true)
        ldgsts[$key]=$(grep -cE 'LDGSTS' "$sass" || true)
        ldgdepbar[$key]=$(grep -cE 'LDGDEPBAR' "$sass" || true)
        depbar_waits[$key]=$(grep -cE 'DEPBAR\.LE SB0, 0x0' "$sass" || true)
        zfill[$key]=$(grep -cE 'LDGSTS[^;]*ZFILL' "$sass" || true)
        fadd[$key]=$(grep -cE '[[:space:]]FADD' "$sass" || true)
        fmul[$key]=$(grep -cE '[[:space:]]FMUL' "$sass" || true)
        ffma[$key]=$(grep -cE '[[:space:]]FFMA' "$sass" || true)
        hadd2[$key]=$(grep -cE '[[:space:]]HADD2' "$sass" || true)
        hmul2[$key]=$(grep -cE '[[:space:]]HMUL2' "$sass" || true)
        for value in "${registers[$key]}" "${stack[$key]}" "${local_bytes[$key]}" \
            "${shared[$key]}" "${instructions[$key]}" "${f16_converts[$key]}" \
            "${f32_converts[$key]}" "${barriers[$key]}" "${qmma[$key]}" \
            "${shfl[$key]}" "${bf16_rounds[$key]}" "${fp8_converts[$key]}" \
            "${swiglu_exp[$key]}" "${fma_ops[$key]}" "${ldgsts[$key]}" \
            "${ldgdepbar[$key]}" "${depbar_waits[$key]}" "${zfill[$key]}" \
            "${fadd[$key]}" "${fmul[$key]}" "${ffma[$key]}" \
            "${hadd2[$key]}" "${hmul2[$key]}"; do
            [[ $value =~ ^[0-9]+$ ]]
        done
        [[ ${stack[$key]} == 0 && ${local_bytes[$key]} == 0 ]]
        (( instructions[$key] > 0 ))
        grep -A1 "Function properties for $symbol" "$ptxas" \
            | grep -q '0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads'
        [[ $(grep -cE '[[:space:]](ATOM|RED|LDL|STL)' "$sass" || true) == 0 ]]
        artifact_files+=("${sass#"$build_dir/"}")
    done

    local extract_dir="$build_dir/${variant}.cubins"
    mkdir "$extract_dir"
    (
        cd "$extract_dir"
        "$cuobjdump_bin" --extract-elf all "$binary" >/dev/null
    )
    local cubin_count=0 cubin cubin_key
    while IFS= read -r cubin; do
        cubin_key="$variant|$cubin_count"
        cubin_hashes[$cubin_key]=$(sha256sum "$cubin" | awk '{print $1}')
        artifact_files+=("${cubin#"$build_dir/"}")
        cubin_count=$((cubin_count + 1))
    done < <(find "$extract_dir" -maxdepth 1 -type f -name '*.cubin' | sort)
    [[ $cubin_count == 7 ]]
    cubin_counts[$variant]=$cubin_count
}

build_variant incumbent 0
build_variant candidate 1

[[ ${cubin_counts[incumbent]} == "${cubin_counts[candidate]}" ]]
different_cubins=0
for ((cubin_index = 0; cubin_index < cubin_counts[incumbent]; ++cubin_index)); do
    incumbent_cubin_key="incumbent|$cubin_index"
    candidate_cubin_key="candidate|$cubin_index"
    if [[ ${cubin_hashes[$incumbent_cubin_key]} != "${cubin_hashes[$candidate_cubin_key]}" ]]; then
        different_cubins=$((different_cubins + 1))
    fi
done
(( different_cubins == 1 ))

incumbent_n128="incumbent|$n128_symbol"
candidate_n128="candidate|$n128_symbol"
incumbent_n256="incumbent|$n256_symbol"
candidate_n256="candidate|$n256_symbol"
[[ ${sass_hashes[$incumbent_n128]} == "${sass_hashes[$candidate_n128]}" ]]
[[ ${sass_hashes[$incumbent_n256]} != "${sass_hashes[$candidate_n256]}" ]]
for key in "$incumbent_n128" "$candidate_n128" "$incumbent_n256" "$candidate_n256"; do
    (( f16_converts[$key] > 0 ))
    [[ ${stack[$key]} == 0 && ${local_bytes[$key]} == 0 ]]
done
for field in registers shared stack local_bytes instructions f16_converts \
    f32_converts barriers qmma shfl bf16_rounds fp8_converts swiglu_exp \
    fma_ops ldgsts ldgdepbar depbar_waits zfill fadd fmul ffma hadd2 hmul2; do
    declare -n metric_ref=$field
    [[ ${metric_ref[$incumbent_n128]} == "${metric_ref[$candidate_n128]}" ]]
    unset -n metric_ref
done
[[ ${registers[$incumbent_n128]} == 128 ]]
[[ ${shared[$incumbent_n128]} == 48128 ]]
[[ ${instructions[$incumbent_n128]} == 2784 ]]
[[ ${qmma[$incumbent_n128]} == 32 ]]
[[ ${bf16_rounds[$incumbent_n128]} == 52 ]]
[[ ${fp8_converts[$incumbent_n128]} == 36 ]]
[[ ${swiglu_exp[$incumbent_n128]} == 4 ]]
[[ ${fma_ops[$incumbent_n128]} == 166 ]]
[[ ${shfl[$incumbent_n128]} == 193 ]]
[[ ${ldgsts[$incumbent_n128]} == 8 ]]
[[ ${ldgdepbar[$incumbent_n128]} == 4 ]]
[[ ${depbar_waits[$incumbent_n128]} == 4 ]]
[[ ${zfill[$incumbent_n128]} == 4 ]]

for field in qmma bf16_rounds f16_converts f32_converts shfl \
    fadd fmul ffma hadd2 hmul2; do
    declare -n metric_ref=$field
    [[ ${metric_ref[$incumbent_n256]} == "${metric_ref[$candidate_n256]}" ]]
    unset -n metric_ref
done
[[ ${qmma[$incumbent_n256]} == 16 ]]
[[ ${bf16_rounds[$incumbent_n256]} == 32 ]]
[[ ${f16_converts[$incumbent_n256]} == 16 ]]
[[ ${f32_converts[$incumbent_n256]} == 0 ]]
[[ ${shfl[$incumbent_n256]} == 16 ]]
[[ ${fadd[$incumbent_n256]} == 32 ]]
[[ ${fmul[$incumbent_n256]} == 40 ]]
[[ ${ffma[$incumbent_n256]} == 0 ]]
[[ ${hadd2[$incumbent_n256]} == 16 ]]
[[ ${hmul2[$incumbent_n256]} == 16 ]]
[[ ${registers[$incumbent_n256]} == 102 ]]
[[ ${shared[$incumbent_n256]} == 19456 ]]
[[ ${instructions[$incumbent_n256]} == 896 ]]
[[ ${barriers[$incumbent_n256]} == 3 ]]
(( registers[$candidate_n256] <= 104 ))
[[ ${registers[$candidate_n256]} == 103 ]]
[[ ${shared[$candidate_n256]} == 19456 ]]
(( instructions[$candidate_n256] > instructions[$incumbent_n256] ))
(( instructions[$candidate_n256] <= 936 ))
[[ ${instructions[$candidate_n256]} == 920 ]]
[[ ${barriers[$candidate_n256]} == 3 ]]
[[ ${ldgsts[$incumbent_n256]} == 2 && ${ldgsts[$candidate_n256]} == 2 ]]
[[ ${ldgdepbar[$incumbent_n256]} == 2 && ${ldgdepbar[$candidate_n256]} == 2 ]]
[[ ${depbar_waits[$incumbent_n256]} == 2 && ${depbar_waits[$candidate_n256]} == 2 ]]
[[ ${zfill[$incumbent_n256]} == 1 ]]
[[ ${zfill[$candidate_n256]} == 2 ]]

awk '
    /LDGSTS/ { last_issue = NR }
    /LDGDEPBAR/ {
        if (!(last_issue < NR)) exit 1
        last_commit = NR
        commits++
    }
    /QMMA\.16832\.F32\.E4M3\.E4M3/ {
        if (commits == 2 && waits == 1) overlap_qmma++
    }
    /DEPBAR\.LE SB0, 0x0/ {
        if (!(last_commit < NR)) exit 1
        if (waits == 1 && overlap_qmma != 16) exit 1
        last_wait = NR
        waits++
    }
    /BAR\.SYNC/ {
        if (last_wait != 0) {
            if (!(last_wait < NR)) exit 1
            publications++
            last_wait = 0
        }
    }
    END {
        if (commits != 2 || waits != 2 || publications != 2 ||
            overlap_qmma != 16 || last_wait != 0) exit 1
    }
' "$build_dir/candidate.$n256_symbol.sass"

inputs_unchanged=1
for dependency in "${dependencies[@]}"; do
    if [[ $(sha256sum "$dependency" | awk '{print $1}') != "${dependency_hashes[$dependency]}" ]]; then
        echo "N256-down continuous-ring A/B input changed during compilation: $dependency" >&2
        inputs_unchanged=0
    fi
done
for tool_record in \
    "$nvcc_bin|$nvcc_real_path|$nvcc_binary_hash" \
    "$cuobjdump_bin|$cuobjdump_real_path|$cuobjdump_binary_hash" \
    "$host_cxx_bin|$host_cxx_real_path|$host_cxx_binary_hash" \
    "$timeout_bin|$timeout_real_path|$timeout_binary_hash"; do
    IFS='|' read -r tool expected_path expected_hash <<<"$tool_record"
    if [[ $(realpath "$tool") != "$expected_path" ]] ||
        [[ $(sha256sum "$tool" | awk '{print $1}') != "$expected_hash" ]]; then
        echo "N256-down continuous-ring A/B tool changed during compilation: $tool" >&2
        inputs_unchanged=0
    fi
done
current_commit=$(git -C "$repo_root" rev-parse HEAD)
current_status_hash=$(git -C "$repo_root" status --porcelain=v1 -uall | sha256sum | awk '{print $1}')
if [[ $(sha256sum "$n256_experiment_wrapper" | awk '{print $1}') != "$generated_wrapper_sha256" ]]; then
    echo "generated N256 experiment wrapper changed during compilation" >&2
    inputs_unchanged=0
fi
if [[ $current_commit != "$git_commit" || $current_status_hash != "$git_status_hash" ]]; then
    echo "repository identity changed during N256-down continuous-ring A/B compilation" >&2
    inputs_unchanged=0
fi
[[ $inputs_unchanged == 1 ]]

receipt="$build_dir/build-receipt.txt"
{
    cat "$manifest"
    echo "pair_id=$pair_id"
    for variant in incumbent candidate; do
        echo "compile_command $variant ${compile_commands[$variant]}"
        echo "compile_command_sha256 $variant ${compile_command_hashes[$variant]}"
        echo "binary_sha256 $variant ${binary_hashes[$variant]}"
    done
    for relative in "${artifact_files[@]}"; do
        artifact_hash=$(sha256sum "$build_dir/$relative" | awk '{print $1}')
        case $relative in
            *.cubin) echo "cubin_sha256 $relative $artifact_hash" ;;
            *.sass) echo "sass_sha256 $relative $artifact_hash" ;;
            *.resources) echo "resource_sha256 $relative $artifact_hash" ;;
        esac
        echo "artifact_sha256 $relative $artifact_hash"
    done
    for variant in incumbent candidate; do
        for symbol in "${target_symbols[@]}"; do
            key="$variant|$symbol"
            echo "resource_census variant=$variant symbol=$symbol REG:${registers[$key]} STACK:${stack[$key]} SHARED:${shared[$key]} LOCAL:${local_bytes[$key]} spills=0"
            echo "sass_census variant=$variant symbol=$symbol instructions=${instructions[$key]} e4m3_f16=${f16_converts[$key]} e4m3_f32=${f32_converts[$key]} barriers=${barriers[$key]} qmma=${qmma[$key]} bf16_rounds=${bf16_rounds[$key]} fp8_converts=${fp8_converts[$key]} swiglu_exp=${swiglu_exp[$key]} fma=${fma_ops[$key]} shfl=${shfl[$key]} fadd=${fadd[$key]} fmul=${fmul[$key]} ffma=${ffma[$key]} hadd2=${hadd2[$key]} hmul2=${hmul2[$key]} ldgsts=${ldgsts[$key]} ldgdepbar=${ldgdepbar[$key]} depbar_waits=${depbar_waits[$key]} zfill=${zfill[$key]}"
        done
    done
    echo "ab_delta symbol=$n128_symbol instructions=$((instructions[$candidate_n128] - instructions[$incumbent_n128])) registers=$((registers[$candidate_n128] - registers[$incumbent_n128])) shared=$((shared[$candidate_n128] - shared[$incumbent_n128])) barriers=$((barriers[$candidate_n128] - barriers[$incumbent_n128]))"
    echo "ab_delta symbol=$n256_symbol instructions=$((instructions[$candidate_n256] - instructions[$incumbent_n256])) registers=$((registers[$candidate_n256] - registers[$incumbent_n256])) shared=$((shared[$candidate_n256] - shared[$incumbent_n256])) barriers=$((barriers[$candidate_n256] - barriers[$incumbent_n256]))"
    echo "cubin_comparison total=${cubin_counts[incumbent]} equal=$((cubin_counts[incumbent] - different_cubins)) different=$different_cubins"
    echo "continuous_ring_sass=PASS packed_e4m3_both=PASS fixed_n256_down_double_buffer1=PASS fixed_route_guard0=PASS fixed_fused_n128_continuous_ring1=PASS unchanged_n128_sass=PASS changed_n256_sass=PASS async_order=PASS arithmetic_census=exact stack=0 local=0 spills=0 atomics=0"
} >"$receipt"
receipt_hash=$(sha256sum "$receipt" | awk '{print $1}')
# END N256 down continuous ring two binary build

# BEGIN N256 down continuous ring generated A/B runner
runner="$build_dir/run-n256-down-continuous-ring-ab.sh"
{
    echo '#!/usr/bin/env bash'
    echo '# SPDX-License-Identifier: AGPL-3.0-only'
    echo 'set -euo pipefail'
    printf 'expected_receipt_sha256=%q\n' "$receipt_hash"
    printf 'expected_pair_id=%q\n' "$pair_id"
    printf 'expected_min_speedup=%q\n' "$min_speedup"
    printf 'expected_timeout_path=%q\n' "$timeout_real_path"
    printf 'expected_timeout_sha256=%q\n' "$timeout_binary_hash"
    cat <<'RUNNER'
export LC_ALL=C
if [[ $# -ne 0 ]]; then
    echo "usage: $0" >&2
    exit 2
fi
runner_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
receipt="$runner_dir/build-receipt.txt"
timeout_bin=$expected_timeout_path

verify_artifacts() {
    local actual_receipt_sha256 relative expected actual
    actual_receipt_sha256=$(sha256sum "$receipt" | awk '{print $1}')
    if [[ $actual_receipt_sha256 != "$expected_receipt_sha256" ]] ||
        [[ $(grep -Fxc "pair_id=$expected_pair_id" "$receipt") -ne 1 ]] ||
        [[ $(grep -Fxc "min_n256_down_continuous_ring_speedup=$expected_min_speedup" "$receipt") -ne 1 ]] ||
        [[ $(grep -Fxc "packed_e4m3=enabled_both_arms" "$receipt") -ne 1 ]] ||
        [[ $(grep -Fxc "fixed_n256_route_guard=0" "$receipt") -ne 1 ]] ||
        [[ $(grep -Fxc "fixed_packed_e4m3=1" "$receipt") -ne 1 ]] ||
        [[ $(grep -Fxc "fixed_n256_down_double_buffer=1" "$receipt") -ne 1 ]] ||
        [[ $(grep -Fxc "fixed_fused_gu_n128_double_buffer=1" "$receipt") -ne 1 ]] ||
        [[ $(grep -Fxc "fixed_fused_gu_n128_continuous_ring=1" "$receipt") -ne 1 ]] ||
        [[ $(grep -Fxc "runtime_device_identity=measured_all_four_runs" "$receipt") -ne 1 ]] ||
        [[ $(grep -Fxc "only_ab_compile_factor=W2A8_N256_DOWN_CONTINUOUS_RING_CANDIDATE" "$receipt") -ne 1 ]] ||
        [[ $(sha256sum "$timeout_bin" | awk '{print $1}') != "$expected_timeout_sha256" ]]; then
        echo "N256-down continuous-ring A/B receipt or tool identity mismatch" >&2
        exit 2
    fi
    while read -r _ relative expected; do
        [[ -n ${relative:-} && -n ${expected:-} ]] || continue
        actual=$(sha256sum "$runner_dir/$relative" | awk '{print $1}')
        if [[ $actual != "$expected" ]]; then
            echo "N256-down continuous-ring A/B artifact mismatch: $relative" >&2
            exit 2
        fi
    done < <(awk '$1 == "artifact_sha256" { print }' "$receipt")
}

verify_artifacts
probe_tmp=$(mktemp -d "${TMPDIR:-/tmp}/atlas-n256-down-continuous-ring-ab-run.XXXXXX")
trap 'rm -rf -- "$probe_tmp"' EXIT
order=(incumbent candidate candidate incumbent)
declare -a candidate_ms device_lines input_hashes output_hashes
number='([0-9]+([.][0-9]+)?|[.][0-9]+)([eE][+-]?[0-9]+)?'

preserve_failure() {
    local reason=$1 evidence
    local failure_dir="$runner_dir/failed-run-$$"
    mkdir -m 0700 -- "$failure_dir"
    for evidence in "$probe_tmp"/stdout-* "$probe_tmp"/stderr-*; do
        [[ -f $evidence ]] && cp -- "$evidence" "$failure_dir/"
    done
    printf 'pair_id=%s\nreason=%s\n' "$expected_pair_id" "$reason" \
        >"$failure_dir/failure.txt"
    echo "bounded failure evidence preserved: $failure_dir" >&2
    for evidence in "$failure_dir"/stderr-*; do
        if [[ -s $evidence ]]; then
            echo "--- $(basename "$evidence") ---" >&2
            sed -n '1,64p' "$evidence" >&2
        fi
    done
}

for index in 0 1 2 3; do
    variant=${order[$index]}
    binary="$runner_dir/exl3-w2a8-n256-down-continuous-ring-${variant}"
    set +e
    (
        ulimit -f 64
        "$timeout_bin" --signal=TERM --kill-after=5s 900s "$binary"
    ) >"$probe_tmp/stdout-$index" 2>"$probe_tmp/stderr-$index"
    probe_rc=$?
    set -e
    stdout_bytes=$(wc -c <"$probe_tmp/stdout-$index")
    stderr_bytes=$(wc -c <"$probe_tmp/stderr-$index")
    stdout_lines=$(wc -l <"$probe_tmp/stdout-$index")
    if [[ $probe_rc -ne 0 || $stdout_bytes -gt 8192 || $stderr_bytes -ne 0 ||
        $stdout_lines -ne 13 ]]; then
        preserve_failure "process-contract-$index-$variant-rc$probe_rc"
        echo "N256-down continuous-ring $variant run rejected: rc=$probe_rc stdout_bytes=$stdout_bytes stderr_bytes=$stderr_bytes stdout_lines=$stdout_lines" >&2
        exit 1
    fi
    mapfile -t lines <"$probe_tmp/stdout-$index"
    if [[ ${lines[0]} != "build_id=$expected_pair_id" ]] ||
        [[ ${lines[1]} != "variant=candidate packed_e4m3=1" ]] ||
        [[ ! ${lines[2]} =~ ^device_uuid=[0-9a-f]{32}\ driver=[0-9]+\ runtime=[0-9]+$ ]] ||
        [[ ${lines[3]} != 'tokens=2410 rows=14460 topk=6 experts=256 hidden=4096 intermediate=2048' ]] ||
        [[ ${lines[4]} != 'routes=balanced,empty-expert,skewed,m64-boundaries,checkpoint-like-synthetic-v1 timing_route=checkpoint-like-synthetic-v1 offsets=257 poison_passes=8' ]] ||
        [[ ${lines[5]} != 'threshold min_speedup=1.01 binary64=1.01 enforcement=parity-only' ]] ||
        [[ ! ${lines[6]} =~ ^baseline_ms=$number\ candidate_ms=$number\ speedup=$number\ abba_samples=4\ timed=whole-chain$ ]] ||
        [[ ${lines[7]} != 'intermediate_fp8=exact intermediate_scale=exact' ]] ||
        [[ ${lines[8]} != 'raw_bf16=exact final_bf16=exact production_alias=exact' ]] ||
        [[ ${lines[9]} != 'guards=clean inputs=immutable malformed_cases=16 all_offsets=validated' ]] ||
        [[ ! ${lines[10]} =~ ^input_hash=[0-9a-f]{16}$ ]] ||
        [[ ! ${lines[11]} =~ ^output_hash=[0-9a-f]{16}$ ]] ||
        [[ ${lines[12]} != 'result=PASS' ]]; then
        preserve_failure "output-contract-$index-$variant"
        echo "N256-down continuous-ring $variant output contract mismatch" >&2
        exit 1
    fi
    candidate_ms[$index]=$(awk -F'[ =]' '{ print $4 }' <<<"${lines[6]}")
    device_lines[$index]=${lines[2]}
    input_hashes[$index]=${lines[10]#input_hash=}
    output_hashes[$index]=${lines[11]#output_hash=}
done

for index in 1 2 3; do
    if [[ ${device_lines[$index]} != "${device_lines[0]}" ]] ||
        [[ ${input_hashes[$index]} != "${input_hashes[0]}" ]] ||
        [[ ${output_hashes[$index]} != "${output_hashes[0]}" ]]; then
        preserve_failure "cross-run-identity"
        echo "N256-down continuous-ring A/B device, input, or output identity mismatch" >&2
        exit 1
    fi
done

if ! n256_down_continuous_ring_speedup=$(awk \
    -v incumbent0="${candidate_ms[0]}" -v candidate0="${candidate_ms[1]}" \
    -v candidate1="${candidate_ms[2]}" -v incumbent1="${candidate_ms[3]}" \
    -v threshold="$expected_min_speedup" '
        BEGIN {
            incumbent = (incumbent0 + incumbent1) / 2.0
            candidate = (candidate0 + candidate1) / 2.0
            if (!(incumbent > 0.0 && candidate > 0.0)) exit 1
            speedup = incumbent / candidate
            if (!(speedup > 0.0) || speedup < threshold) exit 1
            printf "%.17g", speedup
        }
    '); then
    preserve_failure "outer-abba-threshold"
    echo "N256-down continuous-ring ABBA speedup below receipt-bound threshold" >&2
    exit 1
fi
if [[ ! $n256_down_continuous_ring_speedup =~ ^$number$ ]]; then
    preserve_failure "outer-abba-non-finite"
    echo "N256-down continuous-ring ABBA produced a non-finite speedup" >&2
    exit 1
fi

verify_artifacts
actual_receipt_sha256=$(sha256sum "$receipt" | awk '{print $1}')
printf 'receipt_sha256=%s\n' "$actual_receipt_sha256"
printf 'pair_id=%s\n' "$expected_pair_id"
printf 'order=incumbent,candidate,candidate,incumbent timed=composed-chain\n'
printf '%s\n' "${device_lines[0]}"
printf 'n256_down_continuous_ring_selectors=incumbent:0,candidate:1 packed_e4m3=1,1 n256_down_double_buffer=1,1 n256_route_guard=0,0 fused_n128_continuous_ring=1,1\n'
printf 'incumbent_candidate_ms=%s,%s candidate_candidate_ms=%s,%s\n' \
    "${candidate_ms[0]}" "${candidate_ms[3]}" "${candidate_ms[1]}" "${candidate_ms[2]}"
printf 'n256_down_continuous_ring_speedup=%s threshold=%s abba_samples=4\n' \
    "$n256_down_continuous_ring_speedup" "$expected_min_speedup"
printf 'candidate_throughput_over_incumbent=%s threshold=%s\n' \
    "$n256_down_continuous_ring_speedup" "$expected_min_speedup"
printf 'input_hash=%s output_hash=%s variant_outputs=exact\n' \
    "${input_hashes[0]}" "${output_hashes[0]}"
printf 'intermediate_fp8=exact intermediate_scale=exact redzones=clean inputs=immutable\n'
printf 'raw_bf16=exact final_bf16=exact production_alias=exact malformed_cases=16 all_offsets=validated\n'
grep -E '^(resource_census|sass_census|ab_delta|cubin_comparison|continuous_ring_sass=)' "$receipt"
printf 'result=PASS\n'
RUNNER
} >"$runner"
chmod 0755 "$runner"
runner_hash=$(sha256sum "$runner" | awk '{print $1}')
# END N256 down continuous ring generated A/B runner

echo "N256-down continuous-ring composed W2A8 A/B binaries compile for SM121a"
echo "packed E4M3 and prior double buffers are enabled in both arms; only the continuous-ring selector differs"
echo "runtime remains gated behind the generated immutable ABBA runner"
echo "pair_id=$pair_id"
echo "receipt_sha256=$receipt_hash"
echo "runner_sha256=$runner_hash"
if [[ $persist_probe == 1 ]]; then
    echo "probe runner: $runner"
    echo "build receipt: $receipt"
fi
