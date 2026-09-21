#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only

set -euo pipefail
export LC_ALL=C

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <min_n128_to_n256_speedup>" >&2
    exit 2
fi
requested_min_speedup=$1
decimal_pattern='^(0|[1-9][0-9]*)([.][0-9]*[1-9])?$'
if [[ ! $requested_min_speedup =~ $decimal_pattern ]] ||
    ! min_speedup=$(awk -v min_speedup="$requested_min_speedup" '
        BEGIN {
            if (!(min_speedup > 1.0 && min_speedup <= 100.0)) exit 1
            printf "%.17g", min_speedup
        }
    ') || [[ $requested_min_speedup != "$min_speedup" ]]; then
    echo "invalid canonical binary64 threshold" >&2
    exit 2
fi
readonly requested_min_speedup min_speedup

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cuda_root_path=${CUDA_ROOT_PATH:-/usr/local/cuda}
nvcc_bin=${NVCC_BIN:-"$cuda_root_path/bin/nvcc"}
cuobjdump_bin=${CUOBJDUMP_BIN:-"$cuda_root_path/bin/cuobjdump"}
host_cxx_bin=${HOST_CXX_BIN:-/usr/bin/g++}

if [[ -n ${W2A8_FUSED_GU_N256_PROBE_OUTPUT_DIR:-} ]]; then
    build_dir=$W2A8_FUSED_GU_N256_PROBE_OUTPUT_DIR
    if [[ -e $build_dir || -L $build_dir ]]; then
        echo "refusing existing W2A8_FUSED_GU_N256_PROBE_OUTPUT_DIR: $build_dir" >&2
        exit 2
    fi
    mkdir -m 0700 -- "$build_dir"
    persist_probe=1
else
    build_dir=$(mktemp -d /tmp/atlas-exl3-w2a8-fused-gu-n256-probe.XXXXXX)
    trap 'rm -rf -- "$build_dir"' EXIT
    persist_probe=0
fi

for tool in "$nvcc_bin" "$cuobjdump_bin" "$host_cxx_bin"; do
    if [[ ! -x $tool ]]; then
        echo "missing build tool: $tool" >&2
        exit 2
    fi
done
for injected_flags in NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS; do
    if [[ -n ${!injected_flags:-} ]]; then
        echo "refusing unreceipted nvcc flags from $injected_flags" >&2
        exit 2
    fi
done

# BEGIN N256 admission immutable inputs
harness_file="$repo_root/kernels/gb10/experiments/exl3_w2a8_fused_gu_down_emit_n256_probe.cu"
n128_probe_file="$repo_root/kernels/gb10/experiments/exl3_w2a8_fused_gu_down_emit_n128_probe.cu"
n128_component_file="$repo_root/kernels/gb10/experiments/exl3_w2a8_fused_gu_down_emit_n128.cu"
n256_component_file="$repo_root/kernels/gb10/experiments/exl3_w2a8_fused_gu_down_emit_n256.cu"
baseline_file="$repo_root/kernels/gb10/experiments/exl3_w2a8_grouped_prefill_n128.cu"
emitter_file="$repo_root/kernels/gb10/experiments/exl3_w2a8_h128_emit.cu"
common_exl3_file="$repo_root/kernels/gb10/common/exl3_gemv.cu"
common_moe_blend_file="$repo_root/kernels/gb10/common/moe_batched_blend.cuh"
n128_wrapper_file="$repo_root/kernels/gb10/deepseek-v4-flash/nvfp4/exl3_w2a8_fused_gu_down_emit_n128.cu"
n256_wrapper_file="$repo_root/kernels/gb10/deepseek-v4-flash/nvfp4/exl3_w2a8_fused_gu_down_emit_n256.cu"
kernel_build_file="$repo_root/crates/atlas-kernels/build.rs"
state_file="$repo_root/crates/spark-model/src/layers/moe/exl3_decode.rs"
dispatch_file="$repo_root/crates/spark-model/src/layers/moe/forward_prefill_exl3_w2a8.rs"
build_script="$repo_root/scripts/check-exl3-prefill-w2a8-fused-gu-n256-probe-build.sh"
dependencies=(
    "$harness_file"
    "$n128_probe_file"
    "$n128_component_file"
    "$n256_component_file"
    "$baseline_file"
    "$emitter_file"
    "$common_exl3_file"
    "$common_moe_blend_file"
    "$n128_wrapper_file"
    "$n256_wrapper_file"
    "$kernel_build_file"
    "$state_file"
    "$dispatch_file"
    "$build_script"
)
declare -A dependency_hashes
for dependency in "${dependencies[@]}"; do
    if [[ ! -f $dependency ]]; then
        echo "missing N256 admission input: $dependency" >&2
        exit 2
    fi
    dependency_hashes[$dependency]=$(sha256sum "$dependency" | awk '{print $1}')
done

nvcc_real_path=$(realpath "$nvcc_bin")
cuobjdump_real_path=$(realpath "$cuobjdump_bin")
host_cxx_real_path=$(realpath "$host_cxx_bin")
nvcc_version=$("$nvcc_bin" --version | tail -n 1)
cuobjdump_version=$("$cuobjdump_bin" --version | tail -n 1)
host_cxx_version=$("$host_cxx_bin" --version | head -n 1)
nvcc_binary_hash=$(sha256sum "$nvcc_bin" | awk '{print $1}')
cuobjdump_binary_hash=$(sha256sum "$cuobjdump_bin" | awk '{print $1}')
host_cxx_binary_hash=$(sha256sum "$host_cxx_bin" | awk '{print $1}')
git_commit=$(git -C "$repo_root" rev-parse HEAD)
git_status_hash=$(git -C "$repo_root" status --porcelain=v1 | sha256sum | awk '{print $1}')
# END N256 admission immutable inputs

# BEGIN N256 admission receipt
manifest="$build_dir/build-manifest.txt"
{
    echo "receipt_format=atlas-w2a8-fused-gu-n256-v1"
    echo "min_n128_to_n256_speedup=$min_speedup"
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
    echo "compile_command_template=<nvcc> -ccbin <host_cxx> -std=c++17 -O3 --fmad=false -arch=sm_121a '-DW2A8_FUSED_GU_N256_PROBE_BUILD_ID=\"<build_id>\"' -DW2A8_FUSED_GU_N256_PROBE_MIN_SPEEDUP=<canonical-binary64-threshold> '-DW2A8_FUSED_GU_N256_PROBE_MIN_SPEEDUP_TEXT=\"<canonical-binary64-threshold>\"' <harness> -o <binary>"
} >"$manifest"
build_id=$(sha256sum "$manifest" | awk '{print $1}')

binary="$build_dir/exl3-w2a8-fused-gu-n256-probe"
compile_command="$nvcc_bin -ccbin $host_cxx_bin -std=c++17 -O3 --fmad=false -arch=sm_121a '-DW2A8_FUSED_GU_N256_PROBE_BUILD_ID=\"$build_id\"' -DW2A8_FUSED_GU_N256_PROBE_MIN_SPEEDUP=${min_speedup} '-DW2A8_FUSED_GU_N256_PROBE_MIN_SPEEDUP_TEXT=\"$min_speedup\"' $harness_file -o $binary"
compile_command_hash=$(printf '%s' "$compile_command" | sha256sum | awk '{print $1}')
"$nvcc_bin" -ccbin "$host_cxx_bin" -std=c++17 -O3 --fmad=false \
    -arch=sm_121a \
    "-DW2A8_FUSED_GU_N256_PROBE_BUILD_ID=\"$build_id\"" \
    "-DW2A8_FUSED_GU_N256_PROBE_MIN_SPEEDUP=${min_speedup}" \
    "-DW2A8_FUSED_GU_N256_PROBE_MIN_SPEEDUP_TEXT=\"$min_speedup\"" \
    "$harness_file" -o "$binary"
binary_hash=$(sha256sum "$binary" | awk '{print $1}')

resources="$build_dir/fused-gu-n256.resources"
resource_command="$cuobjdump_bin --dump-resource-usage $binary"
resource_command_hash=$(printf '%s' "$resource_command" | sha256sum | awk '{print $1}')
"$cuobjdump_bin" --dump-resource-usage "$binary" >"$resources"
for symbol in \
    exl3_w2a8_fused_gu_down_emit_n128 \
    exl3_w2a8_fused_gu_down_emit_n256; do
    grep -q "Function ${symbol}:" "$resources"
done

extract_dir="$build_dir/fused-gu-n256.cubins"
mkdir "$extract_dir"
extract_command="$cuobjdump_bin --extract-elf all $binary"
extract_command_hash=$(printf '%s' "$extract_command" | sha256sum | awk '{print $1}')
(
    cd "$extract_dir"
    "$cuobjdump_bin" --extract-elf all "$binary" >/dev/null
)
cubin_count=$(find "$extract_dir" -maxdepth 1 -type f -name '*.cubin' | wc -l)
[[ $cubin_count -ge 1 ]]
receipt_lines=("binary_sha256 n256_admission_probe $binary_hash")
while IFS= read -r cubin; do
    cubin_name=$(basename "$cubin")
    cubin_hash=$(sha256sum "$cubin" | awk '{print $1}')
    receipt_lines+=("cubin_sha256 fused-gu-n256.cubins/$cubin_name $cubin_hash")
done < <(find "$extract_dir" -maxdepth 1 -type f -name '*.cubin' | sort)

inputs_unchanged=1
for dependency in "${dependencies[@]}"; do
    if [[ $(sha256sum "$dependency" | awk '{print $1}') != "${dependency_hashes[$dependency]}" ]]; then
        echo "N256 admission input changed during compilation: $dependency" >&2
        inputs_unchanged=0
    fi
done
for tool_record in \
    "$nvcc_bin|$nvcc_real_path|$nvcc_binary_hash" \
    "$cuobjdump_bin|$cuobjdump_real_path|$cuobjdump_binary_hash" \
    "$host_cxx_bin|$host_cxx_real_path|$host_cxx_binary_hash"; do
    IFS='|' read -r tool expected_path expected_hash <<<"$tool_record"
    if [[ $(realpath "$tool") != "$expected_path" ]] ||
        [[ $(sha256sum "$tool" | awk '{print $1}') != "$expected_hash" ]]; then
        echo "N256 admission build tool changed: $tool" >&2
        inputs_unchanged=0
    fi
done
current_commit=$(git -C "$repo_root" rev-parse HEAD)
current_status_hash=$(git -C "$repo_root" status --porcelain=v1 | sha256sum | awk '{print $1}')
if [[ $current_commit != "$git_commit" || $current_status_hash != "$git_status_hash" ]]; then
    echo "repository identity changed during N256 admission compilation" >&2
    inputs_unchanged=0
fi
[[ $inputs_unchanged == 1 ]]

receipt="$build_dir/build-receipt.txt"
{
    cat "$manifest"
    echo "build_id=$build_id"
    echo "compile_command=$compile_command"
    echo "compile_command_sha256=$compile_command_hash"
    echo "resource_command=$resource_command"
    echo "resource_command_sha256=$resource_command_hash"
    echo "extract_command=$extract_command"
    echo "extract_command_sha256=$extract_command_hash"
    printf '%s\n' "${receipt_lines[@]}"
} >"$receipt"
receipt_hash=$(sha256sum "$receipt" | awk '{print $1}')
# END N256 admission receipt

# BEGIN N256 admission zero-argument runner
runner="$build_dir/run-fused-gu-n256-probe.sh"
{
    echo '#!/usr/bin/env bash'
    echo '# SPDX-License-Identifier: AGPL-3.0-only'
    echo 'set -euo pipefail'
    printf 'expected_receipt_sha256=%q\n' "$receipt_hash"
    printf 'expected_binary_sha256=%q\n' "$binary_hash"
    printf 'expected_build_id=%q\n' "$build_id"
    printf 'expected_min_speedup=%q\n' "$min_speedup"
    cat <<'RUNNER'
export LC_ALL=C
if [[ $# -ne 0 ]]; then
    echo "usage: $0" >&2
    exit 2
fi
runner_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
receipt="$runner_dir/build-receipt.txt"
binary="$runner_dir/exl3-w2a8-fused-gu-n256-probe"
actual_receipt_sha256=$(sha256sum "$receipt" | awk '{print $1}')
actual_binary_sha256=$(sha256sum "$binary" | awk '{print $1}')
if [[ $actual_receipt_sha256 != "$expected_receipt_sha256" ||
    $actual_binary_sha256 != "$expected_binary_sha256" ]] ||
    [[ $(grep -Fxc "build_id=$expected_build_id" "$receipt") -ne 1 ]] ||
    [[ $(grep -Fxc "min_n128_to_n256_speedup=$expected_min_speedup" "$receipt") -ne 1 ]] ||
    [[ $(grep -Fxc "binary_sha256 n256_admission_probe $expected_binary_sha256" "$receipt") -ne 1 ]]; then
    echo "N256 admission artifact identity mismatch" >&2
    exit 2
fi
while read -r _ cubin_relative expected_cubin_sha256; do
    [[ -n ${cubin_relative:-} && -n ${expected_cubin_sha256:-} ]] || continue
    actual_cubin_sha256=$(sha256sum "$runner_dir/$cubin_relative" | awk '{print $1}')
    if [[ $actual_cubin_sha256 != "$expected_cubin_sha256" ]]; then
        echo "N256 admission cubin identity mismatch: $cubin_relative" >&2
        exit 2
    fi
done < <(awk '$1 == "cubin_sha256" { print }' "$receipt")

probe_tmp=$(mktemp -d "${TMPDIR:-/tmp}/atlas-w2a8-fused-gu-n256-run.XXXXXX")
trap 'rm -rf -- "$probe_tmp"' EXIT
set +e
"$binary" >"$probe_tmp/stdout" 2>"$probe_tmp/stderr"
probe_rc=$?
set -e
stdout_bytes=$(wc -c <"$probe_tmp/stdout")
stderr_bytes=$(wc -c <"$probe_tmp/stderr")
stdout_lines=$(wc -l <"$probe_tmp/stdout")
if [[ $probe_rc -ne 0 || $stdout_bytes -gt 8192 || $stderr_bytes -ne 0 || $stdout_lines -ne 10 ]]; then
    echo "N256 admission rejected: rc=$probe_rc stdout_bytes=$stdout_bytes stderr_bytes=$stderr_bytes stdout_lines=$stdout_lines" >&2
    exit 1
fi

mapfile -t lines <"$probe_tmp/stdout"
number='([0-9]+([.][0-9]+)?|[.][0-9]+)([eE][+-]?[0-9]+)?'
if [[ ${lines[0]} != "build_id=$expected_build_id" ]] ||
    [[ ! ${lines[1]} =~ ^device_uuid=[0-9a-f]{32}\ driver=[0-9]+\ runtime=[0-9]+$ ]] ||
    [[ ! ${lines[2]} =~ ^input_hash=[0-9a-f]{16}\ routing_hash=[0-9a-f]{16}\ tables_hash=[0-9a-f]{16}$ ]] ||
    [[ ! ${lines[3]} =~ ^n128_output_hash=[0-9a-f]{16}\ n256_output_hash=[0-9a-f]{16}$ ]] ||
    [[ ${lines[4]} != 'rows=14460 fp8_bytes=29614080 scale_bytes=925440 experts=256 routes=4 boundary_counts=10 malformed_routes=2 parity_cases=8' ]] ||
    [[ ${lines[5]} != 'routing=synthetic_production_total representative=0' ]] ||
    [[ ${lines[6]} != "threshold min_speedup=$expected_min_speedup" ]] ||
    [[ ! ${lines[7]} =~ ^n128_ms=$number\ n256_ms=$number\ speedup=$number\ abba_samples=4$ ]] ||
    [[ ${lines[8]} != 'fp8_mismatches=0 scale_mismatches=0 guards=clean poison_a=clean poison_b=clean input_immutable=clean' ]] ||
    [[ ${lines[9]} != 'result=PASS' ]]; then
    echo "N256 admission output contract mismatch" >&2
    exit 1
fi
n128_output_hash=${lines[3]#n128_output_hash=}
n128_output_hash=${n128_output_hash%% *}
n256_output_hash=${lines[3]##*n256_output_hash=}
if [[ $n128_output_hash != "$n256_output_hash" ]]; then
    echo "N256 admission output hash mismatch" >&2
    exit 1
fi
printf 'receipt_sha256=%s\n' "$actual_receipt_sha256"
printf 'binary_sha256=%s\n' "$actual_binary_sha256"
cat "$probe_tmp/stdout"
RUNNER
} >"$runner"
chmod 0755 "$runner"
runner_hash=$(sha256sum "$runner" | awk '{print $1}')
# END N256 admission zero-argument runner

echo "fused W2A8 N128-to-N256 admission probe compiles for SM121a"
echo "threshold=$min_speedup is receipt-bound; runtime evidence requires the generated runner"
echo "build_id=$build_id"
echo "receipt_sha256=$receipt_hash"
echo "binary_sha256=$binary_hash"
echo "runner_sha256=$runner_hash"
if [[ $persist_probe == 1 ]]; then
    echo "probe runner: $runner"
    echo "build receipt: $receipt"
fi
