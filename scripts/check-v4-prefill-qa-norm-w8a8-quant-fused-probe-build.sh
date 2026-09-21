#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only

set -euo pipefail
export LC_ALL=C

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <min_speedup>" >&2
    exit 2
fi
min_speedup=$1
number_pattern='^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$'
if [[ ! $min_speedup =~ $number_pattern ]] ||
    ! awk -v value="$min_speedup" 'BEGIN { exit !(value > 1.0 && value <= 100.0) }'; then
    echo "invalid explicit numeric threshold" >&2
    exit 2
fi
min_speedup=$(awk -v value="$min_speedup" 'BEGIN { printf "%.17g", value + 0.0 }')

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cuda_root_path=${CUDA_ROOT_PATH:-/usr/local/cuda}
nvcc_bin=${NVCC_BIN:-"$cuda_root_path/bin/nvcc"}
cuobjdump_bin=${CUOBJDUMP_BIN:-"$cuda_root_path/bin/cuobjdump"}
host_cxx_bin=${HOST_CXX_BIN:-/usr/bin/g++}

for tool in "$nvcc_bin" "$cuobjdump_bin" "$host_cxx_bin"; do
    [[ -x $tool ]] || { echo "missing build tool: $tool" >&2; exit 2; }
done
for injected_flags in NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS; do
    [[ -z ${!injected_flags:-} ]] || {
        echo "refusing unreceipted nvcc flags from $injected_flags" >&2
        exit 2
    }
done

if [[ -n ${V4_QA_PROBE_OUTPUT_DIR:-} ]]; then
    build_dir=$V4_QA_PROBE_OUTPUT_DIR
    [[ ! -e $build_dir && ! -L $build_dir ]] || {
        echo "refusing existing V4_QA_PROBE_OUTPUT_DIR: $build_dir" >&2
        exit 2
    }
    mkdir -m 0700 -- "$build_dir"
    persist_probe=1
else
    build_dir=$(mktemp -d /tmp/atlas-v4-qa-norm-w8a8-probe.XXXXXX)
    trap 'rm -rf -- "$build_dir"' EXIT
    persist_probe=0
fi

# BEGIN q_a probe immutable inputs
probe_file="$repo_root/kernels/gb10/experiments/v4_prefill_qa_norm_w8a8_quant_fused_probe.cu"
candidate_file="$repo_root/kernels/gb10/experiments/v4_prefill_qa_norm_w8a8_quant_fused.cu"
rms_file="$repo_root/kernels/gb10/common/rms_norm_vanilla.cu"
quant_file="$repo_root/kernels/gb10/common/w8a8_gemm_pipelined.cu"
build_script="$repo_root/scripts/check-v4-prefill-qa-norm-w8a8-quant-fused-probe-build.sh"
dependencies=("$probe_file" "$candidate_file" "$rms_file" "$quant_file" "$build_script")
declare -A dependency_hashes
for dependency in "${dependencies[@]}"; do
    [[ -f $dependency ]] || { echo "missing q_a probe build input: $dependency" >&2; exit 2; }
    dependency_hashes[$dependency]=$(sha256sum "$dependency" | awk '{print $1}')
done
nvcc_real_path=$(realpath "$nvcc_bin")
cuobjdump_real_path=$(realpath "$cuobjdump_bin")
host_cxx_real_path=$(realpath "$host_cxx_bin")
nvcc_binary_hash=$(sha256sum "$nvcc_bin" | awk '{print $1}')
cuobjdump_binary_hash=$(sha256sum "$cuobjdump_bin" | awk '{print $1}')
host_cxx_binary_hash=$(sha256sum "$host_cxx_bin" | awk '{print $1}')
git_commit=$(git -C "$repo_root" rev-parse HEAD)
git_status_hash=$(git -C "$repo_root" status --porcelain=v1 | sha256sum | awk '{print $1}')
# END q_a probe immutable inputs

# BEGIN q_a probe receipt contract
manifest="$build_dir/build-manifest.txt"
{
    echo receipt_format=atlas-v4-qa-norm-w8a8-probe-v1
    echo "git_commit=$git_commit"
    echo "git_status_sha256=$git_status_hash"
    echo "min_speedup=$min_speedup"
    for dependency in "${dependencies[@]}"; do
        echo "source_sha256 ${dependency#"$repo_root/"} ${dependency_hashes[$dependency]}"
    done
    echo "nvcc_path=$nvcc_real_path"
    echo "nvcc_binary_sha256=$nvcc_binary_hash"
    echo "cuobjdump_path=$cuobjdump_real_path"
    echo "cuobjdump_binary_sha256=$cuobjdump_binary_hash"
    echo "host_cxx_path=$host_cxx_real_path"
    echo "host_cxx_binary_sha256=$host_cxx_binary_hash"
    echo "compile_command_template=<nvcc> -ccbin <host_cxx> -std=c++17 -O3 --fmad=false -arch=sm_121a -DV4_QA_PROBE_BUILD_ID=<build_id> <probe> -o <binary>"
} >"$manifest"
build_id=$(sha256sum "$manifest" | awk '{print $1}')
binary="$build_dir/v4-prefill-qa-norm-w8a8-quant-fused-probe"
compile_log="$build_dir/compile.log"
compile_command="$nvcc_bin -ccbin $host_cxx_bin -std=c++17 -O3 --fmad=false -arch=sm_121a '-DV4_QA_PROBE_BUILD_ID=\"$build_id\"' $probe_file -o $binary"
compile_command_hash=$(printf %s "$compile_command" | sha256sum | awk '{print $1}')
"$nvcc_bin" -ccbin "$host_cxx_bin" -std=c++17 -O3 --fmad=false -arch=sm_121a \
    "-DV4_QA_PROBE_BUILD_ID=\"$build_id\"" "$probe_file" -o "$binary" \
    -Xptxas=-v 2>"$compile_log"
binary_hash=$(sha256sum "$binary" | awk '{print $1}')
compile_log_hash=$(sha256sum "$compile_log" | awk '{print $1}')

expect_threshold_rejection() {
    local label=$1
    shift
    local output="$build_dir/reject-$label.out"
    set +e
    "$binary" "$@" >"$output" 2>&1
    local rc=$?
    set -e
    [[ $rc -eq 2 ]] || { echo "q_a probe accepted invalid threshold: $label" >&2; exit 1; }
    if [[ $label == usage* ]]; then
        grep -Eqx 'usage: .* <min_end_to_end_speedup>' "$output"
    else
        grep -qx 'invalid explicit numeric threshold' "$output"
    fi
}
expect_threshold_rejection usage
expect_threshold_rejection usage_extra 1.01 extra
expect_threshold_rejection speed_junk junk
expect_threshold_rejection speed_nan nan
expect_threshold_rejection speed_inf inf
expect_threshold_rejection speed_whitespace ' 1.01'
expect_threshold_rejection speed_hex 0x1.1p1
expect_threshold_rejection speed_negative -1
expect_threshold_rejection speed_zero 0
expect_threshold_rejection speed_no_win 1
expect_threshold_rejection speed_lax 101

resources="$build_dir/resources.txt"
resource_command="$cuobjdump_bin --dump-resource-usage $binary"
resource_command_hash=$(printf %s "$resource_command" | sha256sum | awk '{print $1}')
"$cuobjdump_bin" --dump-resource-usage "$binary" >"$resources"
for symbol in rms_norm_vanilla quantize_a_fp8_rows v4_prefill_qa_norm_w8a8_quant_fused; do
    grep -q "Function $symbol:" "$resources"
done
resources_hash=$(sha256sum "$resources" | awk '{print $1}')
cubin_dir="$build_dir/cubins"
mkdir "$cubin_dir"
extract_command="$cuobjdump_bin --extract-elf all $binary"
extract_command_hash=$(printf %s "$extract_command" | sha256sum | awk '{print $1}')
(cd "$cubin_dir" && "$cuobjdump_bin" --extract-elf all "$binary" >/dev/null)
mapfile -t cubins < <(find "$cubin_dir" -maxdepth 1 -type f -name '*.cubin' | sort)
[[ ${#cubins[@]} -gt 0 ]] || { echo "no cubin extracted" >&2; exit 1; }

for dependency in "${dependencies[@]}"; do
    [[ $(sha256sum "$dependency" | awk '{print $1}') == "${dependency_hashes[$dependency]}" ]] || {
        echo "build input changed during q_a probe compilation: $dependency" >&2
        exit 1
    }
done
for record in \
    "$nvcc_bin|$nvcc_real_path|$nvcc_binary_hash" \
    "$cuobjdump_bin|$cuobjdump_real_path|$cuobjdump_binary_hash" \
    "$host_cxx_bin|$host_cxx_real_path|$host_cxx_binary_hash"; do
    IFS='|' read -r tool tool_path tool_hash <<<"$record"
    [[ $(realpath "$tool") == "$tool_path" && $(sha256sum "$tool" | awk '{print $1}') == "$tool_hash" ]] || {
        echo "build tool changed during q_a probe compilation: $tool" >&2
        exit 1
    }
done
[[ $(git -C "$repo_root" rev-parse HEAD) == "$git_commit" && \
    $(git -C "$repo_root" status --porcelain=v1 | sha256sum | awk '{print $1}') == "$git_status_hash" ]] || {
    echo "repository identity changed during q_a probe compilation" >&2
    exit 1
}

receipt="$build_dir/build-receipt.txt"
{
    cat "$manifest"
    echo "build_id=$build_id"
    echo "compile_command=$compile_command"
    echo "compile_command_sha256=$compile_command_hash"
    echo "compile_log_sha256=$compile_log_hash"
    echo "resource_command=$resource_command"
    echo "resource_command_sha256=$resource_command_hash"
    echo "resources_sha256=$resources_hash"
    echo "extract_command=$extract_command"
    echo "extract_command_sha256=$extract_command_hash"
    echo "binary_sha256 probe $binary_hash"
    for cubin in "${cubins[@]}"; do
        echo "cubin_sha256 cubins/$(basename "$cubin") $(sha256sum "$cubin" | awk '{print $1}')"
    done
} >"$receipt"
receipt_hash=$(sha256sum "$receipt" | awk '{print $1}')
# END q_a probe receipt contract

# BEGIN q_a probe runner verification
runner="$build_dir/run-v4-prefill-qa-norm-w8a8-quant-fused-probe.sh"
{
    echo '#!/usr/bin/env bash'
    echo '# SPDX-License-Identifier: AGPL-3.0-only'
    echo 'set -euo pipefail'
    echo 'export LC_ALL=C'
    printf 'expected_receipt_sha256=%q\nexpected_binary_sha256=%q\nexpected_build_id=%q\nexpected_min_speedup=%q\n' \
        "$receipt_hash" "$binary_hash" "$build_id" "$min_speedup"
    cat <<'RUNNER'
[[ $# -eq 0 ]] || { echo "usage: $0" >&2; exit 2; }
dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
receipt="$dir/build-receipt.txt"
binary="$dir/v4-prefill-qa-norm-w8a8-quant-fused-probe"
actual_receipt_sha256=$(sha256sum "$receipt" | awk '{print $1}')
actual_binary_sha256=$(sha256sum "$binary" | awk '{print $1}')
[[ $actual_receipt_sha256 == "$expected_receipt_sha256" && $actual_binary_sha256 == "$expected_binary_sha256" ]] || {
    echo "q_a probe artifact identity mismatch" >&2
    exit 2
}
[[ $(grep -Fxc "build_id=$expected_build_id" "$receipt") -eq 1 ]] || exit 2
[[ $(grep -Fxc "min_speedup=$expected_min_speedup" "$receipt") -eq 1 ]] || exit 2
[[ $(grep -Fxc "binary_sha256 probe $expected_binary_sha256" "$receipt") -eq 1 ]] || exit 2
while read -r _ relative expected; do
    actual_cubin_sha256=$(sha256sum "$dir/$relative" | awk '{print $1}')
    [[ $actual_cubin_sha256 == "$expected" ]] || exit 2
done < <(grep '^cubin_sha256 ' "$receipt")
output=$(mktemp)
errors=$(mktemp)
trap 'rm -f "$output" "$errors"' EXIT
set +e
"$binary" "$expected_min_speedup" >"$output" 2>"$errors"
rc=$?
set -e
[[ $rc -eq 0 && ! -s $errors && $(wc -c <"$output") -le 4096 && $(wc -l <"$output") -eq 8 ]] || {
    cat "$errors" >&2
    exit 1
}
mapfile -t line <"$output"
[[ ${line[0]} == "build_id=$expected_build_id" ]]
[[ ${line[1]} =~ ^device_uuid=[0-9a-f]{32}\ driver=[0-9]+\ runtime=[0-9]+$ ]]
[[ ${line[2]} =~ ^input_hash=[0-9a-f]{16}\ weight_hash=[0-9a-f]{16}$ ]]
[[ ${line[3]} == 'shapes=1,7,128,2410 hidden=1024 parity_cases=4 poison_cases=2 geometry_cases=15 fp8_bytes=2467840 scale_bytes=9640 normed_bytes=4935680' ]]
[[ ${line[4]} == "threshold min_speedup=$expected_min_speedup" ]]
[[ ${line[5]} =~ ^baseline_ms=.+\ candidate_ms=.+\ speedup=.+\ abba_samples=[1-9][0-9]*$ ]]
[[ ${line[6]} == 'fp8_mismatches=0 scale_mismatches=0 guards=clean poison_a=clean poison_b=clean' ]]
[[ ${line[7]} == 'result=PASS' ]]
cat "$output"
RUNNER
} >"$runner"
chmod 0555 "$runner"
# END q_a probe runner verification

if [[ $persist_probe == 1 ]]; then
    echo "retained_receipt=$receipt"
    echo "retained_runner=$runner"
fi
echo "V4 q_a norm/W8A8 fused promotion probe build: PASS"
echo "min_speedup=$min_speedup"
echo "build_id=$build_id"
echo "receipt_sha256=$receipt_hash"
