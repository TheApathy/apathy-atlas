#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cuda_root_path=${CUDA_ROOT_PATH:-/usr/local/cuda}
nvcc_bin=${NVCC_BIN:-"$cuda_root_path/bin/nvcc"}
cuobjdump_bin=${CUOBJDUMP_BIN:-"$cuda_root_path/bin/cuobjdump"}
host_cxx_bin=${HOST_CXX_BIN:-/usr/bin/g++}

if [[ -n ${W2A8_PROBE_OUTPUT_DIR:-} ]]; then
    w2a8_build_dir=$W2A8_PROBE_OUTPUT_DIR
    if [[ -d $w2a8_build_dir ]] &&
        [[ -n $(find "$w2a8_build_dir" -mindepth 1 -maxdepth 1 -print -quit) ]]; then
        echo "refusing nonempty W2A8_PROBE_OUTPUT_DIR: $w2a8_build_dir" >&2
        exit 2
    fi
    mkdir -p "$w2a8_build_dir"
    persist_probes=1
else
    w2a8_build_dir=$(mktemp -d /tmp/atlas-exl3-w2a8-probe.XXXXXX)
    trap 'rm -rf -- "$w2a8_build_dir"' EXIT
    persist_probes=0
fi

for tool in "$nvcc_bin" "$cuobjdump_bin" "$host_cxx_bin"; do
    if [[ ! -x "$tool" ]]; then
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

source_file="$repo_root/kernels/gb10/experiments/exl3_w2a8_grouped_prefill_probe.cu"
component_file="$repo_root/kernels/gb10/experiments/exl3_w2a8_grouped_prefill.cu"
component_n128_file="$repo_root/kernels/gb10/experiments/exl3_w2a8_grouped_prefill_n128.cu"
component_n256_file="$repo_root/kernels/gb10/experiments/exl3_w2a8_grouped_prefill_n256.cu"
quant_file="$repo_root/kernels/gb10/common/per_token_group_quant_fp8.cu"
baseline_file="$repo_root/kernels/gb10/common/exl3_grouped_prefill.cu"
gu_wrapper="$repo_root/kernels/gb10/common/exl3_grouped_prefill_k64_k2_gu.cu"
down_wrapper="$repo_root/kernels/gb10/common/exl3_grouped_prefill_k64_k2_down.cu"
build_script="$repo_root/scripts/check-exl3-prefill-w2a8-probe-build.sh"
dependencies=(
    "$source_file" "$component_file" "$component_n128_file" "$component_n256_file"
    "$quant_file" "$baseline_file"
    "$gu_wrapper" "$down_wrapper" "$build_script"
)
declare -A dependency_hashes
for dependency in "${dependencies[@]}"; do
    dependency_hashes[$dependency]=$(sha256sum "$dependency" | awk '{print $1}')
done

nvcc_version=$("$nvcc_bin" --version | tail -n 1)
cuobjdump_version=$("$cuobjdump_bin" --version | tail -n 1)
host_cxx_version=$("$host_cxx_bin" --version | head -n 1)
nvcc_real_path=$(realpath "$nvcc_bin")
cuobjdump_real_path=$(realpath "$cuobjdump_bin")
host_cxx_real_path=$(realpath "$host_cxx_bin")
nvcc_binary_hash=$(sha256sum "$nvcc_bin" | awk '{print $1}')
cuobjdump_binary_hash=$(sha256sum "$cuobjdump_bin" | awk '{print $1}')
host_cxx_binary_hash=$(sha256sum "$host_cxx_bin" | awk '{print $1}')
git_commit=$(git -C "$repo_root" rev-parse HEAD)
git_status_hash=$(git -C "$repo_root" status --porcelain=v1 | sha256sum | awk '{print $1}')
build_manifest="$w2a8_build_dir/build-manifest.txt"
{
    echo "receipt_format=atlas-w2a8-v3"
    echo "git_commit=$git_commit"
    echo "git_status_sha256=$git_status_hash"
    echo "nvcc_path=$nvcc_real_path"
    echo "nvcc_version=$nvcc_version"
    echo "nvcc_binary_sha256=$nvcc_binary_hash"
    echo "cuobjdump_path=$cuobjdump_real_path"
    echo "cuobjdump_version=$cuobjdump_version"
    echo "cuobjdump_binary_sha256=$cuobjdump_binary_hash"
    echo "host_cxx_path=$host_cxx_real_path"
    echo "host_cxx_version=$host_cxx_version"
    echo "host_cxx_binary_sha256=$host_cxx_binary_hash"
    echo "compile_command_template=<nvcc> -ccbin <host_cxx> -std=c++17 -O3 --fmad=false -arch=sm_121a -DW2A8_PROBE_GU=<0|1> -DW2A8_PROBE_N_TILE=<64|128|256> '-DW2A8_BUILD_ID=\"<build_id>\"' <source> -o <output>"
    for dependency in "${dependencies[@]}"; do
        relative=${dependency#"$repo_root/"}
        echo "source_sha256 $relative ${dependency_hashes[$dependency]}"
    done
} >"$build_manifest"
build_id=$(sha256sum "$build_manifest" | awk '{print $1}')
receipt_lines=()
compile_commands=()
declare -A binary_hashes

expect_threshold_rejection() {
    local binary=$1
    local kind=$2
    local label=$3
    shift 3
    set +e
    "$binary" "$@" >"$w2a8_build_dir/${kind}.${label}.out" 2>&1
    local probe_rc=$?
    set -e
    if [[ $probe_rc -ne 2 ]]; then
        echo "${kind} probe accepted invalid thresholds: ${label}" >&2
        exit 1
    fi
    if [[ $label == usage* ]]; then
        grep -q 'usage: .* <min_cosine> <max_abs_error> <min_end_to_end_speedup>' \
            "$w2a8_build_dir/${kind}.${label}.out"
    else
        grep -qx 'invalid explicit numeric threshold' \
            "$w2a8_build_dir/${kind}.${label}.out"
    fi
}

expect_dump_rejection() {
    local binary=$1
    local artifact=$2
    local dump_file="$w2a8_build_dir/${artifact}.existing.dump"
    local output_file="$w2a8_build_dir/${artifact}.dump-rejection.out"
    printf 'must-not-be-overwritten\n' >"$dump_file"
    local dump_before
    dump_before=$(sha256sum "$dump_file" | awk '{print $1}')
    set +e
    W2A8_PROBE_DUMP="$dump_file" \
        "$binary" 0.999 0.1 1.1 >"$output_file" 2>&1
    local probe_rc=$?
    set -e
    if [[ $probe_rc -ne 2 ]] ||
        ! grep -q 'exclusive W2A8 probe dump' "$output_file" ||
        [[ $(sha256sum "$dump_file" | awk '{print $1}') != "$dump_before" ]]; then
        echo "${artifact} probe did not preserve an existing dump" >&2
        exit 1
    fi
}

for kind in gu down; do
    if [[ $kind == gu ]]; then
        probe_gu=1
    else
        probe_gu=0
    fi
    for n_tile in 64 128 256; do
        artifact="${kind}_n${n_tile}"
        binary="$w2a8_build_dir/exl3-w2a8-${kind}-n${n_tile}-probe"
        "$nvcc_bin" -ccbin "$host_cxx_bin" -std=c++17 -O3 --fmad=false \
            -arch=sm_121a -DW2A8_PROBE_GU="$probe_gu" \
            -DW2A8_PROBE_N_TILE="$n_tile" \
            "-DW2A8_BUILD_ID=\"$build_id\"" "$source_file" -o "$binary"
        compile_commands+=(
            "compile_command_${kind}_n${n_tile}=$nvcc_bin -ccbin $host_cxx_bin -std=c++17 -O3 --fmad=false -arch=sm_121a -DW2A8_PROBE_GU=$probe_gu -DW2A8_PROBE_N_TILE=$n_tile '-DW2A8_BUILD_ID=\"$build_id\"' $source_file -o $binary"
        )
        expect_threshold_rejection "$binary" "$artifact" usage
        binary_hashes[$artifact]=$(sha256sum "$binary" | awk '{print $1}')
        receipt_lines+=("binary_sha256 ${artifact} ${binary_hashes[$artifact]}")
        expect_threshold_rejection "$binary" "$artifact" junk junk 0.1 1.1
        expect_threshold_rejection "$binary" "$artifact" cosine_nan nan 0.1 1.1
        expect_threshold_rejection "$binary" "$artifact" cosine_inf inf 0.1 1.1
        expect_threshold_rejection "$binary" "$artifact" cosine_weak 0.98 0.1 1.1
        expect_threshold_rejection "$binary" "$artifact" error_nan 0.999 nan 1.1
        expect_threshold_rejection "$binary" "$artifact" error_inf 0.999 inf 1.1
        expect_threshold_rejection "$binary" "$artifact" error_zero 0.999 0 1.1
        expect_threshold_rejection "$binary" "$artifact" error_lax 0.999 1.1 1.1
        expect_threshold_rejection "$binary" "$artifact" speed_nan 0.999 0.1 nan
        expect_threshold_rejection "$binary" "$artifact" speed_inf 0.999 0.1 inf
        expect_threshold_rejection "$binary" "$artifact" speed_no_win 0.999 0.1 1.0
        expect_threshold_rejection "$binary" "$artifact" speed_lax 0.999 0.1 101
        expect_threshold_rejection \
            "$binary" "$artifact" usage_extra 0.999 0.1 1.1 extra
        expect_dump_rejection "$binary" "$artifact"

        resources="$w2a8_build_dir/${artifact}.resources"
        "$cuobjdump_bin" --dump-resource-usage "$binary" >"$resources"
        if [[ $n_tile == 256 ]]; then
            kernel_symbol="exl3_w2a8_grouped_prefill_n256_${kind}"
        elif [[ $n_tile == 128 ]]; then
            kernel_symbol="exl3_w2a8_grouped_prefill_n128_${kind}"
        else
            kernel_symbol="exl3_w2a8_grouped_prefill_${kind}"
        fi
        grep -q "Function ${kernel_symbol}:" "$resources"
        grep -q 'Function per_token_group_quant_fp8:' "$resources"

        extract_dir="$w2a8_build_dir/${artifact}.cubins"
        mkdir "$extract_dir"
        (
            cd "$extract_dir"
            "$cuobjdump_bin" --extract-elf all "$binary" >/dev/null
        )
        cubin_count=$(find "$extract_dir" -maxdepth 1 -type f -name '*.cubin' | wc -l)
        [[ $cubin_count -ge 1 ]]
        while IFS= read -r cubin; do
            cubin_name=$(basename "$cubin")
            cubin_hash=$(sha256sum "$cubin" | awk '{print $1}')
            receipt_lines+=("cubin_sha256 ${artifact}/${cubin_name} ${cubin_hash}")
        done < <(find "$extract_dir" -maxdepth 1 -type f -name '*.cubin' | sort)
    done
done

build_inputs_unchanged=1
for dependency in "${dependencies[@]}"; do
    current_hash=$(sha256sum "$dependency" | awk '{print $1}')
    if [[ $current_hash != "${dependency_hashes[$dependency]}" ]]; then
        echo "build input changed during W2A8 compilation: $dependency" >&2
        build_inputs_unchanged=0
    fi
done
for tool_record in \
    "$nvcc_bin|$nvcc_real_path|$nvcc_binary_hash" \
    "$cuobjdump_bin|$cuobjdump_real_path|$cuobjdump_binary_hash" \
    "$host_cxx_bin|$host_cxx_real_path|$host_cxx_binary_hash"; do
    IFS='|' read -r tool expected_path expected_hash <<<"$tool_record"
    if [[ $(realpath "$tool") != "$expected_path" ]] ||
        [[ $(sha256sum "$tool" | awk '{print $1}') != "$expected_hash" ]]; then
        echo "build tool changed during W2A8 compilation: $tool" >&2
        build_inputs_unchanged=0
    fi
done
current_commit=$(git -C "$repo_root" rev-parse HEAD)
current_status_hash=$(git -C "$repo_root" status --porcelain=v1 | sha256sum | awk '{print $1}')
if [[ $current_commit != "$git_commit" || $current_status_hash != "$git_status_hash" ]]; then
    echo "repository identity changed during W2A8 compilation" >&2
    build_inputs_unchanged=0
fi
[[ $build_inputs_unchanged == 1 ]]

receipt_file="$w2a8_build_dir/build-receipt.txt"
{
    cat "$build_manifest"
    echo "build_id=$build_id"
    printf '%s\n' "${compile_commands[@]}"
    printf '%s\n' "${receipt_lines[@]}"
} | tee "$receipt_file"
receipt_hash=$(sha256sum "$receipt_file" | awk '{print $1}')

for kind in gu down; do
    for n_tile in 64 128 256; do
        artifact="${kind}_n${n_tile}"
        runner="$w2a8_build_dir/run-${kind}-n${n_tile}-probe.sh"
        {
            echo '#!/usr/bin/env bash'
            echo '# SPDX-License-Identifier: AGPL-3.0-only'
            echo 'set -euo pipefail'
            printf 'expected_receipt_sha256=%q\n' "$receipt_hash"
            printf 'expected_binary_sha256=%q\n' "${binary_hashes[$artifact]}"
            printf 'expected_build_id=%q\n' "$build_id"
            printf 'probe_kind=%q\n' "$kind"
            printf 'probe_n_tile=%q\n' "$n_tile"
            cat <<'RUNNER'
runner_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
receipt="$runner_dir/build-receipt.txt"
binary="$runner_dir/exl3-w2a8-${probe_kind}-n${probe_n_tile}-probe"
actual_receipt_sha256=$(sha256sum "$receipt" | awk '{print $1}')
actual_binary_sha256=$(sha256sum "$binary" | awk '{print $1}')
if [[ $actual_receipt_sha256 != "$expected_receipt_sha256" ||
    $actual_binary_sha256 != "$expected_binary_sha256" ]]; then
    echo "W2A8 probe artifact identity mismatch" >&2
    exit 2
fi
echo "receipt_sha256=$actual_receipt_sha256"
echo "binary_sha256=$actual_binary_sha256"
echo "expected_build_id=$expected_build_id"
exec "$binary" "$@"
RUNNER
        } >"$runner"
        chmod 0755 "$runner"
    done
    cp "$w2a8_build_dir/run-${kind}-n64-probe.sh" \
        "$w2a8_build_dir/run-${kind}-probe.sh"

    pair_runner="$w2a8_build_dir/run-${kind}-pair-probe.sh"
    {
        echo '#!/usr/bin/env bash'
        echo '# SPDX-License-Identifier: AGPL-3.0-only'
        echo 'set -euo pipefail'
        printf 'probe_kind=%q\n' "$kind"
        cat <<'PAIR_RUNNER'
runner_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
n64_runner="$runner_dir/run-${probe_kind}-n64-probe.sh"
n128_runner="$runner_dir/run-${probe_kind}-n128-probe.sh"
pair_tmp=$(mktemp -d "${TMPDIR:-/tmp}/atlas-w2a8-pair.XXXXXX")
trap 'rm -rf -- "$pair_tmp"' EXIT
n64_output=$(W2A8_PROBE_DUMP="$pair_tmp/n64.bin" "$n64_runner" "$@")
n128_output=$(W2A8_PROBE_DUMP="$pair_tmp/n128.bin" "$n128_runner" "$@")
printf '%s\n' "=== ${probe_kind} N64 ===" "$n64_output"
printf '%s\n' "=== ${probe_kind} N128 ===" "$n128_output"

extract_w2a8_hashes() {
    sed -n 's/.*w2a8_hash=\([0-9a-f]\{16\}\).*/\1/p'
}
extract_input_identity() {
    sed -n \
        's/.*input_hash=\([0-9a-f]\{16\}\) trellis_hash=\([0-9a-f]\{16\}\) result=PASS.*/\1 \2/p'
}
n64_hashes=$(printf '%s\n' "$n64_output" | extract_w2a8_hashes)
n128_hashes=$(printf '%s\n' "$n128_output" | extract_w2a8_hashes)
hash_count=$(printf '%s\n' "$n64_hashes" | awk 'NF { count++ } END { print count + 0 }')
n128_hash_count=$(printf '%s\n' "$n128_hashes" | awk 'NF { count++ } END { print count + 0 }')
n64_input=$(printf '%s\n' "$n64_output" | extract_input_identity)
n128_input=$(printf '%s\n' "$n128_output" | extract_input_identity)
if [[ $hash_count -ne 9 || $n128_hash_count -ne 9 ||
    $n64_hashes != "$n128_hashes" || -z $n64_input ||
    $n64_input != "$n128_input" ||
    ! -s $pair_tmp/n64.bin || ! -s $pair_tmp/n128.bin ]] ||
    ! cmp -s "$pair_tmp/n64.bin" "$pair_tmp/n128.bin"; then
    echo "cross-width W2A8 hashes do not match exactly" >&2
    exit 1
fi
echo "cross-width W2A8 bytes: exact; 9/9 hashes, input, and trellis identities match"
PAIR_RUNNER
    } >"$pair_runner"
    chmod 0755 "$pair_runner"

    three_width_runner="$w2a8_build_dir/run-${kind}-three-width-probe.sh"
    {
        echo '#!/usr/bin/env bash'
        echo '# SPDX-License-Identifier: AGPL-3.0-only'
        echo 'set -euo pipefail'
        printf 'probe_kind=%q\n' "$kind"
        cat <<'THREE_WIDTH_RUNNER'
runner_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
n64_runner="$runner_dir/run-${probe_kind}-n64-probe.sh"
n128_runner="$runner_dir/run-${probe_kind}-n128-probe.sh"
n256_runner="$runner_dir/run-${probe_kind}-n256-probe.sh"
width_tmp=$(mktemp -d "${TMPDIR:-/tmp}/atlas-w2a8-three-width.XXXXXX")
trap 'rm -rf -- "$width_tmp"' EXIT
n64_output=$(W2A8_PROBE_DUMP="$width_tmp/n64.bin" "$n64_runner" "$@")
n128_output=$(W2A8_PROBE_DUMP="$width_tmp/n128.bin" "$n128_runner" "$@")
n256_output=$(W2A8_PROBE_DUMP="$width_tmp/n256.bin" "$n256_runner" "$@")
printf '%s\n' "=== ${probe_kind} N64 ===" "$n64_output"
printf '%s\n' "=== ${probe_kind} N128 ===" "$n128_output"
printf '%s\n' "=== ${probe_kind} N256 ===" "$n256_output"

extract_w2a8_hashes() {
    sed -n 's/.*w2a8_hash=\([0-9a-f]\{16\}\).*/\1/p'
}
extract_input_identity() {
    sed -n \
        's/.*input_hash=\([0-9a-f]\{16\}\) trellis_hash=\([0-9a-f]\{16\}\) result=PASS.*/\1 \2/p'
}
n64_hashes=$(printf '%s\n' "$n64_output" | extract_w2a8_hashes)
n128_hashes=$(printf '%s\n' "$n128_output" | extract_w2a8_hashes)
n256_hashes=$(printf '%s\n' "$n256_output" | extract_w2a8_hashes)
n64_hash_count=$(printf '%s\n' "$n64_hashes" | awk 'NF { count++ } END { print count + 0 }')
n128_hash_count=$(printf '%s\n' "$n128_hashes" | awk 'NF { count++ } END { print count + 0 }')
n256_hash_count=$(printf '%s\n' "$n256_hashes" | awk 'NF { count++ } END { print count + 0 }')
n64_input=$(printf '%s\n' "$n64_output" | extract_input_identity)
n128_input=$(printf '%s\n' "$n128_output" | extract_input_identity)
n256_input=$(printf '%s\n' "$n256_output" | extract_input_identity)
if [[ $n64_hash_count -ne 9 || $n128_hash_count -ne 9 ||
    $n256_hash_count -ne 9 || $n64_hashes != "$n128_hashes" ||
    $n64_hashes != "$n256_hashes" || -z $n64_input ||
    $n64_input != "$n128_input" || $n64_input != "$n256_input" ||
    ! -s $width_tmp/n64.bin || ! -s $width_tmp/n128.bin ||
    ! -s $width_tmp/n256.bin ]] ||
    ! cmp -s "$width_tmp/n64.bin" "$width_tmp/n128.bin" ||
    ! cmp -s "$width_tmp/n64.bin" "$width_tmp/n256.bin"; then
    echo "three-width W2A8 hashes do not match exactly" >&2
    exit 1
fi
echo "three-width W2A8 bytes: exact; N64/N128/N256 9/9 hashes, input, and trellis identities match"
THREE_WIDTH_RUNNER
    } >"$three_width_runner"
    chmod 0755 "$three_width_runner"
done

echo "W2A8 standalone GU/down N64/N128/N256 promotion probes compile for SM121a"
echo "strict parser rejection is distinct from CUDA failures"
echo "build_id=$build_id"
echo "receipt_sha256=$receipt_hash"
if [[ $persist_probes == 1 ]]; then
    echo "probe runners: $w2a8_build_dir/run-{gu,down}-n{64,128,256}-probe.sh"
    echo "paired parity runners: $w2a8_build_dir/run-{gu,down}-pair-probe.sh"
    echo "three-width parity runners: $w2a8_build_dir/run-{gu,down}-three-width-probe.sh"
    echo "N64 compatibility runners: $w2a8_build_dir/run-{gu,down}-probe.sh"
    echo "build receipt: $receipt_file"
fi
