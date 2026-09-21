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

if [[ -n ${V4_INV_QUANT_PROBE_OUTPUT_DIR:-} ]]; then
    build_dir=$V4_INV_QUANT_PROBE_OUTPUT_DIR
    if [[ -e $build_dir || -L $build_dir ]]; then
        echo "refusing existing V4_INV_QUANT_PROBE_OUTPUT_DIR: $build_dir" >&2
        exit 2
    fi
    mkdir -m 0700 -- "$build_dir"
    persist_probe=1
else
    build_dir=$(mktemp -d /tmp/atlas-v4-inverse-quant-probe.XXXXXX)
    cleanup() {
        local exit_code=$?
        rm -rf -- "$build_dir"
        trap - EXIT
        exit "$exit_code"
    }
    trap cleanup EXIT
    persist_probe=0
fi

# BEGIN V4 inverse quant probe immutable inputs
probe_file="$repo_root/kernels/gb10/experiments/v4_prefill_inverse_rope_w8a8_quant_fused_probe.cu"
candidate_file="$repo_root/kernels/gb10/experiments/v4_prefill_inverse_rope_w8a8_quant_fused.cu"
direct_rope_file="$repo_root/kernels/gb10/experiments/v4_prefill_rope_fused.cu"
quantizer_file="$repo_root/kernels/gb10/common/w8a8_gemm_pipelined.cu"
build_script="$repo_root/scripts/check-v4-prefill-inverse-rope-w8a8-quant-fused-probe-build.sh"
dependencies=("$probe_file" "$candidate_file" "$direct_rope_file" "$quantizer_file" "$build_script")
declare -A dependency_hashes
for dependency in "${dependencies[@]}"; do
    if [[ ! -f $dependency ]]; then
        echo "missing V4 inverse quant probe input: $dependency" >&2
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
nvcc_binary_sha256=$(sha256sum "$nvcc_bin" | awk '{print $1}')
cuobjdump_binary_sha256=$(sha256sum "$cuobjdump_bin" | awk '{print $1}')
host_cxx_binary_sha256=$(sha256sum "$host_cxx_bin" | awk '{print $1}')
git_commit=$(git -C "$repo_root" rev-parse HEAD)
status_paths=(
    "kernels/gb10/experiments/v4_prefill_inverse_rope_w8a8_quant_fused_probe.cu"
    "kernels/gb10/experiments/v4_prefill_inverse_rope_w8a8_quant_fused.cu"
    "kernels/gb10/experiments/v4_prefill_rope_fused.cu"
    "kernels/gb10/common/w8a8_gemm_pipelined.cu"
    "scripts/check-v4-prefill-inverse-rope-w8a8-quant-fused-probe-build.sh"
)
git_status_sha256=$(git -C "$repo_root" status --porcelain=v1 -- "${status_paths[@]}" |
    sha256sum | awk '{print $1}')
# END V4 inverse quant probe immutable inputs

# BEGIN V4 inverse quant probe receipt
manifest="$build_dir/build-manifest.txt"
{
    echo "receipt_format=atlas-v4-prefill-inverse-rope-w8a8-quant-fused-probe-v1"
    echo "git_commit=$git_commit"
    echo "git_status_sha256=$git_status_sha256"
    echo "min_speedup=$min_speedup"
    for dependency in "${dependencies[@]}"; do
        echo "source_sha256 ${dependency#"$repo_root/"} ${dependency_hashes[$dependency]}"
    done
    echo "nvcc_path=$nvcc_real_path"
    echo "nvcc_version=$nvcc_version"
    echo "nvcc_binary_sha256=$nvcc_binary_sha256"
    echo "cuobjdump_path=$cuobjdump_real_path"
    echo "cuobjdump_version=$cuobjdump_version"
    echo "cuobjdump_binary_sha256=$cuobjdump_binary_sha256"
    echo "host_cxx_path=$host_cxx_real_path"
    echo "host_cxx_version=$host_cxx_version"
    echo "host_cxx_binary_sha256=$host_cxx_binary_sha256"
    echo "compile_command_template=<nvcc> -ccbin <host_cxx> -std=c++17 -O3 --fmad=false -arch=sm_121a -DV4_INV_QUANT_PROBE_BUILD_ID=<build_id> <probe> -o <binary>"
    echo "resource_command_template=<cuobjdump> --dump-resource-usage <binary>"
    echo "extract_command_template=<cuobjdump> --extract-elf all <binary>"
} >"$manifest"
build_id=$(sha256sum "$manifest" | awk '{print $1}')

binary="$build_dir/v4-prefill-inverse-rope-w8a8-quant-fused-probe"
compile_log="$build_dir/compile.log"
compile_command="$nvcc_bin -ccbin $host_cxx_bin -std=c++17 -O3 --fmad=false -arch=sm_121a '-DV4_INV_QUANT_PROBE_BUILD_ID=\"$build_id\"' $probe_file -o $binary -Xptxas=-v"
compile_command_sha256=$(printf %s "$compile_command" | sha256sum | awk '{print $1}')
"$nvcc_bin" -ccbin "$host_cxx_bin" -std=c++17 -O3 --fmad=false -arch=sm_121a \
    "-DV4_INV_QUANT_PROBE_BUILD_ID=\"$build_id\"" "$probe_file" -o "$binary" \
    -Xptxas=-v 2>"$compile_log"
binary_sha256=$(sha256sum "$binary" | awk '{print $1}')
compile_log_sha256=$(sha256sum "$compile_log" | awk '{print $1}')

reject_threshold() {
    local label=$1
    shift
    local output="$build_dir/reject-$label.out"
    set +e
    "$binary" "$@" >"$output" 2>&1
    local result=$?
    set -e
    if [[ $result -ne 2 ]]; then
        echo "V4 inverse quant probe accepted invalid threshold: $label" >&2
        exit 1
    fi
    if [[ $label == usage* ]]; then
        grep -Eqx 'usage: .* <min_speedup>' "$output"
    else
        grep -qx 'invalid explicit numeric threshold' "$output"
    fi
}
reject_threshold usage
reject_threshold usage_extra 1.01 extra
reject_threshold junk junk
reject_threshold nan nan
reject_threshold inf inf
reject_threshold whitespace ' 1.01'
reject_threshold trailing '1.01 '
reject_threshold hexadecimal 0x1.1p1
reject_threshold negative -1
reject_threshold zero 0
reject_threshold no_win 1
reject_threshold lax 100.1

resources="$build_dir/resources.txt"
resource_command="$cuobjdump_bin --dump-resource-usage $binary"
resource_command_sha256=$(printf %s "$resource_command" | sha256sum | awk '{print $1}')
"$cuobjdump_bin" --dump-resource-usage "$binary" >"$resources"
for symbol in v4_prefill_rope_fused_inverse quantize_a_fp8_rows \
    v4_prefill_inverse_rope_w8a8_quant_fused; do
    grep -q "Function $symbol:" "$resources"
done
resources_sha256=$(sha256sum "$resources" | awk '{print $1}')

cubin_dir="$build_dir/cubins"
mkdir "$cubin_dir"
extract_command="$cuobjdump_bin --extract-elf all $binary"
extract_command_sha256=$(printf %s "$extract_command" | sha256sum | awk '{print $1}')
(cd "$cubin_dir" && "$cuobjdump_bin" --extract-elf all "$binary" >/dev/null)
mapfile -t cubins < <(find "$cubin_dir" -maxdepth 1 -type f -name '*.cubin' | sort)
if [[ ${#cubins[@]} -lt 1 ]]; then
    echo "V4 inverse quant probe produced no cubin" >&2
    exit 1
fi

for dependency in "${dependencies[@]}"; do
    if [[ $(sha256sum "$dependency" | awk '{print $1}') != "${dependency_hashes[$dependency]}" ]]; then
        echo "build input changed during V4 inverse quant probe compilation: $dependency" >&2
        exit 1
    fi
done
for tool_record in \
    "$nvcc_bin|$nvcc_real_path|$nvcc_binary_sha256" \
    "$cuobjdump_bin|$cuobjdump_real_path|$cuobjdump_binary_sha256" \
    "$host_cxx_bin|$host_cxx_real_path|$host_cxx_binary_sha256"; do
    IFS='|' read -r tool expected_path expected_hash <<<"$tool_record"
    if [[ $(realpath "$tool") != "$expected_path" ]] ||
        [[ $(sha256sum "$tool" | awk '{print $1}') != "$expected_hash" ]]; then
        echo "build tool changed during V4 inverse quant probe compilation: $tool" >&2
        exit 1
    fi
done
if [[ $(git -C "$repo_root" rev-parse HEAD) != "$git_commit" ]] ||
    [[ $(git -C "$repo_root" status --porcelain=v1 -- "${status_paths[@]}" |
        sha256sum | awk '{print $1}') != "$git_status_sha256" ]]; then
    echo "repository identity changed during V4 inverse quant probe compilation" >&2
    exit 1
fi

receipt="$build_dir/build-receipt.txt"
{
    cat "$manifest"
    echo "build_id=$build_id"
    echo "compile_command=$compile_command"
    echo "compile_command_sha256=$compile_command_sha256"
    echo "compile_log_sha256=$compile_log_sha256"
    echo "resource_command=$resource_command"
    echo "resource_command_sha256=$resource_command_sha256"
    echo "resources_sha256=$resources_sha256"
    echo "extract_command=$extract_command"
    echo "extract_command_sha256=$extract_command_sha256"
    echo "binary_sha256 probe $binary_sha256"
    for cubin in "${cubins[@]}"; do
        echo "cubin_sha256 cubins/$(basename "$cubin") $(sha256sum "$cubin" | awk '{print $1}')"
    done
} >"$receipt"
receipt_sha256=$(sha256sum "$receipt" | awk '{print $1}')
# END V4 inverse quant probe receipt

# BEGIN V4 inverse quant runner verification
runner="$build_dir/run-v4-prefill-inverse-rope-w8a8-quant-fused-probe.sh"
{
    echo '#!/usr/bin/env bash'
    echo '# SPDX-License-Identifier: AGPL-3.0-only'
    echo 'set -euo pipefail'
    echo 'export LC_ALL=C'
    printf 'expected_receipt_sha256=%q\n' "$receipt_sha256"
    printf 'expected_binary_sha256=%q\n' "$binary_sha256"
    printf 'expected_build_id=%q\n' "$build_id"
    printf 'expected_min_speedup=%q\n' "$min_speedup"
    cat <<'RUNNER'
if [[ $# -ne 0 ]]; then
    echo "usage: $0" >&2
    exit 2
fi
runner_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
receipt="$runner_dir/build-receipt.txt"
binary="$runner_dir/v4-prefill-inverse-rope-w8a8-quant-fused-probe"
actual_receipt_sha256=$(sha256sum "$receipt" | awk '{print $1}')
actual_binary_sha256=$(sha256sum "$binary" | awk '{print $1}')
if [[ $actual_receipt_sha256 != "$expected_receipt_sha256" ||
    $actual_binary_sha256 != "$expected_binary_sha256" ]] ||
    [[ $(grep -Fxc "build_id=$expected_build_id" "$receipt") -ne 1 ]] ||
    [[ $(grep -Fxc "min_speedup=$expected_min_speedup" "$receipt") -ne 1 ]] ||
    [[ $(grep -Fxc "binary_sha256 probe $expected_binary_sha256" "$receipt") -ne 1 ]]; then
    echo "V4 inverse quant probe artifact identity mismatch" >&2
    exit 2
fi
while read -r _ relative expected_hash; do
    artifact="$runner_dir/$relative"
    if [[ ! -f $artifact ]]; then
        echo "V4 inverse quant probe cubin missing: $relative" >&2
        exit 2
    fi
    actual_cubin_sha256=$(sha256sum "$artifact" | awk '{print $1}')
    if [[ $actual_cubin_sha256 != "$expected_hash" ]]; then
        echo "V4 inverse quant probe cubin identity mismatch: $relative" >&2
        exit 2
    fi
done < <(grep '^cubin_sha256 ' "$receipt")

probe_tmp=$(mktemp -d "${TMPDIR:-/tmp}/atlas-v4-inverse-quant-run.XXXXXX")
cleanup() {
    local exit_code=$?
    rm -rf -- "$probe_tmp"
    trap - EXIT
    exit "$exit_code"
}
trap cleanup EXIT
set +e
"$binary" "$expected_min_speedup" >"$probe_tmp/stdout" 2>"$probe_tmp/stderr"
probe_result=$?
set -e
max_output_bytes=4096
max_output_lines=8
stdout_bytes=$(wc -c <"$probe_tmp/stdout")
stderr_bytes=$(wc -c <"$probe_tmp/stderr")
stdout_lines=$(wc -l <"$probe_tmp/stdout")
if [[ $probe_result -ne 0 || $stdout_bytes -gt $max_output_bytes ||
    $stderr_bytes -ne 0 || $stdout_lines -ne $max_output_lines ]]; then
    echo "V4 inverse quant probe rejected: rc=$probe_result stdout_bytes=$stdout_bytes stderr_bytes=$stderr_bytes stdout_lines=$stdout_lines" >&2
    exit 1
fi
mapfile -t lines <"$probe_tmp/stdout"
number='[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?'
if [[ ${lines[0]} != "build_id=$expected_build_id" ]] ||
    [[ ! ${lines[1]} =~ ^device_uuid=[0-9a-f]{32}\ driver=[0-9]+\ runtime=[0-9]+$ ]] ||
    [[ ! ${lines[2]} =~ ^input_hash=[0-9a-f]{16}\ position_hash=[0-9a-f]{16}\ frequency_hash=[0-9a-f]{16}$ ]] ||
    [[ ${lines[3]} != 'shapes=1,7,128,2410 row=32768 parity_cases=4 poison_cases=2 candidate_input_unchanged=yes' ]] ||
    [[ ${lines[4]} != "threshold min_speedup=$expected_min_speedup" ]] ||
    [[ ! ${lines[5]} =~ ^timing\ baseline_ms=$number\ candidate_ms=$number\ speedup=$number\ abba_rounds=8$ ]] ||
    [[ ${lines[6]} != 'fp8_mismatches=0 scale_mismatches=0 malformed_cases=22 no_write=22 guards=clean poison_a=clean poison_b=clean' ]] ||
    [[ ${lines[7]} != 'result=PASS' ]]; then
    echo "V4 inverse quant probe output contract mismatch" >&2
    exit 1
fi
printf 'receipt_sha256=%s\n' "$actual_receipt_sha256"
printf 'binary_sha256=%s\n' "$actual_binary_sha256"
cat "$probe_tmp/stdout"
RUNNER
} >"$runner"
chmod 0555 "$runner"
# END V4 inverse quant runner verification

echo "V4 inverse-RoPE/W8A8 fused promotion probe build: PASS"
echo "invalid thresholds reject before CUDA; runtime evidence requires the generated runner"
echo "min_speedup=$min_speedup"
echo "build_id=$build_id"
echo "receipt_sha256=$receipt_sha256"
echo "binary_sha256=$binary_sha256"
if [[ $persist_probe == 1 ]]; then
    echo "retained_receipt=$receipt"
    echo "retained_runner=$runner"
fi
