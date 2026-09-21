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

if [[ -n ${V4_ROPE_PROBE_OUTPUT_DIR:-} ]]; then
    build_dir=$V4_ROPE_PROBE_OUTPUT_DIR
    if [[ -e $build_dir || -L $build_dir ]]; then
        echo "refusing existing V4_ROPE_PROBE_OUTPUT_DIR: $build_dir" >&2
        exit 2
    fi
    mkdir -m 0700 -- "$build_dir"
    persist_probe=1
else
    build_dir=$(mktemp -d /tmp/atlas-v4-rope-probe.XXXXXX)
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

# BEGIN V4 RoPE probe immutable inputs
harness_file="$repo_root/kernels/gb10/experiments/v4_prefill_rope_fused_probe.cu"
candidate_file="$repo_root/kernels/gb10/experiments/v4_prefill_rope_fused.cu"
rope_file="$repo_root/kernels/gb10/common/rope.cu"
mla_file="$repo_root/kernels/gb10/deepseek-v4-flash/nvfp4/mla_absorbed.cu"
host_flow_file="$repo_root/crates/spark-model/src/layers/qwen3_attention/prefill/cache_skip_v4.rs"
ops_wrapper_file="$repo_root/crates/spark-model/src/layers/ops/embeddings.rs"
build_script="$repo_root/scripts/check-v4-prefill-rope-fused-probe-build.sh"
dependencies=("$harness_file" "$candidate_file" "$rope_file" "$mla_file" \
    "$host_flow_file" "$ops_wrapper_file" "$build_script")
declare -A dependency_hashes
for dependency in "${dependencies[@]}"; do
    [[ -f $dependency ]] || { echo "missing V4 RoPE probe input: $dependency" >&2; exit 2; }
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
status_paths=(
    "kernels/gb10/experiments/v4_prefill_rope_fused_probe.cu"
    "kernels/gb10/experiments/v4_prefill_rope_fused.cu"
    "kernels/gb10/common/rope.cu"
    "kernels/gb10/deepseek-v4-flash/nvfp4/mla_absorbed.cu"
    "crates/spark-model/src/layers/qwen3_attention/prefill/cache_skip_v4.rs"
    "crates/spark-model/src/layers/ops/embeddings.rs"
    "scripts/check-v4-prefill-rope-fused-probe-build.sh"
)
git_status_hash=$(git -C "$repo_root" status --porcelain=v1 -- "${status_paths[@]}" |
    sha256sum | awk '{print $1}')
# END V4 RoPE probe immutable inputs

# BEGIN V4 RoPE probe receipt contract
manifest="$build_dir/build-manifest.txt"
{
    echo "receipt_format=atlas-v4-prefill-rope-fused-probe-v1"
    echo "git_commit=$git_commit"
    echo "git_status_sha256=$git_status_hash"
    echo "min_speedup=$min_speedup"
    for dependency in "${dependencies[@]}"; do
        echo "source_sha256 ${dependency#"$repo_root/"} ${dependency_hashes[$dependency]}"
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
    echo "compile_command_template=<nvcc> -ccbin <host_cxx> -std=c++17 -O3 --fmad=false -arch=sm_121a '-DV4_ROPE_PROBE_BUILD_ID=\"<build_id>\"' <harness> -o <binary>"
    echo "resource_command_template=<cuobjdump> --dump-resource-usage <binary>"
    echo "cubin_extract_command_template=<cuobjdump> --extract-elf all <binary>"
} >"$manifest"
build_id=$(sha256sum "$manifest" | awk '{print $1}')

binary="$build_dir/v4-prefill-rope-fused-probe"
"$nvcc_bin" -ccbin "$host_cxx_bin" -std=c++17 -O3 --fmad=false -arch=sm_121a \
    "-DV4_ROPE_PROBE_BUILD_ID=\"$build_id\"" "$harness_file" -o "$binary"
compile_command="compile_command=$nvcc_bin -ccbin $host_cxx_bin -std=c++17 -O3 --fmad=false -arch=sm_121a '-DV4_ROPE_PROBE_BUILD_ID=\"$build_id\"' $harness_file -o $binary"
binary_hash=$(sha256sum "$binary" | awk '{print $1}')

reject_threshold() {
    local label=$1
    shift
    local output="$build_dir/reject-$label.out"
    set +e
    "$binary" "$@" >"$output" 2>&1
    local result=$?
    set -e
    [[ $result -eq 2 ]] || { echo "probe accepted invalid threshold: $label" >&2; exit 1; }
    if [[ $label == usage* ]]; then
        grep -Eqx 'usage: .+ <min_speedup>' "$output"
    else
        grep -qx 'invalid explicit numeric threshold' "$output"
    fi
}
reject_threshold usage
reject_threshold usage_extra 1.05 extra
reject_threshold junk junk
reject_threshold nan nan
reject_threshold inf inf
reject_threshold whitespace ' 1.01'
reject_threshold hexadecimal 0x1.1p1
reject_threshold no_win 1
reject_threshold lax 100.1

resources="$build_dir/probe.resources"
"$cuobjdump_bin" --dump-resource-usage "$binary" >"$resources"
for symbol in v4_prefill_rope_fused_forward v4_prefill_rope_fused_inverse \
    mla_q_rope_extract_batched mla_q_rope_writeback_batched \
    rope_forward_yarn_interleaved rope_forward_yarn_interleaved_inv; do
    grep -q "Function $symbol:" "$resources"
done

extract_dir="$build_dir/cubins"
mkdir "$extract_dir"
(cd "$extract_dir" && "$cuobjdump_bin" --extract-elf all "$binary" >/dev/null)
mapfile -t cubins < <(find "$extract_dir" -maxdepth 1 -type f -name '*.cubin' | sort)
[[ ${#cubins[@]} -ge 1 ]]

unchanged=1
for dependency in "${dependencies[@]}"; do
    if [[ $(sha256sum "$dependency" | awk '{print $1}') != "${dependency_hashes[$dependency]}" ]]; then
        echo "build input changed during V4 RoPE probe compilation: $dependency" >&2
        unchanged=0
    fi
done
for tool_record in \
    "$nvcc_bin|$nvcc_real_path|$nvcc_binary_hash" \
    "$cuobjdump_bin|$cuobjdump_real_path|$cuobjdump_binary_hash" \
    "$host_cxx_bin|$host_cxx_real_path|$host_cxx_binary_hash"; do
    IFS='|' read -r tool expected_path expected_hash <<<"$tool_record"
    if [[ $(realpath "$tool") != "$expected_path" ]] ||
        [[ $(sha256sum "$tool" | awk '{print $1}') != "$expected_hash" ]]; then
        echo "build tool changed during V4 RoPE probe compilation: $tool" >&2
        unchanged=0
    fi
done
current_commit=$(git -C "$repo_root" rev-parse HEAD)
current_status_hash=$(git -C "$repo_root" status --porcelain=v1 -- "${status_paths[@]}" |
    sha256sum | awk '{print $1}')
if [[ $current_commit != "$git_commit" || $current_status_hash != "$git_status_hash" ]]; then
    echo "repository identity changed during V4 RoPE probe compilation" >&2
    unchanged=0
fi
[[ $unchanged == 1 ]]

receipt="$build_dir/build-receipt.txt"
{
    cat "$manifest"
    echo "build_id=$build_id"
    echo "$compile_command"
    echo "resource_command=$cuobjdump_bin --dump-resource-usage $binary"
    echo "cubin_extract_command=$cuobjdump_bin --extract-elf all $binary"
    echo "binary_sha256 probe $binary_hash"
    for cubin in "${cubins[@]}"; do
        echo "cubin_sha256 cubins/$(basename "$cubin") $(sha256sum "$cubin" | awk '{print $1}')"
    done
} | tee "$receipt"
receipt_hash=$(sha256sum "$receipt" | awk '{print $1}')
# END V4 RoPE probe receipt contract

# BEGIN V4 RoPE probe runner verification
runner="$build_dir/run-v4-prefill-rope-fused-probe.sh"
{
    echo '#!/usr/bin/env bash'
    echo '# SPDX-License-Identifier: AGPL-3.0-only'
    echo 'set -euo pipefail'
    echo 'export LC_ALL=C'
    printf 'expected_receipt_sha256=%q\n' "$receipt_hash"
    printf 'expected_binary_sha256=%q\n' "$binary_hash"
    printf 'expected_build_id=%q\n' "$build_id"
    printf 'expected_min_speedup=%q\n' "$min_speedup"
    cat <<'RUNNER'
if [[ $# -ne 0 ]]; then
    echo "usage: $0" >&2
    exit 2
fi
runner_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
receipt="$runner_dir/build-receipt.txt"
binary="$runner_dir/v4-prefill-rope-fused-probe"
actual_receipt_sha256=$(sha256sum "$receipt" | awk '{print $1}')
actual_binary_sha256=$(sha256sum "$binary" | awk '{print $1}')
if [[ $actual_receipt_sha256 != "$expected_receipt_sha256" ||
    $actual_binary_sha256 != "$expected_binary_sha256" ]] ||
    [[ $(grep -Fxc "build_id=$expected_build_id" "$receipt") -ne 1 ]] ||
    [[ $(grep -Fxc "min_speedup=$expected_min_speedup" "$receipt") -ne 1 ]] ||
    [[ $(grep -Fxc "binary_sha256 probe $expected_binary_sha256" "$receipt") -ne 1 ]]; then
    echo "V4 RoPE probe artifact identity mismatch" >&2
    exit 2
fi
while read -r _ relative expected_hash; do
    artifact="$runner_dir/$relative"
    if [[ ! -f $artifact || $(sha256sum "$artifact" | awk '{print $1}') != "$expected_hash" ]]; then
        echo "V4 RoPE probe cubin identity mismatch: $relative" >&2
        exit 2
    fi
done < <(grep '^cubin_sha256 ' "$receipt")

probe_tmp=$(mktemp -d "${TMPDIR:-/tmp}/atlas-v4-rope-run.XXXXXX")
trap 'rm -rf -- "$probe_tmp"' EXIT
set +e
"$binary" "$expected_min_speedup" >"$probe_tmp/stdout" 2>"$probe_tmp/stderr"
probe_rc=$?
set -e
max_output_bytes=4096
max_output_lines=8
stdout_bytes=$(wc -c <"$probe_tmp/stdout")
stderr_bytes=$(wc -c <"$probe_tmp/stderr")
stdout_lines=$(wc -l <"$probe_tmp/stdout")
if [[ $probe_rc -ne 0 || $stdout_bytes -gt $max_output_bytes || $stderr_bytes -ne 0 ||
    $stdout_lines -ne $max_output_lines ]]; then
    echo "V4 RoPE probe rejected: rc=$probe_rc stdout_bytes=$stdout_bytes stderr_bytes=$stderr_bytes stdout_lines=$stdout_lines" >&2
    exit 1
fi
mapfile -t lines <"$probe_tmp/stdout"
number='[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?'
if [[ ${lines[0]} != "build_id=$expected_build_id" ]] ||
    [[ ! ${lines[1]} =~ ^device_uuid=[0-9a-f]{32}\ driver=[0-9]+\ runtime=[0-9]+$ ]] ||
    [[ ! ${lines[2]} =~ ^input_hash=[0-9a-f]{16}\ position_hash=[0-9a-f]{16}\ frequency_hash=[0-9a-f]{16}$ ]] ||
    [[ ${lines[3]} != 'parity cases=3 forward_mismatches=0 inverse_mismatches=0' ]] ||
    [[ ${lines[4]} != 'guards poison_cases=38 unchanged=38 redzone_buffers=6 prefix=clean suffix=clean' ]] ||
    [[ ! ${lines[5]} =~ ^timing\ baseline_ms=$number\ candidate_ms=$number\ speedup=$number\ abba_rounds=6$ ]] ||
    [[ ${lines[6]} != "threshold min_speedup=$expected_min_speedup" ]] ||
    [[ ${lines[7]} != 'result=PASS' ]]; then
    echo "V4 RoPE probe output contract mismatch" >&2
    exit 1
fi
printf 'receipt_sha256=%s\n' "$actual_receipt_sha256"
printf 'binary_sha256=%s\n' "$actual_binary_sha256"
cat "$probe_tmp/stdout"
RUNNER
} >"$runner"
chmod 0755 "$runner"
# END V4 RoPE probe runner verification

echo "V4 fused RoPE promotion probe compiles for SM121a"
echo "invalid thresholds reject before CUDA; runtime evidence requires the generated runner"
echo "build_id=$build_id"
echo "receipt_sha256=$receipt_hash"
if [[ $persist_probe == 1 ]]; then
    echo "probe runner: $runner"
    echo "build receipt: $receipt"
fi
