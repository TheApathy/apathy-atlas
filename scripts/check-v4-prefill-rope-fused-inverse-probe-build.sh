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

if [[ -n ${V4_INVERSE_ROPE_PROBE_OUTPUT_DIR:-} ]]; then
    build_dir=$V4_INVERSE_ROPE_PROBE_OUTPUT_DIR
    if [[ -e $build_dir || -L $build_dir ]]; then
        echo "refusing existing V4_INVERSE_ROPE_PROBE_OUTPUT_DIR: $build_dir" >&2
        exit 2
    fi
    mkdir -m 0700 -- "$build_dir"
else
    build_dir=$(mktemp -d /tmp/atlas-v4-inverse-rope-probe.XXXXXX)
    trap 'rm -rf -- "$build_dir"' EXIT
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

probe_file="$repo_root/kernels/gb10/experiments/v4_prefill_rope_fused_inverse_probe.cu"
shared_probe_file="$repo_root/kernels/gb10/experiments/v4_prefill_rope_fused_probe.cu"
candidate_file="$repo_root/kernels/gb10/experiments/v4_prefill_rope_fused.cu"
production_file="$repo_root/kernels/gb10/deepseek-v4-flash/nvfp4/v4_prefill_rope_fused_inverse.cu"
rope_file="$repo_root/kernels/gb10/common/rope.cu"
mla_file="$repo_root/kernels/gb10/deepseek-v4-flash/nvfp4/mla_absorbed.cu"
host_flow_file="$repo_root/crates/spark-model/src/layers/qwen3_attention/prefill/cache_skip_v4.rs"
types_file="$repo_root/crates/spark-model/src/layers/qwen3_attention/types.rs"
init_file="$repo_root/crates/spark-model/src/layers/qwen3_attention/init.rs"
build_script="$repo_root/scripts/check-v4-prefill-rope-fused-inverse-probe-build.sh"
dependencies=(
    "$probe_file"
    "$shared_probe_file"
    "$candidate_file"
    "$production_file"
    "$rope_file"
    "$mla_file"
    "$host_flow_file"
    "$types_file"
    "$init_file"
    "$build_script"
)
declare -A dependency_hashes
for dependency in "${dependencies[@]}"; do
    if [[ ! -f $dependency ]]; then
        echo "missing inverse-only RoPE probe input: $dependency" >&2
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
status_paths=(
    "kernels/gb10/experiments/v4_prefill_rope_fused_inverse_probe.cu"
    "kernels/gb10/experiments/v4_prefill_rope_fused_probe.cu"
    "kernels/gb10/experiments/v4_prefill_rope_fused.cu"
    "kernels/gb10/deepseek-v4-flash/nvfp4/v4_prefill_rope_fused_inverse.cu"
    "kernels/gb10/common/rope.cu"
    "kernels/gb10/deepseek-v4-flash/nvfp4/mla_absorbed.cu"
    "crates/spark-model/src/layers/qwen3_attention/prefill/cache_skip_v4.rs"
    "crates/spark-model/src/layers/qwen3_attention/types.rs"
    "crates/spark-model/src/layers/qwen3_attention/init.rs"
    "scripts/check-v4-prefill-rope-fused-inverse-probe-build.sh"
)
git_status_hash=$(git -C "$repo_root" status --porcelain=v1 -- "${status_paths[@]}" |
    sha256sum | awk '{print $1}')

manifest="$build_dir/build-manifest.txt"
{
    echo "receipt_format=atlas-v4-prefill-inverse-rope-probe-v1"
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
    echo "compile_command_template=<nvcc> -ccbin <host_cxx> -std=c++17 -O3 --fmad=false -arch=sm_121a '-DV4_ROPE_PROBE_BUILD_ID=\"<build_id>\"' <probe> -o <binary>"
    echo "production_compile_command_template=<nvcc> -std=c++17 -O3 --fmad=false -arch=sm_121a -cubin <production_wrapper> -o <production_cubin>"
} >"$manifest"
build_id=$(sha256sum "$manifest" | awk '{print $1}')

binary="$build_dir/v4-prefill-rope-fused-inverse-probe"
production_cubin="$build_dir/v4-prefill-rope-fused-inverse.cubin"
"$nvcc_bin" -ccbin "$host_cxx_bin" -std=c++17 -O3 --fmad=false -arch=sm_121a \
    "-DV4_ROPE_PROBE_BUILD_ID=\"$build_id\"" "$probe_file" -o "$binary"
"$nvcc_bin" -std=c++17 -O3 --fmad=false -arch=sm_121a -cubin \
    "$production_file" -o "$production_cubin"
compile_command="compile_command=$nvcc_bin -ccbin $host_cxx_bin -std=c++17 -O3 --fmad=false -arch=sm_121a '-DV4_ROPE_PROBE_BUILD_ID=\"$build_id\"' $probe_file -o $binary"
production_compile_command="production_compile_command=$nvcc_bin -std=c++17 -O3 --fmad=false -arch=sm_121a -cubin $production_file -o $production_cubin"

reject_threshold() {
    local label=$1
    shift
    local output="$build_dir/reject-$label.out"
    set +e
    "$binary" "$@" >"$output" 2>&1
    local result=$?
    set -e
    [[ $result -eq 2 ]] || { echo "inverse-only probe accepted invalid threshold: $label" >&2; exit 1; }
}
reject_threshold usage
reject_threshold usage_extra 1.01 extra
reject_threshold junk junk
reject_threshold nan nan
reject_threshold no_win 1
reject_threshold lax 100.1

resources="$build_dir/probe.resources"
production_resources="$build_dir/production.resources"
"$cuobjdump_bin" --dump-resource-usage "$binary" >"$resources"
"$cuobjdump_bin" --dump-resource-usage "$production_cubin" >"$production_resources"
for symbol in v4_prefill_rope_fused_inverse mla_q_rope_extract_batched \
    mla_q_rope_writeback_batched rope_forward_yarn_interleaved_inv; do
    grep -q "Function $symbol:" "$resources"
done
grep -q "Function v4_prefill_rope_fused_inverse:" "$production_resources"

extract_dir="$build_dir/cubins"
mkdir "$extract_dir"
(cd "$extract_dir" && "$cuobjdump_bin" --extract-elf all "$binary" >/dev/null)
mapfile -t cubins < <(find "$extract_dir" -maxdepth 1 -type f -name '*.cubin' | sort)
[[ ${#cubins[@]} -ge 1 ]]

unchanged=1
for dependency in "${dependencies[@]}"; do
    if [[ $(sha256sum "$dependency" | awk '{print $1}') != "${dependency_hashes[$dependency]}" ]]; then
        echo "build input changed during inverse-only RoPE probe compilation: $dependency" >&2
        unchanged=0
    fi
done
for record in \
    "$nvcc_bin|$nvcc_real_path|$nvcc_binary_hash" \
    "$cuobjdump_bin|$cuobjdump_real_path|$cuobjdump_binary_hash" \
    "$host_cxx_bin|$host_cxx_real_path|$host_cxx_binary_hash"; do
    IFS='|' read -r tool expected_path expected_hash <<<"$record"
    if [[ $(realpath "$tool") != "$expected_path" ]] ||
        [[ $(sha256sum "$tool" | awk '{print $1}') != "$expected_hash" ]]; then
        echo "build tool changed during inverse-only RoPE probe compilation: $tool" >&2
        unchanged=0
    fi
done
current_status_hash=$(git -C "$repo_root" status --porcelain=v1 -- "${status_paths[@]}" |
    sha256sum | awk '{print $1}')
if [[ $(git -C "$repo_root" rev-parse HEAD) != "$git_commit" ||
    $current_status_hash != "$git_status_hash" ]]; then
    echo "repository identity changed during inverse-only RoPE probe compilation" >&2
    unchanged=0
fi
[[ $unchanged == 1 ]]

binary_hash=$(sha256sum "$binary" | awk '{print $1}')
production_cubin_hash=$(sha256sum "$production_cubin" | awk '{print $1}')
receipt="$build_dir/build-receipt.txt"
{
    cat "$manifest"
    echo "build_id=$build_id"
    echo "$compile_command"
    echo "$production_compile_command"
    echo "binary_sha256 probe $binary_hash"
    echo "cubin_sha256 production $(basename "$production_cubin") $production_cubin_hash"
    for cubin in "${cubins[@]}"; do
        echo "cubin_sha256 probe cubins/$(basename "$cubin") $(sha256sum "$cubin" | awk '{print $1}')"
    done
} >"$receipt"
receipt_hash=$(sha256sum "$receipt" | awk '{print $1}')

runner="$build_dir/run-v4-prefill-rope-fused-inverse-probe.sh"
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
binary="$runner_dir/v4-prefill-rope-fused-inverse-probe"
if [[ $(sha256sum "$receipt" | awk '{print $1}') != "$expected_receipt_sha256" ]]; then
    echo "receipt identity mismatch" >&2
    exit 1
fi
if [[ $(sha256sum "$binary" | awk '{print $1}') != "$expected_binary_sha256" ]]; then
    echo "binary identity mismatch" >&2
    exit 1
fi
if ! grep -Fx "$expected_build_id" < <(strings "$binary") >/dev/null; then
    echo "embedded build ID mismatch" >&2
    exit 1
fi
while read -r _ kind relative expected_hash; do
    if [[ $kind == production ]]; then
        artifact="$runner_dir/$relative"
    else
        artifact="$runner_dir/$relative"
    fi
    if [[ ! -f $artifact || $(sha256sum "$artifact" | awk '{print $1}') != "$expected_hash" ]]; then
        echo "cubin identity mismatch: $relative" >&2
        exit 1
    fi
done < <(awk '$1 == "cubin_sha256" { print }' "$receipt")
stdout_file="$runner_dir/run.stdout"
stderr_file="$runner_dir/run.stderr"
set +e
"$binary" "$expected_min_speedup" >"$stdout_file" 2>"$stderr_file"
probe_rc=$?
set -e
max_output_bytes=4096
max_output_lines=8
stdout_bytes=$(wc -c <"$stdout_file")
stdout_lines=$(wc -l <"$stdout_file")
stderr_bytes=$(wc -c <"$stderr_file")
if [[ $probe_rc -ne 0 || $stdout_bytes -gt $max_output_bytes ||
    $stdout_lines -gt $max_output_lines || $stderr_bytes -ne 0 ]]; then
    echo "inverse-only RoPE probe rejected: rc=$probe_rc stdout_bytes=$stdout_bytes stderr_bytes=$stderr_bytes stdout_lines=$stdout_lines" >&2
    exit 1
fi
grep -qx "build_id=$expected_build_id" "$stdout_file"
grep -Eqx 'device_uuid=[0-9a-f]{32} driver=[0-9]+ runtime=[0-9]+' "$stdout_file"
grep -Eqx 'input_hash=[0-9a-f]{16} position_hash=[0-9a-f]{16} frequency_hash=[0-9a-f]{16}' "$stdout_file"
grep -qx 'parity cases=3 inverse_mismatch_bytes=0' "$stdout_file"
grep -qx 'guards redzone_buffers=2 prefix=clean suffix=clean input_unchanged=1 table_unchanged=1' "$stdout_file"
grep -Eqx 'timing baseline_ms=[^ ]+ candidate_ms=[^ ]+ speedup=[^ ]+ abba_rounds=6' "$stdout_file"
grep -qx "threshold min_speedup=$expected_min_speedup" "$stdout_file"
grep -qx 'result=PASS' "$stdout_file"
cat "$stdout_file"
RUNNER
} >"$runner"
chmod 0700 "$runner"
runner_sha256=$(sha256sum "$runner" | awk '{print $1}')

echo "inverse-only V4 prefill RoPE promotion probe compiles for SM121a"
echo "build_id=$build_id"
echo "receipt_sha256=$receipt_hash"
echo "runner_sha256=$runner_sha256"
echo "runner=$runner"
