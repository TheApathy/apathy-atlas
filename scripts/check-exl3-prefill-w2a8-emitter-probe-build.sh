#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cuda_root_path=${CUDA_ROOT_PATH:-/usr/local/cuda}
nvcc_bin=${NVCC_BIN:-"$cuda_root_path/bin/nvcc"}
cuobjdump_bin=${CUOBJDUMP_BIN:-"$cuda_root_path/bin/cuobjdump"}
host_cxx_bin=${HOST_CXX_BIN:-/usr/bin/g++}

if [[ -n ${W2A8_EMITTER_PROBE_OUTPUT_DIR:-} ]]; then
    emitter_build_dir=$W2A8_EMITTER_PROBE_OUTPUT_DIR
    if [[ -e $emitter_build_dir || -L $emitter_build_dir ]]; then
        echo "refusing existing W2A8_EMITTER_PROBE_OUTPUT_DIR: $emitter_build_dir" >&2
        exit 2
    fi
    mkdir -m 0700 -- "$emitter_build_dir"
    persist_probe=1
else
    emitter_build_dir=$(mktemp -d /tmp/atlas-exl3-w2a8-emitter-probe.XXXXXX)
    trap 'rm -rf -- "$emitter_build_dir"' EXIT
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

# BEGIN emitter probe immutable inputs
harness_file="$repo_root/kernels/gb10/experiments/exl3_w2a8_h128_emit_probe.cu"
emitter_file="$repo_root/kernels/gb10/experiments/exl3_w2a8_h128_emit.cu"
quant_file="$repo_root/kernels/gb10/common/per_token_group_quant_fp8.cu"
common_exl3_file="$repo_root/kernels/gb10/common/exl3_gemv.cu"
build_script="$repo_root/scripts/check-exl3-prefill-w2a8-emitter-probe-build.sh"
dependencies=(
    "$harness_file"
    "$emitter_file"
    "$quant_file"
    "$common_exl3_file"
    "$build_script"
)

declare -A dependency_hashes
for dependency in "${dependencies[@]}"; do
    if [[ ! -f $dependency ]]; then
        echo "missing emitter probe build input: $dependency" >&2
        exit 2
    fi
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
# END emitter probe immutable inputs

# BEGIN emitter probe receipt contract
build_manifest="$emitter_build_dir/build-manifest.txt"
{
    echo "receipt_format=atlas-w2a8-emitter-v1"
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
    echo "compile_command_template=<nvcc> -ccbin <host_cxx> -std=c++17 -O3 --fmad=false -arch=sm_121a '-DW2A8_EMIT_PROBE_BUILD_ID=\"<build_id>\"' <harness> -o <binary>"
} >"$build_manifest"
build_id=$(sha256sum "$build_manifest" | awk '{print $1}')

binary="$emitter_build_dir/exl3-w2a8-h128-emitter-probe"
"$nvcc_bin" -ccbin "$host_cxx_bin" -std=c++17 -O3 --fmad=false \
    -arch=sm_121a "-DW2A8_EMIT_PROBE_BUILD_ID=\"$build_id\"" \
    "$harness_file" -o "$binary"
compile_command="compile_command=$nvcc_bin -ccbin $host_cxx_bin -std=c++17 -O3 --fmad=false -arch=sm_121a '-DW2A8_EMIT_PROBE_BUILD_ID=\"$build_id\"' $harness_file -o $binary"
binary_hash=$(sha256sum "$binary" | awk '{print $1}')

usage_output="$emitter_build_dir/usage-rejection.out"
set +e
"$binary" unexpected-argument >"$usage_output" 2>&1
usage_rc=$?
set -e
if [[ $usage_rc -ne 2 ]] || ! grep -Eqx 'usage: .+' "$usage_output"; then
    echo "emitter probe did not reject arguments before CUDA initialization" >&2
    exit 1
fi

resources="$emitter_build_dir/emitter.resources"
"$cuobjdump_bin" --dump-resource-usage "$binary" >"$resources"
for symbol in \
    exl3_w2a8_h128_pre_dual_emit_h4096 \
    exl3_w2a8_h128_post_silu_pre_emit_h2048 \
    exl3_h128_pre_dual_rows_h4096 \
    exl3_h128_post_silu_pre_rows \
    per_token_group_quant_fp8; do
    grep -q "Function ${symbol}:" "$resources"
done

receipt_lines=("binary_sha256 emitter_probe $binary_hash")
extract_dir="$emitter_build_dir/emitter.cubins"
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
    receipt_lines+=("cubin_sha256 emitter_probe/$cubin_name $cubin_hash")
done < <(find "$extract_dir" -maxdepth 1 -type f -name '*.cubin' | sort)

build_inputs_unchanged=1
for dependency in "${dependencies[@]}"; do
    current_hash=$(sha256sum "$dependency" | awk '{print $1}')
    if [[ $current_hash != "${dependency_hashes[$dependency]}" ]]; then
        echo "build input changed during W2A8 emitter probe compilation: $dependency" >&2
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
        echo "build tool changed during emitter probe compilation: $tool" >&2
        build_inputs_unchanged=0
    fi
done
current_commit=$(git -C "$repo_root" rev-parse HEAD)
current_status_hash=$(git -C "$repo_root" status --porcelain=v1 | sha256sum | awk '{print $1}')
if [[ $current_commit != "$git_commit" || $current_status_hash != "$git_status_hash" ]]; then
    echo "repository identity changed during emitter probe compilation" >&2
    build_inputs_unchanged=0
fi
[[ $build_inputs_unchanged == 1 ]]

receipt_file="$emitter_build_dir/build-receipt.txt"
{
    cat "$build_manifest"
    echo "build_id=$build_id"
    echo "$compile_command"
    printf '%s\n' "${receipt_lines[@]}"
} | tee "$receipt_file"
receipt_hash=$(sha256sum "$receipt_file" | awk '{print $1}')
# END emitter probe receipt contract

# BEGIN emitter probe runner verification
runner="$emitter_build_dir/run-emitter-probe.sh"
{
    echo '#!/usr/bin/env bash'
    echo '# SPDX-License-Identifier: AGPL-3.0-only'
    echo 'set -euo pipefail'
    printf 'expected_receipt_sha256=%q\n' "$receipt_hash"
    printf 'expected_binary_sha256=%q\n' "$binary_hash"
    printf 'expected_build_id=%q\n' "$build_id"
    cat <<'RUNNER'
export LC_ALL=C
if [[ $# -ne 0 ]]; then
    echo "usage: $0" >&2
    exit 2
fi
runner_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
receipt="$runner_dir/build-receipt.txt"
binary="$runner_dir/exl3-w2a8-h128-emitter-probe"
actual_receipt_sha256=$(sha256sum "$receipt" | awk '{print $1}')
actual_binary_sha256=$(sha256sum "$binary" | awk '{print $1}')
if [[ $actual_receipt_sha256 != "$expected_receipt_sha256" ||
    $actual_binary_sha256 != "$expected_binary_sha256" ]] ||
    [[ $(grep -Fxc "build_id=$expected_build_id" "$receipt") -ne 1 ]] ||
    [[ $(grep -Fxc "binary_sha256 emitter_probe $expected_binary_sha256" "$receipt") -ne 1 ]]; then
    echo "W2A8 emitter probe artifact identity mismatch" >&2
    exit 2
fi

probe_tmp=$(mktemp -d "${TMPDIR:-/tmp}/atlas-w2a8-emitter-run.XXXXXX")
trap 'rm -rf -- "$probe_tmp"' EXIT
set +e
"$binary" >"$probe_tmp/stdout" 2>"$probe_tmp/stderr"
probe_rc=$?
set -e
max_output_bytes=4096
max_output_lines=5
stdout_bytes=$(wc -c <"$probe_tmp/stdout")
stderr_bytes=$(wc -c <"$probe_tmp/stderr")
stdout_lines=$(wc -l <"$probe_tmp/stdout")
if [[ $probe_rc -ne 0 || $stdout_bytes -gt $max_output_bytes ||
    $stderr_bytes -ne 0 || $stdout_lines -ne $max_output_lines ]]; then
    echo "W2A8 emitter probe rejected: rc=$probe_rc stdout_bytes=$stdout_bytes stderr_bytes=$stderr_bytes stdout_lines=$stdout_lines" >&2
    exit 1
fi

mapfile -t probe_lines <"$probe_tmp/stdout"
if [[ ${probe_lines[0]} != "build_id=$expected_build_id" ]] ||
    [[ ! ${probe_lines[1]} =~ ^device_uuid=[0-9a-f]{32}\ driver=[0-9]+\ runtime=[0-9]+$ ]] ||
    [[ ! ${probe_lines[2]} =~ ^input_hash=[0-9a-f]{16}\ routing_hash=[0-9a-f]{16}\ tables_hash=[0-9a-f]{16}$ ]] ||
    [[ ${probe_lines[3]} != 'pre_cases=16 post_cases=8 geometry_cases=16 mismatches=0 guards=clean' ]] ||
    [[ ${probe_lines[4]} != 'result=PASS' ]]; then
    echo "W2A8 emitter probe output contract mismatch" >&2
    exit 1
fi
printf 'receipt_sha256=%s\n' "$actual_receipt_sha256"
printf 'binary_sha256=%s\n' "$actual_binary_sha256"
cat "$probe_tmp/stdout"
RUNNER
} >"$runner"
chmod 0755 "$runner"
# END emitter probe runner verification

echo "W2A8 H128 emitter promotion probe compiles for SM121a"
echo "strict no-argument rejection is distinct from CUDA failures"
echo "build_id=$build_id"
echo "receipt_sha256=$receipt_hash"
if [[ $persist_probe == 1 ]]; then
    echo "probe runner: $runner"
    echo "build receipt: $receipt_file"
fi
