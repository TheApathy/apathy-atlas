#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail
export LC_ALL=C

canonicalize_binary64_threshold() {
    local value=$1
    local number_pattern='^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$'
    [[ $value =~ $number_pattern ]] || return 1
    awk -v value="$value" 'BEGIN {
        value += 0.0
        if (!(value > 1.0 && value <= 100.0)) exit 1
        printf "%.17g\n", value
    }'
}

if [[ $# -ne 1 ]] || ! min_speedup=$(canonicalize_binary64_threshold "${1:-}"); then
    echo "usage: $0 <min_speedup>" >&2
    echo "invalid explicit numeric threshold" >&2
    exit 2
fi
[[ $(canonicalize_binary64_threshold 1.00000001) == 1.0000000099999999 ]]
for invalid in ' 1.01' '1.01 ' 0x1.1p1 nan inf -inf 1 1.00000000000000001 100.1; do
    if canonicalize_binary64_threshold "$invalid" >/dev/null 2>&1; then
        echo "threshold canonicalizer accepted invalid regression: $invalid" >&2
        exit 2
    fi
done

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cuda_root_path=${CUDA_ROOT_PATH:-/usr/local/cuda}
nvcc_bin=${NVCC_BIN:-"$cuda_root_path/bin/nvcc"}
cuobjdump_bin=${CUOBJDUMP_BIN:-"$cuda_root_path/bin/cuobjdump"}
host_cxx_bin=${HOST_CXX_BIN:-/usr/bin/g++}

if [[ -n ${V4_TC2_WARP0_PROBE_OUTPUT_DIR:-} ]]; then
    build_dir=$V4_TC2_WARP0_PROBE_OUTPUT_DIR
    [[ ! -e $build_dir && ! -L $build_dir ]] || { echo "refusing existing V4_TC2_WARP0_PROBE_OUTPUT_DIR: $build_dir" >&2; exit 2; }
    mkdir -m 0700 -- "$build_dir"
    persist_probe=1
else
    build_dir=$(mktemp -d /tmp/atlas-v4-tc2-warp0-probe.XXXXXX)
    trap 'rm -rf -- "$build_dir"' EXIT
    persist_probe=0
fi
for tool in "$nvcc_bin" "$cuobjdump_bin" "$host_cxx_bin"; do
    [[ -x $tool ]] || { echo "missing build tool: $tool" >&2; exit 2; }
done
for injected_flags in NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS; do
    [[ -z ${!injected_flags:-} ]] || { echo "refusing unreceipted nvcc flags from $injected_flags" >&2; exit 2; }
done

# BEGIN TC2 warp0 probe immutable inputs
probe_file="$repo_root/kernels/gb10/experiments/v4_prefill_attn_tc2_warp0_probe.cu"
candidate_file="$repo_root/kernels/gb10/experiments/v4_prefill_attn_compressed_tc2_warp0.cu"
production_file="$repo_root/kernels/gb10/deepseek-v4-flash/nvfp4/v4_prefill_attn_compressed_tc2_warp0.cu"
incumbent_file="$repo_root/kernels/gb10/deepseek-v4-flash/nvfp4/prefill_attn_compressed.cu"
host_flow_file="$repo_root/crates/spark-model/src/layers/qwen3_attention/prefill/cache_skip_v4.rs"
types_file="$repo_root/crates/spark-model/src/layers/qwen3_attention/types.rs"
init_file="$repo_root/crates/spark-model/src/layers/qwen3_attention/init.rs"
build_script="$repo_root/scripts/check-v4-prefill-attn-tc2-warp0-probe-build.sh"
dependencies=("$probe_file" "$candidate_file" "$production_file" "$incumbent_file" \
    "$host_flow_file" "$types_file" "$init_file" "$build_script")
declare -A dependency_hashes
for dependency in "${dependencies[@]}"; do
    [[ -f $dependency ]] || { echo "missing TC2 warp-0 probe input: $dependency" >&2; exit 2; }
    dependency_hashes[$dependency]=$(sha256sum "$dependency" | awk '{print $1}')
done
nvcc_real_path=$(realpath "$nvcc_bin")
cuobjdump_real_path=$(realpath "$cuobjdump_bin")
host_cxx_real_path=$(realpath "$host_cxx_bin")
nvcc_binary_hash=$(sha256sum "$nvcc_bin" | awk '{print $1}')
cuobjdump_binary_hash=$(sha256sum "$cuobjdump_bin" | awk '{print $1}')
host_cxx_binary_hash=$(sha256sum "$host_cxx_bin" | awk '{print $1}')
git_commit=$(git -C "$repo_root" rev-parse HEAD)
status_paths=(
    "kernels/gb10/experiments/v4_prefill_attn_tc2_warp0_probe.cu"
    "kernels/gb10/experiments/v4_prefill_attn_compressed_tc2_warp0.cu"
    "kernels/gb10/deepseek-v4-flash/nvfp4/v4_prefill_attn_compressed_tc2_warp0.cu"
    "kernels/gb10/deepseek-v4-flash/nvfp4/prefill_attn_compressed.cu"
    "crates/spark-model/src/layers/qwen3_attention/prefill/cache_skip_v4.rs"
    "crates/spark-model/src/layers/qwen3_attention/types.rs"
    "crates/spark-model/src/layers/qwen3_attention/init.rs"
    "scripts/check-v4-prefill-attn-tc2-warp0-probe-build.sh"
)
git_status_hash=$(git -C "$repo_root" status --porcelain=v1 -- "${status_paths[@]}" |
    sha256sum | awk '{print $1}')
# END TC2 warp0 probe immutable inputs

# BEGIN TC2 warp0 probe receipt contract
manifest="$build_dir/build-manifest.txt"
{
    echo receipt_format=atlas-v4-tc2-warp0-probe-v1
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
    echo "compile_command_template=<nvcc> -ccbin <host_cxx> -std=c++17 -O3 --fmad=false -arch=sm_121a -DV4_TC2_WARP0_PROBE_BUILD_ID=<build_id> -DV4_TC2_WARP0_PROBE_MIN_SPEEDUP=<binary64> -DV4_TC2_WARP0_PROBE_MIN_SPEEDUP_TEXT=<canonical> <probe> -o <binary>"
} >"$manifest"
build_id=$(sha256sum "$manifest" | awk '{print $1}')
binary="$build_dir/v4-prefill-attn-tc2-warp0-probe"
compile_command="$nvcc_bin -ccbin $host_cxx_bin -std=c++17 -O3 --fmad=false -arch=sm_121a '-DV4_TC2_WARP0_PROBE_BUILD_ID=\"$build_id\"' '-DV4_TC2_WARP0_PROBE_MIN_SPEEDUP=$min_speedup' '-DV4_TC2_WARP0_PROBE_MIN_SPEEDUP_TEXT=\"$min_speedup\"' $probe_file -o $binary"
compile_command_hash=$(printf %s "$compile_command" | sha256sum | awk '{print $1}')
"$nvcc_bin" -ccbin "$host_cxx_bin" -std=c++17 -O3 --fmad=false -arch=sm_121a \
    "-DV4_TC2_WARP0_PROBE_BUILD_ID=\"$build_id\"" \
    "-DV4_TC2_WARP0_PROBE_MIN_SPEEDUP=$min_speedup" \
    "-DV4_TC2_WARP0_PROBE_MIN_SPEEDUP_TEXT=\"$min_speedup\"" \
    "$probe_file" -o "$binary"
binary_hash=$(sha256sum "$binary" | awk '{print $1}')

resources="$build_dir/resources.txt"
resource_command="$cuobjdump_bin --dump-resource-usage $binary"
resource_command_hash=$(printf %s "$resource_command" | sha256sum | awk '{print $1}')
"$cuobjdump_bin" --dump-resource-usage "$binary" >"$resources"
for symbol in v4_probe_incumbent_tc2 v4_prefill_attn_compressed_tc2_warp0; do
    grep -q "Function $symbol:" "$resources"
done
cubin_dir="$build_dir/cubins"
mkdir "$cubin_dir"
extract_command="$cuobjdump_bin --extract-elf all $binary"
extract_command_hash=$(printf %s "$extract_command" | sha256sum | awk '{print $1}')
(cd "$cubin_dir" && "$cuobjdump_bin" --extract-elf all "$binary" >/dev/null)
mapfile -t cubins < <(find "$cubin_dir" -maxdepth 1 -type f -name '*.cubin' | sort)
[[ ${#cubins[@]} -gt 0 ]] || { echo "no cubin extracted" >&2; exit 1; }

for dependency in "${dependencies[@]}"; do
    [[ $(sha256sum "$dependency" | awk '{print $1}') == "${dependency_hashes[$dependency]}" ]] || { echo "build input changed during TC2 warp-0 probe compilation: $dependency" >&2; exit 1; }
done
for record in "$nvcc_bin|$nvcc_real_path|$nvcc_binary_hash" "$cuobjdump_bin|$cuobjdump_real_path|$cuobjdump_binary_hash" "$host_cxx_bin|$host_cxx_real_path|$host_cxx_binary_hash"; do
    IFS='|' read -r tool path hash <<<"$record"
    [[ $(realpath "$tool") == "$path" && $(sha256sum "$tool" | awk '{print $1}') == "$hash" ]] || { echo "build tool changed during TC2 warp-0 probe compilation: $tool" >&2; exit 1; }
done
[[ $(git -C "$repo_root" rev-parse HEAD) == "$git_commit" &&
    $(git -C "$repo_root" status --porcelain=v1 -- "${status_paths[@]}" |
        sha256sum | awk '{print $1}') == "$git_status_hash" ]] || {
    echo "repository identity changed during TC2 warp-0 probe compilation" >&2
    exit 1
}

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
    echo "binary_sha256 probe $binary_hash"
    for cubin in "${cubins[@]}"; do
        echo "cubin_sha256 cubins/$(basename "$cubin") $(sha256sum "$cubin" | awk '{print $1}')"
    done
} >"$receipt"
receipt_hash=$(sha256sum "$receipt" | awk '{print $1}')
# END TC2 warp0 probe receipt contract

# BEGIN TC2 warp0 probe runner verification
runner="$build_dir/run-v4-tc2-warp0-probe.sh"
{
    echo '#!/usr/bin/env bash'
    echo '# SPDX-License-Identifier: AGPL-3.0-only'
    echo 'set -euo pipefail'
    printf 'expected_receipt_sha256=%q\nexpected_binary_sha256=%q\nexpected_build_id=%q\nexpected_min_speedup=%q\n' \
        "$receipt_hash" "$binary_hash" "$build_id" "$min_speedup"
    cat <<'RUNNER'
[[ $# -eq 0 ]] || { echo "usage: $0" >&2; exit 2; }
dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
receipt="$dir/build-receipt.txt"
binary="$dir/v4-prefill-attn-tc2-warp0-probe"
actual_receipt_sha256=$(sha256sum "$receipt" | awk '{print $1}')
actual_binary_sha256=$(sha256sum "$binary" | awk '{print $1}')
[[ $actual_receipt_sha256 == "$expected_receipt_sha256" && $actual_binary_sha256 == "$expected_binary_sha256" ]] || { echo "TC2 warp-0 probe artifact identity mismatch" >&2; exit 2; }
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
"$binary" >"$output" 2>"$errors"
rc=$?
set -e
[[ $rc -eq 0 && ! -s $errors && $(wc -c <"$output") -le 4096 && $(wc -l <"$output") -eq 11 ]] || { cat "$errors" >&2; exit 1; }
mapfile -t line <"$output"
[[ ${line[0]} == "build_id=$expected_build_id" ]]
[[ ${line[1]} =~ ^device_uuid=[0-9a-f]{32}\ driver=[0-9]+\ runtime=[0-9]+$ ]]
[[ ${line[2]} =~ ^input_hash=[0-9a-f]{16}$ ]]
[[ ${line[3]} == 'parity_cases=9 poison_cases=2 malformed_cases=36' ]]
[[ ${line[4]} == 'csa_timing_tokens=2410 heads=64 head_dim=512 n_comp=602 ratio=4 window=128' ]]
[[ ${line[5]} == 'dense_timing_tokens=2410 heads=64 head_dim=512 n_comp=0 ratio=1 window=128 kv_alias=K=V=Kc=Vc' ]]
[[ ${line[6]} == "threshold min_speedup=$expected_min_speedup" ]]
[[ ${line[7]} =~ ^csa_baseline_ms=[0-9]+\.[0-9]{6}\ csa_candidate_ms=[0-9]+\.[0-9]{6}\ csa_speedup=[0-9]+\.[0-9]{9}\ abba_samples=6$ ]]
[[ ${line[8]} =~ ^dense_baseline_ms=[0-9]+\.[0-9]{6}\ dense_candidate_ms=[0-9]+\.[0-9]{6}\ dense_speedup=[0-9]+\.[0-9]{9}\ abba_samples=6$ ]]
[[ ${line[9]} == 'output_mismatches=0 inputs=immutable input_guards=clean output_guards=clean poison_a=clean poison_b=clean' ]]
[[ ${line[10]} == 'result=PASS' ]]
cat "$output"
RUNNER
} >"$runner"
chmod 0555 "$runner"
runner_sha256=$(sha256sum "$runner" | awk '{print $1}')
# END TC2 warp0 probe runner verification

if [[ $persist_probe == 1 ]]; then
    echo "retained_receipt=$receipt"
    echo "retained_runner=$runner"
fi
echo "runner_sha256=$runner_sha256"
echo "TC2 warp-0 admission probe build: PASS"
