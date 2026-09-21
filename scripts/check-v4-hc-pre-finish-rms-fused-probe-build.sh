#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only

set -euo pipefail
export LC_ALL=C

# Threshold admission precedes CUDA/tool discovery and output-directory work.
if [[ $# -ne 1 ]]; then
    echo "usage: $0 <min_speedup>" >&2
    exit 2
fi
raw_min_speedup=$1
number_pattern='^([0-9]+([.][0-9]*)?|[.][0-9]+)([eE][+-]?[0-9]+)?$'
if [[ ! $raw_min_speedup =~ $number_pattern ]]; then
    echo "invalid explicit numeric threshold" >&2
    exit 2
fi
python_bin=/usr/bin/python3
if [[ ! -x $python_bin ]]; then
    echo "missing binary64 threshold canonicalizer: $python_bin" >&2
    exit 2
fi

canonicalize_binary64_threshold() {
    "$python_bin" - "$1" <<'PYTHON'
import ctypes
import math
import sys

libc = ctypes.CDLL(None)
libc.strtod.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_char_p)]
libc.strtod.restype = ctypes.c_double
raw = sys.argv[1].encode("ascii")
end = ctypes.c_char_p()
value = libc.strtod(raw, ctypes.byref(end))
if end.value != b"" or not math.isfinite(value) or not (1.0 < value <= 100.0):
    raise SystemExit(1)
print(format(value, '.17g'))
print(value.hex(), end="")
PYTHON
}
if ! canonical_threshold=$(canonicalize_binary64_threshold "$raw_min_speedup"); then
    echo "invalid explicit numeric threshold" >&2
    exit 2
fi
min_speedup_text=${canonical_threshold%%$'\n'*}
min_speedup_hex=${canonical_threshold#*$'\n'}
if [[ -z $min_speedup_text || -z $min_speedup_hex ||
    $min_speedup_text == "$canonical_threshold" ]]; then
    echo "invalid explicit numeric threshold" >&2
    exit 2
fi

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cuda_root_path=${CUDA_ROOT_PATH:-/usr/local/cuda}
nvcc_bin=${NVCC_BIN:-"$cuda_root_path/bin/nvcc"}
cuobjdump_bin=${CUOBJDUMP_BIN:-"$cuda_root_path/bin/cuobjdump"}
host_cxx_bin=${HOST_CXX_BIN:-/usr/bin/g++}

if [[ -n ${V4_HC_PROBE_OUTPUT_DIR:-} ]]; then
    build_dir=$V4_HC_PROBE_OUTPUT_DIR
    if [[ -e $build_dir || -L $build_dir ]]; then
        echo "refusing existing V4_HC_PROBE_OUTPUT_DIR: $build_dir" >&2
        exit 2
    fi
    mkdir -m 0700 -- "$build_dir"
    persist_probe=1
else
    build_dir=$(mktemp -d /tmp/atlas-v4-hc-fused-probe.XXXXXX)
    cleanup() {
        local exit_code=$?
        rm -rf -- "$build_dir"
        trap - EXIT
        exit "$exit_code"
    }
    trap cleanup EXIT
    persist_probe=0
fi
for tool in "$nvcc_bin" "$cuobjdump_bin" "$host_cxx_bin" "$python_bin"; do
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

# BEGIN HC probe immutable inputs
probe_file="$repo_root/kernels/gb10/experiments/v4_hc_pre_finish_rms_fused_probe.cu"
candidate_file="$repo_root/kernels/gb10/experiments/v4_hc_pre_finish_rms_fused.cu"
production_wrapper="$repo_root/kernels/gb10/deepseek-v4-flash/nvfp4/v4_hc_pre_finish_rms_fused.cu"
hyper_file="$repo_root/kernels/gb10/deepseek-v4-flash/nvfp4/hyper_connection.cu"
rms_file="$repo_root/kernels/gb10/common/rms_norm_vanilla.cu"
build_script="$repo_root/scripts/check-v4-hc-pre-finish-rms-fused-probe-build.sh"
dependencies=("$probe_file" "$candidate_file" "$production_wrapper" "$hyper_file" "$rms_file" "$build_script")
declare -A dependency_hashes
for dependency in "${dependencies[@]}"; do
    if [[ ! -f $dependency ]]; then
        echo "missing HC probe build input: $dependency" >&2
        exit 2
    fi
    dependency_hashes[$dependency]=$(sha256sum "$dependency" | awk '{print $1}')
done
nvcc_real_path=$(realpath "$nvcc_bin")
cuobjdump_real_path=$(realpath "$cuobjdump_bin")
host_cxx_real_path=$(realpath "$host_cxx_bin")
python_real_path=$(realpath "$python_bin")
nvcc_version=$("$nvcc_bin" --version | tail -n 1)
cuobjdump_version=$("$cuobjdump_bin" --version | tail -n 1)
host_cxx_version=$("$host_cxx_bin" --version | head -n 1)
python_version=$("$python_bin" --version 2>&1)
nvcc_binary_hash=$(sha256sum "$nvcc_bin" | awk '{print $1}')
cuobjdump_binary_hash=$(sha256sum "$cuobjdump_bin" | awk '{print $1}')
host_cxx_binary_hash=$(sha256sum "$host_cxx_bin" | awk '{print $1}')
python_binary_hash=$(sha256sum "$python_bin" | awk '{print $1}')
git_commit=$(git -C "$repo_root" rev-parse HEAD)
status_paths=(
    "kernels/gb10/experiments/v4_hc_pre_finish_rms_fused_probe.cu"
    "kernels/gb10/experiments/v4_hc_pre_finish_rms_fused.cu"
    "kernels/gb10/deepseek-v4-flash/nvfp4/v4_hc_pre_finish_rms_fused.cu"
    "kernels/gb10/deepseek-v4-flash/nvfp4/hyper_connection.cu"
    "kernels/gb10/common/rms_norm_vanilla.cu"
    "scripts/check-v4-hc-pre-finish-rms-fused-probe-build.sh"
)
git_status_hash=$(git -C "$repo_root" status --porcelain=v1 -- "${status_paths[@]}" |
    sha256sum | awk '{print $1}')
# END HC probe immutable inputs

# BEGIN HC probe receipt contract
manifest="$build_dir/build-manifest.txt"
{
    echo "receipt_format=atlas-v4-hc-fused-probe-v3"
    echo "git_commit=$git_commit"
    echo "git_status_sha256=$git_status_hash"
    echo "min_speedup_text=$min_speedup_text"
    echo "min_speedup_hex=$min_speedup_hex"
    echo "baseline_kernel=rms_norm_vanilla"
    echo "normalization_formula=x*rms*weight"
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
    echo "python_path=$python_real_path"
    echo "python_version=$python_version"
    echo "python_binary_sha256=$python_binary_hash"
    echo "compile_command_template=<nvcc> -ccbin <host_cxx> -std=c++17 -O3 --fmad=false -arch=sm_121a '-DV4_HC_PROBE_BUILD_ID=\"<build_id>\"' '-DV4_HC_PROBE_MIN_SPEEDUP_TEXT=\"<min_speedup_text>\"' -DV4_HC_PROBE_MIN_SPEEDUP_HEX=<min_speedup_hex> <probe> -o <binary>"
    echo "resource_command_template=<cuobjdump> --dump-resource-usage <binary>"
    echo "extract_command_template=<cuobjdump> --extract-elf all <binary>"
} >"$manifest"
build_id=$(sha256sum "$manifest" | awk '{print $1}')

binary="$build_dir/v4-hc-pre-finish-rms-fused-probe"
compile_command="$nvcc_bin -ccbin $host_cxx_bin -std=c++17 -O3 --fmad=false -arch=sm_121a '-DV4_HC_PROBE_BUILD_ID=\"$build_id\"' '-DV4_HC_PROBE_MIN_SPEEDUP_TEXT=\"$min_speedup_text\"' -DV4_HC_PROBE_MIN_SPEEDUP_HEX=$min_speedup_hex $probe_file -o $binary"
compile_command_hash=$(printf %s "$compile_command" | sha256sum | awk '{print $1}')
"$nvcc_bin" -ccbin "$host_cxx_bin" -std=c++17 -O3 --fmad=false -arch=sm_121a \
    "-DV4_HC_PROBE_BUILD_ID=\"$build_id\"" \
    "-DV4_HC_PROBE_MIN_SPEEDUP_TEXT=\"$min_speedup_text\"" \
    "-DV4_HC_PROBE_MIN_SPEEDUP_HEX=$min_speedup_hex" \
    "$probe_file" -o "$binary"
binary_hash=$(sha256sum "$binary" | awk '{print $1}')

# The probe is zero-argument. Exercise only its pre-CUDA argument rejection.
binary_rejection="$build_dir/reject-valid-binary64-extra-arg.out"
set +e
"$binary" 1.00000001 >"$binary_rejection" 2>&1
binary_rejection_rc=$?
set -e
if [[ $binary_rejection_rc -ne 2 ]] ||
    ! grep -qx 'usage: v4-hc-pre-finish-rms-fused-probe' "$binary_rejection"; then
    echo "HC probe accepted a runtime threshold argument" >&2
    exit 1
fi

resources="$build_dir/resources.txt"
resource_command="$cuobjdump_bin --dump-resource-usage $binary"
resource_command_hash=$(printf %s "$resource_command" | sha256sum | awk '{print $1}')
"$cuobjdump_bin" --dump-resource-usage "$binary" >"$resources"
for symbol in hc_pre_finish rms_norm_vanilla v4_hc_pre_finish_rms_fused; do
    grep -q "Function $symbol:" "$resources"
done
resource_hash=$(sha256sum "$resources" | awk '{print $1}')
cubin_dir="$build_dir/cubins"
mkdir "$cubin_dir"
extract_command="$cuobjdump_bin --extract-elf all $binary"
extract_command_hash=$(printf %s "$extract_command" | sha256sum | awk '{print $1}')
(cd "$cubin_dir" && "$cuobjdump_bin" --extract-elf all "$binary" >/dev/null)
mapfile -t cubins < <(find "$cubin_dir" -maxdepth 1 -type f -name '*.cubin' | sort)
if [[ ${#cubins[@]} -lt 1 ]]; then
    echo "no HC probe cubin extracted" >&2
    exit 1
fi

for dependency in "${dependencies[@]}"; do
    if [[ $(sha256sum "$dependency" | awk '{print $1}') != "${dependency_hashes[$dependency]}" ]]; then
        echo "build input changed during V4 HC probe compilation: $dependency" >&2
        exit 1
    fi
done
for record in \
    "$nvcc_bin|$nvcc_real_path|$nvcc_binary_hash" \
    "$cuobjdump_bin|$cuobjdump_real_path|$cuobjdump_binary_hash" \
    "$host_cxx_bin|$host_cxx_real_path|$host_cxx_binary_hash" \
    "$python_bin|$python_real_path|$python_binary_hash"; do
    IFS='|' read -r tool expected_path expected_hash <<<"$record"
    if [[ $(realpath "$tool") != "$expected_path" ]] ||
        [[ $(sha256sum "$tool" | awk '{print $1}') != "$expected_hash" ]]; then
        echo "build tool changed during V4 HC probe compilation: $tool" >&2
        exit 1
    fi
done
if [[ $(git -C "$repo_root" rev-parse HEAD) != "$git_commit" ]] ||
    [[ $(git -C "$repo_root" status --porcelain=v1 -- "${status_paths[@]}" |
        sha256sum | awk '{print $1}') != "$git_status_hash" ]]; then
    echo "repository identity changed during V4 HC probe compilation" >&2
    exit 1
fi

receipt="$build_dir/build-receipt.txt"
{
    cat "$manifest"
    echo "build_id=$build_id"
    echo "compile_command=$compile_command"
    echo "compile_command_sha256=$compile_command_hash"
    echo "resource_command=$resource_command"
    echo "resource_command_sha256=$resource_command_hash"
    echo "resource_sha256 resources.txt $resource_hash"
    echo "extract_command=$extract_command"
    echo "extract_command_sha256=$extract_command_hash"
    echo "binary_sha256 probe $binary_hash"
    for cubin in "${cubins[@]}"; do
        echo "cubin_sha256 cubins/$(basename "$cubin") $(sha256sum "$cubin" | awk '{print $1}')"
    done
} >"$receipt"
receipt_hash=$(sha256sum "$receipt" | awk '{print $1}')
# END HC probe receipt contract

# BEGIN HC probe runner verification
runner="$build_dir/run-v4-hc-fused-probe.sh"
{
    echo '#!/usr/bin/env bash'
    echo '# SPDX-License-Identifier: AGPL-3.0-only'
    echo 'set -euo pipefail'
    echo 'export LC_ALL=C'
    printf 'expected_receipt_sha256=%q\n' "$receipt_hash"
    printf 'expected_binary_sha256=%q\n' "$binary_hash"
    printf 'expected_build_id=%q\n' "$build_id"
    printf 'expected_min_speedup_text=%q\n' "$min_speedup_text"
    printf 'expected_min_speedup_hex=%q\n' "$min_speedup_hex"
    cat <<'RUNNER'
if [[ $# -ne 0 ]]; then
    echo "usage: $0" >&2
    exit 2
fi
dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
receipt="$dir/build-receipt.txt"
binary="$dir/v4-hc-pre-finish-rms-fused-probe"
actual_receipt_sha256=$(sha256sum "$receipt" | awk '{print $1}')
actual_binary_sha256=$(sha256sum "$binary" | awk '{print $1}')
if [[ $actual_receipt_sha256 != "$expected_receipt_sha256" ||
    $actual_binary_sha256 != "$expected_binary_sha256" ]] ||
    [[ $(grep -Fxc "build_id=$expected_build_id" "$receipt") -ne 1 ]] ||
    [[ $(grep -Fxc "min_speedup_text=$expected_min_speedup_text" "$receipt") -ne 1 ]] ||
    [[ $(grep -Fxc "min_speedup_hex=$expected_min_speedup_hex" "$receipt") -ne 1 ]] ||
    [[ $(grep -Fxc "binary_sha256 probe $expected_binary_sha256" "$receipt") -ne 1 ]]; then
    echo "HC probe artifact identity mismatch" >&2
    exit 2
fi
while read -r _ relative expected; do
    artifact="$dir/$relative"
    if [[ ! -f $artifact ]]; then
        echo "HC probe cubin missing: $relative" >&2
        exit 2
    fi
    actual_cubin_sha256=$(sha256sum "$artifact" | awk '{print $1}')
    if [[ $actual_cubin_sha256 != "$expected" ]]; then
        echo "HC probe cubin identity mismatch: $relative" >&2
        exit 2
    fi
done < <(grep '^cubin_sha256 ' "$receipt")

probe_tmp=$(mktemp -d "${TMPDIR:-/tmp}/atlas-v4-hc-run.XXXXXX")
cleanup() {
    local exit_code=$?
    rm -rf -- "$probe_tmp"
    trap - EXIT
    exit "$exit_code"
}
trap cleanup EXIT
output="$probe_tmp/stdout"
errors="$probe_tmp/stderr"
set +e
"$binary" >"$output" 2>"$errors"
rc=$?
set -e
max_output_bytes=4096
max_output_lines=8
if [[ $rc -ne 0 || -s $errors || $(wc -c <"$output") -gt $max_output_bytes ||
    $(wc -l <"$output") -ne $max_output_lines ]]; then
    echo "HC probe rejected: rc=$rc" >&2
    exit 1
fi
mapfile -t line <"$output"
number='[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?'
[[ ${line[0]} == "build_id=$expected_build_id" ]]
[[ ${line[1]} =~ ^device_uuid=[0-9a-f]{32}\ driver=[0-9]+\ runtime=[0-9]+$ ]]
[[ ${line[2]} =~ ^input_hash=[0-9a-f]{16}\ mix_hash=[0-9a-f]{16}\ weight_hash=[0-9a-f]{16}$ ]]
[[ ${line[3]} == 'tokens=2410 hidden=4096 hc=4 parity_cases=6 poison_cases=2 geometry_cases=10 hidden_bytes=19742720 normed_bytes=19742720 post_bytes=38560 comb_bytes=154240' ]]
[[ ${line[4]} == "threshold min_speedup=$expected_min_speedup_text" ]]
[[ ${line[5]} =~ ^baseline_ms=$number\ candidate_ms=$number\ speedup=$number\ abba_samples=6$ ]]
[[ ${line[6]} == 'hidden_mismatches=0 normed_mismatches=0 post_mismatches=0 comb_mismatches=0 guards=clean poison_a=clean poison_b=clean' ]]
[[ ${line[7]} == 'result=PASS' ]]
cat "$output"
RUNNER
} >"$runner"
chmod 0555 "$runner"

# reject-valid-binary64-extra-arg: runner admission must precede every GPU call.
runner_rejection="$build_dir/reject-valid-binary64-extra-arg-runner.out"
set +e
"$runner" 1.00000001 >"$runner_rejection" 2>&1
runner_rejection_rc=$?
set -e
if [[ $runner_rejection_rc -ne 2 ]] ||
    ! grep -Eqx 'usage: .+/run-v4-hc-fused-probe.sh' "$runner_rejection"; then
    echo "HC runner accepted a runtime threshold argument" >&2
    exit 1
fi
# END HC probe runner verification

if [[ $persist_probe == 1 ]]; then
    echo "retained_receipt=$receipt"
    echo "retained_runner=$runner"
fi
echo "HC fused promotion probe build: PASS (no GPU execution)"
