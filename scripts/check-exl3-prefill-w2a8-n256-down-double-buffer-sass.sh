#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cuda_root_path=${CUDA_ROOT_PATH:-/usr/local/cuda}
nvcc_bin=${NVCC_BIN:-"$cuda_root_path/bin/nvcc"}
cuobjdump_bin=${CUOBJDUMP_BIN:-"$cuda_root_path/bin/cuobjdump"}
source_file="$repo_root/kernels/gb10/experiments/exl3_w2a8_grouped_prefill_n256.cu"
wrapper_file="$repo_root/kernels/gb10/deepseek-v4-flash/nvfp4/exl3_w2a8_grouped_prefill_n256_k2_down.cu"
probe_dir=$(mktemp -d /tmp/atlas-exl3-w2a8-n256-down-double-buffer.XXXXXX)
trap 'rm -rf -- "$probe_dir"' EXIT
register_limit=104
shared_limit=19456
instruction_limit=920

if [[ -n ${NVCC_PREPEND_FLAGS:-} || -n ${NVCC_APPEND_FLAGS:-} ]]; then
    echo "NVCC_PREPEND_FLAGS and NVCC_APPEND_FLAGS must be empty" >&2
    exit 2
fi
for tool in "$nvcc_bin" "$cuobjdump_bin"; do
    if [[ ! -x $tool ]]; then
        echo "missing CUDA tool: $tool" >&2
        exit 2
    fi
done

for contract in \
    '#define W2A8_N256_DOWN_DOUBLE_BUFFER_CANDIDATE 0' \
    'static_assert(W2A8_N256_DOWN_DOUBLE_BUFFER_CANDIDATE == 0 ||' \
    'W2A8GemmScratch scratch_buffers[2]' \
    'sizeof(W2A8GemmScratch) == 9 * 1024' \
    'cp.async.ca.shared.global' \
    'cp.async.commit_group' \
    'cp.async.wait_group 0' \
    'scratch_buffers[1]' \
    'scratch_buffers[current_buffer]'; do
    grep -Fq "$contract" "$source_file"
done

resource_value() {
    local file=$1 symbol=$2 field=$3
    awk -v symbol="$symbol" -v field="$field" '
        $0 == " Function " symbol ":" { getline; line = $0 }
        END {
            count = split(line, parts, " ")
            for (i = 1; i <= count; i++) {
                split(parts[i], pair, ":")
                if (pair[1] == field) { print pair[2]; exit }
            }
        }
    ' "$file"
}

instruction_count() {
    awk '/^[[:space:]]*\/\*[0-9a-f]+\*\// { count++ } END { print count + 0 }' "$1"
}

sass_count() {
    grep -cE "$2" "$1" || true
}

compile_variant() {
    local name=$1 selector=$2
    local symbol="exl3_w2a8_grouped_prefill_n256_down_double_buffer_${name}"
    local cubin="$probe_dir/${name}.cubin"
    local log="$probe_dir/${name}.log"
    local resources="$probe_dir/${name}.resources"
    local sass="$probe_dir/${name}.sass"

    "$nvcc_bin" -std=c++17 -O3 --fmad=false -arch=sm_121a \
        -DW2A8_FIXED_N=4096 -DW2A8_FIXED_K=2048 \
        -DW2A8_PACKED_E4M3_CANDIDATE=1 \
        -DW2A8_N256_SINGLE_WARP_ROUTE_GUARD_CANDIDATE=0 \
        -DW2A8_N256_DOWN_DOUBLE_BUFFER_CANDIDATE="$selector" \
        -DW2A8_KERNEL_NAME="$symbol" -cubin "$source_file" \
        -o "$cubin" -Xptxas=-v 2>"$log"
    "$cuobjdump_bin" --dump-resource-usage "$cubin" >"$resources"
    "$cuobjdump_bin" --dump-sass --function "$symbol" "$cubin" >"$sass"

    local registers stack local_bytes shared instructions barriers qmma
    local bf16_rounds fp8_f16 fp8_f32 shuffles fadd fmul ffma hadd2 hmul2
    local ldgsts ldgdepbar depbar_waits zfill
    registers=$(resource_value "$resources" "$symbol" REG)
    stack=$(resource_value "$resources" "$symbol" STACK)
    local_bytes=$(resource_value "$resources" "$symbol" LOCAL)
    shared=$(resource_value "$resources" "$symbol" SHARED)
    instructions=$(instruction_count "$sass")
    barriers=$(sass_count "$sass" 'BAR\.SYNC')
    qmma=$(sass_count "$sass" 'QMMA\.16832\.F32\.E4M3\.E4M3')
    bf16_rounds=$(sass_count "$sass" 'F2F(P)?\.BF16\.F32')
    fp8_f16=$(sass_count "$sass" 'F2FP\.SATFINITE\.E4M3\.F16')
    fp8_f32=$(sass_count "$sass" 'F2FP\.SATFINITE\.E4M3\.F32')
    shuffles=$(sass_count "$sass" 'SHFL\.')
    fadd=$(sass_count "$sass" '[[:space:]]FADD')
    fmul=$(sass_count "$sass" '[[:space:]]FMUL')
    ffma=$(sass_count "$sass" '[[:space:]]FFMA')
    hadd2=$(sass_count "$sass" '[[:space:]]HADD2')
    hmul2=$(sass_count "$sass" '[[:space:]]HMUL2')
    ldgsts=$(sass_count "$sass" 'LDGSTS')
    ldgdepbar=$(sass_count "$sass" 'LDGDEPBAR')
    depbar_waits=$(sass_count "$sass" 'DEPBAR\.LE SB0, 0x0')
    zfill=$(sass_count "$sass" 'LDGSTS[^;]*ZFILL')
    for value in \
        "$registers" "$stack" "$local_bytes" "$shared" "$instructions" \
        "$barriers" "$qmma" "$bf16_rounds" "$fp8_f16" "$fp8_f32" \
        "$shuffles" "$fadd" "$fmul" "$ffma" "$hadd2" "$hmul2" \
        "$ldgsts" "$ldgdepbar" "$depbar_waits" "$zfill"; do
        [[ $value =~ ^[0-9]+$ ]]
    done
    [[ $stack == 0 ]]
    [[ $local_bytes == 0 ]]
    grep -A1 "Function properties for $symbol" "$log" \
        | grep -q '0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads'
    [[ $(sass_count "$sass" '[[:space:]](ATOM|RED|LDL|STL)') == 0 ]]
    printf '%s\n' \
        "registers=$registers" \
        "stack=$stack" \
        "local=$local_bytes" \
        "shared=$shared" \
        "instructions=$instructions" \
        "barriers=$barriers" \
        "qmma=$qmma" \
        "bf16_rounds=$bf16_rounds" \
        "fp8_f16=$fp8_f16" \
        "fp8_f32=$fp8_f32" \
        "shuffles=$shuffles" \
        "fadd=$fadd" \
        "fmul=$fmul" \
        "ffma=$ffma" \
        "hadd2=$hadd2" \
        "hmul2=$hmul2" \
        "ldgsts=$ldgsts" \
        "ldgdepbar=$ldgdepbar" \
        "depbar_waits=$depbar_waits" \
        "zfill=$zfill" >"$probe_dir/${name}.metrics"
}

compile_variant incumbent 0
compile_variant candidate 1

metric() {
    local name=$1 field=$2
    awk -F= -v field="$field" '$1 == field { print $2 }' "$probe_dir/${name}.metrics"
}

for field in qmma bf16_rounds fp8_f16 fp8_f32 shuffles fadd fmul ffma hadd2 hmul2; do
    [[ $(metric candidate "$field") == "$(metric incumbent "$field")" ]]
done
[[ $(metric incumbent qmma) == 16 ]]
[[ $(metric incumbent bf16_rounds) == 32 ]]
[[ $(metric incumbent fp8_f16) == 16 ]]
[[ $(metric incumbent fp8_f32) == 0 ]]
[[ $(metric incumbent shuffles) == 16 ]]
[[ $(metric incumbent ldgsts) == 0 ]]
[[ $(metric incumbent ldgdepbar) == 0 ]]
[[ $(metric incumbent depbar_waits) == 0 ]]
[[ $(metric incumbent zfill) == 0 ]]
[[ $(metric candidate ldgsts) == 2 ]]
[[ $(metric candidate ldgdepbar) == 2 ]]
[[ $(metric candidate depbar_waits) == 2 ]]
[[ $(metric candidate zfill) == 1 ]]

# Each async group must be issued before commit, waited before publication, and
# the second group's wait must follow the current stage's complete QMMA body.
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
' "$probe_dir/candidate.sass"

incumbent_registers=$(metric incumbent registers)
candidate_registers=$(metric candidate registers)
incumbent_shared=$(metric incumbent shared)
candidate_shared=$(metric candidate shared)
incumbent_instructions=$(metric incumbent instructions)
candidate_instructions=$(metric candidate instructions)
incumbent_barriers=$(metric incumbent barriers)
candidate_barriers=$(metric candidate barriers)
[[ $incumbent_registers == 97 ]]
[[ $incumbent_shared == 10240 ]]
[[ $incumbent_instructions == 856 ]]
[[ $incumbent_barriers == 3 ]]
(( candidate_registers <= register_limit ))
(( candidate_shared <= shared_limit ))
(( candidate_instructions <= instruction_limit ))
[[ $candidate_barriers == 3 ]]

# The production wrapper owns both promoted selectors. Prove hostile
# command-line zeroes cannot silently demote either released arm.
production_symbol=exl3_w2a8_grouped_prefill_n256_k2_down
production_cubin="$probe_dir/production-hostile-overrides.cubin"
production_log="$probe_dir/production-hostile-overrides.log"
production_resources="$probe_dir/production-hostile-overrides.resources"
production_sass="$probe_dir/production-hostile-overrides.sass"
"$nvcc_bin" -std=c++17 -O3 --fmad=false -arch=sm_121a \
    -DW2A8_PACKED_E4M3_CANDIDATE=0 \
    -DW2A8_N256_SINGLE_WARP_ROUTE_GUARD_CANDIDATE=0 \
    -DW2A8_N256_DOWN_DOUBLE_BUFFER_CANDIDATE=0 \
    -cubin "$wrapper_file" -o "$production_cubin" -Xptxas=-v 2>"$production_log"
"$cuobjdump_bin" --dump-resource-usage "$production_cubin" >"$production_resources"
"$cuobjdump_bin" --dump-sass --function "$production_symbol" \
    "$production_cubin" >"$production_sass"
[[ $(resource_value "$production_resources" "$production_symbol" REG) == 102 ]]
[[ $(resource_value "$production_resources" "$production_symbol" SHARED) == 19456 ]]
[[ $(instruction_count "$production_sass") == 896 ]]
[[ $(sass_count "$production_sass" 'LDGSTS') == 2 ]]
[[ $(sass_count "$production_sass" 'LDGDEPBAR') == 2 ]]
[[ $(sass_count "$production_sass" 'DEPBAR\.LE SB0, 0x0') == 2 ]]
[[ $(sass_count "$production_sass" 'LDGSTS[^;]*ZFILL') == 1 ]]
[[ $(sass_count "$production_sass" 'F2FP\.SATFINITE\.E4M3\.F16') == 16 ]]
[[ $(sass_count "$production_sass" 'F2FP\.SATFINITE\.E4M3\.F32') == 0 ]]
grep -A1 "Function properties for $production_symbol" "$production_log" \
    | grep -q '0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads'

if "$nvcc_bin" -std=c++17 -O3 --fmad=false -arch=sm_121a \
    -DW2A8_FIXED_N=4096 -DW2A8_FIXED_K=2048 \
    -DW2A8_PACKED_E4M3_CANDIDATE=1 \
    -DW2A8_N256_SINGLE_WARP_ROUTE_GUARD_CANDIDATE=0 \
    -DW2A8_N256_DOWN_DOUBLE_BUFFER_CANDIDATE=2 \
    -DW2A8_KERNEL_NAME=exl3_w2a8_n256_invalid_double_buffer \
    -cubin "$source_file" -o "$probe_dir/invalid-selector.cubin" >/dev/null 2>&1; then
    echo "non-boolean N256 down double-buffer selector unexpectedly compiled" >&2
    exit 1
fi

if "$nvcc_bin" -std=c++17 -O3 --fmad=false -arch=sm_121a \
    -DW2A8_FIXED_N=2048 -DW2A8_FIXED_K=4096 \
    -DW2A8_PACKED_E4M3_CANDIDATE=1 \
    -DW2A8_N256_SINGLE_WARP_ROUTE_GUARD_CANDIDATE=0 \
    -DW2A8_N256_DOWN_DOUBLE_BUFFER_CANDIDATE=1 \
    -DW2A8_KERNEL_NAME=exl3_w2a8_n256_invalid_gu_double_buffer \
    -cubin "$source_file" -o "$probe_dir/invalid-gu.cubin" >/dev/null 2>&1; then
    echo "N256 down double buffering unexpectedly accepted the GU shape" >&2
    exit 1
fi

digest=$(
    sha256sum "$probe_dir"/{incumbent,candidate}.{cubin,resources,sass} \
        | awk '{print $1}' | sha256sum | awk '{print $1}'
)
echo "N256 down K64 double buffer: packed E4M3=1 route_guard=0 one selector arithmetic census exact"
echo "incumbent: instructions=$incumbent_instructions registers=$incumbent_registers shared=$incumbent_shared barriers=$incumbent_barriers ldgsts=$(metric incumbent ldgsts)"
echo "candidate: instructions=$candidate_instructions registers=$candidate_registers shared=$candidate_shared barriers=$candidate_barriers ldgsts=$(metric candidate ldgsts) commits=$(metric candidate ldgdepbar) waits=$(metric candidate depbar_waits) zfill=$(metric candidate zfill)"
echo "production hostile selector zeroes: packed=1 n256_down_double_buffer=1"
echo "ceilings: instructions<=$instruction_limit registers<=$register_limit shared<=$shared_limit stack=0 local=0 spills=0 atomics=0 cubin_set_sha256=$digest"
