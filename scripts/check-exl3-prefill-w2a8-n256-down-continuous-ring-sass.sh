#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cuda_root_path=${CUDA_ROOT_PATH:-/usr/local/cuda}
nvcc_bin=${NVCC_BIN:-"$cuda_root_path/bin/nvcc"}
cuobjdump_bin=${CUOBJDUMP_BIN:-"$cuda_root_path/bin/cuobjdump"}
source_file="$repo_root/kernels/gb10/experiments/exl3_w2a8_grouped_prefill_n256.cu"
wrapper_file="$repo_root/kernels/gb10/deepseek-v4-flash/nvfp4/exl3_w2a8_grouped_prefill_n256_k2_down.cu"
wrapper_sha256_expected=6147daf2b8fca4864f8c58bd477e01979a7479ac729b212aa6f1218ff5409023
probe_dir=$(mktemp -d /tmp/atlas-exl3-w2a8-n256-down-continuous-ring.XXXXXX)
trap 'rm -rf -- "$probe_dir"' EXIT
register_limit=104
shared_limit=19456
instruction_limit=936
production_instruction_expected=920

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
[[ $(sha256sum "$wrapper_file" | awk '{print $1}') == "$wrapper_sha256_expected" ]]
for contract in \
    '#ifdef W2A8_N256_DOWN_CONTINUOUS_RING_CANDIDATE' \
    '#undef W2A8_N256_DOWN_CONTINUOUS_RING_CANDIDATE' \
    '#define W2A8_N256_DOWN_CONTINUOUS_RING_CANDIDATE 1'; do
    grep -Fq "$contract" "$wrapper_file"
done

for contract in \
    '#define W2A8_N256_DOWN_DOUBLE_BUFFER_CANDIDATE 0' \
    'static_assert(W2A8_N256_DOWN_DOUBLE_BUFFER_CANDIDATE == 0 ||' \
    'W2A8GemmScratch scratch_buffers[2]' \
    'sizeof(W2A8GemmScratch) == 9 * 1024' \
    'cp.async.ca.shared.global' \
    'cp.async.commit_group' \
    'cp.async.wait_group 0' \
    '#define W2A8_N256_DOWN_CONTINUOUS_RING_CANDIDATE 0' \
    'static_assert(W2A8_N256_DOWN_CONTINUOUS_RING_CANDIDATE == 0 ||' \
    'N256 continuous ring requires the two staging buffers' \
    'N256 down continuous K64 ring initial publication' \
    'absolute_stage = absolute_k / W2A8_K_STAGE' \
    'next_k < W2A8_FIXED_K' \
    'scratch_buffers[next_buffer]' \
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
    local symbol="exl3_w2a8_grouped_prefill_n256_down_continuous_ring_${name}"
    local cubin="$probe_dir/${name}.cubin"
    local log="$probe_dir/${name}.log"
    local resources="$probe_dir/${name}.resources"
    local sass="$probe_dir/${name}.sass"

    "$nvcc_bin" -std=c++17 -O3 --fmad=false -arch=sm_121a \
        -DW2A8_FIXED_N=4096 -DW2A8_FIXED_K=2048 \
        -DW2A8_PACKED_E4M3_CANDIDATE=1 \
        -DW2A8_N256_SINGLE_WARP_ROUTE_GUARD_CANDIDATE=0 \
        -DW2A8_N256_DOWN_DOUBLE_BUFFER_CANDIDATE=1 \
        -DW2A8_N256_DOWN_CONTINUOUS_RING_CANDIDATE="$selector" \
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
[[ $(metric incumbent fadd) == 32 ]]
[[ $(metric incumbent fmul) == 40 ]]
[[ $(metric incumbent ffma) == 0 ]]
[[ $(metric incumbent hadd2) == 16 ]]
[[ $(metric incumbent hmul2) == 16 ]]
[[ $(metric incumbent ldgsts) == 2 ]]
[[ $(metric incumbent ldgdepbar) == 2 ]]
[[ $(metric incumbent depbar_waits) == 2 ]]
[[ $(metric incumbent zfill) == 1 ]]
[[ $(metric candidate ldgsts) == 2 ]]
[[ $(metric candidate ldgdepbar) == 2 ]]
[[ $(metric candidate depbar_waits) == 2 ]]
[[ $(metric candidate zfill) == 2 ]]

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
[[ $incumbent_registers == 102 ]]
[[ $incumbent_shared == 19456 ]]
[[ $incumbent_instructions == 896 ]]
[[ $incumbent_barriers == 3 ]]
(( candidate_registers <= register_limit ))
[[ $candidate_registers == 103 ]]
[[ $candidate_shared == "$shared_limit" ]]
(( candidate_instructions <= instruction_limit ))
[[ $candidate_instructions == 920 ]]
[[ $candidate_barriers == 3 ]]

# The production wrapper now forces all three promoted selectors. Compare a
# hostile ring0 production compile with a context-identical raw ring1 reference
# so the proof is about generated resources and SASS, not only source text.
production_symbol=exl3_w2a8_grouped_prefill_n256_k2_down
forced_candidate_cubin="$probe_dir/forced-candidate-ring1.cubin"
forced_candidate_log="$probe_dir/forced-candidate-ring1.log"
forced_candidate_resources="$probe_dir/forced-candidate-ring1.resources"
forced_candidate_sass="$probe_dir/forced-candidate-ring1.sass"
production_cubin="$probe_dir/production-hostile-ring0.cubin"
production_log="$probe_dir/production-hostile-ring0.log"
production_resources="$probe_dir/production-hostile-ring0.resources"
production_sass="$probe_dir/production-hostile-ring0.sass"

"$nvcc_bin" -std=c++17 -O3 --fmad=false -arch=sm_121a \
    -DW2A8_FIXED_N=4096 -DW2A8_FIXED_K=2048 \
    -DW2A8_PACKED_E4M3_CANDIDATE=1 \
    -DW2A8_N256_SINGLE_WARP_ROUTE_GUARD_CANDIDATE=0 \
    -DW2A8_N256_DOWN_DOUBLE_BUFFER_CANDIDATE=1 \
    -DW2A8_N256_DOWN_CONTINUOUS_RING_CANDIDATE=1 \
    -DW2A8_KERNEL_NAME="$production_symbol" -cubin "$source_file" \
    -o "$forced_candidate_cubin" -Xptxas=-v 2>"$forced_candidate_log"
"$cuobjdump_bin" --dump-resource-usage "$forced_candidate_cubin" \
    >"$forced_candidate_resources"
"$cuobjdump_bin" --dump-sass --function "$production_symbol" \
    "$forced_candidate_cubin" >"$forced_candidate_sass"

"$nvcc_bin" -std=c++17 -O3 --fmad=false -arch=sm_121a \
    -DW2A8_PACKED_E4M3_CANDIDATE=0 \
    -DW2A8_N256_SINGLE_WARP_ROUTE_GUARD_CANDIDATE=0 \
    -DW2A8_N256_DOWN_DOUBLE_BUFFER_CANDIDATE=0 \
    -DW2A8_N256_DOWN_CONTINUOUS_RING_CANDIDATE=0 \
    -cubin "$wrapper_file" -o "$production_cubin" -Xptxas=-v \
    2>"$production_log"
"$cuobjdump_bin" --dump-resource-usage "$production_cubin" >"$production_resources"
"$cuobjdump_bin" --dump-sass --function "$production_symbol" \
    "$production_cubin" >"$production_sass"

for field in REG STACK SHARED LOCAL; do
    [[ $(resource_value "$production_resources" "$production_symbol" "$field") == \
       "$(resource_value "$forced_candidate_resources" "$production_symbol" "$field")" ]]
done
forced_candidate_resources_sha256=$(sha256sum "$forced_candidate_resources" | awk '{print $1}')
production_resources_sha256=$(sha256sum "$production_resources" | awk '{print $1}')
[[ $production_resources_sha256 == "$forced_candidate_resources_sha256" ]]
[[ $(resource_value "$production_resources" "$production_symbol" REG) == 103 ]]
[[ $(resource_value "$production_resources" "$production_symbol" STACK) == 0 ]]
[[ $(resource_value "$production_resources" "$production_symbol" SHARED) == 19456 ]]
[[ $(resource_value "$production_resources" "$production_symbol" LOCAL) == 0 ]]
[[ $(instruction_count "$production_sass") == "$production_instruction_expected" ]]
[[ $(sass_count "$production_sass" 'BAR\.SYNC') == 3 ]]
[[ $(sass_count "$production_sass" 'QMMA\.16832\.F32\.E4M3\.E4M3') == 16 ]]
[[ $(sass_count "$production_sass" 'F2F(P)?\.BF16\.F32') == 32 ]]
[[ $(sass_count "$production_sass" 'F2FP\.SATFINITE\.E4M3\.F16') == 16 ]]
[[ $(sass_count "$production_sass" 'F2FP\.SATFINITE\.E4M3\.F32') == 0 ]]
[[ $(sass_count "$production_sass" 'SHFL\.') == 16 ]]
[[ $(sass_count "$production_sass" '[[:space:]]FADD') == 32 ]]
[[ $(sass_count "$production_sass" '[[:space:]]FMUL') == 40 ]]
[[ $(sass_count "$production_sass" '[[:space:]]FFMA') == 0 ]]
[[ $(sass_count "$production_sass" '[[:space:]]HADD2') == 16 ]]
[[ $(sass_count "$production_sass" '[[:space:]]HMUL2') == 16 ]]
[[ $(sass_count "$production_sass" 'LDGSTS') == 2 ]]
[[ $(sass_count "$production_sass" 'LDGDEPBAR') == 2 ]]
[[ $(sass_count "$production_sass" 'DEPBAR\.LE SB0, 0x0') == 2 ]]
[[ $(sass_count "$production_sass" 'LDGSTS[^;]*ZFILL') == 2 ]]
[[ $(sass_count "$production_sass" '[[:space:]](ATOM|RED|LDL|STL)') == 0 ]]
for log in "$forced_candidate_log" "$production_log"; do
    grep -A1 "Function properties for $production_symbol" "$log" \
        | grep -q '0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads'
done
forced_candidate_sass_sha256=$(sha256sum "$forced_candidate_sass" | awk '{print $1}')
production_sass_sha256=$(sha256sum "$production_sass" | awk '{print $1}')
[[ $production_sass_sha256 == "$forced_candidate_sass_sha256" ]]

if "$nvcc_bin" -std=c++17 -O3 --fmad=false -arch=sm_121a \
    -DW2A8_FIXED_N=4096 -DW2A8_FIXED_K=2048 \
    -DW2A8_PACKED_E4M3_CANDIDATE=1 \
    -DW2A8_N256_SINGLE_WARP_ROUTE_GUARD_CANDIDATE=0 \
    -DW2A8_N256_DOWN_DOUBLE_BUFFER_CANDIDATE=1 \
    -DW2A8_N256_DOWN_CONTINUOUS_RING_CANDIDATE=2 \
    -DW2A8_KERNEL_NAME=exl3_w2a8_n256_invalid_continuous_ring \
    -cubin "$source_file" -o "$probe_dir/invalid-selector.cubin" >/dev/null 2>&1; then
    echo "non-boolean N256 continuous-ring selector unexpectedly compiled" >&2
    exit 1
fi

if "$nvcc_bin" -std=c++17 -O3 --fmad=false -arch=sm_121a \
    -DW2A8_FIXED_N=4096 -DW2A8_FIXED_K=2048 \
    -DW2A8_PACKED_E4M3_CANDIDATE=0 \
    -DW2A8_N256_SINGLE_WARP_ROUTE_GUARD_CANDIDATE=0 \
    -DW2A8_N256_DOWN_DOUBLE_BUFFER_CANDIDATE=1 \
    -DW2A8_N256_DOWN_CONTINUOUS_RING_CANDIDATE=1 \
    -DW2A8_KERNEL_NAME=exl3_w2a8_n256_invalid_ring_packing \
    -cubin "$source_file" -o "$probe_dir/invalid-packing.cubin" >/dev/null 2>&1; then
    echo "N256 continuous ring unexpectedly accepted packed-E4M3 zero" >&2
    exit 1
fi

if "$nvcc_bin" -std=c++17 -O3 --fmad=false -arch=sm_121a \
    -DW2A8_FIXED_N=2048 -DW2A8_FIXED_K=4096 \
    -DW2A8_PACKED_E4M3_CANDIDATE=1 \
    -DW2A8_N256_SINGLE_WARP_ROUTE_GUARD_CANDIDATE=0 \
    -DW2A8_N256_DOWN_DOUBLE_BUFFER_CANDIDATE=1 \
    -DW2A8_N256_DOWN_CONTINUOUS_RING_CANDIDATE=1 \
    -DW2A8_KERNEL_NAME=exl3_w2a8_n256_invalid_gu_continuous_ring \
    -cubin "$source_file" -o "$probe_dir/invalid-gu.cubin" >/dev/null 2>&1; then
    echo "N256 down continuous ring unexpectedly accepted the GU shape" >&2
    exit 1
fi

if "$nvcc_bin" -std=c++17 -O3 --fmad=false -arch=sm_121a \
    -DW2A8_FIXED_N=4096 -DW2A8_FIXED_K=2048 \
    -DW2A8_PACKED_E4M3_CANDIDATE=1 \
    -DW2A8_N256_SINGLE_WARP_ROUTE_GUARD_CANDIDATE=0 \
    -DW2A8_N256_DOWN_DOUBLE_BUFFER_CANDIDATE=0 \
    -DW2A8_N256_DOWN_CONTINUOUS_RING_CANDIDATE=1 \
    -DW2A8_KERNEL_NAME=exl3_w2a8_n256_invalid_ring_dependency \
    -cubin "$source_file" -o "$probe_dir/invalid-dependency.cubin" >/dev/null 2>&1; then
    echo "N256 continuous ring unexpectedly accepted double-buffer zero" >&2
    exit 1
fi

digest=$(
    sha256sum "$probe_dir"/{incumbent,candidate}.{cubin,resources,sass} \
        "$forced_candidate_cubin" "$forced_candidate_resources" \
        "$forced_candidate_sass" "$production_cubin" \
        "$production_resources" "$production_sass" \
        | awk '{print $1}' | sha256sum | awk '{print $1}'
)
echo "continuous_ring_sass=PASS N256 down K64 ring: packed E4M3=1 double_buffer=1 route_guard=0 one selector arithmetic census exact"
echo "incumbent: instructions=$incumbent_instructions registers=$incumbent_registers shared=$incumbent_shared barriers=$incumbent_barriers ldgsts=$(metric incumbent ldgsts)"
echo "candidate: instructions=$candidate_instructions registers=$candidate_registers shared=$candidate_shared barriers=$candidate_barriers ldgsts=$(metric candidate ldgsts) commits=$(metric candidate ldgdepbar) waits=$(metric candidate depbar_waits) zfill=$(metric candidate zfill)"
echo "production hostile ring zero: continuous_ring=1 instructions=$production_instruction_expected registers=103 shared=19456 barriers=3 sass_sha256=$production_sass_sha256"
echo "ceilings: instructions<=$instruction_limit registers<=$register_limit shared<=$shared_limit stack=0 local=0 spills=0 atomics=0 cubin_set_sha256=$digest"
