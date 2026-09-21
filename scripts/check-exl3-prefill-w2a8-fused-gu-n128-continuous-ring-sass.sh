#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only

set -euo pipefail
export LC_ALL=C

report_error() {
    local status=$?
    echo "continuous-ring SASS gate failed at line ${BASH_LINENO[0]}" >&2
    if [[ -d ${probe_dir:-} ]]; then
        local metrics
        for metrics in "$probe_dir"/*.metrics; do
            [[ -f $metrics ]] || continue
            echo "--- $(basename "$metrics") ---" >&2
            sed -n '1,40p' "$metrics" >&2
        done
    fi
    exit "$status"
}
trap report_error ERR

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cuda_root_path=${CUDA_ROOT_PATH:-/usr/local/cuda}
nvcc_bin=${NVCC_BIN:-"$cuda_root_path/bin/nvcc"}
cuobjdump_bin=${CUOBJDUMP_BIN:-"$cuda_root_path/bin/cuobjdump"}
host_cxx_bin=${HOST_CXX_BIN:-/usr/bin/g++}
source_file="$repo_root/kernels/gb10/experiments/exl3_w2a8_fused_gu_down_emit_n128.cu"
wrapper_file="$repo_root/kernels/gb10/deepseek-v4-flash/nvfp4/exl3_w2a8_fused_gu_down_emit_n128.cu"
harness_file="$repo_root/kernels/gb10/experiments/exl3_w2a8_composed_n128_n256_probe.cu"
emitter_wrapper="$repo_root/kernels/gb10/deepseek-v4-flash/nvfp4/exl3_w2a8_h128_emit.cu"
n64_gu_wrapper="$repo_root/kernels/gb10/deepseek-v4-flash/nvfp4/exl3_w2a8_grouped_prefill_k2_gu.cu"
n64_down_wrapper="$repo_root/kernels/gb10/deepseek-v4-flash/nvfp4/exl3_w2a8_grouped_prefill_k2_down.cu"
n256_down_wrapper="$repo_root/kernels/gb10/deepseek-v4-flash/nvfp4/exl3_w2a8_grouped_prefill_n256_k2_down.cu"
probe_dir=$(mktemp -d /tmp/atlas-exl3-w2a8-fused-gu-n128-continuous-ring.XXXXXX)
trap 'rm -rf -- "$probe_dir"' EXIT
experiment_wrapper="$probe_dir/continuous-ring-experiment-wrapper.cu"
register_limit=128
shared_limit=48128
instruction_limit=2800
production_instruction_expected=2784
wrapper_sha256_expected=4bd2a209b195065d63042a50183e59e60f6773524beed07fc56e4c105e8725b2
compute_body_sha256_expected=7b0773744b91b8eac4b4d290ca1c3f608aa45670304678b4206624b115a0933e

if [[ -n ${NVCC_PREPEND_FLAGS:-} || -n ${NVCC_APPEND_FLAGS:-} ]]; then
    echo "NVCC_PREPEND_FLAGS and NVCC_APPEND_FLAGS must be empty" >&2
    exit 2
fi
for tool in "$nvcc_bin" "$cuobjdump_bin" "$host_cxx_bin"; do
    if [[ ! -x $tool ]]; then
        echo "missing continuous-ring compile tool: $tool" >&2
        exit 2
    fi
done

printf '%s\n' \
    '// SPDX-License-Identifier: AGPL-3.0-only' \
    '#define W2A8_FIXED_N 2048' \
    '#define W2A8_FIXED_K 4096' \
    '#define W2A8_KERNEL_NAME exl3_w2a8_fused_gu_down_emit_n128' \
    '#ifndef W2A8_PACKED_E4M3_CANDIDATE' \
    '#error "W2A8_PACKED_E4M3_CANDIDATE must be supplied by the experiment"' \
    '#endif' \
    '#ifndef W2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE' \
    '#error "W2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE must be supplied by the experiment"' \
    '#endif' \
    '#ifndef W2A8_FUSED_GU_N128_CONTINUOUS_RING_CANDIDATE' \
    '#error "W2A8_FUSED_GU_N128_CONTINUOUS_RING_CANDIDATE must be supplied by the experiment"' \
    '#endif' \
    "#include \"$source_file\"" >"$experiment_wrapper"
if grep -Fq '#define W2A8_FUSED_GU_N128_CONTINUOUS_RING_CANDIDATE' \
    "$experiment_wrapper"; then
    echo "experiment wrapper must not define the A/B selector" >&2
    exit 1
fi

for contract in \
    '#define W2A8_FUSED_GU_N128_CONTINUOUS_RING_CANDIDATE 0' \
    'static_assert(W2A8_FUSED_GU_N128_CONTINUOUS_RING_CANDIDATE == 0 ||' \
    'static_assert(W2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE == 1,' \
    'static_assert(W2A8_PACKED_E4M3_CANDIDATE == 1,' \
    '// BEGIN continuous K64 ring initial publication' \
    '// BEGIN continuous K64 ring schedule' \
    'const unsigned int current_buffer = absolute_stage & 1;' \
    'w2f_stage_async(scratch_buffers[next_buffer]' \
    'if (next_k < W2F_GATE_UP_K) w2f_cp_async_wait();'; do
    grep -Fq "$contract" "$source_file"
done
for contract in \
    '#ifdef W2A8_FUSED_GU_N128_CONTINUOUS_RING_CANDIDATE' \
    '#undef W2A8_FUSED_GU_N128_CONTINUOUS_RING_CANDIDATE' \
    '#define W2A8_FUSED_GU_N128_CONTINUOUS_RING_CANDIDATE 1'; do
    grep -Fq "$contract" "$wrapper_file"
done
[[ $(sha256sum "$wrapper_file" | awk '{print $1}') == "$wrapper_sha256_expected" ]]
compute_body_sha256=$(
    awk '
        /void w2f_compute_leg\(/ { in_helper = 1 }
        /void w2f_emit_group\(/ { in_helper = 0 }
        in_helper && /w2f_decode8|w2f_encode_pair|w2f_repack_b|w2f_mma|activation_scale|isfinite|factor[01]|outer\[|output\[/ {
            gsub(/[[:space:]]+/, "")
            print
        }
    ' "$source_file" | sha256sum | awk '{print $1}'
)
[[ $compute_body_sha256 == "$compute_body_sha256_expected" ]]

resource_value() {
    local file=$1 symbol=$2 field=$3
    awk -v symbol="$symbol" -v field="$field" '
        $0 == " Function " symbol ":" { getline; line = $0; found = 1 }
        END {
            if (!found) exit 1
            count = split(line, parts, " ")
            for (i = 1; i <= count; i++) {
                split(parts[i], pair, ":")
                if (pair[1] == field) { print pair[2]; exit }
            }
            exit 1
        }
    ' "$file"
}

instruction_count() {
    grep -cE '^[[:space:]]*/\*[0-9a-f]+\*/' "$1" || true
}

sass_count() {
    grep -cE "$2" "$1" || true
}

compile_variant() {
    local name=$1 selector=$2
    local symbol=exl3_w2a8_fused_gu_down_emit_n128
    local binary="$probe_dir/${name}.binary"
    local log="$probe_dir/${name}.log"
    local resources="$probe_dir/${name}.resources"
    local sass="$probe_dir/${name}.sass"

    "$nvcc_bin" -ccbin "$host_cxx_bin" -std=c++17 -O3 --fmad=false \
        -arch=sm_121a -Xptxas=-v \
        -DW2A8_PACKED_E4M3_CANDIDATE=1 \
        -DW2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE=1 \
        -DW2A8_FUSED_GU_N128_CONTINUOUS_RING_CANDIDATE="$selector" \
        -DW2A8_N256_SINGLE_WARP_ROUTE_GUARD_CANDIDATE=0 \
        -DW2A8_N256_DOWN_DOUBLE_BUFFER_CANDIDATE=1 \
        -DW2A8_COMPOSED_PROBE_ENFORCE_SPEEDUP=0 \
        '-DW2A8_COMPOSED_PROBE_BUILD_ID="continuous-ring-static"' \
        -DW2A8_COMPOSED_PROBE_MIN_SPEEDUP=1.01 \
        '-DW2A8_COMPOSED_PROBE_MIN_SPEEDUP_TEXT="1.01"' \
        "$harness_file" "$emitter_wrapper" "$n64_gu_wrapper" \
        "$n64_down_wrapper" "$experiment_wrapper" "$n256_down_wrapper" \
        -o "$binary" 2>"$log"
    "$cuobjdump_bin" --dump-resource-usage "$binary" >"$resources"
    "$cuobjdump_bin" --dump-sass --function "$symbol" "$binary" >"$sass"
    grep -Fq "$symbol" "$sass"

    local registers stack local_bytes shared instructions barriers
    local qmma bf16_rounds fp8_converts fp8_f16 fp8_f32 swiglu_exp
    local shuffles fadd fmul ffma hadd2 hmul2 ldgsts commits waits zfill
    registers=$(resource_value "$resources" "$symbol" REG)
    stack=$(resource_value "$resources" "$symbol" STACK)
    local_bytes=$(resource_value "$resources" "$symbol" LOCAL)
    shared=$(resource_value "$resources" "$symbol" SHARED)
    instructions=$(instruction_count "$sass")
    barriers=$(sass_count "$sass" 'BAR\.SYNC')
    qmma=$(sass_count "$sass" 'QMMA\.16832\.F32\.E4M3\.E4M3')
    bf16_rounds=$(sass_count "$sass" 'F2F(P)?\.BF16\.F32')
    fp8_converts=$(sass_count "$sass" 'F2FP\.SATFINITE\.E4M3')
    fp8_f16=$(sass_count "$sass" 'F2FP\.SATFINITE\.E4M3\.F16')
    fp8_f32=$(sass_count "$sass" 'F2FP\.SATFINITE\.E4M3\.F32')
    swiglu_exp=$(sass_count "$sass" 'MUFU\.EX2')
    shuffles=$(sass_count "$sass" 'SHFL\.')
    fadd=$(sass_count "$sass" '[[:space:]]FADD')
    fmul=$(sass_count "$sass" '[[:space:]]FMUL')
    ffma=$(sass_count "$sass" '[[:space:]]FFMA')
    hadd2=$(sass_count "$sass" '[[:space:]]HADD2')
    hmul2=$(sass_count "$sass" '[[:space:]]HMUL2')
    ldgsts=$(sass_count "$sass" 'LDGSTS')
    commits=$(sass_count "$sass" 'LDGDEPBAR')
    waits=$(sass_count "$sass" 'DEPBAR\.LE SB0, 0x0')
    zfill=$(sass_count "$sass" 'LDGSTS[^;]*ZFILL')
    for value in \
        "$registers" "$stack" "$local_bytes" "$shared" "$instructions" \
        "$barriers" "$qmma" "$bf16_rounds" "$fp8_converts" \
        "$fp8_f16" "$fp8_f32" "$swiglu_exp" "$shuffles" "$fadd" \
        "$fmul" "$ffma" "$hadd2" "$hmul2" "$ldgsts" "$commits" \
        "$waits" "$zfill"; do
        [[ $value =~ ^[0-9]+$ ]]
    done
    [[ $stack == 0 && $local_bytes == 0 ]]
    (( registers <= register_limit ))
    (( shared <= shared_limit ))
    (( instructions <= instruction_limit ))
    grep -A1 "Function properties for $symbol" "$log" \
        | grep -q '0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads'
    [[ $(sass_count "$sass" '[[:space:]](ATOM|RED|LDL|STL)') == 0 ]]
    printf '%s\n' \
        "registers=$registers" "stack=$stack" "local=$local_bytes" \
        "shared=$shared" "instructions=$instructions" "barriers=$barriers" \
        "qmma=$qmma" "bf16_rounds=$bf16_rounds" \
        "fp8_converts=$fp8_converts" "fp8_f16=$fp8_f16" \
        "fp8_f32=$fp8_f32" "swiglu_exp=$swiglu_exp" \
        "shuffles=$shuffles" "fadd=$fadd" "fmul=$fmul" "ffma=$ffma" \
        "hadd2=$hadd2" "hmul2=$hmul2" "ldgsts=$ldgsts" \
        "commits=$commits" "waits=$waits" "zfill=$zfill" \
        >"$probe_dir/${name}.metrics"
}

compile_variant incumbent 0
compile_variant candidate 1

metric() {
    local name=$1 field=$2
    awk -F= -v field="$field" '$1 == field { print $2 }' \
        "$probe_dir/${name}.metrics"
}

for field in \
    qmma bf16_rounds fp8_converts fp8_f16 fp8_f32 swiglu_exp \
    shuffles fadd fmul ffma hadd2 hmul2; do
    [[ $(metric candidate "$field") == "$(metric incumbent "$field")" ]]
done
[[ $(metric incumbent qmma) == 32 ]]
[[ $(metric incumbent bf16_rounds) == 52 ]]
[[ $(metric incumbent fp8_converts) == 36 ]]
[[ $(metric incumbent fp8_f16) == 32 ]]
[[ $(metric incumbent fp8_f32) == 4 ]]
[[ $(metric incumbent swiglu_exp) == 4 ]]
[[ $(metric incumbent shuffles) == 193 ]]
[[ $(metric incumbent fadd) == 116 ]]
[[ $(metric incumbent fmul) == 128 ]]
[[ $(metric incumbent ffma) == 166 ]]
[[ $(metric incumbent hadd2) == 44 ]]
[[ $(metric incumbent hmul2) == 32 ]]
for variant in incumbent candidate; do
    [[ $(metric "$variant" ldgsts) == 8 ]]
    [[ $(metric "$variant" commits) == 4 ]]
    [[ $(metric "$variant" waits) == 4 ]]
done
[[ $(metric incumbent zfill) == 2 ]]
[[ $(metric candidate zfill) == 4 ]]

# Both legs have one initial issue/wait and one loop issue whose wait follows
# all sixteen static QMMAs. Every waited async group is published by a barrier.
awk '
    /LDGSTS/ { last_issue = NR }
    /LDGDEPBAR/ {
        if (!(last_issue < NR)) exit 1
        last_commit = NR
        commits++
    }
    /QMMA\.16832\.F32\.E4M3\.E4M3/ {
        if ((commits == 2 && waits == 1) ||
            (commits == 4 && waits == 3)) overlap_qmma++
    }
    /DEPBAR\.LE SB0, 0x0/ {
        if (!(last_commit < NR)) exit 1
        if ((waits == 1 || waits == 3) && overlap_qmma != 16 * ((waits + 1) / 2))
            exit 1
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
        if (commits != 4 || waits != 4 || publications != 4 ||
            overlap_qmma != 32 || last_wait != 0) exit 1
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
[[ $incumbent_registers == 128 ]]
[[ $incumbent_shared == 48128 ]]
[[ $incumbent_instructions == 2752 ]]
[[ $incumbent_barriers == 7 ]]
(( candidate_registers <= register_limit ))
(( candidate_shared <= shared_limit ))
(( candidate_instructions <= instruction_limit ))
(( candidate_barriers <= incumbent_barriers ))
[[ $candidate_shared == "$incumbent_shared" ]]
[[ $(sha256sum "$probe_dir/incumbent.binary" | awk '{print $1}') != \
   "$(sha256sum "$probe_dir/candidate.binary" | awk '{print $1}')" ]]

# The production wrapper now forces all three promoted selectors.  Compare a
# hostile ring0 production compile with a context-identical raw ring1 reference
# so the proof is about generated resources and SASS, not only source text.
production_symbol=exl3_w2a8_fused_gu_down_emit_n128
forced_candidate_cubin="$probe_dir/forced-candidate-ring1.cubin"
forced_candidate_log="$probe_dir/forced-candidate-ring1.log"
forced_candidate_resources="$probe_dir/forced-candidate-ring1.resources"
forced_candidate_sass="$probe_dir/forced-candidate-ring1.sass"
production_cubin="$probe_dir/production-hostile-ring0.cubin"
production_log="$probe_dir/production-hostile-ring0.log"
production_resources="$probe_dir/production-hostile-ring0.resources"
production_sass="$probe_dir/production-hostile-ring0.sass"

"$nvcc_bin" -ccbin "$host_cxx_bin" -std=c++17 -O3 --fmad=false \
    -arch=sm_121a -Xptxas=-v \
    -DW2A8_PACKED_E4M3_CANDIDATE=1 \
    -DW2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE=1 \
    -DW2A8_FUSED_GU_N128_CONTINUOUS_RING_CANDIDATE=1 \
    -cubin "$experiment_wrapper" -o "$forced_candidate_cubin" \
    2>"$forced_candidate_log"
"$cuobjdump_bin" --dump-resource-usage "$forced_candidate_cubin" \
    >"$forced_candidate_resources"
"$cuobjdump_bin" --dump-sass --function "$production_symbol" \
    "$forced_candidate_cubin" >"$forced_candidate_sass"

"$nvcc_bin" -ccbin "$host_cxx_bin" -std=c++17 -O3 --fmad=false \
    -arch=sm_121a -Xptxas=-v \
    -DW2A8_PACKED_E4M3_CANDIDATE=0 \
    -DW2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE=0 \
    -DW2A8_FUSED_GU_N128_CONTINUOUS_RING_CANDIDATE=0 \
    -cubin "$wrapper_file" -o "$production_cubin" 2>"$production_log"
"$cuobjdump_bin" --dump-resource-usage "$production_cubin" \
    >"$production_resources"
"$cuobjdump_bin" --dump-sass --function "$production_symbol" \
    "$production_cubin" >"$production_sass"

for field in REG STACK SHARED LOCAL; do
    [[ $(resource_value "$production_resources" "$production_symbol" "$field") == \
       "$(resource_value "$forced_candidate_resources" "$production_symbol" "$field")" ]]
done
[[ $(resource_value "$production_resources" "$production_symbol" REG) == 128 ]]
[[ $(resource_value "$production_resources" "$production_symbol" STACK) == 0 ]]
[[ $(resource_value "$production_resources" "$production_symbol" SHARED) == 48128 ]]
[[ $(resource_value "$production_resources" "$production_symbol" LOCAL) == 0 ]]
[[ $(instruction_count "$production_sass") == "$production_instruction_expected" ]]
[[ $(sass_count "$production_sass" 'BAR\.SYNC') == 7 ]]
[[ $(sass_count "$production_sass" 'QMMA\.16832\.F32\.E4M3\.E4M3') == 32 ]]
[[ $(sass_count "$production_sass" 'F2F(P)?\.BF16\.F32') == 52 ]]
[[ $(sass_count "$production_sass" 'F2FP\.SATFINITE\.E4M3') == 36 ]]
[[ $(sass_count "$production_sass" 'F2FP\.SATFINITE\.E4M3\.F16') == 32 ]]
[[ $(sass_count "$production_sass" 'F2FP\.SATFINITE\.E4M3\.F32') == 4 ]]
[[ $(sass_count "$production_sass" 'MUFU\.EX2') == 4 ]]
[[ $(sass_count "$production_sass" '[[:space:]]FADD') == 116 ]]
[[ $(sass_count "$production_sass" '[[:space:]]FMUL') == 128 ]]
[[ $(sass_count "$production_sass" '[[:space:]]FFMA') == 166 ]]
[[ $(sass_count "$production_sass" '[[:space:]]HADD2') == 44 ]]
[[ $(sass_count "$production_sass" '[[:space:]]HMUL2') == 32 ]]
[[ $(sass_count "$production_sass" 'LDGSTS') == 8 ]]
[[ $(sass_count "$production_sass" 'LDGDEPBAR') == 4 ]]
[[ $(sass_count "$production_sass" 'DEPBAR\.LE SB0, 0x0') == 4 ]]
[[ $(sass_count "$production_sass" 'LDGSTS[^;]*ZFILL') == 4 ]]
[[ $(sass_count "$production_sass" '[[:space:]](ATOM|RED|LDL|STL)') == 0 ]]
grep -A1 "Function properties for $production_symbol" "$production_log" \
    | grep -q '0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads'
forced_candidate_sass_sha256=$(sha256sum "$forced_candidate_sass" | awk '{print $1}')
production_sass_sha256=$(sha256sum "$production_sass" | awk '{print $1}')
[[ $production_sass_sha256 == "$forced_candidate_sass_sha256" ]]

compile_must_fail() {
    local name=$1 packed=$2 double_buffer=$3 ring=$4
    if "$nvcc_bin" -ccbin "$host_cxx_bin" -std=c++17 -O3 --fmad=false \
        -arch=sm_121a -DW2A8_FIXED_N=2048 -DW2A8_FIXED_K=4096 \
        -DW2A8_PACKED_E4M3_CANDIDATE="$packed" \
        -DW2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE="$double_buffer" \
        -DW2A8_FUSED_GU_N128_CONTINUOUS_RING_CANDIDATE="$ring" \
        -DW2A8_KERNEL_NAME="exl3_w2a8_invalid_${name}" -cubin "$source_file" \
        -o "$probe_dir/invalid-${name}.cubin" >/dev/null 2>&1; then
        echo "invalid continuous-ring configuration compiled: $name" >&2
        exit 1
    fi
}
compile_must_fail non_boolean 1 1 2
compile_must_fail no_double_buffer 1 0 1
compile_must_fail unpacked_e4m3 0 1 1

digest=$(
    sha256sum \
        "$probe_dir"/{incumbent,candidate}.{binary,resources,sass} \
        "$forced_candidate_cubin" "$forced_candidate_resources" \
        "$forced_candidate_sass" "$production_cubin" \
        "$production_resources" "$production_sass" \
        | awk '{print $1}' | sha256sum | awk '{print $1}'
)
echo "fused N128 continuous K64 ring: packed=1 double_buffer=1 one selector arithmetic census exact"
echo "incumbent: instructions=$incumbent_instructions registers=$incumbent_registers shared=$incumbent_shared barriers=$incumbent_barriers"
echo "candidate: instructions=$candidate_instructions registers=$candidate_registers shared=$candidate_shared barriers=$candidate_barriers ldgsts=$(metric candidate ldgsts) commits=$(metric candidate commits) waits=$(metric candidate waits) zfill=$(metric candidate zfill)"
echo "production hostile ring zero: continuous_ring=1 instructions=$production_instruction_expected registers=128 shared=48128 barriers=7 sass_sha256=$production_sass_sha256"
echo "ceilings: instructions<=$instruction_limit registers<=$register_limit shared<=$shared_limit stack=0 local=0 spills=0 atomics=0 cubin_set_sha256=$digest"
