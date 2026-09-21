#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cuda_root_path=${CUDA_ROOT_PATH:-/usr/local/cuda}
nvcc_bin=${NVCC_BIN:-"$cuda_root_path/bin/nvcc"}
cuobjdump_bin=${CUOBJDUMP_BIN:-"$cuda_root_path/bin/cuobjdump"}
nvdisasm_bin=${NVDISASM_BIN:-"$cuda_root_path/bin/nvdisasm"}
source_file="$repo_root/kernels/gb10/experiments/exl3_w2a8_fused_gu_down_emit_n128.cu"
wrapper_file="$repo_root/kernels/gb10/deepseek-v4-flash/nvfp4/exl3_w2a8_fused_gu_down_emit_n128.cu"
probe_dir=$(mktemp -d /tmp/atlas-exl3-w2a8-fused-gu-n128-double-buffer.XXXXXX)
trap 'rm -rf -- "$probe_dir"' EXIT
static_shared_limit=$((48 * 1024))
register_limit=128

if [[ -n ${NVCC_PREPEND_FLAGS:-} || -n ${NVCC_APPEND_FLAGS:-} ]]; then
    echo "NVCC_PREPEND_FLAGS and NVCC_APPEND_FLAGS must be empty" >&2
    exit 2
fi
for tool in "$nvcc_bin" "$cuobjdump_bin" "$nvdisasm_bin"; do
    if [[ ! -x $tool ]]; then
        echo "missing CUDA tool: $tool" >&2
        exit 2
    fi
done

for contract in \
    '#define W2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE 0' \
    'static_assert(W2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE == 0 ||' \
    'W2FGemmScratch gemm[2]' \
    'sizeof(W2FShared) == 46 * 1024' \
    'cp.async.ca.shared.global' \
    'cp.async.commit_group' \
    'cp.async.wait_group 0' \
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
    local symbol="exl3_w2a8_fused_gu_down_emit_n128_${name}"
    local cubin="$probe_dir/${name}.cubin"
    local log="$probe_dir/${name}.log"
    local resources="$probe_dir/${name}.resources"
    local sass="$probe_dir/${name}.sass"

    "$nvcc_bin" -std=c++17 -O3 --fmad=false -arch=sm_121a \
        -DW2A8_FIXED_N=2048 -DW2A8_FIXED_K=4096 \
        -DW2A8_PACKED_E4M3_CANDIDATE=1 \
        -DW2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE="$selector" \
        -DW2A8_KERNEL_NAME="$symbol" -cubin "$source_file" \
        -o "$cubin" -Xptxas=-v 2>"$log"
    "$cuobjdump_bin" --dump-resource-usage "$cubin" >"$resources"
    "$nvdisasm_bin" "$cubin" >"$sass"

    local registers stack local_bytes shared instructions barriers qmma
    local bf16_rounds fp8_converts swiglu_exp fma_ops shuffles ldgsts
    local ldgdepbar depbar_waits zfill
    registers=$(resource_value "$resources" "$symbol" REG)
    stack=$(resource_value "$resources" "$symbol" STACK)
    local_bytes=$(resource_value "$resources" "$symbol" LOCAL)
    shared=$(resource_value "$resources" "$symbol" SHARED)
    instructions=$(instruction_count "$sass")
    barriers=$(sass_count "$sass" 'BAR.SYNC')
    qmma=$(sass_count "$sass" 'QMMA.16832.F32.E4M3.E4M3')
    bf16_rounds=$(sass_count "$sass" 'F2F(P)?\.BF16\.F32')
    fp8_converts=$(sass_count "$sass" 'F2FP.SATFINITE.E4M3')
    swiglu_exp=$(sass_count "$sass" 'MUFU.EX2')
    fma_ops=$(sass_count "$sass" '[[:space:]](FFMA|HFMA2)')
    shuffles=$(sass_count "$sass" 'SHFL\.')
    ldgsts=$(sass_count "$sass" 'LDGSTS')
    ldgdepbar=$(sass_count "$sass" 'LDGDEPBAR')
    depbar_waits=$(sass_count "$sass" 'DEPBAR\.LE SB0, 0x0')
    zfill=$(sass_count "$sass" 'LDGSTS[^;]*ZFILL')
    for value in \
        "$registers" "$stack" "$local_bytes" "$shared" "$instructions" \
        "$barriers" "$qmma" "$bf16_rounds" "$fp8_converts" \
        "$swiglu_exp" "$fma_ops" "$shuffles" "$ldgsts" \
        "$ldgdepbar" "$depbar_waits" "$zfill"; do
        [[ $value =~ ^[0-9]+$ ]]
    done
    [[ $stack == 0 ]]
    [[ $local_bytes == 0 ]]
    (( registers <= register_limit ))
    (( shared <= static_shared_limit ))
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
        "fp8_converts=$fp8_converts" \
        "swiglu_exp=$swiglu_exp" \
        "fma_ops=$fma_ops" \
        "shuffles=$shuffles" \
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

for field in qmma bf16_rounds fp8_converts swiglu_exp fma_ops shuffles; do
    [[ $(metric candidate "$field") == "$(metric incumbent "$field")" ]]
done
[[ $(metric incumbent qmma) == 32 ]]
[[ $(metric incumbent bf16_rounds) == 52 ]]
[[ $(metric incumbent fp8_converts) == 36 ]]
[[ $(metric incumbent swiglu_exp) == 4 ]]
[[ $(metric incumbent fma_ops) == 166 ]]
[[ $(metric incumbent shuffles) == 193 ]]
[[ $(metric incumbent ldgsts) == 0 ]]
[[ $(metric candidate ldgsts) == 8 ]]
[[ $(metric incumbent ldgdepbar) == 0 ]]
[[ $(metric incumbent depbar_waits) == 0 ]]
[[ $(metric incumbent zfill) == 0 ]]
[[ $(metric candidate ldgdepbar) == 4 ]]
[[ $(metric candidate depbar_waits) == 4 ]]
[[ $(metric candidate zfill) == 2 ]]

awk '
    /LDGSTS/ { last_issue = NR }
    /LDGDEPBAR/ {
        if (!(last_issue < NR)) exit 1
        last_commit = NR
        commits++
    }
    /DEPBAR\.LE SB0, 0x0/ {
        if (!(last_commit < NR)) exit 1
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
        if (commits != 4 || waits != 4 || publications != 4 || last_wait != 0)
            exit 1
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
[[ $incumbent_shared == 40960 ]]
[[ $incumbent_instructions == 3280 ]]
[[ $candidate_instructions == 3368 ]]
(( candidate_registers == incumbent_registers ))
[[ $candidate_shared == 48128 ]]
(( candidate_shared <= static_shared_limit ))
(( candidate_barriers == incumbent_barriers ))

# The production wrapper owns both promoted selectors.  Prove hostile command-line
# zeroes cannot silently demote either arm in the real build preprocessor order.
production_symbol=exl3_w2a8_fused_gu_down_emit_n128
production_cubin="$probe_dir/production-hostile-overrides.cubin"
production_log="$probe_dir/production-hostile-overrides.log"
production_resources="$probe_dir/production-hostile-overrides.resources"
production_sass="$probe_dir/production-hostile-overrides.sass"
"$nvcc_bin" -std=c++17 -O3 --fmad=false -arch=sm_121a \
    -DW2A8_PACKED_E4M3_CANDIDATE=0 \
    -DW2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE=0 \
    -cubin "$wrapper_file" -o "$production_cubin" -Xptxas=-v 2>"$production_log"
"$cuobjdump_bin" --dump-resource-usage "$production_cubin" >"$production_resources"
"$nvdisasm_bin" "$production_cubin" >"$production_sass"
[[ $(resource_value "$production_resources" "$production_symbol" REG) == 128 ]]
[[ $(resource_value "$production_resources" "$production_symbol" SHARED) == 48128 ]]
[[ $(sass_count "$production_sass" 'LDGSTS') == 8 ]]
[[ $(sass_count "$production_sass" 'LDGDEPBAR') == 4 ]]
[[ $(sass_count "$production_sass" 'DEPBAR\.LE SB0, 0x0') == 4 ]]
[[ $(sass_count "$production_sass" 'F2FP\.SATFINITE\.E4M3\.F16') == 32 ]]
[[ $(sass_count "$production_sass" 'F2FP\.SATFINITE\.E4M3\.F32') == 4 ]]
grep -A1 "Function properties for $production_symbol" "$production_log" \
    | grep -q '0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads'

if "$nvcc_bin" -std=c++17 -O3 --fmad=false -arch=sm_121a \
    -DW2A8_FIXED_N=2048 -DW2A8_FIXED_K=4096 \
    -DW2A8_PACKED_E4M3_CANDIDATE=1 \
    -DW2A8_FUSED_GU_N128_DOUBLE_BUFFER_CANDIDATE=2 \
    -DW2A8_KERNEL_NAME=exl3_w2a8_fused_invalid_double_buffer \
    -cubin "$source_file" -o "$probe_dir/invalid.cubin" >/dev/null 2>&1; then
    echo "non-boolean double-buffer selector unexpectedly compiled" >&2
    exit 1
fi

digest=$(
    sha256sum "$probe_dir"/{incumbent,candidate}.{cubin,resources,sass} \
        | awk '{print $1}' | sha256sum | awk '{print $1}'
)
echo "fused N128 K64 double buffer: packed E4M3, one selector, arithmetic census exact"
echo "incumbent: instructions=$incumbent_instructions registers=$incumbent_registers shared=$incumbent_shared barriers=$incumbent_barriers ldgsts=$(metric incumbent ldgsts)"
echo "candidate: instructions=$candidate_instructions registers=$candidate_registers shared=$candidate_shared barriers=$candidate_barriers ldgsts=$(metric candidate ldgsts)"
echo "production hostile selector zeroes: packed=1 double_buffer=1"
echo "stack=0 local=0 spills=0 atomics=0 cubin_set_sha256=$digest"
