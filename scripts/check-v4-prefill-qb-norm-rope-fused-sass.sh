#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only

set -euo pipefail

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cuda_root_path=${CUDA_ROOT_PATH:-/usr/local/cuda}
nvcc_bin=${NVCC_BIN:-"$cuda_root_path/bin/nvcc"}
cuobjdump_bin=${CUOBJDUMP_BIN:-"$cuda_root_path/bin/cuobjdump"}
nvdisasm_bin=${NVDISASM_BIN:-"$cuda_root_path/bin/nvdisasm"}
source_file="$repo_root/kernels/gb10/deepseek-v4-flash/nvfp4/v4_prefill_qb_norm_rope_fused.cu"
probe_dir=$(mktemp -d /tmp/atlas-v4-prefill-qb-norm-rope-fused.XXXXXX)
cleanup() {
    local exit_code=$?
    rm -rf -- "$probe_dir"
    trap - EXIT
    exit "$exit_code"
}
trap cleanup EXIT

for tool in "$nvcc_bin" "$cuobjdump_bin" "$nvdisasm_bin"; do
    if [[ ! -x $tool ]]; then
        echo "missing CUDA tool: $tool" >&2
        exit 2
    fi
done
for injected_flags in NVCC_PREPEND_FLAGS NVCC_APPEND_FLAGS; do
    if [[ -n ${!injected_flags:-} ]]; then
        echo "refusing unreceipted nvcc flags from $injected_flags" >&2
        exit 2
    fi
done
if [[ ! -f $source_file ]]; then
    echo "missing isolated V4 Q-B RMSNorm + RoPE source: $source_file" >&2
    exit 1
fi

cubin="$probe_dir/v4-prefill-qb-norm-rope-fused.cubin"
log="$probe_dir/v4-prefill-qb-norm-rope-fused.log"
resources="$probe_dir/v4-prefill-qb-norm-rope-fused.resources"
sass="$probe_dir/v4-prefill-qb-norm-rope-fused.sass"
function_sass="$probe_dir/v4-prefill-qb-norm-rope-fused.function.sass"

"$nvcc_bin" -std=c++17 -O3 --fmad=false -arch=sm_121a -cubin "$source_file" \
    -o "$cubin" -Xptxas=-v 2>"$log"
"$cuobjdump_bin" --dump-resource-usage "$cubin" >"$resources"
"$nvdisasm_bin" "$cubin" >"$sass"

symbol=v4_prefill_qb_norm_rope_fused

resource_value() {
    local field=$1
    awk -v symbol="$symbol" -v field="$field" '
        $0 == " Function " symbol ":" { getline; line = $0 }
        END {
            count = split(line, parts, " ")
            for (i = 1; i <= count; i++) {
                split(parts[i], pair, ":")
                if (pair[1] == field) { print pair[2]; exit }
            }
        }
    ' "$resources"
}

awk -v symbol="$symbol" '
    emit && /^\/\/--------------------- \.text\./ { exit }
    $0 == symbol ":" { emit = 1 }
    emit { print }
' "$sass" >"$function_sass"

opcode_count() {
    local pattern=$1
    grep -cE "$pattern" "$function_sass" || true
}

registers=$(resource_value REG)
stack=$(resource_value STACK)
local_bytes=$(resource_value LOCAL)
shared=$(resource_value SHARED)
instructions=$(grep -cE '^[[:space:]]*/\*[0-9a-f]+\*/' "$function_sass" || true)

[[ $registers == 34 ]]
[[ $stack == 0 ]]
[[ $local_bytes == 0 ]]
[[ $shared == 1152 ]]
[[ $instructions == 730 ]]
grep -A1 "Function properties for $symbol" "$log" \
    | grep -q '0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads'
grep -q 'Used 34 registers, used 1 barriers, 128 bytes smem' "$log"
[[ $(opcode_count 'F2F\.BF16\.F32') == 6 ]]
[[ $(opcode_count '[[:space:]]SHFL([[:space:]]|\.)') == 15 ]]
[[ $(opcode_count '[[:space:]]MUFU\.RSQ') == 1 ]]
[[ $(opcode_count '[[:space:]]MUFU\.(COS|SIN)') == 0 ]]
[[ $(opcode_count '[[:space:]]BAR([[:space:]]|\.)') == 2 ]]
[[ $(opcode_count '[[:space:]]FFMA([[:space:]]|\.)') == 26 ]]
[[ $(opcode_count '[[:space:]]FMUL([[:space:]]|\.)') == 33 ]]
[[ $(opcode_count '[[:space:]]FADD([[:space:]]|\.)') == 27 ]]
[[ $(opcode_count '[[:space:]]LDG([[:space:]]|\.)') == 11 ]]
[[ $(opcode_count '[[:space:]]STG([[:space:]]|\.)') == 2 ]]
[[ $(opcode_count '[[:space:]](ATOM|RED|LDL|STL)(\.|[[:space:]])') == 0 ]]

digest=$(sha256sum "$cubin" | awk '{print $1}')
echo "$symbol: instructions=$instructions registers=$registers shared=$shared ptxas_smem=128 stack=0 local=0 spills=0 atomics=0"
echo "topology: bf16_f2f=6 shfl=15 rsq=1 bar=2 ffma=26 fmul=33 fadd=27 ldg=11 stg=2"
echo "cubin_set_sha256=$digest"
