#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
lib="$here/build/libatlas_gdn_c143_sm121.so"
[[ -f "$lib" ]] || { echo "missing build artifact" >&2; exit 2; }
[[ "$(sha256sum "$lib" | awk '{print $1}')" == 272ce1b115dc16eaef22bc2cbe5d1e82ec4889782475f4c68567eab8435cd3f0 ]] || {
  echo "deterministic c143 library hash drift" >&2; exit 2;
}
[[ "$(grep -Fc '"$strip_bin" --strip-unneeded' "$here/build.sh")" == 2 ]] || {
  echo "both native libraries must pass through the pinned deterministic strip step" >&2; exit 2;
}
! readelf -S --wide "$lib" | grep -Eq '[[:space:]]\.symtab[[:space:]]|[[:space:]]\.strtab[[:space:]]'

actual=$(readelf -Ws "$lib" | awk '$4=="FUNC" && $7!="UND" && $8 ~ /^atlas_gdn_c143_/ {print $8}' | sort -u)
expected=$'atlas_gdn_c143_abi_identity\natlas_gdn_c143_last_error\natlas_gdn_c143_launch\natlas_gdn_c143_launch_v2\natlas_gdn_c143_launch_v3\natlas_gdn_c143_workspace_size\natlas_gdn_c143_workspace_size_v2\natlas_gdn_c143_workspace_size_v3'
[[ "$actual" == "$expected" ]] || { echo "export drift" >&2; exit 3; }
atlas_lib="$here/build/libatlas_gdn_wy32_sm121.so"
atlas_actual=$(readelf -Ws "$atlas_lib" | awk '$4=="FUNC" && $7!="UND" && $8 ~ /^atlas_gdn_wy32_/ {print $8}' | sort -u)
atlas_expected=$'atlas_gdn_wy32_abi_identity\natlas_gdn_wy32_last_error\natlas_gdn_wy32_launch'
[[ "$atlas_actual" == "$atlas_expected" ]] || { echo "Atlas bridge export drift" >&2; exit 3; }
elfs=$(/usr/local/cuda/bin/cuobjdump --list-elf "$lib")
grep -F 'sm_121a.cubin' <<<"$elfs" >/dev/null
ptx=$(/usr/local/cuda/bin/cuobjdump --list-ptx "$lib" 2>/dev/null)
[[ -z "$ptx" ]] || { echo "unexpected PTX fallback in SM121 library" >&2; exit 3; }
python3 - "$lib" <<'PY'
import re
import subprocess
import sys

lib = sys.argv[1]
kernels = (
    "gated_delta_rule_c143_v3_kkt_solve",
    "gated_delta_rule_c143_v3_recompute_wu",
    "gated_delta_rule_c143_v3_delta_h",
    "gated_delta_rule_c143_v3_fwd_o",
)
resources = subprocess.check_output(
    ["/usr/local/cuda/bin/cuobjdump", "--dump-resource-usage", lib], text=True
)
sass = subprocess.check_output(
    ["/usr/local/cuda/bin/cuobjdump", "--dump-sass", lib], text=True
)
for kernel in kernels:
    match = re.search(rf" Function {re.escape(kernel)}:\n([^\n]+)", resources)
    assert match, kernel
    receipt = match.group(1)
    assert "STACK:0" in receipt and "LOCAL:0" in receipt, (kernel, receipt)
    section = re.search(
        rf"Function : {re.escape(kernel)}\n(.*?)(?=\n\s*Function : |\Z)",
        sass,
        re.S,
    )
    assert section and "HMMA.16816.F32.BF16" in section.group(1), kernel
PY
python3 - "$atlas_lib" "$here/src/atlas_wy32_bridge.cu" "$here/../../kernels/gb10/common/gated_delta_rule_wy32_gatecache.cu" <<'PY'
import ctypes, hashlib, sys

lib = ctypes.CDLL(sys.argv[1])
lib.atlas_gdn_wy32_abi_identity.restype = ctypes.c_char_p
bridge = hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest()
source = hashlib.sha256(open(sys.argv[3], "rb").read()).hexdigest()
assert lib.atlas_gdn_wy32_abi_identity().decode("ascii") == (
    "atlas-gdn-wy32-bridge-v1:" + source + ":" + bridge
)
PY
! rg -n 'gdn-sglang-c143' "$here/../../crates" "$here/../../kernels" \
    "$here/../../bench" "$here/../../docs"
python3 -m py_compile "$here/gate.py"
python3 - "$here/gate.py" <<'PY'
import ast
import statistics
import sys

tree = ast.parse(open(sys.argv[1], encoding="utf-8").read())
function = next(
    node for node in tree.body
    if isinstance(node, ast.FunctionDef) and node.name == "evaluate_performance_screen"
)
namespace = {"statistics": statistics}
exec(compile(ast.Module(body=[function], type_ignores=[]), sys.argv[1], "exec"), namespace)
evaluate = namespace["evaluate_performance_screen"]
positions = {
    "atlas": [3, 3, 3, 2],
    "port": [2, 3, 3, 3],
    "port_v3": [3, 2, 3, 3],
    "sglang": [3, 3, 2, 3],
}

def case(atlas, candidate, sglang, *, m=2079, counts=positions):
    times = {
        "atlas": [atlas] * 11,
        "port": [8.3 if m == 2079 else 32.5] * 11,
        "port_v3": [candidate] * 11,
        "sglang": [sglang] * 11,
    }
    timing = {
        name: {
            "median_ms": statistics.median(values),
            "p90_ms": sorted(values)[9],
        }
        for name, values in times.items()
    }
    return evaluate(m, times, timing, counts)

assert case(7.5, 2.4, 2.1)["pass"]
assert case(29.9, 9.9, 8.45, m=8192)["pass"]
assert not case(2.0, 2.4, 1.0)["pass"]  # old absolute-only false pass
assert not case(7.5, 2.4, 1.0)["pass"]  # >1.20x pinned SGLang
assert not case(7.5, 2.5, 2.1)["pass"]  # strict absolute bound
tail_times = {
    "atlas": [3.0] * 11,
    "port": [8.3] * 11,
    "port_v3": [2.0] * 9 + [50.0] * 2,
    "sglang": [2.0] * 11,
}
tail_timing = {
    name: {
        "median_ms": statistics.median(values),
        "p90_ms": sorted(values)[9],
    }
    for name, values in tail_times.items()
}
tail_screen = evaluate(2079, tail_times, tail_timing, positions)
assert tail_screen["port_v3_median_ms"] < tail_screen["atlas_median_ms"]
assert tail_screen["port_v3_p90_ms"] > tail_screen["atlas_p90_ms"]
assert tail_screen["paired_atlas_gain_median_ms"] > 0.0
assert not tail_screen["atlas_p90_comparative_pass"]
assert not tail_screen["pass"]
unbalanced = {name: [11, 0, 0, 0] for name in positions}
assert not case(7.5, 2.4, 2.1, counts=unbalanced)["pass"]
PY
python3 - "$lib" <<'PY'
import ctypes, sys
lib = ctypes.CDLL(sys.argv[1])
lib.atlas_gdn_c143_abi_identity.restype = ctypes.c_char_p
source = __import__("hashlib").sha256(
    open(__import__("pathlib").Path(sys.argv[1]).parents[1] / "src/gated_delta_rule_sglang_c143.cu", "rb").read()
).hexdigest()
assert lib.atlas_gdn_c143_abi_identity().decode("ascii") == (
    "atlas-gdn-c143-abi-v3:qwen38-c1-b1-workspace-v3:" + source
)
lib.atlas_gdn_c143_workspace_size.argtypes = [ctypes.c_uint]
lib.atlas_gdn_c143_workspace_size.restype = ctypes.c_size_t
lib.atlas_gdn_c143_workspace_size_v2.argtypes = [ctypes.c_uint]
lib.atlas_gdn_c143_workspace_size_v2.restype = ctypes.c_size_t
lib.atlas_gdn_c143_workspace_size_v3.argtypes = [ctypes.c_uint]
lib.atlas_gdn_c143_workspace_size_v3.restype = ctypes.c_size_t
assert lib.atlas_gdn_c143_workspace_size(0) == 0
assert lib.atlas_gdn_c143_workspace_size(2079) == 130166784
assert lib.atlas_gdn_c143_workspace_size(8192) == 504889344
assert lib.atlas_gdn_c143_workspace_size_v2(0) == 0
assert lib.atlas_gdn_c143_workspace_size_v2(2079) == 130565952
assert lib.atlas_gdn_c143_workspace_size_v2(8192) == 506462208
assert lib.atlas_gdn_c143_workspace_size_v3(0) == 0
assert lib.atlas_gdn_c143_workspace_size_v3((1 << 32) - 1) == 0
assert lib.atlas_gdn_c143_workspace_size_v3(2079) == 130565952
assert lib.atlas_gdn_c143_workspace_size_v3(8192) == 506462208
assert lib.atlas_gdn_c143_workspace_size_v3(262144) == 16206790656
assert lib.atlas_gdn_c143_workspace_size_v3(1048576) == 64827162624
assert lib.atlas_gdn_c143_workspace_size_v3(4194240) == 259304693760
assert lib.atlas_gdn_c143_workspace_size_v3(4194241) == 0
for m in (1, 63, 64, 65, 2079, 8192):
    assert (lib.atlas_gdn_c143_workspace_size_v2(m)
            - lib.atlas_gdn_c143_workspace_size(m)) == m * 48 * 4
    assert (lib.atlas_gdn_c143_workspace_size_v3(m)
            == lib.atlas_gdn_c143_workspace_size_v2(m))
lib.atlas_gdn_c143_launch.argtypes = [ctypes.c_void_p] * 8 + [
    ctypes.c_size_t, ctypes.c_uint, ctypes.c_void_p
]
lib.atlas_gdn_c143_launch.restype = ctypes.c_int
assert lib.atlas_gdn_c143_launch(*([None] * 8), 0, 2079, None) == -1
lib.atlas_gdn_c143_last_error.restype = ctypes.c_char_p
assert b"invalid null/zero argument" in lib.atlas_gdn_c143_last_error()

lib.atlas_gdn_c143_launch_v2.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t,
    ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
    ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_size_t, ctypes.c_uint, ctypes.c_void_p,
]
lib.atlas_gdn_c143_launch_v2.restype = ctypes.c_int
assert lib.atlas_gdn_c143_launch_v2(
    None, None, 0, 0, 0, 0, 0, 0, None, 0, None, None, 0, 0, None,
) == -1
assert b"v2 invalid null/zero argument" in lib.atlas_gdn_c143_last_error()

lib.atlas_gdn_c143_launch_v3.argtypes = lib.atlas_gdn_c143_launch_v2.argtypes
lib.atlas_gdn_c143_launch_v3.restype = ctypes.c_int
assert lib.atlas_gdn_c143_launch_v3(
    None, None, 0, 0, 0, 0, 0, 0, None, 0, None, None, 0, 0, None,
) == -1
assert b"v3 invalid null/zero argument" in lib.atlas_gdn_c143_last_error()

# Invalid geometry and misalignment must fail before any CUDA call. Fake
# non-null addresses are therefore safe and prove validation ordering.
p = ctypes.c_void_p(0x1000)
stream = ctypes.c_void_p(0x2000)
need = lib.atlas_gdn_c143_workspace_size_v2(1)
assert lib.atlas_gdn_c143_launch_v2(
    p, p, 0, 2048, 4096, 2048, 2048, 10239,
    p, 96, p, p, need, 1, None,
) == -4
assert b"requires exact Qwen3.8 C1 production layout" in lib.atlas_gdn_c143_last_error()
assert lib.atlas_gdn_c143_launch_v2(
    p, p, 0, 2048, 4096, 10240, 10240, 10240,
    p, 96, p, p, need - 1, 1, None,
) == -3
assert b"workspace too small" in lib.atlas_gdn_c143_last_error()
assert lib.atlas_gdn_c143_launch_v2(
    p, p, 0, 2048, 4096, 10240, 10240, 10240,
    p, 95, p, p, need, 1, None,
) == -4
assert b"requires exact Qwen3.8 C1 production layout" in lib.atlas_gdn_c143_last_error()
assert lib.atlas_gdn_c143_launch_v2(
    p, p, (1 << 64) - 1, 2048, 4096, 10240, 10240, 10240,
    p, 96, p, p, need, 1, None,
) == -4
assert b"requires exact Qwen3.8 C1 production layout" in lib.atlas_gdn_c143_last_error()
assert lib.atlas_gdn_c143_launch_v2(
    ctypes.c_void_p(0x1001), p, 0, 2048, 4096,
    10240, 10240, 10240, p, 96, p, p, need, 1, None,
) == -5
assert b"misaligned pointer" in lib.atlas_gdn_c143_last_error()

need_v3 = lib.atlas_gdn_c143_workspace_size_v3(1)
assert lib.atlas_gdn_c143_launch_v3(
    p, p, 0, 2048, 4096, 10240, 10240, 10240,
    p, 96, p, p, need_v3, 1, None,
) == -1
assert b"v3 invalid null/zero argument" in lib.atlas_gdn_c143_last_error()
assert lib.atlas_gdn_c143_launch_v3(
    p, p, 0, 2048, 4096, 2048, 10240, 10240,
    p, 96, p, p, need_v3, 1, stream,
) == -4
assert b"v3 requires exact Qwen3.8 C1 production layout" in lib.atlas_gdn_c143_last_error()
assert lib.atlas_gdn_c143_launch_v3(
    p, p, 0, 2048, 4096, 10240, 10240, 10240,
    p, 96, p, p, need_v3 - 1, 1, stream,
) == -3
assert b"v3 workspace too small" in lib.atlas_gdn_c143_last_error()
assert lib.atlas_gdn_c143_launch_v3(
    ctypes.c_void_p(0x1001), p, 0, 2048, 4096,
    10240, 10240, 10240, p, 96, p, p, need_v3, 1, stream,
) == -5
assert b"v3 misaligned pointer" in lib.atlas_gdn_c143_last_error()
PY

rg -q 'query_base_offset.*key_base_offset.*value_base_offset' "$here/atlas_gdn_c143.h"
rg -q 'strides 10240/10240/10240' "$here/atlas_gdn_c143.h"
rg -q 'exact production stride is 96' "$here/atlas_gdn_c143.h"
rg -q 'gated_delta_rule_alpha_to_log_v2' "$here/src/gated_delta_rule_sglang_c143.cu"
rg -q 'isfinite\(alpha\).*alpha >= 0.0f' "$here/src/gated_delta_rule_sglang_c143.cu"
rg -q 'launch_production' "$here/gate.py"
rg -q 'launch_compact' "$here/gate.py"
rg -q 'compact_v1_equals_production_v2_exact' "$here/gate.py"
rg -q '"compact_v1_equals_production_v2_exact": cross_layout_exact' "$here/gate.py"
rg -q 'launch_production_v3' "$here/gate.py"
rg -q 'comparisons\[f"port_v3_vs_' "$here/gate.py"
rg -q 'performance_screen' "$here/gate.py"
rg -q 'receipt\["qualification"\] = "PASS"' "$here/gate.py"
rg -q 'qualification output already exists' "$here/gate.py"
rg -Fq 'args.output.open("x"' "$here/gate.py"
python3 - "$here/gate.py" <<'PY'
from pathlib import Path
import re
import sys

source = Path(sys.argv[1]).read_text(encoding="utf-8")
create = source.index("qualification_stream = torch.cuda.Stream()")
reject_default = source.index("if qualification_stream_handle == 0:", create)
receipt = source.index('"execution_stream": "explicit_non_default_pytorch_stream"', reject_default)
bind = source.index("with torch.cuda.stream(qualification_stream):", receipt)
verify = source.index(
    "torch.cuda.current_stream().cuda_stream) != qualification_stream_handle", bind
)
run = source.index("receipts = [", verify)
sync = source.index("qualification_stream.synchronize()", run)
assert create < reject_default < receipt < bind < verify < run < sync
diagnostic = source.index('"qualification": "FAIL"')
stage_view = source.index("def workspace_stage_views(")
v2_stages = source.index("production_stages = workspace_stage_views(", stage_view)
v3_stages = source.index("stages = workspace_stage_views(", v2_stages)
stage_evidence = source.index("stage_deterministic = {}", v3_stages)
hard_exit = source.index('raise SystemExit("hard correctness gate failed")', diagnostic)
performance_diagnostic = source.index('"failure": "hard_v3_performance_gate"', hard_exit)
performance_exit = source.index(
    'raise SystemExit("hard v3 performance gate failed")', performance_diagnostic
)
pass_receipt = source.index('receipt["qualification"] = "PASS"', performance_exit)
output_write = source.index('args.output.open("x"', pass_receipt)
assert (
    stage_view < v2_stages < v3_stages < stage_evidence < diagnostic < hard_exit
    < performance_diagnostic < performance_exit < pass_receipt < output_write
)
for field in ("w", "u", "gc", "h", "v_new"):
    assert f'"{field}": take(' in source
assert "cursor + expected_log_bytes != workspace.numel()" in source
PY
python3 - "$here/src/gated_delta_rule_sglang_c143.cu" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1]).read_text(encoding="utf-8")
start = source.index("gated_delta_rule_c143_v3_kkt_solve(")
end = source.index("gated_delta_rule_c143_v3_recompute_wu(", start)
kkt = source[start:end]
size_proof = kkt.index(
    "CHUNK * K_DIM * sizeof(__nv_bfloat16) ==\n"
    "                  CHUNK * CHUNK * sizeof(float)"
)
lower = kkt.index("float* lower = reinterpret_cast<float*>(sk);", size_proof)
gram = kkt.index(
    "float* gram = reinterpret_cast<float*>(sk + CHUNK * K_DIM);", lower
)
inverse = kkt.index("float* inverse = gram + CHUNK * CHUNK;", gram)
mma = kkt.index("mma_gram<8, CHUNK, false>(sk, sk, gram);", inverse)
barrier = kkt.index("__syncthreads();", mma)
gram_read = kkt.index("gram[j * CHUNK + i]", barrier)
lower_write = kkt.index("lower[idx] = value;", gram_read)
solve_read = kkt.index("lower[i * CHUNK + l]", lower_write)
assert size_proof < lower < gram < inverse < mma < barrier
assert barrier < gram_read < lower_write < solve_read
assert "lower[j * CHUNK + i]" not in kkt
PY
python3 - "$here/src/gated_delta_rule_sglang_c143.cu" <<'PY'
from pathlib import Path
import re
import sys

source = Path(sys.argv[1]).read_text(encoding="utf-8")
helper_start = source.index("void c143_v3_mma_qk_causal(")
helper_end = source.index("// One program per (chunk,value-head)", helper_start)
helper = source[helper_start:helper_end]
mma = helper.index("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32")
barrier = helper.index("__syncthreads();", mma)
causal_write = helper.index("causal[m0 * CHUNK + n0]", barrier)
assert mma < barrier < causal_write
assert "causal` aliases `key`" in helper
prefix = helper[:barrier]
suffix = helper[barrier:]
assert "causal[" not in prefix
assert "s_key[" not in suffix
assert helper.count("__syncthreads();") == 1
assert helper.count("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32") == 1
assert helper.count("for (unsigned nt = 0; nt < NTC; ++nt)") == 3

generic_start = source.index("template <int KRED")
generic_end = source.index("// QK-specialized sibling", generic_start)
generic = source[generic_start:generic_end]
generic_acc_start = generic.index("float acc[NTC][4];")
generic_acc_end = generic.index(
    "#pragma unroll\n    for (int nt = 0; nt < NTC; ++nt)",
    generic.index("for (unsigned ks = 0", generic_acc_start),
)
generic_acc = generic[generic_acc_start:generic_acc_end]
special_acc_start = helper.index("float acc[NTC][4];")
special_acc_end = helper.index("// `causal` aliases `key`", special_acc_start)
special_acc = helper[special_acc_start:special_acc_end]

def compact(text):
    return re.sub(r"\s+", "", text)

special_normalized = compact(special_acc)
special_normalized = special_normalized.replace("for(unsignednt", "for(intnt")
special_normalized = special_normalized.replace("ks<K_DIM", "ks<KRED")
for row in ("fr0", "fr1"):
    special_normalized = special_normalized.replace(
        f"s_query[{row}*K_DIM+", f"sA[{row}*ASTRIDE+"
    )
special_normalized = special_normalized.replace("s_key[nc*K_DIM+", "sB[nc*BSTRIDE+")
assert special_normalized == compact(generic_acc)

for exact in (
    "acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.0f;",
    "for (unsigned ks = 0; ks < K_DIM; ks += 16)",
    "for (unsigned nt = 0; nt < NTC; ++nt)",
    "const unsigned nc = nt * 8 + grp;",
    "s_query[fr0 * K_DIM + fc0]",
    "s_query[fr1 * K_DIM + fc0]",
    "s_query[fr0 * K_DIM + fc1]",
    "s_query[fr1 * K_DIM + fc1]",
    "s_key[nc * K_DIM + k0 + 1]",
    "s_key[nc * K_DIM + k0]",
    "s_key[nc * K_DIM + k1 + 1]",
    "s_key[nc * K_DIM + k1]",
    ': "=f"(acc[nt][0]), "=f"(acc[nt][1]), "=f"(acc[nt][2]),',
    '"=f"(acc[nt][3])',
    ': "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),',
    '"f"(acc[nt][0]), "f"(acc[nt][1]), "f"(acc[nt][2]),',
    '"f"(acc[nt][3])',
):
    assert exact in prefix, exact
assert (
    "const unsigned b0 =\n"
    "                (static_cast<unsigned>(s_key[nc * K_DIM + k0 + 1]) << 16) |\n"
    "                static_cast<unsigned>(s_key[nc * K_DIM + k0]);"
) in prefix
assert (
    "const unsigned b1 =\n"
    "                (static_cast<unsigned>(s_key[nc * K_DIM + k1 + 1]) << 16) |\n"
    "                static_cast<unsigned>(s_key[nc * K_DIM + k1]);"
) in prefix
for exact in (
    "gc[base * CHUNK + m0] - gc[base * CHUNK + n0]) *\n"
    "                        acc[nt][0]",
    "gc[base * CHUNK + m0] - gc[base * CHUNK + n1]) *\n"
    "                        acc[nt][1]",
    "gc[base * CHUNK + m1] - gc[base * CHUNK + n0]) *\n"
    "                        acc[nt][2]",
    "gc[base * CHUNK + m1] - gc[base * CHUNK + n1]) *\n"
    "                        acc[nt][3]",
    "causal[m0 * CHUNK + n0] = __float2bfloat16(values[0]);",
    "causal[m0 * CHUNK + n1] = __float2bfloat16(values[1]);",
    "causal[m1 * CHUNK + n0] = __float2bfloat16(values[2]);",
    "causal[m1 * CHUNK + n1] = __float2bfloat16(values[3]);",
):
    assert exact in suffix, exact
assert suffix.count("causal[") == 4

fwd_start = source.index("gated_delta_rule_c143_v3_fwd_o(")
fwd_end = source.index("namespace {", fwd_start)
fwd = source[fwd_start:fwd_end]
sq = fwd.index("__nv_bfloat16* sq = reinterpret_cast<__nv_bfloat16*>(raw);")
sh = fwd.index("__nv_bfloat16* sh = sq + CHUNK * K_DIM;", sq)
o_state = fwd.index(
    "float* o_state = reinterpret_cast<float*>(sh + CHUNK * K_DIM);", sh
)
qh_fill = fwd.index("sh[idx] = h_entries[", o_state)
barrier_before_qh = fwd.index("__syncthreads();", qh_fill)
qh = fwd.index(
    "c143_v3_mma_row_row<128, 8, 128, 128, BV, false>(sq, sh, o_state);",
    o_state,
)
barrier_after_qh = fwd.index("__syncthreads();", qh)
k_reload = fwd.index("sh[idx] = i < ce", qh)
barrier_after_k_reload = fwd.index("__syncthreads();", k_reload)
causal = fwd.index("__nv_bfloat16* causal = sh;", k_reload)
qk = fwd.index("c143_v3_mma_qk_causal(sq, sh, causal, gc, base, ce);", causal)
vt = fwd.index("__nv_bfloat16* vt = causal + CHUNK * CHUNK;", qk)
av = fwd.index("float* av = reinterpret_cast<float*>(sq);", vt)
vt_load = fwd.index("vt[idx] = j < ce", av)
barrier_after_vt = fwd.index("__syncthreads();", vt_load)
av_mma = fwd.index(
    "c143_v3_mma_row_row<64, 8, 64, 64, BV, false>(causal, vt, av);", av
)
barrier_after_av = fwd.index("__syncthreads();", av_mma)
output_read = fwd.index("o_state[idx] + av[idx]", barrier_after_av)
assert (
    sq < sh < o_state < qh_fill < barrier_before_qh < qh
    < barrier_after_qh < k_reload
    < barrier_after_k_reload < causal < qk < vt < av < vt_load
    < barrier_after_vt < av_mma < barrier_after_av < output_read
)
assert fwd.count("__syncthreads();") == 5
assert "float* qk" not in fwd
assert "c143_v3_mma_row_row<128, 8, 128, 128, CHUNK, false>" not in fwd
assert "constexpr unsigned int C143_V3_OUT_SMEM = 49152;" in source
PY
rg -q 'SGLANG_AUDITED_SHA256' "$here/gate.py"
rg -q 'PORT_ABI_PREFIX' "$here/gate.py"
rg -q 'gdn_source_sha' "$here/build.sh"
rg -q 'ATLAS_BRIDGE_ABI_PREFIX' "$here/gate.py"
rg -q 'atlas_bridge_sha' "$here/build.sh"
rg -q '"--porcelain"' "$here/gate.py"
rg -q '"--untracked-files=no"' "$here/gate.py"
! rg -q '^from sglang\.' "$here/gate.py"
rg -Fq 'dim3(4, C143_VALUE_HEADS, 1)' "$here/src/gated_delta_rule_sglang_c143.cu"
rg -Fq 'dim3(2, nt, C143_VALUE_HEADS)' "$here/src/gated_delta_rule_sglang_c143.cu"
rg -Fq 'constexpr unsigned int C143_V3_KKT_SMEM = 49408;' "$here/src/gated_delta_rule_sglang_c143.cu"
rg -Fq 'constexpr unsigned int C143_V3_OUT_SMEM = 49152;' "$here/src/gated_delta_rule_sglang_c143.cu"
rg -Fq 'c143_v3_mma_row_row<128, 4' "$here/src/gated_delta_rule_sglang_c143.cu"
rg -Fq 'c143_v3_mma_row_row<64, 8' "$here/src/gated_delta_rule_sglang_c143.cu"
cc -std=c11 -Wall -Wextra -Werror -fsyntax-only -x c "$here/atlas_gdn_c143.h"
if rg -n 'cuda(Malloc|Free|DeviceSynchronize|StreamSynchronize|Graph)' \
    "$here/src/gated_delta_rule_sglang_c143.cu"; then
    echo "forbidden allocation/synchronization/capture call in source" >&2
    exit 4
fi
