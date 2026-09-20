#!/usr/bin/env python3
import argparse
import ctypes
import hashlib
import importlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

import torch

SGLANG_ROOT = Path("/tmp/sglang-c14312a66420b75ca9a11bf1817c4db1fa26b097")
SGLANG_COMMIT = "c14312a66420b75ca9a11bf1817c4db1fa26b097"
SGLANG_AUDITED_SHA256 = {
    "LICENSE": "1495e1e757ef4d0925a2350563cf5754bb23c51701a8ec4fb3c5cdcbedae6747",
    "python/sglang/kernels/ops/attention/fla/chunk.py": (
        "8edab1f6fc35b86300a91dc6afd61c2456bd7a4ed3986564456977fdb098f2b2"
    ),
    "python/sglang/kernels/ops/attention/fla/chunk_fwd.py": (
        "e6ee7b4601ca12ccda6fd93050acedae25d2b6e6a27a27ebf194a58533a4140c"
    ),
    "python/sglang/kernels/ops/attention/fla/chunk_delta_h.py": (
        "580a24d2e91c885ef180f5135978c3cc35f01e96a17776baa4b13fe06533bb60"
    ),
    "python/sglang/kernels/ops/attention/fla/chunk_o.py": (
        "c5e5b0f7ccdaa744c5e0eede8ec73a5767b322132a72ce46a56f04bfe4c07564"
    ),
}
ATLAS_SOURCE = Path(
    "/home/flocka/atlas/src/kernels/gb10/common/gated_delta_rule_wy32_gatecache.cu"
)
ATLAS_SHA256 = "f73a7071aa8160960e9221bb584caa1e9f733856c0d82bcc39085e9e881d6aed"
HERE = Path(__file__).resolve().parent
HK, HV, K, V, CHUNK = 16, 48, 128, 128, 64
Q_WIDTH, K_WIDTH, V_WIDTH = HK * K, HK * K, HV * V
Q_OFFSET, K_OFFSET, V_OFFSET = 0, Q_WIDTH, Q_WIDTH + K_WIDTH
QKV_ROW_STRIDE = Q_WIDTH + K_WIDTH + V_WIDTH
GATE_BETA_ROW_STRIDE = HV * 2
MIN_COSINE = 0.999
MAX_RELATIVE_RMS = 0.01
PORT_ABI_PREFIX = "atlas-gdn-c143-abi-v3:qwen38-c1-b1-workspace-v3:"
ATLAS_BRIDGE_ABI_PREFIX = "atlas-gdn-wy32-bridge-v1:"


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_pinned_sglang():
    commit = subprocess.check_output(
        ["git", "-C", str(SGLANG_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != SGLANG_COMMIT:
        raise SystemExit(f"SGLang commit drift: {commit}")
    tracked_drift = subprocess.check_output(
        [
            "git",
            "-C",
            str(SGLANG_ROOT),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        text=True,
    )
    if tracked_drift:
        raise SystemExit("SGLang tracked worktree drift")
    audited = {
        relative: file_sha(SGLANG_ROOT / relative) for relative in SGLANG_AUDITED_SHA256
    }
    if audited != SGLANG_AUDITED_SHA256:
        raise SystemExit("SGLang audited source hash drift")
    expected_root = (SGLANG_ROOT / "python").resolve()
    sys.path.insert(0, str(expected_root))
    module = importlib.import_module("sglang.kernels.ops.attention.fla.chunk")
    module_path = Path(module.__file__).resolve()
    if not module_path.is_relative_to(expected_root):
        raise SystemExit(f"SGLang import escaped pinned tree: {module_path}")
    for name, loaded in sys.modules.items():
        if name.startswith("sglang.kernels.ops.attention.fla."):
            loaded_path = getattr(loaded, "__file__", None)
            if loaded_path is not None and not Path(
                loaded_path
            ).resolve().is_relative_to(expected_root):
                raise SystemExit(f"SGLang dependency escaped pinned tree: {name}")
    return commit, audited, module.chunk_gated_delta_rule


def tensor_sha(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def timing_summary(values):
    ordered = sorted(values)
    return {
        "median_ms": statistics.median(values),
        "p90_ms": ordered[math.ceil(0.9 * len(ordered)) - 1],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "reps": len(values),
    }


def evaluate_performance_screen(m, times, timing, position_counts):
    required_arms = {"atlas", "port", "port_v3", "sglang"}
    if set(times) != required_arms or set(position_counts) != required_arms:
        return {"pass": False, "reason": "timing arm set drift"}
    sample_counts = {name: len(values) for name, values in times.items()}
    if len(set(sample_counts.values())) != 1 or next(iter(sample_counts.values())) < 11:
        return {"pass": False, "reason": "timing sample count drift"}
    balanced_positions = all(
        sum(counts) == sample_counts[name] and max(counts) - min(counts) <= 1
        for name, counts in position_counts.items()
    )
    limit_ms = 2.5 if m == 2079 else 10.0
    candidate_ms = timing["port_v3"]["median_ms"]
    candidate_p90_ms = timing["port_v3"]["p90_ms"]
    atlas_ms = timing["atlas"]["median_ms"]
    atlas_p90_ms = timing["atlas"]["p90_ms"]
    sglang_ms = timing["sglang"]["median_ms"]
    paired_atlas_gain_ms = statistics.median(
        atlas - candidate for atlas, candidate in zip(times["atlas"], times["port_v3"])
    )
    sglang_ratio = candidate_ms / max(sglang_ms, 1e-30)
    absolute_pass = candidate_ms < limit_ms
    atlas_p90_pass = candidate_p90_ms < atlas_p90_ms
    atlas_pass = (
        candidate_ms < atlas_ms and atlas_p90_pass and paired_atlas_gain_ms > 0.0
    )
    sglang_pass = sglang_ratio <= 1.20
    return {
        "port_v3_median_limit_ms": limit_ms,
        "port_v3_median_ms": candidate_ms,
        "port_v3_p90_ms": candidate_p90_ms,
        "atlas_median_ms": atlas_ms,
        "atlas_p90_ms": atlas_p90_ms,
        "sglang_median_ms": sglang_ms,
        "paired_atlas_gain_median_ms": paired_atlas_gain_ms,
        "port_v3_to_sglang_median_ratio": sglang_ratio,
        "sglang_ratio_limit": 1.20,
        "position_counts": position_counts,
        "balanced_positions": balanced_positions,
        "absolute_pass": absolute_pass,
        "atlas_comparative_pass": atlas_pass,
        "atlas_p90_comparative_pass": atlas_p90_pass,
        "sglang_comparative_pass": sglang_pass,
        "pass": absolute_pass and atlas_pass and sglang_pass and balanced_positions,
        "scope": "isolated default-unrouted candidate only",
    }


def metrics(candidate: torch.Tensor, reference: torch.Tensor):
    a = candidate.float().reshape(-1).double()
    b = reference.float().reshape(-1).double()
    diff = a - b
    dot = torch.dot(a, b).item()
    an = torch.linalg.vector_norm(a).item()
    bn = torch.linalg.vector_norm(b).item()
    rms = torch.sqrt(torch.mean(diff * diff)).item()
    ref_rms = torch.sqrt(torch.mean(b * b)).item()
    return {
        "max_abs": torch.max(torch.abs(diff)).item(),
        "rms": rms,
        "relative_rms": rms / max(ref_rms, 1e-30),
        "cosine": dot / max(an * bn, 1e-30),
    }


def workspace_stage_views(workspace: torch.Tensor, m: int):
    if (
        workspace.dtype != torch.uint8
        or workspace.ndim != 1
        or not workspace.is_contiguous()
    ):
        raise RuntimeError("workspace stage view requires contiguous CUDA bytes")
    programs = ((m + CHUNK - 1) // CHUNK) * HV
    cursor = 0

    def take(dtype, shape):
        nonlocal cursor
        elements = math.prod(shape)
        item_bytes = torch.empty((), dtype=dtype).element_size()
        byte_count = elements * item_bytes
        if cursor + byte_count > workspace.numel():
            raise RuntimeError("workspace stage layout exceeds allocation")
        view = workspace.narrow(0, cursor, byte_count).view(dtype).view(shape)
        cursor += byte_count
        return view

    stages = {
        "w": take(torch.bfloat16, (programs, CHUNK, K)),
        "u": take(torch.bfloat16, (programs, CHUNK, V)),
        "gc": take(torch.float32, (programs, CHUNK)),
        "h": take(torch.bfloat16, (programs, V, K)),
        "v_new": take(torch.bfloat16, (programs, CHUNK, V)),
    }
    expected_log_bytes = m * HV * torch.empty((), dtype=torch.float32).element_size()
    if cursor + expected_log_bytes != workspace.numel():
        raise RuntimeError(
            "workspace stage/log extent does not exactly cover allocation"
        )
    return stages


class AtlasBridge:
    def __init__(self, path: Path):
        self.lib = ctypes.CDLL(str(path), mode=ctypes.RTLD_LOCAL)
        self.lib.atlas_gdn_wy32_abi_identity.restype = ctypes.c_char_p
        identity = self.lib.atlas_gdn_wy32_abi_identity().decode("ascii", "strict")
        expected_identity = (
            ATLAS_BRIDGE_ABI_PREFIX
            + file_sha(ATLAS_SOURCE)
            + ":"
            + file_sha(HERE / "src/atlas_wy32_bridge.cu")
        )
        if identity != expected_identity:
            raise RuntimeError(
                "Atlas loaded-bridge ABI/source identity mismatch: "
                f"loaded={identity!r} expected={expected_identity!r}"
            )
        self.abi_identity = identity
        self.lib.atlas_gdn_wy32_launch.argtypes = [ctypes.c_void_p] * 7 + [
            ctypes.c_uint,
            ctypes.c_void_p,
        ]
        self.lib.atlas_gdn_wy32_launch.restype = ctypes.c_int
        self.lib.atlas_gdn_wy32_last_error.restype = ctypes.c_char_p

    def launch(self, state, q, k, v, alpha, beta, output):
        tensors = (state, q, k, v, alpha, beta, output)
        if not all(x.is_cuda and x.is_contiguous() for x in tensors):
            raise RuntimeError("Atlas inputs must be contiguous CUDA tensors")
        stream = torch.cuda.current_stream().cuda_stream
        status = self.lib.atlas_gdn_wy32_launch(
            *(ctypes.c_void_p(x.data_ptr()) for x in tensors),
            ctypes.c_uint(q.shape[1]),
            ctypes.c_void_p(stream),
        )
        if status:
            raise RuntimeError(
                self.lib.atlas_gdn_wy32_last_error().decode("utf-8", "replace")
            )


class PortBridge:
    def __init__(self, path: Path):
        self.lib = ctypes.CDLL(str(path), mode=ctypes.RTLD_LOCAL)
        self.lib.atlas_gdn_c143_abi_identity.restype = ctypes.c_char_p
        identity = self.lib.atlas_gdn_c143_abi_identity().decode("ascii", "strict")
        expected_identity = PORT_ABI_PREFIX + file_sha(
            HERE / "src/gated_delta_rule_sglang_c143.cu"
        )
        if identity != expected_identity:
            raise RuntimeError(
                "c143 loaded-library ABI/source identity mismatch: "
                f"loaded={identity!r} expected={expected_identity!r}"
            )
        self.abi_identity = identity
        self.lib.atlas_gdn_c143_workspace_size.argtypes = [ctypes.c_uint]
        self.lib.atlas_gdn_c143_workspace_size.restype = ctypes.c_size_t
        self.lib.atlas_gdn_c143_workspace_size_v2.argtypes = [ctypes.c_uint]
        self.lib.atlas_gdn_c143_workspace_size_v2.restype = ctypes.c_size_t
        self.lib.atlas_gdn_c143_workspace_size_v3.argtypes = [ctypes.c_uint]
        self.lib.atlas_gdn_c143_workspace_size_v3.restype = ctypes.c_size_t
        self.lib.atlas_gdn_c143_launch.argtypes = [ctypes.c_void_p] * 8 + [
            ctypes.c_size_t,
            ctypes.c_uint,
            ctypes.c_void_p,
        ]
        self.lib.atlas_gdn_c143_launch.restype = ctypes.c_int
        self.lib.atlas_gdn_c143_launch_v2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_size_t,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_uint,
            ctypes.c_void_p,
        ]
        self.lib.atlas_gdn_c143_launch_v2.restype = ctypes.c_int
        self.lib.atlas_gdn_c143_launch_v3.argtypes = (
            self.lib.atlas_gdn_c143_launch_v2.argtypes
        )
        self.lib.atlas_gdn_c143_launch_v3.restype = ctypes.c_int
        self.lib.atlas_gdn_c143_last_error.restype = ctypes.c_char_p

    def workspace_bytes(self, m, *, production=False, v3=False):
        if v3:
            query = self.lib.atlas_gdn_c143_workspace_size_v3
        else:
            query = (
                self.lib.atlas_gdn_c143_workspace_size_v2
                if production
                else self.lib.atlas_gdn_c143_workspace_size
            )
        value = query(ctypes.c_uint(m))
        if value <= 0:
            raise RuntimeError("c143 workspace query returned zero")
        return value

    def launch_compact(self, state, q, k, v, log_gate, beta, output, workspace):
        tensors = (state, q, k, v, log_gate, beta, output, workspace)
        if not all(x.is_cuda and x.is_contiguous() for x in tensors):
            raise RuntimeError("c143 inputs must be contiguous CUDA tensors")
        stream = torch.cuda.current_stream().cuda_stream
        status = self.lib.atlas_gdn_c143_launch(
            *(ctypes.c_void_p(x.data_ptr()) for x in tensors),
            ctypes.c_size_t(workspace.numel()),
            ctypes.c_uint(q.shape[1]),
            ctypes.c_void_p(stream),
        )
        if status:
            raise RuntimeError(
                self.lib.atlas_gdn_c143_last_error().decode("utf-8", "replace")
            )

    def launch_production(self, state, qkv, gate_beta, output, workspace):
        tensors = (state, qkv, gate_beta, output, workspace)
        if not all(x.is_cuda and x.is_contiguous() for x in tensors):
            raise RuntimeError("c143 v2 inputs must be contiguous CUDA tensors")
        if (
            qkv.shape[-1] != QKV_ROW_STRIDE
            or gate_beta.shape[-1] != GATE_BETA_ROW_STRIDE
        ):
            raise RuntimeError("c143 v2 production layout drift")
        stream = torch.cuda.current_stream().cuda_stream
        status = self.lib.atlas_gdn_c143_launch_v2(
            ctypes.c_void_p(state.data_ptr()),
            ctypes.c_void_p(qkv.data_ptr()),
            ctypes.c_size_t(Q_OFFSET),
            ctypes.c_size_t(K_OFFSET),
            ctypes.c_size_t(V_OFFSET),
            ctypes.c_uint(QKV_ROW_STRIDE),
            ctypes.c_uint(QKV_ROW_STRIDE),
            ctypes.c_uint(QKV_ROW_STRIDE),
            ctypes.c_void_p(gate_beta.data_ptr()),
            ctypes.c_uint(GATE_BETA_ROW_STRIDE),
            ctypes.c_void_p(output.data_ptr()),
            ctypes.c_void_p(workspace.data_ptr()),
            ctypes.c_size_t(workspace.numel()),
            ctypes.c_uint(qkv.shape[1]),
            ctypes.c_void_p(stream),
        )
        if status:
            raise RuntimeError(
                self.lib.atlas_gdn_c143_last_error().decode("utf-8", "replace")
            )

    def launch_production_v3(self, state, qkv, gate_beta, output, workspace):
        tensors = (state, qkv, gate_beta, output, workspace)
        if not all(x.is_cuda and x.is_contiguous() for x in tensors):
            raise RuntimeError("c143 v3 inputs must be contiguous CUDA tensors")
        if (
            qkv.shape[-1] != QKV_ROW_STRIDE
            or gate_beta.shape[-1] != GATE_BETA_ROW_STRIDE
        ):
            raise RuntimeError("c143 v3 production layout drift")
        stream = torch.cuda.current_stream().cuda_stream
        status = self.lib.atlas_gdn_c143_launch_v3(
            ctypes.c_void_p(state.data_ptr()),
            ctypes.c_void_p(qkv.data_ptr()),
            ctypes.c_size_t(Q_OFFSET),
            ctypes.c_size_t(K_OFFSET),
            ctypes.c_size_t(V_OFFSET),
            ctypes.c_uint(QKV_ROW_STRIDE),
            ctypes.c_uint(QKV_ROW_STRIDE),
            ctypes.c_uint(QKV_ROW_STRIDE),
            ctypes.c_void_p(gate_beta.data_ptr()),
            ctypes.c_uint(GATE_BETA_ROW_STRIDE),
            ctypes.c_void_p(output.data_ptr()),
            ctypes.c_void_p(workspace.data_ptr()),
            ctypes.c_size_t(workspace.numel()),
            ctypes.c_uint(qkv.shape[1]),
            ctypes.c_void_p(stream),
        )
        if status:
            raise RuntimeError(
                self.lib.atlas_gdn_c143_last_error().decode("utf-8", "replace")
            )


CANARY_BYTES = 4096


def guarded_tensor(source: torch.Tensor, prefix: int, suffix: int):
    payload_bytes = source.numel() * source.element_size()
    raw = torch.empty(
        CANARY_BYTES + payload_bytes + CANARY_BYTES, dtype=torch.uint8, device="cuda"
    )
    raw[:CANARY_BYTES].fill_(prefix)
    raw[CANARY_BYTES + payload_bytes :].fill_(suffix)
    payload = (
        raw[CANARY_BYTES : CANARY_BYTES + payload_bytes]
        .view(source.dtype)
        .view(source.shape)
    )
    payload.copy_(source)
    return payload, raw, prefix, suffix


def guarded_empty(shape, dtype, fill: int, prefix: int, suffix: int):
    source = torch.empty(shape, dtype=dtype, device="cuda")
    source.view(torch.uint8).fill_(fill)
    return guarded_tensor(source, prefix, suffix)


def assert_canary(item):
    payload, raw, prefix, suffix = item
    payload_bytes = payload.numel() * payload.element_size()
    if not bool((raw[:CANARY_BYTES] == prefix).all().item()):
        raise RuntimeError("prefix canary corrupted")
    if not bool((raw[CANARY_BYTES + payload_bytes :] == suffix).all().item()):
        raise RuntimeError("suffix canary corrupted")


def make_inputs(m: int):
    generator = torch.Generator(device="cuda")
    generator.manual_seed(0x38C143 + m)
    q_raw = torch.randn(
        (1, m, HK, K), generator=generator, device="cuda", dtype=torch.bfloat16
    )
    k_raw = torch.randn(
        (1, m, HK, K), generator=generator, device="cuda", dtype=torch.bfloat16
    )
    # Both arms consume the exact same post-QK-normalization BF16 bytes.
    q = (
        torch.nn.functional.normalize(q_raw.float(), dim=-1)
        .to(torch.bfloat16)
        .contiguous()
    )
    k = (
        torch.nn.functional.normalize(k_raw.float(), dim=-1)
        .to(torch.bfloat16)
        .contiguous()
    )
    v = (
        torch.randn(
            (1, m, HV, V), generator=generator, device="cuda", dtype=torch.bfloat16
        )
        * 0.125
    ).contiguous()
    # Checkpoint-like construction: g_log=-exp(A_log)*softplus(a+dt_bias), beta=sigmoid(b).
    a_log = torch.linspace(-4.5, -2.0, HV, device="cuda", dtype=torch.float32)
    dt_bias = torch.linspace(-3.0, 0.5, HV, device="cuda", dtype=torch.float32)
    a_raw = torch.randn(
        (1, m, HV), generator=generator, device="cuda", dtype=torch.float32
    )
    b_raw = torch.randn(
        (1, m, HV), generator=generator, device="cuda", dtype=torch.float32
    )
    log_gate = (
        -torch.exp(a_log) * torch.nn.functional.softplus(a_raw + dt_bias)
    ).contiguous()
    beta = torch.sigmoid(b_raw).contiguous()
    alpha = torch.exp(log_gate).contiguous()
    # Canonical state is SGLang's documented [N,H,V,K]; Atlas receives [H,K,V].
    state_v_k = (
        torch.randn(
            (1, HV, V, K), generator=generator, device="cuda", dtype=torch.float32
        )
        * 0.01
    ).contiguous()
    return q, k, v, log_gate, alpha, beta, state_v_k


def run_case(
    atlas_bridge: AtlasBridge,
    port_bridge: PortBridge,
    chunk_gated_delta_rule,
    m: int,
    warmup: int,
    reps: int,
):
    names = ("q", "k", "v", "log_gate", "alpha", "beta", "initial_v_k")
    input_guards = {}
    input_tensors = {}
    for index, (name, source) in enumerate(zip(names, make_inputs(m))):
        guard = guarded_tensor(source, 0x20 + index, 0xD0 + index)
        input_guards[name] = guard
        input_tensors[name] = guard[0]
    cu_guard = guarded_tensor(
        torch.tensor([0, m], dtype=torch.int32, device="cuda"), 0x31, 0xE1
    )
    index_guard = guarded_tensor(
        torch.tensor([0], dtype=torch.int32, device="cuda"), 0x32, 0xE2
    )
    input_guards["cu_seqlens"] = cu_guard
    input_guards["state_index"] = index_guard
    input_tensors["cu_seqlens"] = cu_guard[0]
    input_tensors["state_index"] = index_guard[0]
    q, k, v = (input_tensors[name] for name in ("q", "k", "v"))
    log_gate, alpha, beta = (
        input_tensors[name] for name in ("log_gate", "alpha", "beta")
    )
    initial_v_k = input_tensors["initial_v_k"]
    cu, state_index = input_tensors["cu_seqlens"], input_tensors["state_index"]

    production_qkv_source = torch.empty(
        (1, m, QKV_ROW_STRIDE), dtype=torch.bfloat16, device="cuda"
    )
    production_qkv_source[:, :, Q_OFFSET : Q_OFFSET + Q_WIDTH].copy_(
        q.reshape(1, m, Q_WIDTH)
    )
    production_qkv_source[:, :, K_OFFSET : K_OFFSET + K_WIDTH].copy_(
        k.reshape(1, m, K_WIDTH)
    )
    production_qkv_source[:, :, V_OFFSET : V_OFFSET + V_WIDTH].copy_(
        v.reshape(1, m, V_WIDTH)
    )
    production_qkv_guard = guarded_tensor(production_qkv_source, 0x33, 0xE3)
    production_gate_beta_source = torch.empty(
        (1, m, GATE_BETA_ROW_STRIDE), dtype=torch.float32, device="cuda"
    )
    production_gate_beta_source[:, :, :HV].copy_(alpha)
    production_gate_beta_source[:, :, HV:].copy_(beta)
    production_gate_beta_guard = guarded_tensor(production_gate_beta_source, 0x34, 0xE4)
    input_guards["production_qkv"] = production_qkv_guard
    input_guards["production_gate_beta"] = production_gate_beta_guard
    input_tensors["production_qkv"] = production_qkv_guard[0]
    input_tensors["production_gate_beta"] = production_gate_beta_guard[0]
    production_qkv = input_tensors["production_qkv"]
    production_gate_beta = input_tensors["production_gate_beta"]
    before = {name: tensor_sha(value) for name, value in input_tensors.items()}

    def atlas_once(fill=0x61):
        state_source = initial_v_k.transpose(-1, -2).reshape(HV, K, V).contiguous()
        state_guard = guarded_tensor(state_source, 0x41, 0xC1)
        output_guard = guarded_empty((1, m, HV, V), torch.bfloat16, fill, 0x42, 0xC2)
        atlas_bridge.launch(state_guard[0], q, k, v, alpha, beta, output_guard[0])
        state_v_k = state_guard[0].reshape(1, HV, K, V).transpose(-1, -2).contiguous()
        return {
            "output": output_guard[0],
            "state": state_v_k,
            "guards": (state_guard, output_guard),
        }

    def port_once(fill=0x71):
        state_source = initial_v_k.transpose(-1, -2).reshape(HV, K, V).contiguous()
        production_state_guard = guarded_tensor(state_source, 0x51, 0xB1)
        production_output_guard = guarded_empty(
            (1, m, HV, V), torch.bfloat16, fill, 0x52, 0xB2
        )
        production_workspace_guard = guarded_empty(
            (port_bridge.workspace_bytes(m, production=True),),
            torch.uint8,
            fill ^ 0xFF,
            0x53,
            0xB3,
        )
        port_bridge.launch_production(
            production_state_guard[0],
            production_qkv,
            production_gate_beta,
            production_output_guard[0],
            production_workspace_guard[0],
        )
        production_stages = workspace_stage_views(production_workspace_guard[0], m)

        # v2's compact log-decay tail is the exact conversion source for the
        # compact-v1 cross-layout oracle. Same-stream order makes it visible to
        # v1 without a host synchronization or a second numerical conversion.
        compact_workspace_bytes = port_bridge.workspace_bytes(m)
        converted_log_gate = (
            production_workspace_guard[0][
                compact_workspace_bytes : compact_workspace_bytes + m * HV * 4
            ]
            .view(torch.float32)
            .view(1, m, HV)
        )
        compact_state_guard = guarded_tensor(state_source, 0x54, 0xB4)
        compact_output_guard = guarded_empty(
            (1, m, HV, V), torch.bfloat16, fill, 0x55, 0xB5
        )
        compact_workspace_guard = guarded_empty(
            (compact_workspace_bytes,), torch.uint8, fill ^ 0xAA, 0x56, 0xB6
        )
        port_bridge.launch_compact(
            compact_state_guard[0],
            q,
            k,
            v,
            converted_log_gate,
            beta,
            compact_output_guard[0],
            compact_workspace_guard[0],
        )
        production_state = (
            production_state_guard[0]
            .reshape(1, HV, K, V)
            .transpose(-1, -2)
            .contiguous()
        )
        compact_state = (
            compact_state_guard[0].reshape(1, HV, K, V).transpose(-1, -2).contiguous()
        )
        return {
            "output": production_output_guard[0],
            "state": production_state,
            "compact_output": compact_output_guard[0],
            "compact_state": compact_state,
            "converted_log_gate": converted_log_gate,
            "stages": production_stages,
            "guards": (
                production_state_guard,
                production_output_guard,
                production_workspace_guard,
                compact_state_guard,
                compact_output_guard,
                compact_workspace_guard,
            ),
        }

    def port_v3_once(fill=0x81):
        state_source = initial_v_k.transpose(-1, -2).reshape(HV, K, V).contiguous()
        state_guard = guarded_tensor(state_source, 0x57, 0xB7)
        output_guard = guarded_empty((1, m, HV, V), torch.bfloat16, fill, 0x58, 0xB8)
        workspace_bytes = port_bridge.workspace_bytes(m, v3=True)
        workspace_guard = guarded_empty(
            (workspace_bytes,), torch.uint8, fill ^ 0xCC, 0x59, 0xB9
        )
        port_bridge.launch_production_v3(
            state_guard[0],
            production_qkv,
            production_gate_beta,
            output_guard[0],
            workspace_guard[0],
        )
        stages = workspace_stage_views(workspace_guard[0], m)
        state = state_guard[0].reshape(1, HV, K, V).transpose(-1, -2).contiguous()
        log_bytes = m * HV * 4
        converted_log_gate = (
            workspace_guard[0][workspace_bytes - log_bytes : workspace_bytes]
            .view(torch.float32)
            .view(1, m, HV)
        )
        return {
            "output": output_guard[0],
            "state": state,
            "converted_log_gate": converted_log_gate,
            "stages": stages,
            "guards": (state_guard, output_guard, workspace_guard),
        }

    def sglang_once(_fill=0):
        state_guard = guarded_tensor(initial_v_k, 0x61, 0xA1)
        output, _, _ = chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=log_gate,
            beta=beta,
            initial_state=state_guard[0],
            initial_state_indices=state_index,
            cu_seqlens=cu,
            head_first=False,
            use_qk_l2norm_in_kernel=False,
        )
        return {"output": output, "state": state_guard[0], "guards": (state_guard,)}

    arms = {
        "atlas": atlas_once,
        "port": port_once,
        "port_v3": port_v3_once,
        "sglang": sglang_once,
    }
    for order in (
        ("atlas", "port", "port_v3", "sglang"),
        ("sglang", "port_v3", "port", "atlas"),
    ):
        for _ in range(warmup):
            warm_results = [arms[name]() for name in order]
            torch.cuda.synchronize()
            for result in warm_results:
                for guard in result["guards"]:
                    assert_canary(guard)

    first = {
        name: invoke(0x60 + index) for index, (name, invoke) in enumerate(arms.items())
    }
    torch.cuda.synchronize()
    for result in first.values():
        for guard in result["guards"]:
            assert_canary(guard)
    second = {
        name: invoke(0x90 + index) for index, (name, invoke) in enumerate(arms.items())
    }
    torch.cuda.synchronize()
    for result in second.values():
        for guard in result["guards"]:
            assert_canary(guard)

    hashes = {}
    deterministic = True
    finite = {}
    for name in arms:
        fields = (
            ("output", "state", "compact_output", "compact_state")
            if name == "port"
            else ("output", "state")
        )
        for field in fields:
            digest = tensor_sha(first[name][field])
            hashes[f"{name}_{field}"] = digest
            deterministic &= digest == tensor_sha(second[name][field])
            finite[f"{name}_{field}"] = bool(
                torch.isfinite(first[name][field]).all().item()
            )
    cross_layout_exact = all(
        hashes[f"port_{field}"] == hashes[f"port_compact_{field}"]
        for field in ("output", "state")
    )
    hashes["port_converted_log_gate"] = tensor_sha(first["port"]["converted_log_gate"])
    hashes["port_v3_converted_log_gate"] = tensor_sha(
        first["port_v3"]["converted_log_gate"]
    )
    converted_log_deterministic = hashes["port_converted_log_gate"] == tensor_sha(
        second["port"]["converted_log_gate"]
    ) and hashes["port_v3_converted_log_gate"] == tensor_sha(
        second["port_v3"]["converted_log_gate"]
    )
    converted_log_cross_version_exact = (
        hashes["port_converted_log_gate"] == hashes["port_v3_converted_log_gate"]
    )
    finite["port_converted_log_gate"] = bool(
        torch.isfinite(first["port"]["converted_log_gate"]).all().item()
    )
    finite["port_v3_converted_log_gate"] = bool(
        torch.isfinite(first["port_v3"]["converted_log_gate"]).all().item()
    )
    deterministic &= converted_log_deterministic and converted_log_cross_version_exact
    after = {name: tensor_sha(value) for name, value in input_tensors.items()}
    immutable = before == after
    for guard in input_guards.values():
        assert_canary(guard)

    timed_state_source = initial_v_k.transpose(-1, -2).reshape(HV, K, V).contiguous()
    timed_atlas_state = guarded_tensor(timed_state_source, 0x71, 0x91)
    timed_atlas_output = guarded_empty((1, m, HV, V), torch.bfloat16, 0x11, 0x72, 0x92)
    timed_port_state = guarded_tensor(timed_state_source, 0x73, 0x93)
    timed_port_output = guarded_empty((1, m, HV, V), torch.bfloat16, 0x12, 0x74, 0x94)
    timed_port_workspace = guarded_empty(
        (port_bridge.workspace_bytes(m, production=True),),
        torch.uint8,
        0x5A,
        0x75,
        0x95,
    )
    timed_v3_state = guarded_tensor(timed_state_source, 0x76, 0x96)
    timed_v3_output = guarded_empty((1, m, HV, V), torch.bfloat16, 0x13, 0x77, 0x97)
    timed_v3_workspace = guarded_empty(
        (port_bridge.workspace_bytes(m, v3=True),),
        torch.uint8,
        0x6A,
        0x78,
        0x98,
    )
    timed_sglang_state = guarded_tensor(initial_v_k, 0x79, 0x99)
    timed_guards = {
        "atlas": (timed_atlas_state, timed_atlas_output),
        "port": (timed_port_state, timed_port_output, timed_port_workspace),
        "port_v3": (timed_v3_state, timed_v3_output, timed_v3_workspace),
        "sglang": (timed_sglang_state,),
    }

    def prepare_timed(name, fill):
        if name == "atlas":
            timed_atlas_state[0].copy_(timed_state_source)
            timed_atlas_output[0].view(torch.uint8).fill_(fill)
        elif name == "port":
            timed_port_state[0].copy_(timed_state_source)
            timed_port_output[0].view(torch.uint8).fill_(fill)
        elif name == "port_v3":
            timed_v3_state[0].copy_(timed_state_source)
            timed_v3_output[0].view(torch.uint8).fill_(fill)
        else:
            timed_sglang_state[0].copy_(initial_v_k)

    def invoke_timed(name):
        if name == "atlas":
            atlas_bridge.launch(
                timed_atlas_state[0], q, k, v, alpha, beta, timed_atlas_output[0]
            )
            return timed_atlas_output[0]
        if name == "port":
            port_bridge.launch_production(
                timed_port_state[0],
                production_qkv,
                production_gate_beta,
                timed_port_output[0],
                timed_port_workspace[0],
            )
            return timed_port_output[0]
        if name == "port_v3":
            port_bridge.launch_production_v3(
                timed_v3_state[0],
                production_qkv,
                production_gate_beta,
                timed_v3_output[0],
                timed_v3_workspace[0],
            )
            return timed_v3_output[0]
        output, _, _ = chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=log_gate,
            beta=beta,
            initial_state=timed_sglang_state[0],
            initial_state_indices=state_index,
            cu_seqlens=cu,
            head_first=False,
            use_qk_l2norm_in_kernel=False,
        )
        return output

    times = {name: [] for name in arms}
    base_order = ("atlas", "port", "port_v3", "sglang")
    position_counts = {name: [0] * len(base_order) for name in base_order}
    for index in range(reps):
        shift = index % len(base_order)
        order = base_order[shift:] + base_order[:shift]
        for position, name in enumerate(order):
            position_counts[name][position] += 1
            prepare_timed(name, 0x30 + index)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            timed_output = invoke_timed(name)
            end.record()
            torch.cuda.synchronize()
            times[name].append(start.elapsed_time(end))
            if not bool(torch.isfinite(timed_output).all().item()):
                raise RuntimeError(f"{name} timed output is non-finite")
            for guard in timed_guards[name]:
                assert_canary(guard)

    comparisons = {}
    for reference in ("atlas", "sglang"):
        comparisons[f"port_vs_{reference}"] = {
            field: metrics(first["port"][field], first[reference][field])
            for field in ("output", "state")
        }
    for reference in ("atlas", "sglang", "port"):
        comparisons[f"port_v3_vs_{reference}"] = {
            field: metrics(first["port_v3"][field], first[reference][field])
            for field in ("output", "state")
        }
    numerical_screen = all(
        value["cosine"] >= MIN_COSINE and value["relative_rms"] <= MAX_RELATIVE_RMS
        for comparison in comparisons.values()
        for value in comparison.values()
    )
    timing = {name: timing_summary(values) for name, values in times.items()}
    performance_screen = evaluate_performance_screen(m, times, timing, position_counts)
    receipt = {
        "m": m,
        "geometry": {"batch": 1, "key_heads": HK, "value_heads": HV, "k": K, "v": V},
        "production_layout": {
            "qkv_row_stride_bf16": QKV_ROW_STRIDE,
            "qkv_base_offsets_bf16": [Q_OFFSET, K_OFFSET, V_OFFSET],
            "gate_beta_row_stride_fp32": GATE_BETA_ROW_STRIDE,
            "gate_beta_layout": "linear alpha[48], beta[48]",
        },
        "normalization": {
            "qk": "identical pre-normalized BF16; SGLang in-kernel normalization disabled",
            "gate": (
                "SGLang uses original FP32 log_gate; Atlas uses FP32 alpha; "
                "port v2/v3 consume the same alpha and write identical compact "
                "log(max(alpha,1e-30)); v2 supplies the compact-v1 cross-layout oracle"
            ),
            "state": "canonical [1,H,V,K] FP32; CUDA arms transpose to [H,K,V] and back",
        },
        "workspace_bytes": {
            "compact_v1": port_bridge.workspace_bytes(m),
            "production_v2": port_bridge.workspace_bytes(m, production=True),
            "production_v3": port_bridge.workspace_bytes(m, v3=True),
            "sharing_contract": (
                "one caller allocation may be reused sequentially across layers on its bound "
                "stream; concurrent streams require distinct allocations"
            ),
        },
        "hashes": hashes,
        "deterministic_full_bytes": deterministic,
        "converted_log_deterministic": converted_log_deterministic,
        "converted_log_v2_equals_v3_exact": converted_log_cross_version_exact,
        "compact_v1_equals_production_v2_exact": cross_layout_exact,
        "immutable_inputs": immutable,
        "canaries": "PASS",
        "finite": finite,
        "comparisons": comparisons,
        "numerical_screen": {
            "min_cosine": MIN_COSINE,
            "max_relative_rms": MAX_RELATIVE_RMS,
            "pass": numerical_screen,
            "scope": "default-off port/shadow admission only; not production promotion",
        },
        "timing": timing,
        "performance_screen": {
            **performance_screen,
        },
        "inputs": before,
    }
    hard_checks = {
        "deterministic_full_bytes": deterministic,
        "compact_v1_equals_production_v2_exact": cross_layout_exact,
        "immutable_inputs": immutable,
        "all_finite": all(finite.values()),
        "numerical_screen": numerical_screen,
    }
    if not all(hard_checks.values()):
        stage_hashes = {}
        stage_deterministic = {}
        stage_comparisons = {}
        for stage in ("w", "u", "gc", "h", "v_new"):
            for arm in ("port", "port_v3"):
                first_hash = tensor_sha(first[arm]["stages"][stage])
                second_hash = tensor_sha(second[arm]["stages"][stage])
                stage_hashes[f"{arm}_{stage}_first"] = first_hash
                stage_hashes[f"{arm}_{stage}_second"] = second_hash
                stage_deterministic[f"{arm}_{stage}"] = first_hash == second_hash
            stage_comparisons[f"port_v3_vs_port_{stage}"] = metrics(
                first["port_v3"]["stages"][stage], first["port"]["stages"][stage]
            )
        print(
            json.dumps(
                {
                    "qualification": "FAIL",
                    "m": m,
                    "hard_checks": hard_checks,
                    "hashes": hashes,
                    "finite": finite,
                    "comparisons": comparisons,
                    "stage_hashes": stage_hashes,
                    "stage_deterministic": stage_deterministic,
                    "stage_comparisons": stage_comparisons,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        raise SystemExit("hard correctness gate failed")
    if not performance_screen["pass"]:
        print(
            json.dumps(
                {
                    "qualification": "FAIL",
                    "failure": "hard_v3_performance_gate",
                    "m": m,
                    "hard_checks": hard_checks,
                    "performance_screen": performance_screen,
                    "timing": timing,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        raise SystemExit("hard v3 performance gate failed")
    receipt["qualification"] = "PASS"
    print(json.dumps(receipt, sort_keys=True), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, action="append", default=[])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--reps", type=int, default=11)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise SystemExit("qualification output already exists; choose a fresh path")
    lengths = args.m or [2079, 8192]
    if lengths != [2079, 8192] or args.warmup < 2 or args.reps < 11:
        raise SystemExit(
            "qualification requires --m 2079 --m 8192, warmup>=2, reps>=11"
        )
    commit, audited_sglang, chunk_gated_delta_rule = load_pinned_sglang()
    if file_sha(ATLAS_SOURCE) != ATLAS_SHA256:
        raise SystemExit("Atlas source drift")
    atlas_path = HERE / "build/libatlas_gdn_wy32_sm121.so"
    port_path = HERE / "build/libatlas_gdn_c143_sm121.so"
    atlas_bridge = AtlasBridge(atlas_path)
    port_bridge = PortBridge(port_path)
    qualification_stream = torch.cuda.Stream()
    qualification_stream_handle = int(qualification_stream.cuda_stream)
    if qualification_stream_handle == 0:
        raise SystemExit("qualification requires an explicit non-default CUDA stream")
    header = {
        "sglang_commit": commit,
        "sglang_tracked_worktree": "clean",
        "sglang_audited_sha256": audited_sglang,
        "atlas_source_sha256": ATLAS_SHA256,
        "atlas_bridge_sha256": file_sha(atlas_path),
        "atlas_bridge_abi_identity": atlas_bridge.abi_identity,
        "port_library_sha256": file_sha(port_path),
        "port_source_sha256": file_sha(HERE / "src/gated_delta_rule_sglang_c143.cu"),
        "port_abi_identity": port_bridge.abi_identity,
        "harness_sha256": file_sha(Path(__file__)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "capability": torch.cuda.get_device_capability(),
        "execution_stream": "explicit_non_default_pytorch_stream",
        "execution_stream_native_handle_nonzero": True,
    }
    with torch.cuda.stream(qualification_stream):
        if int(torch.cuda.current_stream().cuda_stream) != qualification_stream_handle:
            raise SystemExit("failed to bind explicit qualification CUDA stream")
        receipts = [
            run_case(
                atlas_bridge,
                port_bridge,
                chunk_gated_delta_rule,
                m,
                args.warmup,
                args.reps,
            )
            for m in lengths
        ]
    qualification_stream.synchronize()
    document = {"provenance": header, "cases": receipts}
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {"result_path": str(args.output), "result_sha256": file_sha(args.output)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
