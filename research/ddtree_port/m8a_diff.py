#!/usr/bin/env python3
"""
M8A precision validation: load /tmp/m8a_dump_*.bin, run Python tree GDN
reference, bit-diff against Atlas kernel output.

Usage:
    # 1) Run Atlas with ATLAS_M8A_DUMP=1 to populate /tmp/m8a_dump_*.bin
    # 2) python3 m8a_diff.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

# Make the prototype importable.
sys.path.insert(0, str(Path(__file__).parent / "ddtree_src" / "prototypes"))
from ddtree_gdn_reference import tree_gated_delta_reference  # noqa: E402


DUMP_DIR = Path("/tmp")


def bf16_to_fp32(buf: bytes) -> np.ndarray:
    u = np.frombuffer(buf, dtype=np.uint16)
    f = (u.astype(np.uint32) << 16).view(np.float32)
    return f


def load_bf16(name: str, shape: tuple[int, ...]) -> torch.Tensor:
    buf = (DUMP_DIR / f"m8a_dump_{name}.bin").read_bytes()
    arr = bf16_to_fp32(buf).reshape(shape)
    return torch.from_numpy(arr).to(torch.bfloat16)


def load_fp32(name: str, shape: tuple[int, ...]) -> torch.Tensor:
    buf = (DUMP_DIR / f"m8a_dump_{name}.bin").read_bytes()
    arr = np.frombuffer(buf, dtype=np.float32).reshape(shape).copy()
    return torch.from_numpy(arr)


def load_i32(name: str, n: int) -> torch.Tensor:
    buf = (DUMP_DIR / f"m8a_dump_{name}.bin").read_bytes()
    arr = np.frombuffer(buf, dtype=np.int32)[:n].copy()
    return torch.from_numpy(arr)


def main() -> int:
    meta_path = DUMP_DIR / "m8a_dump_meta.json"
    if not meta_path.exists():
        print(f"ERROR: {meta_path} not found. Run Atlas with ATLAS_M8A_DUMP=1 first.")
        return 1
    meta = json.loads(meta_path.read_text())
    T = meta["num_tokens"]
    nv = meta["num_v_heads"]
    nk = meta["num_k_heads"]
    kd = meta["k_dim"]
    vd = meta["v_dim"]
    qk_stride = meta["qk_stride"]
    v_stride = meta["v_stride"]
    gb_stride = meta["gb_stride"]
    print(f"=== M8A precision diff ===")
    print(f"  T={T} nk={nk} nv={nv} kd={kd} vd={vd}")
    print(f"  qk_stride={qk_stride} v_stride={v_stride} gb_stride={gb_stride}")

    # Atlas's deinterleaved layout per token: contiguous BF16 of length qk_stride
    # = num_k_heads*k_dim*2 + num_v_heads*v_dim.
    # Within that buffer: Q lives at offset 0..nk*kd, K at nk*kd..2*nk*kd,
    # V at 2*nk*kd..2*nk*kd+nv*vd.
    # The dumped q_ptr is already offset at Q within the row, so layout is
    # [T, qk_stride] BF16 but only first nk*kd valid; same for k_ptr.
    # v_ptr similarly: first nv*vd valid per row at v_stride span.
    q_full = load_bf16("q", (T, qk_stride))[:, : nk * kd].reshape(T, nk, kd)
    k_full = load_bf16("k", (T, qk_stride))[:, : nk * kd].reshape(T, nk, kd)
    v_full = load_bf16("v", (T, v_stride))[:, : nv * vd].reshape(T, nv, vd)
    # GQA: repeat q,k from nk → nv heads (Atlas's kernel does kh = vh / hr).
    if nv != nk:
        repeat = nv // nk
        q_full = q_full.repeat_interleave(repeat, dim=1)  # [T, nv, kd]
        k_full = k_full.repeat_interleave(repeat, dim=1)

    # gate/beta are interleaved [gate(nv) | beta(nv)] per token at gb_stride.
    # gate/beta are dumped as (T, nv) FP32 — see dump code in trait_decode_batched_conv_gdn.rs.
    gate_raw = load_fp32("gate", (T, nv))
    beta_raw = load_fp32("beta", (T, nv))

    parents = load_i32("parent_ids", T)

    # h_in: Atlas stores h_state as [batch, vh, kd*vd] FP32. Just one vh-batch
    # slice was dumped, but layout is contiguous per (vh, b).
    # Full layout: nv * kd * vd
    h_in = load_fp32("h_in", (nv, kd, vd))
    h_out_inter = load_fp32("h_out_inter", (T, nv, kd, vd))
    output = load_bf16("output", (T, nv, vd))

    print(f"\n  q shape={list(q_full.shape)} mean={q_full.float().mean().item():.5f}")
    print(f"  k shape={list(k_full.shape)} mean={k_full.float().mean().item():.5f}")
    print(f"  v shape={list(v_full.shape)} mean={v_full.float().mean().item():.5f}")
    print(f"  gate range=[{gate_raw.min():.4f}, {gate_raw.max():.4f}]")
    print(f"  beta range=[{beta_raw.min():.4f}, {beta_raw.max():.4f}]")
    print(f"  parents={parents.tolist()}")
    print(f"  h_in norm={h_in.norm():.4f}")

    # === Run the Python reference ===
    # Reference expects:
    #   q,k: [T, H, K]  (H = nk for q/k? or nv?)
    #   v:   [T, HV, V]
    #   gate, beta: [T, HV]
    #   parent_ids: [T]
    #   initial_state: [HV, V, K]
    #
    # Atlas convention: gate's pre-exp value is already in gate_raw (post
    # softplus etc.). Reference's `tree_gated_delta_reference` calls
    # `torch.exp(gate)` internally. Atlas's wy_k kernels apply
    # fminf(fmaxf(gate, 1e-6), 1-1e-6) — that's the value Atlas treats AS
    # the multiplier, NOT a log-domain value to exp. So we need to FEED the
    # reference with log(gate_clamped) so that exp(log(g)) = g matches.
    gate_clamped = gate_raw.clamp(1e-6, 1.0 - 1e-6)
    gate_for_ref = torch.log(gate_clamped)  # ref will exp it back

    # Reference's `beta` is passed through sigmoid internally. Atlas's wy_k
    # uses beta as a direct multiplier (no sigmoid). So we feed
    # logit(beta) = log(beta/(1-beta)) so sigmoid(logit(beta)) = beta.
    beta_clamped = beta_raw.clamp(1e-5, 1.0 - 1e-5)
    beta_for_ref = torch.log(beta_clamped / (1.0 - beta_clamped))

    # Reference's initial_state shape is [HV, V, K]. Atlas's h_state is
    # [nv, kd, vd] — i.e. axis order (head, k_dim, v_dim) = (HV, K, V).
    # Need to permute to (HV, V, K).
    initial_state = h_in.permute(0, 2, 1).contiguous()

    print("\n=== running Python reference ===")
    ref_output, ref_states = tree_gated_delta_reference(
        q_full.float(),  # q [T, H, K] — H = nk
        k_full.float(),
        v_full.float(),
        gate_for_ref,
        beta_for_ref,
        parents,
        initial_state,
    )
    # ref_output shape [T, HV, V]; ref_states shape [T, HV, V, K].

    # === Apply 1/sqrt(d) scale (Atlas applies it; reference doesn't unless
    # it's part of the algorithm). Atlas's gated_delta_rule_tree.cu does
    # `qd * rsqrt(k_dim)` at the end.
    ref_output_scaled = ref_output * (1.0 / (kd ** 0.5))

    # === Bit-diff vs Atlas output ===
    atlas_output_f32 = output.float()
    diff = (ref_output_scaled.float() - atlas_output_f32).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    rel = diff / atlas_output_f32.abs().clamp(min=1e-6)
    max_rel = rel.max().item()
    cos_sim = (
        (ref_output_scaled.float() * atlas_output_f32).sum()
        / (ref_output_scaled.float().norm() * atlas_output_f32.norm() + 1e-12)
    ).item()

    print(f"\n=== OUTPUT DIFF (ref vs Atlas kernel) ===")
    print(f"  max abs diff:  {max_abs:.6f}")
    print(f"  mean abs diff: {mean_abs:.6f}")
    print(f"  max rel diff:  {max_rel:.6f}")
    print(f"  cosine sim:    {cos_sim:.6f}")

    # Per-token breakdown.
    print(f"\n  per-token mean abs diff:")
    for t in range(T):
        td = (ref_output_scaled[t].float() - atlas_output_f32[t]).abs().mean().item()
        tcos = (
            (ref_output_scaled[t].float() * atlas_output_f32[t]).sum()
            / (ref_output_scaled[t].float().norm() * atlas_output_f32[t].norm() + 1e-12)
        ).item()
        print(f"    t={t:>2} parent={parents[t].item():>3}  mean_abs={td:.6f}  cos={tcos:.6f}")

    # === States diff (FP32 reads) ===
    # ref_states shape [T, HV, V, K]; Atlas h_out_inter [T, HV, K, V] → permute.
    atlas_states_permuted = h_out_inter.permute(0, 1, 3, 2).contiguous()
    sdiff = (ref_states.float() - atlas_states_permuted).abs()
    print(f"\n=== STATE DIFF (ref vs Atlas h_state_inter) ===")
    print(f"  max abs diff:  {sdiff.max().item():.6f}")
    print(f"  mean abs diff: {sdiff.mean().item():.6f}")
    print(f"  per-token state cos:")
    for t in range(T):
        sc = (
            (ref_states[t].float() * atlas_states_permuted[t]).sum()
            / (ref_states[t].float().norm() * atlas_states_permuted[t].norm() + 1e-12)
        ).item()
        print(f"    t={t:>2}  state_cos={sc:.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
