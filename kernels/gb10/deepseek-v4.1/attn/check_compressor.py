"""Rebuild ckv/ik on the kv-source layers from attn_x taps + checkpoint weights, against
runF_faithful (checkpoint numerics). This pins the compressor recipe the Rust compress()
implements. CPU only.

  r=2: kvl = x.f32 @ wkv.f32^T, sc = x.f32 @ wgate.f32^T (TRUE fp32), pairwise softmax over r,
       latent = rmsnorm(bf16(sum_r kvl*w), comp_norm)
  r=1: latent = rmsnorm(bf16(x @ wkv^T), comp_norm)
  ik  = rope_c(rmsnorm(bf16(latent @ wk^T), k_norm), pos = j*r)   -- from the PRE-RoPE latent
  ckv = rope_c(latent, pos = j*r)
"""
import sys

import numpy as np
import torch

import check_spec_sections as C
from oracle_io import Run, f32_to_bf16_bits, round_bf16

torch.set_num_threads(16)
EPS = 1e-20


def bf(x):
    return torch.from_numpy(round_bf16(x.float().numpy()))


def rmsnorm(x, w):
    xf = x.float()
    return bf(w.float() * (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + EPS)))


def compress(R, L, occ, pending, variant="ref"):
    r = C.RATIO[L]
    x = torch.from_numpy(R.load(L, "attn_x", occ))
    S = R.meta(L, "attn_x", occ)["S"]
    norm = C.weight(L, "attn.compressor.norm.weight")
    wk = C.weight(L, "attn.indexer.wk.weight").float()
    kn = C.weight(L, "attn.indexer.k_norm.weight")
    if r == 2:
        wkv = C.weight(L, "attn.compressor.wkv.weight").double()
        wg = C.weight(L, "attn.compressor.wgate.weight").double()
        kvl = (x.double() @ wkv.T).float(); sc = (x.double() @ wg.T).float()
        first = S
        if pending is not None:
            kvl = torch.cat([pending[0][None], kvl]); sc = torch.cat([pending[1][None], sc]); first = S - 1
        n = kvl.shape[0]; cut = n - n % r
        pending = (kvl[-1], sc[-1]) if n % r else None
        g_kv, g_sc = kvl[:cut].unflatten(0, (-1, r)), sc[:cut].unflatten(0, (-1, r))
        if variant == "no_softmax":            # CONTROL: plain mean instead of the gated combine
            lat = g_kv.mean(1)
        else:
            lat = (g_kv * g_sc.softmax(dim=1)).sum(1)
        latent = rmsnorm(bf(lat), norm)
        j0 = first // r
    else:
        latent = rmsnorm(bf((x.double() @ C.weight(L, "attn.compressor.wkv.weight").double().T).float()), norm)
        j0 = S
    nj = latent.shape[0]
    jp = (j0 + torch.arange(nj)) * (r if variant != "pos_j" else 1)
    fc = C.TABLES["freqs_c"][jp]
    k = rmsnorm(bf(latent.float() @ wk.T), kn)
    ik = bf(C.rope_tail(k, fc))
    src = bf(C.rope_tail(latent, fc)) if variant == "ik_after_rope" else latent
    if variant == "ik_after_rope":             # CONTROL: index key from the RoPE'd latent
        ik = bf(C.rope_tail(rmsnorm(bf(src.float() @ wk.T), kn), fc))
    ckv = bf(C.rope_tail(latent, fc))
    return j0, ckv, ik, pending


def main(run="runF_faithful"):
    R = Run(run)
    bad = 0
    for L in (2, 8, 14, 20):
        if L not in R.manifest["layers"]:
            continue
        for variant in ("ref", "no_softmax", "pos_j", "ik_after_rope"):
            if variant in ("no_softmax", "pos_j") and C.RATIO[L] != 2:
                continue   # at r=1 there is no combine, and j*r == j: these controls CANNOT fail
            pending = None
            res = []
            for occ in (0, 1):
                j0, ckv, ik, pending = compress(R, L, occ, pending, variant)
                tc = R.load(L, "ckv", occ, raw=True)[j0:j0 + ckv.shape[0]]
                ti = R.load(L, "ik", occ, raw=True)[j0:j0 + ik.shape[0]]
                res.append(((f32_to_bf16_bits(ckv.numpy()) == tc).mean(),
                            (f32_to_bf16_bits(ik.numpy()) == ti).mean()))
            ck = min(a for a, _ in res); ii = min(b for _, b in res)
            if variant == "ref":
                # r=2 lands at >= 99.67% bit-exact. r=1 lands at 99.75% (ckv) / 97.6% (ik), rel
                # 2e-4 / 5.5e-4, spread 2-11 flips per row over nope AND rope dims: its latent is
                # a bf16-OUTPUT GEMM, and fp64 vs fp32 accumulation alone moves only 0.03% of those
                # outputs. UNVERIFIED hypothesis: torch's default bf16 reduced-precision reduction
                # in the reference's cuBLAS call. Atlas's fp32-compute GEMM will not reproduce it
                # either, so the gate is bit-exact >= 97% AND the controls below at < 90%.
                ok = ck > 0.97 and ii > 0.97
                bad += not ok
            else:
                ok = ck < 0.9 or ii < 0.9          # the control must MISS
                bad += not ok
            print(f"L{L:02d} r={C.RATIO[L]} {variant:14s} ckv bit-exact {ck:.4f}  ik bit-exact {ii:.4f}  "
                  f"{'PASS' if ok else ('FAIL' if variant == 'ref' else 'CONTROL DID NOT FAIL')}")
    print("RESULT", "FAIL" if bad else "PASS")


if __name__ == "__main__":
    main(*sys.argv[1:])
