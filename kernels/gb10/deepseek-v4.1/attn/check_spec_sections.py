"""Validate SPEC.md sections 1-3 (cache sourcing, RoPE table selection, inverse RoPE on the
output, the gather, the layer-20 candidate hand-off) against real taps. CPU only, no GPU.

  python3 check_spec_sections.py            # runD_L20_kernel + runE_torch

Every PASS below is paired with a CONTROL that must FAIL on the same data; a check whose
control does not fail is reported as UNRUNNABLE, not PASS.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch
from safetensors import safe_open

from oracle_io import Run, bf16_to_f32, f32_to_bf16_bits

MODEL = "/home/flocka/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K"
RING, WIN, RD = 4096, 128, 64
KV_SOURCE = (2, 8, 14, 20)
RATIO = [0, 0] + [2] * 18 + [1] * 20
NORM_EPS = 1e-20
results: list[tuple[str, bool]] = []


def verdict(name: str, ok: bool, ctrl_failed: bool | None = None, detail: str = ""):
    if ctrl_failed is False:
        state = "UNRUNNABLE (control did not fail)"
        ok = False
    else:
        state = "PASS" if ok else "FAIL"
    print(f"[{state}] {name} {detail}")
    results.append((name, ok))


# ------------------------------------------------------------------ RoPE (v41_ref, verbatim math)
def freqs(dim, seqlen, original_seq_len, base, factor=16, beta_fast=32, beta_slow=1):
    f = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        def corrected_dim(rot):
            return dim * math.log(original_seq_len / (rot * 2 * math.pi)) / (2 * math.log(base))
        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = ((torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)).clamp(0, 1)
        smooth = 1 - ramp
        f = f / factor * (1 - smooth) + f * smooth
    f = torch.outer(torch.arange(seqlen), f)
    return torch.polar(torch.ones_like(f), f)


TABLES = {"freqs_c": freqs(RD, 8200, 65536, 160000.0), "freqs_w": freqs(RD, 8200, 0, 10000.0)}


def rope_tail(x: torch.Tensor, fc: torch.Tensor, inverse=False) -> torch.Tensor:
    """x [T, D] or [T, H, D] float32; rotary on the last 64 dims; fp32 result."""
    tail = torch.view_as_complex(x[..., -RD:].float().contiguous().unflatten(-1, (-1, 2)))
    f = fc.conj() if inverse else fc
    if tail.ndim == 3:
        f = f[:, None, :]
    y = torch.view_as_real(tail * f).flatten(-2)
    return torch.cat([x[..., :-RD].float(), y], dim=-1)


def T(a):
    return torch.from_numpy(np.ascontiguousarray(a))


def bits(x: torch.Tensor) -> np.ndarray:
    return f32_to_bf16_bits(x.float().numpy())


def weight(L, name):
    import json
    idx = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]
    k = f"layers.{L}.{name}"
    with safe_open(os.path.join(MODEL, idx[k]), "pt") as f:
        return f.get_tensor(k)


def chunks(R: Run, L: int):
    """(occurrence, S) for each chunk this layer was captured in.

    S is the first POSITION of the rows, not the recorder's chunk offset: layers above the
    candidate source (20) run only in the decoder replay over the prompt tail, where the
    manifest's S is still the enclosing chunk's (512) but the rows are positions win_lo.. (896..).
    """
    out = []
    occ = 0
    while R.has(L, "kv_new", occ):
        S = R.meta(L, "kv_new", occ)["S"]
        if L > 20:
            S = int(R.load(L, "win_lo", occ))
        out.append((occ, S))
        occ += 1
    return out


def topk_occ(R: Run, L: int, occ: int) -> int:
    """The kernel path taps `topk` twice per chunk (in _compressed and again in attention)."""
    n_ch = len(chunks(R, L))
    n_tk = sum(1 for i in range(2 * n_ch) if R.has(L, "topk", i))
    if n_tk == 2 * n_ch:
        a, b = R.load(L, "topk", 2 * occ), R.load(L, "topk", 2 * occ + 1)
        assert np.array_equal(a, b), (L, occ)
        return 2 * occ
    return occ


# ------------------------------------------------------------------ §1 cache sourcing
def check_sourcing(R: Run, name: str):
    layers = R.manifest["layers"]
    for L in layers:
        for occ, S in chunks(R, L):
            Tn = R.meta(L, "kv_new", occ)["shape"][0]
            pos = np.arange(S, S + Tn)
            # the ratio tap exists iff the layer has compressed rows
            has_c = R.has(L, "ratio", occ)
            ok = has_c == (RATIO[L] != 0)
            if has_c:
                r = int(R.load(L, "ratio", occ))
                cl = R.load(L, "compress_lens", occ)
                ok &= r == RATIO[L] and np.array_equal(cl, (pos + 1) // r)
                # control: compress_lens computed with CHUNK-RELATIVE positions must disagree
                ctrl = not np.array_equal(cl, (np.arange(Tn) + 1) // r) if S > 0 else None
            else:
                ctrl = None
            wpos = R.load(L, "wpos", occ)
            want = pos[:, None] - np.arange(WIN - 1, -1, -1)[None, :]
            want = np.where(want >= 0, want, -1)
            ok &= np.array_equal(wpos, want)
            verdict(f"{name} L{L:02d} occ{occ} S={S}: ratio/compress_lens(abs pos)/wpos", ok,
                    ctrl, f"ratio={RATIO[L]} win_lo={int(R.load(L, 'win_lo', occ))}")
    # inheritance: a non-kv-source layer's ckv/ik IS its source layer's cache, bit for bit
    if 24 in layers:
        m24 = R.meta(24, "ckv", 0)
        src_occ = max(o for o, S in chunks(R, 20) if S <= m24["S"])
        a = R.load(24, "ckv", 0, raw=True); b = R.load(20, "ckv", src_occ, raw=True)
        ai = R.load(24, "ik", 0, raw=True); bi = R.load(20, "ik", src_occ, raw=True)
        # control: layer 14's cache (the previous kv-source, ratio 2) is not layer 24's
        c14 = R.load(14, "ckv", src_occ, raw=True)
        ctrl = c14.shape != a.shape or not np.array_equal(c14, a[:c14.shape[0]])
        verdict(f"{name} L24 ckv/ik == L20's (kv-source inheritance)",
                np.array_equal(a, b) and np.array_equal(ai, bi), ctrl,
                f"{a.shape} vs L20.occ{src_occ} {b.shape}")


# ------------------------------------------------------------------ §2 inverse RoPE, table choice
def check_output_rope(R: Run, name: str):
    for L in R.manifest["layers"]:
        want_tab = "freqs_c" if RATIO[L] else "freqs_w"
        for occ, S in chunks(R, L):
            pre = T(R.load(L, "attn_o_pre_inverse_rope", occ))
            post = R.load(L, "attn_o_post_inverse_rope", occ, raw=True)
            Tn = pre.shape[0]
            fr = {}
            for tab, F in TABLES.items():
                fc = F[S:S + Tn]
                fr[tab] = (bits(rope_tail(pre, fc, inverse=True)) == post)[..., -RD:].mean()
            fr["forward_c"] = (bits(rope_tail(pre, TABLES["freqs_c"][S:S + Tn])) == post)[..., -RD:].mean()
            fr["none"] = (bits(pre) == post)[..., -RD:].mean()
            nope_same = np.array_equal(bits(pre)[..., :-RD], post[..., :-RD])
            best = max(fr, key=fr.get)
            others = max(v for k, v in fr.items() if k != want_tab)
            ok = best == want_tab and fr[want_tab] > 0.99 and nope_same
            verdict(f"{name} L{L:02d} occ{occ}: output inverse-RoPE uses {want_tab}", ok,
                    others < 0.9,
                    " ".join(f"{k}={v:.4f}" for k, v in fr.items()) + f" nope_untouched={nope_same}")


# ------------------------------------------------------------------ §3 compressed rows: freqs_c at j*r
def check_compressed_rope(R: Run, name: str):
    for L in (2, 14, 20):
        if L not in R.manifest["layers"]:
            continue
        r = RATIO[L]
        wk = weight(L, "attn.indexer.wk.weight").float()
        kn = weight(L, "attn.indexer.k_norm.weight").float()
        occ = chunks(R, L)[-1][0]
        ckv = T(R.load(L, "ckv", occ)); ik = R.load(L, "ik", occ, raw=True)
        n = ckv.shape[0]
        j = torch.arange(n)
        scores = {}
        for label, tab, jp in (("freqs_c@j*r", "freqs_c", j * r), ("freqs_c@j", "freqs_c", j),
                               ("freqs_w@j*r", "freqs_w", j * r)):
            fc = TABLES[tab][jp]
            lat = torch.from_numpy(bf16_to_f32(bits(rope_tail(ckv, fc, inverse=True))))  # pre-RoPE latent
            k = lat @ wk.T
            k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + NORM_EPS) * kn
            k = torch.from_numpy(bf16_to_f32(bits(k)))
            k = rope_tail(k, fc)
            got = bf16_to_f32(ik)[1:, -RD:]          # row 0 is angle 0 under every table
            mine = k.numpy()[1:, -RD:]
            scores[label] = float(np.linalg.norm(mine - got) / np.linalg.norm(got))
        ok = scores["freqs_c@j*r"] < 2e-2
        ctrl = min(scores["freqs_c@j"], scores["freqs_w@j*r"]) > 10 * scores["freqs_c@j*r"] if r > 1 \
            else scores["freqs_w@j*r"] > 10 * scores["freqs_c@j*r"]
        verdict(f"{name} L{L:02d}: index keys are RoPE'd with freqs_c at abs pos j*r (from pre-RoPE latent)",
                ok, ctrl, " ".join(f"{k}={v:.3e}" for k, v in scores.items()))


# ------------------------------------------------------------------ gather taps (torch path only)
def check_gather(R: Run, name: str):
    for L in R.manifest["layers"]:
        for occ, S in chunks(R, L):
            if not R.has(L, "win_kv", occ):
                continue
            ring = R.load(L, "ring", occ, raw=True)
            wpos = R.load(L, "wpos", occ); win_lo = int(R.load(L, "win_lo", occ))
            wkv = R.load(L, "win_kv", occ, raw=True)
            ok = np.array_equal(wkv, ring[np.maximum(wpos, 0) % RING])
            wm = R.load(L, "win_mask", occ)
            ok &= np.array_equal(wm, (wpos >= win_lo) if win_lo else (wpos >= 0))
            ctrl = not np.array_equal(wkv, ring[np.maximum(wpos - 1, 0) % RING])
            det = ""
            if R.has(L, "ckv_rows", occ):
                ckv = R.load(L, "ckv", occ, raw=True)
                tk = R.load(L, "topk", topk_occ(R, L, occ))
                rows = R.load(L, "ckv_rows", occ, raw=True)
                ok &= np.array_equal(rows, ckv[np.maximum(tk, 0)])
                ok &= np.array_equal(R.load(L, "c_mask", occ), tk >= 0)
                det = f"valid_c={int((tk >= 0).sum())} valid_w={int(wm.sum())}"
            verdict(f"{name} L{L:02d} occ{occ}: win_kv=ring[wpos%RING], ckv_rows=ckv[topk], masks", ok, ctrl, det)


# ------------------------------------------------------------------ attention recompute (fp64)
def attend(q, ring, wpos, win_lo, ckv, cidx, sink):
    """Reference sinked softmax in float64 over the gathered window + compressed rows."""
    Tn, H, D = q.shape
    out = torch.empty(Tn, H, D, dtype=torch.float64)
    scale = D ** -0.5
    for i in range(0, Tn, 64):
        sl = slice(i, i + 64)
        wp = wpos[sl]
        k = ring[wp.clamp_min(0) % RING]
        m = (wp >= win_lo) if win_lo else (wp >= 0)
        if ckv is not None:
            ci = cidx[sl]
            k = torch.cat([k, ckv[ci.clamp_min(0)]], 1)
            m = torch.cat([m, ci >= 0], 1)
        s = torch.einsum("thd,tnd->thn", q[sl], k) * scale
        s = s.masked_fill(~m[:, None, :], float("-inf"))
        mx = s.amax(-1, keepdim=True).clamp_min(-1e30)
        p = torch.exp(s - mx)
        den = p.sum(-1, keepdim=True) + torch.exp(sink[None, :, None] - mx)
        out[sl] = torch.einsum("thn,tnd->thd", p / den, k)
    return out


def check_attention(R: Run, name: str, layers=(2, 20, 24)):
    for L in layers:
        if L not in R.manifest["layers"]:
            continue
        occ, S = chunks(R, L)[-1]
        d = lambda tap: T(R.load(L, tap, occ)).double()
        q, ring = d("q"), d("ring")
        wpos = T(R.load(L, "wpos", occ)); win_lo = int(R.load(L, "win_lo", occ))
        ckv = d("ckv"); cidx = T(R.load(L, "topk", topk_occ(R, L, occ))); sink = d("attn_sink")
        ref = R.load(L, "attn_o_pre_inverse_rope", occ)
        ref_bits = R.load(L, "attn_o_pre_inverse_rope", occ, raw=True)

        def score(o):
            rl = float(np.linalg.norm(o - ref) / np.linalg.norm(ref))
            ex = float((f32_to_bf16_bits(o.astype(np.float32)) == ref_bits).mean())
            return rl, ex

        rl, ex = score(attend(q, ring, wpos, win_lo, ckv, cidx, sink).numpy())
        # CONTROL 1: the "ckv cleared per chunk" bug -- rows written by earlier chunks read as zero
        chunk_start = R.meta(L, "kv_new", occ)["S"]
        ck0 = ckv.clone(); ck0[: (chunk_start // RATIO[L])] = 0
        c1 = score(attend(q, ring, wpos, win_lo, ck0, cidx, sink).numpy())
        # CONTROL 2: no attention sink in the denominator (a SMALL term: rel_l2 cannot see it at
        # bf16 tolerance, so the gate is the bf16 bit-exact fraction, which can)
        c2 = score(attend(q, ring, wpos, win_lo, ckv, cidx, sink * 0 - 1e30).numpy())
        algo = R.meta(L, "attn_o_pre_inverse_rope", occ)["algo"]
        if algo == "softmax_fp32":   # like-for-like: fp32-P reference, bf16 output
            ok, ctrl = ex > 0.99, c1[1] < 0.9 and c2[1] < 0.9
        else:                        # kernel path rounds P to bf16: an L2 comparison only
            ok, ctrl = rl < 5e-3, c1[0] > 5e-2
        verdict(f"{name} L{L:02d} occ{occ} pos0={S}: fp64 recompute vs attn_o_pre_inverse_rope", ok, ctrl,
                f"rel_l2={rl:.3e} bf16_bit_exact={ex:.4f} | CTRL cleared-ckv {c1[0]:.3e}/{c1[1]:.4f} "
                f"no-sink {c2[0]:.3e}/{c2[1]:.4f} (algo={algo}, win_lo={win_lo})")


# ------------------------------------------------------------------ candidate hand-off 20 -> 24
def check_candidates(R: Run, name: str):
    if 24 not in R.manifest["layers"]:
        return
    ci = R.load(24, "cand_in", 0)
    S24 = R.meta(24, "cand_in", 0)["S"]
    occ20, S20 = [c for c in chunks(R, 20) if c[1] <= S24][-1]
    co = R.load(20, "cand_out", occ20)
    win_lo = int(R.load(24, "win_lo", 0))
    # replay rows are prompt positions win_lo .. win_lo+127, i.e. the tail of layer 20's last chunk
    rows = np.arange(win_lo, win_lo + ci.shape[0]) - S20
    ok = ci.shape[1] == co.shape[1] and np.array_equal(ci, co[rows])
    ctrl = not np.array_equal(ci, co[rows - 1])
    sc = R.load(24, "idx_score", 0)
    ok_mask = bool(np.all(np.isneginf(sc[~ci])))
    verdict(f"{name} L24 cand_in == L20 cand_out[replay rows], and L24 score is -inf off-candidate",
            ok and ok_mask, ctrl,
            f"L24 S={S24} win_lo={win_lo} rows {rows[0]}..{rows[-1]} of L20 occ{occ20}; "
            f"cand density {ci.mean():.3f}")


if __name__ == "__main__":
    runs = sys.argv[1:] or ["runD_L20_kernel", "runE_torch"]
    torch.set_num_threads(16)
    for rn in runs:
        R = Run(rn)
        print(f"==== {rn} (algo {R.manifest['algo']})")
        check_sourcing(R, rn)
        check_gather(R, rn)
        check_candidates(R, rn)
        check_output_rope(R, rn)
        check_compressed_rope(R, rn)
        check_attention(R, rn)
    bad = [n for n, ok in results if not ok]
    print(f"RESULT {'FAIL' if bad else 'PASS'}: {len(results) - len(bad)}/{len(results)}")
    for n in bad:
        print("  not passing:", n)
