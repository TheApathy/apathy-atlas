#!/usr/bin/env python3
"""Linear-algebra structure probe for MXFP4 expert weights.

Questions (all quality-exact analyses; no requant assumed):
  A. Per-expert spectral decay: what energy does a rank-r factorization keep?
  B. Shared right-basis: do experts in a layer share a subspace? (basis
     amortization: read basis once/layer, small coefficients per expert)
  C. Cross-expert cosine similarity (delta-coding viability, value level)
  D. Code-level mutual information between experts (delta-coding, symbol level)
"""
import json, sys
import torch
from safetensors import safe_open

MODEL = "/home/flocka/models/DeepSeek-V4-Flash-162B"
LAYERS = [0, 21, 42]
NEXP = 16
RANKS = [256, 512, 1024, 1536]
dev = "cuda"

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=dev)

index = json.load(open(f"{MODEL}/model.safetensors.index.json"))["weight_map"]

_open = {}
def shard(name):
    fn = index[name]
    if fn not in _open:
        _open[fn] = safe_open(f"{MODEL}/{fn}", "pt")
    return _open[fn]

def dequant(lname):
    w = shard(f"{lname}.weight").get_tensor(f"{lname}.weight")
    s = shard(f"{lname}.scale").get_tensor(f"{lname}.scale")
    b = w.view(torch.uint8).to(dev)
    lo, hi = b & 0xF, b >> 4
    idx = torch.stack([lo, hi], dim=-1).reshape(b.shape[0], -1)
    mag = E2M1[(idx & 7).long()]
    val = torch.where((idx & 8).bool(), -mag, mag)
    exp = s.view(torch.uint8).to(dev).to(torch.float32) - 127.0
    val = val * torch.pow(2.0, exp).repeat_interleave(32, dim=1)
    return val, idx  # f32 [R, C_logical], nibble codes [R, C_logical]

report = {}
for L in LAYERS:
    lrep = {}
    Ws, codes = [], []
    for e in range(NEXP):
        v, c = dequant(f"layers.{L}.ffn.experts.{e}.w1")
        Ws.append(v)
        codes.append(c)

    # A. per-expert spectra
    caps = {r: [] for r in RANKS}
    for v in Ws:
        sv = torch.linalg.svdvals(v)
        en = sv.square()
        cum = en.cumsum(0) / en.sum()
        for r in RANKS:
            caps[r].append(cum[r - 1].item())
    lrep["per_expert_energy_at_rank"] = {
        r: sum(c) / len(c) for r, c in caps.items()
    }

    # B. shared right-basis across the 16 experts
    G = torch.zeros(Ws[0].shape[1], Ws[0].shape[1], device=dev)
    tot = 0.0
    for v in Ws:
        G += v.T @ v
        tot += v.square().sum().item()
    ev = torch.linalg.eigvalsh(G)
    ev = ev.flip(0).clamp_min(0)
    cum = ev.cumsum(0) / ev.sum()
    lrep["shared_basis_energy_at_rank"] = {
        r: cum[r - 1].item() for r in RANKS + [2048, 3072]
    }

    # C. cross-expert cosine (flattened values)
    F = torch.stack([v.flatten() for v in Ws])
    Fn = torch.nn.functional.normalize(F, dim=1)
    C = Fn @ Fn.T
    off = C[~torch.eye(NEXP, dtype=bool, device=dev)]
    lrep["cross_expert_cos"] = {
        "mean_abs": off.abs().mean().item(),
        "max_abs": off.abs().max().item(),
    }

    # D. symbol-level mutual information, 3 pairs
    mis = []
    for a, b in [(0, 1), (2, 3), (4, 5)]:
        ca, cb = codes[a].flatten().long(), codes[b].flatten().long()
        joint = torch.zeros(16, 16, device=dev)
        joint.index_put_((ca, cb), torch.ones_like(ca, dtype=torch.float32),
                         accumulate=True)
        p = joint / joint.sum()
        pa, pb = p.sum(1, keepdim=True), p.sum(0, keepdim=True)
        mask = p > 0
        mi = (p[mask] * (p[mask] / (pa @ pb)[mask]).log2()).sum().item()
        mis.append(mi)
    lrep["code_mutual_info_bits"] = mis

    report[f"layer{L}"] = lrep
    del Ws, codes, F, G
    torch.cuda.empty_cache()
    print(f"layer {L} done", file=sys.stderr)

print(json.dumps(report, indent=2, default=float))
