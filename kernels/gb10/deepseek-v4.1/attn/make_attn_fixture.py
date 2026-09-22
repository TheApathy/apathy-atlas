#!/usr/bin/env python3
"""Fixture for sparse_attn.cu -- the V4.1 one-pass streaming gather.

Synthetic, not captured. This gate checks the KERNEL: the gather-by-index, the
online softmax, the learned sink, the masking. It cannot check the
orchestration (which layer sources the cache, which RoPE table, the inverse
RoPE on the output) -- that needs dsv41-oracle's captured tensors and is a
separate gate. See SPEC.md sections 1-4 for what is NOT covered here.

Shapes are the real ones: 64 query heads sharing ONE 512-dim KV row per
position (MQA -> K and V are the same tensor), 128 window rows + 512
indexer-selected compressed rows.

S = 4096 with RING = 4096 so the window wraps the ring (positions 3969..4223
land in slots 3969..4095 and 0..127). A kernel that forgets the modulo passes a
non-wrapping fixture and fails this one.

Token 0 is deliberately ALL-MASKED (every wpos and cidx is -1). The reference
for that row is exactly zero; a kernel without the m_safe clamp gives NaN.

Two references are written, and the gate reports against BOTH (SPEC.md sec 5):
  ar_f32.bin  -- P kept in fp32, i.e. engine/model.py `_softmax_attn`
  ar_bf16.bin -- P rounded to bf16 before the PV product, i.e. the numerics
                 tools/prefill_attn.py accepted for the prefill path
Both are accumulated in float64 from the same bf16 inputs the kernel sees, so
the gap between them is the P-rounding contract alone and nothing else.
"""
import numpy as np, torch

T, NH, D, NW, NC, RING, S = 128, 64, 512, 128, 512, 4096, 4096
torch.manual_seed(0)

q    = (torch.randn(T, NH, D) * 0.05).to(torch.bfloat16)
ring = (torch.randn(RING, D) * 0.05).to(torch.bfloat16)
n_c  = (S + T)                                    # ratio 1 (layer 20): one compressed row per token
ckv  = (torch.randn(n_c, D) * 0.05).to(torch.bfloat16)
sink = (torch.randn(NH) * 0.5).float()            # learned per-head, denominator only

pos  = torch.arange(S, S + T)
wpos = pos[:, None] - torch.arange(NW - 1, -1, -1)[None, :]
wpos = torch.where(wpos >= 0, wpos, torch.full_like(wpos, -1))          # [T, NW]

# The indexer hands back exactly NC columns, ascending, -1 padded (SPEC.md sec 4).
g = torch.Generator().manual_seed(1)
cidx = torch.full((T, NC), -1, dtype=torch.int64)
for t in range(T):
    avail = int(pos[t]) + 1                        # compress_lens[t] at ratio 1
    keep = NC - (t % 7)                            # a few rows short => genuine -1 padding
    sel = torch.randperm(avail, generator=g)[:keep].sort().values
    cidx[t, :keep] = sel

cidx[0, :] = -1                                    # the all-masked control row
wpos[0, :] = -1

def reference(round_p_to_bf16: bool) -> torch.Tensor:
    """Sinked softmax over the union of both row sets, float64, window rows first."""
    out = torch.zeros(T, NH, D, dtype=torch.float64)
    scale = D ** -0.5
    for t in range(T):
        rows, valid = [], []
        for p in wpos[t].tolist():
            rows.append(ring[p % RING] if p >= 0 else torch.zeros(D, dtype=torch.bfloat16))
            valid.append(p >= 0)
        for j in cidx[t].tolist():
            rows.append(ckv[j] if j >= 0 else torch.zeros(D, dtype=torch.bfloat16))
            valid.append(j >= 0)
        K = torch.stack(rows).double()                                   # [640, D]
        v = torch.tensor(valid)
        s = (q[t].double() @ K.T) * scale                                # [NH, 640]
        s = s.masked_fill(~v[None, :], float("-inf"))
        m = s.amax(-1, keepdim=True).clamp_min(-1e30)
        p_ = torch.exp(s - m)
        denom = p_.sum(-1, keepdim=True) + torch.exp(sink[:, None].double() - m)
        p_ = p_ / denom
        if round_p_to_bf16:
            p_ = p_.to(torch.bfloat16).double()
        out[t] = p_ @ K
    return out

ref32 = reference(False)
refbf = reference(True)
gap = ((ref32 - refbf).pow(2).sum() / ref32.pow(2).sum()).sqrt().item()

assert torch.isfinite(ref32).all(), "reference itself is not finite"
assert ref32[0].abs().max() == 0, "the all-masked control row is not zero"

q.view(torch.uint16).numpy().tofile("ar_q.bin")
ring.view(torch.uint16).numpy().tofile("ar_ring.bin")
ckv.view(torch.uint16).numpy().tofile("ar_ckv.bin")
sink.numpy().astype(np.float32).tofile("ar_sink.bin")
wpos.numpy().astype(np.int32).tofile("ar_wpos.bin")
cidx.numpy().astype(np.int32).tofile("ar_cidx.bin")
ref32.numpy().astype(np.float32).tofile("ar_f32.bin")
refbf.numpy().astype(np.float32).tofile("ar_bf16.bin")
open("ar_dims.txt", "w").write(f"{T} {NH} {D} {NW} {NC} {RING} {n_c}\n")

print(f"T={T} NH={NH} D={D} NW={NW} NC={NC} RING={RING} n_c={n_c}")
print(f"masked cols: wpos {(wpos < 0).sum().item()}  cidx {(cidx < 0).sum().item()}")
print(f"fp32-P vs bf16-P reference gap: rel_l2={gap:.3e}   <- the prefill kernel's accepted cost")
