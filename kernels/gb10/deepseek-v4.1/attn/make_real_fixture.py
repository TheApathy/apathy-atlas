#!/usr/bin/env python3
"""Real-tensor fixture for sparse_attn.cu, built entirely from the oracle capture.

The synthetic fixture proves the kernel's arithmetic. This one proves it on the
actual tensors the model produces: q, the window ring, the compressed cache and
the indexer's selection are all captured from runC_2048 (layer 2, 2048 tokens),
and the per-head sink is read straight from the checkpoint. NO GPU run is
needed to build it -- attn_sink is a stored weight, not a tap.

What this still does NOT cover: the inverse RoPE and the grouped o_proj that
follow attention are the CALLER's job (see SPEC.md sec 2), and `attn_out` in the
capture is taken after both, so it is not a target for this kernel.

Token subsampling: the float64 reference is O(T * 640 * 512) and 2048 rows is
wasteful when every query is independent by construction. We take a spread of
rows covering the three regimes that actually differ -- early queries whose
window is clipped and whose compressed set is far below 512, the transition
around compress_lens == 512, and late queries with a full selection.
"""
import json, os, struct
import numpy as np, torch

REF = "/home/flocka/atlas/DSV41_PORT/oracle/ref/runC_2048"
CKPT = "/home/flocka/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K"
LAYER, RATIO = 2, 2
T_FULL, NH, D, NW, NC, RING = 2048, 64, 512, 128, 512, 4096

man = json.load(open(os.path.join(REF, "manifest.json")))
ents = man["tensors"] if isinstance(man, dict) and "tensors" in man else man
ents = ents if isinstance(ents, list) else list(ents.values())
by = {e["file"]: e for e in ents}

def load(name):
    e = by[name]
    raw = np.fromfile(os.path.join(REF, name),
                      dtype=np.uint16 if e["stored_dtype"] == "uint16" else np.int64)
    t = torch.from_numpy(raw.copy())
    if e["stored_dtype"] == "uint16":
        t = t.view(torch.bfloat16)
    return t.reshape(e["shape"]) if e["shape"] else t

q_all    = load(f"L{LAYER:02d}.q.000.bin")        # [2048, 64, 512] bf16, RoPE'd
kv_all   = load(f"L{LAYER:02d}.kv_new.000.bin")   # [2048, 512]     bf16, RoPE'd, MQA
ckv      = load(f"L{LAYER:02d}.latent_1.000.bin") # [1024, 512]     bf16
topk     = load(f"L{LAYER:02d}.topk.000.bin")     # [2048, 512]     i64 absolute, -1 = none
j0       = int(load(f"L{LAYER:02d}.latent_0.000.bin").item())
assert j0 == 0, f"this fixture assumes a single chunk starting at 0, got j0={j0}"

# attn_sink is a WEIGHT, read from the checkpoint -- no forward pass required.
widx = json.load(open(os.path.join(CKPT, "model.safetensors.index.json")))["weight_map"]
name = f"layers.{LAYER}.attn.attn_sink"
with open(os.path.join(CKPT, widx[name]), "rb") as fh:
    n = struct.unpack("<Q", fh.read(8))[0]
    hdr = json.loads(fh.read(n)); meta = hdr[name]; s, e = meta["data_offsets"]
    fh.seek(8 + n + s)
    dt = {"F32": np.float32, "BF16": np.uint16}[meta["dtype"]]
    buf = np.frombuffer(fh.read(e - s), dtype=dt).reshape(meta["shape"]).copy()
sink = torch.from_numpy(buf)
if meta["dtype"] == "BF16":
    sink = sink.view(torch.bfloat16)
sink = sink.float().reshape(-1)
assert sink.numel() == NH, (sink.shape, NH)

# The ring holds every position written so far; pos < 2048 < RING so slot == pos.
ring = torch.zeros(RING, D, dtype=torch.bfloat16)
ring[torch.arange(T_FULL) % RING] = kv_all

sel = sorted(set(list(range(0, 64)) + list(range(1000, 1032)) +
                 list(range(1020, 1052)) + list(range(2016, 2048))))
sel = torch.tensor(sel)
T = sel.numel()

pos  = sel
wpos = pos[:, None] - torch.arange(NW - 1, -1, -1)[None, :]
wpos = torch.where(wpos >= 0, wpos, torch.full_like(wpos, -1))
q    = q_all[sel].contiguous()
cidx = topk[sel].contiguous()

clen = (pos + 1) // RATIO
assert int((cidx >= 0).sum(1).max()) <= NC
for i in range(T):
    v = cidx[i][cidx[i] >= 0]
    assert v.numel() == min(NC, int(clen[i])), (i, v.numel(), clen[i])
    assert v.numel() == 0 or int(v.max()) < int(clen[i])

def reference(round_p_to_bf16: bool) -> torch.Tensor:
    out = torch.zeros(T, NH, D, dtype=torch.float64)
    scale = D ** -0.5
    for i in range(T):
        rows, valid = [], []
        for p in wpos[i].tolist():
            rows.append(ring[p % RING] if p >= 0 else torch.zeros(D, dtype=torch.bfloat16))
            valid.append(p >= 0)
        for j in cidx[i].tolist():
            rows.append(ckv[j] if j >= 0 else torch.zeros(D, dtype=torch.bfloat16))
            valid.append(j >= 0)
        K = torch.stack(rows).double()
        v = torch.tensor(valid)
        s = (q[i].double() @ K.T) * scale
        s = s.masked_fill(~v[None, :], float("-inf"))
        m = s.amax(-1, keepdim=True).clamp_min(-1e30)
        p_ = torch.exp(s - m)
        den = p_.sum(-1, keepdim=True) + torch.exp(sink[:, None].double() - m)
        p_ = p_ / den
        if round_p_to_bf16:
            p_ = p_.to(torch.bfloat16).double()
        out[i] = p_ @ K
    return out

ref32, refbf = reference(False), reference(True)
assert torch.isfinite(ref32).all()
gap = ((ref32 - refbf).pow(2).sum() / ref32.pow(2).sum()).sqrt().item()

q.view(torch.uint16).numpy().tofile("ar_q.bin")
ring.view(torch.uint16).numpy().tofile("ar_ring.bin")
ckv.contiguous().view(torch.uint16).numpy().tofile("ar_ckv.bin")
sink.numpy().astype(np.float32).tofile("ar_sink.bin")
wpos.numpy().astype(np.int32).tofile("ar_wpos.bin")
cidx.numpy().astype(np.int32).tofile("ar_cidx.bin")
ref32.numpy().astype(np.float32).tofile("ar_f32.bin")
refbf.numpy().astype(np.float32).tofile("ar_bf16.bin")
open("ar_dims.txt", "w").write(f"{T} {NH} {D} {NW} {NC} {RING} {ckv.shape[0]}\n")
print(f"REAL fixture: layer {LAYER}, T={T} of {T_FULL}, ckv_rows={ckv.shape[0]}")
print(f"  compress_lens range {int(clen.min())}..{int(clen.max())}, "
      f"rows at full 512: {(clen >= NC).sum().item()}/{T}")
print(f"  cidx -1 fraction {float((cidx < 0).float().mean()):.3f}   "
      f"sink range [{sink.min():.3f}, {sink.max():.3f}]")
print(f"  fp32-P vs bf16-P reference gap: rel_l2={gap:.3e}")
