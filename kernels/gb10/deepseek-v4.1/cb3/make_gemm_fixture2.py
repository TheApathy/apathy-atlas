#!/usr/bin/env python3
"""Fixture for cb3_gemm.cu: a FULL expert w1 [2304, 5120] plus activations.

Full-width on purpose. The reconstruct check ran 32x1024; a real GEMM has to
walk all 10 blocks of K=5120 and all 2304 output rows, which is where a
block-index or row-stride slip shows up.
"""
import sys, json, struct, numpy as np, torch
sys.path.insert(0, "/home/flocka/atlas/dsv41-prefill-work/tools")
from cb3 import dequant_cb3_v2

PACK = "/home/flocka/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K/k154-cb3/layers/layer-00.safetensors"
DIM, INTER, EXPERT, M = 5120, 2304, 0, 64

def read_st(path, names):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; hdr = json.loads(fh.read(n)); base = 8 + n; o = {}
        for nm in names:
            m = hdr[nm]; s, e = m["data_offsets"]; fh.seek(base + s)
            o[nm] = torch.from_numpy(np.frombuffer(fh.read(e - s), dtype=np.uint8).reshape(m["shape"]).copy())
        return o

raw = read_st(PACK, ["w1_lo", "w1_hi", "w1_cb", "s1"])
lo, hi = raw["w1_lo"][EXPERT].contiguous(), raw["w1_hi"][EXPERT].contiguous()
cb, s = raw["w1_cb"][EXPERT].contiguous(), raw["s1"][EXPERT].contiguous()
assert lo.shape == (INTER, DIM // 4) and hi.shape == (INTER, DIM // 8), (lo.shape, hi.shape)

W = dequant_cb3_v2(lo, hi, cb, s).float()            # [INTER, DIM]
torch.manual_seed(0)
X = (torch.randn(M, DIM) * 0.05).to(torch.bfloat16)  # what the engine feeds it

# Reference in float64 from the bf16 activations the kernel will actually see.
Y = (X.double() @ W.double().T)

for nm, a in (("lo", lo), ("hi", hi), ("cb", cb), ("s", s)):
    a.numpy().tofile(f"gx_{nm}.bin")
X.view(torch.uint16).numpy().tofile("gx_x.bin")      # raw bf16 bits
W.numpy().astype(np.float32).tofile("gx_w.bin")
Y.numpy().astype(np.float32).tofile("gx_y.bin")
open("gx_dims.txt", "w").write(f"{M} {INTER} {DIM}\n")
print(f"M={M} N={INTER} K={DIM}  |Y|max={Y.abs().max():.4f}  W nonzero={(W!=0).float().mean():.3f}")
