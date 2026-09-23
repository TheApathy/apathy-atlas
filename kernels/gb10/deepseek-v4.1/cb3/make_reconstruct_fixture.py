#!/usr/bin/env python3
"""Fixture for `cb3_reconstruct.cu`, from the REAL deployed pack.

N=32 rows x K=1024 covers two full 512-blocks, so every scale group g (0..15),
every lane, both r sub-positions AND the block-index arithmetic are exercised.
A K=256 slice would leave g=8..15 and all block indexing untested.

The reference comes from `unpack_cb3_v2` — the K-order settled in
CB3_FORMAT.md and measured against the shipped Triton kernel at rel_l2 2.3e-03
with three wrong orderings at ~1.35.
"""
import sys, json, struct, numpy as np, torch

sys.path.insert(0, "/home/flocka/atlas/dsv41-prefill-work/tools")
from cb3 import dequant_cb3_v2

PACK = "/home/flocka/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K/k154-cb3/layers/layer-00.safetensors"
N, K, EXPERT = 32, 1024, 0


def read_st(path, names):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
        base = 8 + n
        out = {}
        for nm in names:
            m = hdr[nm]
            s, e = m["data_offsets"]
            fh.seek(base + s)
            buf = np.frombuffer(fh.read(e - s), dtype=np.uint8).reshape(m["shape"]).copy()
            out[nm] = torch.from_numpy(buf)
        return out


raw = read_st(PACK, ["w1_lo", "w1_hi", "w1_cb", "s1"])
lo = raw["w1_lo"][EXPERT][:N, : K // 4].contiguous()
hi = raw["w1_hi"][EXPERT][:N, : K // 8].contiguous()
cb = raw["w1_cb"][EXPERT][:N].contiguous()
s = raw["s1"][EXPERT][:N, : K // 32].contiguous()

W = dequant_cb3_v2(lo, hi, cb, s).float().numpy()
for nm, arr in (("lo", lo), ("hi", hi), ("cb", cb), ("s", s)):
    arr.numpy().tofile(f"fx_{nm}.bin")
W.astype(np.float32).tofile("fx_ref.bin")
open("fx_dims.txt", "w").write(f"{N} {K}\n")
print(f"fixture N={N} K={K} nonzero={(W != 0).mean():.3f} absmax={abs(W).max()}")
