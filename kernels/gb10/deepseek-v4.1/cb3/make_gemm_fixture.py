#!/usr/bin/env python3
"""Float64 ground truth for one CB3 expert tile, from the REAL pack.

Not another GPU implementation: the reference is computed in float64 from the
shipped bytes, so a CUDA GEMM checked against it is being checked against the
arithmetic rather than against a second opinion that could share its bug.

The index reconstruction here is the SEMANTIC form (lo two bits, hi one bit,
codebook lookup). `cb3_decode.cuh` reaches the same answer by a packed-nibble
PTX route; that route is already pinned bit-exact by cb3_decode_vectors.txt, so
the two are independent descriptions of one format.
"""
import json, struct, sys
import numpy as np

PACK = "/home/flocka/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K/k154-cb3/layers/layer-00.safetensors"
# e2m1: 4-bit float, sign in bit 3, the eight magnitudes below.
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float64)

def read(path, names):
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
        base = 8 + n
        out = {}
        for k in names:
            e = hdr[k]
            s, t = e["data_offsets"]
            fh.seek(base + s)
            out[k] = np.frombuffer(fh.read(t - s), dtype=np.uint8).reshape(e["shape"])
        return out

def e2m1_of(nibble):
    return np.where(nibble & 8, -1.0, 1.0) * E2M1[nibble & 7]

def dequant_rows(lo, hi, cb, scales, K, rows):
    """[rows, K] float64 weights for one expert."""
    W = np.zeros((rows, K), dtype=np.float64)
    for r in range(rows):
        loj = np.unpackbits(lo[r], bitorder="little").reshape(-1, 2)   # 2 bits/weight
        hij = np.unpackbits(hi[r], bitorder="little")                  # 1 bit/weight
        idx = (loj[:K, 0] | (loj[:K, 1] << 1) | (hij[:K] << 2)).astype(np.int64)
        vals = e2m1_of(cb[r][idx].astype(np.int64) & 0xF)
        # UE8M0 scales: one exponent per 32 weights, applied per group.
        sc = np.repeat(scales[r][: K // 32].astype(np.int64), 32)
        W[r] = vals * np.exp2(sc.astype(np.float64) - 127.0)
    return W

if __name__ == "__main__":
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    K = 5120
    t = read(PACK, ["w1_lo", "w1_hi", "w1_cb", "s1"])
    lo, hi, cb, s = t["w1_lo"][0], t["w1_hi"][0], t["w1_cb"][0], t["s1"][0]
    print(f"expert 0  w1: lo{lo.shape} hi{hi.shape} cb{cb.shape} s{s.shape}")
    W = dequant_rows(lo[:rows], hi[:rows], cb[:rows], s[:rows], K, rows)
    rng = np.random.default_rng(20260921)
    X = rng.standard_normal((4, K))                       # 4 activation rows
    Y = X @ W.T                                            # float64 reference
    np.save("fixture_W.npy", W); np.save("fixture_X.npy", X); np.save("fixture_Y.npy", Y)
    nz = int((W != 0).sum())
    print(f"  W nonzero {nz}/{W.size} ({100*nz/W.size:.1f}%)  |W|max {np.abs(W).max():.4g}")
    print(f"  Y shape {Y.shape}  |Y|mean {np.abs(Y).mean():.4g}")
    assert nz > 0, "an all-zero weight tile would make any GEMM look correct"
