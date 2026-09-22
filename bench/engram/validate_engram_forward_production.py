#!/usr/bin/env python3
"""Same check as validate_engram_forward.py, but running PRODUCTION'S OWN
tools/v41_ref.py EngramWeights + engram_forward unmodified, not a hand
reproduction. Rules out "the spec was mistranscribed" as an explanation for any
residual: if this script and validate_engram_forward.py disagree, the hand
version has a bug; if they agree, the hand version is a faithful copy and any
gap vs the oracle capture is either input provenance or engine-vs-CPU numerics,
not a spec error.

Needs tools/v41_ref.py importable (no GPU, no triton required for the branch
this exercises -- fp4_linear/triton imports are soft-failed inside v41_ref.py).

Usage: validate_engram_forward_production.py [v41-ref-dir] [ref-dir] [occurrence]
"""
import sys
import os
import json
import struct
import numpy as np
import torch

V41_REF_DIR = sys.argv[1] if len(sys.argv) > 1 else "/home/flocka/atlas/dsv41-prefill-work/tools"
REF_DIR = sys.argv[2] if len(sys.argv) > 2 else "/home/flocka/atlas/DSV41_PORT/oracle/ref/runE_image"
OCC = sys.argv[3] if len(sys.argv) > 3 else "000"
LAYER = 1
MODEL_DIR = "/home/flocka/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K"

sys.path.insert(0, V41_REF_DIR)
os.environ.setdefault("DSV41_DENSE_FP8", "1")  # matches every oracle capture's env
import v41_ref as R  # noqa: E402


def load_bf16_bin(path, shape):
    raw = np.fromfile(path, dtype=np.uint16)
    f32 = (raw.astype(np.uint32) << 16).view(np.float32)
    return torch.from_numpy(f32.copy()).reshape(shape).to(torch.bfloat16)


def load_f32_bin(path, shape):
    return torch.from_numpy(np.fromfile(path, dtype=np.float32).copy()).reshape(shape)


def make_getter():
    with open(f"{MODEL_DIR}/model.safetensors.index.json") as f:
        wm = json.load(f)["weight_map"]
    cache = {}

    def get(name):
        if name in cache:
            return cache[name]
        shard = wm[name]
        with open(f"{MODEL_DIR}/{shard}", "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
            meta = hdr[name]
            dtmap = {"F8_E4M3": torch.float8_e4m3fn, "F8_E8M0": torch.uint8, "BF16": torch.bfloat16}
            dt = dtmap[meta["dtype"]]
            start, end = meta["data_offsets"]
            f.seek(8 + n + start)
            t = torch.frombuffer(bytearray(f.read(end - start)), dtype=dt).reshape(meta["shape"])
            cache[name] = t
            return t

    return get


def main():
    with open(f"{REF_DIR}/manifest.json") as f:
        manifest = json.load(f)
    ent = {e["file"]: e for e in manifest["tensors"]}
    h_shape = ent[f"L00.h.{OCC}.bin"]["shape"]
    rows_shape = ent[f"L01.engram_rows.{OCC}.bin"]["shape"]
    out_shape = ent[f"L01.engram_out.{OCC}.bin"]["shape"]

    h = load_bf16_bin(f"{REF_DIR}/L00.h.{OCC}.bin", h_shape)
    rows = load_f32_bin(f"{REF_DIR}/L01.engram_rows.{OCC}.bin", rows_shape)
    want = load_bf16_bin(f"{REF_DIR}/L01.engram_out.{OCC}.bin", out_shape).float()

    get = make_getter()
    ew = R.EngramWeights(get, LAYER, "cpu")
    print("wkv type:", type(ew.wkv).__name__)

    R.MM_TILE = 16  # matches engine/model.py:65
    args = R.Args(dim=h_shape[-1], hc_mult=h_shape[1], norm_eps=1e-20)
    got = R.engram_forward(h, rows, ew, args).float()

    diff = (got - want).abs()
    rel_l2 = (diff.pow(2).sum() / want.pow(2).sum().clamp_min(1e-30)).sqrt().item()
    print(f"PRODUCTION v41_ref.EngramWeights + engram_forward: rel_l2={rel_l2:.9e} max_abs={diff.max().item():.6e}")


if __name__ == "__main__":
    main()
