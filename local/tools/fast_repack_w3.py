#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Fast GPU/CPU-vectorized W3 repacker for Qwen3.8 / Atlas.

Matches `repack_w3.py` bit-for-bit, but runs full tensor quantization in parallel.
"""

import argparse
import json
import os
import struct
import sys
import time
import numpy as np
import torch

W3_MAGS_NP = np.array([0.0, 1.0, 2.0, 4.0], dtype=np.float32)
W3_LUT_NP = np.concatenate([W3_MAGS_NP, -W3_MAGS_NP]).astype(np.float32)  # [8]
W3_LMAX = 4.0
W4_LMAX = 6.0
SCALE2_RESCALE = 1.5
GROUP_SIZE = 16

E2M1_LUT_NP = np.array(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
    dtype=np.float32,
)

def _build_e4m3_decode_table() -> np.ndarray:
    out = np.zeros(256, dtype=np.float32)
    for b in range(256):
        s = -1.0 if (b & 0x80) else 1.0
        e = (b >> 3) & 0xF
        m = b & 0x7
        if e == 0xF and m == 0x7:
            out[b] = np.nan
        elif e == 0:
            out[b] = s * (m / 8.0) * 2.0**-6
        else:
            out[b] = s * (1.0 + m / 8.0) * 2.0 ** (e - 7)
    return out

E4M3_DECODE_NP = _build_e4m3_decode_table()
_E4M3_POS = np.sort(np.unique(E4M3_DECODE_NP[np.isfinite(E4M3_DECODE_NP) & (E4M3_DECODE_NP >= 0)]))
_E4M3_POS_BYTE = np.zeros(len(_E4M3_POS), dtype=np.uint8)
for _b in range(0x80):
    v = E4M3_DECODE_NP[_b]
    if np.isfinite(v):
        _E4M3_POS_BYTE[np.searchsorted(_E4M3_POS, v)] = _b

class SafetensorsReader:
    def __init__(self, path):
        self.files = {}
        self.headers = {}
        self.bases = {}
        self.weight_map = {}
        self.header = {}
        if os.path.isdir(path):
            index_path = os.path.join(path, "model.safetensors.index.json")
            single_path = os.path.join(path, "model.safetensors")
            if os.path.exists(index_path):
                with open(index_path, "r") as f:
                    idx = json.load(f)
                self.weight_map = idx.get("weight_map", {})
                self.dir = path
                self.header = {k: None for k in self.weight_map}
            elif os.path.exists(single_path):
                self._load_file(single_path, default=True)
            else:
                raise FileNotFoundError(f"No safetensors found in directory: {path}")
        else:
            self._load_file(path, default=True)

    def _load_file(self, path, default=False):
        f = open(path, "rb")
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        fname = os.path.basename(path)
        self.files[fname] = f
        self.headers[fname] = header
        self.bases[fname] = base
        if default:
            self.header = header
            self.weight_map = {k: fname for k in header}
        return fname

    def _get_file_and_meta(self, name):
        if name not in self.weight_map:
            raise KeyError(f"Tensor {name} not found in weights")
        fname = self.weight_map[name]
        if fname not in self.files:
            fpath = os.path.join(self.dir, fname) if hasattr(self, "dir") else fname
            self._load_file(fpath)
        return self.files[fname], self.headers[fname][name], self.bases[fname]

    def tensor(self, name):
        f, m, base = self._get_file_and_meta(name)
        off = m["data_offsets"]
        f.seek(base + off[0])
        buf = f.read(off[1] - off[0])
        return np.frombuffer(buf, dtype=np.uint8).copy(), m["dtype"], m["shape"]

    def scalar_f32(self, name):
        raw, dt, _ = self.tensor(name)
        assert dt == "F32", f"{name}: {dt}"
        return float(np.frombuffer(raw.tobytes(), dtype=np.float32)[0])


def write_safetensors(path, tensors):
    header = {}
    off = 0
    for name, (buf, dt, shape) in tensors.items():
        header[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + len(buf)]}
        off += len(buf)
    hj = json.dumps(header, sort_keys=True).encode()
    pad = (8 - len(hj) % 8) % 8
    hj += b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for name, (buf, _, _) in tensors.items():
            f.write(buf)


# Torch constants on device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
E4M3_POS_T = torch.from_numpy(_E4M3_POS).to(DEVICE, dtype=torch.float32)
E4M3_POS_BYTE_T = torch.from_numpy(_E4M3_POS_BYTE).to(DEVICE, dtype=torch.uint8)
E4M3_DECODE_T = torch.from_numpy(E4M3_DECODE_NP).to(DEVICE, dtype=torch.float32)
E2M1_LUT_T = torch.from_numpy(E2M1_LUT_NP).to(DEVICE, dtype=torch.float32)
W3_LUT_T = torch.from_numpy(W3_LUT_NP).to(DEVICE, dtype=torch.float32)
BOUNDS_T = torch.tensor([0.5, 1.5, 3.0], device=DEVICE, dtype=torch.float32)


def torch_e4m3_encode(x: torch.Tensor) -> torch.Tensor:
    """Vectorized f32 -> nearest e4m3 byte tensor."""
    sign = torch.signbit(x)
    a = torch.clamp(torch.abs(x), 0.0, 448.0)
    idx = torch.searchsorted(E4M3_POS_T, a)
    idx = torch.clamp(idx, 0, len(_E4M3_POS) - 1)
    lo = torch.clamp(idx - 1, 0, len(_E4M3_POS) - 1)
    below = E4M3_POS_T[lo]
    above = E4M3_POS_T[idx]
    d_below = a - below
    d_above = above - a
    below_byte = E4M3_POS_BYTE_T[lo]
    pick_below = (d_below < d_above) | ((d_below == d_above) & ((below_byte % 2) == 0))
    chosen = torch.where(pick_below, lo, idx)
    byte = E4M3_POS_BYTE_T[chosen]
    return torch.where(sign, byte | 0x80, byte)


def torch_e4m3_decode(b: torch.Tensor) -> torch.Tensor:
    return E4M3_DECODE_T[b.long()]


def torch_repack_tensor(reader, prefix, factors_list):
    pk_raw, dt, shape = reader.tensor(prefix + ".weight")
    n, kh = shape
    k = kh * 2
    sc_raw, _, _ = reader.tensor(prefix + ".weight_scale")
    scale2_w4 = reader.scalar_f32(prefix + ".weight_scale_2")
    scale2_w3 = scale2_w4 * SCALE2_RESCALE

    # Move to GPU
    pk_t = torch.from_numpy(pk_raw.reshape(n, kh)).to(DEVICE)
    sc_t = torch.from_numpy(sc_raw.reshape(n, k // GROUP_SIZE)).to(DEVICE)

    # Dequant NVFP4
    codes_t = torch.empty((n, k), dtype=torch.uint8, device=DEVICE)
    codes_t[:, 0::2] = pk_t & 0xF
    codes_t[:, 1::2] = pk_t >> 4
    w_raw = E2M1_LUT_T[codes_t.long()].reshape(n, k // GROUP_SIZE, GROUP_SIZE)
    sv_raw = torch_e4m3_decode(sc_t).reshape(n, k // GROUP_SIZE, 1) * scale2_w4
    w = (w_raw * sv_raw).reshape(-1, GROUP_SIZE)  # [G, 16]

    g = w.shape[0]
    gmax = torch.abs(w).max(dim=1, keepdim=True).values  # [G, 1]
    base = gmax / (W3_LMAX * scale2_w3)
    base = torch.where(base == 0, torch.tensor(1.0, device=DEVICE), base)

    best_err = torch.full((g,), float("inf"), dtype=torch.float32, device=DEVICE)
    best_sb = torch.zeros((g,), dtype=torch.uint8, device=DEVICE)
    best_codes = torch.zeros((g, GROUP_SIZE), dtype=torch.uint8, device=DEVICE)

    for fac in factors_list:
        sb = torch_e4m3_encode(base * float(fac))  # [G, 1]
        s = torch_e4m3_decode(sb)
        s = torch.where(s == 0, torch.tensor(1e-8, device=DEVICE), s)
        eff = s * scale2_w3
        a = torch.abs(w) / eff  # [G, 16]
        mag_idx = torch.bucketize(a, BOUNDS_T).to(torch.uint8)  # 0..3
        neg = torch.signbit(w).to(torch.uint8)
        codes = mag_idx | (neg << 2)
        recon = W3_LUT_T[codes.long()] * eff
        err = torch.sum((w - recon) ** 2, dim=1)
        better = err < best_err
        best_err = torch.where(better, err, best_err)
        best_sb = torch.where(better, sb.squeeze(1), best_sb)
        best_codes = torch.where(better.unsqueeze(1), codes, best_codes)

    # Zero groups
    zero_grp = (gmax.squeeze(1) == 0)
    best_sb = torch.where(zero_grp, torch.tensor(0x38, dtype=torch.uint8, device=DEVICE), best_sb)
    best_codes[zero_grp] = 0

    # Pack W3
    codes_cpu = best_codes.view(n, k).cpu().numpy()
    scale_cpu = best_sb.view(n, k // GROUP_SIZE).cpu().numpy()

    # Pack 8 codes -> 3 bytes LE
    c = codes_cpu.reshape(-1, k // 8, 8).astype(np.uint32)
    u24 = np.zeros(c.shape[:2], dtype=np.uint32)
    for i in range(8):
        u24 |= c[:, :, i] << (3 * i)
    w3_packed = np.empty((c.shape[0], c.shape[1], 3), dtype=np.uint8)
    w3_packed[:, :, 0] = u24 & 0xFF
    w3_packed[:, :, 1] = (u24 >> 8) & 0xFF
    w3_packed[:, :, 2] = (u24 >> 16) & 0xFF
    w3_packed = w3_packed.reshape(n, 3 * k // 8)

    sq_err = float(best_err.sum().cpu())
    sq_ref = float(torch.sum(w ** 2).cpu())
    rel_mse = sq_err / max(sq_ref, 1e-30)

    return {
        "packed": w3_packed,
        "scale": scale_cpu,
        "scale2": scale2_w3,
        "n": n,
        "k": k,
        "rel_mse": rel_mse,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/flocka/atlas/qwen38/optimized-qwen")
    ap.add_argument("--layers", default="")
    ap.add_argument("--projs", default="gate_proj,up_proj,down_proj")
    ap.add_argument("--out", default=None)
    ap.add_argument("--factors", type=int, default=13)
    ap.add_argument("--fmin", type=float, default=0.75)
    ap.add_argument("--fmax", type=float, default=1.25)
    args = ap.parse_args()

    reader = SafetensorsReader(args.model)
    layers = sorted({int(x) for x in args.layers.split(",") if x.strip() != ""})
    projs = [p.strip() for p in args.projs.split(",")]
    factors = np.linspace(args.fmin, args.fmax, args.factors).tolist()

    out_path = args.out or f"{args.model}/w3_ffn_sidecar.safetensors"
    tensors = {}

    t0 = time.time()
    for i in layers:
        l_t0 = time.time()
        for proj in projs:
            prefix = f"model.language_model.layers.{i}.mlp.{proj}"
            r = torch_repack_tensor(reader, prefix, factors)
            tensors[prefix + ".w3_weight"] = (
                r["packed"].tobytes(), "U8", [r["n"], 3 * r["k"] // 8]
            )
            tensors[prefix + ".w3_weight_scale"] = (
                r["scale"].tobytes(), "U8", [r["n"], r["k"] // GROUP_SIZE]
            )
            tensors[prefix + ".w3_weight_scale_2"] = (
                np.float32(r["scale2"]).tobytes(), "F32", [1]
            )
            print(f"{prefix}: relMSE={r['rel_mse']:.5f}", flush=True)
        print(f"Layer {i} done in {time.time() - l_t0:.2f}s", flush=True)

    write_safetensors(out_path, tensors)
    total = sum(len(b) for b, _, _ in tensors.values())
    print(f"\nWrote {out_path} ({total / 1e6:.1f} MB, {len(tensors)} tensors, {len(layers)} layers in {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
