#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Offline NVFP4 (W4) -> W3 FFN weight repack tool for the Atlas engine.

W3 FORMAT SPEC (v1, "w3a16") ------------------------------------------------
  * 3-bit weight codes, ONE global 8-entry LUT mirroring the engine's
    E2M1 FP4 LUT approach (sign bit on top, magnitude below):

        W3_LUT[8] = { 0.0, 1.0, 2.0, 4.0, -0.0, -1.0, -2.0, -4.0 }
        code = (sign << 2) | mag          # mag in 0..3

    Magnitude set {0,1,2,4} is the e2m1-subset; empirically beats the
    linear {0,1,2,3} codebook by ~15% relMSE on AEON-Q36-27B FFN weights
    (see w3_sensitivity.py).

  * Packing: 8 weights -> 3 bytes ("octet"), byte-aligned, little-endian
    24-bit word:  u24 = sum_{i=0..7} code_i << (3*i)
        byte 0 = u24 & 0xFF, byte 1 = (u24 >> 8) & 0xFF, byte 2 = u24 >> 16
    Non-transposed (GEMV) layout: [N, 3*K/8] u8 row-major (octet j of row n
    occupies bytes [n, 3j .. 3j+2]).  Transposed (GEMM) layout is built by
    the Rust loader: [3*K/8, N_pad64] (row 3j+b = byte-plane b of octet j).

  * Group scales: UNCHANGED scheme from NVFP4 — one FP8-E4M3 scale per
    16 consecutive K values: [N, K/16] u8.  Scale VALUES are re-optimized
    per group (candidate-factor sweep, min-MSE, e4m3-snapped BEFORE
    evaluation so the sim matches the runtime dequant bit-for-bit).

  * Per-tensor scale2 (f32): scale2_w3 = scale2_w4 * (6/4) = 1.5x.
    Rescaling keeps group scales inside the finite e4m3 range (Lmax drops
    6 -> 4; without this the largest groups overflow 448 -> NaN scales).

  * Dequant contract (must match kernels + Rust host mirror EXACTLY):
        sv = f32(e4m3_decode(scale_byte)) * scale2        # f32 multiply
        w  = W3_LUT[code] * sv                            # f32 multiply
    The m32_n64 GEMM additionally rounds w through FP8 E4M3
    (cvt.rn.satfinite) before the MMA — mirrored by `dequant_w3_e4m3`.

  * Bytes: packed 3/8 B/wt vs 4/8 (-25%); incl. scales 0.4375 vs 0.5625
    B/wt (-22.2%).

SIDECAR FILE -----------------------------------------------------------------
  safetensors-format file (default <model>/w3_ffn_sidecar.safetensors) with,
  per selected layer i and proj in {gate_proj, up_proj, down_proj}:
      model.language_model.layers.{i}.mlp.{proj}.w3_weight        U8  [N, 3K/8]
      model.language_model.layers.{i}.mlp.{proj}.w3_weight_scale  U8  [N, K/16]
      model.language_model.layers.{i}.mlp.{proj}.w3_weight_scale_2 F32 [1]
  The Rust loader (ATLAS_FFN_W3_LAYERS gate) picks these up at model load.

USAGE ------------------------------------------------------------------------
  python3 repack_w3.py --model /path/to/models/AEON-Q36-27B-Full \
      --layers 3,7,12 --out /path/to/models/AEON-Q36-27B-Full/w3_ffn_sidecar.safetensors
  python3 repack_w3.py --golden       # print golden vectors for the Rust test
"""

import argparse
import json
import struct
import sys

import numpy as np

# ── Format constants ─────────────────────────────────────────────────────────
W3_MAGS = np.array([0.0, 1.0, 2.0, 4.0], dtype=np.float32)
W3_LUT = np.concatenate([W3_MAGS, -W3_MAGS]).astype(np.float32)  # [8]
W3_LMAX = 4.0
W4_LMAX = 6.0
SCALE2_RESCALE = np.float32(W4_LMAX / W3_LMAX)  # 1.5
GROUP_SIZE = 16

E2M1_LUT = np.array(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
    dtype=np.float32,
)

# ── FP8 E4M3 (fn variant: bias 7, no inf, NaN=S.1111.111, max 448) ──────────


def _build_e4m3_decode_table() -> np.ndarray:
    """256-entry byte -> f32 decode table for FP8 E4M3 (e4m3fn)."""
    out = np.zeros(256, dtype=np.float32)
    for b in range(256):
        s = -1.0 if (b & 0x80) else 1.0
        e = (b >> 3) & 0xF
        m = b & 0x7
        if e == 0xF and m == 0x7:
            out[b] = np.nan
        elif e == 0:
            out[b] = s * (m / 8.0) * 2.0**-6  # subnormal
        else:
            out[b] = s * (1.0 + m / 8.0) * 2.0 ** (e - 7)
    return out


E4M3_DECODE = _build_e4m3_decode_table()
# All finite non-negative magnitudes, sorted, for RNE encode.
_E4M3_POS = np.sort(np.unique(E4M3_DECODE[np.isfinite(E4M3_DECODE) & (E4M3_DECODE >= 0)]))
# Byte for each positive magnitude value.
_E4M3_POS_BYTE = np.zeros(len(_E4M3_POS), dtype=np.uint8)
for _b in range(0x80):
    v = E4M3_DECODE[_b]
    if np.isfinite(v):
        _E4M3_POS_BYTE[np.searchsorted(_E4M3_POS, v)] = _b


def e4m3_decode(bytes_u8: np.ndarray) -> np.ndarray:
    return E4M3_DECODE[bytes_u8.astype(np.uint8)]


def e4m3_encode(x: np.ndarray) -> np.ndarray:
    """f32 -> nearest finite e4m3 byte, round-to-nearest-even, saturating.

    Matches CUDA `cvt.rn.satfinite.e4m3x2.f32` on finite inputs.
    """
    x = np.asarray(x, dtype=np.float32)
    sign = np.signbit(x)
    a = np.clip(np.abs(x), 0.0, 448.0)
    idx = np.searchsorted(_E4M3_POS, a)  # first pos >= a
    idx = np.clip(idx, 0, len(_E4M3_POS) - 1)
    lo = np.clip(idx - 1, 0, len(_E4M3_POS) - 1)
    below = _E4M3_POS[lo]
    above = _E4M3_POS[idx]
    d_below = a - below
    d_above = above - a
    pick_below = (d_below < d_above) | (
        (d_below == d_above) & (_E4M3_POS_BYTE[lo] % 2 == 0)
    )
    chosen = np.where(pick_below, lo, idx)
    byte = _E4M3_POS_BYTE[chosen]
    return (byte | np.where(sign, 0x80, 0)).astype(np.uint8)


# ── W4 (NVFP4) dequant ───────────────────────────────────────────────────────


def dequant_w4(packed: np.ndarray, scale: np.ndarray, scale2: float, n: int, k: int):
    """[N, K/2] u8 + [N, K/16] e4m3 + f32 -> [N, K] f32."""
    codes = np.empty((n, k), dtype=np.uint8)
    codes[:, 0::2] = packed & 0xF
    codes[:, 1::2] = packed >> 4
    w = E2M1_LUT[codes].reshape(n, k // GROUP_SIZE, GROUP_SIZE)
    sv = e4m3_decode(scale).reshape(n, k // GROUP_SIZE, 1) * np.float32(scale2)
    return (w * sv).reshape(n, k)


# ── W3 requant / pack / dequant ─────────────────────────────────────────────


def requant_w3_groups(w, scale2_w3, factors):
    """Requantize [G, 16] f32 groups to W3.

    Per group: sweep candidate scales (e4m3-snapped), nearest-level codes,
    keep the min-MSE candidate. Returns (codes [G,16] u8 in 0..7,
    scale_bytes [G] u8, sq_err [G] f32).
    """
    g = w.shape[0]
    gmax = np.abs(w).max(axis=1, keepdims=True)  # [G,1]
    base = gmax / (W3_LMAX * scale2_w3)
    base[base == 0] = 1.0  # all-zero group: any scale, codes all 0
    bounds = (W3_MAGS[:-1] + W3_MAGS[1:]) / 2.0  # [0.5, 1.5, 3.0]

    best_err = np.full((g,), np.inf, dtype=np.float32)
    best_scale_byte = np.zeros((g,), dtype=np.uint8)
    best_codes = np.zeros((g, GROUP_SIZE), dtype=np.uint8)

    for fac in factors:
        sb = e4m3_encode(base * np.float32(fac))  # [G,1] u8
        s = e4m3_decode(sb)  # snapped, [G,1]
        s = np.where(s == 0, np.float32(1e-8), s)
        eff = (s * scale2_w3).astype(np.float32)
        a = np.abs(w) / eff
        mag_idx = np.searchsorted(bounds, a.ravel()).reshape(a.shape).astype(np.uint8)
        neg = np.signbit(w)
        codes = mag_idx | (neg.astype(np.uint8) << 2)
        # Reconstruct EXACTLY as the runtime dequant does:
        #   sv = e4m3 * scale2 (f32), w = LUT[code] * sv (f32)
        recon = W3_LUT[codes] * eff
        err = ((w - recon) ** 2).sum(axis=1)
        better = err < best_err
        best_err = np.where(better, err, best_err)
        best_scale_byte = np.where(better, sb.ravel(), best_scale_byte)
        best_codes = np.where(better[:, None], codes, best_codes)

    # All-zero groups: canonicalize to scale byte 0x38 (=1.0) and codes 0.
    zero_grp = (gmax.ravel() == 0)
    best_scale_byte = np.where(zero_grp, np.uint8(0x38), best_scale_byte)
    best_codes[zero_grp] = 0
    best_err = np.where(zero_grp, np.float32(0.0), best_err)
    return best_codes, best_scale_byte, best_err


def pack_w3(codes: np.ndarray) -> np.ndarray:
    """[..., K] codes (0..7) -> [..., 3K/8] u8, 8 weights per 3 bytes LE."""
    shp = codes.shape
    k = shp[-1]
    assert k % 8 == 0
    c = codes.reshape(-1, k // 8, 8).astype(np.uint32)
    u24 = np.zeros(c.shape[:2], dtype=np.uint32)
    for i in range(8):
        u24 |= c[:, :, i] << (3 * i)
    out = np.empty((c.shape[0], c.shape[1], 3), dtype=np.uint8)
    out[:, :, 0] = u24 & 0xFF
    out[:, :, 1] = (u24 >> 8) & 0xFF
    out[:, :, 2] = (u24 >> 16) & 0xFF
    return out.reshape(*shp[:-1], 3 * k // 8)


def unpack_w3(packed: np.ndarray, k: int) -> np.ndarray:
    """[..., 3K/8] u8 -> [..., K] codes (0..7)."""
    shp = packed.shape
    p = packed.reshape(-1, k // 8, 3).astype(np.uint32)
    u24 = p[:, :, 0] | (p[:, :, 1] << 8) | (p[:, :, 2] << 16)
    codes = np.empty((p.shape[0], k // 8, 8), dtype=np.uint8)
    for i in range(8):
        codes[:, :, i] = (u24 >> (3 * i)) & 7
    return codes.reshape(*shp[:-1], k)


def dequant_w3(packed, scale_bytes, scale2, n, k):
    """Runtime-exact W3 dequant (f32 path, mirrors w3a16_gemv*)."""
    codes = unpack_w3(packed.reshape(n, 3 * k // 8), k)
    sv = (e4m3_decode(scale_bytes).reshape(n, k // GROUP_SIZE, 1)
          * np.float32(scale2)).astype(np.float32)
    w = (W3_LUT[codes].reshape(n, k // GROUP_SIZE, GROUP_SIZE) * sv)
    return w.reshape(n, k).astype(np.float32)


def dequant_w3_e4m3(packed, scale_bytes, scale2, n, k):
    """W3 dequant with the m32_n64 GEMM's extra FP8-E4M3 round-trip."""
    w = dequant_w3(packed, scale_bytes, scale2, n, k)
    return e4m3_decode(e4m3_encode(w))


# ── safetensors I/O (minimal, dependency-free) ──────────────────────────────


class SafetensorsReader:
    def __init__(self, path):
        self.f = open(path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.header = json.loads(self.f.read(n))
        self.base = 8 + n

    def tensor(self, name):
        m = self.header[name]
        off = m["data_offsets"]
        self.f.seek(self.base + off[0])
        buf = self.f.read(off[1] - off[0])
        return np.frombuffer(buf, dtype=np.uint8).copy(), m["dtype"], m["shape"]

    def scalar_f32(self, name):
        raw, dt, _ = self.tensor(name)
        assert dt == "F32", f"{name}: {dt}"
        return float(np.frombuffer(raw.tobytes(), dtype=np.float32)[0])


def write_safetensors(path, tensors):
    """tensors: dict name -> (bytes, dtype_str, shape_list)."""
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


# ── Per-tensor repack ────────────────────────────────────────────────────────

PROJS = ("gate_proj", "up_proj", "down_proj")


def repack_tensor(reader, prefix, factors, row_chunk=2048, row_stride=1):
    """Repack one FFN projection W4 -> W3.

    Streams in row chunks to bound RAM. `row_stride > 1` subsamples rows
    (stats-only mode for the sensitivity script). Returns dict with packed
    arrays (None when subsampling) + error stats.
    """
    pk_raw, dt, shape = reader.tensor(prefix + ".weight")
    assert dt == "U8", f"{prefix}: expected packed NVFP4 U8, got {dt}"
    n, kh = shape
    k = kh * 2
    pk = pk_raw.reshape(n, kh)
    sc_raw, _, _ = reader.tensor(prefix + ".weight_scale")
    sc = sc_raw.reshape(n, k // GROUP_SIZE)
    scale2_w4 = reader.scalar_f32(prefix + ".weight_scale_2")
    scale2_w3 = np.float32(scale2_w4) * SCALE2_RESCALE

    full = row_stride == 1
    w3_packed = np.empty((n, 3 * k // 8), dtype=np.uint8) if full else None
    w3_scale = np.empty((n, k // GROUP_SIZE), dtype=np.uint8) if full else None

    sq_err = 0.0
    sq_ref = 0.0
    max_err = 0.0
    rows = range(0, n, row_chunk)
    for r0 in rows:
        r1 = min(r0 + row_chunk, n)
        sel = slice(r0, r1, row_stride)
        w4 = dequant_w4(pk[sel], sc[sel], scale2_w4, len(range(r0, r1, row_stride)), k)
        grp = w4.reshape(-1, GROUP_SIZE)
        codes, sbytes, err = requant_w3_groups(grp, scale2_w3, factors)
        nrows = grp.shape[0] // (k // GROUP_SIZE)
        if full:
            w3_packed[r0:r1] = pack_w3(codes.reshape(nrows, k))
            w3_scale[r0:r1] = sbytes.reshape(nrows, k // GROUP_SIZE)
        sq_err += float(err.sum())
        sq_ref += float((w4 ** 2).sum())
        # max abs err (recompute recon for the chunk via runtime dequant path)
        sv = (e4m3_decode(sbytes).reshape(-1, 1) * scale2_w3).astype(np.float32)
        recon = W3_LUT[codes] * sv
        max_err = max(max_err, float(np.abs(grp - recon).max()))

    rel_mse = sq_err / max(sq_ref, 1e-30)
    return {
        "packed": w3_packed,
        "scale": w3_scale,
        "scale2": float(scale2_w3),
        "n": n,
        "k": k,
        "rel_mse": rel_mse,
        "max_abs_err": max_err,
        "rms_ref": (sq_ref / (n // row_stride * k)) ** 0.5,
    }


def layer_prefix(i, proj):
    return f"model.language_model.layers.{i}.mlp.{proj}"


# ── Golden vectors for the Rust unit test ───────────────────────────────────


def emit_golden():
    """Deterministic golden case: 1 row, K=32 (2 groups), fixed codes/scales."""
    rng_codes = np.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0,
         2, 2, 5, 0, 1, 7, 3, 6, 4, 0, 2, 1, 6, 5, 7, 3], dtype=np.uint8)
    scale_bytes = np.array([0x44, 0x2E], dtype=np.uint8)  # e4m3: 6.0, 0.21875
    scale2 = np.float32(0.0123456)
    packed = pack_w3(rng_codes[None, :])
    assert np.array_equal(unpack_w3(packed, 32)[0], rng_codes)
    deq = dequant_w3(packed, scale_bytes[None, :], scale2, 1, 32)[0]
    deq_e4m3 = dequant_w3_e4m3(packed, scale_bytes[None, :], scale2, 1, 32)[0]
    print("// Auto-generated by local/tools/repack_w3.py --golden")
    print("codes:", list(rng_codes))
    print("packed bytes:", list(packed[0]))
    print("scale bytes:", list(scale_bytes))
    print("scale2 bits:", hex(np.float32(scale2).view(np.uint32)))
    print("dequant f32 bits:", [hex(v) for v in deq.view(np.uint32)])
    print("dequant e4m3 f32 bits:", [hex(v) for v in deq_e4m3.view(np.uint32)])


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="/path/to/models/AEON-Q36-27B-Full")
    ap.add_argument("--layers", default="", help="comma list, e.g. 3,7,12 (or 'all')")
    ap.add_argument("--projs", default="gate_proj,up_proj,down_proj")
    ap.add_argument("--out", default=None, help="sidecar path (default <model>/w3_ffn_sidecar.safetensors)")
    ap.add_argument("--stats-out", default=None, help="JSON stats path")
    ap.add_argument("--factors", type=int, default=13)
    ap.add_argument("--fmin", type=float, default=0.75)
    ap.add_argument("--fmax", type=float, default=1.25)
    ap.add_argument("--golden", action="store_true", help="print Rust golden vectors and exit")
    args = ap.parse_args()

    if args.golden:
        emit_golden()
        return

    reader = SafetensorsReader(f"{args.model}/model.safetensors")
    if args.layers == "all":
        layers = sorted(
            {int(k.split(".layers.")[1].split(".")[0])
             for k in reader.header if ".mlp.gate_proj.weight" in k})
    else:
        layers = sorted({int(x) for x in args.layers.split(",") if x.strip() != ""})
    if not layers:
        ap.error("--layers required (comma list or 'all')")
    projs = [p.strip() for p in args.projs.split(",")]
    factors = np.linspace(args.fmin, args.fmax, args.factors)

    out_path = args.out or f"{args.model}/w3_ffn_sidecar.safetensors"
    tensors = {}
    stats = {}
    for i in layers:
        for proj in projs:
            prefix = layer_prefix(i, proj)
            r = repack_tensor(reader, prefix, factors)
            tensors[prefix + ".w3_weight"] = (
                r["packed"].tobytes(), "U8", [r["n"], 3 * r["k"] // 8])
            tensors[prefix + ".w3_weight_scale"] = (
                r["scale"].tobytes(), "U8", [r["n"], r["k"] // GROUP_SIZE])
            tensors[prefix + ".w3_weight_scale_2"] = (
                np.float32(r["scale2"]).tobytes(), "F32", [1])
            stats[prefix] = {k: r[k] for k in ("rel_mse", "max_abs_err", "rms_ref", "n", "k")}
            print(f"{prefix}: relMSE={r['rel_mse']:.5f} maxerr={r['max_abs_err']:.4f} "
                  f"rms={r['rms_ref']:.4f}", flush=True)

    write_safetensors(out_path, tensors)
    total = sum(len(b) for b, _, _ in tensors.values())
    print(f"\nwrote {out_path} ({total / 1e6:.1f} MB, {len(tensors)} tensors, "
          f"{len(layers)} layers)")
    if args.stats_out:
        with open(args.stats_out, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"stats -> {args.stats_out}")


if __name__ == "__main__":
    main()
