#!/usr/bin/env python3
"""Validate the engram_forward math spec against the real oracle capture, on CPU.

This is a PRE-CHECK for the Rust implementation (which will run wkv through
whatever shared fp8-linear primitive dsv41-integrate builds): it re-derives
tools/v41_ref.py's engram_forward (v41_ref.py:782) from the checkpoint's real
weights and a real captured input, independent of any GPU kernel, to catch a
spec error before anyone builds against it.

Inputs, all real:
  - layers.1.engram.{wkv.weight,wkv.scale,q_weight,k_weight} from the checkpoint
  - h  = L00.h.<occ>.bin      (layer 1's engram_forward INPUT: the loop's h after
                                layer 0's full block, tapped at the loop's own
                                h-tap just before layer 1 is entered)
  - rows = L01.engram_rows.<occ>.bin   (POST dead-head-mask rows, already the
                                seam's contract)
Compared against: L01.engram_out.<occ>.bin.

Usage: validate_engram_forward.py [ref-dir] [occurrence]
"""
import sys
import json
import struct
import numpy as np
import torch
import torch.nn.functional as F

MODEL_DIR = "/home/flocka/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K"
REF_DIR = sys.argv[1] if len(sys.argv) > 1 else "/home/flocka/atlas/DSV41_PORT/oracle/ref/runE_image"
OCC = sys.argv[2] if len(sys.argv) > 2 else "000"
LAYER = 1
HC_MULT, DIM, EPS = 4, 5120, 1e-20


def load_bf16_bin(path, shape):
    raw = np.fromfile(path, dtype=np.uint16)
    f32 = (raw.astype(np.uint32) << 16).view(np.float32)
    return torch.from_numpy(f32.copy()).reshape(shape)


def load_f32_bin(path, shape):
    return torch.from_numpy(np.fromfile(path, dtype=np.float32).copy()).reshape(shape)


def find_shard(tensor_name):
    with open(f"{MODEL_DIR}/model.safetensors.index.json") as f:
        wm = json.load(f)["weight_map"]
    return wm[tensor_name]


def read_tensor(name):
    shard = find_shard(name)
    with open(f"{MODEL_DIR}/{shard}", "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        meta = hdr[name]
        dtype_map = {"F8_E4M3": torch.float8_e4m3fn, "F8_E8M0": torch.uint8, "BF16": torch.bfloat16}
        dt = dtype_map[meta["dtype"]]
        start, end = meta["data_offsets"]
        f.seek(8 + n + start)
        raw = f.read(end - start)
        t = torch.frombuffer(bytearray(raw), dtype=dt).reshape(meta["shape"])
        return t


def e8m0_to_float(scale_u8: torch.Tensor) -> torch.Tensor:
    return torch.exp2(scale_u8.float() - 127.0)


def dequant_fp8_block(weight_e4m3: torch.Tensor, scale_u8: torch.Tensor, block: int = 32) -> torch.Tensor:
    n, k = weight_e4m3.shape
    s = e8m0_to_float(scale_u8)
    s = s.repeat_interleave(block, 0)[:n].repeat_interleave(block, 1)[:, :k]
    return (weight_e4m3.float() * s).to(torch.bfloat16)


def act_qdq_fp8(x: torch.Tensor, block: int = 32) -> torch.Tensor:
    shape = x.shape
    xf = x.float().reshape(-1, block)
    amax = xf.abs().amax(dim=1, keepdim=True).clamp_min(1e-4)
    s = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    y = (xf / s).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float() * s
    return y.reshape(shape).to(torch.bfloat16)


def rms_rsqrt(x, eps):
    return torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)


def engram_forward(h, rows, wkv_bf16, q_weight, k_weight):
    T = h.size(0)
    kv = F.linear(act_qdq_fp8(rows.reshape(T, -1).to(torch.bfloat16)), wkv_bf16)  # qlinear
    key, value = kv.split([HC_MULT * DIM, DIM], dim=-1)
    key = key.float().view(T, HC_MULT, DIM)
    weight = q_weight * k_weight
    hf = h.float()

    def gate(hh, kk):
        rstd = rms_rsqrt(hh, EPS) * rms_rsqrt(kk, EPS)
        dot = (hh * weight * kk).sum(-1, keepdim=True) * rstd * DIM ** -0.5
        return torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))

    g = gate(hf, key)
    return (hf + g * value.float().unsqueeze(1)).to(h.dtype)


def main():
    with open(f"{REF_DIR}/manifest.json") as f:
        manifest = json.load(f)
    ent = {e["file"]: e for e in manifest["tensors"]}
    h_ent = ent[f"L00.h.{OCC}.bin"]
    rows_ent = ent[f"L01.engram_rows.{OCC}.bin"]
    out_ent = ent[f"L01.engram_out.{OCC}.bin"]
    T = h_ent["shape"][0]
    assert h_ent["shape"] == [T, HC_MULT, DIM]
    assert rows_ent["shape"] == [T, 24, 256]
    assert out_ent["shape"] == [T, HC_MULT, DIM]

    h = load_bf16_bin(f"{REF_DIR}/{h_ent['file']}", h_ent["shape"])
    rows = load_f32_bin(f"{REF_DIR}/{rows_ent['file']}", rows_ent["shape"])
    want = load_bf16_bin(f"{REF_DIR}/{out_ent['file']}", out_ent["shape"]).float()

    wkv_w = read_tensor(f"layers.{LAYER}.engram.wkv.weight")
    wkv_s = read_tensor(f"layers.{LAYER}.engram.wkv.scale")
    q_w = read_tensor(f"layers.{LAYER}.engram.q_weight").float()
    k_w = read_tensor(f"layers.{LAYER}.engram.k_weight").float()
    wkv_bf16 = dequant_fp8_block(wkv_w, wkv_s)

    got = engram_forward(h, rows, wkv_bf16, q_w, k_w).float()

    diff = (got - want).abs()
    rel_l2 = (diff.pow(2).sum() / want.pow(2).sum().clamp_min(1e-30)).sqrt().item()
    max_abs = diff.max().item()
    print(f"T={T} occ={OCC}: rel_l2={rel_l2:.6e} max_abs={max_abs:.6e} "
          f"want_absmax={want.abs().max().item():.4f} got_absmax={got.abs().max().item():.4f}")
    tol = 3e-3  # bf16 tolerance, matching compare.py's DEFAULT_TOL
    verdict = "PASS" if rel_l2 < tol else "FAIL"
    print(verdict, f"(tol {tol})")

    # Negative control: skip the gate entirely (h passed through unchanged). Must be far off,
    # or this spec check proves nothing about whether the gate/value term matters.
    control_diff = (h.float() - want).abs()
    control_rel_l2 = (control_diff.pow(2).sum() / want.pow(2).sum().clamp_min(1e-30)).sqrt().item()
    print(f"negative control (h alone, no gate/value): rel_l2={control_rel_l2:.6e} "
          f"({'correctly FAR' if control_rel_l2 > tol * 10 else 'TOO CLOSE -- control is weak'})")


if __name__ == "__main__":
    main()
