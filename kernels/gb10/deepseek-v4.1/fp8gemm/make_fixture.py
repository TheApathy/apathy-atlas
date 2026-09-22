#!/usr/bin/env python3
"""Real-weight fixtures for the fused fp8-weight GEMM gate: layer-2 dense weights (fp8 e4m3 +
ue8m0 32x32 block scales) straight from the checkpoint, with real activations where a tap
exists (runF_faithful L2 attn_x / qr) and a fixed-seed bf16 normal otherwise."""
import json, os, sys
import numpy as np
import torch
from safetensors import safe_open
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "attn"))
from oracle_io import Run

MODEL = "/home/flocka/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fx")
idx = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]


def get(name):
    with safe_open(os.path.join(MODEL, idx[name]), "pt") as f:
        return f.get_tensor(name)


R = Run("runF_faithful")
acts = {"attn_x": R.load(2, "attn_x", 0, raw=True), "qr": R.load(2, "qr", 0, raw=True)}
cases = [("wq_b", "layers.2.attn.wq_b", "qr"), ("wo_b", "layers.2.attn.wo_b", None),
         ("w1", "layers.2.ffn.shared_experts.w1", "attn_x"), ("wq_a", "layers.2.attn.wq_a", "attn_x"),
         ("wkv", "layers.2.attn.wkv", "attn_x"), ("w2", "layers.2.ffn.shared_experts.w2", None),
         ("wo_a_g0", "layers.2.attn.wo_a", None)]
os.makedirs(OUT, exist_ok=True)
for name, p, act in cases:
    w = get(p + ".weight").view(torch.uint8).numpy()
    s = get(p + ".scale").view(torch.uint8).numpy()
    if name == "wo_a_g0":   # the grouped wo_a: group 0 = rows 0..1023 (x K 4096), as ops issues it
        w, s = w[:1024], s[:32]
    n, k = w.shape
    if act:
        a = acts[act]
    else:
        g = torch.Generator().manual_seed(0)
        a = (torch.randn(512, k, generator=g) * 0.05).to(torch.bfloat16).view(torch.uint16).numpy()
    assert a.shape == (512, k), (name, a.shape, k)
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    a.tofile(f"{d}/a.bin"); w.tofile(f"{d}/w.bin"); s.tofile(f"{d}/s.bin")
    open(f"{d}/dims.txt", "w").write(f"512 {n} {k} {s.shape[0]} {s.shape[1]}\n")
    print(d, "M=512 N", n, "K", k, "scale", s.shape, "act", act or "randn")
