#!/usr/bin/env python3
"""Load Qwen3.6-27B-base via HF transformers, capture hiddens at layers
[1, 16, 31, 46, 61] for tokens dumped from Atlas, compare to Atlas's
/tmp/atlas_target_hidden.bin element-wise.

This isolates whether Atlas's SSM/attention kernels produce numerically
the same hidden states as HF transformers — if not, the z-lab DFlash
drafter (trained against HF hiddens) cannot match Atlas's hiddens, and
that's the root cause of the 1% accept rate.

Inputs:
    /tmp/atlas_tokens.json          — token IDs from Atlas
    /tmp/atlas_target_hidden.bin    — Atlas captures [ctx, 5*hidden] BF16

Outputs:
    /tmp/hf_target_hidden.bin       — HF captures, same layout
    diff stats printed to stdout
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

MODEL = "/home/flocka/models/Qwen3.6-27B-base"
CAPTURE_LAYERS = [1, 16, 31, 46, 61]
HIDDEN = 5120

TOKENS_JSON = "/tmp/atlas_tokens.json"
ATLAS_BIN = "/tmp/atlas_target_hidden.bin"
HF_BIN = "/tmp/hf_target_hidden.bin"


def bf16_bytes_to_f32(b):
    u = np.frombuffer(b, dtype=np.uint16)
    return ((u.astype(np.uint32)) << 16).view(np.float32)


def main():
    if not os.path.exists(TOKENS_JSON):
        print(f"ERR: {TOKENS_JSON} missing. Run Atlas with ATLAS_DFLASH_DEBUG_DUMP_FULL=1.")
        sys.exit(1)
    info = json.load(open(TOKENS_JSON))
    tokens = info["all_tokens"]
    print(f"[hf_capture] {len(tokens)} tokens from Atlas")

    print(f"[hf_capture] loading {MODEL} on CUDA (BF16, ~52 GB; SSM kernels need GPU)...")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    model.eval()
    print(f"[hf_capture] loaded; type={type(model).__name__}")

    # Find the layers list (path differs by arch)
    layers = None
    for path in [
        lambda m: m.model.language_model.layers,
        lambda m: m.language_model.model.layers,
        lambda m: m.model.layers,
        lambda m: m.language_model.layers,
    ]:
        try:
            layers = path(model)
            break
        except AttributeError:
            continue
    if layers is None:
        print(f"[hf_capture] ERROR: couldn't find .layers list. Model structure:")
        for n, _ in model.named_modules():
            if "layers" in n and ".0." not in n:
                print(f"  {n}")
        sys.exit(2)
    print(f"[hf_capture] found {len(layers)} layers")

    # Register hooks
    caps = {}

    def make_hook(idx):
        def hook_fn(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            caps[idx] = hs.detach().to(torch.bfloat16).cpu().contiguous()
        return hook_fn

    hooks = [layers[i].register_forward_hook(make_hook(i)) for i in CAPTURE_LAYERS]

    # Forward
    print(f"[hf_capture] forward pass...")
    input_ids = torch.tensor([tokens], dtype=torch.long, device="cuda:0")
    with torch.no_grad():
        _ = model(input_ids=input_ids, use_cache=False)
    for h in hooks:
        h.remove()

    # caps[idx] shape: [1, n_tokens, 5120]
    n = len(tokens)
    print(f"[hf_capture] captures ready: {[caps[i].shape for i in CAPTURE_LAYERS]}")

    # Pack HF into same layout as Atlas: [n_ctx, 5*HIDDEN] BF16
    hf_arr = np.zeros((n, 5 * HIDDEN), dtype=np.uint16)
    for slot, layer_idx in enumerate(CAPTURE_LAYERS):
        h = caps[layer_idx][0].view(torch.uint16).numpy()  # [n, 5120]
        hf_arr[:, slot * HIDDEN:(slot + 1) * HIDDEN] = h
    Path(HF_BIN).write_bytes(hf_arr.tobytes())
    print(f"[hf_capture] wrote {HF_BIN} ({hf_arr.nbytes} bytes)")

    # Compare
    atlas_raw = open(ATLAS_BIN, "rb").read()
    n_atlas = len(atlas_raw) // (5 * HIDDEN * 2)
    print(f"[hf_capture] atlas dump: {n_atlas} ctx slots, hf: {n} slots")
    n_compare = min(n_atlas, n)
    atlas_f = bf16_bytes_to_f32(atlas_raw).reshape(n_atlas, 5, HIDDEN)
    hf_f = bf16_bytes_to_f32(hf_arr.tobytes()).reshape(n, 5, HIDDEN)

    print(f"\n=== Per-layer per-slot diff (cosine similarity) ===")
    for slot, layer_idx in enumerate(CAPTURE_LAYERS):
        print(f"\nLayer {layer_idx}:")
        for s in range(n_compare):
            a = atlas_f[s, slot]
            h = hf_f[s, slot]
            cos = float(np.dot(a, h) / (np.linalg.norm(a) * np.linalg.norm(h) + 1e-9))
            l2 = float(np.linalg.norm(a - h))
            ratio = l2 / (np.linalg.norm(h) + 1e-9)
            print(f"  slot{s:2d}: cos={cos:.4f} L2={l2:.2f} L2/HF={ratio:.4f}  "
                  f"atlas[mean={a.mean():.4f} std={a.std():.4f} norm={np.linalg.norm(a):.2f}]  "
                  f"hf[mean={h.mean():.4f} std={h.std():.4f} norm={np.linalg.norm(h):.2f}]")


if __name__ == "__main__":
    main()
