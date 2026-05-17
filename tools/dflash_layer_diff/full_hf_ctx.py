#!/usr/bin/env python3
"""Generate the FULL HF/FLA hidden-state context for a prompt + greedy
continuation, write to /tmp/hf_target_hidden.bin in Atlas's layout.

This lets us validate Option A (FLA-equivalent ctx → higher drafter
accept rate) end-to-end without doing the full Rust HTTP client wire-in.

Sequence of operations:
  1. Load HF Qwen3.6-27B-base on GPU.
  2. Tokenize the prompt.
  3. Greedy-decode N tokens, capturing hiddens at the 5 drafter
     capture layers [1, 16, 31, 46, 61] for EVERY position.
  4. Pack into [n_total, 5*HIDDEN] BF16 (Atlas's ctx_hidden_acc slot
     layout) and write /tmp/hf_target_hidden.bin.
  5. Also write the matching tokens to /tmp/hf_atlas_tokens.json so
     the benchmark can be re-run consistently.

Atlas will read this file via ATLAS_DFLASH_HF_OVERRIDE on every
propose() call, so the drafter gets FLA-equivalent ctx for the whole
generation (assuming Atlas's greedy continuation matches HF's, which
should hold at temp=0).
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

CAPTURE_LAYERS = [1, 16, 31, 46, 61]
HIDDEN = 5120


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/path/to/models/Qwen3.6-27B-base")
    ap.add_argument("--prompt", default="1 2 3 4 5")
    ap.add_argument("--n-tokens", type=int, default=20)
    ap.add_argument("--out-bin", default="/tmp/hf_target_hidden.bin")
    ap.add_argument("--out-tokens", default="/tmp/hf_atlas_tokens.json")
    args = ap.parse_args()

    print(f"[full_hf_ctx] loading {args.model}...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
    )
    model.eval()

    # Find layers list
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
        print("ERROR: couldn't find layers")
        sys.exit(1)
    print(f"[full_hf_ctx] {len(layers)} layers, capturing {CAPTURE_LAYERS}")

    # Tokenize
    enc = tok(args.prompt, return_tensors="pt").to("cuda:0")
    prompt_ids = enc["input_ids"][0].tolist()
    n_prompt = len(prompt_ids)
    print(f"[full_hf_ctx] prompt -> {n_prompt} tokens: {prompt_ids}")

    # Greedy generate, collecting hiddens at each step
    print(f"[full_hf_ctx] greedy-decoding {args.n_tokens} tokens with hidden capture...")
    all_tokens = list(prompt_ids)
    n_total = n_prompt + args.n_tokens

    # Run ONE big forward of prompt + all generated tokens (greedy) to get hiddens cleanly
    # Step 1: get the greedy continuation
    with torch.no_grad():
        out = model.generate(
            enc["input_ids"],
            max_new_tokens=args.n_tokens,
            do_sample=False,
            temperature=1.0,
            use_cache=True,
        )
    full_ids = out[0].tolist()
    print(f"[full_hf_ctx] full sequence ({len(full_ids)} tokens): {full_ids}")

    # Step 2: re-run forward once with hooks for hidden capture (no kv cache)
    print(f"[full_hf_ctx] re-running with hooks for hidden capture...")
    caps = {}
    def hook(idx):
        def fn(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            caps[idx] = hs.detach().to(torch.bfloat16).cpu().contiguous()
        return fn
    handles = [layers[i].register_forward_hook(hook(i)) for i in CAPTURE_LAYERS]
    try:
        full_ids_t = torch.tensor([full_ids], dtype=torch.long, device="cuda:0")
        with torch.no_grad():
            _ = model(input_ids=full_ids_t, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    # Pack [n_total, 5*HIDDEN] BF16
    n = len(full_ids)
    out_arr = np.zeros((n, 5 * HIDDEN), dtype=np.uint16)
    for slot, layer_idx in enumerate(CAPTURE_LAYERS):
        h = caps[layer_idx][0].view(torch.uint16).numpy()  # [n, 5120]
        out_arr[:, slot * HIDDEN:(slot + 1) * HIDDEN] = h
    Path(args.out_bin).write_bytes(out_arr.tobytes())
    print(f"[full_hf_ctx] wrote {args.out_bin} ({out_arr.nbytes} bytes, {n} positions)")

    info = {
        "prompt": args.prompt,
        "prompt_ids": prompt_ids,
        "full_ids": full_ids,
        "n_total": n,
    }
    Path(args.out_tokens).write_text(json.dumps(info, indent=2))
    print(f"[full_hf_ctx] wrote {args.out_tokens}")


if __name__ == "__main__":
    main()
