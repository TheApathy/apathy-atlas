#!/usr/bin/env python3
"""FLA sidecar — Python HTTP server that runs the HF/FLA reference target
forward on demand and returns captured ctx hiddens for Atlas's DFlash
drafter.

Why this exists:
    Atlas's SSM kernels diverge ~14-53% from HF transformers' reference
    (FLA's `chunk_gated_delta_rule`) by layer 61. The z-lab DFlash
    drafter was trained against FLA hiddens, so Atlas-captured ctx is
    out-of-distribution → drafter accept rate collapses to ~1%.

    This sidecar wraps HF Qwen3.6-27B with FLA's bit-tested kernels and
    exposes a POST /capture endpoint. Atlas calls it at prefill time
    (and optionally per-step) to get FLA-equivalent ctx hiddens.

Usage:
    # On server (run on same GPU as Atlas):
    python3 fla_sidecar.py --model /home/flocka/models/Qwen3.6-27B-base --port 8890

    # From Atlas: set env to point at sidecar (production wire-in TBD):
    export ATLAS_DFLASH_FLA_SIDECAR=http://localhost:8890

Limitations / TODO:
    - First-cut: prefill-only override (Atlas's per-step append still
      uses Atlas's drifted hiddens).  Full fix needs per-step calls.
    - GPU contention: HF + Atlas both want the GPU.  Co-scheduling
      via CUDA stream sync or split GPU mem.
    - Token-by-token Triton compile delay on first call (~30s).  Warm
      up with a dummy prompt.

This is the FIRST stepping stone of "Option A" — wire FLA's reference
kernels into Atlas's SSM target forward.  Production wire-in still
requires:
    1. Rust HTTP client in spark-server that calls /capture
    2. Atlas-side caching so we don't call sidecar every step
    3. Eventually: replace HTTP/JSON with shared CUDA memory via DLPack
       to avoid serialization overhead at decode latency.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Lock

import numpy as np
import torch

CAPTURE_LAYERS = [1, 16, 31, 46, 61]
HIDDEN = 5120

_model = None
_layers = None
_lock = Lock()


def init_model(model_path: str):
    global _model, _layers
    print(f"[fla_sidecar] loading {model_path} on cuda:0 (BF16 + FLA kernels)...")
    from transformers import AutoModelForCausalLM
    _model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    _model.eval()

    # Locate layers list
    for path in [
        lambda m: m.model.language_model.layers,
        lambda m: m.language_model.model.layers,
        lambda m: m.model.layers,
        lambda m: m.language_model.layers,
    ]:
        try:
            _layers = path(_model)
            break
        except AttributeError:
            continue
    if _layers is None:
        raise RuntimeError("Couldn't find layers list in model")
    print(f"[fla_sidecar] loaded; {len(_layers)} layers")


def capture(tokens: list[int]) -> bytes:
    """Run forward, capture hiddens at CAPTURE_LAYERS, return packed BF16
    bytes shaped [n_tokens, 5, HIDDEN] (matches Atlas's ctx_hidden_acc
    slot layout)."""
    n = len(tokens)
    caps = {}

    def hook(idx):
        def fn(module, inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            caps[idx] = hs.detach().to(torch.bfloat16).cpu().contiguous()
        return fn

    handles = [_layers[i].register_forward_hook(hook(i)) for i in CAPTURE_LAYERS]
    try:
        input_ids = torch.tensor([tokens], dtype=torch.long, device="cuda:0")
        with torch.no_grad():
            _ = _model(input_ids=input_ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    # Pack: [n, 5, HIDDEN] BF16, flatten to bytes
    out = np.zeros((n, 5 * HIDDEN), dtype=np.uint16)
    for slot, layer_idx in enumerate(CAPTURE_LAYERS):
        h = caps[layer_idx][0].view(torch.uint16).numpy()
        out[:, slot * HIDDEN:(slot + 1) * HIDDEN] = h
    return out.tobytes()


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/capture":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        req = json.loads(body)
        tokens = req["tokens"]

        with _lock:
            payload = capture(tokens)

        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        sys.stderr.write(f"[fla_sidecar] {self.address_string()} - {fmt % args}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8890)
    args = ap.parse_args()
    init_model(args.model)
    print(f"[fla_sidecar] listening on 0.0.0.0:{args.port}")
    HTTPServer(("0.0.0.0", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
