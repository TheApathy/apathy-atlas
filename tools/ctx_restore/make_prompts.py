#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Build exact-token prompt pairs for the context-checkpoint gate.

Each case is (C, N): turn 1 sends the first C tokens, turn 2 the first N
(N > C, so turn 2 starts with turn 1 byte for byte, as a multi-turn chat
does). Cases never share a prefix. Also writes a same-length "forged" prefix (one token changed) used
by the stale-file control. Tokens come from real text (this repo's Rust
sources) tokenized with the target model's own tokenizer.

usage: make_prompts.py <model_dir> <out_dir> <C:N> [<C:N> ...] [--max-tokens M]
"""
import argparse
import glob
import hashlib
import json
import os
import struct

from tokenizers import Tokenizer


def prefix_hash(tokens):
    # Mirrors spark_runtime::ctx_store::prefix_hash.
    d = hashlib.sha256(b"".join(struct.pack("<I", t) for t in tokens)).digest()
    return struct.unpack("<Q", d[:8])[0]


def corpus(n_tokens, tok):
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.normpath(os.path.join(here, "..", "..", "crates"))
    ids = []
    for path in sorted(glob.glob(os.path.join(root, "**", "*.rs"), recursive=True)):
        ids.extend(tok.encode(open(path, errors="replace").read()).ids)
        if len(ids) >= n_tokens:
            return ids[:n_tokens]
    raise SystemExit(f"corpus too small: {len(ids)} < {n_tokens}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("out_dir")
    ap.add_argument("cases", nargs="+")
    ap.add_argument("--max-tokens", type=int, default=64)
    a = ap.parse_args()
    tok = Tokenizer.from_file(os.path.join(a.model_dir, "tokenizer.json"))
    cases = [tuple(int(x) for x in c.split(":")) for c in a.cases]
    # Each case starts at a different corpus offset so no case's prompt is a
    # prefix of another's (a deeper checkpoint supersedes its prefixes).
    all_ids = corpus(max(n for _, n in cases) + 1000 * len(cases), tok)
    os.makedirs(a.out_dir, exist_ok=True)
    manifest = []
    for i, (c, n) in enumerate(cases):
        assert 0 < c < n
        ids = all_ids[1000 * i:]
        forged = list(ids[:c])
        forged[c // 2] = (forged[c // 2] + 1) % 100000
        for name, toks, mt in [
            (f"p1-{c}", ids[:c], 1),
            (f"p2-{c}-{n}", ids[:n], a.max_tokens),
            (f"forged-{c}-{n}", forged + ids[c:n], a.max_tokens),
        ]:
            body = {"model": "m", "prompt_token_ids": toks, "max_tokens": mt,
                    "temperature": 0.0, "stream": False}
            json.dump(body, open(os.path.join(a.out_dir, name + ".json"), "w"))
        manifest.append({"c": c, "n": n,
                         "p1_hash": f"{prefix_hash(ids[:c]):016x}",
                         "forged_hash": f"{prefix_hash(forged):016x}"})
    json.dump(manifest, open(os.path.join(a.out_dir, "manifest.json"), "w"), indent=1)
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
