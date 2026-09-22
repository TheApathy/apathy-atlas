#!/usr/bin/env python3
"""Dump each occurrence's LOCAL chunk token ids from an oracle manifest.

engram_dead_heads is called fresh per forward chunk on that chunk's own `ids`
(engine/model.py:747) -- no cross-chunk carry -- so validating it needs exactly
the token ids that occurrence's forward call saw, not the whole prompt.

Writes ids_<occurrence>.bin (u32 little-endian) for every `engram_dead` tap on
the requested layer, plus prints S (chunk start) and n (chunk length) so the
caller can name/compare against the matching L0N.engram_dead.<occ>.bin.
"""
import json
import sys
import numpy as np

ref_dir = sys.argv[1]
layer = int(sys.argv[2]) if len(sys.argv) > 2 else 1

with open(f"{ref_dir}/manifest.json") as f:
    m = json.load(f)

token_ids = np.array(m["token_ids"], dtype=np.uint32)
entries = [e for e in m["tensors"] if e.get("tap") == "engram_dead" and e.get("layer") == layer]
entries.sort(key=lambda e: e["occurrence"])
assert entries, f"no engram_dead taps for layer {layer} in {ref_dir}/manifest.json"

for e in entries:
    s, n = e["S"], e["shape"][0]
    occ = f"{e['occurrence']:03d}"
    chunk = token_ids[s : s + n]
    chunk.tofile(f"ids_{occ}.bin")
    print(f"occ={occ} S={s} n={n} file={e['file']} -> ids_{occ}.bin")
