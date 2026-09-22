#!/usr/bin/env python3
"""Chunk invariance: the same prompt prefilled under different chunkings must give BYTE-IDENTICAL
h at every layer and identical logits. Encoder layers (<=20) tap one occurrence per chunk, so
their occurrences are concatenated along T before comparing; replay layers (>=21) tap once."""
import os, re, sys, numpy as np
base = sys.argv[1]
arms = sorted(d for d in os.listdir(base) if d.startswith("split_"))
def load(arm):
    out = {}
    files = sorted(os.listdir(os.path.join(base, arm)))
    by = {}
    for f in files:
        m = re.match(r"L(\d+)\.(\w+)\.(\d+)\.bin$", f)
        if m:
            by.setdefault((int(m[1]), m[2]), []).append((int(m[3]), f))
    for k, v in by.items():
        v.sort()
        out[k] = b"".join(open(os.path.join(base, arm, f), "rb").read() for _, f in v)
    return out
data = {a: load(a) for a in arms}
ref = arms[0]
worst_first = None
for key in sorted(data[ref]):
    row = [f"L{key[0]:02d}.{key[1]:12s}"]
    for a in arms[1:]:
        x, y = data[ref][key], data[a].get(key)
        if y is None or len(x) != len(y):
            row.append(f"{a}: SHAPE/MISSING"); continue
        if x == y:
            row.append(f"{a}: identical"); continue
        xa = (np.frombuffer(x, np.uint16).astype(np.uint32) << 16).view(np.float32)
        ya = (np.frombuffer(y, np.uint16).astype(np.uint32) << 16).view(np.float32)
        nd = int((xa != ya).sum())
        rel = float(np.linalg.norm(xa - ya) / max(np.linalg.norm(xa), 1e-30))
        row.append(f"{a}: {nd}/{xa.size} differ rel {rel:.3e}")
        if worst_first is None:
            worst_first = key
    print("  ".join(row))
print("FIRST NON-IDENTICAL:", worst_first if worst_first else "none -- all arms byte-identical")
