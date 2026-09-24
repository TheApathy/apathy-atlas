#!/usr/bin/env python3
"""Quick per-arm view: median decode tok/s and the distinct output hashes per prompt.

usage: arms.py <sweep-dir> [arm ...]
"""
import collections
import json
import statistics
import sys
from pathlib import Path

d = Path(sys.argv[1])
arms = sys.argv[2:] or [l.split("|")[0] for l in open(d / "arms.txt") if l.strip()]
for a in arms:
    f = d / a / "trials.jsonl"
    if not f.exists():
        continue
    tps = collections.defaultdict(list)
    shas = collections.defaultdict(set)
    for line in open(f):
        r = json.loads(line)
        if r["rep"] != "warmup":
            tps[r["prompt"]].append(r["decode_tok_s"])
            shas[r["prompt"]].add(r["sha"][:6])
    cells = [f"{p}={statistics.median(v):6.2f}{sorted(shas[p])}" for p, v in tps.items()]
    print(f"{a:>6} " + " ".join(cells))
