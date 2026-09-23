#!/usr/bin/env python3
"""statehash_compare.py <reference.txt> <candidate.txt>

Both files come from ATLAS_GLM53_STATE_HASH. Lines are grouped into requests (a new
request starts when the position does not increase). For each request index and each
position both files reached, the committed KDA conv/recurrent hashes of layers 0/17/33
must be equal. Prints per-source counts (walk/prefix/full) and every mismatch.
Exit 0 = all shared positions equal, 1 = mismatch, 2 = nothing comparable.
"""
import collections, re, sys


def load(path):
    reqs, cur, last = [], {}, None
    for line in open(path):
        f = dict(kv.split("=", 1) for kv in line.split())
        pos = int(f.pop("pos"))
        if last is not None and pos <= last:
            reqs.append(cur); cur = {}
        cur[pos] = f
        last = pos
    if cur:
        reqs.append(cur)
    return reqs


ref, cand = load(sys.argv[1]), load(sys.argv[2])
checked = collections.Counter(); bad = []
for r, (a, b) in enumerate(zip(ref, cand)):
    for pos in sorted(set(a) & set(b)):
        src = b[pos]["src"]
        ka = {k: v for k, v in a[pos].items() if k != "src"}
        kb = {k: v for k, v in b[pos].items() if k != "src"}
        checked[src] += 1
        if ka != kb:
            diff = [k for k in ka if ka[k] != kb.get(k)]
            bad.append((r, pos, src, diff))
print(f"requests ref={len(ref)} cand={len(cand)} compared positions by candidate source: {dict(checked)}")
for r, pos, src, diff in bad[:20]:
    print(f"MISMATCH request {r} pos {pos} ({src}): {diff}")
if not sum(checked.values()):
    print("UNAVAILABLE: no shared positions"); sys.exit(2)
print("STATE GATE:", "FAIL" if bad else "PASS")
sys.exit(1 if bad else 0)
