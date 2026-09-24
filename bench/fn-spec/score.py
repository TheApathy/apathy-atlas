#!/usr/bin/env python3
"""Score a fn-spec sweep: per-arm per-prompt median decode tok/s, self-determinism,
identity against the control arms, and each treatment's speedup vs its two
neighbouring controls (paired local controls, C T C T C ordering).

usage: score.py <sweep-dir>
"""
import json
import re
import statistics
import sys
from pathlib import Path

d = Path(sys.argv[1])
arms = [l.split("|")[0] for l in open(d / "arms.txt") if l.strip() and not l.startswith("#")]
data = {}
for a in arms:
    f = d / a / "trials.jsonl"
    if not f.exists():
        continue
    recs = [json.loads(l) for l in open(f)]
    data[a] = recs

prompts = []
for recs in data.values():
    for r in recs:
        if r["prompt"] not in prompts:
            prompts.append(r["prompt"])

is_ctrl = lambda a: a.startswith("C")
# Control reference text per prompt: the set of shas seen in ANY control trial.
ctrl_shas = {p: {} for p in prompts}
for a, recs in data.items():
    if is_ctrl(a):
        for r in recs:
            if r["rep"] != "warmup":
                ctrl_shas[r["prompt"]][r["sha"]] = ctrl_shas[r["prompt"]].get(r["sha"], 0) + 1


def med(a, p):
    v = [r["decode_tok_s"] for r in data.get(a, []) if r["prompt"] == p and r["rep"] != "warmup"]
    return statistics.median(v) if v else float("nan")


def accept(a):
    log = d / a / "server.log"
    if not log.exists():
        return ""
    last = ""
    for line in open(log, errors="replace"):
        if "MTP_ACCEPT summary" in line:
            last = line.strip()
    return re.sub(r"^.*?INFO\s+", "", last)[:240]


print("control sha census:", {p: ctrl_shas[p] for p in prompts})
hdr = f"{'arm':>6} " + " ".join(f"{p:>12}" for p in prompts)
print(hdr)
for i, a in enumerate(arms):
    if a not in data:
        print(f"{a:>6} MISSING")
        continue
    cells = []
    for p in prompts:
        trials = [r for r in data[a] if r["prompt"] == p and r["rep"] != "warmup"]
        shas = {r["sha"] for r in trials}
        ident = all(r["sha"] in ctrl_shas[p] for r in trials) and len(shas) == 1
        cells.append(f"{med(a, p):7.2f}{'=' if ident else ('~' if len(shas) == 1 else '!')}{len(shas)}")
    print(f"{a:>6} " + " ".join(f"{c:>12}" for c in cells))
print("legend: = one sha, in control census; ~ one sha, NOT in census; !N N distinct shas")

print("\nspeedup vs mean of neighbouring controls:")
for i, a in enumerate(arms):
    if is_ctrl(a) or a not in data:
        continue
    nb = [arms[j] for j in (i - 1, i + 1) if 0 <= j < len(arms) and is_ctrl(arms[j]) and arms[j] in data]
    out = []
    for p in prompts:
        c = statistics.mean(med(n, p) for n in nb) if nb else float("nan")
        out.append(f"{p}={med(a, p) / c:.3f}x")
    print(f"{a:>6} vs {','.join(nb)}: " + " ".join(out))
    acc = accept(a)
    if acc:
        print(f"       accept: {acc}")

print("\nwarmup decode tok/s by run index (a trend here invalidates the run):")
for a in arms:
    if a in data:
        w = [r["decode_tok_s"] for r in data[a] if r["rep"] == "warmup"]
        print(f"{a:>6} " + " ".join(f"{x:7.2f}" for x in w))
