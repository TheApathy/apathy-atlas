#!/usr/bin/env python3
"""aggregate.py <window-dir> : interleaved-window summary.

Arms named T<k> are target-only; other arms are grouped by name minus a trailing letter
(G3a, G3b -> G3). For each spec arm, the ratio uses the mean of the target arms immediately
before and after it in window order (bracketing), so between-restart drift cancels.
Reports per config and prompt: mean tok/s, mean bracketed ratio, min ratio; then the geomean
over prompts and the worst prompt.
"""
import collections, math, os, re, statistics, sys

win = sys.argv[1]
order = []
for line in open(os.path.join(win, "window.log")):
    m = re.search(r"arm (\S+) \(", line)
    if m and "starting" in line:
        order.append(m.group(1))


def speeds(arm):
    out = collections.defaultdict(list)
    p = os.path.join(win, arm, "timing.txt")
    for line in open(p):
        m = re.match(r"trial-\d+-(\w+) \w+: .*completion=(\d+).*decode_tok_s=([\d.]+)", line)
        if m and m.group(2) == "160":
            out[m.group(1)].append(float(m.group(3)))
    return {k: statistics.mean(v) for k, v in out.items()}


order = [a for a in order if os.path.exists(os.path.join(win, a, "timing.txt"))]
sp = {a: speeds(a) for a in order}
targets = [i for i, a in enumerate(order) if re.fullmatch(r"T\d*", a)]
prompts = sorted({p for a in order for p in sp[a]} - {"ctlperturb"})
cfg = collections.defaultdict(lambda: collections.defaultdict(list))
for i, a in enumerate(order):
    if i in targets:
        continue
    before = max((t for t in targets if t < i), default=None)
    after = min((t for t in targets if t > i), default=None)
    brackets = [order[t] for t in (before, after) if t is not None]
    for p in prompts:
        base = statistics.mean(sp[b][p] for b in brackets)
        cfg[re.sub(r"[a-z]$", "", a)][p].append((sp[a][p], sp[a][p] / base))
tmean = {p: statistics.mean(sp[order[t]][p] for t in targets) for p in prompts}
print("target mean tok/s:", {p: round(v, 2) for p, v in tmean.items()},
      " spread:", {p: f"{100*(max(sp[order[t]][p] for t in targets)/min(sp[order[t]][p] for t in targets)-1):.1f}%" for p in prompts})
for c, d in cfg.items():
    ratios = {}
    print(f"\n{c}:")
    for p in prompts:
        v = d[p]
        r = statistics.mean(x[1] for x in v)
        ratios[p] = r
        print(f"  {p:8s} tok/s={statistics.mean(x[0] for x in v):6.2f}  bracketed ratio mean={r:.3f} min={min(x[1] for x in v):.3f} n={len(v)}")
    g = math.exp(statistics.mean(math.log(r) for r in ratios.values()))
    w = min(ratios, key=ratios.get)
    print(f"  GEOMEAN {g:.3f}x   WORST {w} {ratios[w]:.3f}x")
