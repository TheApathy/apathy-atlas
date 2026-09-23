#!/usr/bin/env python3
"""Per-kernel GPU time inside the DECODE window of a profile.py capture.

usage: prof_decode.py <prof-dir> [steps]
Decode window = [first kernel + ttft, last kernel]; ttft from prof.jsonl.
If <steps> is given, per-step figures are printed (e.g. tokens / tokens-per-step).
"""
import json
import sqlite3
import sys
from collections import defaultdict

d = sys.argv[1]
steps = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
rec = json.loads(open(f"{d}/prof.jsonl").readline())
c = sqlite3.connect(f"{d}/trace.sqlite")
names = dict(c.execute("select id,value from StringIds").fetchall())
k = c.execute("select start,end,shortName from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
t0 = k[0][0] + rec["ttft_ms"] * 1e6
k = [x for x in k if x[0] >= t0]
t1 = max(e for _, e, _ in k)
busy = 0
cs, ce = k[0][0], k[0][1]
for s, e, _ in k[1:]:
    if s > ce:
        busy += ce - cs
        cs, ce = s, e
    else:
        ce = max(ce, e)
busy += ce - cs
span = t1 - t0
print(f"decode window {span/1e6:.1f} ms, gpu busy {busy/1e6:.1f} ms ({busy/span:.1%}), "
      f"per step: span {span/1e6/steps:.2f} ms busy {busy/1e6/steps:.2f} ms")
agg = defaultdict(lambda: [0, 0])
for s, e, n in k:
    agg[names[n]][0] += 1
    agg[names[n]][1] += e - s
for n, (cnt, t) in sorted(agg.items(), key=lambda x: -x[1][1])[:25]:
    print(f"{t/1e6/steps:8.3f} ms/step {cnt/steps:8.1f} calls/step {t/cnt/1e3:8.1f} us/call  {n[:80]}")
