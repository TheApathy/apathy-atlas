#!/usr/bin/env python3
"""Build the complete prefill waterfall from an ATLAS_PROFILE=1 serve log.

Parses every profiled bucket line emitted after a marker timestamp, sums per
bucket, and reports the total attributed vs the measured TTFT — the
difference is the unattributed remainder that profiling has not yet named.

Usage: python3 scripts/prefill_waterfall.py <serve-log> [since-epoch-seconds]
"""
import re
import sys
from collections import defaultdict

log = sys.argv[1]
since = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0

# Match any "[bucket] ...: 1234µs" or "...: 1234us" profile line.
pat = re.compile(r"\[([\w./ -]+)\][^:]*:\s*(\d+)\s*[µu]s")
ts_pat = re.compile(r"^\x1b?\[?2?m?(\d{4}-\d{2}-\d{2})T(\d{2}):(\d{2}):(\d{2})")

buckets = defaultdict(lambda: [0, 0])  # label -> [total_us, count]
for line in open(log, errors="replace"):
    m = ts_pat.search(line.replace("\x1b[2m", ""))
    if m and since:
        h, mi, s = int(m.group(2)), int(m.group(3)), int(m.group(4))
        tod = h * 3600 + mi * 60 + s
        if tod < since:
            continue
    pm = pat.search(line)
    if pm:
        label, us = pm.group(1), int(pm.group(2))
        buckets[label][0] += us
        buckets[label][1] += 1

total = sum(v[0] for v in buckets.values())
print(f"{'bucket':40s} {'total ms':>10s} {'calls':>6s} {'ms/call':>8s}")
for label, (us, cnt) in sorted(buckets.items(), key=lambda kv: -kv[1][0]):
    print(f"{label:40s} {us/1000:10.1f} {cnt:6d} {us/1000/max(cnt,1):8.2f}")
print(f"{'TOTAL ATTRIBUTED':40s} {total/1000:10.1f}")
