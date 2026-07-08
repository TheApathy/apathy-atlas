#!/usr/bin/env python3
"""Task #20 — parse a probe serve log slice into step-phase statistics.

Usage: async20_analyze.py <serve_log> <bench_json>

Per modality (using the byte windows recorded by async20_bench.py):
  * accept distribution + full-accept rate (DFLASH K=γ verify lines)
  * step-phase means (DFLASH step timing lines)
  * inter-step wall gap (consecutive step-timing timestamps minus total)
  * ASYNC_PROBE commit-tail GPU duration + propose enqueue/GPU split
"""

import json
import re
import statistics
import sys
from datetime import datetime

LOG, BENCH = sys.argv[1], sys.argv[2]

RE_TS = re.compile(r"^\x1b?\[?2?m?(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)Z")
RE_STEP = re.compile(
    r"DFLASH step timing: total=(\d+)μs sync_secondary=(\d+)μs verify=(\d+)μs "
    r"commit=(\d+)μs save_hidden=(\d+)μs trim=(\d+)μs propose=(\d+)μs "
    r"other=(\d+)μs accepted=(\d+)")
RE_VERIFY = re.compile(r"DFLASH K=γ verify: γ=(\d+) accepted=(\d+)/(\d+)")
RE_CTAIL = re.compile(r"ASYNC_PROBE commit_tail: enqueue=(\d+)μs gpu_total=(\d+)μs")
RE_PROP = re.compile(r"ASYNC_PROBE propose: enqueue=(\d+)μs gpu_total=(\d+)μs")


def strip_ansi(s):
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


def parse_window(data):
    steps, accepts, ctails, props, ts_totals = [], [], [], [], []
    for line in data.splitlines():
        line = strip_ansi(line)
        m = RE_STEP.search(line)
        if m:
            v = [int(x) for x in m.groups()]
            steps.append(dict(zip(
                ["total", "sync", "verify", "commit", "save", "trim",
                 "propose", "other", "accepted"], v)))
            tm = re.match(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)Z", line)
            if tm:
                ts = datetime.fromisoformat(tm.group(1))
                ts_totals.append((ts, v[0]))
        m = RE_VERIFY.search(line)
        if m:
            g, a, tot = int(m.group(1)), int(m.group(2)), int(m.group(3))
            accepts.append((a, tot))
        m = RE_CTAIL.search(line)
        if m:
            ctails.append((int(m.group(1)), int(m.group(2))))
        m = RE_PROP.search(line)
        if m:
            props.append((int(m.group(1)), int(m.group(2))))
    # Inter-step gap: wall delta between consecutive step-END timestamps
    # minus the later step's own total.
    gaps = []
    for (t0, _), (t1, tot1) in zip(ts_totals, ts_totals[1:]):
        d_us = (t1 - t0).total_seconds() * 1e6
        gap = d_us - tot1
        if 0 <= gap < 200_000:  # ignore inter-request boundaries
            gaps.append(gap)
    return steps, accepts, ctails, props, gaps


def mean(v):
    return statistics.mean(v) if v else 0.0


def main():
    bench = json.load(open(BENCH))
    raw = open(LOG, "rb").read()
    print(f"=== {BENCH} (label={bench['label']}) md5_ok={bench.get('md5_ok')} ===")
    for name, b in bench["benches"].items():
        blobs = []
        for off0, off1 in b["log_windows"]:
            blobs.append(raw[off0:off1].decode("utf-8", "replace"))
        steps, accepts, ctails, props, gaps = parse_window("\n".join(blobs))
        n = len(accepts)
        full = sum(1 for a, t in accepts if a == t)
        acc_mean = mean([a for a, _ in accepts])
        print(f"\n[{name}] tok/s mean={b['mean']:.1f} steps={len(steps)} verifies={n}")
        if n:
            print(f"  accept: mean={acc_mean:.2f}/{accepts[0][1]} "
                  f"full-accept={full}/{n} ({100*full/max(n,1):.1f}%)")
        if steps:
            for k in ["total", "sync", "verify", "commit", "save", "trim",
                      "propose", "other"]:
                print(f"  step.{k:8s} mean={mean([s[k] for s in steps])/1000:8.2f} ms")
        if gaps:
            print(f"  inter-step gap: mean={mean(gaps)/1000:.2f} ms "
                  f"median={statistics.median(gaps)/1000:.2f} ms n={len(gaps)}")
        if ctails:
            print(f"  commit-tail GPU: enqueue={mean([c[0] for c in ctails])/1000:.2f} ms "
                  f"total={mean([c[1] for c in ctails])/1000:.2f} ms")
        if props:
            print(f"  propose split: enqueue={mean([p[0] for p in props])/1000:.2f} ms "
                  f"gpu_total={mean([p[1] for p in props])/1000:.2f} ms")


if __name__ == "__main__":
    main()
