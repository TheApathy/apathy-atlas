#!/usr/bin/env python3
"""Compare ATLAS_LOGITS_FNV row hashes across trials and arms.

usage: fnv.py <sweep-dir> <reference-arm> [arm ...]

Requests are split on the server's per-request "Jinja rendered" line and
matched to trials.jsonl in order. For every (arm, prompt, trial) this prints
how many logged rows share the reference arm's first trial hash at the same
output index and the first index where they differ.
"""
import json
import re
import sys
from pathlib import Path

ANSI = re.compile(r"\x1b\[[0-9;]*m")
FNV = re.compile(r"LOGITS_FNV idx=(\d+) fnv=([0-9a-f]+)")


def requests(arm_dir: Path):
    reqs, cur = [], None
    for line in open(arm_dir / "server.log", errors="replace"):
        line = ANSI.sub("", line)
        if "Jinja rendered" in line:
            cur = {}
            reqs.append(cur)
            continue
        m = FNV.search(line)
        if m and cur is not None:
            cur.setdefault(int(m.group(1)), m.group(2))
    trials = [json.loads(l) for l in open(arm_dir / "trials.jsonl")]
    return list(zip(trials, reqs[-len(trials):]))


d = Path(sys.argv[1])
ref_arm = sys.argv[2]
arms = sys.argv[3:] or [ref_arm]
# Reference trial per prompt: rep REF_REP (default 1) of the reference arm,
# so a race-hit warmup cannot become the reference.
import os
ref_rep = os.environ.get("REF_REP", "1")
ref = {}
for trial, rows in requests(d / ref_arm):
    if str(trial["rep"]) == ref_rep:
        ref.setdefault(trial["prompt"], rows)
for arm in arms:
    for trial, rows in requests(d / arm):
        base = ref[trial["prompt"]]
        shared = sorted(set(rows) & set(base))
        same = [i for i in shared if rows[i] == base[i]]
        diff = [i for i in shared if rows[i] != base[i]]
        print(f"{arm:>5} {trial['prompt']:>10} {str(trial['rep']):>6} sha={trial['sha'][:6]} "
              f"rows={len(rows):4d} shared={len(shared):4d} same={len(same):4d} "
              f"first_diff={diff[0] if diff else '-'}")
