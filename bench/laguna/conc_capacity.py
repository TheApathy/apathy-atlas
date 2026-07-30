#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Concurrency CAPACITY measurement: aggregate tok/s vs concurrent requests C.

READ THIS BEFORE QUOTING ANY NUMBER FROM THIS SCRIPT.

This is a CAPACITY measurement, not a DFlash A/B. `scheduler/mod.rs:530-556`
gates all three speculative branches (ngram, self-spec, and MTP -- which is
what DFlash rides) on `active.len() == 1`. So:

    C == 1  ->  DFlash speculative decode
    C >= 2  ->  DFlash is OFF; `step_decode_only` batches every active seq

The C>=2 rows therefore answer "what is our aggregate throughput under load",
and CANNOT be compared against a single-stream DFlash figure as if the same
decode mode produced both. Two different engines.

Lockstep: `min_tokens` is hardcoded 0 at both API call sites and is not a
serde field, and there is no `ignore_eos`, so requests cannot be forced to run
equal step counts. Instead every prompt is chosen to SATURATE max_tokens, and
`lockstep_ok` below reports whether they actually did. If it is false the
aggregate is diluted by a ragged tail (and worse, DFlash re-arms once the tail
drops to one active seq) and the row must not be quoted.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import statistics
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchenv import CHAT_URL, default_log, log_lines  # noqa: E402

# Distinct prompts, matched shape. Distinct because C identical prompts would
# let the prefix cache serve C-1 of them, so a high-C row would be measuring
# cache hits rather than concurrent decode -- and unequally across C.
# All are the decode_bench "repetitive" family, which is the one prompt
# measured to reliably hit the token cap rather than emitting EOS early.
_NAMES = [
    ("Point", "norm"), ("Vec", "length"), ("Node", "weight"), ("Cell", "area"),
    ("Mark", "scale"), ("Slot", "span"), ("Tile", "extent"), ("Zone", "radius"),
    ("Face", "depth"), ("Edge", "gap"), ("Bead", "mass"), ("Ring", "girth"),
]


def prompt_for(i: int) -> str:
    base, meth = _NAMES[i % len(_NAMES)]
    names = ", ".join(f"{base}{n}" for n in range(2, 10))
    return (
        f"Write a Python module defining 8 dataclasses named {names}, "
        f"each with float fields x,y and a method {meth}(). Code only."
    )


@dataclass(frozen=True)
class OneResult:
    idx: int
    t_start: float
    t_end: float
    wall: float
    completion_tokens: int
    prompt_tokens: int
    tok_s: float
    hash: str
    capped: bool


def one(url: str, idx: int, max_tokens: int, epoch: float) -> OneResult:
    body = {
        "model": "laguna",
        "messages": [{"role": "user", "content": prompt_for(idx)}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    t_start = time.time()
    resp = json.loads(urllib.request.urlopen(req, timeout=900).read())
    t_end = time.time()
    text = resp["choices"][0]["message"]["content"]
    n = int(resp["usage"]["completion_tokens"])
    return OneResult(
        idx=idx,
        t_start=t_start - epoch,
        t_end=t_end - epoch,
        wall=t_end - t_start,
        completion_tokens=n,
        prompt_tokens=int(resp["usage"]["prompt_tokens"]),
        tok_s=n / (t_end - t_start),
        hash=hashlib.sha256(text.encode()).hexdigest()[:16],
        capped=n >= max_tokens,
    )


def run_level(url: str, conc: int, max_tokens: int) -> dict:
    """Fire `conc` requests concurrently and report AGGREGATE throughput."""
    epoch = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as pool:
        futs = [pool.submit(one, url, i, max_tokens, epoch) for i in range(conc)]
        rows = [f.result() for f in futs]

    rows.sort(key=lambda r: r.idx)
    total = sum(r.completion_tokens for r in rows)
    window = max(r.t_end for r in rows) - min(r.t_start for r in rows)
    # The stretch during which ALL conc sequences were in flight. Aggregate
    # over the full window is diluted by ramp-up and tail; this bounds it.
    overlap = min(r.t_end for r in rows) - max(r.t_start for r in rows)
    return {
        "conc": conc,
        "aggregate_tok_s": total / window,
        "per_req_tok_s_mean": statistics.fmean(r.tok_s for r in rows),
        "total_tokens": total,
        "window_s": window,
        "overlap_s": overlap,
        # Fraction of the window in which every seq was active. Below ~0.9 the
        # tail is long enough that DFlash re-arms at active.len()==1 and this
        # row is a mixture of two decode modes.
        "overlap_frac": overlap / window if window > 0 else 0.0,
        "lockstep_ok": all(r.capped for r in rows),
        "n_capped": sum(1 for r in rows if r.capped),
        "tokens_spread": (min(r.completion_tokens for r in rows),
                          max(r.completion_tokens for r in rows)),
        "rows": [asdict(r) for r in rows],
    }


def scrape_window(path: str, start_line: int) -> dict:
    """Server-side truth for the window: Done lines, arrivals, DFlash steps."""
    try:
        with open(path, errors="ignore") as fh:
            lines = fh.readlines()[start_line:]
    except OSError:
        return {"error": f"cannot read {path}"}
    done = [float(m.group(1))
            for line in lines
            if (m := re.search(r"Done:.*?([\d.]+) tok/s", line))]
    # Arrival marker: the same string prefill_cublas.py's ARRIVE_RE keys on.
    # Verified against a real serve log to be present 46/46, in exact agreement
    # with "Prefill first token" and "Done:". An earlier version of this
    # function counted "POST /v1/chat/completions", which the serve NEVER logs
    # -- so the foreign-traffic guard silently reported 0 arrivals forever and
    # could not fire. Any change to this regex must be re-checked against a real
    # serve log, not assumed.
    arrivals = sum(1 for line in lines if "Chunked prefill start" in line)
    return {
        "done_count": len(done),
        "done_tok_s": done,
        # Sum of per-request server-side rates. For sequences that overlap for
        # the whole window this approximates aggregate throughput; it is an
        # OVERESTIMATE whenever overlap_frac < 1.
        "done_tok_s_sum": sum(done),
        "arrivals": arrivals,
        # Two independent per-request markers. They agreed exactly on the
        # reference log, so a disagreement here means the window boundary
        # clipped a request and the row is not trustworthy.
        "arrivals_vs_done_agree": arrivals == len(done),
        # accepted= only appears on speculative steps, so a nonzero count at
        # C>=2 means DFlash re-armed during a ragged tail.
        "spec_steps": sum(1 for line in lines if "accepted=" in line),
        # WEAK hint only. Keying contamination on thinking is broken in BOTH
        # directions (our own eval traffic thinks by default; foreign traffic
        # need not). The arrival census above is the real guard. Kept because
        # THIS harness sends enable_thinking=False on every request, so a hit
        # inside our window is at least worth looking at.
        "thinking_hint": sum(1 for line in lines if "Thinking enabled" in line),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=CHAT_URL)
    ap.add_argument("--log", default=default_log("serve-conc.log"))
    ap.add_argument("--conc", default="1,2,4,8")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    levels = [int(c) for c in args.conc.split(",")]
    if max(levels) > len(_NAMES):
        raise SystemExit(f"--conc max {max(levels)} exceeds {len(_NAMES)} distinct prompts")

    out = []
    for c in levels:
        start = log_lines(args.log)
        res = run_level(args.url, c, args.tokens)
        time.sleep(1.0)  # let the serve flush its Done lines
        res["server"] = scrape_window(args.log, start)
        out.append(res)

        s = res["server"]
        flags = []
        if not res["lockstep_ok"]:
            flags.append(f"RAGGED({res['n_capped']}/{c} capped)")
        if res["overlap_frac"] < 0.9:
            flags.append(f"LOW-OVERLAP({res['overlap_frac']:.2f})")
        if s.get("done_count") != c:
            flags.append(f"DONE={s.get('done_count')}!={c}")
        if s.get("arrivals", 0) > c:
            flags.append(f"FOREIGN(+{s['arrivals'] - c})")
        if not s.get("arrivals_vs_done_agree", True):
            flags.append(f"MARKERS-DISAGREE(arr={s['arrivals']} done={s['done_count']})")
        # THE BATCHING PROOF, and it is an INVERSION worth stating plainly.
        # DFlash is gated on active.len() == 1. So at C>=2:
        #   spec_steps == 0            -> the scheduler really did batch. GOOD.
        #   spec_steps ~ a handful     -> a ragged tail dropped to 1 active seq
        #                                 and DFlash re-armed; row is a mixture.
        #   spec_steps ~ full run      -> the serve SERIALIZED our requests
        #                                 (never admitted a 2nd seq), so this
        #                                 row is not a concurrency measurement
        #                                 at all -- it is C=1 repeated C times.
        # This replaces an earlier `grep max_batch_size` clamp check, which was
        # vacuous: the serve never logs that string, so the guard could not fire.
        if c > 1:
            spec = s.get("spec_steps", 0)
            if spec == 0:
                flags.append("batched-confirmed(0 spec steps)")
            elif spec > 0.2 * args.tokens:
                flags.append(f"** SERIALIZED? {spec} spec steps -- NOT a C={c} row")
            else:
                flags.append(f"tail-rearmed({spec} spec steps)")
        if s.get("thinking_hint", 0) > 0:
            flags.append(f"thinking-hint({s['thinking_hint']})")
        mode = "DFlash (spec)" if c == 1 else "batched serial (DFlash OFF)"
        print(
            f"  C={c:<2} aggregate {res['aggregate_tok_s']:6.1f} tok/s   "
            f"per-req {res['per_req_tok_s_mean']:5.1f}   "
            f"tok={res['total_tokens']:<5} window={res['window_s']:5.1f}s   "
            f"[{mode}]" + ("  ** " + " ".join(flags) if flags else "  clean"),
            flush=True,
        )

    if len(out) > 1:
        # Label the ACTUAL denominator. This said "vs C=1" while dividing by
        # out[0], which is only C=1 when the sweep happens to start there --
        # a --conc 3,5,6,7 run printed ratios against C=3 under a "vs C=1"
        # heading. A ratio whose operands are mislabelled is worse than no
        # ratio, because it reads as comparable across runs when it is not.
        base_c = out[0]["conc"]
        base = out[0]["aggregate_tok_s"]
        note = "" if base_c == 1 else "  (NOT vs C=1 -- this sweep did not run C=1)"
        print(f"\n  scaling vs C={base_c} ({base:.1f} tok/s):{note}")
        for r in out:
            print(f"    C={r['conc']:<2} {r['aggregate_tok_s'] / base:5.2f}x")

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\n  wrote {args.json_out}")


if __name__ == "__main__":
    main()
