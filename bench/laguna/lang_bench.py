#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Decode tok/s across C / Python / Go on MATCHED coding tasks.

WHY THIS EXISTS: every prompt in decode_bench.py's SUITE is Python, so the
campaign's headline "33.7 weighted tok/s for coding" is strictly a PYTHON
number. This asks whether that rate holds per-language.

DESIGN -- the tasks are MATCHED, which is the whole point. Each task is the
same algorithm expressed in all three languages, phrased identically apart
from the language name and its idiom words. If the prompts differed in
difficulty as well as language, a per-language gap would not be attributable
to language. Anything that is not matched (e.g. C needing an explicit length
parameter) is a known, recorded asymmetry, not a silent one.

The expected mechanism, stated UP FRONT so the result can falsify it: DFlash's
speed is driven by the drafter's accept rate, and accept rate tracks how
PREDICTABLE the token stream is. C and Go carry more boilerplate per unit of
logic (type declarations, `if err != nil`, braces) than Python. If that
boilerplate is more predictable, C/Go should accept MORE and run FASTER than
Python -- the opposite of the usual "Python is the model's best language"
intuition. Report accept alongside tok/s so the two can be checked together.

Run TWICE (--pass 1, --pass 2) and compare: with no repeat there is no floor,
and a per-language gap cannot be distinguished from run-to-run noise.

RESULT -- the hypothesis above is FALSIFIED, in the direction the intuition
predicted rather than the mechanism did. Python accepts MOST (3.30) and is
fastest; Go accepts least (2.87) and is slowest. Boilerplate volume does not buy
predictability; the model's per-language fluency does. Accept still explains the
speed (r=0.971, r2=0.942 over the 13 speculative rows), so the accept->tok/s
mechanism survives -- only the language->accept prediction died.

AND THE THING THIS HARNESS ALMOST GOT WRONG: 2 of 15 rows are not DFlash rows
at all. adaptive_spec.rs suspends speculation when the rolling 12-step mean
accept drops below ATLAS_DFLASH_ADAPTIVE_MIN (1.2 in production) and serial-
decodes from there. Both such rows landed at the SERIAL baseline (~23 tok/s)
while still reporting a small, entirely plausible accept figure, because
`accepted=` is logged on speculative steps ONLY. See benchenv.spec_fraction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchenv import (  # noqa: E402
    CHAT_URL, default_log, is_dflash, log_lines, scrape, spec_fraction,
)

LANGS = ("c", "python", "go")

_LANG_NAME = {"c": "C", "python": "Python", "go": "Go"}

# Per-language phrasing of "give me a self-contained unit of code, no prose".
# Kept as close to identical as the languages allow.
_STYLE = {
    "c": "Write a C function{extra}. Include the function signature and body only, no main(), no explanation.",
    "python": "Write a Python function{extra}. Include type hints and a docstring, no explanation.",
    "go": "Write a Go function{extra}. Include a doc comment, no package clause, no explanation.",
}

# (task_id, per-language task clause). The clause completes "...a <LANG>
# function {clause}". C gets an explicit length/size parameter where Python and
# Go carry length in the slice/list -- that is a genuine language asymmetry,
# recorded here rather than hidden.
TASKS = [
    ("binsearch", {
        "c": " int binary_search(const int *a, size_t n, int target) returning the index or -1",
        "python": " binary_search(a: list[int], target: int) -> int returning the index or -1",
        "go": " BinarySearch(a []int, target int) int returning the index or -1",
    }),
    ("mergesort", {
        "c": " void merge_sort(int *a, size_t n) that sorts in place",
        "python": " merge_sort(a: list[int]) -> list[int] that returns a new sorted list",
        "go": " MergeSort(a []int) []int that returns a new sorted slice",
    }),
    ("modinv", {
        "c": " int mod_inverse(int a, int m) using the extended Euclidean algorithm, returning -1 when no inverse exists",
        "python": " mod_inverse(a: int, m: int) -> int using the extended Euclidean algorithm, raising ValueError when no inverse exists",
        "go": " ModInverse(a, m int) (int, error) using the extended Euclidean algorithm, returning an error when no inverse exists",
    }),
    ("intervals", {
        "c": " int min_points(const int *starts, const int *ends, size_t n) returning the minimum number of points so every interval contains at least one point",
        "python": " min_points(intervals: list[tuple[int, int]]) -> int returning the minimum number of points so every interval contains at least one point",
        "go": " MinPoints(intervals [][2]int) int returning the minimum number of points so every interval contains at least one point",
    }),
    # Boilerplate-heavy. "repetitive" is our fastest class in the Python suite,
    # so this is the row where the predictability mechanism should show most.
    ("boilerplate", {
        "c": "s: eight structs named Point2..Point9, each with float fields x,y, and for each a function <name>_norm returning the Euclidean norm",
        "python": "s: eight dataclasses named Point2..Point9, each with float fields x,y and a method norm() returning the Euclidean norm",
        "go": "s: eight structs named Point2..Point9, each with float64 fields X,Y and for each a method Norm() returning the Euclidean norm",
    }),
]


def prompt_for(lang: str, task_id: str, clause: str) -> str:
    # The boilerplate row is plural ("functions"), so its clause starts with
    # "s:" and grafts onto the style string's "function" -> "functions:".
    return _STYLE[lang].format(extra=clause).replace("a C functions:", "C functions:") \
        .replace("a Python functions:", "Python functions:") \
        .replace("a Go functions:", "Go functions:")


def one(url: str, prompt: str, max_tokens: int) -> dict:
    body = {
        "model": "laguna",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    resp = json.loads(urllib.request.urlopen(req, timeout=600).read())
    wall = time.time() - t0
    text = resp["choices"][0]["message"]["content"]
    n = int(resp["usage"]["completion_tokens"])
    return {
        "wall": wall,
        "completion_tokens": n,
        "prompt_tokens": int(resp["usage"]["prompt_tokens"]),
        "tok_s": n / wall,
        "hash": hashlib.sha256(text.encode()).hexdigest()[:16],
        "capped": n >= max_tokens,
        "text": text,
    }


def window_census(path: str, start: int) -> dict:
    """Per-request census fields that benchenv.scrape does not carry.

    benchenv.scrape owns the accept distribution and the adaptive-suspend/reprobe
    counters. This adds only the request-completion markers this harness needs to
    tell a clean single-request window from a contended one: the serve's own
    `Done: ... tok/s` figures and the "Chunked prefill start" arrival count.

    Markers verified against a real serve log: "Chunked prefill start" /
    "Prefill first token" / "Done:" all agreed at 46/46. Never key a census on a
    string you have not counted in a real log.
    """
    try:
        with open(path, errors="ignore") as fh:
            lines = fh.readlines()[start:]
    except OSError:
        return {"server_tok_s": [], "arrivals": 0, "done_count": 0}
    done = [float(m.group(1)) for line in lines
            if (m := re.search(r"Done:.*?([\d.]+) tok/s", line))]
    return {
        "server_tok_s": done,
        "arrivals": sum(1 for line in lines if "Chunked prefill start" in line),
        "done_count": len(done),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=CHAT_URL)
    ap.add_argument("--log", default=default_log("serve-persist.log"))
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--pass", dest="pass_n", type=int, default=1)
    ap.add_argument("--retries", type=int, default=3,
                    help="re-runs allowed per row when the window is contended")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    out: list[dict] = []
    print(f"=== lang_bench pass {args.pass_n} :: max_tokens={args.tokens} "
          f"retries<={args.retries} ===")
    print(f"{'task':<13}{'lang':<8}{'tok/s':>8}{'srv':>7}{'tok':>6}{'cap':>5}{'accept':>8}{'try':>5}  hash")
    for task_id, clauses in TASKS:
        for lang in LANGS:
            prompt = prompt_for(lang, task_id, clauses[lang])
            row = None
            for attempt in range(1, args.retries + 2):
                start = log_lines(args.log)
                r = one(args.url, prompt, args.tokens)
                time.sleep(0.6)
                s = scrape(args.log, start) or {}
                cen = window_census(args.log, start)
                srv = cen["server_tok_s"]
                row = {
                    "task": task_id, "lang": lang, "pass": args.pass_n,
                    "tok_s": r["tok_s"], "server_tok_s": srv[0] if len(srv) == 1 else None,
                    "completion_tokens": r["completion_tokens"], "capped": r["capped"],
                    "hash": r["hash"], "mean_accept": s.get("mean_accept"),
                    "steps": s.get("steps"), "dist": s.get("dist"),
                    "arrivals": cen["arrivals"], "done_count": cen["done_count"],
                    "adapt_suspend": s.get("adapt_suspend"),
                    "adapt_reprobe": s.get("adapt_reprobe"),
                    "attempts": attempt, "text": r["text"],
                }
                # Engine label, from TWO independent signals that must agree:
                # the serve's own suspend log, and the token accounting in
                # spec_fraction. is_dflash returns (verdict, frac, agree) where
                # verdict is None (UNGRADED) when the scrape has nothing to grade.
                row["dflash"], row["spec_frac"], row["engine_agree"] = is_dflash(
                    s, row["completion_tokens"])
                # Keep spec_fraction imported and callable for anyone re-deriving
                # the ratio from a row's stored (steps, mean_accept, tokens).
                if row["spec_frac"] is None:
                    row["spec_frac"] = spec_fraction(
                        row["steps"], row["mean_accept"], row["completion_tokens"])
                row["clean"] = row["arrivals"] == 1 and row["done_count"] == 1
                if row["clean"]:
                    break
                # The serve port may be shared with periodic external clients
                # (a foreign client sending bursts of short requests will queue
                # ahead of ours). A lock does NOT keep such traffic out -- it is
                # a CLIENT, not a competing serve. So a contended row must be
                # RE-RUN, never averaged in: a contended request measures
                # queueing, not decode. Back off past the burst rather than
                # retrying straight into it.
                if attempt <= args.retries:
                    time.sleep(20.0)
            assert row is not None
            out.append(row)
            marks = []
            if not row["clean"]:
                marks.append(f"** CONTENDED arrivals={row['arrivals']} "
                             f"done={row['done_count']} -- EXCLUDED")
            # THREE engine states, not two: True (DFlash), False (adaptive-
            # suspended to serial), and None (UNGRADED -- the scrape had nothing
            # to grade, e.g. an empty accept histogram). "ungraded" is not the
            # same as "ran serial"; a zero count must never read as a finding.
            frac = f"{row['spec_frac']:.2f}" if row["spec_frac"] is not None else "?"
            acc = f"{row['mean_accept']:.2f}" if row["mean_accept"] is not None else "?"
            if row["dflash"] is False:
                marks.append(f"** SUSPENDED->serial (spec_frac={frac}, "
                             f"{row['adapt_suspend']} suspend / {row['adapt_reprobe']} reprobe) "
                             f"-- accept {acc} is over the SPEC SUBSET only")
            elif row["dflash"] is None:
                marks.append(f"** UNGRADED (spec_frac={frac}) -- the scrape produced "
                             f"nothing to classify; not a serial row, just not graded")
            if row["engine_agree"] is False:
                marks.append(f"** ENGINE SIGNALS DISAGREE: spec_frac={row['spec_frac']} vs "
                             f"adapt_suspend={row['adapt_suspend']} -- do not quote this row")
            # Print "-" for an unmeasured cell, never 0. `or 0` would render an
            # empty scrape as accept 0.00, which reads as "the drafter accepted
            # nothing" -- a finding -- when the truth is that nothing was
            # measured. Same rule as the three-state verdict above.
            srv_c = f"{row['server_tok_s']:.1f}" if row["server_tok_s"] is not None else "-"
            print(f"{task_id:<13}{_LANG_NAME[lang]:<8}{row['tok_s']:>8.1f}"
                  f"{srv_c:>7}{row['completion_tokens']:>6}"
                  f"{('Y' if row['capped'] else '-'):>5}"
                  f"{acc:>8}{row['attempts']:>5}"
                  f"  {row['hash'][:10]}" + ("  " + "  ".join(marks) if marks else ""),
                  flush=True)

    dirty = [r for r in out if not r["clean"]]
    serial = [r for r in out if r["clean"] and r["dflash"] is False]
    ungraded = [r for r in out if r["clean"] and r["dflash"] is None]
    # TWO ENGINES, TWO TABLES -- never one. A row whose DFlash suspended itself
    # is running the serial decoder, so averaging it into a per-language DFlash
    # figure reports the mean of two different machines. Same error class as the
    # concurrency table, where C>=2 rows have DFlash off entirely.
    for label, grp in (("DFlash speculative", [r for r in out if r["clean"] and r["dflash"]]),
                       ("adaptive-SUSPENDED (serial decode)", serial)):
        print(f"\n== {label} :: n={len(grp)}")
        if not grp:
            print("   (none)")
            continue
        print(f"   {'lang':<8}{'mean tok/s':>12}{'tok-weighted':>14}{'mean accept':>13}"
              f"{'range':>15}{'tokens':>9}{'capped':>8}{'n':>4}")
        for lang in LANGS:
            rs = [r for r in grp if r["lang"] == lang]
            if not rs:
                print(f"   {_LANG_NAME[lang]:<8}{'-':>12}")
                continue
            tot_tok = sum(r["completion_tokens"] for r in rs)
            tot_wall = sum(r["completion_tokens"] / r["tok_s"] for r in rs)
            accs = [r["mean_accept"] for r in rs if r["mean_accept"] is not None]
            rates = [r["tok_s"] for r in rs]
            print(f"   {_LANG_NAME[lang]:<8}{statistics.fmean(rates):>12.1f}"
                  f"{tot_tok / tot_wall:>14.1f}"
                  f"{(statistics.fmean(accs) if accs else 0):>13.2f}"
                  f"{f'{min(rates):.1f}-{max(rates):.1f}':>15}"
                  f"{tot_tok:>9}{sum(1 for r in rs if r['capped']):>8}{len(rs):>4}")
    if serial:
        names = ", ".join("{}/{}".format(r["task"], r["lang"]) for r in serial)
        print(f"\n  NOTE {len(serial)} row(s) fell OUT of speculation mid-request and are "
              f"tabled separately: {names}")
        print("     This is adaptive_spec.rs working as designed (never materially slower "
              "than plain decode), not a DFlash regression -- but their tok/s is a SERIAL "
              "figure and their accept is over the speculative subset only.")
    if ungraded:
        names = ", ".join("{}/{}".format(r["task"], r["lang"]) for r in ungraded)
        print(f"\n  NOTE {len(ungraded)} row(s) ran a clean window but produced nothing "
              f"to grade (empty accept scrape) and are UNGRADED, not serial: {names}")
        print("     'ran but ungraded' is a third state distinct from 'graded and clean' "
              "-- these are excluded from BOTH engine tables above.")
    if dirty:
        names = ", ".join("{}/{}".format(r["task"], r["lang"]) for r in dirty)
        print(f"\n  !! {len(dirty)} row(s) never got a clean window and are EXCLUDED "
              f"from the aggregates above: {names}")
        print("     Per-language means above are over UNEQUAL task sets, so they are "
              "not strictly comparable -- read the per-task rows instead.")

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
