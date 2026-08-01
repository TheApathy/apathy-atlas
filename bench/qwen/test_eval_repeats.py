#!/usr/bin/env python3
"""Offline checks for eval_repeats.py. No GPU, no network, ~1s.

    python3 bench/qwen/test_eval_repeats.py

The rule these exist to defend: a repeated A/B must not convert overlapping
samples into a direction. Every check below is a way that could happen.
"""

from __future__ import annotations

import io
import sys

import eval_repeats as er

FAILS = []


def check(name, got, want):
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {name}")
    if not ok:
        print(f"       got {got!r}, want {want!r}")
        FAILS.append(name)


def pas(cells_vals, clean=True):
    """One pass: {cell: accept}. Rows carry the guard flags the scorer reads."""
    return {c: {"cell": c, "lang": c.split("/")[-1], "mean_accept": v,
                "clean": clean, "dflash": True, "width_ok": True}
            for c, v in cells_vals.items()}


def run(a, b):
    buf = io.StringIO()
    code = er.report(a, b, out=buf)
    return code, buf.getvalue()


print("== separated(): the entire statistical claim")
check("clean separation, A above", er.separated([8.0, 8.1, 7.9], [7.0, 7.1]), True)
check("clean separation, B above", er.separated([6.0, 6.1], [7.0, 7.1]), True)
check("one value overlapping kills it", er.separated([8.0, 6.9], [7.0, 7.1]), False)
check("touching endpoints are NOT separated", er.separated([7.0, 8.0], [6.0, 7.0]), False)
# A single sample has no spread, so it cannot establish separation from anything.
check("n=1 on a side is never separated", er.separated([8.0], [7.0]), False)

print("\n== a direction is only reported when the samples do not overlap")
sep_a = [pas({"x/go": 8.0}), pas({"x/go": 8.1}), pas({"x/go": 7.9})]
sep_b = [pas({"x/go": 7.0}), pas({"x/go": 7.1}), pas({"x/go": 6.9})]
code, txt = run(sep_a, sep_b)
check("separated cell resolves", "A WINS" in txt, True)
check("...and exit is 0", code, 0)

ov_a = [pas({"x/go": 8.0}), pas({"x/go": 6.0}), pas({"x/go": 7.0})]
ov_b = [pas({"x/go": 7.5}), pas({"x/go": 6.5}), pas({"x/go": 7.2})]
code, txt = run(ov_a, ov_b)
check("overlapping cell does NOT resolve", "unresolved" in txt, True)
check("no WINS anywhere in the report", "WINS" in txt, False)
# The point of the whole tool: a mean delta exists, and is deliberately not
# promoted to a verdict.
check("exit 5 = ran clean, resolved nothing", code, 5)
check("...and the text refuses the tie reading",
      "never as a tie in one side's" in txt, True)

print("\n== identical is a settled no-difference, not an unresolved one")
id_a = [pas({"x/go": 5.0}), pas({"x/go": 5.0})]
id_b = [pas({"x/go": 5.0}), pas({"x/go": 5.0})]
code, txt = run(id_a, id_b)
check("flat and equal reads as identical", "identical" in txt, True)
check("...and not as unresolved", "unresolved" in txt.split("-- verdict")[1].split("identical")[0].replace("unresolved  0", ""), False)

print("\n== a cell that failed to measure is a hole, never a tie")
# Second A pass has the cell ungraded (contended window), leaving n=1 on that
# side -- which must drop the cell rather than silently compare 1-vs-2.
drop_a = [pas({"x/go": 8.0}), pas({"x/go": 8.1}, clean=False)]
drop_b = [pas({"x/go": 7.0}), pas({"x/go": 7.1})]
code, txt = run(drop_a, drop_b)
check("dropped cell is named", "DROPPED" in txt and "x/go" in txt, True)
check("...and explicitly not counted as a tie",
      "not counted as a tie" in txt, True)
check("nothing resolved from a dropped cell", "WINS" in txt, False)

print("\n== the tool refuses to invent a repeat it was not given")
code, txt = run([pas({"x/go": 8.0})], [pas({"x/go": 7.0})])
check("k=1 per side is refused", code, 4)
check("...and points at the n=1 tool", "eval_verdict.py for the n=1 case" in txt, True)

print("\n== within-side spread is measured, not assumed")
noisy_a = [pas({"n/go": 8.0}), pas({"n/go": 6.0})]   # 28.6% spread
flat_b = [pas({"n/go": 7.0}), pas({"n/go": 7.0})]    # 0%
code, txt = run(noisy_a, flat_b)
check("A's spread is reported nonzero", "A: max 28.57%" in txt, True)
check("B's spread is reported zero", "B: max  0.00%" in txt, True)
check("a noisy A vs flat B still does not resolve", "WINS" in txt, False)

print()
if FAILS:
    print(f"FAILED {len(FAILS)}: {', '.join(FAILS)}")
    sys.exit(1)
print("all checks passed")
