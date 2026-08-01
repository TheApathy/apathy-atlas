#!/usr/bin/env python3
"""Score a repeated A/B: n passes per side, per cell, with a real spread.

WHY THIS EXISTS. `eval_verdict.py` compares one A arm to one B arm and uses a
single A/A repeat as the floor. That is enough when the floor is small relative
to the effect. It was NOT enough for the 2026-08-01 Go question: 4 of the 5 Go
cells moved under a FIXED drafter, the Go delta landed at 1.4x its own floor,
and the honest verdict was UNRESOLVED. n=1 cannot separate a 2% effect from 2%
noise -- no amount of care in the scoring fixes a missing repeat.

So: run each side k times and compare distributions instead of points.

WHAT IT REFUSES TO DO.

  * It does not average away a cell's spread and quote the mean as if it were
    a measurement. Every cell prints mean, min-max, and n.

  * It does not call a winner on overlapping ranges. With k=3 per side the
    comparison is nonparametric and deliberately crude: a cell is RESOLVED
    only if every pass of one side beats every pass of the other (a separation
    with no overlap, p=1/20 under the null at k=3 by exact rank count). Anything
    else is UNRESOLVED. Crude and honest beats a t-test on n=3.

  * It does not pool across cells before showing the per-cell picture, and it
    never pools accept over each arm's own steps -- see eval_verdict.pool() and
    the accept-pooling confound it documents.

READ THE PER-CELL BLOCK FIRST. The aggregate at the bottom answers "on this
mix", which is a different question from "on this workload".
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys


def load(paths):
    """[{cell: row}] -- one dict per pass, keyed by cell."""
    out = []
    for p in paths:
        with open(p) as fh:
            d = json.load(fh)
        out.append({r["cell"]: r for r in d["rows"]})
    return out


def graded(r):
    """A row only counts if the harness said its window was trustworthy."""
    return bool(r) and r.get("clean") and r.get("dflash") and r.get("width_ok") \
        and r.get("mean_accept") is not None


def series(passes, cell, key="mean_accept"):
    """Values for one cell across passes -- only the graded ones."""
    vs = []
    for p in passes:
        r = p.get(cell)
        if graded(r):
            vs.append(r[key])
    return vs


def separated(xs, ys):
    """True when the two samples do not overlap at all.

    Direction is returned separately. This is the whole statistical claim and
    it is intentionally conservative: with k=3 a side, complete separation is
    the 1-in-20 tail, and anything short of it at these sample sizes is not
    something to publish a direction on.
    """
    if len(xs) < 2 or len(ys) < 2:
        return False
    return max(xs) < min(ys) or max(ys) < min(xs)


def fmt_rel(new, old):
    if not old:
        return "     -"
    return f"{(new - old) / old * 100:+5.1f}%"


def report(a_passes, b_passes, out=sys.stdout):
    p = lambda s="": print(s, file=out)
    ka, kb = len(a_passes), len(b_passes)
    cells = sorted(set().union(*[set(x) for x in a_passes + b_passes]))

    p("=== repeated A/B ===")
    p(f"    A passes: {ka}    B passes: {kb}    cells: {len(cells)}")
    if ka < 2 or kb < 2:
        p("\n!! FEWER THAN 2 PASSES ON A SIDE -- this tool cannot add a repeat")
        p("   that was not run. Use eval_verdict.py for the n=1 case; it says")
        p("   UNRESOLVED honestly rather than implying a spread it lacks.")
        return 4

    p("\n-- per cell, mean accept (min-max over passes)")
    p(f"  {'cell':<20}{'A mean':>9} {'A range':>15}   "
      f"{'B mean':>9} {'B range':>15}   {'delta':>7}  verdict")

    resolved, unresolved, dropped, identical = [], [], [], []
    for c in cells:
        xa, xb = series(a_passes, c), series(b_passes, c)
        if len(xa) < 2 or len(xb) < 2:
            dropped.append(c)
            continue
        ma, mb = statistics.fmean(xa), statistics.fmean(xb)
        # A cell where both sides are perfectly reproducible AND equal is a
        # resolved no-difference, not an unresolved one. Lumping it in with
        # genuinely-overlapping cells would understate how much of the matrix
        # is actually settled -- and on this stack it is a real category:
        # boilerplate/go returned 8.84 from both drafters, twice.
        flat = (min(xa) == max(xa) and min(xb) == max(xb))
        if flat and ma == mb:
            v = "identical"
            identical.append(c)
        elif separated(xa, xb):
            v = "B WINS" if mb > ma else "A WINS"
            resolved.append((c, v, ma, mb))
        else:
            v = "unresolved"
            unresolved.append(c)
        p(f"  {c:<20}{ma:9.2f} {f'{min(xa):.2f}-{max(xa):.2f}':>15}   "
          f"{mb:9.2f} {f'{min(xb):.2f}-{max(xb):.2f}':>15}   "
          f"{fmt_rel(mb, ma):>7}  {v}")

    if dropped:
        p(f"\n  !! DROPPED (fewer than 2 graded passes on a side): "
          f"{', '.join(dropped)}")
        p("     Not scored, and not counted as a tie. A cell that failed to")
        p("     measure is a hole in the evidence, not a null result.")

    # -- the within-side spread, which is the number that decided the last run.
    p("\n-- within-side spread (the floor, measured not assumed)")
    for name, passes in (("A", a_passes), ("B", b_passes)):
        rels = []
        for c in cells:
            xs = series(passes, c)
            if len(xs) >= 2 and statistics.fmean(xs):
                rels.append((max(xs) - min(xs)) / statistics.fmean(xs))
        if rels:
            p(f"    {name}: max {max(rels)*100:5.2f}%   "
              f"median {statistics.median(rels)*100:5.2f}%   "
              f"cells with any movement: "
              f"{sum(1 for r in rels if r > 1e-9)}/{len(rels)}")

    p("\n-- verdict")
    p(f"    resolved   {len(resolved):>2}  " +
      (", ".join(f"{c} ({v})" for c, v, _, _ in resolved) or "-"))
    p(f"    unresolved {len(unresolved):>2}  " + (", ".join(unresolved) or "-"))
    p(f"    identical  {len(identical):>2}  " + (", ".join(identical) or "-")
      + "   (both sides reproducible AND equal -- a settled no-difference)")

    a_wins = [c for c, v, _, _ in resolved if v == "A WINS"]
    b_wins = [c for c, v, _, _ in resolved if v == "B WINS"]

    # Aggregate, stated as what it is: a claim about this mix.
    per_cell = []
    for c in cells:
        xa, xb = series(a_passes, c), series(b_passes, c)
        if len(xa) >= 2 and len(xb) >= 2 and statistics.fmean(xa):
            per_cell.append((statistics.fmean(xb) - statistics.fmean(xa))
                            / statistics.fmean(xa))
    if per_cell:
        p(f"\n    mean accept across cells, B vs A: "
          f"{statistics.fmean(per_cell)*100:+.1f}%  (unweighted, per cell)")

    p()
    if not resolved:
        p(f"RESULT: NO CELL RESOLVES A DIRECTION"
          + (f" ({len(identical)} identical, {len(unresolved)} overlapping)."
             if identical else "."))
        p("        NOTHING RESOLVED. Every cell's two samples overlap, so no")
        p("        direction is supportable at these sample sizes. This is a")
        p("        real outcome -- it means the effect, if any, is smaller than")
        p("        this box's run-to-run noise on these cells. Report it as")
        p("        'no measurable difference', never as a tie in one side's")
        p("        favour, and never by quoting the mean delta alone.")
        return 5
    # Separation answers "is there a direction", NOT "is it worth anything".
    # boilerplate/go resolved for A at 0.8% on the 2026-08-01 run: four passes
    # a side, zero overlap, and an effect nobody would deploy for. Printing the
    # resolved count without the magnitudes invites reading 3-of-5 as three
    # equal wins when one of them is a rounding error with good hygiene.
    trivial = [(c, v, ma, mb) for c, v, ma, mb in resolved
               if ma and abs(mb - ma) / ma < 0.01]
    if trivial:
        p(f"\n    !! RESOLVED BUT TRIVIAL (<1%): "
          + ", ".join(f"{c} ({(mb-ma)/ma*100:+.1f}%)" for c, _, ma, mb in trivial))
        p("       Direction is reproducible; the magnitude is not worth a")
        p("       decision. Do not count these alongside the material wins.")

    p(f"RESULT: {len(a_wins)} cell(s) resolve for A, {len(b_wins)} for B, "
      f"{len(unresolved)} unresolved.")
    p("        Resolved means the two samples did not overlap at all. Quote")
    p("        the per-cell block, not the aggregate, when they disagree.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", required=True, metavar="JSON")
    ap.add_argument("--b", nargs="+", required=True, metavar="JSON")
    args = ap.parse_args()
    return report(load(args.a), load(args.b))


if __name__ == "__main__":
    sys.exit(main())
