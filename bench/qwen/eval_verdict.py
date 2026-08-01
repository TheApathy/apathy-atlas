#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Score an evaluation matrix: per-cell verdicts, coverage gate, mix sensitivity.

    python3 bench/qwen/eval_verdict.py --a ab/x/a.json --b ab/x/b.json \
        --aa ab/x/a-p2.json --claims lang:go,task:boilerplate

Pure: reads JSON written by eval_matrix.py, touches no GPU, no serve, no
network. Every rule below is exercised offline by test_eval_verdict.py.

THE THREE THINGS THIS DOES THAT A POOLED A/B DOES NOT

1. COVERAGE IS A GATE, NOT A FOOTNOTE. `--claims` names the workload the
   change is supposed to help. If the matrix does not GRADE those cells, this
   refuses to print a verdict and exits 3. Our in-house drafter was warm-
   started to raise the Go share of its training corpus, shipped on a Go-only
   A/B, and was later judged by a suite containing no Go at all. Both numbers
   were right; neither was an answer. A harness that will happily score a
   change on cells it was never about is how that happens twice.

2. EVERY CELL IS JUDGED SEPARATELY, AGAINST ITS OWN REPEAT. The A/A arm is
   mandatory -- a floor belongs to a harness+config+session, and a delta
   quoted against a floor from another sitting is quoted against nothing. Cell
   floors are n=1: a LOWER BOUND on spread, not a tolerance. A delta that
   barely clears one is not established.

3. THE POOLED NUMBER IS PRINTED NEXT TO THE MIX IT DEPENDS ON. The same rows
   pooled three ways -- by tokens, by cell, by language -- can disagree, and
   if they disagree in SIGN then "A beats B" is a statement about the prompt
   mix and not about A. That gets said out loud here rather than discovered
   three weeks later.

WHAT IS DELIBERATELY NOT A GATE. Completion hashes. On this stack committed
text is drafter-dependent at temperature 0, on a greedy target where it ought
not to be; the mechanism is unidentified. Hashes are tabled and explicitly
marked ungraded. Do not convert that table into a pass/fail without an
explanation of the mechanism first.

ACCEPT LEADS. Accept is the only quantity a drafter alone controls, and on
this harness its A/A floor came out ~7x tighter than tok/s (0.34% vs 2.36%).
tok/s is downstream of acceptance AND of the target's cost, so it moves for
reasons that have nothing to do with the change under test.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys

# A cell is comparable only if it ran a clean window, genuinely ran speculative
# decode end to end, and ran at the documented verify width. Anything else is
# EXCLUDED and named, never averaged.
GRADE_KEYS = ("clean", "dflash", "width_ok")


def load(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def graded(payload: dict) -> dict:
    """{cell_id: row} for the rows that may be compared at all."""
    out = {}
    for r in payload.get("rows", []):
        if all(r.get(k) is True for k in GRADE_KEYS):
            out[r["cell"]] = r
    return out


def ungraded_reasons(payload: dict) -> dict:
    """{cell_id: why it is not comparable}. Named, so exclusion is visible."""
    out = {}
    for r in payload.get("rows", []):
        if all(r.get(k) is True for k in GRADE_KEYS):
            continue
        if r.get("census") is None:
            why = "log unreadable"
        elif r.get("clean") is not True:
            why = f"contended (census={r.get('census')})"
        elif r.get("dflash") is False:
            why = f"adaptive-suspended to serial (spec_frac={r.get('spec_frac')})"
        elif r.get("dflash") is None:
            why = "ungraded: nothing to classify"
        elif r.get("width_ok") is not True:
            why = f"wrong verify width {r.get('denoms')}"
        else:
            why = "unknown"
        out[r["cell"]] = why
    return out


def rel(new, old):
    """Relative change, or None when either side is unmeasured.

    Returns None rather than 0.0 for a missing operand. A zero here would read
    as "measured, no change", which is a finding; the truth would be that no
    comparison happened.
    """
    if new is None or old is None or not old:
        return None
    return (new - old) / old


def cell_rows(a: dict, b: dict, aa: dict) -> list:
    """Per-cell comparison over the cells graded in ALL THREE arms.

    Intersecting is the conservative choice: a cell missing its A/A repeat has
    no floor, and a cell graded in only one treatment arm has no comparison.
    Both are reported as coverage losses by `coverage()` rather than silently
    shrinking the denominator.
    """
    rows = []
    for cell in sorted(set(a) & set(b) & set(aa)):
        ra, rb, raa = a[cell], b[cell], aa[cell]
        rows.append({
            "cell": cell, "task": ra.get("task"), "lang": ra.get("lang"),
            "accept_a": ra.get("mean_accept"), "accept_b": rb.get("mean_accept"),
            "accept_aa": raa.get("mean_accept"),
            "d_accept": rel(rb.get("mean_accept"), ra.get("mean_accept")),
            "f_accept": rel(raa.get("mean_accept"), ra.get("mean_accept")),
            "tok_a": ra.get("tok_s"), "tok_b": rb.get("tok_s"),
            "tok_aa": raa.get("tok_s"),
            "d_tok": rel(rb.get("tok_s"), ra.get("tok_s")),
            "f_tok": rel(raa.get("tok_s"), ra.get("tok_s")),
            "tokens_a": ra.get("completion_tokens"),
            "tokens_b": rb.get("completion_tokens"),
            "steps_a": ra.get("steps"), "steps_b": rb.get("steps"),
            "hash_a": ra.get("hash"), "hash_b": rb.get("hash"),
            "hash_aa": raa.get("hash"),
        })
    for r in rows:
        r["v_accept"] = verdict(r["d_accept"], r["f_accept"])
        r["v_tok"] = verdict(r["d_tok"], r["f_tok"])
    return rows


def verdict(delta, floor):
    """WIN / LOSS / TIE / '-' for one cell on one metric.

    TIE means |delta| did not exceed this cell's own A/A spread. That spread is
    n=1, so TIE is "not distinguishable here", never "proven equal" -- the two
    are different claims and only the first one is supported.
    """
    if delta is None or floor is None:
        return "-"
    if abs(delta) <= abs(floor):
        return "TIE"
    return "WIN" if delta > 0 else "LOSS"


def parse_claims(spec: str) -> list:
    """`lang:go,task:boilerplate,cell:modinv/go` -> [(axis, value), ...]."""
    claims = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"claim {part!r} is not axis:value "
                             f"(axes: lang, task, cell)")
        axis, value = part.split(":", 1)
        axis = axis.strip()
        if axis not in ("lang", "task", "cell"):
            raise ValueError(f"claim axis {axis!r} is not one of lang/task/cell")
        claims.append((axis, value.strip()))
    return claims


def coverage(rows: list, claims: list) -> dict:
    """Does the graded matrix actually cover what the change claims to help?

    Returns {claim: [cells that satisfy it]}. An empty list is the refusal
    case, and it is the whole point of this file: a change tuned for Go and
    graded only on Python has ZERO evidence either way, which is a different
    outcome from "no significant difference" and must not print as one.
    """
    got = {}
    for axis, value in claims:
        if axis == "lang":
            hit = [r["cell"] for r in rows if r["lang"] == value]
        elif axis == "task":
            hit = [r["cell"] for r in rows if r["task"] == value]
        else:
            hit = [r["cell"] for r in rows if r["cell"] == value]
        got[f"{axis}:{value}"] = hit
    return got


def pool(rows: list, metric: str, weighting: str):
    """Pooled A->B delta under one weighting. (delta, n) or (None, 0).

    THREE weightings, because the choice of weighting is a choice about what
    workload you are claiming to serve, and it is normally made by accident:

      tokens  -- the mix the suite happens to emit. What a pooled bench prints.
      cell    -- every matrix cell equal. Removes the suite's own composition.
      lang    -- every language equal, tasks averaged within it. The mix a
                 multi-language user actually sees, and the one under which a
                 Go-tuned change is allowed to show up at all.

    THE WEIGHTS ARE TAKEN FROM ARM A FOR BOTH ARMS, and that is not a detail.

    The obvious thing -- average each arm over its own steps, which is what the
    serve's own global `mean=` figure does -- is CONFOUNDED for accept, because
    the weight is a function of the metric. A speculative step commits
    accept+1 tokens, so a drafter that accepts MORE on some cell takes FEWER
    steps there, and its own average therefore gives that cell LESS weight.
    Every arm is scored on a mix tilted toward the cells it is worst at, and
    the better drafter is penalised for being better.

    It is not a rounding effect. On the 2026-07-31 drafter A/B, prose alone
    carried 41% of the step weight purely because prose is the cell both
    drafters accept least on. Re-pooled with the weights held fixed, the same
    six rows move from +4.3% to +7.8% on accept -- and the shipped conclusion
    ("ours is 3.6% worse") was read off the confounded figure.

    Holding the weights fixed asks the question actually being asked: on ONE
    workload, which arm does better. It also means the by-tokens column is not
    comparable to a `mean=` line scraped from a serve log; that is intended.
    """
    key_d = "d_accept" if metric == "accept" else "d_tok"
    key_a = "accept_a" if metric == "accept" else "tok_a"
    key_b = "accept_b" if metric == "accept" else "tok_b"
    usable = [r for r in rows if r[key_a] is not None and r[key_b] is not None]
    if not usable:
        return None, 0

    if weighting == "tokens":
        # Weight by the work each cell represents. For accept the natural
        # weight is verify STEPS (accept is per step); for tok/s it is emitted
        # TOKENS. Using one weight for both would silently re-scale one metric.
        wkey = "steps_a" if metric == "accept" else "tokens_a"
        wt = [(r, r.get(wkey) or 0) for r in usable]
        tot = sum(w for _, w in wt)
        if not tot:
            return None, 0
        a = sum(r[key_a] * w for r, w in wt) / tot
        b = sum(r[key_b] * w for r, w in wt) / tot
        return rel(b, a), len(usable)

    if weighting == "cell":
        a = statistics.fmean(r[key_a] for r in usable)
        b = statistics.fmean(r[key_b] for r in usable)
        return rel(b, a), len(usable)

    if weighting == "lang":
        langs = sorted({r["lang"] for r in usable})
        per_a, per_b = [], []
        for lg in langs:
            rs = [r for r in usable if r["lang"] == lg]
            per_a.append(statistics.fmean(r[key_a] for r in rs))
            per_b.append(statistics.fmean(r[key_b] for r in rs))
        return rel(statistics.fmean(per_b), statistics.fmean(per_a)), len(usable)

    raise ValueError(f"unknown weighting {weighting!r}")


def sign_flip(deltas: dict) -> bool:
    """True when the pooled verdicts do not agree on direction.

    Only counts deltas that are actually present. Two weightings that disagree
    in sign mean the pooled claim is a claim about the mix.
    """
    signs = {(d > 0) for d in deltas.values() if d is not None and d != 0}
    return len(signs) > 1


def _fmt(x, width=8, prec=1, pct=True):
    if x is None:
        return "-".rjust(width)
    return (f"{x * 100:+.{prec}f}%" if pct else f"{x:.{prec}f}").rjust(width)


def report(a_pay, b_pay, aa_pay, claims, out=sys.stdout) -> int:
    """Print the whole verdict. Returns the process exit code.

    Exit codes are distinct on purpose:
      0  verdict emitted
      3  REFUSED -- the matrix does not cover a claimed workload
      4  REFUSED -- too few cells graded in all three arms to compare anything
    A refusal must never share an exit code with a null result.
    """
    p = lambda *a: print(*a, file=out)  # noqa: E731

    ga, gb, gaa = graded(a_pay), graded(b_pay), graded(aa_pay)
    rows = cell_rows(ga, gb, gaa)

    p("=== eval matrix verdict ===")
    p(f"    A  {a_pay.get('arm')}   B  {b_pay.get('arm')}   "
      f"A/A  {aa_pay.get('arm')}")
    p(f"    graded cells: A={len(ga)}  B={len(gb)}  A/A={len(gaa)}  "
      f"comparable in all three={len(rows)}")

    excluded = {}
    for name, pay in (("A", a_pay), ("B", b_pay), ("A/A", aa_pay)):
        for cell, why in ungraded_reasons(pay).items():
            excluded.setdefault(cell, []).append(f"{name}: {why}")
    if excluded:
        p("\n  EXCLUDED cells (named, not silently dropped):")
        for cell, whys in sorted(excluded.items()):
            p(f"    {cell:<22} {'; '.join(whys)}")

    if len(rows) < 2:
        p(f"\nREFUSING TO SCORE: only {len(rows)} cell(s) graded in all three "
          f"arms. There is nothing here to compare, which is NOT the same as "
          f"finding no difference.")
        return 4

    # -- coverage gate, BEFORE any number, so a refusal cannot be read past.
    cov = coverage(rows, claims)
    if claims:
        p("\n-- coverage of the claimed workload")
        for k, hits in cov.items():
            p(f"    {k:<24} {len(hits):>2} graded cell(s)"
              + (f"  {', '.join(hits)}" if hits else "   <-- NONE"))
        missing = [k for k, hits in cov.items() if not hits]
        if missing:
            p(f"\nREFUSING TO SCORE: the matrix grades ZERO cells for "
              f"{', '.join(missing)}.")
            p("  The change claims to help there, so this run carries no "
              "evidence about it -- in either direction. Scoring the cells it "
              "does cover would answer a question nobody asked; that is the "
              "exact failure this gate exists to stop. Add the cells, or drop "
              "the claim and say plainly what the change was graded on.")
            return 3
    else:
        p("\n-- coverage: NO --claims given, so nothing is being checked for")
        p("     A verdict below is scoped to the cells listed, and to nothing "
          "else. State the target workload with --claims to have that enforced.")

    # -- per-cell table. Accept first: it is the drafter-controlled quantity.
    p("\n-- per cell (A/A is this cell's own floor; n=1, a lower bound on "
      "spread, not a tolerance)")
    p(f"  {'cell':<22}{'acc A':>7}{'acc B':>7}{'d acc':>9}{'floor':>9}{'':>6}"
      f"{'tok A':>8}{'tok B':>8}{'d tok':>9}{'floor':>9}")
    for r in rows:
        p(f"  {r['cell']:<22}"
          f"{_fmt(r['accept_a'], 7, 2, False)}{_fmt(r['accept_b'], 7, 2, False)}"
          f"{_fmt(r['d_accept'], 9)}{_fmt(r['f_accept'], 9)}"
          f"{r['v_accept']:>6}"
          f"{_fmt(r['tok_a'], 8, 1, False)}{_fmt(r['tok_b'], 8, 1, False)}"
          f"{_fmt(r['d_tok'], 9)}{_fmt(r['f_tok'], 9)}  {r['v_tok']}")

    # -- sign counts. The headline, because it survives re-weighting and a
    #    pooled mean does not.
    p("\n-- cell verdicts (B relative to A)")
    for metric, key in (("accept", "v_accept"), ("tok/s", "v_tok")):
        tally = {v: [r["cell"] for r in rows if r[key] == v]
                 for v in ("WIN", "LOSS", "TIE", "-")}
        p(f"    {metric:<8} WIN {len(tally['WIN']):>2}   "
          f"LOSS {len(tally['LOSS']):>2}   TIE {len(tally['TIE']):>2}   "
          f"ungraded {len(tally['-']):>2}")
        for v in ("WIN", "LOSS"):
            if tally[v]:
                p(f"             {v}: {', '.join(tally[v])}")

    # -- per language, because that is the axis a drafter change acts on.
    #
    # The FLOOR column is the point of this block. An earlier version printed
    # the language delta alone, and on the 2026-08-01 GoHeavy run that read as
    # "go -2.4%, ours wins on Go" -- the one claim the run existed to settle.
    # It is not a win: 4 of the 5 go cells move under a FIXED drafter, so the
    # aggregate go floor is 1.7% and the signal is 1.4x it. Meanwhile c is
    # 19.3x its floor. A language delta without its own floor next to it is
    # unreadable, because per-language reproducibility is itself per-language.
    p("\n-- by language (mean accept vs the A/A floor ON THE SAME CELLS)")
    p(f"    {'lang':<8} {'n':<4} {'B vs A':>9} {'A/A floor':>10} {'ratio':>7}"
      f"   cell verdicts")
    for lg in sorted({r["lang"] for r in rows}):
        rs = [r for r in rows if r["lang"] == lg]
        accs_a = [r["accept_a"] for r in rs if r["accept_a"] is not None]
        accs_b = [r["accept_b"] for r in rs if r["accept_b"] is not None]
        accs_aa = [r["accept_aa"] for r in rs if r["accept_aa"] is not None]
        d = rel(statistics.fmean(accs_b), statistics.fmean(accs_a)) \
            if accs_a and accs_b else None
        f = rel(statistics.fmean(accs_aa), statistics.fmean(accs_a)) \
            if accs_a and accs_aa else None
        # Ratio of signal to same-axis noise. Below ~2x, n=1 cannot separate
        # them and the honest word is UNRESOLVED, not the sign of d.
        ratio = abs(d / f) if (d is not None and f) else None
        moved = sum(1 for r in rs if r["f_accept"] not in (None, 0.0))
        w = sum(1 for r in rs if r["v_accept"] == "WIN")
        l = sum(1 for r in rs if r["v_accept"] == "LOSS")
        # A zero floor at n=1 is "did not move in one repeat", NOT "cannot
        # move". Dividing by it would print inf and read as infinitely
        # resolvable. The ratio is undefined, and says so.
        rtxt = f"{ratio:6.1f}x" if ratio is not None else \
               ("   n/a" if d is not None else "     -")
        p(f"    {lg:<8} n={len(rs):<3}{_fmt(d, 9)}{_fmt(f, 10)} {rtxt}"
          f"   {w} win / {l} loss / {len(rs) - w - l} tie-or-ungraded")
        if ratio is not None and ratio < 2.0:
            p(f"             !! UNRESOLVED at n=1: the {lg} delta is only "
              f"{ratio:.1f}x its own A/A floor")
            p(f"                ({moved} of {len(rs)} {lg} cells move under a "
              f"FIXED drafter). Do not report a direction here.")

    # -- mix sensitivity. The pooled number, and what it is standing on.
    p("\n-- pooled, under three weightings (see pool() for why three)")
    p(f"  {'metric':<10}{'by tokens':>12}{'by cell':>12}{'by language':>13}")
    flips = []
    for metric in ("accept", "tok/s"):
        m = "accept" if metric == "accept" else "tok"
        ds = {w: pool(rows, m, w)[0] for w in ("tokens", "cell", "lang")}
        p(f"  {metric:<10}{_fmt(ds['tokens'], 12)}{_fmt(ds['cell'], 12)}"
          f"{_fmt(ds['lang'], 13)}")
        if sign_flip(ds):
            flips.append(metric)
    if flips:
        p(f"\n  !! SIGN FLIP on {', '.join(flips)}: the pooled verdict changes "
          f"direction with the workload mix.")
        p("     'B beats A' is then a claim about the prompt mix, not about B. "
          "Quote the per-cell and per-language rows instead, and say which mix "
          "the deployment actually has.")

    # -- hashes: tabled, explicitly ungraded.
    diff_ab = [r["cell"] for r in rows if r["hash_a"] != r["hash_b"]]
    unstable = [r["cell"] for r in rows if r["hash_a"] != r["hash_aa"]]
    p(f"\n-- completion hashes: REPORTED, NOT GRADED")
    p(f"    A vs B differ on   {len(diff_ab)}/{len(rows)}"
      + (f": {', '.join(diff_ab)}" if diff_ab else ""))
    p(f"    A vs A/A differ on {len(unstable)}/{len(rows)}"
      + (f": {', '.join(unstable)}" if unstable else ""))
    if diff_ab:
        p("    At temperature 0 the committed tokens are the TARGET's argmax "
          "and should not depend on who proposed them. Cells that differ "
          "A-vs-B but NOT A-vs-A/A isolate the change as the cause -- that is "
          "an open finding about the verify path, not a quality signal. Do not "
          "clear it because the output looks correct.")

    p("\nRESULT: read the per-cell and per-language rows before the pooled row.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="baseline arm JSON")
    ap.add_argument("--b", required=True, help="treatment arm JSON")
    ap.add_argument("--aa", required=True,
                    help="repeat of the baseline -- this session's noise floor")
    ap.add_argument("--claims", default="",
                    help="workload the change targets, e.g. lang:go,task:boilerplate. "
                         "Uncovered claims are a REFUSAL, not a warning.")
    args = ap.parse_args()
    try:
        claims = parse_claims(args.claims)
    except ValueError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 2
    return report(load(args.a), load(args.b), load(args.aa), claims)


if __name__ == "__main__":
    sys.exit(main())
