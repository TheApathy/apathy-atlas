#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Offline truth-table test for eval_verdict.py. No GPU, no serve, no network.

    python3 bench/qwen/test_eval_verdict.py

WHY IT IS A SEPARATE FILE AND NOT A LIVE CHECK. eval_verdict.py's job is to
REFUSE in cases that should not reach a verdict. A guard tested by running the
real thing is tested only on the path where it does nothing: if the refusal is
broken, the test's failure mode IS the event the refusal exists to prevent.
So every state is asserted against a fixture instead, including the states a
healthy run never reaches.

The last section replays the REAL numbers from the 2026-07-31 drafter A/B --
the one whose evidence did not cover the workload its subject was built for --
and asserts that this scorer refuses it. That is the regression test for the
whole idea.
"""

from __future__ import annotations

import io
import sys

import eval_verdict as ev

FAILS = []


def check(name, got, want):
    ok = got == want
    print(f"  {'ok  ' if ok else 'FAIL'} {name}")
    if not ok:
        print(f"       got  {got!r}\n       want {want!r}")
        FAILS.append(name)


_DEFAULT = object()  # so census=None can mean "the log was UNREADABLE" and not
                     # "caller said nothing" -- the fixture has to be able to
                     # express the fourth state it is here to test.


def row(cell, lang, task, accept, tok_s, tokens=256, sha="deadbeefdeadbeef",
        clean=True, dflash=True, width_ok=True, census=_DEFAULT, **extra):
    """One matrix row in eval_matrix.py's on-disk shape.

    `steps` is derived as tokens/(accept+1) because each speculative step
    commits accept+1 tokens -- the same identity benchenv.spec_fraction uses.
    Deriving it here rather than inventing a number keeps the fixture's
    internal accounting consistent with the guard that reads it.
    """
    r = {
        "cell": cell, "lang": lang, "task": task,
        "mean_accept": accept, "tok_s": tok_s, "completion_tokens": tokens,
        "steps": round(tokens / (accept + 1.0)) if accept is not None else None,
        "hash": sha, "clean": clean, "dflash": dflash, "width_ok": width_ok,
        "census": [tokens] if census is _DEFAULT else census,
        "denoms": [16], "spec_frac": 1.0,
    }
    r.update(extra)
    return r


def arm(name, rows):
    return {"arm": name, "gamma": 16, "rows": rows}


def run(a, b, aa, claims=""):
    """Returns (exit_code, printed_text)."""
    buf = io.StringIO()
    code = ev.report(a, b, aa, ev.parse_claims(claims), out=buf)
    return code, buf.getvalue()


# --------------------------------------------------------------- unit rules

print("== rel(): a missing operand is None, never 0.0")
check("rel(2,1)", ev.rel(2.0, 1.0), 1.0)
check("rel(None,1) is None", ev.rel(None, 1.0), None)
check("rel(1,None) is None", ev.rel(1.0, None), None)
# 0.0 would print as '+0.0%' -- 'measured, no change', a finding. The truth is
# that no comparison happened, and the two must not render alike.
check("rel(1,0) is None (no divide, no fake zero)", ev.rel(1.0, 0.0), None)

print("\n== verdict(): TIE means 'not distinguishable', never 'proven equal'")
check("clears floor upward -> WIN", ev.verdict(0.10, 0.02), "WIN")
check("clears floor downward -> LOSS", ev.verdict(-0.10, 0.02), "LOSS")
check("inside floor -> TIE", ev.verdict(0.01, 0.02), "TIE")
check("exactly at floor -> TIE (not a win)", ev.verdict(0.02, 0.02), "TIE")
check("negative floor is used by magnitude", ev.verdict(0.10, -0.20), "TIE")
check("no delta -> '-'", ev.verdict(None, 0.02), "-")
check("no floor -> '-' (a delta with no floor is not a verdict)",
      ev.verdict(0.10, None), "-")

print("\n== parse_claims()")
check("two claims", ev.parse_claims("lang:go, task:modinv"),
      [("lang", "go"), ("task", "modinv")])
check("empty is empty", ev.parse_claims(""), [])
for bad in ("go", "langgo", "flavour:go"):
    try:
        ev.parse_claims(bad)
        check(f"rejects {bad!r}", "accepted", "ValueError")
    except ValueError:
        check(f"rejects {bad!r}", "ValueError", "ValueError")

print("\n== sign_flip()")
check("all positive", ev.sign_flip({"a": 0.1, "b": 0.2}), False)
check("mixed", ev.sign_flip({"a": 0.1, "b": -0.2}), True)
check("None ignored", ev.sign_flip({"a": 0.1, "b": None}), False)
check("all None", ev.sign_flip({"a": None}), False)

print("\n== graded(): a row must pass ALL of clean/dflash/width_ok")
pay = arm("x", [
    row("ok/python", "python", "ok", 5.0, 40.0),
    row("dirty/python", "python", "dirty", 5.0, 40.0, clean=False, census=[256, 9]),
    row("serial/python", "python", "serial", 5.0, 23.0, dflash=False),
    row("nograde/python", "python", "nograde", None, 40.0, dflash=None),
    row("width/python", "python", "width", 5.0, 40.0, width_ok=False, denoms=[8]),
])
check("only the clean DFlash row is graded", sorted(ev.graded(pay)), ["ok/python"])
check("every exclusion is NAMED", sorted(ev.ungraded_reasons(pay)),
      ["dirty/python", "nograde/python", "serial/python", "width/python"])
# 'ungraded' and 'ran serial' are different states; the reasons must not collide.
check("ungraded != serial in the reason text",
      ev.ungraded_reasons(pay)["nograde/python"]
      != ev.ungraded_reasons(pay)["serial/python"], True)
# An unreadable log is a FOURTH state -- not clean, not contended, not empty.
check("unreadable log says so",
      ev.ungraded_reasons(arm("x", [row("u/python", "python", "u", 5.0, 40.0,
                                        clean=False, census=None)]))["u/python"],
      "log unreadable")


# ------------------------------------------------------- the coverage gate

print("\n== COVERAGE GATE: a claim with zero graded cells is a REFUSAL")
py_only = [row(f"t{i}/python", "python", f"t{i}", 5.0, 40.0) for i in range(4)]
py_better = [row(f"t{i}/python", "python", f"t{i}", 6.0, 46.0) for i in range(4)]
A, B, AA = arm("a", py_only), arm("b", py_better), arm("a-p2", py_only)

code, txt = run(A, B, AA, claims="lang:go")
check("exit 3 when the claimed language is absent", code, 3)
check("says REFUSING", "REFUSING TO SCORE" in txt, True)
check("names the missing claim", "lang:go" in txt, True)
# The refusal must precede any pooled number, or it can be read past.
check("refusal comes BEFORE the per-cell table",
      "REFUSING TO SCORE" in txt and "-- per cell" not in txt, True)
check("and before the pooled row", "-- pooled" not in txt, True)

code, txt = run(A, B, AA, claims="lang:python")
check("exit 0 when the claim IS covered", code, 0)
check("scores it", "-- per cell" in txt, True)

code, txt = run(A, B, AA, claims="")
check("no claims -> still scores", code, 0)
check("but says nothing is being checked for", "NO --claims given" in txt, True)

code, txt = run(A, B, AA, claims="lang:python,task:t9")
check("a MISSING task also refuses", code, 3)
check("names task:t9", "task:t9" in txt and "<-- NONE" in txt, True)

print("\n== too few comparable cells is its own refusal, not a null result")
one_cell = [row("t0/python", "python", "t0", 5.0, 40.0)]
code, txt = run(arm("a", one_cell), arm("b", one_cell), arm("a-p2", one_cell))
check("exit 4", code, 4)
check("distinct from the coverage refusal (3 vs 4)", code != 3, True)
check("says nothing to compare", "nothing here to compare" in txt, True)

print("\n== a cell graded in only SOME arms drops out of the comparison")
half = arm("b", [row("t0/python", "python", "t0", 5.0, 40.0),
                 row("t1/python", "python", "t1", 5.0, 40.0, clean=False,
                     census=[256, 7]),
                 row("t2/python", "python", "t2", 5.0, 40.0),
                 row("t3/python", "python", "t3", 5.0, 40.0)])
code, txt = run(A, half, AA)
check("exit 0", code, 0)
check("3 of 4 comparable", "comparable in all three=3" in txt, True)
check("the dropped cell is named with its reason",
      "t1/python" in txt and "contended" in txt, True)


# ------------------------------------------------------- mix sensitivity

print("\n== pool(): three weightings, and a SIGN FLIP is called out")
# Two languages. B is better on go (+15%) and slightly worse on python (-5%).
# With SIX python cells and TWO go cells, the six small python losses outweigh
# the two large go wins when every cell counts equally, and do not when every
# language counts equally: by cell B loses, by language B wins. That is exactly
# the situation where one pooled number is a statement about the prompt mix.
# (Sized deliberately: an earlier draft of this fixture gave go +40%, which is
# large enough to win under BOTH weightings and so tested nothing.)
mixed_a = ([row(f"p{i}/python", "python", f"p{i}", 5.00, 40.0) for i in range(6)]
           + [row(f"g{i}/go", "go", f"g{i}", 4.00, 34.0) for i in range(2)])
mixed_b = ([row(f"p{i}/python", "python", f"p{i}", 4.75, 38.5) for i in range(6)]
           + [row(f"g{i}/go", "go", f"g{i}", 4.60, 39.0) for i in range(2)])
mixed_aa = ([row(f"p{i}/python", "python", f"p{i}", 5.01, 40.2) for i in range(6)]
            + [row(f"g{i}/go", "go", f"g{i}", 4.01, 34.2) for i in range(2)])
mrows = ev.cell_rows(ev.graded(arm("a", mixed_a)), ev.graded(arm("b", mixed_b)),
                     ev.graded(arm("a-p2", mixed_aa)))
d_cell = ev.pool(mrows, "accept", "cell")[0]
d_lang = ev.pool(mrows, "accept", "lang")[0]
check("by-cell says B is worse", d_cell < 0, True)
check("by-language says B is better", d_lang > 0, True)
check("sign_flip sees it", ev.sign_flip({"cell": d_cell, "lang": d_lang}), True)

code, txt = run(arm("a", mixed_a), arm("b", mixed_b), arm("a-p2", mixed_aa),
                claims="lang:go")
check("exit 0 -- go IS covered here", code, 0)
check("prints the SIGN FLIP warning", "SIGN FLIP" in txt, True)
check("names it as a claim about the mix", "not about B" in txt, True)
lang_block = txt.split("-- by language")[1].split("-- pooled")[0]
check("per-language: go is 2 win / 0 loss", "go" in lang_block
      and "2 win / 0 loss" in lang_block, True)
check("per-language: python is 0 win / 6 loss",
      "0 win / 6 loss" in lang_block, True)

print("\n== pool(): accept weights by STEPS, tok/s by TOKENS")
# One long cell and one short one, differing in direction. If accept were
# weighted by tokens instead of steps, the two metrics would silently share a
# weighting and one of them would be re-scaled by the other's units.
uneven_a = [row("long/python", "python", "long", 2.0, 30.0, tokens=512),
            row("short/python", "python", "short", 8.0, 60.0, tokens=64),
            row("mid/python", "python", "mid", 4.0, 40.0, tokens=128)]
uneven_b = [row("long/python", "python", "long", 2.4, 33.0, tokens=512),
            row("short/python", "python", "short", 7.0, 55.0, tokens=64),
            row("mid/python", "python", "mid", 4.0, 40.0, tokens=128)]
uneven_aa = [row(r["cell"], "python", r["task"], r["mean_accept"], r["tok_s"],
                 tokens=r["completion_tokens"]) for r in uneven_a]
urows = ev.cell_rows(ev.graded(arm("a", uneven_a)), ev.graded(arm("b", uneven_b)),
                     ev.graded(arm("a-p2", uneven_aa)))
check("token weighting differs from cell weighting",
      ev.pool(urows, "accept", "tokens")[0] != ev.pool(urows, "accept", "cell")[0],
      True)
check("pool returns (None, 0) when nothing is usable",
      ev.pool([], "accept", "cell"), (None, 0))


# --------------------------------------------------- hashes are NOT a gate

print("\n== hashes are reported, never graded")
h_a = [row(f"t{i}/python", "python", f"t{i}", 5.0, 40.0, sha=f"aaaa{i}") for i in range(4)]
h_b = [row(f"t{i}/python", "python", f"t{i}", 5.0, 40.0, sha=f"bbbb{i}") for i in range(4)]
code, txt = run(arm("a", h_a), arm("b", h_b), arm("a-p2", h_a))
check("every hash differs and it STILL exits 0", code, 0)
check("reported as ungraded", "REPORTED, NOT GRADED" in txt, True)
check("A vs B differ on 4/4", "A vs B differ on   4/4" in txt, True)
check("A vs A/A differ on 0/4", "A vs A/A differ on 0/4" in txt, True)
check("and says a stable-A cell isolates the change as the cause",
      "isolate the change as the cause" in txt, True)


# ------------------------------------------- replay of the real 2026-07-31 A/B

print("\n== REPLAY: the real drafter A/B, whose evidence missed its own target")
# Verbatim from bench/qwen/ab/drafter-ab, 2026-07-31. A = our in-house
# retrain, warm-started specifically to raise the GO share of its corpus;
# B = the public checkpoint it was warm-started from; a-p2 = A repeated.
# Five of the six suite prompts are Python and the sixth is prose. There is no
# Go anywhere in it.
REAL = {
    #  cell            lang      A accept, A tok/s, A tokens, A sha
    "repetitive":  ("python", (10.43, 67.5, 256, "c9f270a964bca544"),
                              (10.43, 68.2, 256, "c9f270a964bca544"),
                              (10.43, 62.4, 256, "c9f270a964bca544")),
    "easy-code":   ("python", (9.13, 51.7, 165, "0dd8f2b7e02cb3c3"),
                              (11.83, 64.3, 165, "0dd8f2b7e02cb3c3"),
                              (9.13, 56.8, 165, "0dd8f2b7e02cb3c3")),
    "common-algo": ("python", (9.78, 59.0, 256, "1007997cc9909ac0"),
                              (9.61, 62.6, 256, "cc6ba5e8f5cf4075"),
                              (9.78, 63.0, 256, "1007997cc9909ac0")),
    "novel-logic": ("python", (5.25, 36.7, 256, "89524f70c87146ea"),
                              (4.93, 38.2, 256, "37e4df3c3743fae0"),
                              (5.25, 39.3, 256, "89524f70c87146ea")),
    "math":        ("python", (5.06, 37.2, 226, "847ca2b3a724588d"),
                              (7.03, 48.4, 256, "9ef77fd24e888cfa"),
                              (4.89, 37.3, 226, "847ca2b3a724588d")),
    "prose":       ("none",   (1.58, 17.3, 256, "b9dd90c67c391cfc"),
                              (1.50, 16.3, 256, "59edafe9c9b09b79"),
                              (1.60, 17.5, 256, "30fba0ac6ff9610c")),
}
real = {"a": [], "b": [], "a-p2": []}
for task, (lang, *arms3) in REAL.items():
    for tag, (acc, tok, n, sha) in zip(("a", "b", "a-p2"), arms3):
        real[tag].append(row(f"{task}/{lang}", lang, task, acc, tok,
                             tokens=n, sha=sha))
RA, RB, RAA = arm("a", real["a"]), arm("b", real["b"]), arm("a-p2", real["a-p2"])

# THE REGRESSION TEST FOR THE WHOLE IDEA. The drafter under test claims to help
# Go. This evidence contains no Go. The old harness printed "-3.6% accept,
# clears the floor" -- a real number about the wrong workload. The new one
# must refuse.
code, txt = run(RA, RB, RAA, claims="lang:go")
check("the real run REFUSES under its own claim", code, 3)
check("and says the matrix grades zero go cells",
      "lang:go" in txt and "<-- NONE" in txt, True)

# Scored honestly -- i.e. with the claim dropped -- it still says what it said,
# but now per cell, and scoped out loud to what it actually measured.
code, txt = run(RA, RB, RAA, claims="lang:python")
check("scoped to python it scores", code, 0)
rrows = ev.cell_rows(ev.graded(RA), ev.graded(RB), ev.graded(RAA))
check("all six cells comparable", len(rrows), 6)
by_cell = {r["cell"]: r for r in rrows}
# math has an A/A floor of -3.4% on accept and a +38.9% delta: a real win for B.
check("math is a WIN for B on accept", by_cell["math/python"]["v_accept"], "WIN")
# repetitive/common-algo/novel-logic have a ZERO A/A accept floor, so any
# nonzero delta clears it. That is the n=1 floor being a lower bound, and the
# report says so in the header rather than pretending it is a tolerance.
check("repetitive ties (identical accept both arms)",
      by_cell["repetitive/python"]["v_accept"], "TIE")
check("novel-logic is a LOSS for B", by_cell["novel-logic/python"]["v_accept"], "LOSS")
check("pooled accept by cell is positive (B ahead on this mix)",
      ev.pool(rrows, "accept", "cell")[0] > 0, True)

# THE CONFOUND THE OLD HARNESS HAD. Pooling accept over each arm's OWN steps
# under-credits the better drafter, because a higher accept means fewer steps
# on that cell, which means that cell gets less weight in its own average.
def _pool_own_weights(rows):
    a = sum(r["accept_a"] * r["steps_a"] for r in rows) / sum(r["steps_a"] for r in rows)
    b = sum(r["accept_b"] * r["steps_b"] for r in rows) / sum(r["steps_b"] for r in rows)
    return ev.rel(b, a)


fixed = ev.pool(rrows, "accept", "tokens")[0]
own = _pool_own_weights(rrows)
check("fixed-weight pooling credits B MORE than own-weight pooling",
      fixed > own, True)
check("and the gap is material, not rounding (>3 points)",
      round((fixed - own) * 100, 1) > 3.0, True)
# Sanity on the mechanism: prose is the cell BOTH arms accept least on, and it
# therefore dominates the step weight for exactly that reason.
prose = by_cell["prose/none"]
check("prose has the lowest accept of any cell",
      prose["accept_a"] == min(r["accept_a"] for r in rrows), True)
check("...and the largest step weight",
      prose["steps_a"] == max(r["steps_a"] for r in rrows), True)

check("hash divergence surfaces: 4 of 6 differ A vs B",
      "A vs B differ on   4/6" in txt, True)
check("and exactly 1 is not self-reproducible (prose)",
      "A vs A/A differ on 1/6" in txt, True)
check("prose is the unstable one", "1/6: prose/none" in txt, True)


print()
if FAILS:
    print(f"FAILED {len(FAILS)}: {', '.join(FAILS)}")
    sys.exit(1)
print("all checks passed")
