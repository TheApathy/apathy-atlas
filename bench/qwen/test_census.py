#!/usr/bin/env python3
"""Offline truth-table test for the contamination census. NO GPU, NO SERVE.

    python3 bench/qwen/test_census.py

WHY THIS EXISTS, AND WHY IT DOES NOT RUN THE BENCH. The census guard's whole job
is to refuse to score an arm whose port had a foreign client on it. If the guard
is buggy, the only way a live test can demonstrate that is by producing the very
event the guard exists to prevent -- a contaminated, scored, plausible-looking
arm. So the predicate is evaluated in ISOLATION against fixtures, and the states
are asserted by reading, not by racing another process for a port.

THE FIXTURE IS A VERBATIM SERVE LINE. `SAMPLE` below was copied out of a real log
from this stack, not written from memory. That matters more than it sounds: the
real line carries ANSI colour codes and a `spark::scheduler::lifecycle` target
prefix, and a from-memory fixture has neither. A guard tested only against clean
hand-written lines can pass its own test and still match nothing in production --
which is the exact shape of the accept-scrape bug this project already ate once,
where a regex wanted `accepted=3/5` and the serve emitted `accepted=3)`.

If you have a real serve log handy, point QWEN_CENSUS_FIXTURE at it and this will
additionally cross-check the built-in sample against it. Absent that, the format
claim rests on the embedded line, and the test says so rather than implying it
verified something it did not.

The states, and why each is separate:

    clean        n == expect            score it
    contaminated n >  expect            foreign client; move ports, do not re-run
    short        0 < n <  expect        graded from the wrong log
    zero-match   n == 0                 guard cannot read this format -- UNGRADED
    unreadable   log missing            guard could not run at all -- UNGRADED

The last two are the point. Collapsing them into "clean" is the failure this
project keeps paying for: a zero meaning "nothing to see" read as "nothing
wrong". Every instance so far produced the flattering answer.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchenv import census  # noqa: E402

# Verbatim from a serve log produced by this binary, ANSI and target prefix
# intact. Contains no paths -- a completion line never names one.
SAMPLE = (
    "\x1b[2m2026-07-21T15:23:48.991537Z\x1b[0m \x1b[32m INFO\x1b[0m "
    "\x1b[2mspark::scheduler::lifecycle\x1b[0m\x1b[2m:\x1b[0m "
    "Done: {n} tokens (stop) 19.3 tok/s, TTFT=890.6ms\n"
)
# Non-completion traffic that must NOT be counted. Also verbatim in shape.
NOISE = [
    "INFO Done processing 12 layers\n",
    "INFO prefill: 512 tokens in 210ms\n",
    "INFO Done: with warmup\n",
    "DFLASH K=y verify: y=16 accepted=15/16 (94%) seq_len=100\n",
    "INFO spark::server: listening\n",
]

EXPECT = 7  # 1 warmup + 6 suite prompts, matching decode_bench.SUITE

fails = []


def check(label, got, want):
    ok = got == want
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: got {got!r}")
    if not ok:
        fails.append(f"{label}: got {got!r}, want {want!r}")


def verdict(seen):
    """The exact predicate decode_bench.py computes. Spelled out here so a drift
    between this test and the bench is a visible edit rather than a silent one."""
    return seen is None or len(seen) != EXPECT


tmp = tempfile.mkdtemp(prefix="census-test-")


def write(name, lines):
    p = os.path.join(tmp, name)
    with open(p, "w") as fh:
        fh.writelines(lines)
    return p


def done(n):
    return SAMPLE.format(n=n)


def window(count, sizes=(256,)):
    """`count` completions, interleaved with noise so the guard is counting
    completions rather than lines."""
    out = []
    for i in range(count):
        out.append(NOISE[i % len(NOISE)])
        out.append(done(sizes[i % len(sizes)]))
    return out


# ---------------------------------------------------------------------------
# 0. Assert on the fixture itself before using it -- the input, not the output.
# ---------------------------------------------------------------------------
print("=== fixture ===")
check("sample line carries ANSI", "\x1b[" in SAMPLE, True)
check("sample parses to its token count", census(write("one.log", [done(204)]), 0), [204])

real = os.environ.get("QWEN_CENSUS_FIXTURE")
if real and os.path.exists(real):
    got = census(real, 0)
    check("real log cross-check: parses", got is not None and len(got) > 0, True)
    print(f"         {len(got)} completions in {real}, sizes {sorted(set(got))}")
else:
    print("  [NOTE] QWEN_CENSUS_FIXTURE unset or missing -- format claim rests on the")
    print("         embedded sample only. This is UNVERIFIED against a live log in")
    print("         this environment, which is not the same as verified.")

# ---------------------------------------------------------------------------
# 1..5 the five states
# ---------------------------------------------------------------------------
print("\n=== truth table ===")

seen = census(write("clean.log", window(EXPECT)), 0)
check("clean: count", len(seen), EXPECT)
check("clean: verdict is NOT contaminated", verdict(seen), False)

seen = census(write("dirty.log", window(EXPECT + 3)), 0)
check("contaminated: count", len(seen), EXPECT + 3)
check("contaminated: verdict", verdict(seen), True)
check("contaminated: routes to the foreign-client branch", len(seen) > EXPECT, True)

seen = census(write("short.log", window(EXPECT - 3)), 0)
check("short: count", len(seen), EXPECT - 3)
check("short: verdict", verdict(seen), True)
check("short: NOT the foreign branch", len(seen) > EXPECT, False)
check("short: NOT the zero branch", not seen, False)

# zero-match: a full log body with every completion line absent. This is what a
# renamed log line, a changed formatter, or the wrong --log looks like.
seen = census(write("zeromatch.log", NOISE * 20), 0)
check("zero-match: count", len(seen), 0)
check("zero-match: verdict is CONTAMINATED, not clean", verdict(seen), True)
check("zero-match: routes to the UNGRADED branch", seen is not None and not seen, True)

seen = census(os.path.join(tmp, "does-not-exist.log"), 0)
check("unreadable: returns None", seen, None)
check("unreadable: verdict", verdict(seen), True)
check("unreadable: is distinguishable from zero-match", seen is None, True)

# ---------------------------------------------------------------------------
# 6. Windowing. `start` is the entire basis for "this arm's window". If it were
#    ignored the guard would count the serve's whole history and every arm would
#    read contaminated -- which fails safe, but fails EVERY run, so it would be
#    switched off within a day and then there is no guard at all.
# ---------------------------------------------------------------------------
print("\n=== windowing ===")
lines = window(12)
p = write("full.log", lines)
check("start=0 sees all 12", len(census(p, 0)), 12)
# each completion is preceded by one noise line, so 2 lines per completion
check("start past the first 7 hides exactly 7", len(census(p, 14)), 5)
check("start past EOF sees nothing", len(census(p, len(lines) + 100)), 0)

# ---------------------------------------------------------------------------
# 7. False-positive resistance. An over-counting guard cries contamination on a
#    clean arm, and a guard that cries wolf gets disabled.
# ---------------------------------------------------------------------------
print("\n=== false-positive resistance ===")
check("noise alone counts 0", len(census(write("noise.log", NOISE), 0)), 0)
check("noise does not inflate a clean window",
      len(census(write("mixed.log", window(EXPECT) + NOISE), 0)), EXPECT)
check("varied completion sizes all counted",
      sorted(census(write("sizes.log", [done(x) for x in (2, 16, 204, 375)]), 0)),
      [2, 16, 204, 375])

print("\n" + "=" * 62)
if fails:
    print(f"FAILED {len(fails)} check(s):")
    for f in fails:
        print(f"  - {f}")
    sys.exit(1)
print("ALL CHECKS PASSED -- census predicate is sound on all five states.")
