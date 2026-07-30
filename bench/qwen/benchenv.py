"""Shared configuration and log-scraping helpers for the Qwen champion harness.

Companion to env.sh: same variables, same defaults, so a Python script and the
shell script that launched its serve cannot disagree about which port they mean.

    from benchenv import CHAT_URL, default_log, spec_fraction

Nothing here has a machine-specific default. Paths are derived from this file's
location and everything else comes from the environment, which is what lets the
harness run from a fresh clone.

This is the sibling of bench/laguna/benchenv.py but the scrape is NOT the same,
because the two servers do not log the same lines. Copying the Laguna regex here
yields an empty histogram on every run -- and an empty histogram is the failure
mode that looks like a clean result. Read GAMMA_DENOM below before editing.
"""
import os
import re

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("QWEN_REPO") or os.path.abspath(os.path.join(BENCH_DIR, "..", ".."))
OUT_ROOT = os.environ.get("QWEN_OUT") or os.path.join(BENCH_DIR, "ab")

HOST = os.environ.get("QWEN_HOST", "127.0.0.1")
PORT = os.environ.get("QWEN_PORT", "8890")
BASE_URL = f"http://{HOST}:{PORT}"
CHAT_URL = f"{BASE_URL}/v1/chat/completions"

MODEL_NAME = os.environ.get("QWEN_MODEL_NAME", "aeon-27b-dflash")
GAMMA = int(os.environ.get("QWEN_GAMMA", "16"))

# THE ACCEPT ANCHOR.
#
# Three different log families in this server contain the substring `accepted=`,
# and they are emitted under different conditions:
#
#   1. "DFLASH K=y verify: y=16 accepted=15/16 (94%) seq_len=..."
#        ungated tracing::info!, one per speculative step.  <-- the one we want
#   2. "DFLASH step timing: total=...us ... propose=...us accepted=15"
#        emitted only when ATLAS_DFLASH_STEP_TIMING=1, also one per step.
#   3. "DFLASH_EE_VERIFY: accepted=15/17 drafts[..min(6)]=[...]"
#        the early-exit verify path.
#
# A bare `accepted=(\d+)` matches all three. With step timing enabled that counts
# every step TWICE, which does not look like a bug: the histogram keeps its
# shape, the mean accept is unchanged, and only `steps` doubles -- so the guard
# in spec_fraction below reads ~2.0 instead of ~1.0 and every row is misreported
# as "not really DFlash". The failure is silent and points at the wrong thing.
#
# The anchor therefore requires the `N/M (P%)` shape, which family 1 has and the
# other two do not: family 2 has no slash, family 3 has no percentage. It is
# also deliberately ASCII-only -- the real line contains a Greek gamma, and
# matching on it would make the scrape depend on the log's encoding surviving a
# round trip through whatever locale the reader happens to have.
ACCEPT_RE = re.compile(r"accepted=(\d+)/(\d+)\s*\((\d+)%\)")

# Phase timings are microseconds here, not milliseconds as in the Laguna tree,
# and they live only on family 2. Gate on the ASCII substring first so the regex
# never has to see the mu.
STEP_TIMING_MARK = "DFLASH step timing:"
VERIFY_US_RE = re.compile(r"verify=(\d+)")
PROPOSE_US_RE = re.compile(r"propose=(\d+)")


def default_log(name="serve.log"):
    """Serve log path under the run-artifact root."""
    return os.path.join(OUT_ROOT, name)


def log_lines(path):
    """Line count, used to bracket a scrape window. Missing file counts as 0."""
    try:
        with open(path, errors="ignore") as fh:
            return len(fh.readlines())
    except OSError:
        return 0


def scrape(path, start):
    """Accept distribution and mean verify/propose ms over a log window.

    `start` is a line index from log_lines() taken BEFORE the request, so the
    window covers only what this request produced.

    `denoms` records every distinct M seen in `accepted=N/M`. It should be the
    single value {16}. Anything else means the serve is not running the width
    this harness documents, and callers surface it rather than averaging over it.
    """
    accepts, denoms, verify_us, propose_us = {}, set(), [], []
    try:
        with open(path, errors="ignore") as fh:
            lines = fh.readlines()[start:]
    except OSError:
        return None
    for line in lines:
        m = ACCEPT_RE.search(line)
        if m:
            n = int(m.group(1))
            accepts[n] = accepts.get(n, 0) + 1
            denoms.add(int(m.group(2)))
        if STEP_TIMING_MARK in line:
            mv = VERIFY_US_RE.search(line)
            if mv:
                verify_us.append(int(mv.group(1)))
            mp = PROPOSE_US_RE.search(line)
            if mp:
                propose_us.append(int(mp.group(1)))
    total = sum(accepts.values())
    return {
        "steps": total,
        "mean_accept": (sum(k * v for k, v in accepts.items()) / total) if total else None,
        "dist": dict(sorted(accepts.items())),
        "denoms": sorted(denoms),
        # None, never 0.0, when step timing was off. An unmeasured phase must not
        # render as a measured zero.
        "verify_ms": (sum(verify_us) / len(verify_us) / 1000.0) if verify_us else None,
        "propose_ms": (sum(propose_us) / len(propose_us) / 1000.0) if propose_us else None,
    }


def spec_fraction(steps, mean_accept, tokens):
    """Fraction of emitted tokens that a speculative step actually committed.

    THE GUARD. `accepted=` is logged on speculative steps ONLY, so `mean_accept`
    is an average over the speculative SUBSET of a decode, not over the decode.
    Any path that quietly finishes a request on the serial decoder still reports
    the accept figure from the steps that ran before it switched, which reads
    perfectly plausibly as "DFlash ran and accepted little" -- and a serial row
    averaged into a DFlash table drags the mean down for a reason the table does
    not name.

    Each speculative step commits `accepted + 1` tokens, so for a row that really
    ran DFlash end to end:

        steps * (mean_accept + 1) / completion_tokens ~= 1

    On the sibling Laguna stack, where an adaptive controller suspends
    speculation mid-request, this separated genuine from suspended rows with no
    ambiguous middle (0.93-0.98 against 0.17-0.20). The champion config here does
    NOT enable that controller, so the expected outcome is that every row passes.
    The guard is kept anyway because it is cheap and because "the mechanism that
    would break this is turned off" is an assumption about the config, which is
    exactly the kind of assumption that stops being true without announcing it.

    Returns None when the inputs cannot support the ratio -- which callers must
    treat as "ungraded", a third state distinct from "graded and clean".
    """
    if not steps or mean_accept is None or tokens <= 0:
        return None
    return steps * (mean_accept + 1.0) / tokens


def is_dflash(scraped, tokens):
    """True if this row genuinely ran speculative decode end to end.

    Returns (verdict, spec_frac, width_ok) where `width_ok` records whether the
    verify width the serve reported is the single documented value. A row at the
    wrong width is not comparable to the published numbers even if it is fast.
    """
    if not scraped:
        return None, None, None
    frac = spec_fraction(scraped["steps"], scraped["mean_accept"], tokens)
    if frac is None:
        # The scrape produced nothing to grade -- an empty accept histogram,
        # usually a serve that never logged a verify step or an anchor that
        # stopped matching. Returning False here would report "ran serial",
        # which is a finding; the truth is that no comparison happened. Keep
        # those two distinct.
        return None, None, None
    verdict = frac >= 0.9
    width_ok = scraped.get("denoms") == [GAMMA]
    return verdict, frac, width_ok
