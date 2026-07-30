"""Shared configuration and log-scraping helpers for the Laguna decode harness.

Companion to env.sh: same variables, same defaults, so a Python script and the
shell script that launched its serve cannot disagree about which port they mean.

    from benchenv import CHAT_URL, default_log, spec_fraction

Nothing here has a machine-specific default. Paths are derived from this file's
location and everything else comes from the environment, which is what lets the
harness run from a fresh clone.
"""
import os
import re

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("LAGUNA_REPO") or os.path.abspath(os.path.join(BENCH_DIR, "..", ".."))
OUT_ROOT = os.environ.get("LAGUNA_OUT") or os.path.join(BENCH_DIR, "ab")

HOST = os.environ.get("LAGUNA_HOST", "127.0.0.1")
PORT = os.environ.get("LAGUNA_PORT", "8890")
BASE_URL = f"http://{HOST}:{PORT}"
CHAT_URL = f"{BASE_URL}/v1/chat/completions"


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
    """
    accepts, verify, propose, suspend, reprobe = {}, [], [], 0, 0
    try:
        with open(path, errors="ignore") as fh:
            lines = fh.readlines()[start:]
    except OSError:
        return None
    for line in lines:
        # The serve emits "(K=6, accepted=3)" -- no "/N". An earlier version of
        # this regex required the "/N" form and therefore matched NOTHING for a
        # whole campaign: `dist` came back empty on every arm, every arm looked
        # histogram-identical, and the comparison that was supposed to
        # discriminate a numerics divergence silently never happened. A count of
        # zero must never be readable as "nothing wrong" -- see the non-empty
        # assertion in spec_fraction's callers.
        m = re.search(r"accepted=(\d+)", line)
        if m:
            accepts[int(m.group(1))] = accepts.get(int(m.group(1)), 0) + 1
        m = re.search(r"verify=([\d.]+)ms", line)
        if m:
            verify.append(float(m.group(1)))
        m = re.search(r"propose=([\d.]+)ms", line)
        if m:
            propose.append(float(m.group(1)))
        if "adaptive spec: SUSPENDED" in line:
            suspend += 1
        if "adaptive spec: RE-PROBING" in line:
            reprobe += 1
    total = sum(accepts.values())
    return {
        "steps": total,
        "mean_accept": (sum(k * v for k, v in accepts.items()) / total) if total else None,
        "dist": dict(sorted(accepts.items())),
        "verify_ms": (sum(verify) / len(verify)) if verify else None,
        "propose_ms": (sum(propose) / len(propose)) if propose else None,
        "adapt_suspend": suspend,
        "adapt_reprobe": reprobe,
    }


def spec_fraction(steps, mean_accept, tokens):
    """Fraction of emitted tokens that a speculative step actually committed.

    THE GUARD. `accepted=` is logged on speculative steps ONLY, so `mean_accept`
    is an average over the speculative SUBSET of a decode -- not over the decode.
    When ATLAS_DFLASH_ADAPTIVE suspends speculation for a request, the request
    finishes on the serial decoder but still reports the accept figure from the
    handful of steps that ran before the suspension. That reads perfectly
    plausibly as "DFlash ran and accepted little", and a serial row averaged into
    a DFlash table drags the mean down for a reason the table does not name.

    Each speculative step commits `accepted + 1` tokens, so for a row that really
    ran DFlash end to end:

        steps * (mean_accept + 1) / completion_tokens ~= 1

    Measured across a 15-row sweep: 13 genuine DFlash rows landed at 0.93-0.98
    and the 2 suspended rows at 0.17 and 0.20. There is no ambiguous middle, so
    a 0.9 cutoff separates them cleanly.

    Cross-check it against the serve's own `adaptive spec: SUSPENDED` lines
    (see scrape) rather than trusting either signal alone.

    Returns None when the inputs cannot support the ratio -- which callers must
    treat as "ungraded", a third state distinct from "graded and clean".
    """
    if not steps or mean_accept is None or tokens <= 0:
        return None
    return steps * (mean_accept + 1.0) / tokens


def is_dflash(scraped, tokens):
    """True if this row genuinely ran speculative decode end to end.

    Returns (verdict, spec_frac, agree) where `agree` records whether the two
    independent signals -- the token-accounting ratio and the serve's own
    suspension tracing -- tell the same story. They should. When they do not,
    the row is not evidence of anything until you find out why.
    """
    if not scraped:
        return None, None, None
    frac = spec_fraction(scraped["steps"], scraped["mean_accept"], tokens)
    if frac is None:
        # The scrape produced nothing to grade -- an empty accept histogram,
        # usually a serve that never logged step timing or a regex that stopped
        # matching. Returning False here would report "ran serial", which is a
        # finding; the truth is that no comparison happened. Keep them distinct.
        return None, None, None
    verdict = frac >= 0.9
    agree = verdict == (scraped.get("adapt_suspend", 0) == 0)
    return verdict, frac, agree
