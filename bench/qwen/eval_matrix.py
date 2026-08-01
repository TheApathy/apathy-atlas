#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Run one arm of the evaluation MATRIX: matched coding tasks x languages.

    python3 bench/qwen/eval_matrix.py --tag a --log ab/x/serve-a.log \
        --json-out ab/x/a.json --dump-text ab/x/text

WHY THIS EXISTS, stated as the failure it is meant to stop.

decode_bench.py's SUITE is five Python prompts and one prose prompt. Our
in-house drafter was retrained specifically to raise the GO share of its
corpus (1.6% -> 11.5%) and shipped on an A/B that measured `go +4.6%,
counting +1.2%, py neutral`. Neither piece of evidence overlapped the other.
Three weeks later the Python suite said it was 3.6% WORSE on accept than the
public checkpoint it was warm-started from -- and both results were correct,
because neither harness measured the workload the other one was about. A
change that is graded only where it was tuned, and a suite that grades only
where the change was not tuned, cannot between them answer whether to ship.

So this file is not "decode_bench with more prompts". The differences that
matter are all in the scorer (eval_verdict.py):

  * every cell is graded SEPARATELY against its own repeat, because a pooled
    mean over an arbitrary workload mix is exactly the number both sides of
    the GoHeavy argument were able to quote in their own favour;
  * the scorer REFUSES to emit a verdict unless the matrix covers the
    workload the change claims to target (--claims), so "we never tested the
    thing it was built for" is a hard failure and not a footnote;
  * the pooled delta is reported next to the delta under a re-weighted mix,
    and a SIGN FLIP between them is called out, because a pooled verdict that
    depends on the prompt mix is a statement about the mix.

MATCHED TASKS. Each task is one algorithm expressed in all three languages,
phrased identically apart from the language name and its idiom words. If the
prompts differed in difficulty as well as in language, a per-language gap
would not be attributable to language. The one genuine asymmetry -- C needs an
explicit length parameter where Python and Go carry length in the slice --
is recorded in TASKS rather than hidden.

WHAT WE ALREADY KNOW, so this run can falsify it rather than rediscover it.
On the sibling Laguna stack the per-TASK spread was 1.7x while the
per-LANGUAGE spread was only 1.11x -- so for a KERNEL change, task diversity
matters more than language diversity. But accept DID track language (Python
3.30 / C 3.15 / Go 2.87), and accept is the only quantity a drafter alone
controls. Hence: language axis for drafter changes, task axis for everything,
and both here because one harness that covers both is cheaper than two that
each cover half.

HASHES ARE REPORTED, NOT GRADED. On this stack committed text is
drafter-dependent at temperature 0 -- four of six rows changed when only the
drafter changed, on a greedy target where the committed tokens are the
target's argmax and ought not to. That is an open finding about the verify
path. Until it is explained, hash equality is not a correctness gate here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchenv import (  # noqa: E402
    CHAT_URL, GAMMA, MODEL_NAME, census, default_log, is_dflash, log_lines,
    scrape,
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

# (task_id, per-language clause). The clause completes "...a <LANG> function
# {clause}". C gets an explicit length/size parameter where Python and Go carry
# length in the slice/list -- a genuine language asymmetry, recorded not hidden.
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
    # Boilerplate-heavy. "repetitive" is the fastest class in the Python suite,
    # so this is the row where a predictability effect should show most.
    ("boilerplate", {
        "c": "s: eight structs named Point2..Point9, each with float fields x,y, and for each a function <name>_norm returning the Euclidean norm",
        "python": "s: eight dataclasses named Point2..Point9, each with float fields x,y and a method norm() returning the Euclidean norm",
        "go": "s: eight structs named Point2..Point9, each with float64 fields X,Y and for each a method Norm() returning the Euclidean norm",
    }),
]

# Non-code anchors, carried at lang "none" so they never contaminate a
# per-language mean. They are here because prose is the LOWEST-acceptance row
# in every suite we have run and therefore the one that anchors any
# token-weighted pooled figure. Dropping it would make the pooled number look
# better without anything getting faster.
ANCHORS = [
    ("prose", "Write three paragraphs about the history of the bicycle."),
    ("mathword", "A train leaves at 60 mph and another at 80 mph two hours later "
                 "on the same track. Work out when the second catches the first, "
                 "showing each step."),
]


def prompt_for(lang: str, clause: str) -> str:
    # The boilerplate row is plural ("functions"), so its clause starts with
    # "s:" and grafts onto the style string's "function" -> "functions:".
    return _STYLE[lang].format(extra=clause) \
        .replace("a C functions:", "C functions:") \
        .replace("a Python functions:", "Python functions:") \
        .replace("a Go functions:", "Go functions:")


def cells():
    """The full matrix, in run order. (cell_id, task, lang, prompt)."""
    for task_id, clauses in TASKS:
        for lang in LANGS:
            yield f"{task_id}/{lang}", task_id, lang, prompt_for(lang, clauses[lang])
    for task_id, prompt in ANCHORS:
        yield f"{task_id}/none", task_id, "none", prompt


def one(url: str, prompt: str, max_tokens: int) -> dict:
    body = {
        "model": MODEL_NAME,
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
        "tok_s": n / wall if wall > 0 else None,
        "hash": hashlib.sha256(text.encode()).hexdigest()[:16],
        "capped": n >= max_tokens,
        "text": text,
    }


_SERVER_TOK_S = re.compile(r"Done:.*?([\d.]+) tok/s")


def server_rates(path: str, start: int) -> list:
    """The serve's OWN tok/s for every request finished in the window.

    Client-side wall time bills queueing to the model. Under --max-batch-size 1
    the scheduler is FIFO, so a foreign request that lands mid-row pushes ours
    back and the client clock reports half the speed with a byte-IDENTICAL
    completion hash -- every correctness gate passes. The server figure is
    immune to that, so it is the one to prefer when the two disagree.
    """
    try:
        with open(path, errors="ignore") as fh:
            lines = fh.readlines()[start:]
    except OSError:
        return []
    return [float(m.group(1)) for line in lines
            if (m := _SERVER_TOK_S.search(line))]


def run_cell(url, log, prompt, max_tokens, retries, settle, backoff):
    """One matrix cell, re-run while the log window shows foreign traffic.

    A contended row is RE-RUN, never averaged in: it measures queueing, not
    decode. Back off past the burst rather than retrying straight into it. If
    every attempt is contended the row is still returned, flagged unclean, and
    the scorer excludes it -- an excluded cell is visible, a silently averaged
    one is not.
    """
    row = None
    for attempt in range(1, retries + 2):
        start = log_lines(log)
        r = one(url, prompt, max_tokens)
        time.sleep(settle)
        s = scrape(log, start)
        cen = census(log, start)
        srv = server_rates(log, start)
        verdict, frac, width_ok = is_dflash(s, r["completion_tokens"])
        row = {
            "tok_s": r["tok_s"],
            "server_tok_s": srv[0] if len(srv) == 1 else None,
            "completion_tokens": r["completion_tokens"],
            "capped": r["capped"],
            "hash": r["hash"],
            "text": r["text"],
            "mean_accept": (s or {}).get("mean_accept"),
            "steps": (s or {}).get("steps"),
            "dist": (s or {}).get("dist"),
            "denoms": (s or {}).get("denoms"),
            # census() returns None when the log could not be READ. That is a
            # fourth state -- not clean, not contaminated, not empty -- and it
            # must never collapse into any of them.
            "census": cen,
            "dflash": verdict,
            "spec_frac": frac,
            "width_ok": width_ok,
            "attempts": attempt,
        }
        row["clean"] = cen is not None and len(cen) == 1
        if row["clean"]:
            break
        if attempt <= retries:
            time.sleep(backoff)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=CHAT_URL)
    ap.add_argument("--log", default=default_log("serve.log"))
    ap.add_argument("--tag", default="run", help="arm name, e.g. a / b / a-p2")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--retries", type=int, default=2,
                    help="re-runs allowed per cell when the window is contended")
    ap.add_argument("--settle", type=float, default=0.6,
                    help="seconds to let the serve flush its log before scraping")
    ap.add_argument("--backoff", type=float, default=20.0,
                    help="seconds to wait past a foreign burst before retrying")
    ap.add_argument("--dump-text", default=None, metavar="DIR")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--langs", default=None, metavar="L,L",
                    help="restrict to these languages (e.g. 'go'). Narrows the "
                         "matrix on purpose -- for buying REPEATS on one noisy "
                         "axis, not for shopping a suite that flatters a change. "
                         "The arm JSON records it so eval_verdict's coverage "
                         "gate still sees what was and wasn't run.")
    args = ap.parse_args()

    if args.dump_text:
        os.makedirs(args.dump_text, exist_ok=True)

    matrix = list(cells())
    if args.langs:
        want = {s.strip() for s in args.langs.split(",") if s.strip()}
        unknown = want - set(LANGS) - {"none"}
        if unknown:
            print(f"FATAL: unknown language(s): {', '.join(sorted(unknown))}",
                  file=sys.stderr)
            return 2
        matrix = [c for c in matrix if c[2] in want]
        if not matrix:
            print("FATAL: --langs selected zero cells", file=sys.stderr)
            return 2
    print(f"=== eval_matrix [{args.tag}] :: {len(matrix)} cells, "
          f"gamma={GAMMA} max_tokens={args.tokens} ===")
    print(f"  {'cell':<22}{'tok/s':>8}{'srv':>7}{'tok':>6}{'cap':>5}"
          f"{'accept':>8}{'try':>5}  hash")

    rows = []
    for cell_id, task_id, lang, prompt in matrix:
        row = run_cell(args.url, args.log, prompt, args.tokens,
                       args.retries, args.settle, args.backoff)
        row.update({"cell": cell_id, "task": task_id, "lang": lang,
                    "arm": args.tag, "prompt": prompt})
        rows.append(row)

        if args.dump_text:
            with open(os.path.join(args.dump_text,
                                   f"{args.tag}.{cell_id.replace('/', '-')}.txt"),
                      "w") as fh:
                fh.write(row["text"])

        # "-" for an unmeasured cell, never 0. `or 0` renders an empty scrape
        # as accept 0.00, which reads as "the drafter accepted nothing" -- a
        # finding -- when the truth is that nothing was measured.
        acc = f"{row['mean_accept']:.2f}" if row["mean_accept"] is not None else "-"
        srv = f"{row['server_tok_s']:.1f}" if row["server_tok_s"] is not None else "-"
        marks = []
        if row["census"] is None:
            marks.append("** LOG UNREADABLE -- not clean, not contaminated, UNKNOWN")
        elif not row["clean"]:
            marks.append(f"** CONTENDED census={row['census']} -- EXCLUDED")
        if row["dflash"] is False:
            marks.append(f"** SUSPENDED->serial (spec_frac={row['spec_frac']:.2f}) "
                         f"-- accept is over the SPEC SUBSET only")
        elif row["dflash"] is None:
            marks.append("** UNGRADED -- nothing to classify; NOT the same as serial")
        if row["width_ok"] is False:
            marks.append(f"** WIDTH {row['denoms']} != [{GAMMA}] -- not comparable")
        print(f"  {cell_id:<22}{row['tok_s']:>8.1f}{srv:>7}"
              f"{row['completion_tokens']:>6}{('Y' if row['capped'] else '-'):>5}"
              f"{acc:>8}{row['attempts']:>5}  {row['hash'][:10]}"
              + ("  " + "  ".join(marks) if marks else ""), flush=True)

    graded = [r for r in rows if r["clean"] and r["dflash"] and r["width_ok"]]
    print(f"\n  {len(graded)}/{len(rows)} cells graded "
          f"(clean window, genuine DFlash, width {GAMMA})")
    for lang in LANGS + ("none",):
        n = sum(1 for r in graded if r["lang"] == lang)
        if n == 0:
            print(f"  !! ZERO graded cells for lang={lang} -- this arm carries NO "
                  f"evidence about {lang}")

    if args.json_out:
        payload = {
            "arm": args.tag, "gamma": GAMMA, "tokens": args.tokens,
            # The languages actually RUN, not the ones the module can produce.
            # Recording list(LANGS) here would let a --langs-narrowed arm claim
            # full coverage in its own JSON -- the "0 comparisons rendered as 0
            # differences" failure, one layer down.
            "langs": sorted({r["lang"] for r in rows}),
            "langs_available": list(LANGS),
            "langs_filter": args.langs, "n_cells": len(rows),
            # Text lives in the sidecar; keeping it out of the JSON keeps the
            # scorer's input small enough to read by eye when it disagrees.
            "rows": [{k: v for k, v in r.items() if k not in ("text", "prompt")}
                     for r in rows],
        }
        with open(args.json_out, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"  wrote {args.json_out}")

    # Exit non-zero when nothing was graded. A run that measured nothing must
    # not look like a run that measured no difference.
    return 0 if graded else 4


if __name__ == "__main__":
    sys.exit(main())
