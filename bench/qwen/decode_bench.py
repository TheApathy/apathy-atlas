#!/usr/bin/env python3
"""Canonical single-stream decode benchmark for the Qwen3.6-27B champion stack.

Fixed 6-prompt suite spanning the workload spread (repetitive -> novel logic).
Reports per-prompt decode tok/s, the suite mean, and the accept distribution
scraped from the serve log window opened by this run. Deterministic:
temperature 0, thinking off, fixed max_tokens.

Usage: decode_bench.py [--log SERVE_LOG] [--tag NAME] [--tokens N]
                       [--dump-text DIR] [--json-out FILE]

--dump-text writes each completion verbatim to DIR/<tag>.<name>.txt. The sha in
the JSON is over the completion TEXT, so a hash divergence between two arms says
only THAT they differ, never HOW. Dump both arms, then `diff -u`. Text goes to a
sidecar rather than into --json-out so downstream hash tables keep parsing.

Every row is classified DFlash-or-serial before it is aggregated; see
benchenv.spec_fraction for why a speculative-decode table that does not do this
can silently average serial rows into its mean.

NOTE ON THINKING. This suite sends enable_thinking=false, so it measures the
plain decode path and does NOT exercise ATLAS_THINK_SPEC. That gate was
quality-gated on a thinking-mode eval, not on this suite -- do not read these
numbers as evidence about it either way.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchenv import (  # noqa: E402
    CHAT_URL, GAMMA, MODEL_NAME, default_log, is_dflash, log_lines, scrape,
)

URL = CHAT_URL

# (name, prompt) -- ordered easy -> hard so the printout reads as a curve.
# Identical to the Laguna suite on purpose: the two stacks serve different models
# and are not directly comparable, but the workload spread should be, so a
# per-prompt shape difference is about the stack rather than about the prompts.
SUITE = [
    ("repetitive", "Write a Python module defining 8 dataclasses named Point2, Point3, Point4, Point5, Point6, Point7, Point8, Point9, each with float fields x,y and a method norm(). Code only."),
    ("easy-code",  "Write a Python binary search function with a docstring. Code only."),
    ("common-algo","Write a Python implementation of merge sort with a docstring and type hints. Code only."),
    ("novel-logic","Write a Python function that takes a list of (start,end) intervals and returns the minimum number of points needed so every interval contains at least one point. Include a docstring. Code only."),
    ("math",       "Write a Python function that computes the modular multiplicative inverse using the extended Euclidean algorithm, raising ValueError when no inverse exists. Code only."),
    ("prose",      "Explain in three paragraphs why speculative decoding improves LLM inference latency without changing output distribution."),
]


def one(prompt, max_tokens):
    body = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    resp = json.loads(urllib.request.urlopen(req, timeout=600).read())
    wall = time.time() - t0
    text = resp["choices"][0]["message"]["content"]
    usage = resp["usage"]
    return {
        "wall": wall,
        "completion_tokens": usage["completion_tokens"],
        "prompt_tokens": usage["prompt_tokens"],
        "tok_s": usage["completion_tokens"] / wall,
        "hash": hashlib.sha256(text.encode()).hexdigest()[:16],
        # Popped before --json-out is written; the sidecar owns the text.
        "text": text,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=default_log("serve-champion.log"))
    ap.add_argument("--tag", default="run")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--dump-text", default=None, metavar="DIR",
                    help="write each completion verbatim to DIR/<tag>.<name>.txt")
    args = ap.parse_args()

    if args.dump_text:
        os.makedirs(args.dump_text, exist_ok=True)

    start = log_lines(args.log)
    # warm the runtime so the first prompt isn't paying graph-capture cost
    one("Say OK.", 8)

    results = []
    print(f"=== decode_bench [{args.tag}] gamma={GAMMA} max_tokens={args.tokens} ===")
    for name, prompt in SUITE:
        # Per-prompt accept window: scrape only the log lines this prompt
        # produced. The hash is per-prompt, so a pooled suite-level histogram
        # could not be paired with it -- and that pairing is the whole
        # discriminator when two arms differ: same accept histogram + different
        # hash means a genuine numerics divergence, whereas a different
        # histogram means a different number of committed tokens per step, which
        # is expected and benign. Pooled stats cannot tell those apart.
        p_start = log_lines(args.log)
        r = one(prompt, args.tokens)
        r["name"] = name
        r["accept"] = scrape(args.log, p_start)
        r["dflash"], r["spec_frac"], r["width_ok"] = is_dflash(
            r["accept"], r["completion_tokens"]
        )
        results.append(r)
        if args.dump_text:
            dest = os.path.join(args.dump_text, f"{args.tag}.{name}.txt")
            with open(dest, "w") as fh:
                fh.write(r["text"])
        acc = r["accept"] or {}
        # A row that ran serial is marked in the row, not silently folded into
        # the mean below. `UNGRADED` means the scrape could not support the ratio
        # at all -- a third state, and not the same as clean.
        mark = "" if r["dflash"] else (
            " ** FELL BACK->serial" if r["dflash"] is False else " ** UNGRADED (empty scrape)"
        )
        if r["width_ok"] is False:
            mark += f" ** WIDTH {acc.get('denoms')} != [{GAMMA}]"
        print(
            f"  {name:<12} {r['tok_s']:6.1f} tok/s  "
            f"({r['completion_tokens']:4d} tok / {r['wall']:5.1f}s)  sha={r['hash']}"
            f"  acc={acc.get('dist')}{mark}"
        )

    mean = sum(r["tok_s"] for r in results) / len(results)
    # token-weighted: total tokens / total wall, the number that matters end-to-end
    weighted = sum(r["completion_tokens"] for r in results) / sum(r["wall"] for r in results)
    print(f"  {'-' * 54}")
    print(f"  suite mean      {mean:6.1f} tok/s")
    print(f"  token-weighted  {weighted:6.1f} tok/s")

    # Report the DFlash-only mean separately. The all-rows mean above is still
    # the honest end-to-end number for a user; this one is the only mean that may
    # be quoted as a property of speculative decoding.
    spec = [r for r in results if r["dflash"]]
    if len(spec) != len(results):
        n_serial = sum(1 for r in results if r["dflash"] is False)
        n_ungraded = sum(1 for r in results if r["dflash"] is None)
        if spec:
            print(f"  DFlash-only     {sum(r['tok_s'] for r in spec) / len(spec):6.1f} tok/s"
                  f"  ({len(spec)}/{len(results)} rows)")
        print(f"  !! {n_serial} row(s) fell back to serial decode,"
              f" {n_ungraded} ungraded -- excluded from the DFlash mean")

    s = scrape(args.log, start)
    if s and s["steps"]:
        print(f"  accept: steps={s['steps']} mean={s['mean_accept']:.2f}/{GAMMA} dist={s['dist']}")
        # Print "-" for an unmeasured phase, never 0. Step timing is off in the
        # champion config, so these are legitimately absent most of the time, and
        # "verify_ms=0.0" would read as a measured zero rather than as no
        # measurement. Same rule as the UNGRADED state above.
        v = f"{s['verify_ms']:.1f}" if s["verify_ms"] is not None else "-"
        p = f"{s['propose_ms']:.1f}" if s["propose_ms"] is not None else "-"
        print(f"  verify_ms={v} propose_ms={p}   (set QWEN_STEP_TIMING=1 for the phase breakdown)")
    else:
        # An empty scrape is reported, not skipped. Silence here would be
        # indistinguishable from a clean run with nothing to say.
        print("  !! accept scrape EMPTY -- no verify steps matched in this window.")
        print("     Either the serve is not running DFlash, or --log points at the wrong file.")

    if args.json_out:
        # Drop the text so --json-out stays a compact hash table; the sidecar is
        # the only place text lives.
        payload = []
        for r in results:
            row = dict(r)
            row.pop("text", None)
            payload.append(row)
        with open(args.json_out, "w") as fh:
            json.dump({"tag": args.tag, "gamma": GAMMA, "tokens": args.tokens,
                       "rows": payload, "suite_mean": mean, "weighted": weighted}, fh, indent=2)
        print(f"  wrote {args.json_out}")


if __name__ == "__main__":
    main()
