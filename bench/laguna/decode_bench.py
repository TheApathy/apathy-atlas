#!/usr/bin/env python3
"""Canonical single-stream decode benchmark for Laguna DFlash.

Fixed 6-prompt suite spanning the workload spread we measured (repetitive ->
novel logic). Reports per-prompt decode tok/s, the suite mean, and the
accept/step-timing distribution scraped from the serve log window opened by
this run. Deterministic: temperature 0, thinking off, fixed max_tokens.

Usage: decode_bench.py [--log SERVE_LOG] [--tag NAME] [--tokens N]
                       [--dump-text DIR]

--dump-text writes each completion verbatim to DIR/<tag>.<name>.txt. The sha in
the JSON is over the completion TEXT, so a hash divergence between two arms says
only THAT they differ, never HOW -- which cost a day of alarm on a drafter-FP8
ablation before anyone could tell a one-token near-tie flip from a semantic
fork. Dump both arms, then `diff -u`. Text goes to a sidecar rather than into
--json-out so downstream hash tables keep parsing.

Every row is classified DFlash-or-serial before it is aggregated; see
benchenv.spec_fraction for why a speculative-decode table that does not do this
silently averages serial rows into its mean.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchenv import CHAT_URL, census, default_log, is_dflash, log_lines, scrape  # noqa: E402

URL = CHAT_URL

# (name, prompt) — ordered easy -> hard so the printout reads as a curve.
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
        "model": "laguna",
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
    ap.add_argument("--log", default=default_log("serve-persist.log"))
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
    print(f"=== decode_bench [{args.tag}] max_tokens={args.tokens} ===")
    for name, prompt in SUITE:
        # Per-prompt accept window: scrape only the log lines this prompt
        # produced. The suite-level scrape below is unchanged and still owns
        # the reported mean/dist; this is strictly additional.
        #
        # WHY PER-PROMPT. The hash is per-prompt but the suite-level accept
        # dist is pooled over all 6, so the two cannot be paired. Pairing is
        # the whole discriminator for the FP8 divergence question: same accept
        # histogram + different hash = genuine numerics divergence; different
        # histogram = the accept-count -> batch-shape -> reduction-order path,
        # which is expected. Pooled stats cannot tell those apart.
        p_start = log_lines(args.log)
        r = one(prompt, args.tokens)
        r["name"] = name
        r["accept"] = scrape(args.log, p_start)
        r["dflash"], r["spec_frac"], r["engine_agree"] = is_dflash(
            r["accept"], r["completion_tokens"]
        )
        results.append(r)
        if args.dump_text:
            dest = os.path.join(args.dump_text, f"{args.tag}.{name}.txt")
            with open(dest, "w") as fh:
                fh.write(r["text"])
        acc = r["accept"] or {}
        # A row that ran serial is marked in the row, not silently folded into
        # the mean below. `?` means the scrape could not support the ratio at
        # all -- ungraded, which is a third state and not the same as clean.
        mark = "" if r["dflash"] else (" ** SUSPENDED->serial" if r["dflash"] is False else " ** UNGRADED")
        if r["engine_agree"] is False:
            mark += " ** ENGINE SIGNALS DISAGREE"
        print(
            f"  {name:<12} {r['tok_s']:6.1f} tok/s  "
            f"({r['completion_tokens']:4d} tok / {r['wall']:5.1f}s)  sha={r['hash']}"
            f"  acc={acc.get('dist')}{mark}"
        )

    mean = sum(r["tok_s"] for r in results) / len(results)
    # token-weighted: total tokens / total wall, the number that matters end-to-end
    weighted = sum(r["completion_tokens"] for r in results) / sum(r["wall"] for r in results)
    print(f"  {'-'*54}")
    print(f"  suite mean      {mean:6.1f} tok/s")
    print(f"  token-weighted  {weighted:6.1f} tok/s")

    # Report the DFlash-only mean separately. The all-rows mean above is still
    # the honest end-to-end number for a user; this one is the only mean that
    # may be quoted as a property of speculative decoding.
    spec = [r for r in results if r["dflash"]]
    if len(spec) != len(results):
        n_serial = sum(1 for r in results if r["dflash"] is False)
        n_ungraded = sum(1 for r in results if r["dflash"] is None)
        if spec:
            print(f"  DFlash-only     {sum(r['tok_s'] for r in spec) / len(spec):6.1f} tok/s"
                  f"  ({len(spec)}/{len(results)} rows)")
        print(f"  !! {n_serial} row(s) adaptive-suspended to serial decode,"
              f" {n_ungraded} ungraded -- excluded from the DFlash mean")

    s = scrape(args.log, start)
    if s and s["steps"]:
        print(
            f"  accept: steps={s['steps']} mean={s['mean_accept']:.2f} dist={s['dist']}"
        )
        if s["verify_ms"]:
            print(f"  verify_ms={s['verify_ms']:.1f} propose_ms={s['propose_ms'] or 0:.1f}")

    # Contamination census. We issued one warmup plus the suite; anything else
    # the serve finished in this window belonged to someone else, and under
    # FIFO it was spending our wall clock. This is checked AFTER the table is
    # printed so the forensic numbers are still visible, but the arm is not
    # allowed to produce a scoreable artifact.
    expect = 1 + len(SUITE)
    seen = census(args.log, start)
    # Not `> expect`. An unreadable log or a short count is also a failed
    # census, not a passed one: "we could not check" must not render the same
    # as "we checked and it was clean".
    contaminated = seen is None or len(seen) != expect
    if contaminated:
        ours = sorted(r["completion_tokens"] for r in results)
        print(f"\n  !! CENSUS FAILED: serve completed "
              f"{'<log unreadable>' if seen is None else len(seen)} requests, we issued {expect}.")
        if seen is not None:
            print(f"     completions seen : {sorted(seen)}")
            print(f"     ours (+ 8-tok warmup): {ours}")
        if seen is not None and len(seen) > expect:
            print("     A foreign client is on this port. Under --max-batch-size 1 its")
            print("     requests queue ahead of ours and the wait is billed as decode time,")
            print("     so rows read SLOW while their completion hashes stay correct.")
            print("     Free the port and re-run this arm; these tok/s are not about the model.")
        else:
            print("     Fewer completions than requests issued -- the log this arm was")
            print("     graded from is not the log its traffic went to. Do not score it.")

    if args.json_out:
        # Drop the text so --json-out stays byte-identical to pre-flag runs --
        # 256-token completions inline would bloat it and break eyeball diffs
        # of the hash table. The sidecar is the only place text lives.
        slim = [{k: v for k, v in r.items() if k != "text"} for r in results]
        # A contaminated arm is diverted to a sidecar path rather than written
        # to the one the driver grades. repro_table.sh refuses to score a
        # campaign with a missing arm JSON, so diverting turns a plausible-but-
        # wrong table into a loud refusal. Writing the real path and merely
        # exiting nonzero would still leave a full table on screen, and a
        # printed table is read as a result no matter what follows it.
        dest = (args.json_out + ".contaminated") if contaminated else args.json_out
        with open(dest, "w") as fh:
            json.dump({"tag": args.tag, "results": slim, "mean": mean,
                       "weighted": weighted, "scrape": s,
                       "contaminated": contaminated, "census": seen}, fh, indent=2)

    if contaminated:
        return 3


if __name__ == "__main__":
    sys.exit(main() or 0)
