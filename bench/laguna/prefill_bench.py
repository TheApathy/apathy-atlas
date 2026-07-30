#!/usr/bin/env python3
"""Measure PREFILL throughput (prompt tokens/sec) against a live Atlas serve.

Why a slope and not a ratio: a single `N / latency` at one length is
contaminated by the fixed per-request cost -- HTTP, chat-template render,
sampler setup, and the one decode step we are forced to pay because the API
has no "prefill only" mode. That fixed cost is on the order of ~2 s/turn, which
at N=1024 would understate prefill by more than 2x. Sweeping N and taking the
SLOPE of latency vs
prompt_tokens cancels every N-independent term; the intercept is then a
free readout of that fixed cost, which is a cross-check, not a nuisance.

Prefix caching is the trap. Atlas caches prompt prefixes across requests
(measured 2.8x on warm turns), so:
  - every trial gets a unique random prefix as its FIRST tokens, and
  - the unique part comes first, so no two prompts share a leading token.
Without that, the 4096 arm silently reuses the 1024 arm's KV and reports a
prefill throughput that no real first-turn request will ever see.

Usage:
  prefill_bench.py [--base-url URL] [--trials 3] [--json-out F]
  (--base-url defaults to benchenv.BASE_URL + "/v1"; override for a remote serve)
"""
import argparse
import json
import os
import random
import statistics
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchenv import BASE_URL  # noqa: E402

# Fixed vocabulary so tokens-per-word stays roughly constant across lengths --
# the sweep must vary token COUNT, not token difficulty.
VOCAB = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo "
         "lima mike november oscar papa quebec romeo sierra tango uniform "
         "victor whiskey xray yankee zulu").split()

# Words, not tokens: this vocabulary runs ~1.8 tokens/word, so 4096 words is
# ~7.3K tokens and the top of the sweep sits just under --max-seq-len 8192.
# Anything larger is rejected by the serve with a 400 and contributes nothing.
LENGTHS = [256, 512, 1024, 2048, 3072, 4096]


def make_prompt(n_words: int, seed: int) -> str:
    """Unique-prefix filler. The seed word goes FIRST so no prompt in this
    sweep shares a leading token with any other -- that is what defeats the
    prefix cache."""
    rng = random.Random(seed)
    head = f"session-{seed:08x}-{rng.getrandbits(48):012x}"
    body = " ".join(rng.choice(VOCAB) for _ in range(n_words))
    return f"{head} {body}"


def post(url: str, payload: dict, timeout: float):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    return time.perf_counter() - t0, body


def fit(xs, ys):
    """Least squares y = a*x + b, plus R^2."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    a = sxy / sxx
    b = my - a * mx
    ss_res = sum((y - (a * x + b)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return a, b, (1 - ss_res / ss_tot if ss_tot else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=f"{BASE_URL}/v1")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--json-out")
    a = ap.parse_args()
    url = a.base_url.rstrip("/") + "/chat/completions"

    # Warm-up: the first request after boot pays graph capture and allocator
    # growth. Grading it would attribute one-time cost to prefill.
    try:
        post(url, {"model": "laguna", "messages": [{"role": "user", "content": "hi"}],
                   "max_tokens": 1, "temperature": 0.0}, a.timeout)
    except urllib.error.URLError as e:
        sys.exit(f"FATAL: cannot reach {url}: {e}")

    rows, seed = [], 0
    print(f"{'req_words':>10}{'prompt_tok':>12}{'latency_s':>11}{'raw_tok/s':>11}")
    for n in LENGTHS:
        for t in range(a.trials):
            seed += 1
            payload = {"model": "laguna",
                       "messages": [{"role": "user", "content": make_prompt(n, seed)}],
                       "max_tokens": 1, "temperature": 0.0, "stream": False}
            try:
                dt, body = post(url, payload, a.timeout)
            except Exception as e:                      # noqa: BLE001
                print(f"{n:>10}  request failed: {e}")
                continue
            ptok = body.get("usage", {}).get("prompt_tokens")
            if not ptok:
                sys.exit("FATAL: serve returned no usage.prompt_tokens -- cannot "
                         "measure prefill against a token count we did not observe")
            rows.append({"words": n, "trial": t, "prompt_tokens": ptok, "latency_s": dt})
            print(f"{n:>10}{ptok:>12}{dt:>11.3f}{ptok / dt:>11.1f}")

    if len(rows) < 4:
        sys.exit("FATAL: too few successful trials to fit a slope")

    # Median per length first: the fit should describe the typical request, not
    # be dragged by one scheduler hiccup.
    by_len = {}
    for r in rows:
        by_len.setdefault(r["words"], []).append(r)
    xs = [statistics.median(x["prompt_tokens"] for x in v) for v in by_len.values()]
    ys = [statistics.median(x["latency_s"] for x in v) for v in by_len.values()]

    slope, icept, r2 = fit(xs, ys)
    prefill_tps = 1.0 / slope
    big = max(rows, key=lambda r: r["prompt_tokens"])
    raw_big = big["prompt_tokens"] / big["latency_s"]

    print(f"\nfit: latency = {slope * 1e3:.4f} ms/tok * N + {icept * 1e3:.0f} ms   R^2={r2:.4f}")
    print(f"PREFILL THROUGHPUT (slope, overhead removed): {prefill_tps:>8.0f} tok/s")
    print(f"raw N/latency at N={big['prompt_tokens']}          : {raw_big:>8.0f} tok/s")
    print(f"fixed per-request overhead (intercept)       : {icept * 1e3:>8.0f} ms")
    if r2 < 0.97:
        print("!! R^2 < 0.97 -- latency is not linear in N here. Either the prefix "
              "cache is hitting or the run is contended; do NOT quote the slope.")

    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump({"rows": rows, "slope_s_per_tok": slope, "intercept_s": icept,
                       "r2": r2, "prefill_tok_s": prefill_tps,
                       "raw_tok_s_at_max": raw_big,
                       "max_prompt_tokens": big["prompt_tokens"]}, f, indent=2)
        print(f"wrote {a.json_out}")


if __name__ == "__main__":
    main()
