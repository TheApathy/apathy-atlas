#!/usr/bin/env python3
"""Long-prompt greedy correctness probe for the MoE prefill tile gates.

WHY THIS EXISTS, AND WHY decode_bench.py CANNOT DO ITS JOB.

The tile gates only affect `run_routed_grouped_gemm`, and `forward_prefill`
routes there only when `num_tokens > 64`. Every prompt in decode_bench.py's
suite is a single sentence -- well under 64 tokens -- so its prefill takes
`forward_batched` and NEVER EXECUTES THE GATED CODE. Comparing those output
hashes across arms is a comparison that cannot fail: zero differences because
zero exposure, which is the exact "0 differences vs 0 comparisons" trap.

So this probe drives prompts of 256/1024/4096 words (~460/1850/7350 tokens),
all firmly in the grouped path, and checks the completion two ways:

  1. HASH, cross-arm. Greedy + identical prompt bytes => a correct arm is
     bit-identical to base. Sharp, but only meaningful RELATIVE to base.

  2. NEEDLE, absolute. Distinctive facts are planted at known fractional
     depths in the filler and the model is asked to recall one. This is the
     check that catches TRUNCATION: `MAX_LOAD_FACTOR=<N>` silently DROPS the
     overflow rows of any expert hotter than N x avg, degrading the hidden
     states rather than crashing. A dropped-row arm can still emit fluent
     text -- it just stops knowing things. Fluency is not correctness; we
     have mis-cleared a divergence on "the output looked plausible" before.

The needle is only usable if BASE recalls it. If base misses, the probe says
so and the scorer must fall back to hash-equality rather than scoring every
arm against a broken oracle.

Usage:
  prefill_hash_probe.py --tag base --json-out hash-base.json
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchenv import CHAT_URL  # noqa: E402

VOCAB = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo "
         "lima mike november oscar papa quebec romeo sierra tango uniform "
         "victor whiskey xray yankee zulu").split()

# All > 64 tokens so the grouped path is guaranteed. 4096 words ~= 7.3K tokens,
# just under --max-seq-len 8192 once the completion is allowed for.
LENGTHS = [256, 1024, 4096]

# Fixed: both arms MUST send byte-identical prompts or the hash check is
# meaningless. Never seed from the clock here.
SEED = 20260728

# Needles land at these fractional depths. The deep one matters most: an
# expert-row truncation early in the prompt corrupts everything downstream.
DEPTHS = [0.15, 0.55, 0.90]


def build(n_words: int):
    """Filler with planted needles. Returns (prompt, expected_code)."""
    rng = random.Random(SEED + n_words)
    words = [rng.choice(VOCAB) for _ in range(n_words)]
    codes = []
    for i, d in enumerate(DEPTHS):
        code = f"{rng.randrange(100000, 999999)}"
        codes.append(code)
        sentence = f"( the access code for sector {i} is {code} )"
        pos = min(int(n_words * d), max(0, n_words - 1))
        words[pos] = sentence
    # Query the DEEPEST needle: it is the one whose recall depends on the most
    # prefill having stayed intact.
    target = len(DEPTHS) - 1
    prompt = (
        "Read the following log and answer the question at the end.\n\n"
        + " ".join(words)
        + f"\n\nQuestion: what is the access code for sector {target}? "
        f"Reply with only the six digits."
    )
    return prompt, codes[target]


def ask(url, prompt, max_tokens, timeout):
    body = {
        "model": "laguna",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    return time.perf_counter() - t0, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=CHAT_URL)
    ap.add_argument("--tag", default="run")
    # 48 was too small and produced a silently USELESS oracle: the model opens
    # with a reasoning preamble ("Okay, let's see. The user is asking...") and
    # the cap truncated every completion before the digits, so base scored
    # 0/3 needles and looked like it couldn't recall. It could -- it was cut
    # off. Budget for the preamble, not just the answer.
    ap.add_argument("--max-tokens", type=int, default=320)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()

    print(f"=== prefill hash probe [{a.tag}] (grouped path, >64-token prompts) ===")
    print(f"    {'words':>7}{'p_tok':>8}{'wall':>8}  {'needle':>7}  hash")

    results = []
    for n in LENGTHS:
        prompt, expected = build(n)
        try:
            wall, out = ask(a.url, prompt, a.max_tokens, a.timeout)
        except Exception as e:                                   # noqa: BLE001
            sys.exit(f"FATAL: probe at {n} words failed: {e}\n"
                     f"A missing row is not a passing row -- refusing to write "
                     f"a partial artifact that would score as clean.")
        text = out["choices"][0]["message"]["content"]
        ptok = out.get("usage", {}).get("prompt_tokens")
        if not ptok:
            sys.exit("FATAL: no usage.prompt_tokens -- cannot confirm this "
                     "prompt was long enough to take the grouped path")
        if ptok <= 64:
            sys.exit(f"FATAL: {n} words tokenized to {ptok} <= 64, so this "
                     f"probe took forward_batched and never touched the gated "
                     f"kernel. The probe would prove nothing.")
        found = expected in text.replace(",", "").replace(" ", "")
        results.append({
            "words": n, "prompt_tokens": ptok, "wall": wall,
            "hash": hashlib.sha256(text.encode()).hexdigest()[:16],
            "needle_expected": expected, "needle_found": found,
            "text": text,
        })
        print(f"    {n:>7}{ptok:>8}{wall:>8.2f}  "
              f"{'HIT' if found else 'miss':>7}  {results[-1]['hash']}")

    hits = sum(r["needle_found"] for r in results)
    print(f"    needle recall: {hits}/{len(results)}")
    if a.json_out:
        with open(a.json_out, "w") as fh:
            json.dump({"tag": a.tag, "seed": SEED, "results": results}, fh, indent=2)
        print(f"    wrote {a.json_out}")


if __name__ == "__main__":
    main()
