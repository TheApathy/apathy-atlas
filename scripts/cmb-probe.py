#!/usr/bin/env python3
"""Four-prompt greedy transparency + throughput probe.

Speculative decoding is supposed to be LOSSLESS: for temperature=0 the text a
speculative config emits must be byte-identical to plain greedy decode. So the
probe hashes the generated text rather than eyeballing it — a config that is
faster but shifts a hash has changed the model's output, not just its speed.

Usage:  scripts/cmb-probe.py <server-log> [max_tokens]
Prints one line per prompt (hash + tok/s) then the step-timing summary scraped
from the log lines this run appended.
"""

import hashlib
import json
import re
import subprocess
import sys
import time
import urllib.request

PORT = 8977
PROMPTS = [
    "Explain how paged attention works in a modern LLM inference server.",
    "Write a Python function that merges two sorted lists.",
    "What is the capital of France?",
    "Describe the tradeoffs between speculative decoding and larger batch sizes.",
]


def complete(prompt: str, max_tokens: int) -> tuple[str, float]:
    body = json.dumps(
        {"model": "deepseek", "prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0}
    ).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    dt = time.time() - t0
    text = d["choices"][0]["text"]
    n = d.get("usage", {}).get("completion_tokens") or max_tokens
    return text, n / dt


def main() -> int:
    log = sys.argv[1]
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 128

    # Wait for load (~6 min cold, less with a warm page cache).
    deadline = time.time() + 1200
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5).read()
            break
        except Exception:
            time.sleep(5)
    else:
        print("server never became healthy", file=sys.stderr)
        return 1

    mark = sum(1 for _ in open(log, errors="replace"))

    for i, p in enumerate(PROMPTS):
        text, tps = complete(p, max_tokens)
        h = hashlib.sha256(text.encode()).hexdigest()[:16]
        print(f"probe{i}: hash={h} tok/s={tps:.2f} chars={len(text)}")

    lines = open(log, errors="replace").read().splitlines()[mark:]
    # "DFLASH STEP_TIMING: verify=128.0ms propose=30.4ms (K=7, accepted=2)"
    steps = [
        (float(m[1]), float(m[2]), int(m[3]))
        for line in lines
        if (m := re.search(
            r"verify=([0-9.]+)ms propose=([0-9.]+)ms .*?accepted=(\d+)", line))
    ]
    if steps:
        n = len(steps)
        v = sum(s[0] for s in steps) / n
        pr = sum(s[1] for s in steps) / n
        st = v + pr
        ac = sum(s[2] for s in steps) / n
        print(
            f"steps={n} verify={v:.1f}ms propose={pr:.1f}ms step={st:.1f}ms "
            f"accepted={ac:.2f} tok/step={ac + 1:.2f} implied={1000 * (ac + 1) / st:.1f} tok/s"
        )
    else:
        print("no STEP_TIMING lines matched — check the log format")

    for pat in ("V4-tree route", "V4-msdecode route", "DFLASH_TREE", "DSpark markov chain"):
        hits = [line for line in lines if pat in line]
        if hits:
            print(f"[{pat}] x{len(hits)}: {hits[0].strip()[-160:]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
