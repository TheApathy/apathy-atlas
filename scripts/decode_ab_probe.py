#!/usr/bin/env python3
"""Four-workload decode probe against a local Atlas server (default :8977).

Reproduces the 2026-08-04 workload ladder (code/repeat/quote/prose) with
usage-block tok/s and a sha256 of each completion for text-exactness
comparison across configs. First run of a fresh server is warmup — run twice.

Usage: python3 scripts/decode_ab_probe.py [tag] [port] [runs]
Writes JSON to probe-<tag>.json and prints a table.
"""
import hashlib
import json
import sys
import time
import urllib.request

TAG = sys.argv[1] if len(sys.argv) > 1 else "probe"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8977
RUNS = int(sys.argv[3]) if len(sys.argv) > 3 else 2
BASE = f"http://127.0.0.1:{PORT}"

PROMPTS = {
    "code": "Write a Python class implementing an LRU cache with get/put in "
            "O(1) using a doubly linked list and dict. Include type hints and "
            "docstrings.",
    "repeat": "Repeat the phrase 'the quick brown fox jumps over the lazy dog' "
              "exactly 40 times, numbered 1. to 40.",
    "quote": "Quote the first stanza of 'The Road Not Taken' by Robert Frost, "
             "then explain each line briefly.",
    "prose": "Write a thoughtful 400-word essay on the history of lighthouse "
             "keeping and why it faded as a profession.",
}


def model_id() -> str:
    with urllib.request.urlopen(f"{BASE}/v1/models", timeout=30) as r:
        return json.load(r)["data"][0]["id"]


def run_one(model: str, prompt: str) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 300,
    }
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    wall = time.time() - t0
    u = d.get("usage", {})
    content = d["choices"][0]["message"].get("content") or ""
    # Atlas reports response_token/s in the usage block when available.
    toks = u.get("completion_tokens", 0)
    tps = u.get("response_token/s") or u.get("response_tokens_per_second")
    if tps is None and toks:
        tps = toks / wall  # includes TTFT — lower bound
    return {
        "tok_s": round(float(tps), 2) if tps else None,
        "completion_tokens": toks,
        "wall_s": round(wall, 2),
        "sha256": hashlib.sha256(content.encode()).hexdigest()[:16],
        "prefix": content[:60].replace("\n", " "),
    }


def main() -> None:
    model = model_id()
    out = {"tag": TAG, "model": model, "runs": []}
    for run in range(RUNS):
        row = {}
        for name, prompt in PROMPTS.items():
            row[name] = run_one(model, prompt)
            print(
                f"run{run} {name:7s} {row[name]['tok_s']} tok/s  "
                f"sha={row[name]['sha256']}  {row[name]['prefix'][:40]}"
            )
        out["runs"].append(row)
    with open(f"probe-{TAG}.json", "w") as f:
        json.dump(out, f, indent=2)
    # Summary: last run (steady state)
    last = out["runs"][-1]
    print("\nsteady-state:", " ".join(f"{k}={v['tok_s']}" for k, v in last.items()))


if __name__ == "__main__":
    main()
