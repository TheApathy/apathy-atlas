#!/usr/bin/env python3
"""AEON-27B dense bench — 4-prompt suite matching atlas_patches/PERF_LOG.md.

Usage:
  bench_aeon27b.py <port> [label] [runs] [max_tokens]

Works against any OpenAI-compatible /v1/chat/completions (Atlas spark + vLLM
container both expose this).
"""
import sys
import time
import json
import requests

PORT = sys.argv[1] if len(sys.argv) > 1 else "8890"
LABEL = sys.argv[2] if len(sys.argv) > 2 else "atlas"
RUNS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
MAX_TOKENS = int(sys.argv[4]) if len(sys.argv) > 4 else 256

URL = f"http://localhost:{PORT}/v1/chat/completions"
try:
    MODEL = requests.get(f"http://localhost:{PORT}/v1/models", timeout=5).json()["data"][0]["id"]
except Exception as e:
    print(f"ERROR: cannot reach :{PORT} /v1/models — {e}", file=sys.stderr)
    sys.exit(2)

PROMPTS = [
    # Forced-length prompts so TTFT amortizes and we measure decode tok/s, not
    # request-overhead. Each is designed to produce ≥256 output tokens with
    # max_tokens=512.
    ("count100",  "Count from 1 to 100 separated by commas. Output ONLY the numbers, nothing else."),
    ("code_long", "Write a complete Python implementation of a binary search tree class with insert, delete, search, in-order traversal, height, and balance-check methods. Include full docstrings and inline comments. Just the code, no prose."),
    ("essay",     "Write a 500-word essay about the history of computing, covering Babbage, Turing, von Neumann, and the transistor. Use full paragraphs."),
    ("creative",  "Write a vivid short story of at least 400 words about a lone astronaut discovering an ancient signal on Mars. Include sensory detail."),
]

print(f"[{LABEL}] model={MODEL}  port={PORT}  runs={RUNS}  max_tokens={MAX_TOKENS}")
print(f"{'prompt':<10}  {'mean':>7}  {'peak':>7}  {'ttft':>7}  {'ct':>5}")

results = {}
for name, prompt in PROMPTS:
    tps_list = []
    ttft_list = []
    ct_list = []
    for _ in range(RUNS):
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
        }
        t0 = time.time()
        try:
            r = requests.post(URL, json=payload, timeout=240)
            r.raise_for_status()
            d = r.json()
        except Exception as e:
            print(f"  {name:<10}  ERROR  {e}")
            continue
        wall = time.time() - t0
        u = d.get("usage", {})
        # Try vendor fields; fall back to wall.
        tps = u.get("response_token/s") or u.get("response_tokens_per_second")
        ttft = u.get("time_to_first_token_ms", 0)
        ct = u.get("completion_tokens", 0)
        if not tps and ct and wall > 0:
            tps = ct / wall
        tps_list.append(tps or 0.0)
        ttft_list.append(ttft or 0.0)
        ct_list.append(ct or 0)
    if not tps_list:
        continue
    mean = sum(tps_list) / len(tps_list)
    peak = max(tps_list)
    avg_ttft = sum(ttft_list) / len(ttft_list)
    avg_ct = sum(ct_list) / len(ct_list)
    results[name] = {"mean": mean, "peak": peak, "ttft": avg_ttft, "ct": avg_ct}
    print(f"  {name:<10}  {mean:7.2f}  {peak:7.2f}  {avg_ttft:7.0f}  {avg_ct:5.0f}")

if results:
    means = [v["mean"] for v in results.values()]
    print(f"  {'MEAN':<10}  {sum(means)/len(means):7.2f}")

# Emit JSON for downstream aggregation
out_path = f"/tmp/bench_aeon_{LABEL}.json"
with open(out_path, "w") as f:
    json.dump({"label": LABEL, "port": PORT, "model": MODEL, "results": results}, f, indent=2)
print(f"  → {out_path}")
