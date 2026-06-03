#!/usr/bin/env python3
"""Fire N concurrent requests, report aggregate throughput.

Used to validate ATLAS_SSM_MULTI_SEQ_BATCHED: at concurrency >= 2 the
scheduler batches the SSM projections across active sequences; the
single-request bench can't surface that win because it never hits
num_seqs > 1 in decode_multi_seq_inner.

Usage:
  bench_concurrent_simple.py <port> <concurrency> [max_tokens]
"""
import sys
import time
import concurrent.futures
import requests

PORT = sys.argv[1] if len(sys.argv) > 1 else "8889"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 4
MAX_TOKENS = int(sys.argv[3]) if len(sys.argv) > 3 else 256

URL = f"http://localhost:{PORT}/v1/chat/completions"
PROMPT = (
    "Write a complete Python implementation of a binary search tree class "
    "with insert, delete, search, in-order traversal, height, and "
    "balance-check methods. Include full docstrings and inline comments. "
    "Just the code, no prose."
)
try:
    MODEL = requests.get(f"http://localhost:{PORT}/v1/models", timeout=5).json()["data"][0]["id"]
except Exception as e:
    sys.stderr.write(f"ERROR: cannot reach :{PORT} /v1/models — {e}\n")
    sys.exit(2)


def one_request(idx):
    t0 = time.time()
    r = requests.post(
        URL,
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        timeout=180,
    )
    wall = time.time() - t0
    u = r.json().get("usage", {})
    tps = u.get("response_token/s") or 0
    ct = u.get("completion_tokens", 0)
    return idx, ct, tps, wall


t_start = time.time()
with concurrent.futures.ThreadPoolExecutor(max_workers=N) as ex:
    futs = [ex.submit(one_request, i) for i in range(N)]
    results = [f.result() for f in concurrent.futures.as_completed(futs)]
total_wall = time.time() - t_start

total_ct = sum(r[1] for r in results)
agg_tps = total_ct / total_wall
print(f"concurrency={N}  max_tokens={MAX_TOKENS}")
for idx, ct, tps, wall in sorted(results):
    print(f"  req {idx}: ct={ct:4d} tps={tps:5.2f} wall={wall:.1f}s")
print(f"AGGREGATE: total_tokens={total_ct}  wall={total_wall:.1f}s  agg_tps={agg_tps:.2f}")
