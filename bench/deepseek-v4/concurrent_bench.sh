#!/usr/bin/env bash
# Measure AGGREGATE decode throughput under concurrency.
#
# This is a different number from `decode_bench.sh`, and the difference is the
# whole point. Single-stream decode is bandwidth-bound: every token re-reads the
# full weight set, so tok/s is pinned by GB-per-token divided by achievable
# memory bandwidth. Under concurrency the dense weights are read ONCE per step
# and amortized across all in-flight sequences, so aggregate tok/s keeps climbing
# well past the single-stream ceiling even though each individual stream slows
# down.
#
# The published DGX Spark headline for this model -- "59 tok/s multi agent
# serving" -- is this measurement at 12 concurrent requests, not a single-stream
# number (that same post reports ~28 tok/s single-stream with speculation). So
# compare against 59 with this script and against 28 with decode_bench.sh; using
# the wrong one makes the engine look ~6x better or worse than it is.
#
#   PORT=8977 CONC=12 NTOK=256 bash bench/deepseek-v4/concurrent_bench.sh
#
# The server must have been started with --max-batch-size >= CONC, otherwise the
# extra requests queue instead of batching and this just measures single-stream
# throughput with extra steps. serve_single.sh takes MAX_BATCH for that.
set -uo pipefail
PORT="${PORT:-8977}" CONC="${CONC:-12}" NTOK="${NTOK:-256}"
MODEL="${MODEL:-/home/flocka/models/DeepSeek-V4-Flash-162B}"
export PORT CONC NTOK MODEL

python3 - <<'PY'
import json, os, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

PORT, CONC = int(os.environ["PORT"]), int(os.environ["CONC"])
NTOK, MODEL = int(os.environ["NTOK"]), os.environ["MODEL"]

# Distinct prompts on purpose. Identical prompts would share prefix-cache
# entries and make prefill look free, which flatters the aggregate number.
TOPICS = ["virtual memory paging", "TCP congestion control", "B-tree indexes",
          "CPU branch prediction", "garbage collection", "public-key crypto",
          "database MVCC", "CUDA memory coalescing", "the CAP theorem",
          "Linux io_uring", "column-oriented storage", "consistent hashing",
          "write-ahead logging", "SIMD vectorization", "RDMA networking",
          "copy-on-write forks"]

def one(i):
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user",
                      "content": f"Write a detailed technical explanation of {TOPICS[i % len(TOPICS)]}."}],
        "max_tokens": NTOK, "temperature": 0.0, "stream": False,
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    return d.get("usage", {}).get("completion_tokens", 0), time.time() - t0

# Warm up serially: the first request pays lazy init and graph capture, and
# folding that into the concurrent window would understate steady-state.
print("warmup...", flush=True)
one(0)

for conc in [int(c) for c in os.environ.get("SWEEP", str(CONC)).split(",")]:
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=conc) as ex:
        res = list(ex.map(one, range(conc)))
    wall = time.time() - t0
    toks = sum(n for n, _ in res)
    lat = sorted(t for _, t in res)
    # Aggregate = all tokens produced divided by the wall time of the whole
    # window, which is what "N tok/s serving" means. Per-stream is what any one
    # user actually feels.
    print(f"  conc={conc:>3}  aggregate={toks/wall:6.1f} tok/s  "
          f"per-stream={toks/wall/conc:5.1f}  "
          f"total={toks} tok in {wall:.1f}s  "
          f"latency p50={lat[len(lat)//2]:.1f}s p100={lat[-1]:.1f}s", flush=True)
PY
