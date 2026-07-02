#!/usr/bin/env python3
"""TTFT prefill bench. Uses server-reported time_to_first_token_ms; unique prompts defeat prefix cache."""
import json, sys, time, urllib.request, random

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8890
MODEL = sys.argv[2] if len(sys.argv) > 2 else "aeon-27b-dflash"
SIZES = [50, 500, 2000, 4000]
if len(sys.argv) > 3:
    SIZES = [int(x) for x in sys.argv[3].split(",")]

def make_prompt(n_words):
    # unique tokens to defeat prefix caching
    tag = random.randint(0, 1_000_000_000)
    words = [f"w{tag}{i}" for i in range(n_words)]
    return " ".join(words)

def ttft(n_words, reps=2):
    best = None
    measured = []
    for _ in range(reps):
        prompt = make_prompt(n_words)
        body = json.dumps({
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": True,
        }).encode()
        req = urllib.request.Request(
            f"http://localhost:{PORT}/v1/chat/completions",
            data=body, headers={"Content-Type": "application/json"})
        t0 = time.time()
        server_ttft = None
        wall_first = None
        with urllib.request.urlopen(req, timeout=300) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except Exception:
                    continue
                ch = obj.get("choices", [{}])
                delta = ch[0].get("delta", {}) if ch else {}
                if wall_first is None and delta.get("content"):
                    wall_first = time.time() - t0
                u = obj.get("usage")
                if u and u.get("time_to_first_token_ms") is not None:
                    server_ttft = u["time_to_first_token_ms"]
        val = server_ttft if server_ttft is not None else (wall_first * 1000 if wall_first else None)
        if val is not None:
            measured.append(val)
            best = val if best is None else min(best, val)
    return best, measured

print(f"port={PORT} model={MODEL}")
ttft(50, reps=1)  # warmup
results = {}
for n in SIZES:
    best, measured = ttft(n)
    results[n] = best
    print(f"  {n:5d} tok prompt: best TTFT={best:8.1f} ms  (all: {[round(x,1) for x in measured]})")

# isolate per-token prefill rate using floor from smallest size
if len(SIZES) >= 2:
    s_lo, s_hi = SIZES[0], SIZES[-1]
    if results.get(s_lo) and results.get(s_hi):
        dt = (results[s_hi] - results[s_lo]) / 1000.0
        dtok = s_hi - s_lo
        if dt > 0:
            print(f"\n  linear-fit prefill rate ({s_lo}->{s_hi}): {dtok/dt:.1f} tok/s  (floor~{results[s_lo]:.0f}ms)")
