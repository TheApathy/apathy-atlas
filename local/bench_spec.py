#!/usr/bin/env python3
"""Bench Atlas Spark spec decode: tok/s per workload via /v1/completions."""
import json, sys, time, urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8897
MAXTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 800
URL = f"http://localhost:{PORT}/v1/completions"

WORKLOADS = {
    "counting": "Count from 1 to 400, one number per line.",
    "coding": "Write a complete Rust implementation of a generic LRU cache with get and put methods, full doc comments, and unit tests. Then explain the design.",
    "prose": "Write a detailed, vivid story about a lighthouse keeper who discovers a mysterious light in the fog. At least 700 words.",
}

def run(name, prompt):
    body = json.dumps({
        "model": "q122b-atlas", "prompt": prompt,
        "max_tokens": MAXTOK, "temperature": 0, "stream": False,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    dt = time.time() - t0
    ct = d["usage"]["completion_tokens"]
    text = d["choices"][0]["text"]
    toks = ct / dt
    garbage = text.count("�") > 0 or "NaN" in text[:200]
    print(f"{name:9s} tok={ct:4d} wall={dt:6.2f}s tok/s={toks:5.1f} garbage={garbage} head={text[:60]!r}")
    return toks

if __name__ == "__main__":
    print(f"=== BENCH port={PORT} max_tokens={MAXTOK} ===")
    for n, p in WORKLOADS.items():
        run(n, p)
