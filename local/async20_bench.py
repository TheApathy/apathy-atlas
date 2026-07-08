#!/usr/bin/env python3
"""Task #20 async propose‖verify — probe + A/B bench runner.

Usage: async20_bench.py <port> <label> [n_runs] [serve_log]

Runs: md5 gate, then counting/coding/prose n_runs each, recording the
serve-log byte offset before/after every request so the analysis can
slice per-modality step-timing windows.
"""

import hashlib
import json
import os
import statistics
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8890
LABEL = sys.argv[2] if len(sys.argv) > 2 else "probe"
N = int(sys.argv[3]) if len(sys.argv) > 3 else 5
SERVE_LOG = sys.argv[4] if len(sys.argv) > 4 else "/tmp/async20/serve_probe.log"
MODEL = "aeon-27b-dflash"
URL = f"http://localhost:{PORT}/v1/completions"

MD5_PROMPT = "Count from 1 to 120 separated by commas:\n1, 2, 3,"
MD5_REF = "91a6ff90d50736f779c09db67a96db2d"

BENCHES = {
    "counting": ("1, 2, 3, ", 500),
    "coding": (
        "Write a Python module with a BankAccount class (deposit, withdraw with "
        "overdraft check, balance property, transaction history) and a unittest "
        "TestCase with 6 test methods. Docstrings + type hints. Code only.",
        1200,
    ),
    "prose": (
        "Write a short story about a lighthouse keeper who discovers a "
        "mysterious light beneath the waves. Be vivid and original.",
        500,
    ),
}


def call(prompt, maxtok, seed=None):
    body = {"model": MODEL, "prompt": prompt, "max_tokens": maxtok,
            "temperature": 0, "stream": False}
    if seed is not None:
        body["seed"] = seed
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    dt = time.time() - t0
    return d["choices"][0]["text"], d["usage"]["completion_tokens"], dt


def log_size():
    try:
        return os.path.getsize(SERVE_LOG)
    except OSError:
        return 0


def main():
    out = {"label": LABEL, "n": N, "benches": {}}
    text, _, _ = call(MD5_PROMPT, 400, seed=0)
    h = hashlib.md5(text.encode()).hexdigest()
    out["md5"] = h
    out["md5_ok"] = h == MD5_REF
    print(f"[md5] {h} {'== REF OK' if out['md5_ok'] else '!= REF MISMATCH ' + MD5_REF}")

    for name, (prompt, maxtok) in BENCHES.items():
        rates, windows, toks_tot, wall_tot = [], [], 0, 0.0
        for i in range(N):
            off0 = log_size()
            text, ct, dt = call(prompt, maxtok)
            off1 = log_size()
            rate = ct / dt if dt > 0 else 0
            rates.append(rate)
            windows.append([off0, off1])
            toks_tot += ct
            wall_tot += dt
            print(f"  {name} run{i}: tok={ct:4d} wall={dt:6.2f}s tok/s={rate:5.1f}")
        out["benches"][name] = {
            "mean": statistics.mean(rates),
            "median": statistics.median(rates),
            "rates": rates,
            "log_windows": windows,
            "tokens": toks_tot,
            "wall": wall_tot,
        }
        print(f"[{name}] mean={statistics.mean(rates):5.1f} "
              f"median={statistics.median(rates):5.1f} tok/s (n={N}) {LABEL}")

    path = f"/tmp/async20/bench_{LABEL}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[saved] {path}")


if __name__ == "__main__":
    main()
