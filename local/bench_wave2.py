#!/usr/bin/env python3
"""Wave-2 kernel A/B: counting md5 gate + n-run counting/coding tok/s.

Usage: bench_wave2.py <port> [n_runs] [label]
"""
import hashlib
import json
import statistics
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8890
N = int(sys.argv[2]) if len(sys.argv) > 2 else 5
LABEL = sys.argv[3] if len(sys.argv) > 3 else ""
MODEL = "aeon-27b-dflash"
URL = f"http://localhost:{PORT}/v1/completions"

MD5_PROMPT = "Count from 1 to 120 separated by commas:\n1, 2, 3,"
MD5_REF = "91a6ff90d50736f779c09db67a96db2d"

COUNTING = ("1, 2, 3, ", 500)
CODING = (
    "Write a Python module with a BankAccount class (deposit, withdraw with "
    "overdraft check, balance property, transaction history) and a unittest "
    "TestCase with 6 test methods. Docstrings + type hints. Code only.",
    1200,
)


def call(prompt, maxtok, seed=None):
    body = {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": maxtok,
        "temperature": 0,
        "stream": False,
    }
    if seed is not None:
        body["seed"] = seed
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        URL, data=data, headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.loads(r.read())
    dt = time.time() - t0
    ct = d["usage"]["completion_tokens"]
    text = d["choices"][0]["text"]
    return text, ct, dt


def md5_gate():
    text, ct, dt = call(MD5_PROMPT, 400, seed=0)
    h = hashlib.md5(text.encode()).hexdigest()
    ok = h == MD5_REF
    print(f"[md5] {h} {'== REF OK' if ok else '!= REF ' + MD5_REF + ' MISMATCH'}")
    print(f"[md5] head: {text[:80]!r}")
    return ok


def bench(name, prompt, maxtok):
    rates = []
    for i in range(N):
        text, ct, dt = call(prompt, maxtok)
        toks = ct / dt if dt > 0 else 0
        garbage = "�" in text
        rates.append(toks)
        print(f"  {name} run{i}: tok={ct:4d} wall={dt:6.2f}s tok/s={toks:5.1f}"
              f"{' GARBAGE' if garbage else ''}")
    mean = statistics.mean(rates)
    med = statistics.median(rates)
    print(f"[{name}] mean={mean:5.1f} median={med:5.1f} tok/s  (n={N}) {LABEL}")
    return mean, med


if __name__ == "__main__":
    print(f"=== WAVE2 BENCH port={PORT} n={N} label={LABEL!r} ===")
    md5_gate()
    which = "all"
    for arg in sys.argv[4:]:
        which = arg
    if which in ("all", "counting"):
        bench("counting", *COUNTING)
    if which in ("all", "coding"):
        bench("coding", *CODING)
