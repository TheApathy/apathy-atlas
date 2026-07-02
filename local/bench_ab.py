#!/usr/bin/env python3
"""A/B bench for DFlash accept-fallback. total tok/s = completion_tokens / wall."""
import json, sys, time, urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8890
MODEL = sys.argv[2] if len(sys.argv) > 2 else "aeon-27b-dflash"
URL = f"http://localhost:{PORT}/v1/completions"

WORKLOADS = {
    "counting": ("1, 2, 3, ", 500),
    "coding": (
        "Write a Python module with a BankAccount class (deposit, withdraw with "
        "overdraft check, balance property, transaction history) and a unittest "
        "TestCase with 6 test methods. Docstrings + type hints. Code only.",
        1200,
    ),
}

def run(name, prompt, maxtok):
    body = json.dumps({
        "model": MODEL, "prompt": prompt,
        "max_tokens": maxtok, "temperature": 0, "stream": False,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    dt = time.time() - t0
    ct = d["usage"]["completion_tokens"]
    text = d["choices"][0]["text"]
    toks = ct / dt
    garbage = text.count("�") > 0 or "NaN" in text[:200]
    print(f"{name:9s} tok={ct:4d} wall={dt:6.2f}s tok/s={toks:5.1f} garbage={garbage}")
    print(f"  head: {text[:120]!r}")
    return toks, text

if __name__ == "__main__":
    which = sys.argv[3] if len(sys.argv) > 3 else "all"
    print(f"=== BENCH port={PORT} model={MODEL} ===")
    for n, (p, mt) in WORKLOADS.items():
        if which != "all" and which != n:
            continue
        run(n, p, mt)
