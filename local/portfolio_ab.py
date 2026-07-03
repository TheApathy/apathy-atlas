#!/usr/bin/env python3
"""Portfolio-verify A/B: counting md5 gate + accept/tok-s on coding & prose.

Usage: portfolio_ab.py <port> <label>
Reads server /v1/completions. The counting prompt + settings match the
canonical md5 gate (ref 91a6ff90d50736f779c09db67a96db2d).
"""
import hashlib
import json
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8890
LABEL = sys.argv[2] if len(sys.argv) > 2 else "run"
URL = f"http://localhost:{PORT}/v1/completions"

MD5_PROMPT = "Count from 1 to 120 separated by commas:\n1, 2, 3,"
CODING = (
    "Write a Python module with a BankAccount class (deposit, withdraw with "
    "overdraft check, balance property, transaction history) and a unittest "
    "TestCase with 6 test methods. Docstrings + type hints. Code only."
)
PROSE = "Describe a thunderstorm rolling over a quiet fishing village at dusk."


def call(prompt, maxtok):
    body = json.dumps(
        {"model": "aeon-27b-dflash", "prompt": prompt,
         "max_tokens": maxtok, "temperature": 0, "seed": 0, "stream": False}
    ).encode()
    req = urllib.request.Request(
        URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    dt = time.time() - t0
    ct = d["usage"]["completion_tokens"]
    text = d["choices"][0]["text"]
    return text, ct, ct / dt if dt > 0 else 0.0


def main():
    print(f"=== PORTFOLIO A/B [{LABEL}] port={PORT} ===")
    # 1) md5 gate (max 400, matches canonical)
    text, ct, tps = call(MD5_PROMPT, 400)
    md5 = hashlib.md5(text.encode()).hexdigest()
    print(f"[{LABEL}] counting md5={md5} ct={ct} tok/s={tps:.1f} "
          f"{'== REF 91a6ff90' if md5 == '91a6ff90d50736f779c09db67a96db2d' else '!! DIVERGED'}")
    print(f"  head={text[:80]!r}")
    # 2) coding tok/s
    _, ct, tps = call(CODING, 1200)
    print(f"[{LABEL}] coding    ct={ct} tok/s={tps:.1f}")
    # 3) prose tok/s
    _, ct, tps = call(PROSE, 400)
    print(f"[{LABEL}] prose     ct={ct} tok/s={tps:.1f}")


if __name__ == "__main__":
    main()
