#!/usr/bin/env python3
"""Probe one bisection leg: 3x the oracle prompt, report hashes vs the
plain-greedy oracle (0d20ac629078b9f9) + steady-state acceptance.

Usage: bisect-probe.py <server-log>
"""
import hashlib
import json
import re
import sys
import time
import urllib.request

PORT = 8977
ORACLE = "0d20ac629078b9f9"
P = "Explain how paged attention works in a modern LLM inference server."


def main() -> int:
    log = sys.argv[1]
    deadline = time.time() + 1200
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5).read()
            break
        except Exception:
            time.sleep(5)
    else:
        print("server never became healthy", file=sys.stderr)
        return 1

    mark = sum(1 for _ in open(log, errors="replace"))
    hits = 0
    for i in range(3):
        body = json.dumps(
            {"model": "d", "prompt": P, "max_tokens": 128, "temperature": 0.0}
        ).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/v1/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=900) as r:
            text = json.loads(r.read())["choices"][0]["text"]
        h = hashlib.sha256(text.encode()).hexdigest()[:16]
        ok = h == ORACLE
        hits += ok
        print(f"req{i}: hash={h} {'== ORACLE' if ok else '!= oracle'} chars={len(text)}")

    lines = open(log, errors="replace").read().splitlines()[mark:]
    st = [
        (float(m[1]), float(m[2]), int(m[3]))
        for line in lines
        if (m := re.search(r"verify=([0-9.]+)ms propose=([0-9.]+)ms .*?accepted=(\d+)", line))
    ]
    if st:
        n = len(st)
        v = sum(s[0] for s in st) / n
        p = sum(s[1] for s in st) / n
        a = sum(s[2] for s in st) / n
        print(
            f"steps={n} verify={v:.1f}ms propose={p:.1f}ms accepted={a:.2f} "
            f"tok/step={a + 1:.2f} implied={1000 * (a + 1) / (v + p):.1f} tok/s "
            f"oracle_hits={hits}/3"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
