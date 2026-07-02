#!/usr/bin/env python3
"""Capture deterministic greedy completions for fixed prompts, for byte-identical
lever verification. Writes one file per prompt; diff across lever configs.

Usage: greedy_capture.py <port> <out_prefix> [max_tokens]
Produces <out_prefix>.<name>.txt for each probe prompt.
"""
import json, sys, urllib.request

PORT = int(sys.argv[1])
OUT = sys.argv[2]
MAXTOK = int(sys.argv[3]) if len(sys.argv) > 3 else 256
URL = f"http://localhost:{PORT}/v1/completions"

PROBES = {
    "count": "Count from 1 to 120, one number per line.",
    "code": "Write a Rust function that reverses a linked list. Include a doc comment.",
    "prose": "Describe a thunderstorm rolling over a quiet fishing village at dusk.",
}

def run(name, prompt):
    body = json.dumps({
        "model": "aeon-27b-dflash", "prompt": prompt,
        "max_tokens": MAXTOK, "temperature": 0, "stream": False,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read())
    text = d["choices"][0]["text"]
    with open(f"{OUT}.{name}.txt", "w") as f:
        f.write(text)
    print(f"{name}: {len(text)} chars, head={text[:50]!r}")

if __name__ == "__main__":
    for n, p in PROBES.items():
        run(n, p)
