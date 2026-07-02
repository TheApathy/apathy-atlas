#!/usr/bin/env python3
"""Long-context single-stream decode bench for the AEON TPS challenge.

Prepends a large filler context (~target token count) before a generation
instruction, then measures decode tok/s via the usage method
(completion_tokens / total wall). This exposes KV-cache levers (TurboQuant+,
nvfp4 KV, sliding window) whose win grows with context length — short-decode
is weight-bandwidth-bound and stays flat across KV dtypes.

Usage: bench_longctx.py <port> <max_tokens> <approx_prompt_tokens>
"""
import json, sys, time, urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8890
MAXTOK = int(sys.argv[2]) if len(sys.argv) > 2 else 800
PROMPT_TOK = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
URL = f"http://localhost:{PORT}/v1/completions"

# ~1 token per word-ish; build deterministic filler so runs are comparable.
_FILLER_SENT = (
    "The lighthouse stood on the rocky promontory, its weathered stone walls "
    "bearing witness to a century of storms, tides, and the slow patient work "
    "of salt and wind upon the ancient mortar that held it fast against the sea. "
)
# ~45 tokens per sentence; repeat to reach approx target.
reps = max(1, PROMPT_TOK // 40)
context = (_FILLER_SENT * reps)
PROMPT = (
    "Read the following passage carefully:\n\n" + context +
    "\n\nNow continue the narrative in the same vivid style for at least "
    "700 words, introducing a mysterious light seen through the fog."
)


def run():
    body = json.dumps({
        "model": "aeon-27b-dflash", "prompt": PROMPT,
        "max_tokens": MAXTOK, "temperature": 0, "stream": False,
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    dt = time.time() - t0
    ct = d["usage"]["completion_tokens"]
    pt = d["usage"].get("prompt_tokens", "?")
    text = d["choices"][0]["text"]
    toks = ct / dt
    garbage = text.count("�") > 0 or "NaN" in text[:200]
    print(f"longctx  prompt_tok={pt} gen={ct:4d} wall={dt:6.2f}s tok/s={toks:5.1f} "
          f"garbage={garbage} head={text[:50]!r}")
    return toks


if __name__ == "__main__":
    print(f"=== LONGCTX BENCH port={PORT} max_tokens={MAXTOK} approx_prompt_tok={PROMPT_TOK} ===")
    run()
