#!/usr/bin/env python3
"""#39 CROSS-SEQ BATCHED DFLASH VERIFY — token-exactness + aggregate tok/s.

Fires `c` concurrent IDENTICAL counting completions (temp0, deterministic) and
reports each response's md5 + per-request tok/s + aggregate tok/s. The
token-exactness gate: every response md5 must equal the c=1 baseline md5
(recorded first). Also runs a MIXED-max_tokens batch to exercise mid-batch
compaction (sequences finishing at different steps).

Usage:
  python validate_v39_batched.py --port 8899 --model aeon-27b-v39 \
      --concurrency 1 2 4 8 --max-tokens 500
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import time
import urllib.request

COUNTING_PROMPT = "1, 2, 3, "


def complete(port: int, model: str, prompt: str, max_tokens: int):
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        }
    ).encode()
    url = f"http://127.0.0.1:{port}/v1/completions"
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    dt = time.time() - t0
    text = d["choices"][0]["text"]
    ct = d["usage"]["completion_tokens"]
    return {
        "md5": hashlib.md5(text.encode()).hexdigest()[:8],
        "tokens": ct,
        "wall": dt,
        "tok_s": ct / dt if dt > 0 else 0.0,
        "head": text[:60],
        "garbage": text.count("�") > 0,
    }


def fire_batch(port, model, jobs):
    """jobs: list of (prompt, max_tokens). Returns list of results, concurrently."""
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        futs = [ex.submit(complete, port, model, p, mt) for (p, mt) in jobs]
        results = [f.result() for f in futs]
    wall = time.time() - t0
    agg_tokens = sum(r["tokens"] for r in results)
    return results, wall, agg_tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--model", default="aeon-27b-v39")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--max-tokens", type=int, default=500)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    print(f"=== #39 VALIDATE port={args.port} model={args.model} {args.label} ===")

    # Baseline: c=1 counting md5 (the constitution reference for THIS boot).
    base, _, _ = fire_batch(
        args.port, args.model, [(COUNTING_PROMPT, args.max_tokens)]
    )
    base_md5 = base[0]["md5"]
    print(
        f"[c=1 baseline] md5={base_md5} tokens={base[0]['tokens']} "
        f"tok/s={base[0]['tok_s']:.1f} head={base[0]['head']!r}"
    )

    print("\n--- token-exactness + aggregate curve (identical counting prompts) ---")
    for c in args.concurrency:
        jobs = [(COUNTING_PROMPT, args.max_tokens)] * c
        results, wall, agg_tokens = fire_batch(args.port, args.model, jobs)
        md5s = [r["md5"] for r in results]
        all_match = all(m == base_md5 for m in md5s)
        agg_tok_s = agg_tokens / wall if wall > 0 else 0.0
        per_seq = agg_tok_s / c
        garbage = any(r["garbage"] for r in results)
        status = "EXACT" if all_match else "DIVERGE"
        print(
            f"c={c:2d}  md5s={md5s}  {status}  "
            f"agg_tok/s={agg_tok_s:6.1f}  per_seq={per_seq:5.1f}  "
            f"wall={wall:5.2f}s  garbage={garbage}"
        )

    # Mixed max_tokens: sequences finish at different steps → mid-batch
    # compaction. Each still must match the SAME-max_tokens single-stream md5,
    # so we record a reference per distinct max_tokens first.
    print("\n--- mixed max_tokens (mid-batch compaction) ---")
    mixed_mt = [200, 350, 500, 650]
    refs = {}
    for mt in sorted(set(mixed_mt)):
        r, _, _ = fire_batch(args.port, args.model, [(COUNTING_PROMPT, mt)])
        refs[mt] = r[0]["md5"]
    jobs = [(COUNTING_PROMPT, mt) for mt in mixed_mt]
    results, wall, agg_tokens = fire_batch(args.port, args.model, jobs)
    ok = True
    for mt, r in zip(mixed_mt, results):
        match = r["md5"] == refs[mt]
        ok = ok and match
        print(
            f"  max_tokens={mt:4d}  md5={r['md5']}  ref={refs[mt]}  "
            f"{'EXACT' if match else 'DIVERGE'}  tokens={r['tokens']}"
        )
    print(f"  mixed-batch aggregate tok/s = {agg_tokens / wall:.1f}  all_exact={ok}")


if __name__ == "__main__":
    main()
