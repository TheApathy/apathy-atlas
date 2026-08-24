#!/usr/bin/env python3
"""Locked single-stream decode sweep for an OpenAI-compatible Mia server."""

import argparse
import hashlib
import json
import statistics
import time
import urllib.request
import uuid


PROMPTS = {
    "code": "Write a complete Python LRU cache implementation and explain its invariants.",
    "math": "Derive the quadratic formula carefully, checking every algebraic step.",
    "prose": "Write a coherent short story about a lighthouse keeper during a storm.",
    "json": "Return a JSON array of 80 objects with id, name, category, and score fields.",
}


def stream_once(url: str, model: str, prompt: str, max_tokens: int) -> dict:
    nonce = uuid.uuid4().hex
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": f"run={nonce}\n{prompt}"}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    request = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first = None
    last = None
    text = []
    usage = {}
    with urllib.request.urlopen(request, timeout=1800) as response:
        for raw in response:
            if not raw.startswith(b"data: "):
                continue
            payload = raw[6:].strip()
            if payload == b"[DONE]":
                break
            event = json.loads(payload)
            usage = event.get("usage") or usage
            choices = event.get("choices") or []
            content = choices[0].get("delta", {}).get("content") if choices else None
            if content:
                now = time.perf_counter()
                first = now if first is None else first
                last = now
                text.append(content)
    ended = time.perf_counter()
    tokens = int(usage.get("completion_tokens", 0))
    decode_seconds = max(0.0, (last or ended) - (first or ended))
    decode_rate = (tokens - 1) / decode_seconds if tokens > 1 and decode_seconds else 0.0
    output = "".join(text)
    return {
        "completion_tokens": tokens,
        "ttft_seconds": (first or ended) - started,
        "decode_seconds": decode_seconds,
        "decode_tok_s": decode_rate,
        "wall_seconds": ended - started,
        "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8888")
    parser.add_argument("--model", default="deepseek-v4-flash-k2")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--output", default="mia-decode-sweep.json")
    args = parser.parse_args()

    results = {"config": vars(args), "cases": {}}
    for name, prompt in PROMPTS.items():
        stream_once(args.url, args.model, prompt, min(64, args.max_tokens))
        runs = [
            stream_once(args.url, args.model, prompt, args.max_tokens)
            for _ in range(args.reps)
        ]
        results["cases"][name] = {
            "median_decode_tok_s": statistics.median(r["decode_tok_s"] for r in runs),
            "runs": runs,
        }
        print(f"{name}: {results['cases'][name]['median_decode_tok_s']:.2f} tok/s")
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
