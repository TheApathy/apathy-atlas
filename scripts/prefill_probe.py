#!/usr/bin/env python3
"""Prefill throughput probe: streaming TTFT at several prompt lengths.

prefill tok/s = prompt_tokens / TTFT (time to first streamed token).
First request of a fresh server is warmup — each length runs twice, second
reported. Usage: python3 scripts/prefill_probe.py [port]
"""
import json
import sys
import time
import urllib.request

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8977
BASE = f"http://127.0.0.1:{PORT}"


def model_id() -> str:
    with urllib.request.urlopen(f"{BASE}/v1/models", timeout=30) as r:
        return json.load(r)["data"][0]["id"]


def make_prompt(target_words: int) -> str:
    body = " ".join(
        f"Fact {i}: division {i} reported revenue of {1000 + i * 7} units at a "
        f"margin of {10 + i % 20} percent in quarter {1 + i % 4}."
        for i in range(target_words // 18)
    )
    return "Summarize the key figures in one sentence: " + body


def run(model: str, prompt: str) -> tuple[int, float, float]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 8,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    t0 = time.time()
    ttft = None
    ptoks = 0
    for line in urllib.request.urlopen(req, timeout=1800):
        line = line.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        d = json.loads(line[5:])
        if ttft is None and d.get("choices") and d["choices"][0].get("delta", {}).get("content"):
            ttft = time.time() - t0
        if d.get("usage"):
            ptoks = d["usage"].get("prompt_tokens", 0)
    return ptoks, ttft or 0.0, time.time() - t0


def main() -> None:
    model = model_id()
    for words in (700, 1500, 3000):
        prompt = make_prompt(words)
        run(model, prompt)  # warmup / cache-state settle
        ptoks, ttft, wall = run(model, prompt + " Also note the final quarter.")
        rate = ptoks / ttft if ttft else 0.0
        print(f"prompt={ptoks:5d} tok  TTFT={ttft:6.2f}s  prefill={rate:7.1f} tok/s  (wall {wall:.2f}s)")


if __name__ == "__main__":
    main()
