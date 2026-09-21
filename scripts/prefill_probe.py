#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only

"""Uncached prefill throughput probe at several prompt lengths.

prefill tok/s = verified fresh prompt tokens / streaming TTFT. Each request
uses a distinct nonce and must report exactly zero cached tokens in its final
usage chunk. Each length gets one warmup plus five measured requests; the probe
reports their median/range and emits every raw sample as JSON.
Usage: python3 scripts/prefill_probe.py [port]
"""

import json
import os
from pathlib import Path
import secrets
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_receipt  # noqa: E402

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8977
BASE = f"http://127.0.0.1:{PORT}"
MEASURED_RUNS = 5


def load_receipt_binding(path_value: str | None) -> dict:
    if not path_value:
        return {
            "classification": "UNGRADED",
            "benchmark_receipt_sha256": None,
            "gpu_identity": None,
            "gpu_activation_state": None,
        }
    envelope = benchmark_receipt.read_envelope(Path(path_value))
    activation = envelope.get("activation")
    return {
        "classification": benchmark_receipt.receipt_state(envelope, BASE),
        "benchmark_receipt_sha256": benchmark_receipt.receipt_digest(envelope),
        "gpu_identity": activation.get("gpu_identity") if activation else None,
        "gpu_activation_state": activation.get("gpu_state") if activation else None,
    }


def model_id() -> str:
    with urllib.request.urlopen(f"{BASE}/v1/models", timeout=30) as r:
        return json.load(r)["data"][0]["id"]


def make_prompt(target_words: int, nonce: str) -> str:
    body = " ".join(
        f"Fact {i}: division {i} reported revenue of {1000 + i * 7} units at a "
        f"margin of {10 + i % 20} percent in quarter {1 + i % 4}."
        for i in range(target_words // 18)
    )
    return f"run={nonce}\nSummarize the key figures in one sentence: " + body


def run(model: str, prompt: str) -> tuple[int, int, float, float]:
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
    t0 = time.perf_counter()
    ttft = None
    final_usage = None
    for line in urllib.request.urlopen(req, timeout=1800):
        line = line.decode().strip()
        if not line.startswith("data:") or line == "data: [DONE]":
            continue
        d = json.loads(line[5:])
        if ttft is None and d.get("choices"):
            delta = d["choices"][0].get("delta", {})
            if delta.get("reasoning_content") or delta.get("content"):
                ttft = time.perf_counter() - t0
        if d.get("choices") == [] and d.get("usage") is not None:
            final_usage = d["usage"]

    if not isinstance(final_usage, dict):
        raise ValueError("final SSE usage chunk is missing or malformed")
    prompt_tokens = final_usage.get("prompt_tokens")
    if type(prompt_tokens) is not int:
        raise ValueError("final usage prompt_tokens is missing or malformed")
    details = final_usage.get("prompt_tokens_details")
    if not isinstance(details, dict):
        raise ValueError("final usage prompt_tokens_details is missing or malformed")
    cached_tokens = details.get("cached_tokens")
    if type(cached_tokens) is not int:
        raise ValueError("final usage cached_tokens is missing or malformed")
    if cached_tokens != 0:
        raise ValueError(
            f"prefill benchmark requires zero cached tokens, got {cached_tokens}"
        )
    if ttft is None or ttft <= 0.0:
        raise ValueError("first streamed output timestamp is missing")
    fresh_tokens = prompt_tokens - cached_tokens
    return prompt_tokens, fresh_tokens, ttft, time.perf_counter() - t0


def sample_record(result: tuple[int, int, float, float]) -> dict:
    total, fresh, ttft, wall = result
    if ttft <= 0.0:
        raise ValueError("TTFT must be positive")
    return {
        "total_tokens": total,
        "fresh_tokens": fresh,
        "ttft_seconds": ttft,
        "wall_seconds": wall,
        "prefill_tok_s": fresh / ttft,
    }


def main() -> None:
    model = model_id()
    receipt_path = os.environ.get("ATLAS_BENCH_RECEIPT")
    receipt_binding = load_receipt_binding(receipt_path)
    report = {
        "schema": "atlas-prefill-probe-v1",
        "base_url": BASE,
        "model": model,
        "measured_runs": MEASURED_RUNS,
        "lengths": [],
        **receipt_binding,
    }
    print(
        f"classification={receipt_binding['classification']} "
        f"receipt={receipt_binding['benchmark_receipt_sha256'] or 'none'}"
    )
    for words in (700, 1500, 3000):
        warmup = make_prompt(words, nonce=secrets.token_hex(16))
        warmup_sample = sample_record(run(model, warmup))
        samples = []
        for _ in range(MEASURED_RUNS):
            prompt = make_prompt(words, nonce=secrets.token_hex(16))
            prompt += " Also note the final quarter."
            samples.append(sample_record(run(model, prompt)))
        rates = [sample["prefill_tok_s"] for sample in samples]
        ttfts = [sample["ttft_seconds"] for sample in samples]
        length_report = {
            "target_words": words,
            "warmup": warmup_sample,
            "samples": samples,
            "median_prefill_tok_s": round(statistics.median(rates), 6),
            "min_prefill_tok_s": round(min(rates), 6),
            "max_prefill_tok_s": round(max(rates), 6),
            "median_ttft_seconds": round(statistics.median(ttfts), 6),
        }
        report["lengths"].append(length_report)
        print(
            f"target_words={words:4d} samples={MEASURED_RUNS} "
            f"median={length_report['median_prefill_tok_s']:7.1f} tok/s "
            f"range={length_report['min_prefill_tok_s']:.1f}.."
            f"{length_report['max_prefill_tok_s']:.1f} "
            f"median_TTFT={length_report['median_ttft_seconds']:.2f}s"
        )
    if receipt_path:
        final_binding = load_receipt_binding(receipt_path)
        if final_binding != receipt_binding:
            raise ValueError("benchmark receipt binding changed during measurement")
    print("RESULT_JSON=" + json.dumps(report, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
