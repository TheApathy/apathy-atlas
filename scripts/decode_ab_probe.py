#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only

"""Four-workload decode probe against a local Atlas server (default :8977).

Reproduces the code/repeat/quote/prose ladder with usage-block tok/s, full
completion hashes, response identity, finish reasons, and token accounting.
Warmups are stored separately; summaries cover five measured runs by default.

Usage: python3 scripts/decode_ab_probe.py [tag] [port] [measured-runs] [warmups]
Writes JSON to probe-<tag>.json and prints a table.
"""

import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_receipt  # noqa: E402

TAG = sys.argv[1] if len(sys.argv) > 1 else "probe"
PORT = int(sys.argv[2]) if len(sys.argv) > 2 else 8977
MEASURED_RUNS = int(sys.argv[3]) if len(sys.argv) > 3 else 5
WARMUP_RUNS = int(sys.argv[4]) if len(sys.argv) > 4 else 1
BASE = f"http://127.0.0.1:{PORT}"

PROMPTS = {
    "code": "Write a Python class implementing an LRU cache with get/put in "
    "O(1) using a doubly linked list and dict. Include type hints and "
    "docstrings.",
    "repeat": "Repeat the phrase 'the quick brown fox jumps over the lazy dog' "
    "exactly 40 times, numbered 1. to 40.",
    "quote": "Quote the first stanza of 'The Road Not Taken' by Robert Frost, "
    "then explain each line briefly.",
    "prose": "Write a thoughtful 400-word essay on the history of lighthouse "
    "keeping and why it faded as a profession.",
}

ENGINE_CLASSES = {
    "LOW_GEAR",
    "LOW_GEAR_MIXED",
    "MIXED",
    "SERIAL",
    "SPECULATIVE",
    "UNVERIFIED",
}


def parse_engine_usage(usage: dict) -> dict:
    engine = usage.get("atlas_engine")
    if type(engine) is not dict:
        return {
            "classification": "UNVERIFIED",
            "speculative_steps": 0,
            "serial_tokens": 0,
            "low_gear_steps": 0,
        }
    required = {
        "classification",
        "speculative_steps",
        "serial_tokens",
        "low_gear_steps",
    }
    if set(engine) != required or type(engine["classification"]) is not str:
        raise ValueError("atlas_engine usage is malformed")
    if engine["classification"] not in ENGINE_CLASSES:
        raise ValueError("atlas_engine classification is unsupported")
    for key in required - {"classification"}:
        if type(engine[key]) is not int or engine[key] < 0:
            raise ValueError("atlas_engine counters are malformed")
    expected = "UNVERIFIED"
    if engine["low_gear_steps"] > 0 and engine["speculative_steps"] > 0:
        expected = "LOW_GEAR_MIXED"
    elif engine["low_gear_steps"] > 0:
        expected = "LOW_GEAR"
    elif engine["speculative_steps"] > 0 and engine["serial_tokens"] > 0:
        expected = "MIXED"
    elif engine["speculative_steps"] > 0:
        expected = "SPECULATIVE"
    elif engine["serial_tokens"] > 0:
        expected = "SERIAL"
    if engine["classification"] != expected:
        raise ValueError("atlas_engine classification disagrees with counters")
    return engine


def model_id() -> str:
    with urllib.request.urlopen(f"{BASE}/v1/models", timeout=30) as r:
        return json.load(r)["data"][0]["id"]


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


def run_one(model: str, prompt: str) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 300,
    }
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    wall = time.perf_counter() - t0
    if type(d) is not dict or d.get("model") != model:
        raise ValueError(
            f"response model mismatch: expected {model!r}, got {d.get('model')!r}"
        )
    choices = d.get("choices")
    if type(choices) is not list or len(choices) != 1 or type(choices[0]) is not dict:
        raise ValueError("response choices are missing or malformed")
    choice = choices[0]
    finish_reason = choice.get("finish_reason")
    if finish_reason not in {"stop", "length"}:
        raise ValueError(f"unsupported finish_reason: {finish_reason!r}")
    message = choice.get("message")
    if type(message) is not dict or type(message.get("content")) is not str:
        raise ValueError("response content is missing or malformed")
    content = message["content"]
    reasoning = message.get("reasoning_content", "")
    if type(reasoning) is not str:
        raise ValueError("response reasoning_content is malformed")
    u = d.get("usage")
    if type(u) is not dict:
        raise ValueError("response usage is missing or malformed")
    prompt_tokens = u.get("prompt_tokens")
    completion_tokens = u.get("completion_tokens")
    total_tokens = u.get("total_tokens")
    if any(
        type(value) is not int
        for value in (prompt_tokens, completion_tokens, total_tokens)
    ):
        raise ValueError("response token accounting is missing or malformed")
    if (
        prompt_tokens < 0
        or completion_tokens <= 0
        or total_tokens != prompt_tokens + completion_tokens
    ):
        raise ValueError("response token accounting is inconsistent")
    # A/B results must use one comparable server-side decode metric. Wall-clock
    # completion_tokens/s includes TTFT and is intentionally not substituted.
    if u.get("response_token/s") is not None:
        tps = u["response_token/s"]
        metric_source = "usage.response_token/s"
    elif u.get("response_tokens_per_second") is not None:
        tps = u["response_tokens_per_second"]
        metric_source = "usage.response_tokens_per_second"
    else:
        raise ValueError("server-side response throughput is missing")
    if type(tps) not in (int, float) or isinstance(tps, bool) or tps <= 0.0:
        raise ValueError("response throughput is missing or malformed")
    engine = parse_engine_usage(u)
    return {
        "tok_s": round(float(tps), 2),
        "metric_source": metric_source,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "wall_s": round(wall, 2),
        "response_model": d["model"],
        "finish_reason": finish_reason,
        "truncated": finish_reason == "length",
        "sha256": hashlib.sha256(
            json.dumps(
                {"content": content, "reasoning_content": reasoning},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "engine": engine,
        "prefix": content[:60].replace("\n", " "),
    }


def main() -> None:
    if MEASURED_RUNS <= 0 or WARMUP_RUNS < 0:
        raise ValueError("measured runs must be positive and warmups nonnegative")
    model = model_id()
    receipt_path = os.environ.get("ATLAS_BENCH_RECEIPT")
    receipt_binding = load_receipt_binding(receipt_path)
    classification = receipt_binding["classification"]
    receipt_sha256 = receipt_binding["benchmark_receipt_sha256"]
    gpu_identity = receipt_binding["gpu_identity"]
    gpu_activation_state = receipt_binding["gpu_activation_state"]
    out = {
        "tag": TAG,
        "model": model,
        "warmup_runs": WARMUP_RUNS,
        "measured_runs": MEASURED_RUNS,
        "warmups": [],
        "runs": [],
        "classification": classification,
        "benchmark_receipt_sha256": receipt_sha256,
        "gpu_identity": gpu_identity,
        "gpu_activation_state": gpu_activation_state,
    }
    print(f"classification={classification} receipt={receipt_sha256 or 'none'}")

    def collect(phase: str, run: int) -> dict:
        row = {}
        for name, prompt in PROMPTS.items():
            row[name] = run_one(model, prompt)
            print(
                f"{phase}{run} {name:7s} {row[name]['tok_s']} tok/s  "
                f"engine={row[name]['engine']['classification']} "
                f"finish={row[name]['finish_reason']} sha={row[name]['sha256'][:16]}  "
                f"{row[name]['prefix'][:40]}"
            )
        return row

    for run in range(WARMUP_RUNS):
        out["warmups"].append(collect("warmup", run))
    for run in range(MEASURED_RUNS):
        out["runs"].append(collect("run", run))
    if receipt_path:
        final_binding = load_receipt_binding(receipt_path)
        if final_binding != receipt_binding:
            raise ValueError("benchmark receipt binding changed during measurement")
    classes = {
        row[name]["engine"]["classification"] for row in out["runs"] for name in PROMPTS
    }
    out["engine_classification"] = (
        next(iter(classes)) if len(classes) == 1 else "MIXED_CLASSIFICATIONS"
    )
    out["median_tok_s_by_engine"] = {}
    for name in PROMPTS:
        grouped = {}
        for row in out["runs"]:
            classification = row[name]["engine"]["classification"]
            grouped.setdefault(classification, []).append(row[name]["tok_s"])
        out["median_tok_s_by_engine"][name] = {
            classification: round(statistics.median(rates), 2)
            for classification, rates in sorted(grouped.items())
        }
    out["median_tok_s"] = (
        {
            name: round(statistics.median(row[name]["tok_s"] for row in out["runs"]), 2)
            for name in PROMPTS
        }
        if len(classes) == 1 and "UNVERIFIED" not in classes
        else None
    )
    with Path(f"probe-{TAG}.json").open("x", encoding="utf-8") as report_file:
        json.dump(out, report_file, indent=2, sort_keys=True)
        report_file.write("\n")
    print(
        f"\n{MEASURED_RUNS}-run median:",
        out["engine_classification"],
        " ".join(
            f"{name}={rates}" for name, rates in out["median_tok_s_by_engine"].items()
        ),
    )


if __name__ == "__main__":
    main()
