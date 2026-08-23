#!/usr/bin/env python3
"""Deterministic reproduction of Weschera's C=1 MinHeap decode probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
import urllib.request
from pathlib import Path
from typing import Any


PROMPT = (
    "Write a complete MinHeap class in Python with insert, extract_min, and heapify, "
    "then explain the complexity of each method."
)


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def get_json(url: str, *, body: dict[str, Any] | None = None) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        None if body is None else json.dumps(body).encode(),
        {} if body is None else {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=1200) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8896/v1/chat/completions")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=1500)
    parser.add_argument("--store-output", action="store_true")
    parser.add_argument("--thinking-default", action="store_true")
    args = parser.parse_args()

    models_url = args.endpoint.rsplit("/chat", 1)[0] + "/models"
    model = get_json(models_url)["data"][0]["id"]
    body: dict[str, Any] = {
        "model": model,
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "messages": [{"role": "user", "content": PROMPT}],
    }
    if not args.thinking_default:
        body["reasoning_effort"] = "none"

    runs = []
    for index in range(args.repetitions):
        started = time.monotonic()
        response = get_json(args.endpoint, body=body)
        wall_seconds = time.monotonic() - started
        choice = response["choices"][0]
        message = choice["message"]
        content = message.get("content") or ""
        reasoning = message.get("reasoning_content") or ""
        stable_output = {
            "content": content,
            "reasoning_content": reasoning,
            "finish_reason": choice.get("finish_reason"),
        }
        usage = response.get("usage", {})
        run = {
            "index": index,
            "wall_seconds": wall_seconds,
            "server_response_tokens_per_second": usage.get("response_token/s"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "finish_reason": choice.get("finish_reason"),
            "content_bytes": len(content.encode()),
            "content_sha256": sha256(content.encode()),
            "reasoning_bytes": len(reasoning.encode()),
            "reasoning_sha256": sha256(reasoning.encode()),
            "stable_output_sha256": sha256(canonical_bytes(stable_output)),
        }
        if args.store_output:
            run["content"] = content
            run["reasoning_content"] = reasoning
        runs.append(run)
        print(json.dumps(run, sort_keys=True), flush=True)

    rates = [run["server_response_tokens_per_second"] for run in runs]
    fingerprints = {
        (
            run["stable_output_sha256"],
            run["completion_tokens"],
            run["finish_reason"],
        )
        for run in runs
    }
    report = {
        "schema": "weschera-minheap-determinism-v1",
        "protocol": "thinking-default" if args.thinking_default else "effort-none",
        "concurrency": 1,
        "repetitions": args.repetitions,
        "endpoint": args.endpoint,
        "model": model,
        "prompt_sha256": sha256(PROMPT.encode()),
        "request_body_sha256": sha256(canonical_bytes(body)),
        "request_body": body,
        "runs": runs,
        "median_server_response_tokens_per_second": statistics.median(rates),
        "deterministic": len(fingerprints) == 1,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bytes(report) + b"\n")
    print(json.dumps({"output": str(args.output), **report}, sort_keys=True), flush=True)
    return 0 if report["deterministic"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
