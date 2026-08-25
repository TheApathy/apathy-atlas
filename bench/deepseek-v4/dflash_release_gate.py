#!/usr/bin/env python3
"""Two-phase, single-GPU DeepSeek DFlash correctness and speed gate."""

import argparse
import hashlib
import json
import pathlib
import re
import tempfile
import time
import urllib.request


PROMPTS = {
    "code": "Write a complete Python LRU cache implementation with tests.",
    "math": "Derive the quadratic formula and verify the result algebraically.",
    "prose": "Write a detailed story about a lighthouse keeper during a storm.",
    "json": "Return a JSON array of 100 objects with id, name, category, and score.",
}
ACCEPT_RE = re.compile(
    r"DSPARK accept: ([0-9.]+) tok/step over ([0-9]+) steps .*?draft accept ([0-9.]+)%"
)


def parse_accept_log(path: pathlib.Path | None, start_offset: int = 0):
    if path is None:
        return None
    with path.open("rb") as handle:
        if path.stat().st_size >= start_offset:
            handle.seek(start_offset)
        text = handle.read().decode(errors="replace")
    matches = ACCEPT_RE.findall(text)
    if not matches:
        raise RuntimeError(f"no DSPARK accept summary in {path}")
    tok_step, steps, accept = matches[-1]
    return {
        "committed_tokens_per_step": float(tok_step),
        "steps": int(steps),
        "draft_accept_percent": float(accept),
    }


def write_json_atomic(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = pathlib.Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def stream_once(url: str, model: str, prompt: str, max_tokens: int) -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        separators=(",", ":"),
    ).encode()
    request = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first = None
    last = None
    reasoning = []
    content = []
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
            delta = choices[0].get("delta", {}) if choices else {}
            reasoning_piece = delta.get("reasoning_content") or delta.get("reasoning")
            content_piece = delta.get("content")
            if reasoning_piece or content_piece:
                now = time.perf_counter()
                first = first or now
                last = now
                if reasoning_piece:
                    reasoning.append(reasoning_piece)
                if content_piece:
                    content.append(content_piece)
    ended = time.perf_counter()
    reasoning_text = "".join(reasoning)
    content_text = "".join(content)
    transcript = json.dumps(
        {"reasoning": reasoning_text, "content": content_text},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    tokens = int(usage.get("completion_tokens", 0))
    decode_seconds = max(0.0, (last or ended) - (first or ended))
    return {
        "completion_tokens": tokens,
        "ttft_seconds": (first or ended) - started,
        "decode_seconds": decode_seconds,
        "decode_tok_s": (tokens - 1) / decode_seconds
        if tokens > 1 and decode_seconds
        else 0.0,
        "reasoning_sha256": hashlib.sha256(reasoning_text.encode()).hexdigest(),
        "content_sha256": hashlib.sha256(content_text.encode()).hexdigest(),
        "output_sha256": hashlib.sha256(transcript.encode()).hexdigest(),
    }


def run(args) -> None:
    if args.max_tokens <= 1:
        raise RuntimeError("max-tokens must be greater than one for decode timing")
    if args.reps <= 0:
        raise RuntimeError("reps must be positive")
    if not args.model_identity.strip() or not args.implementation_identity.strip():
        raise RuntimeError("model and implementation identities must be non-empty")
    if args.output.exists() and not args.overwrite:
        raise RuntimeError(f"refusing to overwrite existing output: {args.output}")
    accept_offset = args.server_log.stat().st_size if args.server_log else 0
    cases = {}
    for name, prompt in PROMPTS.items():
        runs = [
            stream_once(args.url, args.model, prompt, args.max_tokens)
            for _ in range(args.reps)
        ]
        cases[name] = {
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "runs": runs,
        }
    all_runs = [run for case in cases.values() for run in case["runs"]]
    decode_seconds = sum(item["decode_seconds"] for item in all_runs)
    decoded_intervals = sum(max(0, item["completion_tokens"] - 1) for item in all_runs)
    if decode_seconds <= 0 or decoded_intervals <= 0:
        raise RuntimeError("benchmark produced no measurable decode intervals")
    result = {
        "label": args.label,
        "model_identity": args.model_identity,
        "implementation_identity": args.implementation_identity,
        "contract": {
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "reps": args.reps,
        },
        "aggregate_decode_tok_s": decoded_intervals / decode_seconds,
        "acceptance": parse_accept_log(args.server_log, accept_offset),
        "cases": cases,
    }
    write_json_atomic(args.output, result)
    print(
        f"{args.label}: {result['aggregate_decode_tok_s']:.2f} tok/s -> {args.output}"
    )


def compare_results(baseline: dict, candidate: dict, min_tok_s: float, min_tok_step: float) -> dict:
    failures = []
    if not baseline.get("model_identity") or not candidate.get("model_identity"):
        failures.append("both records must declare model_identity")
    elif baseline["model_identity"] != candidate["model_identity"]:
        failures.append("model identities differ")
    if not baseline.get("implementation_identity"):
        failures.append("baseline has no implementation_identity")
    if not candidate.get("implementation_identity"):
        failures.append("candidate has no implementation_identity")
    if baseline["contract"] != candidate["contract"]:
        failures.append("benchmark contracts differ")
    for name in PROMPTS:
        left = baseline["cases"][name]
        right = candidate["cases"][name]
        if left["prompt_sha256"] != right["prompt_sha256"]:
            failures.append(f"{name}: prompt hash differs")
            continue
        left_hashes = [run["output_sha256"] for run in left["runs"]]
        right_hashes = [run["output_sha256"] for run in right["runs"]]
        if left_hashes != right_hashes:
            failures.append(f"{name}: output hashes differ")
    speed = candidate["aggregate_decode_tok_s"]
    if speed < min_tok_s:
        failures.append(f"decode {speed:.2f} tok/s is below {min_tok_s:.2f}")
    acceptance = candidate.get("acceptance")
    if min_tok_step:
        if acceptance is None:
            failures.append("candidate has no acceptance summary")
        elif acceptance["committed_tokens_per_step"] < min_tok_step:
            failures.append(
                f"acceptance {acceptance['committed_tokens_per_step']:.2f} tok/step "
                f"is below {min_tok_step:.2f}"
            )
    return {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "candidate_tok_s": speed,
        "acceptance": acceptance,
    }


def compare(args) -> None:
    baseline = json.loads(args.baseline.read_text())
    candidate = json.loads(args.candidate.read_text())
    report = compare_results(
        baseline, candidate, args.min_tok_s, args.min_tok_step
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["failures"]:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--url", default="http://127.0.0.1:8977")
    run_parser.add_argument("--model", default="deepseek-v4-flash-k2")
    run_parser.add_argument("--label", required=True)
    run_parser.add_argument(
        "--model-identity",
        required=True,
        help="immutable checkpoint revision or content hash shared by both runs",
    )
    run_parser.add_argument(
        "--implementation-identity",
        required=True,
        help="binary plus launch-config identity for this run",
    )
    run_parser.add_argument("--max-tokens", type=int, default=512)
    run_parser.add_argument("--reps", type=int, default=3)
    run_parser.add_argument("--server-log", type=pathlib.Path)
    run_parser.add_argument("--output", type=pathlib.Path, required=True)
    run_parser.add_argument("--overwrite", action="store_true")
    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("baseline", type=pathlib.Path)
    compare_parser.add_argument("candidate", type=pathlib.Path)
    compare_parser.add_argument("--min-tok-s", type=float, default=65.0)
    compare_parser.add_argument("--min-tok-step", type=float, default=3.0)
    args = parser.parse_args()
    run(args) if args.command == "run" else compare(args)


if __name__ == "__main__":
    main()
