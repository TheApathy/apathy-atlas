#!/usr/bin/env python3
"""Two-phase, single-GPU DeepSeek DFlash correctness and speed gate."""

import argparse
import hashlib
import json
import math
import os
import pathlib
import re
import statistics
import sys
import tempfile
import time
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
import benchmark_receipt  # noqa: E402

PROMPTS = {
    "code": "Write a complete Python LRU cache implementation with tests.",
    "math": "Derive the quadratic formula and verify the result algebraically.",
    "prose": "Write a detailed story about a lighthouse keeper during a storm.",
    "json": "Return a JSON array of 100 objects with id, name, category, and score.",
}
ACCEPT_RE = re.compile(
    r"DSPARK accept: ([0-9.]+) tok/step over ([0-9]+) steps .*?draft accept ([0-9.]+)%"
)


def is_sha256(value) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def load_receipt_binding(path: pathlib.Path, url: str, model: str) -> dict:
    envelope = benchmark_receipt.read_envelope(path)
    state = benchmark_receipt.receipt_state(envelope, url)
    if state != "ACTIVE_VERIFIED":
        raise RuntimeError("benchmark receipt is not ACTIVE_VERIFIED")
    activation = envelope["activation"]
    if activation["model_id"] != model:
        raise RuntimeError("benchmark receipt model does not match request model")
    return {
        "model_identity": hashlib.sha256(
            benchmark_receipt.canonical_json(envelope["manifest"]["model"])
        ).hexdigest(),
        "implementation_identity": envelope["manifest_sha256"],
        "benchmark_receipt_sha256": envelope["activation_sha256"],
        "host_boot_id": activation["host_boot_id"],
        "gpu_identity": activation["gpu_identity"],
        "benchmark_receipt": envelope,
    }


def parse_accept_log(
    path: pathlib.Path | None,
    start_offset: int = 0,
    expected_identity: tuple[int, int] | None = None,
    expected_prefix_sha256: str | None = None,
):
    if path is None:
        return None
    stat = path.stat()
    identity = (stat.st_dev, stat.st_ino)
    if expected_identity is not None and identity != expected_identity:
        raise RuntimeError(f"server log was replaced during benchmark: {path}")
    if stat.st_size < start_offset:
        raise RuntimeError(f"server log shrank during benchmark: {path}")
    with path.open("rb") as handle:
        if expected_prefix_sha256 is not None:
            prefix = handle.read(start_offset)
            if hashlib.sha256(prefix).hexdigest() != expected_prefix_sha256:
                raise RuntimeError(
                    f"server log prefix changed during benchmark: {path}"
                )
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


def write_json_atomic(
    path: pathlib.Path, payload: dict, *, overwrite: bool = False
) -> None:
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
        handle.flush()
        os.fsync(handle.fileno())
    try:
        if overwrite:
            temporary.replace(path)
        else:
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                raise RuntimeError(f"output already exists: {path}") from error
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
    finish_reason = None
    saw_done = False
    with benchmark_receipt.urlopen_no_redirect(request, timeout=1800) as response:
        for raw in response:
            if not raw.startswith(b"data: "):
                continue
            payload = raw[6:].strip()
            if payload == b"[DONE]":
                saw_done = True
                break
            event = json.loads(payload)
            usage = event.get("usage") or usage
            choices = event.get("choices") or []
            event_finish_reason = (
                choices[0].get("finish_reason")
                if choices and isinstance(choices[0], dict)
                else None
            )
            if event_finish_reason is not None:
                if event_finish_reason not in {"stop", "length"}:
                    raise RuntimeError(
                        f"unsupported finish_reason: {event_finish_reason!r}"
                    )
                if finish_reason is not None:
                    raise RuntimeError("stream emitted more than one finish_reason")
                finish_reason = event_finish_reason
            delta = choices[0].get("delta", {}) if choices else {}
            reasoning_piece = delta.get("reasoning_content") or delta.get("reasoning")
            content_piece = delta.get("content")
            if finish_reason is not None and (reasoning_piece or content_piece):
                raise RuntimeError("stream emitted content after finish_reason")
            if reasoning_piece or content_piece:
                now = time.perf_counter()
                first = first or now
                last = now
                if reasoning_piece:
                    reasoning.append(reasoning_piece)
                if content_piece:
                    content.append(content_piece)
    ended = time.perf_counter()
    if not saw_done:
        raise RuntimeError("stream ended without [DONE]")
    if finish_reason is None:
        raise RuntimeError("stream ended without finish_reason")
    reasoning_text = "".join(reasoning)
    content_text = "".join(content)
    transcript = json.dumps(
        {"reasoning": reasoning_text, "content": content_text},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    tokens = usage.get("completion_tokens")
    if type(tokens) is not int or tokens <= 0:
        raise RuntimeError("completion_tokens must be a positive integer")
    decode_seconds = max(0.0, (last or ended) - (first or ended))
    return {
        "completion_tokens": tokens,
        "finish_reason": finish_reason,
        "truncated": finish_reason == "length",
        "ttft_seconds": (first or ended) - started,
        "decode_seconds": decode_seconds,
        "decode_tok_s": (
            (tokens - 1) / decode_seconds if tokens > 1 and decode_seconds else 0.0
        ),
        "reasoning_sha256": hashlib.sha256(reasoning_text.encode()).hexdigest(),
        "content_sha256": hashlib.sha256(content_text.encode()).hexdigest(),
        "output_sha256": hashlib.sha256(transcript.encode()).hexdigest(),
    }


def run(args) -> None:
    if args.max_tokens <= 1:
        raise RuntimeError("max-tokens must be greater than one for decode timing")
    if args.reps <= 0:
        raise RuntimeError("reps must be positive")
    if args.output.exists() and not args.overwrite:
        raise RuntimeError(f"refusing to overwrite existing output: {args.output}")
    if not args.label.strip():
        raise RuntimeError("label must be non-empty")
    receipt_binding = load_receipt_binding(args.receipt, args.url, args.model)
    accept_stat = args.server_log.stat() if args.server_log else None
    accept_offset = accept_stat.st_size if accept_stat else 0
    accept_identity = (accept_stat.st_dev, accept_stat.st_ino) if accept_stat else None
    accept_prefix_sha256 = None
    if args.server_log:
        with args.server_log.open("rb") as handle:
            accept_prefix_sha256 = hashlib.sha256(handle.read()).hexdigest()
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
    acceptance = parse_accept_log(
        args.server_log,
        accept_offset,
        accept_identity,
        accept_prefix_sha256,
    )
    final_receipt_binding = load_receipt_binding(args.receipt, args.url, args.model)
    if final_receipt_binding != receipt_binding:
        raise RuntimeError("benchmark receipt binding changed during measurement")
    receipt_binding = final_receipt_binding
    result = {
        "label": args.label,
        **receipt_binding,
        "contract": {
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "reps": args.reps,
        },
        "aggregate_decode_tok_s": decoded_intervals / decode_seconds,
        "median_decode_tok_s": statistics.median(
            item["decode_tok_s"] for item in all_runs
        ),
        "acceptance": acceptance,
        "cases": cases,
    }
    write_json_atomic(args.output, result, overwrite=args.overwrite)
    print(
        f"{args.label}: median {result['median_decode_tok_s']:.2f} tok/s -> "
        f"{args.output}"
    )


def _compare_results(
    baseline: dict, candidate: dict, min_tok_s: float, min_tok_step: float
) -> dict:
    failures = []
    for label, record in (("baseline", baseline), ("candidate", candidate)):
        envelope = record.get("benchmark_receipt")
        try:
            if not benchmark_receipt.verify_envelope(envelope):
                raise ValueError("digest")
            benchmark_receipt.validate_manifest(envelope["manifest"])
            benchmark_receipt.validate_activation(envelope)
            if "activation" not in envelope:
                raise ValueError("planned")
            activation = envelope["activation"]
            derived = {
                "model_identity": hashlib.sha256(
                    benchmark_receipt.canonical_json(envelope["manifest"]["model"])
                ).hexdigest(),
                "implementation_identity": envelope["manifest_sha256"],
                "benchmark_receipt_sha256": envelope["activation_sha256"],
                "host_boot_id": activation["host_boot_id"],
                "gpu_identity": activation["gpu_identity"],
            }
            for field, value in derived.items():
                if record.get(field) != value:
                    failures.append(f"{label} {field} disagrees with embedded receipt")
        except (KeyError, TypeError, ValueError):
            failures.append(f"{label} embedded benchmark receipt is malformed")
        for field in (
            "model_identity",
            "implementation_identity",
            "benchmark_receipt_sha256",
        ):
            if not is_sha256(record.get(field)):
                failures.append(f"{label} {field} is malformed")
        try:
            benchmark_receipt.validate_boot_id(record.get("host_boot_id"))
        except ValueError:
            failures.append(f"{label} host_boot_id is malformed")
        gpu_identity = record.get("gpu_identity")
        if (
            type(gpu_identity) is not dict
            or set(gpu_identity) != {"uuid", "name", "driver_version"}
            or any(
                type(value) is not str or not value for value in gpu_identity.values()
            )
        ):
            failures.append(f"{label} gpu_identity is malformed")
        for field in ("median_decode_tok_s", "aggregate_decode_tok_s"):
            value = record.get(field)
            if (
                type(value) not in (int, float)
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0
            ):
                failures.append(f"{label} {field} must be a finite positive number")
    if not baseline.get("model_identity") or not candidate.get("model_identity"):
        failures.append("both records must declare model_identity")
    elif baseline["model_identity"] != candidate["model_identity"]:
        failures.append("model identities differ")
    if not baseline.get("implementation_identity"):
        failures.append("baseline has no implementation_identity")
    if not candidate.get("implementation_identity"):
        failures.append("candidate has no implementation_identity")
    elif (
        baseline.get("implementation_identity") == candidate["implementation_identity"]
    ):
        failures.append(
            "baseline and candidate implementation identities are identical"
        )
    if baseline.get("host_boot_id") != candidate.get("host_boot_id"):
        failures.append("host_boot_id differs")
    if baseline.get("gpu_identity") != candidate.get("gpu_identity"):
        failures.append("gpu_identity differs")
    # This establishes one Linux boot cohort only. Separate server/model loads can
    # still choose different autotune results and require repeated paired trials.
    if baseline["contract"] != candidate["contract"]:
        failures.append("benchmark contracts differ")
    for name in PROMPTS:
        left = baseline["cases"][name]
        right = candidate["cases"][name]
        for label, case in (("baseline", left), ("candidate", right)):
            if not is_sha256(case.get("prompt_sha256")):
                failures.append(f"{name}: {label} prompt_sha256 is malformed")
        if left["prompt_sha256"] != right["prompt_sha256"]:
            failures.append(f"{name}: prompt hash differs")
            continue
        for label, case in (("baseline", left), ("candidate", right)):
            for run_index, run in enumerate(case["runs"]):
                if not is_sha256(run.get("output_sha256")):
                    failures.append(
                        f"{name} run {run_index}: {label} output_sha256 is malformed"
                    )
        left_hashes = [run["output_sha256"] for run in left["runs"]]
        right_hashes = [run["output_sha256"] for run in right["runs"]]
        if left_hashes != right_hashes:
            failures.append(f"{name}: output hashes differ")
        if len(left["runs"]) != len(right["runs"]):
            failures.append(f"{name}: run counts differ")
            continue
        for run_index, (left_run, right_run) in enumerate(
            zip(left["runs"], right["runs"])
        ):
            for record in (left_run, right_run):
                run_speed = record.get("decode_tok_s")
                if (
                    type(run_speed) not in (int, float)
                    or isinstance(run_speed, bool)
                    or not math.isfinite(run_speed)
                    or run_speed < 0
                ):
                    failures.append(
                        f"{name} run {run_index}: decode_tok_s must be finite"
                    )
                    break
            left_finish = left_run.get("finish_reason")
            right_finish = right_run.get("finish_reason")
            if left_finish not in {"stop", "length"} or right_finish not in {
                "stop",
                "length",
            }:
                failures.append(f"{name} run {run_index}: finish_reason is malformed")
            elif left_finish != right_finish:
                failures.append(f"{name} run {run_index}: finish_reason differs")
            for record in (left_run, right_run):
                truncated = record.get("truncated")
                if type(truncated) is not bool or truncated != (
                    record.get("finish_reason") == "length"
                ):
                    failures.append(
                        f"{name} run {run_index}: truncated marker is malformed"
                    )
                    break
            left_tokens = left_run.get("completion_tokens")
            right_tokens = right_run.get("completion_tokens")
            if (
                type(left_tokens) is not int
                or left_tokens <= 0
                or type(right_tokens) is not int
                or right_tokens <= 0
            ):
                failures.append(
                    f"{name} run {run_index}: completion_tokens is malformed"
                )
            elif left_tokens != right_tokens:
                failures.append(f"{name} run {run_index}: completion_tokens differs")
    speed = candidate["median_decode_tok_s"]
    if speed < min_tok_s:
        failures.append(f"decode {speed:.2f} tok/s is below {min_tok_s:.2f}")
    acceptance = candidate.get("acceptance")
    if min_tok_step:
        if type(acceptance) is not dict:
            failures.append("candidate has no acceptance summary")
        else:
            committed = acceptance.get("committed_tokens_per_step")
            if (
                type(committed) not in (int, float)
                or isinstance(committed, bool)
                or not math.isfinite(committed)
            ):
                failures.append("candidate acceptance must be a finite number")
            elif committed < min_tok_step:
                failures.append(
                    f"acceptance {committed:.2f} tok/step is below {min_tok_step:.2f}"
                )
    return {
        "status": "pass" if not failures else "fail",
        "failures": failures,
        "candidate_median_tok_s": speed,
        "acceptance": acceptance,
    }


def compare_results(
    baseline: dict, candidate: dict, min_tok_s: float, min_tok_step: float
) -> dict:
    try:
        if type(baseline) is not dict or type(candidate) is not dict:
            raise TypeError("records")
        return _compare_results(baseline, candidate, min_tok_s, min_tok_step)
    except (KeyError, TypeError, ValueError, IndexError) as error:
        return {
            "status": "fail",
            "failures": [f"malformed benchmark result: {type(error).__name__}"],
            "candidate_median_tok_s": None,
            "acceptance": None,
        }


def compare(args) -> None:
    if (
        not math.isfinite(args.min_tok_s)
        or not math.isfinite(args.min_tok_step)
        or args.min_tok_s <= 0
        or args.min_tok_step < 0
    ):
        raise RuntimeError(
            "thresholds must be finite; min-tok-s must be positive and "
            "min-tok-step non-negative"
        )
    try:

        def reject_constant(value):
            raise ValueError(f"non-finite JSON number: {value}")

        baseline = json.loads(args.baseline.read_text(), parse_constant=reject_constant)
        candidate = json.loads(
            args.candidate.read_text(), parse_constant=reject_constant
        )
        report = compare_results(baseline, candidate, args.min_tok_s, args.min_tok_step)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        report = {
            "status": "fail",
            "failures": [f"malformed benchmark artifact: {type(error).__name__}"],
            "candidate_median_tok_s": None,
            "acceptance": None,
        }
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
    run_parser.add_argument("--receipt", type=pathlib.Path, required=True)
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
