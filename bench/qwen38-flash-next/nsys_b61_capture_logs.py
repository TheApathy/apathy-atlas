# SPDX-License-Identifier: AGPL-3.0-only
"""Exact b61 server-route and Nsight-control log census."""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request
from pathlib import Path
from typing import Any

import nsys_b61_capture_support as support
import nsys_b61_capture_control as control_support

REQUIRED_SERVER = {
    "ple_discovery": (r"Qwen4 PLE: discovered sparse offload manifest", 1),
    "prefix_disabled": (r"Prefix caching: disabled", 1),
    "loader": (
        r"Qwen3\.5 weight loader: 48 layers \(12 attention, 36 linear_attn\)",
        1,
    ),
    "mtp_disabled": (r"No MTP weights found . speculative decoding disabled", 1),
    "marconi_zero": (r"SSM snapshot pool: Marconi 0 slots", 1),
    "scheduler": (
        r"Scheduler started .*max_batch=1, mtp=false, ngram=false, num_drafts=0, "
        r"policy=fifo, chunked_prefill=true, max_prefill_tokens=512",
        1,
    ),
    "listening": (r"Listening on 127\.0\.0\.1:8998", 1),
    "request_64": (r"Request: .*temp=Some\(0\.0\), max_tokens=64(?:\D|$)", 1),
    "request_400": (r"Request: .*temp=Some\(0\.0\), max_tokens=400(?:\D|$)", 1),
    "session": (r"Session .*: 38 prompt tokens, tools=false \(0 defined\)", 2),
    "chunk_64": (
        r"Chunked prefill start: 38 prompt tokens, .*max_tokens=64(?:\D|$)",
        1,
    ),
    "chunk_400": (
        r"Chunked prefill start: 38 prompt tokens, .*max_tokens=400(?:\D|$)",
        1,
    ),
    "first_token": (r"Prefill first token: 71093", 2),
    "prefilled_64": (
        r"Prefilled \(single chunk\): seq_len=38, remaining=63(?:\D|$)",
        1,
    ),
    "prefilled_400": (
        r"Prefilled \(single chunk\): seq_len=38, remaining=399(?:\D|$)",
        1,
    ),
    "segmented_graph": (r"Qwen4 PLE segmented graph captured for slot=0", 2),
    "done_64": (r"Done: 64 tokens \(length\)", 1),
    "done_400": (r"Done: 400 tokens \(length\)", 1),
}
FORBIDDEN_ROUTE = re.compile(
    r"Speculative decoding: ENABLED|DFLASH|DFlash propose|spec-policy|K=.?\d+|"
    r"ENGAGED ATLAS_|prefix cache hit|QSA.*(?:enabled|engaged)",
    re.IGNORECASE,
)
KNOWN_FALLBACK = "Skipping MoE weight transposition"
CONTROL_TIMEOUT_SECONDS = {
    "start": 60,
    "stop": 60,
    "cancel": 60,
    "shutdown": 60,
    "cuda_gpu_trace": 300,
    "cuda_gpu_kern_gb_sum": 300,
}


def run_control(
    label: str,
    command: list[str],
    log: Any,
    events: list[dict[str, Any]],
    *,
    check: bool = True,
) -> None:
    marker = {"argv": command, "label": label}
    log.write(b"CONTROL " + support.canonical_bytes(marker) + b"\n")
    result = control_support.run_bounded(
        command,
        cwd=support.REPO,
        env=support.TARGET_ENV,
        output=log,
        timeout_seconds=CONTROL_TIMEOUT_SECONDS[label],
    )
    event = {**marker, **result}
    event["argv_sha256"] = hashlib.sha256(support.canonical_bytes(command)).hexdigest()
    log.write(b"\nCONTROL_RESULT " + support.canonical_bytes(event) + b"\n")
    log.flush()
    os.fsync(log.fileno())
    events.append(event)
    if result["timed_out"]:
        raise RuntimeError(f"Nsight {label} timed out and was reaped")
    if check and result["returncode"] != 0:
        raise RuntimeError(f"Nsight {label} failed rc={result['returncode']}")


def freeze(path: Path) -> dict[str, Any]:
    info = support.require_regular(path)
    path.chmod(0o444)
    return {"sha256": support.sha_file(path), "size": info.st_size}


def write_json(path: Path, receipt: dict[str, Any]) -> None:
    with path.open("xb") as stream:
        stream.write(support.canonical_bytes(receipt) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)


def http_json(
    url: str, body: dict[str, Any] | None = None
) -> tuple[dict[str, Any], str]:
    raw = None if body is None else support.canonical_bytes(body)
    headers = {} if raw is None else {"Content-Type": "application/json"}
    with urllib.request.urlopen(
        urllib.request.Request(url, raw, headers), timeout=1200
    ) as response:
        payload = response.read()
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def validate_response(response: dict[str, Any], max_tokens: int) -> dict[str, Any]:
    choices = response["choices"]
    if not isinstance(choices, list) or len(choices) != 1:
        raise RuntimeError("response must have one choice")
    choice = choices[0]
    message = choice["message"]
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    usage = response["usage"]
    stable = {
        "content": content,
        "reasoning_content": reasoning,
        "finish_reason": choice.get("finish_reason"),
    }
    got = {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "finish_reason": choice.get("finish_reason"),
        "content_bytes": len(content.encode()),
        "content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "reasoning_bytes": len(reasoning.encode()),
        "reasoning_sha256": hashlib.sha256(reasoning.encode()).hexdigest(),
        "stable_output_sha256": hashlib.sha256(
            support.canonical_bytes(stable)
        ).hexdigest(),
    }
    expected = {
        64: (
            38,
            64,
            "length",
            273,
            "724e2dcaffefa24aa4cd8630cbf0aae27776822ee6177b05c7270ec23ff8a04d",
            "723248e0bde86ae19be445ff895f52c5bb9f7af3c6e63b23ef61a296a58efdda",
        ),
        400: (
            38,
            400,
            "length",
            1490,
            "fce20d91ead45078417ef0aecef08e7d2b749e3d3f88cc77efebf824e11b90b6",
            "f811cdc565dff063074f8bb1d0bb3fd55c8b10f3438c517cf4d59128f23cf790",
        ),
    }[max_tokens]
    actual = tuple(
        got[key]
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "finish_reason",
            "content_bytes",
            "content_sha256",
            "stable_output_sha256",
        )
    )
    if actual != expected or got["reasoning_bytes"] != 0:
        raise RuntimeError(f"semantic oracle mismatch for max_tokens={max_tokens}")
    if got["reasoning_sha256"] != support.EMPTY_SHA:
        raise RuntimeError(f"semantic oracle mismatch for max_tokens={max_tokens}")
    return got


def validate_server_log(path: Path) -> dict[str, int]:
    text = path.read_text(errors="strict")
    census = {
        name: len(re.findall(pattern, text))
        for name, (pattern, _) in REQUIRED_SERVER.items()
    }
    wrong = {
        name: {"actual": census[name], "expected": expected}
        for name, (_, expected) in REQUIRED_SERVER.items()
        if census[name] != expected
    }
    if wrong:
        raise RuntimeError(f"server route census mismatch: {wrong}")
    ple_lines = [
        line
        for line in text.splitlines()
        if "Qwen4 PLE sparse NVFP4 offload enabled" in line
    ]
    if (
        len(ple_lines) != 1
        or "cache_mb=0" not in ple_lines[0]
        or "io_mode=Direct" not in ple_lines[0]
    ):
        raise RuntimeError("PLE direct/cache-zero route census mismatch")
    suspicious = []
    known_fallbacks = 0
    for line in text.splitlines():
        if re.search(r"WARN|ERROR|FATAL|panic|fallback", line, re.IGNORECASE):
            if (
                KNOWN_FALLBACK in line
                and "Prefill will use fallback grouped GEMM" in line
            ):
                known_fallbacks += 1
            else:
                suspicious.append(line[:240])
    if known_fallbacks != 1 or suspicious:
        raise RuntimeError(
            f"unexpected server warning/error/fallback: known={known_fallbacks} other={suspicious}"
        )
    forbidden = FORBIDDEN_ROUTE.findall(text)
    if forbidden:
        raise RuntimeError(f"unexpected selector/route engagement: {forbidden[:4]}")
    if len(re.findall(r"Request:", text)) != 2 or len(re.findall(r"Done:", text)) != 2:
        raise RuntimeError("request/completion log cardinality mismatch")
    census["known_prefill_fallback"] = known_fallbacks
    census["ple_direct_cache_zero"] = 1
    return census


def validate_control_log(path: Path, events: list[dict[str, Any]]) -> dict[str, Any]:
    expected = ["start", "stop", "cuda_gpu_trace", "cuda_gpu_kern_gb_sum", "shutdown"]
    labels = [event.get("label") for event in events]
    if labels != expected or any(event.get("returncode") != 0 for event in events):
        raise RuntimeError(f"Nsight control sequence/status mismatch: {labels}")
    text = path.read_text(errors="strict")
    for event in events:
        if text.count(f'"label":"{event["label"]}"') != 2:
            raise RuntimeError("Nsight control log marker cardinality mismatch")
    if re.search(r"\b(?:ERROR|FATAL|panic|failed)\b", text, re.IGNORECASE):
        raise RuntimeError("Nsight control log reports an error")
    return {"event_labels": labels, "event_count": len(events)}
