#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Fail-closed schemas and immutable receipt I/O for the 1M quality gate."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from pathlib import Path
from typing import Any

PROVENANCE_SCHEMA = "qwen38-long-context-provenance-v1"
RECEIPT_SCHEMA = "qwen38-long-context-quality-v1"
HEX64_KEYS = (
    "binary_sha256",
    "build_receipt_sha256",
    "kernel_bundle_sha256",
    "model_config_sha256",
    "model_index_sha256",
    "tokenizer_sha256",
    "effective_environment_sha256",
    "server_receipt_sha256",
    "server_run_nonce",
)
PROVENANCE_KEYS = frozenset(
    {"schema", "source_revision", "model", "runtime", *HEX64_KEYS}
)
REQUIRED_RUNTIME = {
    "runtime_mode": "no-spec",
    "max_seq_len": 1_000_000,
    "rope_yarn_factor": "4",
    "max_batch_size": 1,
    "block_size": 16,
    "kv_cache_dtype": "bf16",
    "prefix_caching": False,
    "high_speed_swap": False,
    "speculative": False,
    "dflash": False,
    "self_speculative": False,
    "ngram_speculative": False,
}
RESPONSE_KEYS = frozenset({"id", "object", "created", "model", "choices", "usage"})
USAGE_KEYS = frozenset(
    {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_tokens_details",
        "completion_tokens_details",
        "time_to_first_token_ms",
        "response_token/s",
    }
)


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("ascii")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _exact_dict(value: object, keys: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        present = set(value) if isinstance(value, dict) else set()
        raise ValueError(
            f"{label} keys differ: missing={sorted(keys - present)!r} "
            f"unknown={sorted(present - keys)!r}"
        )
    return value


def _hex(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(byte in "0123456789abcdef" for byte in value)
    )


def validate_provenance(value: object) -> dict[str, Any]:
    provenance = _exact_dict(value, PROVENANCE_KEYS, "provenance")
    if provenance["schema"] != PROVENANCE_SCHEMA:
        raise ValueError("wrong provenance schema")
    if not _hex(provenance["source_revision"], 40):
        raise ValueError("source_revision must be lowercase SHA-1")
    for key in HEX64_KEYS:
        if not _hex(provenance[key], 64):
            raise ValueError(f"{key} must be lowercase SHA-256")
    if not isinstance(provenance["model"], str) or not provenance["model"]:
        raise ValueError("model must be a non-empty string")
    if provenance["runtime"] != REQUIRED_RUNTIME:
        raise ValueError("runtime must equal the target-only BF16 1M contract")
    return provenance


def validate_tokenize(value: object) -> list[int]:
    response = _exact_dict(value, frozenset({"tokens", "count"}), "tokenize response")
    tokens = response["tokens"]
    if not isinstance(tokens, list) or not tokens:
        raise ValueError("tokenize response must contain tokens")
    if any(type(token) is not int or not 0 <= token <= 0xFFFF_FFFF for token in tokens):
        raise ValueError("token IDs must be u32 integers")
    if type(response["count"]) is not int or response["count"] != len(tokens):
        raise ValueError("tokenize count differs from token vector")
    return tokens


def expected_answer(codes: dict[str, str]) -> str:
    if set(codes) != {"early", "middle", "late"}:
        raise ValueError("needle codes differ")
    if any(not code.isascii() or not code.isalnum() for code in codes.values()):
        raise ValueError("needle codes must be ASCII alphanumeric")
    return f"EARLY={codes['early']} MIDDLE={codes['middle']} LATE={codes['late']}"


def _finite_positive(value: object, label: str) -> float:
    if type(value) not in {int, float} or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return float(value)


def validate_completion(
    value: object, target: int, codes: dict[str, str]
) -> dict[str, Any]:
    response = _exact_dict(value, RESPONSE_KEYS, "completion response")
    if response["object"] != "text_completion" or not isinstance(response["id"], str):
        raise ValueError("wrong completion identity")
    if type(response["created"]) is not int or not isinstance(response["model"], str):
        raise ValueError("invalid completion metadata")
    choices = response["choices"]
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("completion must contain exactly one choice")
    choice = _exact_dict(
        choices[0], frozenset({"index", "text", "finish_reason"}), "completion choice"
    )
    answer = expected_answer(codes)
    if choice != {"index": 0, "text": answer, "finish_reason": "stop"}:
        raise ValueError("completion did not return the exact three-needle answer")
    usage = _exact_dict(response["usage"], USAGE_KEYS, "completion usage")
    if type(target) is not int or target <= 0 or usage["prompt_tokens"] != target:
        raise ValueError("prompt token accounting differs from target")
    completion = usage["completion_tokens"]
    if type(completion) is not int or not 1 <= completion <= 64:
        raise ValueError("completion token accounting is out of bounds")
    if usage["total_tokens"] != target + completion:
        raise ValueError("total token accounting differs")
    prompt_details = _exact_dict(
        usage["prompt_tokens_details"],
        frozenset({"cached_tokens", "audio_tokens"}),
        "prompt token details",
    )
    if prompt_details != {"cached_tokens": 0, "audio_tokens": 0}:
        raise ValueError("quality request must be uncached and text-only")
    completion_details = _exact_dict(
        usage["completion_tokens_details"],
        frozenset(
            {
                "reasoning_tokens",
                "audio_tokens",
                "accepted_prediction_tokens",
                "rejected_prediction_tokens",
            }
        ),
        "completion token details",
    )
    if any(type(item) is not int or item != 0 for item in completion_details.values()):
        raise ValueError(
            "quality request must have no reasoning/speculative/audio tokens"
        )
    ttft = _finite_positive(usage["time_to_first_token_ms"], "TTFT")
    decode_tps = _finite_positive(usage["response_token/s"], "decode throughput")
    return {
        "prompt_tokens": target,
        "completion_tokens": completion,
        "cached_tokens": 0,
        "finish_reason": "stop",
        "stable_output_sha256": sha256(answer.encode("ascii")),
        "time_to_first_token_ms": ttft,
        "effective_prefill_tokens_per_second": target * 1000.0 / ttft,
        "response_tokens_per_second": decode_tps,
    }


def _directory_fd(parent: Path) -> int:
    absolute = parent.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if stat.S_ISLNK(os.lstat(current).st_mode):
            raise ValueError(f"receipt directory component is a symlink: {current}")
    return os.open(absolute, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)


def seal_receipt(path: Path, value: object) -> str:
    payload = canonical_bytes(value)
    directory_fd = _directory_fd(path.parent)
    descriptor = -1
    try:
        descriptor = os.open(
            path.name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        view = memoryview(payload)
        while view:
            view = view[os.write(descriptor, view) :]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        held = os.fstat(descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(directory_fd)
    final = os.lstat(path)
    if (
        not stat.S_ISREG(final.st_mode)
        or final.st_nlink != 1
        or final.st_mode & 0o222
        or final.st_size != len(payload)
        or (held.st_dev, held.st_ino) != (final.st_dev, final.st_ino)
    ):
        raise RuntimeError("sealed receipt identity or mode changed")
    return sha256(payload)
