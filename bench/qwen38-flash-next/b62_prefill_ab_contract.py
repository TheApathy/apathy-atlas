# SPDX-License-Identifier: AGPL-3.0-only
"""Immutable workload and arm contract for the sealed b62 prefill A/B."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BINARY = Path("/var/tmp/atlas-flash-b62-build-D6fncRTX/release/spark")
BINARY_SHA256 = "77fe8bc1ad3ed90c3b3fb4db5323e55d36c5fedc5b76852f36a445d25193d5bd"
BINARY_SIZE = 38_569_952
BUILD_ID = "5d9d0cc7cf59c20cd2f12af01e6221a332698645"
MODEL = Path("/home/flocka/models/Qwen3.8-Flash-Next-NVFP4-Offload")
MODEL_MANIFEST = Path("/var/tmp/atlas-b62-qwen38-flash-next-model-manifest.json")
# Deliberately non-hex until root can capture and independently review the exact 197 shards.
MODEL_MANIFEST_SHA256 = "UNPINNED_REQUIRES_FOREIGN_STORAGE_LANE_RELEASE"
MODEL_CONTENT_ROOT_SHA256 = "UNPINNED_REQUIRES_FOREIGN_STORAGE_LANE_RELEASE"
MODEL_NAME = "qwen3.8-flash-next"
PORT = 8998
ENDPOINT = f"http://127.0.0.1:{PORT}/v1/chat/completions"
EXECUTE = "EXECUTE_SEALED_B62_PREFILL_AB_WITH_ACTIVE_GPU_RESERVATION"

BUILD_FILES = {
    "BUILD_RECEIPT.md": "c40ad52e50c6af734e6ff0c573d2ca34db80da9e8525a8dedf17b41bc793547a",
    "artifact-manifest.sha256": "2e7b25500a16d98087afff633d72e0aaa19b7a5cfc501c75db9fd0a4be7a0b0a",
    "ptx-manifest.sha256": "cbe5fd1ad856ae41a41a46f03870aa1e66563f38d9cf377152b5e40957281253",
    "selected-source-manifest.sha256": "997218b8bce625a57bc2bfd52d384d54abd070d2ace0e8d85d079fe3b1e54ef7",
    "source-manifest.sha256": "703d2d4c651c1ae5c3a868f172f530000fd4c14050b569c74572b893c600b504",
}
MANIFEST_LINES = {
    "ptx-manifest.sha256": 154,
    "selected-source-manifest.sha256": 25,
    "source-manifest.sha256": 1_839,
}
ARM_SELECTORS = {
    "control": ("0", "0"),
    "attention": ("1", "0"),
    "ssm": ("0", "1"),
}
ARM_ORDER = tuple(ARM_SELECTORS)

WESCHERA_PROMPT = (
    "Write a complete MinHeap class in Python with insert, extract_min, and heapify, "
    "then explain the complexity of each method."
)
M2013_ANSWER = "ORCHID-7391, COBALT-2846, SAFFRON-9153"
M2013_SEMANTIC_SHA256 = (
    "5664095aa5014d64566be16ee7492601c1a62030cb14c091d8f2a50a68241432"
)
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


@dataclass(frozen=True)
class RequestSpec:
    name: str
    body: dict[str, Any]
    wire: bytes
    wire_sha256: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    content_bytes: int
    content_sha256: str
    stable_sha256: str


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _wire(body: dict[str, Any]) -> bytes:
    return json.dumps(body, ensure_ascii=True, allow_nan=False).encode()


def _m2013_prompt() -> str:
    records = []
    for index in range(1, 124):
        if index == 10:
            records.append("CHECKPOINT_ALPHA is ORCHID-7391.")
        elif index == 61:
            records.append("CHECKPOINT_BETA is COBALT-2846.")
        elif index == 118:
            records.append("CHECKPOINT_GAMMA is SAFFRON-9153.")
        else:
            records.append(
                f"Record {index}: the archive entry is ordinary and contains no checkpoint value."
            )
    return " ".join(records) + (
        " Return only CHECKPOINT_ALPHA, CHECKPOINT_BETA, and CHECKPOINT_GAMMA "
        "in that order, separated by commas."
    )


def request_specs() -> dict[str, RequestSpec]:
    canary_body = {
        "model": MODEL_NAME,
        "temperature": 0.0,
        "max_tokens": 400,
        "messages": [{"role": "user", "content": WESCHERA_PROMPT}],
        "reasoning_effort": "none",
    }
    m2013_body = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": _m2013_prompt()}],
        "reasoning_effort": "none",
        "temperature": 0.0,
        "max_tokens": 32,
        "stream": False,
    }
    canary_wire, m2013_wire = _wire(canary_body), _wire(m2013_body)
    checks = {
        "canary": (
            len(canary_wire),
            sha256(canary_wire),
            268,
            "cd22f3df9cef8042f6b30dfa82a76863dd1f42732f1b29736693c719e621fa93",
        ),
        "m2013": (
            len(m2013_wire),
            sha256(m2013_wire),
            9_377,
            "077c25c42b9c911affc71a6660159d4e13c2708c8d636938cd84f14b7d439095",
        ),
    }
    if any(
        actual != expected
        for actual, digest, expected, expected_digest in checks.values()
    ) or any(
        digest != expected_digest
        for actual, digest, expected, expected_digest in checks.values()
    ):
        raise RuntimeError("hard-pinned request wire drift")
    if (
        sha256(_m2013_prompt().encode())
        != "faf31fc16fa554c0ef82fa9c59cb15f007d35007a903f3617e9c665908e2e72f"
    ):
        raise RuntimeError("M2013 prompt drift")
    return {
        "canary": RequestSpec(
            "canary",
            canary_body,
            canary_wire,
            checks["canary"][1],
            38,
            400,
            "length",
            1_490,
            "fce20d91ead45078417ef0aecef08e7d2b749e3d3f88cc77efebf824e11b90b6",
            "f811cdc565dff063074f8bb1d0bb3fd55c8b10f3438c517cf4d59128f23cf790",
        ),
        "m2013": RequestSpec(
            "m2013",
            m2013_body,
            m2013_wire,
            checks["m2013"][1],
            2_013,
            27,
            "stop",
            len(M2013_ANSWER.encode()),
            sha256(M2013_ANSWER.encode()),
            "1dac947db1eb23cab6c428f8c53c10eb01333a5546557c869e37811497e462c9",
        ),
    }


def percentile(values: list[float], fraction: float) -> float:
    if not values or not 0.0 <= fraction <= 1.0:
        raise ValueError("invalid percentile input")
    if any(type(value) is not float or not math.isfinite(value) for value in values):
        raise ValueError("percentile values must be finite floats")
    ordered = sorted(values)
    rank = max(1, math.ceil(len(ordered) * fraction))
    return ordered[rank - 1]
