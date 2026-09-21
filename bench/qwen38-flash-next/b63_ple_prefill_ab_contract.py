# SPDX-License-Identifier: AGPL-3.0-only
"""Immutable workload and sole-delta arm contract for sealed b63 PLE A/B."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BINARY = Path("/var/tmp/atlas-flash-b63-ple-build-XOX6w3vJ/release/spark")
BINARY_SHA256 = "23db269e1fdefaaa54885df6dff9a8297edac67521332dec5fcda076b0cfc644"
BINARY_SIZE = 38_862_088
BUILD_ID = "e048074990ec9aa6ebd1faf5511553e44f3b793a"
MODEL = Path("/home/flocka/models/Qwen3.8-Flash-Next-NVFP4-Offload")
MODEL_MANIFEST = Path("/var/tmp/atlas-b62-qwen38-flash-next-model-manifest.json")
# Deliberately non-hex until root can capture and independently review the exact 197 shards.
MODEL_MANIFEST_SHA256 = "UNPINNED_REQUIRES_FOREIGN_STORAGE_LANE_RELEASE"
MODEL_CONTENT_ROOT_SHA256 = "UNPINNED_REQUIRES_FOREIGN_STORAGE_LANE_RELEASE"
RAW_PARITY_RECEIPT = Path("/var/tmp/atlas-b63-ple-raw-parity-qualification.json")
RAW_PARITY_RECEIPT_SHA256 = "UNPINNED_REQUIRES_RAW_PARITY_AND_CLEAN_REBUILD"
MODEL_NAME = "qwen3.8-flash-next"
PORT = 8998
ENDPOINT = f"http://127.0.0.1:{PORT}/v1/chat/completions"
EXECUTE = "EXECUTE_SEALED_B63_WHOLE_PROMPT_PLE_AB_WITH_ACTIVE_GPU_RESERVATION"

BUILD_FILES = {
    "BUILD_RECEIPT.md": "b467e45ada25eaad68a5e33114806c9e4e13d101fb2fed1b340e287c2f030ff4",
    "artifact-manifest.sha256": "a18ad17da63a609ee22b5722827f967bf55af87d04038088655772b146468c8b",
    "artifact-manifest-check.log": "b1654149a47f5c7d7df48151a384b45b4ddebdd373e8a393231c9dfe6c69d777",
    "ptx-manifest.sha256": "ffae0a181188459858705eb640efe3eb1c473af4c74b9c443b2d320e038448d7",
    "selected-source-manifest.sha256": "bc098d6fbb54a56cd9361ded5a9c6a358d19c1c2765d74fb266257590bc9bc73",
    "source-manifest.sha256": "426ad39862d1cf6a64c1186c33a05da1228b25381f17563977f1cfd166faab08",
}
MANIFEST_LINES = {
    "artifact-manifest.sha256": 18,
    "artifact-manifest-check.log": 18,
    "ptx-manifest.sha256": 154,
    "selected-source-manifest.sha256": 25,
    "source-manifest.sha256": 1_840,
}
ARM_SELECTORS = {
    "control": "0",
    "candidate": "1",
}
ARM_ORDER = tuple(ARM_SELECTORS)

ROUTE_ENV = {
    "ATLAS_DFLASH_ASYNC": "0",
    "ATLAS_DFLASH_MARKOV": "0",
    "ATLAS_DFLASH_SPEC_CYCLE_V2": "0",
    "ATLAS_GDN_PREFILL_TUNED": "0",
    "ATLAS_MOE_EXACT_PREFILL_GRID": "0",
    "ATLAS_MULTISEQ_GRAPHS": "0",
    "ATLAS_NVFP4_MOE_WORKLIST": "0",
    "ATLAS_PLE_CACHE_MB": "0",
    "ATLAS_PLE_PAGE_CACHE": "0",
    "ATLAS_PREFILL_FFN_FAST": "0",
    "ATLAS_PREFILL_FFN_PIPE": "0",
    "ATLAS_PREFILL_PHASE_PROFILE": "0",
    "ATLAS_PREFILL_PROJ_FAST": "0",
    "ATLAS_PREFILL_PROJ_PIPE": "0",
    "ATLAS_QWEN4_ATTN_PREFILL_BATCH": "0",
    "ATLAS_QWEN4_HYPER_PREFILL_GEMM": "0",
    "ATLAS_QWEN4_PLE_SEGMENTED_GRAPHS": "0",
    "ATLAS_QWEN4_QSA_PREFILL_GEMM": "0",
    "ATLAS_QWEN4_SSM_PREFILL_BATCH": "0",
    "ATLAS_QWEN4_SSM_PREFILL_FP32": "0",
    "ATLAS_SSM_MULTI_SEQ_GRAPH": "0",
    "ATLAS_WEIGHT_CACHE": "0",
}

WESCHERA_PROMPT = (
    "Write a complete MinHeap class in Python with insert, extract_min, and heapify, "
    "then explain the complexity of each method."
)
M2013_ANSWER = "ORCHID-7391, COBALT-2846, SAFFRON-9153"
M2013_MIN_MEDIAN_PROMPT_TOKENS_PER_SECOND = 2_000.0
M2013_MAX_P90_TTFT_MS = 1_006.5
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


def performance_gate(control: dict[str, Any], candidate: dict[str, Any]) -> dict:
    control_rate = control["prompt_tokens_per_second"]["median"]
    candidate_rate = candidate["prompt_tokens_per_second"]["median"]
    control_p90 = control["ttft_ms"]["p90"]
    candidate_p90 = candidate["ttft_ms"]["p90"]
    checks = {
        "target_median_prompt_tokens_per_second_pass": (
            candidate_rate >= M2013_MIN_MEDIAN_PROMPT_TOKENS_PER_SECOND
        ),
        "target_p90_ttft_ms_pass": candidate_p90 <= M2013_MAX_P90_TTFT_MS,
        "candidate_median_prompt_tokens_per_second_win": candidate_rate > control_rate,
        "candidate_p90_ttft_ms_non_regression": candidate_p90 <= control_p90,
    }
    if not all(checks.values()):
        failed = ",".join(key for key, value in checks.items() if not value)
        raise RuntimeError(f"M2013 performance qualification failed: {failed}")
    return {
        "schema": "atlas-b63-m2013-prefill-performance-gate-v1",
        "minimum_median_prompt_tokens_per_second": (
            M2013_MIN_MEDIAN_PROMPT_TOKENS_PER_SECOND
        ),
        "maximum_p90_ttft_ms": M2013_MAX_P90_TTFT_MS,
        "control_metrics": control,
        "candidate_metrics": candidate,
        **checks,
    }
