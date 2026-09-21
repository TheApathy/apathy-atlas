# SPDX-License-Identifier: AGPL-3.0-only
"""Strict HTTP response, selector-receipt, log, and campaign gates."""

from __future__ import annotations

import math
import re
import shlex
import statistics
from pathlib import Path
from typing import Any

import b63_ple_prefill_ab_contract as contract
import b63_ple_prefill_ab_identity as identity

RECEIPT_MARKER = "QWEN4_PREFILL_ENGAGED"
COMMON_RECEIPT = {
    "family": "ple",
    "selector": "ATLAS_QWEN4_PLE_PREFILL_BATCH",
    "projection": "cublaslt_bf16_non_bit_exact",
    "projection_parity_required": "true",
    "performance_claim_allowed": "false",
    "heads": "8",
    "hidden_size": "2560",
    "hc_count": "2",
    "reset_state": "true",
}

RAW_PARITY_SCHEMA = "atlas-b63-ple-raw-parity-qualification-v1"


def attest_raw_parity() -> dict[str, Any]:
    digest = contract.RAW_PARITY_RECEIPT_SHA256
    if not isinstance(digest, str) or not identity.HEX64.fullmatch(digest):
        raise RuntimeError("authoritative raw PLE parity receipt is not pinned")
    document, evidence = identity.load_sealed_json(contract.RAW_PARITY_RECEIPT, digest)
    required = {
        "schema",
        "qualified",
        "performance_claim_allowed",
        "binary_sha256",
        "build_files",
        "model_manifest_sha256",
        "model_content_root_sha256",
        "control_selector",
        "candidate_selector",
        "prompt_tokens",
        "projection",
        "hidden_equal",
        "live_state_equal",
        "checkpoint_state_equal",
        "canonical_output_equal",
        "capture_source_bundle_sha256",
        "capture_artifacts_root_sha256",
    }
    expected = {
        "schema": RAW_PARITY_SCHEMA,
        "qualified": True,
        "performance_claim_allowed": False,
        "binary_sha256": contract.BINARY_SHA256,
        "build_files": contract.BUILD_FILES,
        "model_manifest_sha256": contract.MODEL_MANIFEST_SHA256,
        "model_content_root_sha256": contract.MODEL_CONTENT_ROOT_SHA256,
        "control_selector": 0,
        "candidate_selector": 1,
        "prompt_tokens": [38, 2013],
        "projection": "cublaslt_bf16_non_bit_exact",
        "hidden_equal": True,
        "live_state_equal": True,
        "checkpoint_state_equal": True,
        "canonical_output_equal": True,
    }
    if not isinstance(document, dict) or set(document) != required:
        raise RuntimeError("invalid raw PLE parity receipt schema")
    if (
        type(document["control_selector"]) is not int
        or type(document["candidate_selector"]) is not int
        or not isinstance(document["prompt_tokens"], list)
        or any(type(value) is not int for value in document["prompt_tokens"])
    ):
        raise RuntimeError("raw PLE parity selector/token types are not exact integers")
    bool_keys = (key for key, value in expected.items() if type(value) is bool)
    if any(type(document[key]) is not bool for key in bool_keys):
        raise RuntimeError("raw PLE parity Boolean fields must be exact booleans")
    if any(document.get(key) != value for key, value in expected.items()):
        raise RuntimeError("raw PLE parity qualification mismatch")
    for key in ("capture_source_bundle_sha256", "capture_artifacts_root_sha256"):
        if not isinstance(document[key], str) or not identity.HEX64.fullmatch(
            document[key]
        ):
            raise RuntimeError(f"invalid raw PLE parity field: {key}")
    return {"document": document, "file": evidence}


def parse_receipts(text: str) -> list[dict[str, str]]:
    receipts = []
    for line in text.splitlines():
        if RECEIPT_MARKER not in line:
            continue
        payload = line.split(RECEIPT_MARKER, 1)[1].strip()
        fields: dict[str, str] = {}
        try:
            tokens = shlex.split(payload)
        except ValueError as error:
            raise RuntimeError("malformed quoted selector receipt") from error
        for token in tokens:
            if token.count("=") != 1:
                raise RuntimeError("malformed selector receipt token")
            key, value = token.split("=", 1)
            if not key or not value or key in fields:
                raise RuntimeError("duplicate/empty selector receipt field")
            fields[key] = value
        receipts.append(fields)
    return receipts


def validate_server_log(path: Path, arm: str) -> dict[str, Any]:
    if path.stat().st_size > 64 << 20:
        raise RuntimeError("server log exceeds bounded parser")
    text = path.read_text(errors="strict")
    required = {
        "listen": (r"Listening on 127\.0\.0\.1:8998", 1),
        "prefix_off": (r"Prefix caching: disabled", 1),
        "marconi_zero": (r"SSM snapshot pool: Marconi 0 slots", 1),
        "maxseq1": (
            r"Scheduler started .*max_batch=1, mtp=false, ngram=false, num_drafts=0, "
            r"policy=fifo, chunked_prefill=true, max_prefill_tokens=2048",
            1,
        ),
        "session_m38": (r"Session .*: 38 prompt tokens, tools=false \(0 defined\)", 1),
        "session_m2013": (
            r"Session .*: 2013 prompt tokens, tools=false \(0 defined\)",
            5,
        ),
        "request_400": (r"Request: .*max_tokens=400(?:\D|$)", 1),
        "request_32": (r"Request: .*max_tokens=32(?:\D|$)", 5),
        "prefill_m38": (
            r"Chunked prefill start: 38 prompt tokens, .*max_tokens=400",
            1,
        ),
        "prefill_m2013": (
            r"Chunked prefill start: 2013 prompt tokens, .*max_tokens=32",
            5,
        ),
        "done_400": (r"Done: 400 tokens \(length\)", 1),
        "done_27": (r"Done: 27 tokens \(stop\)", 5),
    }
    census = {
        name: len(re.findall(pattern, text)) for name, (pattern, _) in required.items()
    }
    if any(census[name] != expected for name, (_, expected) in required.items()):
        raise RuntimeError(f"server log request/startup census mismatch: {census}")
    if text.count("Request:") != 6 or text.count("Done:") != 6:
        raise RuntimeError("server log request/completion cardinality mismatch")
    forbidden = re.compile(
        r"\b(?:WARN|ERROR|FATAL|panic|fallback|cache hit|NaN|OOM)\b|"
        r"Speculative decoding: ENABLED|DFLASH|QWEN4_PREFILL_SELECTOR_RECEIPT",
        re.I,
    )
    if forbidden.search(text):
        raise RuntimeError(
            "server log contains error/cache/speculation/fallback evidence"
        )
    receipts = parse_receipts(text)
    if arm == "control":
        if receipts:
            raise RuntimeError("control emitted selector receipts")
    elif arm == "candidate":
        if [item.get("num_tokens") for item in receipts] != ["38"] + ["2013"] * 5:
            raise RuntimeError("PLE receipt token/order/cardinality mismatch")
        for receipt in receipts:
            target = {**COMMON_RECEIPT, "num_tokens": receipt["num_tokens"]}
            if receipt != target:
                raise RuntimeError("PLE receipt projection/geometry/policy mismatch")
    else:
        raise RuntimeError("unknown PLE A/B arm")
    return {"census": census, "selector_receipts": len(receipts)}


def metrics(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if len(runs) != 5 or any(run["kind"] != "m2013" for run in runs):
        raise RuntimeError("metrics require exactly five M2013 trials")
    ttfts = [run["semantic"]["ttft_ms"] for run in runs]
    if any(type(v) is not float or not math.isfinite(v) or v <= 0 for v in ttfts):
        raise RuntimeError("metrics require finite positive exact-float TTFT")
    if any(
        type(run["semantic"].get("prompt_tokens")) is not int
        or run["semantic"]["prompt_tokens"] != 2_013
        for run in runs
    ):
        raise RuntimeError("metrics require exact M2013 prompt counts")
    rates = [2_013_000.0 / value for value in ttfts]
    if any(
        type(run["semantic"].get("prompt_tokens_per_second")) is not float
        or run["semantic"]["prompt_tokens_per_second"] != rate
        for run, rate in zip(runs, rates, strict=True)
    ):
        raise RuntimeError("reported prefill rate differs from recomputed TTFT rate")
    return {
        key: {
            "values": values,
            "median": statistics.median(values),
            "p90": contract.percentile(values, 0.9),
        }
        for key, values in {"ttft_ms": ttfts, "prompt_tokens_per_second": rates}.items()
    }


def campaign_gate(arms: list[dict[str, Any]]) -> dict[str, Any]:
    if [arm["arm"] for arm in arms] != list(contract.ARM_ORDER):
        raise RuntimeError("campaign arm ordering drift")
    identities = {
        (arm["process"]["pid"], arm["process"]["starttime_ticks"]) for arm in arms
    }
    if len(identities) != len(contract.ARM_ORDER):
        raise RuntimeError("each arm must use a fresh server process")
    expected = contract.request_specs()
    baseline = [
        {
            key: value
            for key, value in run["semantic"].items()
            if key not in {"ttft_ms", "prompt_tokens_per_second"}
        }
        for run in arms[0]["runs"]
    ]
    reports = {}
    for arm in arms:
        stable = [run["semantic"]["stable_sha256"] for run in arm["runs"]]
        required = [expected["canary"].stable_sha256] + [
            expected["m2013"].stable_sha256
        ] * 5
        if stable != required:
            raise RuntimeError("cross-arm request output/semantic oracle drift")
        current = [
            {
                key: value
                for key, value in run["semantic"].items()
                if key not in {"ttft_ms", "prompt_tokens_per_second"}
            }
            for run in arm["runs"]
        ]
        if current != baseline:
            raise RuntimeError("cross-arm canonical output/usage equality drift")
        report = metrics(arm["runs"][1:])
        if arm.get("metrics") != report:
            raise RuntimeError("arm metrics differ from recomputed admitted trials")
        reports[arm["arm"]] = report
    return contract.performance_gate(reports["control"], reports["candidate"])
