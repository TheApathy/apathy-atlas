# SPDX-License-Identifier: AGPL-3.0-only
"""Strict HTTP response, selector-receipt, log, and campaign gates."""

from __future__ import annotations

import re
import statistics
from pathlib import Path
from typing import Any

import b62_prefill_ab_contract as contract

RECEIPT_MARKER = "QWEN4_PREFILL_SELECTOR_RECEIPT "
COMMON_RECEIPT = {
    "value": "1",
    "path_success": "enqueued",
    "serialized_fallback": "false",
    "H": "2560",
    "L": "48",
    "E": "512",
    "TOPK": "10",
    "I": "640",
    "SI": "640",
}
FAMILY_RECEIPT = {
    "attention": {
        "family": "attention",
        "selector": "ATLAS_QWEN4_ATTN_PREFILL_BATCH",
        "attention_selector": "1",
        "ssm_selector": "0",
        "expected_layers": "12",
        "engaged_layers": "12",
        "Q": "24",
        "KV": "2",
        "HD": "256",
    },
    "ssm": {
        "family": "ssm",
        "selector": "ATLAS_QWEN4_SSM_PREFILL_BATCH",
        "attention_selector": "0",
        "ssm_selector": "1",
        "expected_layers": "36",
        "engaged_layers": "36",
        "NK": "16",
        "KD": "128",
        "NV": "48",
        "VD": "128",
        "D": "4",
        "QKVZ": "16384",
    },
}


def parse_receipts(text: str) -> list[dict[str, str]]:
    receipts = []
    for line in text.splitlines():
        if RECEIPT_MARKER not in line:
            continue
        payload = line.split(RECEIPT_MARKER, 1)[1].strip()
        fields: dict[str, str] = {}
        for token in payload.split():
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
        r"Speculative decoding: ENABLED|DFLASH",
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
    else:
        expected = {**COMMON_RECEIPT, **FAMILY_RECEIPT[arm]}
        if [item.get("M") for item in receipts] != ["38"] + ["2013"] * 5:
            raise RuntimeError("selector receipt M/order/cardinality mismatch")
        for receipt in receipts:
            target = {**expected, "M": receipt["M"]}
            if receipt != target:
                raise RuntimeError("selector receipt family/geometry/census mismatch")
    return {"census": census, "selector_receipts": len(receipts)}


def metrics(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if len(runs) != 5 or any(run["kind"] != "m2013" for run in runs):
        raise RuntimeError("metrics require exactly five M2013 trials")
    result = {}
    for key in ("ttft_ms", "prompt_tokens_per_second"):
        values = [float(run["semantic"][key]) for run in runs]
        result[key] = {
            "values": values,
            "median": statistics.median(values),
            "p90": contract.percentile(values, 0.9),
        }
    return result


def campaign_gate(arms: list[dict[str, Any]]) -> None:
    if [arm["arm"] for arm in arms] != list(contract.ARM_ORDER):
        raise RuntimeError("campaign arm ordering drift")
    identities = {
        (arm["process"]["pid"], arm["process"]["starttime_ticks"]) for arm in arms
    }
    if len(identities) != 3:
        raise RuntimeError("each arm must use a fresh server process")
    expected = contract.request_specs()
    for arm in arms:
        if (
            arm["runs"][0]["semantic"]["stable_sha256"]
            != expected["canary"].stable_sha256
        ):
            raise RuntimeError("cross-arm canary oracle drift")
        if any(
            run["semantic"]["stable_sha256"] != expected["m2013"].stable_sha256
            for run in arm["runs"][1:]
        ):
            raise RuntimeError("cross-arm M2013 output/semantic oracle drift")
