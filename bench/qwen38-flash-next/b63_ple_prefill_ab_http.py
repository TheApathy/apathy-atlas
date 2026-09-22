# SPDX-License-Identifier: AGPL-3.0-only
"""Closed HTTP and exact blocking-response boundary for the b63 PLE harness."""

from __future__ import annotations

import hashlib
import http.client
import math
from typing import Any

import b63_ple_prefill_ab_contract as contract
import b63_ple_prefill_ab_identity as identity

TIMING_FIELDS = {"ttft_ms", "prompt_tokens_per_second"}


def http_json(method: str, path: str, body: bytes | None = None) -> tuple[Any, str]:
    connection = http.client.HTTPConnection("127.0.0.1", contract.PORT, timeout=1200)
    headers = {} if body is None else {"Content-Type": "application/json"}
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        raw = response.read((64 << 20) + 1)
        content_type = response.getheader("Content-Type")
    finally:
        connection.close()
    if response.status != 200 or content_type != "application/json":
        raise RuntimeError(f"HTTP identity mismatch: {response.status} {content_type}")
    if len(raw) > 64 << 20:
        raise RuntimeError("HTTP response exceeds bounded parser")
    return identity.strict_json(raw), hashlib.sha256(raw).hexdigest()


def wait_models() -> dict[str, Any]:
    response, digest = http_json("GET", "/v1/models")
    if not isinstance(response, dict) or set(response) != {"object", "data"}:
        raise RuntimeError("model endpoint schema mismatch")
    data = response["data"]
    if response["object"] != "list" or not isinstance(data, list) or len(data) != 1:
        raise RuntimeError("model endpoint cardinality mismatch")
    item = data[0]
    if not isinstance(item, dict) or set(item) != {
        "id",
        "object",
        "created",
        "owned_by",
    }:
        raise RuntimeError("model endpoint record schema mismatch")
    if item["id"] != contract.MODEL_NAME or item["object"] != "model":
        raise RuntimeError("model endpoint identity mismatch")
    if type(item["created"]) is not int or item["created"] <= 0 or not item["owned_by"]:
        raise RuntimeError("model endpoint provenance fields invalid")
    return {"response_sha256": digest, "record": item}


def _exact_response_shape(response: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    top = {"id", "object", "created", "model", "system_fingerprint", "choices", "usage"}
    if not isinstance(response, dict) or set(response) != top:
        raise RuntimeError("chat response top-level schema mismatch")
    if (
        not isinstance(response["id"], str)
        or not response["id"].startswith("chatcmpl-")
        or response["object"] != "chat.completion"
        or type(response["created"]) is not int
        or response["created"] <= 0
        or response["model"] != contract.MODEL_NAME
        or response["system_fingerprint"] != "fp_atlas"
    ):
        raise RuntimeError("chat response identity mismatch")
    choices = response["choices"]
    if not isinstance(choices, list) or len(choices) != 1:
        raise RuntimeError("chat response choice cardinality mismatch")
    choice = choices[0]
    if not isinstance(choice, dict) or set(choice) != {
        "index",
        "message",
        "finish_reason",
        "logprobs",
    }:
        raise RuntimeError("chat choice schema mismatch")
    message = choice["message"]
    if choice["index"] != 0 or choice["logprobs"] is not None:
        raise RuntimeError("chat choice identity mismatch")
    if not isinstance(message, dict) or set(message) != {"role", "content"}:
        raise RuntimeError("chat message schema mismatch")
    if message["role"] != "assistant" or not isinstance(message["content"], str):
        raise RuntimeError("chat message identity mismatch")
    return choice, response["usage"]


def validate_response(response: Any, spec: contract.RequestSpec) -> dict[str, Any]:
    choice, usage = _exact_response_shape(response)
    usage_keys = {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_tokens_details",
        "completion_tokens_details",
        "time_to_first_token_ms",
        "response_token/s",
    }
    if not isinstance(usage, dict) or set(usage) != usage_keys:
        raise RuntimeError("usage schema mismatch")
    if usage["prompt_tokens_details"] != {"cached_tokens": 0, "audio_tokens": 0}:
        raise RuntimeError("nonzero or malformed prompt cache evidence")
    if usage["completion_tokens_details"] != {
        "reasoning_tokens": 0,
        "audio_tokens": 0,
        "accepted_prediction_tokens": 0,
        "rejected_prediction_tokens": 0,
    }:
        raise RuntimeError("completion policy evidence mismatch")
    integer_fields = (
        usage["prompt_tokens"],
        usage["completion_tokens"],
        usage["total_tokens"],
        *usage["prompt_tokens_details"].values(),
        *usage["completion_tokens_details"].values(),
    )
    if any(type(value) is not int for value in integer_fields):
        raise RuntimeError("usage count fields must be exact integers")
    counts = (usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"])
    expected_counts = (
        spec.prompt_tokens,
        spec.completion_tokens,
        spec.prompt_tokens + spec.completion_tokens,
    )
    if counts != expected_counts or choice["finish_reason"] != spec.finish_reason:
        raise RuntimeError("response token/finish oracle mismatch")
    content = choice["message"]["content"]
    stable = {
        "content": content,
        "reasoning_content": "",
        "finish_reason": choice["finish_reason"],
    }
    got = {
        "content_bytes": len(content.encode()),
        "content_sha256": contract.sha256(content.encode()),
        "stable_sha256": contract.sha256(contract.canonical_bytes(stable)),
    }
    if tuple(got.values()) != (
        spec.content_bytes,
        spec.content_sha256,
        spec.stable_sha256,
    ):
        raise RuntimeError(f"{spec.name} exact output/semantic oracle mismatch")
    ttft, decode_rate = usage["time_to_first_token_ms"], usage["response_token/s"]
    if any(
        type(value) is not float or not math.isfinite(value) or value <= 0
        for value in (ttft, decode_rate)
    ):
        raise RuntimeError("non-finite/nonpositive server timing evidence")
    semantic_tuple = [
        spec.prompt_tokens,
        usage["prompt_tokens_details"]["cached_tokens"],
        spec.completion_tokens,
        spec.finish_reason,
        usage["completion_tokens_details"]["reasoning_tokens"],
        content,
    ]
    semantic_sha256 = contract.sha256(contract.canonical_bytes(semantic_tuple))
    if spec.name == "m2013" and semantic_sha256 != contract.M2013_SEMANTIC_SHA256:
        raise RuntimeError("M2013 semantic tuple digest mismatch")
    return {
        **got,
        "prompt_tokens": spec.prompt_tokens,
        "completion_tokens": spec.completion_tokens,
        "finish_reason": spec.finish_reason,
        "semantic_sha256": semantic_sha256,
        "ttft_ms": ttft,
        "prompt_tokens_per_second": spec.prompt_tokens * 1000.0 / ttft,
    }


def persistent_semantic(semantic: dict[str, Any]) -> dict[str, Any]:
    result = {key: value for key, value in semantic.items() if key not in TIMING_FIELDS}
    if TIMING_FIELDS & set(result):
        raise RuntimeError("persistent request evidence contains timing")
    return result
