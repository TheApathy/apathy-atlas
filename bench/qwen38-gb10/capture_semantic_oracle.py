#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Capture an unapproved semantic-oracle candidate from a frozen no-spec server."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import prefill_decode_repro as harness
import semantic_oracle_capture_io as capture_io
import semantic_oracle_capture_schema as schema

DECODE_COMPLETION_TOKENS = schema.DECODE_COMPLETION_TOKENS
PREFILL_COMPLETION_TOKENS = schema.PREFILL_COMPLETION_TOKENS
PREFILL_CONFIRMATIONS = schema.PREFILL_CONFIRMATIONS
REPETITIONS = schema.REPETITIONS
build_candidate = schema.build_candidate
validate_abi_attestation = schema.validate_abi_attestation
validate_rows = schema.validate_rows


def capture_identity() -> dict[str, Any]:
    paths = {
        "capture_semantic_oracle.py": Path(__file__),
        "semantic_oracle_capture_io.py": Path(capture_io.__file__),
        "semantic_oracle_capture_schema.py": Path(schema.__file__),
        "prefill_decode_repro.py": Path(harness.__file__),
    }
    components = {
        name: harness.sha256(path.read_bytes()) for name, path in paths.items()
    }
    return {
        "components": components,
        "manifest_sha256": harness.sha256(harness.canonical_bytes(components)),
    }


def _observation(response: object, requested: int, *, decode: bool) -> dict[str, Any]:
    if (
        not isinstance(response, dict)
        or not isinstance(response.get("choices"), list)
        or len(response["choices"]) != 1
    ):
        raise ValueError("response must contain exactly one choice")
    choice = response["choices"][0]
    usage = response.get("usage")
    if not isinstance(choice, dict) or not isinstance(usage, dict):
        raise ValueError("response choice and usage must be objects")
    details = usage.get("prompt_tokens_details")
    prompt = usage.get("prompt_tokens")
    completed = usage.get("completion_tokens")
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    finish = choice.get("finish_reason")
    if (
        type(prompt) is not int
        or prompt <= 0
        or type(completed) is not int
        or completed != requested
    ):
        raise ValueError(
            "response token counts are missing, non-integral, or incomplete"
        )
    if (
        type(cached) is not int
        or cached != 0
        or finish != harness.DECODE_COMPLETE_FINISH_REASON
    ):
        raise ValueError(
            "response must explicitly prove zero cache and finish_reason=length"
        )
    if decode:
        message = choice.get("message")
        if not isinstance(message, dict):
            raise ValueError("decode response message is missing")
        content = message.get("content")
        reasoning = message.get("reasoning_content")
        if content is not None and not isinstance(content, str):
            raise ValueError("decode content must be a string or null")
        if reasoning is not None and not isinstance(reasoning, str):
            raise ValueError("decode output fields must be strings or null")
        content, reasoning = content or "", reasoning or ""
        if not content.strip() and not reasoning.strip():
            raise ValueError("decode output must be nonempty")
        output = {
            "content": content,
            "reasoning_content": reasoning,
            "finish_reason": finish,
        }
    else:
        text = choice.get("text")
        if text is not None and not isinstance(text, str):
            raise ValueError("prefill output text must be a string or null")
        text = text or ""
        if not text.strip():
            raise ValueError("prefill output must be nonempty")
        output = {"text": text, "finish_reason": finish}
    return {
        "prompt_tokens": prompt,
        "cached_prompt_tokens": cached,
        "completion_tokens": completed,
        "finish_reason": finish,
        "stable_output_sha256": harness.sha256(harness.canonical_bytes(output)),
    }


def capture_prefill_rows(
    endpoint: str, model: str, post: Callable = harness.post_json
) -> list[dict[str, Any]]:
    rows = []
    for target in harness.DEFAULT_INPUTS:
        for index in range(REPETITIONS):
            body = {
                "model": model,
                "prompt": harness.build_prefill_prompt(
                    target, index, PREFILL_COMPLETION_TOKENS
                ),
                "temperature": 0.0,
                "max_tokens": PREFILL_COMPLETION_TOKENS,
                "logit_bias": {
                    str(token): -100.0
                    for token in harness.PREFILL_SUPPRESSED_STOP_TOKEN_IDS
                },
            }
            observations = [
                _observation(
                    post(endpoint, body), PREFILL_COMPLETION_TOKENS, decode=False
                )
                for _ in range(PREFILL_CONFIRMATIONS)
            ]
            if observations[0] != observations[1]:
                raise ValueError(
                    f"prefill request {(target, index)} is not deterministic"
                )
            rows.append(
                {
                    "target_prompt_tokens": target,
                    "index": index,
                    "request_body_sha256": harness.sha256(
                        harness.canonical_bytes(body)
                    ),
                    "requested_continuation_tokens": PREFILL_COMPLETION_TOKENS,
                    "observations": observations,
                }
            )
    validate_rows(rows, [])
    return rows


def capture_decode_rows(
    endpoint: str, model: str, post: Callable = harness.post_json
) -> list[dict[str, Any]]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": harness.DECODE_PROMPT}],
        "temperature": 0.0,
        "reasoning_effort": "none",
        "max_tokens": DECODE_COMPLETION_TOKENS,
    }
    request_hash = harness.sha256(harness.canonical_bytes(body))
    rows = []
    for index in range(REPETITIONS):
        observation = _observation(
            post(endpoint, body), DECODE_COMPLETION_TOKENS, decode=True
        )
        rows.append(
            {
                "index": index,
                "request_body_sha256": request_hash,
                "requested_completion_tokens": DECODE_COMPLETION_TOKENS,
                **observation,
            }
        )
    if (
        len(
            {
                (
                    row["prompt_tokens"],
                    row["completion_tokens"],
                    row["finish_reason"],
                    row["stable_output_sha256"],
                )
                for row in rows
            }
        )
        != 1
    ):
        raise ValueError("decode requests are not deterministic")
    validate_rows([], rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture an unapproved semantic-oracle candidate; this tool cannot approve or convert it."
    )
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--route-log", type=Path, required=True)
    parser.add_argument("--provenance-manifest", type=Path, required=True)
    parser.add_argument("--expected-provenance-sha256", required=True)
    parser.add_argument("--abi-attestation", type=Path, required=True)
    parser.add_argument("--expected-abi-attestation-sha256", required=True)
    parser.add_argument("--expected-capture-components-sha256", required=True)
    args = parser.parse_args()
    identity = capture_identity()
    if identity["manifest_sha256"] != args.expected_capture_components_sha256:
        parser.error(
            "capture component manifest SHA-256 mismatch: "
            f"expected {args.expected_capture_components_sha256}, got {identity['manifest_sha256']}"
        )
    try:
        output, digest = capture_io.run_capture(
            args,
            validate_abi_attestation,
            capture_prefill_rows,
            capture_decode_rows,
            build_candidate,
            capture_identity,
            identity,
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        parser.error(str(error))
    print(
        json.dumps(
            {
                "candidate": str(output),
                "sha256": digest,
                "status": "provisional-unapproved",
                "performance_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
