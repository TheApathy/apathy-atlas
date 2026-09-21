# SPDX-License-Identifier: AGPL-3.0-only
"""Exact provisional semantic-oracle candidate schema and admission rules."""

from __future__ import annotations

from typing import Any

import prefill_decode_repro as harness

CANDIDATE_SCHEMA = "qwen38-semantic-oracle-candidate-v1"
ABI_ATTESTATION_SCHEMA = "qwen38-corrected-attention-abi-attestation-v1"
REPETITIONS = 5
PREFILL_CONFIRMATIONS = 2
PREFILL_COMPLETION_TOKENS = harness.DEFAULT_PREFILL_CONTINUATION_TOKENS
DECODE_COMPLETION_TOKENS = 400
ABI_KEYS = frozenset(
    {
        "schema",
        "source_revision",
        "binary_sha256",
        "kernel_bundle_sha256",
        "ptx_entry",
        "host_argument_count",
        "ptx_parameter_count",
        "ptx_sha256",
    }
)
PREFILL_KEYS = frozenset(
    {
        "target_prompt_tokens",
        "index",
        "request_body_sha256",
        "requested_continuation_tokens",
        "observations",
    }
)
OBSERVATION_KEYS = frozenset(
    {
        "prompt_tokens",
        "cached_prompt_tokens",
        "completion_tokens",
        "finish_reason",
        "stable_output_sha256",
    }
)
DECODE_KEYS = frozenset(
    {
        "index",
        "request_body_sha256",
        "prompt_tokens",
        "cached_prompt_tokens",
        "requested_completion_tokens",
        "completion_tokens",
        "finish_reason",
        "stable_output_sha256",
    }
)
APPROVAL_BOUNDARY = {
    "status": "independent-review-required",
    "accepted_by_harness": False,
    "target_schema": harness.SEMANTIC_ORACLE_SCHEMA,
    "required_action": "independent-review-and-explicit-conversion",
}
CAPTURE_COMPONENT_NAMES = {
    "capture_semantic_oracle.py",
    "semantic_oracle_capture_io.py",
    "semantic_oracle_capture_schema.py",
    "prefill_decode_repro.py",
}


def _exact_keys(value: object, keys: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{label} must contain exactly {sorted(keys)!r}")
    return value


def _hash(value: object, label: str) -> str:
    if not isinstance(value, str) or not harness._is_hex(value, 64):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def validate_abi_attestation(value, provenance):
    abi = _exact_keys(value, ABI_KEYS, "corrected-attention ABI attestation")
    if (
        abi["schema"] != ABI_ATTESTATION_SCHEMA
        or abi["ptx_entry"] != "inferspark_prefill"
    ):
        raise ValueError("ABI attestation schema or PTX entry is not canonical")
    for key in ("source_revision", "binary_sha256", "kernel_bundle_sha256"):
        if abi[key] != provenance[key]:
            raise ValueError(f"ABI attestation {key} differs from provenance")
    if (
        type(abi["host_argument_count"]) is not int
        or abi["host_argument_count"] != 13
        or type(abi["ptx_parameter_count"]) is not int
        or abi["ptx_parameter_count"] != 13
        or abi["ptx_sha256"] != harness.CORRECTED_ATTN_PTX_SHA256
    ):
        raise ValueError("ABI attestation must prove pinned inferspark_prefill 13/13")
    return dict(abi)


def _valid_prefill_rows(rows: list[dict[str, Any]]) -> None:
    expected = {(t, i) for t in harness.DEFAULT_INPUTS for i in range(REPETITIONS)}
    observed = {(row.get("target_prompt_tokens"), row.get("index")) for row in rows}
    if len(rows) != 15 or observed != expected:
        raise ValueError("candidate requires exactly 15 canonical prefill rows")
    requests = set()
    prompts = {target: set() for target in harness.DEFAULT_INPUTS}
    for row in rows:
        _exact_keys(row, PREFILL_KEYS, "prefill row")
        if (
            type(row["target_prompt_tokens"]) is not int
            or type(row["index"]) is not int
        ):
            raise ValueError("prefill target and index must be plain integers")
        _hash(row["request_body_sha256"], "prefill request hash")
        observations = row["observations"]
        if (
            row["request_body_sha256"] in requests
            or type(row["requested_continuation_tokens"]) is not int
            or row["requested_continuation_tokens"] != PREFILL_COMPLETION_TOKENS
            or not isinstance(observations, list)
            or len(observations) != PREFILL_CONFIRMATIONS
        ):
            raise ValueError(
                "prefill request identity, count, or confirmation is invalid"
            )
        requests.add(row["request_body_sha256"])
        if observations[0] != observations[1]:
            raise ValueError("prefill confirmation differs")
        for observation in observations:
            _exact_keys(observation, OBSERVATION_KEYS, "prefill observation")
            _hash(observation["stable_output_sha256"], "prefill output hash")
            if (
                type(observation["prompt_tokens"]) is not int
                or observation["prompt_tokens"] <= 0
                or type(observation["cached_prompt_tokens"]) is not int
                or observation["cached_prompt_tokens"] != 0
                or type(observation["completion_tokens"]) is not int
                or observation["completion_tokens"] != PREFILL_COMPLETION_TOKENS
                or observation["finish_reason"] != "length"
            ):
                raise ValueError("prefill observation is incomplete or cached")
        prompts[row["target_prompt_tokens"]].add(observations[0]["prompt_tokens"])
    if any(len(values) != 1 for values in prompts.values()):
        raise ValueError("prefill prompt counts drift within a bin")


def _valid_decode_rows(rows: list[dict[str, Any]]) -> None:
    if len(rows) != REPETITIONS or {row.get("index") for row in rows} != set(
        range(REPETITIONS)
    ):
        raise ValueError("candidate requires exactly five decode rows")
    for row in rows:
        _exact_keys(row, DECODE_KEYS, "decode row")
        _hash(row["request_body_sha256"], "decode request hash")
        _hash(row["stable_output_sha256"], "decode output hash")
        if (
            type(row["index"]) is not int
            or type(row["prompt_tokens"]) is not int
            or row["prompt_tokens"] <= 0
            or type(row["cached_prompt_tokens"]) is not int
            or row["cached_prompt_tokens"] != 0
            or type(row["requested_completion_tokens"]) is not int
            or row["requested_completion_tokens"] != DECODE_COMPLETION_TOKENS
            or type(row["completion_tokens"]) is not int
            or row["completion_tokens"] != DECODE_COMPLETION_TOKENS
            or row["finish_reason"] != "length"
        ):
            raise ValueError("decode row is incomplete or cached")
    identities = {
        (row["request_body_sha256"], row["prompt_tokens"], row["stable_output_sha256"])
        for row in rows
    }
    if len(identities) != 1:
        raise ValueError("decode rows are not deterministic")


def validate_rows(prefill: list[dict[str, Any]], decode: list[dict[str, Any]]) -> None:
    if prefill:
        _valid_prefill_rows(prefill)
    if decode:
        _valid_decode_rows(decode)


def _validate_capture_identity(value: object) -> dict[str, Any]:
    identity = _exact_keys(
        value, frozenset({"components", "manifest_sha256"}), "capture identity"
    )
    components = identity["components"]
    if not isinstance(components, dict) or set(components) != CAPTURE_COMPONENT_NAMES:
        raise ValueError("capture identity does not name the exact component set")
    for name, digest in components.items():
        _hash(digest, f"capture component {name}")
    manifest = _hash(identity["manifest_sha256"], "capture component manifest")
    if harness.sha256(harness.canonical_bytes(components)) != manifest:
        raise ValueError("capture component manifest hash is inconsistent")
    return {"components": dict(components), "manifest_sha256": manifest}


def build_candidate(
    model,
    provenance,
    provenance_sha,
    abi,
    abi_sha,
    route_evidence,
    prefill,
    decode,
    capture_identity,
):
    if (
        provenance["runtime_mode"] != "no-spec"
        or provenance["same_mode_environment_delta"]
    ):
        raise ValueError("candidate capture requires exact no-spec provenance")
    if not isinstance(model, str) or not model:
        raise ValueError("candidate model is malformed")
    if any(char in model for char in "\x00\r\n"):
        raise ValueError("candidate model is malformed")
    if type(prefill) is not list or type(decode) is not list:
        raise ValueError("candidate row collections must be lists")
    abi = validate_abi_attestation(abi, provenance)
    validate_rows(prefill, decode)
    route_evidence = harness.validate_route_evidence_record(
        route_evidence, provenance["required_route_markers"], provenance
    )
    for value, label in ((provenance_sha, "provenance"), (abi_sha, "ABI attestation")):
        _hash(value, label)
    capture_identity = _validate_capture_identity(capture_identity)
    return {
        "schema": CANDIDATE_SCHEMA,
        "status": "provisional-unapproved",
        "performance_claim": False,
        "approval_boundary": dict(APPROVAL_BOUNDARY),
        "model": model,
        "capture_identity": capture_identity,
        "provenance_manifest_sha256": provenance_sha,
        "provenance": provenance,
        "corrected_attention_abi_attestation_sha256": abi_sha,
        "corrected_attention_abi_attestation": abi,
        "route_evidence": route_evidence,
        "prefill_rows": prefill,
        "decode_rows": decode,
    }
