# SPDX-License-Identifier: AGPL-3.0-only
"""Bit-first raw PLE parity and timing-free response qualification."""

from __future__ import annotations

import hashlib
import json
import math
import signal
import struct
from typing import Any

import ple_prefill_parity_contract as contract

TOKEN_FIELDS = (
    "request_tokens_sha256",
    "ple_prior_m",
    "ple_prior_tokens_sha256",
    "ple_ordered_m",
    "ple_ordered_tokens_sha256",
    "chunk_start",
    "chunk_m",
    "chunk_tokens_sha256",
    "reset_state",
    "continuation",
    "slot_idx",
)
LIFECYCLE_KEYS = {
    "actions",
    "timeout_after_sigterm_seconds",
    "returncode",
    "clean_exit",
    "sigterm_identity_rechecked",
    "sigkill_identity_rechecked",
    "session_drain_signal_count",
    "final_session_empty",
    "final_listener_empty",
    "final_gpu_inventory_empty",
    "build_model_authority_stable",
}
RECEIPT_IDENTITY_KEYS = {
    "sha256",
    "bytes",
    "dev",
    "ino",
    "mode",
    "nlink",
    "mtime_ns",
}


def _bounded_lifecycle_tuple(value: object) -> str:
    """Return one canonical, bounded diagnostic without echoing hostile values."""

    document = value if type(value) is dict else {}
    actions = document.get("actions")
    if (
        type(actions) is list
        and len(actions) <= 3
        and all(
            type(action) is str
            and 0 < len(action) <= 16
            and action.isascii()
            and all(char.isalnum() or char == "_" for char in action)
            for action in actions
        )
    ):
        bounded_actions: object = actions
    else:
        bounded_actions = "<invalid>"

    def bounded_integer(key: str, *, optional: bool = False) -> object:
        item = document.get(key)
        if optional and item is None:
            return None
        if type(item) is int and -3_600 <= item <= 3_600:
            return item
        return "<invalid>"

    def bounded_boolean(key: str) -> object:
        item = document.get(key)
        return item if type(item) is bool else "<invalid>"

    fields = (
        ("actions", bounded_actions),
        (
            "timeout_after_sigterm_seconds",
            bounded_integer("timeout_after_sigterm_seconds", optional=True),
        ),
        ("returncode", bounded_integer("returncode")),
        ("clean_exit", bounded_boolean("clean_exit")),
        (
            "sigterm_identity_rechecked",
            bounded_boolean("sigterm_identity_rechecked"),
        ),
        (
            "sigkill_identity_rechecked",
            bounded_boolean("sigkill_identity_rechecked"),
        ),
        ("session_drain_signal_count", bounded_integer("session_drain_signal_count")),
        ("final_session_empty", bounded_boolean("final_session_empty")),
        ("final_listener_empty", bounded_boolean("final_listener_empty")),
        ("final_gpu_inventory_empty", bounded_boolean("final_gpu_inventory_empty")),
        (
            "build_model_authority_stable",
            bounded_boolean("build_model_authority_stable"),
        ),
    )
    return json.dumps(fields, ensure_ascii=True, separators=(",", ":"))


def _reject_lifecycle(reason: str, value: object) -> None:
    raise RuntimeError(f"{reason}; lifecycle_tuple={_bounded_lifecycle_tuple(value)}")


class ParityMismatch(RuntimeError):
    def __init__(self, diagnostics: list[dict[str, Any]]) -> None:
        super().__init__("STOP-NO-TIMING: raw post-PLE byte mismatch")
        self.diagnostics = diagnostics


def _response_shape(response: object) -> tuple[dict[str, Any], dict[str, Any]]:
    top = {"id", "object", "created", "model", "system_fingerprint", "choices", "usage"}
    if not isinstance(response, dict) or set(response) != top:
        raise RuntimeError("response top-level schema drift")
    if (
        type(response["id"]) is not str
        or not response["id"].startswith("chatcmpl-")
        or response["object"] != "chat.completion"
        or type(response["created"]) is not int
        or response["created"] <= 0
        or response["model"] != contract.MODEL_NAME
        or response["system_fingerprint"] != "fp_atlas"
    ):
        raise RuntimeError("response identity drift")
    choices = response["choices"]
    if not isinstance(choices, list) or len(choices) != 1:
        raise RuntimeError("response choice cardinality drift")
    choice = choices[0]
    if not isinstance(choice, dict) or set(choice) != {
        "index",
        "message",
        "finish_reason",
        "logprobs",
    }:
        raise RuntimeError("response choice schema drift")
    message = choice["message"]
    if choice["index"] != 0 or choice["logprobs"] is not None:
        raise RuntimeError("response choice identity drift")
    if not isinstance(message, dict) or set(message) != {"role", "content"}:
        raise RuntimeError("response message schema drift")
    if message["role"] != "assistant" or type(message["content"]) is not str:
        raise RuntimeError("response message identity drift")
    if not isinstance(response["usage"], dict):
        raise RuntimeError("response usage type drift")
    return choice, response["usage"]


def canonical_response(response: object, spec: contract.RequestSpec) -> dict[str, Any]:
    choice, usage = _response_shape(response)
    usage_keys = {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_tokens_details",
        "completion_tokens_details",
        "time_to_first_token_ms",
        "response_token/s",
    }
    if set(usage) != usage_keys:
        raise RuntimeError("response usage schema drift")
    prompt_details = {"cached_tokens": 0, "audio_tokens": 0}
    completion_details = {
        "reasoning_tokens": 0,
        "audio_tokens": 0,
        "accepted_prediction_tokens": 0,
        "rejected_prediction_tokens": 0,
    }
    if usage["prompt_tokens_details"] != prompt_details:
        raise RuntimeError("response cache evidence drift")
    if usage["completion_tokens_details"] != completion_details:
        raise RuntimeError("response speculative/reasoning evidence drift")
    integer_values = (
        usage["prompt_tokens"],
        usage["completion_tokens"],
        usage["total_tokens"],
        *usage["prompt_tokens_details"].values(),
        *usage["completion_tokens_details"].values(),
    )
    if any(type(value) is not int for value in integer_values):
        raise RuntimeError("response usage counts are not exact integers")
    counts = (usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"])
    wanted = (
        spec.prompt_tokens,
        spec.completion_tokens,
        spec.prompt_tokens + spec.completion_tokens,
    )
    if counts != wanted or choice["finish_reason"] != spec.finish_reason:
        raise RuntimeError("response token/finish oracle drift")
    content = choice["message"]["content"]
    stable = {
        "content": content,
        "reasoning_content": "",
        "finish_reason": spec.finish_reason,
    }
    if (
        len(content.encode()),
        hashlib.sha256(content.encode()).hexdigest(),
        hashlib.sha256(contract.canonical_bytes(stable)).hexdigest(),
    ) != (spec.content_bytes, spec.content_sha256, spec.stable_sha256):
        raise RuntimeError("response exact output oracle drift")
    # Timing fields are admitted only as server-wire compatibility. Their
    # values are never compared, returned, persisted, or used for a claim.
    if any(
        type(usage[key]) is not float or not math.isfinite(usage[key])
        for key in ("time_to_first_token_ms", "response_token/s")
    ):
        raise RuntimeError("response contains non-finite wire-only timing")
    return {
        "content_bytes": spec.content_bytes,
        "content_sha256": spec.content_sha256,
        "stable_sha256": spec.stable_sha256,
        "finish_reason": spec.finish_reason,
        "usage": {
            "prompt_tokens": usage["prompt_tokens"],
            "completion_tokens": usage["completion_tokens"],
            "total_tokens": usage["total_tokens"],
            "cached_tokens": 0,
            "reasoning_tokens": 0,
            "accepted_prediction_tokens": 0,
            "rejected_prediction_tokens": 0,
        },
    }


def _ordered_bf16(bits: int) -> int:
    return (~bits & 0xFFFF) if bits & 0x8000 else bits | 0x8000


def mismatch_diagnostics(
    label: str, control: bytes, candidate: bytes
) -> dict[str, Any]:
    if control == candidate:
        raise RuntimeError("diagnostics requested without a byte mismatch")
    if len(control) != len(candidate):
        raise RuntimeError(
            "STOP-NO-TIMING: mismatched raw extents; diagnostics forbidden"
        )
    if not control or len(control) % 2:
        raise RuntimeError("STOP-NO-TIMING: invalid BF16 mismatch extent")
    max_abs = max_rel = dot = left_sq = right_sq = 0.0
    max_ulp = 0
    left = struct.iter_unpack("<H", control)
    right = struct.iter_unpack("<H", candidate)
    for (left_bits,), (right_bits,) in zip(left, right, strict=True):
        left_value = struct.unpack("<f", struct.pack("<I", left_bits << 16))[0]
        right_value = struct.unpack("<f", struct.pack("<I", right_bits << 16))[0]
        if not math.isfinite(left_value) or not math.isfinite(right_value):
            raise RuntimeError(
                "STOP-NO-TIMING: non-finite BF16 mismatch; diagnostics forbidden"
            )
        delta = abs(left_value - right_value)
        max_abs = max(max_abs, delta)
        max_rel = max(max_rel, delta / max(abs(left_value), 1.0e-30))
        max_ulp = max(
            max_ulp, abs(_ordered_bf16(left_bits) - _ordered_bf16(right_bits))
        )
        dot += left_value * right_value
        left_sq += left_value * left_value
        right_sq += right_value * right_value
    left_norm, right_norm = math.sqrt(left_sq), math.sqrt(right_sq)
    cosine = dot / (left_norm * right_norm) if left_norm and right_norm else 0.0
    if not all(
        math.isfinite(value)
        for value in (max_abs, max_rel, cosine, dot, left_sq, right_sq)
    ):
        raise RuntimeError("STOP-NO-TIMING: non-finite mismatch metrics")
    return {
        "label": label,
        "bytes": len(control),
        "max_abs": max_abs,
        "max_rel": max_rel,
        "max_bf16_ulp": max_ulp,
        "cosine": cosine,
    }


def admit_lifecycle(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != LIFECYCLE_KEYS:
        _reject_lifecycle("fresh-process lifecycle receipt schema drift", value)
    boolean_keys = (
        "clean_exit",
        "sigterm_identity_rechecked",
        "sigkill_identity_rechecked",
        "final_session_empty",
        "final_listener_empty",
        "final_gpu_inventory_empty",
        "build_model_authority_stable",
    )
    if (
        type(value["actions"]) is not list
        or any(type(action) is not str for action in value["actions"])
        or type(value["returncode"]) is not int
        or type(value["session_drain_signal_count"]) is not int
        or any(type(value[key]) is not bool for key in boolean_keys)
    ):
        _reject_lifecycle("fresh-process lifecycle receipt type drift", value)
    common = (
        value["sigterm_identity_rechecked"]
        and value["session_drain_signal_count"] == 0
        and value["final_session_empty"]
        and value["final_listener_empty"]
        and value["final_gpu_inventory_empty"]
        and value["build_model_authority_stable"]
    )
    graceful = (
        value["actions"] == ["sigterm"]
        and value["timeout_after_sigterm_seconds"] is None
        and value["returncode"] == 0
        and value["clean_exit"]
        and not value["sigkill_identity_rechecked"]
    )
    terminated = (
        value["actions"] == ["sigterm"]
        and value["timeout_after_sigterm_seconds"] is None
        and value["returncode"] == -int(signal.SIGTERM)
        and not value["clean_exit"]
        and not value["sigkill_identity_rechecked"]
    )
    forced = (
        value["actions"] == ["sigterm", "sigkill"]
        and type(value["timeout_after_sigterm_seconds"]) is int
        and value["timeout_after_sigterm_seconds"] == 15
        and value["returncode"] == -int(signal.SIGKILL)
        and not value["clean_exit"]
        and value["sigkill_identity_rechecked"]
    )
    if not common or not (graceful or terminated or forced):
        _reject_lifecycle("fresh-process lifecycle receipt is not admissible", value)
    return value


def _sealed_frame_gate(frame: object) -> dict[str, Any]:
    if not isinstance(frame, dict) or set(frame) != {
        "receipt",
        "receipt_identity",
        "hidden",
        "live",
        "checkpoint",
    }:
        raise RuntimeError("complete sealed frame schema drift")
    identity = frame["receipt_identity"]
    if (
        not isinstance(identity, dict)
        or set(identity) != RECEIPT_IDENTITY_KEYS
        or type(identity["sha256"]) is not str
        or len(identity["sha256"]) != 64
        or any(ch not in "0123456789abcdef" for ch in identity["sha256"])
        or any(
            type(identity[key]) is not int for key in RECEIPT_IDENTITY_KEYS - {"sha256"}
        )
        or identity["bytes"] <= 0
        or identity["dev"] <= 0
        or identity["ino"] <= 0
        or identity["mode"] != 0o444
        or identity["nlink"] != 1
        or identity["mtime_ns"] <= 0
    ):
        raise RuntimeError("complete sealed frame receipt identity drift")
    receipt = frame["receipt"]
    if (
        not isinstance(receipt, dict)
        or receipt.get("frame_commit_mode") != "0500"
        or receipt.get("performance_claim_allowed") is not False
        or receipt.get("producer_stream_synchronized") is not True
    ):
        raise RuntimeError("complete sealed frame commit receipt drift")
    return frame


def _process_gate(run: dict[str, Any]) -> tuple[int, int]:
    process = run.get("process")
    if not isinstance(process, dict) or set(process) != {
        "pid",
        "starttime_ticks",
        "elf_sha256",
        "argv",
        "environment",
        "lifecycle",
    }:
        raise RuntimeError("fresh-process receipt schema drift")
    if (
        type(process["pid"]) is not int
        or process["pid"] <= 1
        or type(process["starttime_ticks"]) is not int
        or process["starttime_ticks"] <= 0
        or process["elf_sha256"] != contract.PINS["elf"]
        or process["argv"] != contract.server_argv()
        or process["environment"]
        != contract.arm_environment(run["arm"], run["nonce"], run["capture_root"])
    ):
        raise RuntimeError("fresh-process identity/sole-delta drift")
    admit_lifecycle(process["lifecycle"])
    return process["pid"], process["starttime_ticks"]


def validate_campaign(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if [(run.get("name"), run.get("arm")) for run in runs] != list(contract.SCHEDULE):
        raise RuntimeError("PLE parity arm/request ordering drift")
    seen_processes, seen_nonces, diagnostics = set(), set(), []
    specs = contract.request_specs()
    for run in runs:
        if set(run) != {
            "name",
            "arm",
            "nonce",
            "capture_root",
            "process",
            "response",
            "frames",
        }:
            raise RuntimeError("PLE parity run schema drift")
        nonce = run["nonce"]
        if (
            type(nonce) is not str
            or len(nonce) != 64
            or any(ch not in "0123456789abcdef" for ch in nonce)
        ):
            raise RuntimeError("PLE parity run nonce drift")
        identity = _process_gate(run)
        if identity in seen_processes or nonce in seen_nonces:
            raise RuntimeError("PLE parity process or nonce was reused")
        seen_processes.add(identity)
        seen_nonces.add(nonce)
        wanted = len(contract.FRAME_LAYOUT[specs[run["name"]].prompt_tokens])
        if not isinstance(run["frames"], list) or len(run["frames"]) != wanted:
            raise RuntimeError("PLE parity loaded-frame census drift")
        for frame in run["frames"]:
            frame = _sealed_frame_gate(frame)
            receipt = frame["receipt"]
            expected = contract.TOKEN_PINS[(run["name"], receipt["chunk_start"])]
            actual = tuple(
                receipt[key]
                for key in (
                    "request_tokens_sha256",
                    "ple_prior_tokens_sha256",
                    "ple_ordered_tokens_sha256",
                    "chunk_tokens_sha256",
                )
            )
            if actual != expected:
                raise RuntimeError("PLE parity canonical token receipt drift")
    for index in (0, 2):
        control, candidate = runs[index], runs[index + 1]
        spec = specs[control["name"]]
        for frame_index, (left, right) in enumerate(
            zip(control["frames"], candidate["frames"], strict=True)
        ):
            if any(
                left["receipt"][key] != right["receipt"][key] for key in TOKEN_FIELDS
            ):
                raise RuntimeError("PLE parity token/state-bound receipt drift")
            for field in ("hidden", "live", "checkpoint"):
                if left[field] != right[field]:
                    diagnostics.append(
                        mismatch_diagnostics(
                            f"{spec.name}.frame{frame_index}.{field}",
                            left[field],
                            right[field],
                        )
                    )
        if diagnostics:
            raise ParityMismatch(diagnostics)
        if canonical_response(control["response"], spec) != canonical_response(
            candidate["response"], spec
        ):
            raise RuntimeError("STOP-NO-TIMING: canonical response/usage mismatch")
    return {
        "schema": contract.RESULT_SCHEMA,
        "qualified": True,
        "performance_claim_allowed": False,
        "timing_allowed": False,
        "schedule": [list(item) for item in contract.SCHEDULE],
        "raw_bit_equal": True,
        "canonical_response_equal": True,
    }
