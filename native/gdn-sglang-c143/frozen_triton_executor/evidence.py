# SPDX-License-Identifier: AGPL-3.0-only
"""Typed real/adversarial fixture aggregation for raw receipts."""

from __future__ import annotations

import hashlib
from typing import Any

import frozen_triton_c143_gate as sealed_gate
from frozen_triton_c143_io import GateError, require_exact_keys, require_sha256

FIXTURE_KEYS = {
    "kind",
    "input_before",
    "input_after",
    "stage_run1",
    "stage_run2",
    "adapter_checks",
    "finite",
    "identities",
    "comparisons",
    "deterministic",
    "all_guards_clean",
}


def combine_hashes(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        if type(value) is not str or len(value) != 64:
            raise GateError("fixture evidence contains malformed SHA256")
        try:
            digest.update(bytes.fromhex(value))
        except ValueError as exc:
            raise GateError("fixture evidence contains malformed SHA256") from exc
    return digest.hexdigest()


def worst_metrics(values: list[dict]) -> dict[str, float]:
    return {
        "max_abs": max(value["max_abs"] for value in values),
        "rms": max(value["rms"] for value in values),
        "relative_rms": max(value["relative_rms"] for value in values),
        "cosine": min(value["cosine"] for value in values),
    }


def validate_fixture_evidence(fixtures: Any) -> None:
    if (
        type(fixtures) is not list
        or any(type(item) is not dict for item in fixtures)
        or [item.get("kind") for item in fixtures] != ["real", "adversarial"]
    ):
        raise GateError("executor fixture family/order drift")
    for fixture in fixtures:
        require_exact_keys(fixture, FIXTURE_KEYS, f"fixture:{fixture['kind']}")
        inputs_before = require_exact_keys(
            fixture["input_before"], set(sealed_gate.IMMUTABLE_NAMES), "fixture inputs"
        )
        inputs_after = require_exact_keys(
            fixture["input_after"],
            set(sealed_gate.IMMUTABLE_NAMES),
            "fixture inputs after",
        )
        if inputs_after != inputs_before:
            raise GateError("fixture immutable input drift")
        require_exact_keys(
            fixture["stage_run1"], set(sealed_gate.STAGE_NAMES), "fixture stages"
        )
        require_exact_keys(
            fixture["stage_run2"],
            set(sealed_gate.STAGE_NAMES),
            "fixture stages replay",
        )
        require_exact_keys(
            fixture["adapter_checks"],
            set(sealed_gate.ADAPTER_CHECKS),
            "fixture adapters",
        )
        require_exact_keys(
            fixture["finite"], set(sealed_gate.FINITE_NAMES), "fixture finite"
        )
        identities = require_exact_keys(
            fixture["identities"],
            {
                f"{arm}_run{run}"
                for arm in ("candidate", "atlas", "sglang")
                for run in (1, 2)
            },
            "fixture identities",
        )
        require_exact_keys(
            fixture["comparisons"],
            {"candidate_vs_atlas", "candidate_vs_sglang"},
            "fixture comparisons",
        )
        if (
            fixture["deterministic"] is not True
            or fixture["all_guards_clean"] is not True
            or not all(value is True for value in fixture["adapter_checks"].values())
            or not all(value is True for value in fixture["finite"].values())
        ):
            raise GateError(f"{fixture['kind']}: fixture evidence failed")
        for identity in identities.values():
            require_exact_keys(
                identity,
                {"output_sha256", "state_sha256", "guards_clean", "finite"},
                "fixture identity",
            )
            if identity["guards_clean"] is not True or identity["finite"] is not True:
                raise GateError(f"{fixture['kind']}: reference evidence failed")
            for field in ("output_sha256", "state_sha256"):
                require_sha256(identity[field], f"fixture identity.{field}")
        for arm in ("candidate", "atlas", "sglang"):
            if any(
                identities[f"{arm}_run1"][field] != identities[f"{arm}_run2"][field]
                for field in ("output_sha256", "state_sha256")
            ):
                raise GateError(f"{fixture['kind']}: identity replay mismatch")


def aggregate_fixtures(fixtures: list[dict]) -> dict:
    validate_fixture_evidence(fixtures)
    return {
        "input_hashes_before": {
            name: combine_hashes([item["input_before"][name] for item in fixtures])
            for name in sealed_gate.IMMUTABLE_NAMES
        },
        "input_hashes_after": {
            name: combine_hashes([item["input_after"][name] for item in fixtures])
            for name in sealed_gate.IMMUTABLE_NAMES
        },
        "stage_hashes": {
            name: {
                run: combine_hashes([item[f"stage_{run}"][name] for item in fixtures])
                for run in ("run1", "run2")
            }
            for name in sealed_gate.STAGE_NAMES
        },
        "adapter_checks": {
            name: all(item["adapter_checks"][name] for item in fixtures)
            for name in sealed_gate.ADAPTER_CHECKS
        },
        "finite": {
            name: all(item["finite"][name] for item in fixtures)
            for name in sealed_gate.FINITE_NAMES
        },
        "comparisons": {
            reference: {
                field: worst_metrics(
                    [item["comparisons"][reference][field] for item in fixtures]
                )
                for field in ("output", "state")
            }
            for reference in ("candidate_vs_sglang", "candidate_vs_atlas")
        },
    }
