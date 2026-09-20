# SPDX-License-Identifier: AGPL-3.0-only
"""Existing raw-receipt composition and provenance envelope publication."""

from __future__ import annotations

import hashlib

import frozen_triton_c143_gate as sealed_gate
from frozen_triton_c143_io import GateError

from .contract import PINNED
from .publication import canonical_bytes, validate_envelope


def build_raw_receipt(
    case: dict,
    attestation: dict,
    contract: dict,
    authorization: dict,
    stream_identity_sha256: str,
    prior_sha256: str | None,
    artifacts_unchanged: bool,
) -> dict:
    m = case["m"]
    return {
        "schema": "atlas.gdn_c143.frozen_triton_raw_receipt.v1",
        "qualification": "PASS",
        "production_authorized": False,
        "default_off": True,
        "m": m,
        "nt": sealed_gate.workspace_layout(m)["nt"],
        "manifest_sha256": attestation["manifest_sha256"],
        "gate_sha256": contract["gate_source_sha256"],
        "artifact_attestation_sha256": attestation["artifact_attestation_sha256"],
        "reservation_receipt_sha256": authorization["reservation_receipt_sha256"],
        "m2079_pass_receipt_sha256": prior_sha256,
        "geometry": sealed_gate.EXPECTED_GEOMETRY,
        "workspace": sealed_gate.workspace_layout(m),
        "stream": {
            "nondefault": True,
            "same_stream_all_stages": True,
            "stream_identity_sha256": stream_identity_sha256,
        },
        "artifacts_matched_before": True,
        "artifacts_matched_after": artifacts_unchanged,
        "adapter_checks": case["adapter_checks"],
        "canaries": {
            "all_clean": case["canaries_clean"],
            "checked_regions": sealed_gate.CANARY_REGIONS,
        },
        "input_hashes_before": case["input_hashes_before"],
        "input_hashes_after": case["input_hashes_after"],
        "stage_hashes": case["stage_hashes"],
        "finite": case["finite"],
        "comparisons": case["comparisons"],
        "stage_timing_ms": case["stage_timing_ms"],
        "full_timing": case["full_timing"],
        "publication": {
            "exclusive_create": True,
            "written_after_all_gates": True,
            "preexisting": False,
            "result_path": contract["result_path"],
        },
    }


def build_envelope(
    case_results: list[dict],
    attestation: dict,
    contract: dict,
    contract_sha256: str,
    authorization: dict,
    device: dict,
    stream_handle: int,
) -> dict:
    if [case["m"] for case in case_results] != [2_079, 8_192]:
        raise GateError("executor must run M2079 before M8192")
    current = sealed_gate.attest_manifest()
    artifacts_unchanged = (
        current["manifest_sha256"] == attestation["manifest_sha256"]
        and current["artifact_attestation_sha256"]
        == attestation["artifact_attestation_sha256"]
    )
    stream_sha = hashlib.sha256(
        f"{stream_handle}:{contract_sha256}".encode("ascii")
    ).hexdigest()
    cases = []
    prior_sha = None
    for result in case_results:
        raw = build_raw_receipt(
            result,
            attestation,
            contract,
            authorization,
            stream_sha,
            prior_sha,
            artifacts_unchanged,
        )
        sealed_gate.validate_raw_receipt(
            raw,
            attestation,
            prior_m2079_receipt_sha256=prior_sha,
        )
        raw_sha = hashlib.sha256(canonical_bytes(raw)).hexdigest()
        cases.append(
            {
                "m": result["m"],
                "raw_receipt": raw,
                "raw_receipt_sha256": raw_sha,
                "fixtures": result["fixtures"],
            }
        )
        prior_sha = raw_sha
    envelope = {
        "schema": "atlas.gdn_c143.frozen_triton_executor_receipt.v1",
        "qualification": "PASS",
        "production_authorized": False,
        "default_off": True,
        "execution_contract_sha256": contract_sha256,
        **{
            key: contract[key]
            for key in (
                "manifest_sha256",
                "artifact_attestation_sha256",
                "gate_source_sha256",
                "executor_source_sha256",
                "python_executable_sha256",
                *PINNED,
                "reservation_receipt_sha256",
            )
        },
        "device": device,
        "m_sequence": [2_079, 8_192],
        "cases": cases,
    }
    validate_envelope(envelope, attestation, contract, contract_sha256)
    return envelope
