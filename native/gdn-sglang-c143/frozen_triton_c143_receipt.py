"""Strict schema and integrity validation for raw-oracle receipts."""

from pathlib import Path
from typing import Any

from frozen_triton_c143_constants import (
    ADAPTER_CHECKS,
    CANARY_REGIONS,
    EXPECTED_GEOMETRY,
    FINITE_NAMES,
    IMMUTABLE_NAMES,
    STAGE_NAMES,
)
from frozen_triton_c143_io import (
    GateError,
    require_bool,
    require_exact_keys,
    require_int,
    require_sha256,
)
from frozen_triton_c143_layout import workspace_layout
from frozen_triton_c143_receipt_metrics import validate_metrics_and_timing

RECEIPT_KEYS = {
    "schema",
    "qualification",
    "production_authorized",
    "default_off",
    "m",
    "nt",
    "manifest_sha256",
    "gate_sha256",
    "artifact_attestation_sha256",
    "reservation_receipt_sha256",
    "m2079_pass_receipt_sha256",
    "geometry",
    "workspace",
    "stream",
    "artifacts_matched_before",
    "artifacts_matched_after",
    "adapter_checks",
    "canaries",
    "input_hashes_before",
    "input_hashes_after",
    "stage_hashes",
    "finite",
    "comparisons",
    "stage_timing_ms",
    "full_timing",
    "publication",
}


def validate_raw_receipt(
    receipt: Any,
    attestation: dict[str, Any],
    *,
    gate_source_sha256: str,
    prior_m2079_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    value = require_exact_keys(receipt, RECEIPT_KEYS, "raw receipt")
    if value["schema"] != "atlas.gdn_c143.frozen_triton_raw_receipt.v1":
        raise GateError("raw receipt schema drift")
    if value["qualification"] != "PASS":
        raise GateError("raw receipt is not PASS")
    if require_bool(value["production_authorized"], "production_authorized"):
        raise GateError("raw receipt cannot authorize production")
    if not require_bool(value["default_off"], "default_off"):
        raise GateError("raw receipt must remain default off")
    m = require_int(value["m"], "m", minimum=1)
    layout = workspace_layout(m)
    if value["nt"] != layout["nt"]:
        raise GateError("raw receipt NT drift")
    if value["manifest_sha256"] != attestation["manifest_sha256"]:
        raise GateError("raw receipt manifest binding mismatch")
    if value["gate_sha256"] != gate_source_sha256:
        raise GateError("raw receipt gate source binding mismatch")
    if (
        value["artifact_attestation_sha256"]
        != attestation["artifact_attestation_sha256"]
    ):
        raise GateError("raw receipt artifact attestation mismatch")
    require_sha256(value["reservation_receipt_sha256"], "reservation_receipt_sha256")
    prior = value["m2079_pass_receipt_sha256"]
    if m == 2_079:
        if prior is not None:
            raise GateError("M2079 receipt must not claim a prior M2079 receipt")
        if prior_m2079_receipt_sha256 is not None:
            raise GateError("M2079 validation must not supply a prior receipt")
    else:
        claimed = require_sha256(prior, "m2079_pass_receipt_sha256")
        if prior_m2079_receipt_sha256 is None:
            raise GateError("M8192 requires an actual validated M2079 PASS receipt")
        actual = require_sha256(
            prior_m2079_receipt_sha256, "actual M2079 PASS receipt SHA256"
        )
        if claimed != actual:
            raise GateError("M8192 prior M2079 PASS receipt binding mismatch")
    if value["geometry"] != EXPECTED_GEOMETRY:
        raise GateError("raw receipt geometry drift")
    if value["workspace"] != layout:
        raise GateError("raw receipt workspace layout drift")
    stream = require_exact_keys(
        value["stream"],
        {"nondefault", "same_stream_all_stages", "stream_identity_sha256"},
        "stream",
    )
    if not require_bool(stream["nondefault"], "stream.nondefault"):
        raise GateError("default stream is forbidden")
    if not require_bool(
        stream["same_stream_all_stages"], "stream.same_stream_all_stages"
    ):
        raise GateError("all stages must use the same stream")
    require_sha256(stream["stream_identity_sha256"], "stream identity")
    if not require_bool(value["artifacts_matched_before"], "artifacts before"):
        raise GateError("pre-run artifact attestation failed")
    if not require_bool(value["artifacts_matched_after"], "artifacts after"):
        raise GateError("post-run artifact attestation failed")
    adapters = require_exact_keys(
        value["adapter_checks"], set(ADAPTER_CHECKS), "adapter_checks"
    )
    if not all(
        require_bool(adapters[name], f"adapter_checks.{name}")
        for name in ADAPTER_CHECKS
    ):
        raise GateError("adapter contract failed")
    canaries = require_exact_keys(
        value["canaries"], {"all_clean", "checked_regions"}, "canaries"
    )
    if not require_bool(canaries["all_clean"], "canaries.all_clean"):
        raise GateError("canary corruption")
    if canaries["checked_regions"] != CANARY_REGIONS:
        raise GateError("canary region set drift")
    before = require_exact_keys(
        value["input_hashes_before"], set(IMMUTABLE_NAMES), "input hashes before"
    )
    after = require_exact_keys(
        value["input_hashes_after"], set(IMMUTABLE_NAMES), "input hashes after"
    )
    for name in IMMUTABLE_NAMES:
        require_sha256(before[name], f"input before:{name}")
        require_sha256(after[name], f"input after:{name}")
        if before[name] != after[name]:
            raise GateError(f"immutable input changed: {name}")
    stage_hashes = require_exact_keys(
        value["stage_hashes"], set(STAGE_NAMES), "stage hashes"
    )
    for stage in STAGE_NAMES:
        hashes = require_exact_keys(
            stage_hashes[stage], {"run1", "run2"}, f"stage:{stage}"
        )
        if require_sha256(hashes["run1"], f"stage:{stage}.run1") != require_sha256(
            hashes["run2"], f"stage:{stage}.run2"
        ):
            raise GateError(f"nondeterministic stage bytes: {stage}")
    finite = require_exact_keys(value["finite"], set(FINITE_NAMES), "finite")
    if not all(require_bool(finite[name], f"finite.{name}") for name in FINITE_NAMES):
        raise GateError("non-finite stage/output/state")
    metrics = validate_metrics_and_timing(value, m)
    publication = require_exact_keys(
        value["publication"],
        {"exclusive_create", "written_after_all_gates", "preexisting", "result_path"},
        "publication",
    )
    result_path = Path(publication["result_path"])
    if (
        not require_bool(
            publication["exclusive_create"], "publication.exclusive_create"
        )
        or not require_bool(
            publication["written_after_all_gates"],
            "publication.written_after_all_gates",
        )
        or require_bool(publication["preexisting"], "publication.preexisting")
        or not result_path.is_absolute()
    ):
        raise GateError("fail-closed publication contract failed")
    return {"m": m, **metrics, "qualification": "PASS", "production_authorized": False}
