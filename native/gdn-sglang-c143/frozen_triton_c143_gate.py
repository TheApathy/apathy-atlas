#!/usr/bin/env python3
"""CPU attestation and receipt gates for the frozen Triton c143 raw oracle.

No module in this isolated family imports a GPU library or implements loading or
launching. A separately reviewed future executor must retain this authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from frozen_triton_c143_attest import attest_manifest, attest_manifest_unsealed
from frozen_triton_c143_authorization import authorize_gpu_execution as _authorize
from frozen_triton_c143_constants import (
    ADAPTER_CHECKS,
    CANARY_REGIONS,
    DEFAULT_MANIFEST,
    EXPECTED_GEOMETRY,
    FINITE_NAMES,
    IMMUTABLE_NAMES,
    POINTER_ALIGNMENT,
    STAGE_NAMES,
    SUPPORTED_M,
    UINT64_MAX,
    WORKSPACE_ALIGNMENT,
)
from frozen_triton_c143_io import GateError, decode_json, hash_file, stable_read
from frozen_triton_c143_layout import (
    resolved_launch_plan,
    validate_runtime_extents,
    workspace_layout,
)
from frozen_triton_c143_receipt import validate_raw_receipt as _validate_receipt

__all__ = [
    "ADAPTER_CHECKS",
    "CANARY_REGIONS",
    "EXPECTED_GEOMETRY",
    "FINITE_NAMES",
    "GateError",
    "IMMUTABLE_NAMES",
    "POINTER_ALIGNMENT",
    "STAGE_NAMES",
    "SUPPORTED_M",
    "UINT64_MAX",
    "WORKSPACE_ALIGNMENT",
    "attest_manifest",
    "authorize_gpu_execution",
    "main",
    "resolved_launch_plan",
    "validate_raw_receipt",
    "validate_runtime_extents",
    "workspace_layout",
]

_stable_read = stable_read
_hash_file = hash_file

SOURCE_MODULES = (
    "frozen_triton_c143_artifact.py",
    "frozen_triton_c143_attest.py",
    "frozen_triton_c143_authorization.py",
    "frozen_triton_c143_constants.py",
    "frozen_triton_c143_gate.py",
    "frozen_triton_c143_io.py",
    "frozen_triton_c143_layout.py",
    "frozen_triton_c143_manifest_policy.py",
    "frozen_triton_c143_provenance.py",
    "frozen_triton_c143_receipt.py",
    "frozen_triton_c143_receipt_metrics.py",
)


def _gate_source_sha256() -> str:
    records = {}
    root = Path(__file__).resolve().parent
    for name in SOURCE_MODULES:
        raw, digest, _ = stable_read(root / name, f"gate source:{name}")
        records[name] = [digest, len(raw)]
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def _attest_manifest(
    manifest_path: Path, *, expected_manifest_sha256: str | None
) -> dict[str, Any]:
    return attest_manifest_unsealed(
        manifest_path, expected_manifest_sha256=expected_manifest_sha256
    )


def authorize_gpu_execution(
    *,
    explicit_flag: str | None,
    nonce: str | None,
    reservation_path: Path | None,
    manifest_sha256: str,
    now: Any = None,
) -> dict[str, Any]:
    return _authorize(
        explicit_flag=explicit_flag,
        nonce=nonce,
        reservation_path=reservation_path,
        manifest_sha256=manifest_sha256,
        gate_source_sha256=_gate_source_sha256(),
        now=now,
    )


def validate_raw_receipt(
    receipt: Any,
    attestation: dict[str, Any],
    *,
    prior_m2079_receipt_sha256: str | None = None,
) -> dict[str, Any]:
    return _validate_receipt(
        receipt,
        attestation,
        gate_source_sha256=_gate_source_sha256(),
        prior_m2079_receipt_sha256=prior_m2079_receipt_sha256,
    )


def _attestation_summary(attestation: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": "atlas.gdn_c143.frozen_triton_attestation.v1",
        "mode": "ATTEST_ONLY",
        "qualification": "ATTESTED_CPU_ONLY",
        "gpu_execution": False,
        "gpu_executor_implemented": False,
        "production_authorized": False,
        "manifest_sha256": attestation["manifest_sha256"],
        "artifact_attestation_sha256": attestation["artifact_attestation_sha256"],
        "gate_source_bundle_sha256": _gate_source_sha256(),
        "attested_file_count": len(attestation["attested_files"]),
        "supported_m": list(SUPPORTED_M),
        "workspace_bytes": {
            str(m): workspace_layout(m)["total_bytes"] for m in SUPPORTED_M
        },
        "launch_grids": {
            str(m): [entry["grid"] for entry in resolved_launch_plan(attestation, m)]
            for m in SUPPORTED_M
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--attest-only", action="store_true")
    modes.add_argument("--validate-receipt", type=Path)
    modes.add_argument("--execute-gpu", action="store_true")
    parser.add_argument("--m2079-pass-receipt", type=Path)
    parser.add_argument("--gpu-execution-flag")
    parser.add_argument("--reservation-nonce")
    parser.add_argument("--reservation", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        attestation = attest_manifest(args.manifest)
        if args.execute_gpu:
            authorize_gpu_execution(
                explicit_flag=args.gpu_execution_flag,
                nonce=args.reservation_nonce,
                reservation_path=args.reservation,
                manifest_sha256=attestation["manifest_sha256"],
            )
            raise GateError(
                "GPU executor is intentionally absent from this CPU-only preparation; "
                "a separate reviewed implementation claim is required"
            )
        if any(
            value is not None
            for value in (
                args.gpu_execution_flag,
                args.reservation_nonce,
                args.reservation,
            )
        ):
            raise GateError("GPU authorization arguments require --execute-gpu")
        if args.m2079_pass_receipt is not None and args.validate_receipt is None:
            raise GateError("--m2079-pass-receipt requires --validate-receipt")
        if args.validate_receipt is not None:
            raw, _, _ = stable_read(
                args.validate_receipt, "raw receipt", maximum_size=8 << 20
            )
            prior_sha = None
            if args.m2079_pass_receipt is not None:
                prior_raw, prior_sha, _ = stable_read(
                    args.m2079_pass_receipt,
                    "prior M2079 PASS receipt",
                    maximum_size=8 << 20,
                )
                prior_result = validate_raw_receipt(
                    decode_json(prior_raw, "prior M2079 PASS receipt"), attestation
                )
                if prior_result["m"] != 2_079:
                    raise GateError("prior receipt is not an M2079 PASS receipt")
            result = validate_raw_receipt(
                decode_json(raw, "raw receipt"),
                attestation,
                prior_m2079_receipt_sha256=prior_sha,
            )
            print(json.dumps(result, sort_keys=True))
            return 0
        print(json.dumps(_attestation_summary(attestation), sort_keys=True))
        return 0
    except (GateError, OSError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
