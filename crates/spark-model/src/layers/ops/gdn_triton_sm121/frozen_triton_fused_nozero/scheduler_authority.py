# SPDX-License-Identifier: AGPL-3.0-only
"""Externally held scheduler authority for the fused GDN raw candidate."""

from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path
from typing import Any

from frozen_triton_c143_io import (
    GateError,
    decode_json,
    require_bool,
    require_exact_keys,
    require_int,
    require_sha256,
)

from .scheduler_trust import HeldRootFile


SCHEDULER_MANIFEST_PATH = Path("/run/atlas/qwen38-gdn-c143-fused-nozero-scheduler.json")
SCHEDULER_KEYS = {
    "schema",
    "variant_source_sha256",
    "adapter_source_sha256",
    "adapter_library_sha256",
    "adapter_library_bytes",
    "base_executor_source_sha256",
    "python_executable",
    "python_executable_sha256",
    "base_manifest_sha256",
    "base_gate_sha256",
    "execution_contract_sha256",
    "build_receipt_path",
    "build_receipt_sha256",
}
RESERVATION_KEYS = {
    "schema",
    "claim",
    "owner",
    "scope",
    "authorization",
    "released",
    "nonce_sha256",
    "base_manifest_sha256",
    "base_gate_sha256",
    "scheduler_manifest_sha256",
    "expires_utc",
}


def bind_scheduler_manifest(
    value: Any, identity: dict[str, Any], contract_sha256: str
) -> dict[str, Any]:
    value = require_exact_keys(value, SCHEDULER_KEYS, "scheduler manifest")
    expected = {
        "schema": "atlas.gdn_c143.fused_nozero_scheduler.v1",
        "variant_source_sha256": identity["variant_source_sha256"],
        "adapter_source_sha256": identity["adapter_source_sha256"],
        "adapter_library_sha256": identity["adapter_library_sha256"],
        "adapter_library_bytes": identity["adapter_library_bytes"],
        "base_executor_source_sha256": identity["base_executor_source_sha256"],
        "python_executable": identity["python_executable"],
        "python_executable_sha256": identity["python_executable_sha256"],
        "base_manifest_sha256": identity["manifest_sha256"],
        "base_gate_sha256": identity["gate_source_sha256"],
        "execution_contract_sha256": contract_sha256,
    }
    for key, wanted in expected.items():
        if value[key] != wanted or type(value[key]) is not type(wanted):
            raise GateError(f"scheduler manifest: {key} drift")
    require_int(value["adapter_library_bytes"], "scheduler adapter bytes", minimum=1)
    for key in SCHEDULER_KEYS - {
        "schema",
        "adapter_library_bytes",
        "python_executable",
        "build_receipt_path",
    }:
        require_sha256(value[key], f"scheduler manifest.{key}")
    if type(value["build_receipt_path"]) is not str:
        raise GateError("scheduler build receipt path is not a string")
    return value


def bind_variant_reservation(
    value: Any,
    identity: dict[str, Any],
    scheduler_sha256: str,
    nonce: str,
    now: dt.datetime,
) -> str:
    value = require_exact_keys(value, RESERVATION_KEYS, "variant GPU reservation")
    if (
        value["schema"] != "atlas.gpu.variant_reservation.v1"
        or value["claim"] != "qwen38-gdn-c143-fused-nozero-raw-gpu"
        or type(value["owner"]) is not str
        or not (value["owner"] == "/root" or value["owner"].startswith("/root/"))
        or value["scope"] != "local-gb10-cuda-execution"
        or value["authorization"] != "GPU_EXECUTION_APPROVED"
        or require_bool(value["released"], "variant reservation.released")
    ):
        raise GateError("variant GPU reservation is inactive or mis-scoped")
    expected = {
        "nonce_sha256": hashlib.sha256(nonce.encode("ascii")).hexdigest(),
        "base_manifest_sha256": identity["manifest_sha256"],
        "base_gate_sha256": identity["gate_source_sha256"],
        "scheduler_manifest_sha256": scheduler_sha256,
    }
    for key, wanted in expected.items():
        if value[key] != wanted:
            raise GateError(f"variant GPU reservation: {key} drift")
        require_sha256(value[key], f"variant GPU reservation.{key}")
    try:
        expires = dt.datetime.fromisoformat(value["expires_utc"].replace("Z", "+00:00"))
    except (AttributeError, ValueError) as error:
        raise GateError("variant GPU reservation expiry is invalid") from error
    if expires.tzinfo is None or expires.utcoffset() != dt.timedelta(0):
        raise GateError("variant GPU reservation expiry is not UTC")
    if expires <= now or expires > now + dt.timedelta(hours=24):
        raise GateError("variant GPU reservation expired or exceeds 24h")
    return value["owner"]


class SchedulerAuthority:
    def __init__(
        self,
        manifest: HeldRootFile,
        reservation: HeldRootFile,
        build: HeldRootFile,
        owner: str,
    ):
        self.manifest, self.reservation, self.build, self.owner = (
            manifest,
            reservation,
            build,
            owner,
        )

    def recheck(self) -> None:
        self.manifest.recheck()
        self.reservation.recheck()
        self.build.recheck()

    def close(self) -> None:
        for held in (self.build, self.reservation, self.manifest):
            held.close()


def authorize_gpu_execution(
    *,
    explicit_flag: str | None,
    nonce: str,
    reservation_path: Path,
    identity: dict[str, Any],
    contract_sha256: str,
) -> SchedulerAuthority:
    if (
        explicit_flag != "I_UNDERSTAND_THIS_RUNS_CUDA"
        or len(nonce) != 64
        or any(c not in "0123456789abcdef" for c in nonce)
    ):
        raise GateError("variant GPU authorization flag or nonce is invalid")
    manifest = HeldRootFile.open(SCHEDULER_MANIFEST_PATH, "scheduler manifest", {0o444})
    reservation = build = None
    try:
        bound = bind_scheduler_manifest(
            decode_json(manifest.raw, "scheduler manifest"), identity, contract_sha256
        )
        build = HeldRootFile.open(
            Path(bound["build_receipt_path"]), "scheduler build receipt", {0o444}
        )
        if build.sha256 != bound["build_receipt_sha256"]:
            raise GateError("scheduler build receipt digest drift")
        reservation = HeldRootFile.open(
            reservation_path, "variant GPU reservation", {0o400, 0o444}
        )
        owner = bind_variant_reservation(
            decode_json(reservation.raw, "variant GPU reservation"),
            identity,
            manifest.sha256,
            nonce,
            dt.datetime.now(dt.timezone.utc),
        )
        return SchedulerAuthority(manifest, reservation, build, owner)
    except Exception:
        for held in (build, reservation, manifest):
            if held is not None:
                held.close()
        raise
