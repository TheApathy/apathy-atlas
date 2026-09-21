# SPDX-License-Identifier: AGPL-3.0-only
"""Hostile validation and exclusive publication for executor envelopes."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import frozen_triton_c143_gate as sealed_gate
from frozen_triton_c143_io import (
    GateError,
    require_exact_keys,
    require_int,
    stable_read,
)

from .contract import PINNED
from .evidence import aggregate_fixtures

ENVELOPE_KEYS = {
    "schema",
    "qualification",
    "production_authorized",
    "default_off",
    "execution_contract_sha256",
    "manifest_sha256",
    "artifact_attestation_sha256",
    "gate_source_sha256",
    "executor_source_sha256",
    "python_executable_sha256",
    *PINNED,
    "reservation_receipt_sha256",
    "device",
    "m_sequence",
    "cases",
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def validate_envelope(
    envelope: Any,
    attestation: dict,
    contract: dict,
    contract_sha256: str,
) -> None:
    value = require_exact_keys(envelope, ENVELOPE_KEYS, "executor receipt")
    if (
        value["schema"] != "atlas.gdn_c143.frozen_triton_executor_receipt.v1"
        or value["qualification"] != "PASS"
        or value["production_authorized"] is not False
        or value["default_off"] is not True
        or value["execution_contract_sha256"] != contract_sha256
        or value["m_sequence"] != [2_079, 8_192]
    ):
        raise GateError("executor receipt policy drift")
    excluded = {
        "schema",
        "qualification",
        "production_authorized",
        "default_off",
        "execution_contract_sha256",
        "device",
        "m_sequence",
        "cases",
    }
    for key in ENVELOPE_KEYS - excluded:
        if value[key] != contract[key]:
            raise GateError(f"executor receipt binding drift: {key}")
    device = require_exact_keys(
        value["device"],
        {"ordinal", "name", "major", "minor", "multiprocessor_count"},
        "executor device",
    )
    identity = tuple(
        require_int(device[key], f"device.{key}")
        for key in ("ordinal", "major", "minor", "multiprocessor_count")
    )
    if (
        type(device["name"]) is not str
        or "GB10" not in device["name"]
        or identity != (0, 12, 1, 48)
    ):
        raise GateError("executor receipt device drift")
    if (
        type(value["cases"]) is not list
        or any(type(item) is not dict for item in value["cases"])
        or [item.get("m") for item in value["cases"]] != [2_079, 8_192]
    ):
        raise GateError("executor receipt case order drift")
    prior_sha = None
    for item in value["cases"]:
        if set(item) != {"m", "raw_receipt", "raw_receipt_sha256", "fixtures"}:
            raise GateError("executor case schema drift")
        actual = hashlib.sha256(canonical_bytes(item["raw_receipt"])).hexdigest()
        if actual != item["raw_receipt_sha256"]:
            raise GateError("nested raw receipt hash drift")
        sealed_gate.validate_raw_receipt(
            item["raw_receipt"],
            attestation,
            prior_m2079_receipt_sha256=prior_sha,
        )
        aggregate = aggregate_fixtures(item["fixtures"])
        for key, expected in aggregate.items():
            if item["raw_receipt"][key] != expected:
                raise GateError(f"nested raw receipt fixture aggregate drift: {key}")
        prior_sha = actual


def publish_exclusive(path: Path, envelope: dict) -> str:
    raw = (
        json.dumps(envelope, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = None
    created = None
    try:
        descriptor = os.open(path, flags, 0o400)
        created = os.fstat(descriptor)
        try:
            view = memoryview(raw)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise OSError("short executor receipt write")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
            descriptor = None
        os.chmod(path, 0o444)
        _, digest, stat = stable_read(
            path, "published executor receipt", maximum_size=16 << 20
        )
        if stat.st_mode & 0o222 or stat.st_size != len(raw):
            raise GateError("published executor receipt mode/size drift")
        return digest
    except Exception:
        if descriptor is not None:
            os.close(descriptor)
        if created is not None:
            try:
                current = path.lstat()
            except FileNotFoundError:
                pass
            else:
                if (current.st_dev, current.st_ino) == (created.st_dev, created.st_ino):
                    path.unlink()
        raise
