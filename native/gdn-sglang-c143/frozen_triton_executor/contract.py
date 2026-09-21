# SPDX-License-Identifier: AGPL-3.0-only
"""Immutable execution-plan and local-source identity checks."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

from frozen_triton_c143_io import (
    GateError,
    decode_json,
    require_exact_keys,
    require_sha256,
    stable_read,
)

ROOT = Path(__file__).resolve().parent.parent
EXECUTOR_FILES = (
    "run_frozen_triton_c143.py",
    "frozen_triton_executor/__init__.py",
    "frozen_triton_executor/contract.py",
    "frozen_triton_executor/verified_loader.py",
    "frozen_triton_executor/cuda_driver.py",
    "frozen_triton_executor/buffers.py",
    "frozen_triton_executor/launch.py",
    "frozen_triton_executor/references.py",
    "frozen_triton_executor/case.py",
    "frozen_triton_executor/timing.py",
    "frozen_triton_executor/evidence.py",
    "frozen_triton_executor/receipt.py",
    "frozen_triton_executor/publication.py",
    "frozen_triton_executor/main.py",
    "frozen_triton_c143_output_overwrite.py",
    "run_frozen_triton_c143_nozero.py",
    "frozen_triton_nozero/__init__.py",
    "frozen_triton_nozero/launch.py",
    "frozen_triton_nozero/case.py",
    "frozen_triton_nozero/main.py",
)
ATLAS_SOURCE = (
    ROOT.parent.parent / "kernels/gb10/common/gated_delta_rule_wy32_gatecache.cu"
)
ATLAS_BRIDGE_SOURCE = ROOT / "src/atlas_wy32_bridge.cu"
ATLAS_LIBRARY = ROOT / "build/libatlas_gdn_wy32_sm121.so"
ATLAS_LIBRARY_BYTES = 724_688
LEGACY_GATE = ROOT / "gate.py"
PINNED = {
    "atlas_parent_source_sha256": "f73a7071aa8160960e9221bb584caa1e9f733856c0d82bcc39085e9e881d6aed",
    "atlas_bridge_source_sha256": "f9c4f61dae7613770ad20d9f6e975e87d947757970406e1dfd91fccc8ee9e07f",
    "atlas_bridge_sha256": "6e567af30fcd82d353b80f800bc47ca49ad33d803a2e7ef324cf47523a063a31",
    "legacy_gate_sha256": "95a718029f8094c52f18a0897d7312d2855afa91301790fec00a7f35cafed465",
}
CONTRACT_KEYS = {
    "schema",
    "m_sequence",
    "manifest_sha256",
    "artifact_attestation_sha256",
    "gate_source_sha256",
    "executor_source_sha256",
    "python_executable_sha256",
    *PINNED,
    "reservation_receipt_sha256",
    "nonce_sha256",
    "result_path",
}


def sha256_file(path: Path, label: str, maximum_size: int = 16 << 20) -> str:
    return stable_read(path.resolve(strict=True), label, maximum_size=maximum_size)[1]


def executor_source_sha256() -> str:
    records = {}
    for name in EXECUTOR_FILES:
        raw, digest, _ = stable_read(ROOT / name, f"executor source:{name}")
        records[name] = [digest, len(raw)]
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def local_identity(attestation: dict[str, Any], gate_sha256: str) -> dict[str, Any]:
    executable = Path(sys.executable).resolve(strict=True)
    values = {
        "manifest_sha256": attestation["manifest_sha256"],
        "artifact_attestation_sha256": attestation["artifact_attestation_sha256"],
        "gate_source_sha256": gate_sha256,
        "executor_source_sha256": executor_source_sha256(),
        "python_executable_sha256": sha256_file(
            executable, "Python executable", 64 << 20
        ),
        "atlas_parent_source_sha256": sha256_file(ATLAS_SOURCE, "Atlas parent source"),
        "atlas_bridge_source_sha256": sha256_file(
            ATLAS_BRIDGE_SOURCE, "Atlas bridge source"
        ),
        "atlas_bridge_sha256": sha256_file(ATLAS_LIBRARY, "Atlas bridge", 64 << 20),
        "legacy_gate_sha256": sha256_file(LEGACY_GATE, "legacy reference gate"),
    }
    if any(values[key] != expected for key, expected in PINNED.items()):
        raise GateError("current Atlas parent/reference identity drift")
    return {"python_executable": str(executable), **values}


def contract_template(
    attestation: dict[str, Any],
    gate_sha256: str,
    *,
    reservation_sha256: str,
    nonce: str,
    result_path: Path,
) -> dict[str, Any]:
    return {
        "schema": "atlas.gdn_c143.frozen_triton_execution_contract.v1",
        "m_sequence": [2_079, 8_192],
        **{
            k: v
            for k, v in local_identity(attestation, gate_sha256).items()
            if k != "python_executable"
        },
        "reservation_receipt_sha256": reservation_sha256,
        "nonce_sha256": hashlib.sha256(nonce.encode("ascii")).hexdigest(),
        "result_path": str(result_path),
    }


def validate_contract(
    path: Path,
    attestation: dict[str, Any],
    gate_sha256: str,
    authorization: dict[str, Any],
    nonce: str,
) -> tuple[dict[str, Any], str]:
    raw, contract_sha, stat = stable_read(
        path, "execution contract", maximum_size=64 << 10
    )
    if stat.st_mode & 0o022:
        raise GateError("execution contract is group/world writable")
    value = require_exact_keys(
        decode_json(raw, "execution contract"), CONTRACT_KEYS, "execution contract"
    )
    if type(value["result_path"]) is not str:
        raise GateError("execution contract result path must be a string")
    expected = contract_template(
        attestation,
        gate_sha256,
        reservation_sha256=authorization["reservation_receipt_sha256"],
        nonce=nonce,
        result_path=Path(value["result_path"]),
    )
    if value != expected:
        raise GateError("execution contract identity or shape drift")
    result = Path(value["result_path"])
    if not result.is_absolute() or result.exists() or result.is_symlink():
        raise GateError("result path must be fresh, absolute, and absent")
    parent = result.parent.resolve(strict=True)
    if parent != result.parent or os.stat(parent).st_mode & 0o022:
        raise GateError(
            "result directory must be canonical and not group/world writable"
        )
    for key in CONTRACT_KEYS - {"schema", "m_sequence", "result_path"}:
        require_sha256(value[key], f"execution contract.{key}")
    return value, contract_sha
