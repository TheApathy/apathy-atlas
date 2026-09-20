# SPDX-License-Identifier: AGPL-3.0-only
"""Independent source, artifact, and execution-contract authority."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent.parent
REPO = HERE.parents[5]
NATIVE = REPO / "native/gdn-sglang-c143"
if str(NATIVE) not in sys.path:
    sys.path.insert(0, str(NATIVE))

from frozen_triton_c143_io import (  # noqa: E402
    GateError,
    decode_json,
    require_exact_keys,
    require_sha256,
    stable_read,
)
from frozen_triton_executor.contract import (  # noqa: E402
    PINNED,
    local_identity,
    sha256_file,
)


ADAPTER_SOURCE = NATIVE / "src/atlas_triton_adapters.cu"
ADAPTER_SOURCE_SHA256 = (
    "dc7a507e3279606827ced026d4a53d2fcfbdc661921f7d11d36d69ca6216f6e2"
)
ADAPTER_LIBRARY = NATIVE / "build/libatlas_gdn_c143_fused_input_sm121.so"
ADAPTER_LIBRARY_SHA256 = "UNRELEASED_REQUIRES_GDN_C143_FUSED_INPUT_SO"
ADAPTER_LIBRARY_BYTES = 0
BASE_EXECUTOR_SOURCE_SHA256 = (
    "34f627c202bfe2b484fe1a7b8330c35220d6fec895ea4896e33bd430ae1bb2e5"
)
VARIANT_FILES = (
    "run_frozen_triton_c143_fused_nozero.py",
    "frozen_triton_fused_nozero/__init__.py",
    "frozen_triton_fused_nozero/adapter.py",
    "frozen_triton_fused_nozero/contract.py",
    "frozen_triton_fused_nozero/case.py",
    "frozen_triton_fused_nozero/main.py",
    "frozen_triton_fused_nozero/scheduler_authority.py",
    "frozen_triton_fused_nozero/scheduler_trust.py",
    "test_frozen_triton_c143_fused_nozero_static.py",
    "test_frozen_triton_c143_fused_nozero_contract.py",
    "test_frozen_triton_c143_fused_nozero_scheduler.py",
)
IDENTITY_KEYS = {
    "manifest_sha256",
    "artifact_attestation_sha256",
    "gate_source_sha256",
    "base_executor_source_sha256",
    "variant_source_sha256",
    "adapter_source_sha256",
    "adapter_library_sha256",
    "python_executable_sha256",
    *PINNED,
}
CONTRACT_KEYS = {
    "schema",
    "m_sequence",
    *IDENTITY_KEYS,
    "nonce_sha256",
    "result_path",
}


def variant_source_sha256() -> str:
    records = {}
    for name in VARIANT_FILES:
        raw, digest, _ = stable_read(HERE / name, f"variant source:{name}")
        records[name] = [digest, len(raw)]
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def source_identity(attestation: dict[str, Any], gate_sha256: str) -> dict[str, Any]:
    base = local_identity(attestation, gate_sha256)
    if base["executor_source_sha256"] != BASE_EXECUTOR_SOURCE_SHA256:
        raise GateError("approved nozero/base executor source identity drift")
    raw, digest, _ = stable_read(ADAPTER_SOURCE, "fused input adapter source")
    if digest != ADAPTER_SOURCE_SHA256 or not raw:
        raise GateError("fused input adapter source identity drift")
    base.pop("executor_source_sha256")
    return {
        **base,
        "base_executor_source_sha256": BASE_EXECUTOR_SOURCE_SHA256,
        "variant_source_sha256": variant_source_sha256(),
        "adapter_source_sha256": ADAPTER_SOURCE_SHA256,
        "adapter_library_sha256": ADAPTER_LIBRARY_SHA256,
    }


def runtime_identity(attestation: dict[str, Any], gate_sha256: str) -> dict[str, Any]:
    require_sha256(ADAPTER_LIBRARY_SHA256, "adapter library SHA256")
    if ADAPTER_LIBRARY_BYTES <= 0:
        raise GateError("adapter library byte size is unreleased")
    identity = source_identity(attestation, gate_sha256)
    _, digest, _ = stable_read(
        ADAPTER_LIBRARY,
        "fused input adapter library",
        expected_size=ADAPTER_LIBRARY_BYTES,
        maximum_size=1 << 20,
    )
    if digest != ADAPTER_LIBRARY_SHA256:
        raise GateError("fused input adapter library identity drift")
    return identity


def cpu_attestation(attestation: dict[str, Any], gate_sha256: str) -> dict[str, Any]:
    identity = source_identity(attestation, gate_sha256)
    return {
        "schema": "atlas.gdn_c143.fused_nozero_attestation.v1",
        "qualification": "ATTESTED_CPU_ONLY",
        "variant": "fused-input-plus-no-output-clear",
        "gpu_execution": False,
        "production_authorized": False,
        "default_off": True,
        "m_sequence": [2_079, 8_192],
        **{key: identity[key] for key in IDENTITY_KEYS},
    }


def expected_contract(
    identity: dict[str, Any], nonce: str, result: Path
) -> dict[str, Any]:
    return {
        "schema": "atlas.gdn_c143.fused_nozero_execution_contract.v2",
        "m_sequence": [2_079, 8_192],
        **{key: identity[key] for key in IDENTITY_KEYS},
        "nonce_sha256": hashlib.sha256(nonce.encode("ascii")).hexdigest(),
        "result_path": str(result),
    }


def validate_contract(
    path: Path,
    identity: dict[str, Any],
    nonce: str,
) -> tuple[dict[str, Any], str]:
    raw, digest, stat = stable_read(
        path, "fused-nozero contract", maximum_size=64 << 10
    )
    if stat.st_mode & 0o022:
        raise GateError("fused-nozero contract is group/world writable")
    value = require_exact_keys(
        decode_json(raw, "fused-nozero contract"),
        CONTRACT_KEYS,
        "fused-nozero contract",
    )
    if type(value["result_path"]) is not str:
        raise GateError("fused-nozero result path must be a string")
    expected = expected_contract(identity, nonce, Path(value["result_path"]))
    if value != expected:
        raise GateError("fused-nozero contract identity or shape drift")
    result = Path(value["result_path"])
    if not result.is_absolute() or result.exists() or result.is_symlink():
        raise GateError("fused-nozero result path must be fresh absolute and absent")
    parent = result.parent.resolve(strict=True)
    if parent != result.parent or os.stat(parent).st_mode & 0o022:
        raise GateError("fused-nozero result directory is not canonical/private")
    for key in CONTRACT_KEYS - {"schema", "m_sequence", "result_path"}:
        require_sha256(value[key], f"fused-nozero contract.{key}")
    return value, digest


def recheck_file(path: Path, expected: str, label: str) -> None:
    if sha256_file(path.resolve(strict=True), label) != expected:
        raise GateError(f"{label} drift")
