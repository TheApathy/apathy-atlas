"""Composition root for the frozen Triton artifact attestation."""

import hashlib
import json
from pathlib import Path
from typing import Any

from frozen_triton_c143_artifact import attest_kernels
from frozen_triton_c143_constants import DEFAULT_MANIFEST, EXPECTED_MANIFEST_SHA256
from frozen_triton_c143_io import GateError, decode_json, stable_read
from frozen_triton_c143_layout import validate_workspace_manifest
from frozen_triton_c143_manifest_policy import (
    validate_manifest_policy,
    validate_receipt_contract,
)
from frozen_triton_c143_provenance import attest_autotune, attest_provenance


def attest_manifest_unsealed(
    manifest_path: Path, *, expected_manifest_sha256: str | None
) -> dict[str, Any]:
    """Test-only structural validator; public entry always requires the sealed hash."""
    raw, manifest_sha, _ = stable_read(manifest_path, "manifest", maximum_size=1 << 20)
    if (
        expected_manifest_sha256 is not None
        and manifest_sha != expected_manifest_sha256
    ):
        raise GateError(
            "manifest: SHA256 mismatch; "
            f"expected={expected_manifest_sha256} actual={manifest_sha}"
        )
    manifest = validate_manifest_policy(decode_json(raw, "manifest"))
    attested: dict[str, dict[str, Any]] = {}
    attest_kernels(manifest["kernels"], attested)
    attest_autotune(manifest["autotune"], attested)
    attest_provenance(manifest["provenance"], attested)
    validate_workspace_manifest(manifest["workspace_layouts"])
    validate_receipt_contract(manifest["receipt_contract"])
    canonical = json.dumps(attested, sort_keys=True, separators=(",", ":")).encode()
    return {
        "manifest": manifest,
        "manifest_sha256": manifest_sha,
        "artifact_attestation_sha256": hashlib.sha256(canonical).hexdigest(),
        "attested_files": attested,
    }


def attest_manifest(manifest_path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    """Attest the single sealed manifest accepted by the command-line gate."""
    return attest_manifest_unsealed(
        manifest_path, expected_manifest_sha256=EXPECTED_MANIFEST_SHA256
    )
