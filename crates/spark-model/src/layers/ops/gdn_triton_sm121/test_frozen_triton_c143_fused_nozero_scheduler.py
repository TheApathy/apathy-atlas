# SPDX-License-Identifier: AGPL-3.0-only
"""Dependency-free scheduler authority and source-rebinding hostiles."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from frozen_triton_fused_nozero import contract as _contract
from frozen_triton_c143_io import GateError
from frozen_triton_fused_nozero.scheduler_authority import (
    bind_scheduler_manifest,
    bind_variant_reservation,
)
from frozen_triton_fused_nozero.scheduler_trust import HeldRootFile


H = "1" * 64
IDENTITY = {
    "variant_source_sha256": "2" * 64,
    "adapter_source_sha256": "3" * 64,
    "adapter_library_sha256": "4" * 64,
    "adapter_library_bytes": 123,
    "base_executor_source_sha256": "5" * 64,
    "python_executable": "/usr/bin/python3.12",
    "python_executable_sha256": "6" * 64,
    "manifest_sha256": "7" * 64,
    "gate_source_sha256": "8" * 64,
}
CONTRACT_SHA = "9" * 64


def scheduler() -> dict:
    return {
        "schema": "atlas.gdn_c143.fused_nozero_scheduler.v1",
        "variant_source_sha256": IDENTITY["variant_source_sha256"],
        "adapter_source_sha256": IDENTITY["adapter_source_sha256"],
        "adapter_library_sha256": IDENTITY["adapter_library_sha256"],
        "adapter_library_bytes": IDENTITY["adapter_library_bytes"],
        "base_executor_source_sha256": IDENTITY["base_executor_source_sha256"],
        "python_executable": IDENTITY["python_executable"],
        "python_executable_sha256": IDENTITY["python_executable_sha256"],
        "base_manifest_sha256": IDENTITY["manifest_sha256"],
        "base_gate_sha256": IDENTITY["gate_source_sha256"],
        "execution_contract_sha256": CONTRACT_SHA,
        "build_receipt_path": "/run/atlas/gdn-build.json",
        "build_receipt_sha256": "a" * 64,
    }


def reservation(nonce: str, scheduler_sha: str) -> dict:
    return {
        "schema": "atlas.gpu.variant_reservation.v1",
        "claim": "qwen38-gdn-c143-fused-nozero-raw-gpu",
        "owner": "/root/gdn-owner",
        "scope": "local-gb10-cuda-execution",
        "authorization": "GPU_EXECUTION_APPROVED",
        "released": False,
        "nonce_sha256": hashlib.sha256(nonce.encode("ascii")).hexdigest(),
        "base_manifest_sha256": IDENTITY["manifest_sha256"],
        "base_gate_sha256": IDENTITY["gate_source_sha256"],
        "scheduler_manifest_sha256": scheduler_sha,
        "expires_utc": "2026-08-28T11:00:00Z",
    }


class SchedulerAuthorityTests(unittest.TestCase):
    def test_scheduler_binds_every_external_identity_with_exact_types(self) -> None:
        good = scheduler()
        self.assertEqual(bind_scheduler_manifest(good, IDENTITY, CONTRACT_SHA), good)
        for key in (
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
        ):
            bad = dict(good)
            bad[key] = True if key == "adapter_library_bytes" else H
            with self.assertRaises(GateError, msg=key):
                bind_scheduler_manifest(bad, IDENTITY, CONTRACT_SHA)
        for key in ("build_receipt_sha256", "build_receipt_path"):
            bad = dict(good)
            bad[key] = 7
            with self.assertRaises(GateError, msg=key):
                bind_scheduler_manifest(bad, IDENTITY, CONTRACT_SHA)

    def test_variant_reservation_binds_scheduler_nonce_base_and_expiry(self) -> None:
        nonce, scheduler_sha = "b" * 64, "c" * 64
        now = dt.datetime(2026, 8, 28, 10, tzinfo=dt.timezone.utc)
        good = reservation(nonce, scheduler_sha)
        self.assertEqual(
            bind_variant_reservation(good, IDENTITY, scheduler_sha, nonce, now),
            "/root/gdn-owner",
        )
        for key in (
            "claim",
            "owner",
            "nonce_sha256",
            "base_manifest_sha256",
            "base_gate_sha256",
            "scheduler_manifest_sha256",
            "expires_utc",
        ):
            bad = dict(good)
            bad[key] = "wrong"
            with self.assertRaises(GateError, msg=key):
                bind_variant_reservation(bad, IDENTITY, scheduler_sha, nonce, now)
        bad = dict(good)
        bad["released"] = 0
        with self.assertRaises(GateError):
            bind_variant_reservation(bad, IDENTITY, scheduler_sha, nonce, now)
        bad = dict(good)
        bad["owner"] = "/root-impersonator"
        with self.assertRaises(GateError):
            bind_variant_reservation(bad, IDENTITY, scheduler_sha, nonce, now)

    def test_appended_module_rebinding_changes_externally_pinned_bundle(self) -> None:
        def digest(overrides: dict[str, bytes]) -> str:
            records = {}
            for name in _contract.VARIANT_FILES:
                raw = overrides.get(name, (_contract.HERE / name).read_bytes())
                records[name] = [hashlib.sha256(raw).hexdigest(), len(raw)]
            wire = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
            return hashlib.sha256(wire).hexdigest()

        approved = digest({})
        self.assertEqual(approved, _contract.variant_source_sha256())
        for name, suffix in (
            (
                "frozen_triton_fused_nozero/adapter.py",
                b"\nFusedBuffers.adapter_checks=lambda self:{}\n",
            ),
            (
                "frozen_triton_fused_nozero/case.py",
                b"\ntiming=lambda *args:({}, {})\n",
            ),
        ):
            mutated = (_contract.HERE / name).read_bytes() + suffix
            mutant_digest = digest({name: mutated})
            self.assertNotEqual(mutant_digest, approved)
            identity = {**IDENTITY, "variant_source_sha256": mutant_digest}
            with self.assertRaises(GateError):
                bind_scheduler_manifest(scheduler(), identity, CONTRACT_SHA)

    def test_user_owned_mutable_and_symlink_authority_files_reject(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            regular = root / "authority.json"
            regular.write_text("{}")
            with self.assertRaises((GateError, OSError)):
                HeldRootFile.open(regular, "hostile", {0o444})
            regular.chmod(0o444)
            with self.assertRaises((GateError, OSError)):
                HeldRootFile.open(regular, "hostile", {0o444})
            link = root / "authority-link.json"
            link.symlink_to(regular)
            with self.assertRaises((GateError, OSError)):
                HeldRootFile.open(link, "hostile", {0o444})

    def test_scheduler_authority_wraps_gpu_import_and_publication(self) -> None:
        main = (_contract.HERE / "frozen_triton_fused_nozero/main.py").read_text()
        contract = (
            _contract.HERE / "frozen_triton_fused_nozero/contract.py"
        ).read_text()
        authorize = main.index("authorization = authorize_gpu_execution(")
        execute = main.index("return _execute_authorized(")
        torch_import = main.index("    import torch")
        recheck = main.index("    authorization.recheck()")
        publish = main.index("    receipt_sha = publish_exclusive(")
        self.assertLess(authorize, execute)
        self.assertLess(execute, torch_import)
        self.assertLess(recheck, publish)
        self.assertIn("finally:\n        authorization.close()", main)
        self.assertNotIn("sealed_gate.authorize_gpu_execution", main)
        self.assertIn("fused_nozero_execution_contract.v2", contract)
        self.assertIn("fused_nozero_raw_receipt.v2", main)


if __name__ == "__main__":
    unittest.main()
