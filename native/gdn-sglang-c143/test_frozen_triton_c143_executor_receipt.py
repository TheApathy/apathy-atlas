# SPDX-License-Identifier: AGPL-3.0-only
"""Hostile CPU-only tests for executor receipt aggregation/publication."""

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import frozen_triton_c143_gate as sealed_gate
from frozen_triton_c143_io import GateError
from frozen_triton_c143_test_support import valid_receipt
from frozen_triton_executor import contract, evidence, publication, receipt


def digest(kind: str, label: str) -> str:
    return hashlib.sha256(f"{kind}:{label}".encode()).hexdigest()


def fixture_evidence(kind: str) -> dict:
    identity = {
        "output_sha256": digest(kind, "output"),
        "state_sha256": digest(kind, "state"),
        "guards_clean": True,
        "finite": True,
    }
    return {
        "kind": kind,
        "input_before": {
            name: digest(kind, f"input:{name}") for name in sealed_gate.IMMUTABLE_NAMES
        },
        "input_after": {
            name: digest(kind, f"input:{name}") for name in sealed_gate.IMMUTABLE_NAMES
        },
        "stage_run1": {
            name: digest(kind, f"stage:{name}") for name in sealed_gate.STAGE_NAMES
        },
        "stage_run2": {
            name: digest(kind, f"stage:{name}") for name in sealed_gate.STAGE_NAMES
        },
        "adapter_checks": {name: True for name in sealed_gate.ADAPTER_CHECKS},
        "finite": {name: True for name in sealed_gate.FINITE_NAMES},
        "identities": {
            f"{arm}_run{run}": dict(identity)
            for arm in ("candidate", "atlas", "sglang")
            for run in (1, 2)
        },
        "comparisons": {
            reference: {
                field: {"max_abs": 0.0, "rms": 0.0, "relative_rms": 0.0, "cosine": 1.0}
                for field in ("output", "state")
            }
            for reference in ("candidate_vs_sglang", "candidate_vs_atlas")
        },
        "deterministic": True,
        "all_guards_clean": True,
    }


def case_from_valid(attestation: dict, m: int) -> dict:
    raw = valid_receipt(attestation, m)
    fixtures = [fixture_evidence("real"), fixture_evidence("adversarial")]
    return {
        "m": m,
        "fixtures": fixtures,
        **evidence.aggregate_fixtures(fixtures),
        "canaries_clean": True,
        "stage_timing_ms": raw["stage_timing_ms"],
        "full_timing": raw["full_timing"],
    }


class FrozenTritonExecutorReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.attestation = sealed_gate.attest_manifest()
        cls.gate_sha = sealed_gate._gate_source_sha256()

    def test_receipt_rejects_device_order_hash_fixture_and_aggregate_drift(
        self,
    ) -> None:
        nonce = "fedcba9876543210" * 4
        authorization = {
            "reservation_receipt_sha256": hashlib.sha256(b"reservation").hexdigest()
        }
        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory) / "result.json"
            plan = contract.contract_template(
                self.attestation,
                self.gate_sha,
                reservation_sha256=authorization["reservation_receipt_sha256"],
                nonce=nonce,
                result_path=result_path,
            )
            contract_sha = hashlib.sha256(b"contract").hexdigest()
            device = {
                "ordinal": 0,
                "name": "NVIDIA GB10",
                "major": 12,
                "minor": 1,
                "multiprocessor_count": 48,
            }
            envelope = receipt.build_envelope(
                [case_from_valid(self.attestation, m) for m in (2_079, 8_192)],
                self.attestation,
                plan,
                contract_sha,
                authorization,
                device,
                0x1234,
            )
            publication.validate_envelope(
                envelope, self.attestation, plan, contract_sha
            )
            mutations = []
            for path, value in (
                (("device", "multiprocessor_count"), 47),
                (("cases", 0, "raw_receipt_sha256"), "0" * 64),
                (("cases", 0, "fixtures", 1, "deterministic"), False),
                (("cases", 0, "fixtures", 0, "stage_run1", "A"), "1" * 64),
            ):
                hostile = copy.deepcopy(envelope)
                target = hostile
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                mutations.append(hostile)
            wrong_order = copy.deepcopy(envelope)
            wrong_order["cases"].reverse()
            mutations.append(wrong_order)
            for index, hostile in enumerate(mutations):
                with self.subTest(index=index), self.assertRaises(GateError):
                    publication.validate_envelope(
                        hostile, self.attestation, plan, contract_sha
                    )
            publication.publish_exclusive(result_path, envelope)
            self.assertEqual(result_path.stat().st_mode & 0o777, 0o444)
            with self.assertRaises(FileExistsError):
                publication.publish_exclusive(result_path, envelope)
            broken_path = Path(directory) / "broken.json"
            with (
                mock.patch.object(
                    publication.os, "write", side_effect=OSError("hostile short write")
                ),
                self.assertRaises(OSError),
            ):
                publication.publish_exclusive(broken_path, envelope)
            self.assertFalse(broken_path.exists())


if __name__ == "__main__":
    unittest.main()
