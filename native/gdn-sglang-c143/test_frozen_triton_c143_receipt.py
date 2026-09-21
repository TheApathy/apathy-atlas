#!/usr/bin/env python3
"""Hostile CPU-only receipt sequencing and qualification tests."""

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import frozen_triton_c143_gate as gate
from frozen_triton_c143_test_support import digest, valid_receipt


class FrozenTritonC143ReceiptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.attestation = gate.attest_manifest()

    def test_valid_m2079_and_sequenced_m8192_receipts_pass(self) -> None:
        first = gate.validate_raw_receipt(
            valid_receipt(self.attestation), self.attestation
        )
        second = gate.validate_raw_receipt(
            valid_receipt(self.attestation, 8_192),
            self.attestation,
            prior_m2079_receipt_sha256=digest("m2079-pass"),
        )
        self.assertEqual(first["qualification"], "PASS")
        self.assertEqual(second["qualification"], "PASS")
        self.assertFalse(first["production_authorized"])

    def test_cli_m8192_validation_binds_actual_m2079_receipt_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "m2079.json"
            first_raw = json.dumps(
                valid_receipt(self.attestation), sort_keys=True
            ).encode()
            first_path.write_bytes(first_raw)
            second = valid_receipt(self.attestation, 8_192)
            second["m2079_pass_receipt_sha256"] = hashlib.sha256(first_raw).hexdigest()
            second_path = Path(directory) / "m8192.json"
            second_path.write_text(json.dumps(second), encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(gate.main(["--validate-receipt", str(second_path)]), 2)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    gate.main(
                        [
                            "--validate-receipt",
                            str(second_path),
                            "--m2079-pass-receipt",
                            str(first_path),
                        ]
                    ),
                    0,
                )

    def test_receipt_rejects_nondeterminism_immutability_and_missing_stage(
        self,
    ) -> None:
        nondeterministic = valid_receipt(self.attestation)
        nondeterministic["stage_hashes"]["A"]["run2"] = digest("different")
        with self.assertRaisesRegex(gate.GateError, "nondeterministic stage"):
            gate.validate_raw_receipt(nondeterministic, self.attestation)
        mutated = valid_receipt(self.attestation)
        mutated["input_hashes_after"]["atlas_qkv_bf16"] = digest("mutated")
        with self.assertRaisesRegex(gate.GateError, "immutable input changed"):
            gate.validate_raw_receipt(mutated, self.attestation)
        missing = valid_receipt(self.attestation)
        del missing["stage_timing_ms"]["A"]
        with self.assertRaisesRegex(gate.GateError, "key set mismatch"):
            gate.validate_raw_receipt(missing, self.attestation)

    def test_receipt_rejects_alias_claim_quality_drift_and_timing_regression(
        self,
    ) -> None:
        alias = valid_receipt(self.attestation)
        alias["adapter_checks"]["all_extents_nonalias"] = False
        with self.assertRaisesRegex(gate.GateError, "adapter contract failed"):
            gate.validate_raw_receipt(alias, self.attestation)
        quality = valid_receipt(self.attestation)
        quality["comparisons"]["candidate_vs_sglang"]["state"]["cosine"] = 0.998
        with self.assertRaisesRegex(gate.GateError, "cosine below"):
            gate.validate_raw_receipt(quality, self.attestation)
        timing = valid_receipt(self.attestation)
        timing["full_timing"]["samples_ms"]["candidate"][-3:] = [4.0, 4.0, 4.0]
        with self.assertRaisesRegex(
            gate.GateError,
            "performance screen.*candidate_median_ms.*atlas_p90.*stage_medians_ms",
        ):
            gate.validate_raw_receipt(timing, self.attestation)

    def test_receipt_rejects_impossible_metric_domains(self) -> None:
        for key in ("max_abs", "rms", "relative_rms"):
            with self.subTest(key=key):
                receipt = valid_receipt(self.attestation)
                receipt["comparisons"]["candidate_vs_sglang"]["output"][key] = -1.0
                with self.assertRaisesRegex(gate.GateError, "must be nonnegative"):
                    gate.validate_raw_receipt(receipt, self.attestation)
        for cosine in (-1.001, 1.001):
            with self.subTest(cosine=cosine):
                receipt = valid_receipt(self.attestation)
                receipt["comparisons"]["candidate_vs_atlas"]["state"]["cosine"] = cosine
                with self.assertRaisesRegex(gate.GateError, "cosine outside"):
                    gate.validate_raw_receipt(receipt, self.attestation)

    def test_receipt_rejects_unbalanced_or_incomplete_timing_scope(self) -> None:
        unbalanced = valid_receipt(self.attestation)
        unbalanced["full_timing"]["position_counts"]["candidate"] = [22, 0, 0]
        with self.assertRaisesRegex(gate.GateError, "not balanced"):
            gate.validate_raw_receipt(unbalanced, self.attestation)
        incomplete = valid_receipt(self.attestation)
        incomplete["full_timing"]["scope_includes"].remove("both_state_transposes")
        with self.assertRaisesRegex(gate.GateError, "excludes required work"):
            gate.validate_raw_receipt(incomplete, self.attestation)

    def test_m8192_receipt_requires_prior_m2079_pass_hash(self) -> None:
        receipt = valid_receipt(self.attestation, 8_192)
        receipt["m2079_pass_receipt_sha256"] = None
        with self.assertRaisesRegex(gate.GateError, "lowercase SHA256"):
            gate.validate_raw_receipt(receipt, self.attestation)
        receipt["m2079_pass_receipt_sha256"] = digest("m2079-pass")
        with self.assertRaisesRegex(gate.GateError, "actual validated"):
            gate.validate_raw_receipt(receipt, self.attestation)
        with self.assertRaisesRegex(gate.GateError, "binding mismatch"):
            gate.validate_raw_receipt(
                receipt,
                self.attestation,
                prior_m2079_receipt_sha256=digest("different-prior"),
            )


if __name__ == "__main__":
    unittest.main()
