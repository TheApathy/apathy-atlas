# SPDX-License-Identifier: AGPL-3.0-only
"""CUDA-hidden hostile source gates for the no-output-clear candidate."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from frozen_triton_executor import contract


HERE = Path(__file__).resolve().parent


class NoZeroStaticTests(unittest.TestCase):
    def test_candidate_sources_are_bound_spdx_capped_and_parse(self) -> None:
        names = (
            "frozen_triton_c143_output_overwrite.py",
            "run_frozen_triton_c143_nozero.py",
            "frozen_triton_nozero/__init__.py",
            "frozen_triton_nozero/launch.py",
            "frozen_triton_nozero/case.py",
            "frozen_triton_nozero/main.py",
        )
        self.assertTrue(set(names).issubset(contract.EXECUTOR_FILES))
        for name in names:
            source = (HERE / name).read_text()
            self.assertEqual(
                source.splitlines()[0], "# SPDX-License-Identifier: AGPL-3.0-only"
            )
            self.assertLessEqual(len(source.splitlines()), 250, name)
            compile(source, name, "exec")

    def test_default_attestation_is_cuda_hidden_and_default_off(self) -> None:
        environment = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        result = subprocess.run(
            [sys.executable, "-B", str(HERE / "run_frozen_triton_c143_nozero.py")],
            cwd=HERE,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        value = json.loads(result.stdout)
        self.assertEqual(value["qualification"], "ATTESTED_CPU_ONLY")
        self.assertEqual(value["variant"], "sole-delta-skip-output-clear")
        self.assertFalse(value["gpu_execution"])
        self.assertFalse(value["production_authorized"])
        self.assertTrue(value["default_off"])

    def test_double_environment_gate_precedes_torch_import(self) -> None:
        main = (HERE / "frozen_triton_nozero/main.py").read_text()
        self.assertLess(
            main.index("ATLAS_GDN_C143_NOZERO_RAW_GPU"), main.index("import torch")
        )
        self.assertLess(
            main.index("authorize_gpu_execution"), main.index("import torch")
        )
        self.assertIn(
            "ATLAS_GDN_C143_TRITON_RAW_GPU",
            (HERE / "frozen_triton_c143_authorization.py").read_text(),
        )

    def test_missing_candidate_environment_fails_cuda_hidden(self) -> None:
        environment = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        environment.pop("ATLAS_GDN_C143_NOZERO_RAW_GPU", None)
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                str(HERE / "run_frozen_triton_c143_nozero.py"),
                "--execute-gpu",
            ],
            cwd=HERE,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertIn("FAIL: nozero GPU environment gate", result.stderr)
        self.assertNotIn("PASS", result.stderr)

    def test_sole_delta_retains_A_clear_and_exact_five_launches(self) -> None:
        source = (HERE / "frozen_triton_nozero/launch.py").read_text()
        self.assertIn('v["A_bf16"].zero_()', source)
        self.assertNotIn("output.zero_", source)
        self.assertEqual(source.count("oracle._launch("), 5)
        roles = (
            "cumsum kkt_bc16_solve recompute_w_u chunk_recurrence_state output".split()
        )
        positions = [source.index(f'"{role}"') for role in roles]
        self.assertEqual(positions, sorted(positions))
        self.assertEqual(source.count("(U64, 0)"), 10)

    def test_two_hostile_seeds_and_all_strict_timing_predicates(self) -> None:
        source = (HERE / "frozen_triton_nozero/case.py").read_text()
        self.assertIn("for seed in (0x6A, 0xC3)", source)
        self.assertIn(
            '"output_seed_before_kernel"',
            (HERE / "frozen_triton_nozero/launch.py").read_text(),
        )
        for name in (
            "absolute",
            "atlas_median",
            "atlas_p90",
            "paired_atlas",
            "sglang_ratio",
            "zero_median",
            "zero_p90",
            "paired_zero",
        ):
            self.assertIn(f'"{name}"', source)
        self.assertIn("range(24)", source)
        self.assertIn("count != 6", source)
        self.assertIn("enable_timing=True", source)
        self.assertIn(
            "for m in (2_079, 8_192)",
            (HERE / "frozen_triton_nozero/main.py").read_text(),
        )

    def test_exclusive_publication_follows_proof_cases_and_rechecks(self) -> None:
        source = (HERE / "frozen_triton_nozero/main.py").read_text()
        proof = source.index("proofs =")
        cases = source.index("cases =")
        recheck = source.index("post_gate_sha =")
        publish = source.index("publish_exclusive(result_path, receipt)")
        self.assertLess(proof, cases)
        self.assertLess(cases, recheck)
        self.assertLess(recheck, publish)
        self.assertIn("post_attestation = sealed_gate.attest_manifest()", source)
        self.assertIn('"full_write_artifact_sha256": EXPECTED_SHA256', source)
        self.assertIn('"execution_contract": contract', source)
        self.assertIn('"qualification": "PASS"', source[publish - 1000 : publish])


if __name__ == "__main__":
    unittest.main()
