# SPDX-License-Identifier: AGPL-3.0-only
"""CUDA-hidden source gates for the fused-input plus nozero candidate."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
ENTRY = HERE / "run_frozen_triton_c143_fused_nozero.py"
PACKAGE = HERE / "frozen_triton_fused_nozero"
FILES = (
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


class FusedNozeroStaticTests(unittest.TestCase):
    def test_complete_new_bundle_is_spdx_capped_and_parses(self) -> None:
        for name in FILES:
            source = (HERE / name).read_text()
            self.assertEqual(
                source.splitlines()[0], "# SPDX-License-Identifier: AGPL-3.0-only"
            )
            self.assertLessEqual(len(source.splitlines()), 250, name)
            self.assertNotRegex(source, r"(?m)[ \t]+$")
            compile(source, name, "exec")
        contract = (PACKAGE / "contract.py").read_text()
        for name in FILES:
            self.assertIn(f'"{name}"', contract)

    def test_default_attestation_is_cuda_hidden_and_inert(self) -> None:
        environment = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        result = subprocess.run(
            [sys.executable, "-B", str(ENTRY)],
            cwd=HERE,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        value = json.loads(result.stdout)
        self.assertEqual(value["qualification"], "ATTESTED_CPU_ONLY")
        self.assertEqual(value["variant"], "fused-input-plus-no-output-clear")
        self.assertFalse(value["gpu_execution"])
        self.assertFalse(value["production_authorized"])
        self.assertTrue(value["default_off"])
        self.assertTrue(value["adapter_library_sha256"].startswith("UNRELEASED_"))

    def test_double_environment_and_nonhex_stop_before_gpu_or_paths(self) -> None:
        environment = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        result = subprocess.run(
            [sys.executable, "-B", str(ENTRY), "--execute-gpu"],
            cwd=HERE,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertEqual(result.stdout, "")
        self.assertIn("fused-nozero GPU environment gate", result.stderr)
        environment.update(
            ATLAS_GDN_C143_TRITON_RAW_GPU="1",
            ATLAS_GDN_C143_FUSED_NOZERO_RAW_GPU="1",
        )
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                str(ENTRY),
                "--execute-gpu",
                "--gpu-execution-flag",
                "x",
                "--reservation-nonce",
                "x",
                "--reservation",
                "/definitely/absent-reservation",
                "--execution-contract",
                "/definitely/absent-contract",
            ],
            cwd=HERE,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("adapter library SHA256", result.stderr)
        self.assertNotIn("absent-reservation", result.stderr)

    def test_fused_adapter_replaces_only_three_parent_hooks(self) -> None:
        adapter = (PACKAGE / "adapter.py").read_text()
        self.assertEqual(adapter.count("self._function("), 1)
        self.assertIn("except AttributeError as error:", adapter)
        self.assertIn("fused input adapter symbol missing", adapter)
        self.assertIn("class FusedBuffers(FrozenBuffers):", adapter)
        self.assertIn("def adapter_qkv(self)", adapter)
        self.assertIn("def adapter_gate(self)", adapter)
        self.assertIn("def adapter_state_in(self)", adapter)
        self.assertIn(
            "run_nozero(oracle, buffers, stream", (PACKAGE / "case.py").read_text()
        )
        for proof in (
            '"q_split_bytes_exact"',
            '"k_split_bytes_exact"',
            '"v_split_bytes_exact"',
            '"beta_split_bytes_exact"',
            '"state_in_transpose_exact"',
            '"log_gate_bytes_exact"',
        ):
            self.assertIn(proof, adapter)

    def test_exact_parent_parity_boundaries_and_final_identity(self) -> None:
        case = (PACKAGE / "case.py").read_text()
        for value in ('float("nan")', 'float("inf")', '-float("inf")', "-0.0"):
            self.assertIn(value, case)
        self.assertIn("candidate_ids[0] == candidate_ids[1] == parent_id", case)
        self.assertIn("candidate_before == candidate_after", case)
        self.assertIn("parent_before == parent_after", case)
        self.assertIn('"atlas_gate_beta_f32": tensor_sha256(', case)
        self.assertIn("set(checks) != {", case)
        self.assertIn('"output_seed_before_kernel"', case)
        self.assertIn('"state_out_transpose_exact"', case)
        self.assertIn("all_canaries_clean()", case)
        self.assertIn("metric_pair", case)

    def test_balanced_five_arm_timing_and_all_strict_predicates(self) -> None:
        case = (PACKAGE / "case.py").read_text()
        for arm in ("fused", "nozero", "zero", "atlas", "sglang"):
            self.assertIn(f'"{arm}"', case)
        self.assertIn("range(25)", case)
        self.assertIn("count != 5", case)
        for predicate in (
            "absolute",
            "nozero_median",
            "nozero_p90",
            "paired_nozero",
            "zero_median",
            "zero_p90",
            "paired_zero",
            "atlas_median",
            "atlas_p90",
            "paired_atlas",
            "sglang_ratio",
        ):
            self.assertIn(f'"{predicate}"', case)

    def test_m2079_precedes_m8192_and_publication_follows_rechecks(self) -> None:
        main = (PACKAGE / "main.py").read_text()
        self.assertIn("for m in (2_079, 8_192)", main)
        cases = main.index("cases =")
        post = main.index("post_attestation =")
        publish = main.index("publish_exclusive(")
        self.assertLess(cases, post)
        self.assertLess(post, publish)
        self.assertIn('"qualification": "PASS"', main[post:publish])
        self.assertIn('if key not in {"schema", "result_path"}', main)
        self.assertNotIn('if key != "result_path"', main)


if __name__ == "__main__":
    unittest.main()
