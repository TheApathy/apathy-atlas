# SPDX-License-Identifier: AGPL-3.0-only
"""Hostile CPU-only contracts for the frozen-cubin executor."""

import contextlib
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import frozen_triton_c143_gate as sealed_gate
from frozen_triton_c143_io import GateError
from frozen_triton_executor import contract, main, verified_loader

HERE = Path(__file__).resolve().parent
RECEIPT_TEST = HERE / "test_frozen_triton_c143_executor_receipt.py"


class FrozenTritonExecutorStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.attestation = sealed_gate.attest_manifest()
        cls.gate_sha = sealed_gate._gate_source_sha256()

    def test_all_new_sources_are_spdx_capped_and_parse(self) -> None:
        paths = [HERE / name for name in contract.EXECUTOR_FILES]
        paths.extend([Path(__file__), RECEIPT_TEST])
        for path in paths:
            source = path.read_text(encoding="utf-8")
            self.assertEqual(
                source.splitlines()[0], "# SPDX-License-Identifier: AGPL-3.0-only"
            )
            self.assertLessEqual(len(source.splitlines()), 250, path)
            compile(source, str(path), "exec")

    def test_default_attestation_is_cuda_hidden_and_nonexecuting(self) -> None:
        environment = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        result = subprocess.run(
            [sys.executable, "-B", str(HERE / "run_frozen_triton_c143.py")],
            cwd=HERE,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        value = json.loads(result.stdout)
        self.assertEqual(value["qualification"], "ATTESTED_CPU_ONLY")
        self.assertFalse(value["gpu_execution"])
        self.assertFalse(value["production_authorized"])
        self.assertEqual(value["m_sequence"], [2_079, 8_192])

    def test_gpu_arguments_and_missing_authorization_fail_without_pass(self) -> None:
        for argv in (["--reservation-nonce", "a" * 64], ["--execute-gpu"]):
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                self.assertEqual(main.main(argv), 2)
            self.assertEqual(stdout.getvalue(), "")
            self.assertIn("FAIL:", stderr.getvalue())
            self.assertNotIn("PASS", stderr.getvalue() + stdout.getvalue())

    def test_execution_contract_binds_every_runtime_identity(self) -> None:
        nonce = "0123456789abcdef" * 4
        authorization = {
            "reservation_receipt_sha256": hashlib.sha256(b"reservation").hexdigest()
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path = root / "result.json"
            value = contract.contract_template(
                self.attestation,
                self.gate_sha,
                reservation_sha256=authorization["reservation_receipt_sha256"],
                nonce=nonce,
                result_path=result_path,
            )
            path = root / "contract.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            path.chmod(0o600)
            parsed, _ = contract.validate_contract(
                path.resolve(), self.attestation, self.gate_sha, authorization, nonce
            )
            self.assertEqual(parsed, value)
            keys = "manifest_sha256 artifact_attestation_sha256 executor_source_sha256 python_executable_sha256 atlas_bridge_sha256 reservation_receipt_sha256 nonce_sha256"
            for key in keys.split():
                hostile = dict(value)
                hostile[key] = "0" * 64
                path.write_text(json.dumps(hostile), encoding="utf-8")
                with self.subTest(key=key), self.assertRaises(GateError):
                    contract.validate_contract(
                        path.resolve(),
                        self.attestation,
                        self.gate_sha,
                        authorization,
                        nonce,
                    )

    def test_driver_and_launch_sources_pin_external_cubin_abi(self) -> None:
        driver = (HERE / "frozen_triton_executor/cuda_driver.py").read_text()
        launch = (HERE / "frozen_triton_executor/launch.py").read_text()
        self.assertIn("cuModuleLoadData", driver)
        self.assertNotIn("self.lib.cuModuleLoad(", driver)
        self.assertIn("create_string_buffer(raw, len(raw))", driver)
        self.assertGreaterEqual(driver.count("sha256(storage.raw)"), 2)
        self.assertIn('contract["files"]["cubin"]', driver)
        self.assertRegex(driver, r"cuLaunchKernel[\s\S]*\(12, 1, 48\)")
        self.assertIn("- TRITON_RESERVED_SHARED_BYTES", driver)
        self.assertIn("actual={actual.value}", driver)
        roles = [kernel["role"] for kernel in self.attestation["manifest"]["kernels"]]
        expected = "cumsum kkt_bc16_solve recompute_w_u chunk_recurrence_state output"
        self.assertEqual(roles, expected.split())
        positions = [launch.index(f'"{role}"') for role in roles]
        self.assertEqual(positions, sorted(positions))
        self.assertGreaterEqual(launch.count("(U64, 0)"), 10)
        for m in sealed_gate.SUPPORTED_M:
            for item in sealed_gate.resolved_launch_plan(self.attestation, m):
                self.assertEqual(item["trailing_scratch_values_u64"], [0, 0])
        expected_pointers = {
            "cumsum": [
                "log_gate_f32",
                "g_cumsum_f32",
                "cu_seqlens_i32",
                "chunk_indices_i32",
            ],
            "kkt_bc16_solve": [
                "k_bf16",
                "g_cumsum_f32",
                "beta_f32",
                "A_bf16",
                "cu_seqlens_i32",
                "chunk_indices_i32",
            ],
            "recompute_w_u": [
                "k_bf16",
                "v_bf16",
                "beta_f32",
                "w_bf16",
                "u_bf16",
                "A_bf16",
                "g_cumsum_f32",
                "cu_seqlens_i32",
                "chunk_indices_i32",
            ],
            "chunk_recurrence_state": [
                "k_bf16",
                "u_bf16",
                "w_bf16",
                "v_new_bf16",
                "g_cumsum_f32",
                "h_bf16",
                "state_hvk_f32",
                "state_index_i32",
                "cu_seqlens_i32",
                "chunk_offsets_i64",
            ],
            "output": [
                "q_bf16",
                "k_bf16",
                "v_new_bf16",
                "h_bf16",
                "g_cumsum_f32",
                "output",
                "cu_seqlens_i32",
                "chunk_indices_i32",
            ],
        }
        for role, expected_names in expected_pointers.items():
            block = launch.split(f'self._launch(\n                "{role}",', 1)[1]
            block = block.split("\n                ],\n            ),", 1)[0]
            matches = re.findall(
                r'device_pointer\((?:v\["([^"]+)"\]|(output))\)', block
            )
            self.assertEqual([left or right for left, right in matches], expected_names)
        self.assertIn("(U32, 786_432)", launch)
        self.assertIn("(F32, 0.08838834764831845)", launch)

    def test_verified_load_rejects_swap_and_memfd_survives_source_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = (Path(directory) / "image.bin").resolve()
            original = b"sealed-original-image"
            source.write_bytes(original)
            digest = hashlib.sha256(original).hexdigest()
            captured = []

            def swap(raw: bytes) -> None:
                captured.append(raw)
                replacement = source.with_suffix(".replacement")
                replacement.write_bytes(original)
                os.replace(replacement, source)

            with self.assertRaises(GateError):
                verified_loader.load_verified_file(
                    source, "hostile", digest, len(original), swap
                )
            self.assertEqual(captured, [original])
            source.write_bytes(original)
            image = verified_loader.SealedMemfd.from_file(
                source, digest, len(original), "atlas-test-seal"
            )
            try:
                source.write_bytes(b"source-now-hostile")
                self.assertEqual(image.verify(), digest)
                with self.assertRaises(OSError):
                    os.pwrite(image.fd, b"x", 0)
            finally:
                image.close()
        references = (HERE / "frozen_triton_executor/references.py").read_text()
        self.assertNotIn("AtlasBridge(ATLAS_LIBRARY)", references)
        self.assertLess(
            references.index("SealedMemfd.from_file"),
            references.index("AtlasBridge(self.atlas_image.path)"),
        )
        self.assertIn("verify_executable_mapping()", references)

    def test_workspace_contract_is_exact_disjoint_and_guarded(self) -> None:
        source = (HERE / "frozen_triton_executor/buffers.py").read_text()
        self.assertIn("CANARY_BYTES = 4096", source)
        self.assertIn("validate_runtime_extents", source)
        for m, total in ((2_079, 188_241_152), (8_192, 729_286_400)):
            layout = sealed_gate.workspace_layout(m)
            self.assertEqual(layout["total_bytes"], total)
            intervals = [(o, o + n) for _, o, n in layout["segments"]]
            self.assertTrue(
                all(
                    left[1] <= right[0] for left, right in zip(intervals, intervals[1:])
                )
            )

    def test_authorization_precedes_gpu_import_and_publication(self) -> None:
        source = (HERE / "frozen_triton_executor/main.py").read_text()
        self.assertLess(
            source.index("authorize_gpu_execution"), source.index("import torch")
        )
        self.assertLess(
            source.index("build_envelope"), source.index("publish_exclusive(")
        )
        self.assertIn("for m in (2_079, 8_192)", source)


if __name__ == "__main__":
    unittest.main()
