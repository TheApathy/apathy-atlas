#!/usr/bin/env python3
"""Hostile CPU-only artifact, geometry, and authorization tests."""

import ast
import contextlib
import copy
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import frozen_triton_c143_gate as gate
from frozen_triton_c143_test_support import (
    MANIFEST,
    reservation,
    valid_buffers,
    write_manifest,
)


class FrozenTritonC143GateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.attestation = gate.attest_manifest()

    def test_exact_attestation_is_cpu_only_and_external(self) -> None:
        summary = gate._attestation_summary(self.attestation)
        self.assertEqual(summary["qualification"], "ATTESTED_CPU_ONLY")
        self.assertFalse(summary["gpu_execution"])
        self.assertFalse(summary["gpu_executor_implemented"])
        self.assertFalse(summary["production_authorized"])
        self.assertEqual(summary["attested_file_count"], 49)
        self.assertEqual(
            summary["workspace_bytes"], {"2079": 188_241_152, "8192": 729_286_400}
        )
        self.assertEqual(
            summary["launch_grids"],
            {
                "2079": [
                    [33, 48, 1],
                    [33, 48, 1],
                    [33, 48, 1],
                    [4, 48, 1],
                    [2, 33, 48],
                ],
                "8192": [
                    [128, 48, 1],
                    [128, 48, 1],
                    [128, 48, 1],
                    [4, 48, 1],
                    [2, 128, 48],
                ],
            },
        )
        for m in gate.SUPPORTED_M:
            for launch in gate.resolved_launch_plan(self.attestation, m):
                self.assertEqual(
                    launch["driver_params"][-2:],
                    [["global_scratch_null", "u64"], ["profile_scratch_null", "u64"]],
                )
                self.assertEqual(launch["trailing_scratch_values_u64"], [0, 0])

    def test_workspace_layouts_are_aligned_disjoint_and_exact(self) -> None:
        for m, expected in ((2_079, 188_241_152), (8_192, 729_286_400)):
            layout = gate.workspace_layout(m)
            self.assertEqual(layout["total_bytes"], expected)
            prior_end = 0
            for _, offset, size in layout["segments"]:
                self.assertEqual(offset % gate.WORKSPACE_ALIGNMENT, 0)
                self.assertGreaterEqual(offset, prior_end)
                prior_end = offset + size
            self.assertLessEqual(prior_end, expected)

    def test_unsupported_shape_and_non_int_shape_fail_closed(self) -> None:
        for hostile in (32_768, 2_080, True, "2079"):
            with self.assertRaises(gate.GateError):
                gate.workspace_layout(hostile)  # type: ignore[arg-type]

    def test_runtime_extents_accept_exact_and_reject_alias(self) -> None:
        buffers = valid_buffers(2_079)
        gate.validate_runtime_extents(2_079, buffers, stream=0x1234)
        hostile = copy.deepcopy(buffers)
        hostile["output_bf16"]["address"] = hostile["atlas_qkv_bf16"]["address"]
        with self.assertRaisesRegex(gate.GateError, "runtime alias"):
            gate.validate_runtime_extents(2_079, hostile, stream=0x1234)

    def test_runtime_extents_reject_short_unaligned_default_stream_and_overflow(
        self,
    ) -> None:
        short = valid_buffers(2_079)
        short["workspace"]["bytes"] -= 1
        with self.assertRaises(gate.GateError):
            gate.validate_runtime_extents(2_079, short, stream=1)
        unaligned = valid_buffers(2_079)
        unaligned["atlas_qkv_bf16"]["address"] += 1
        with self.assertRaises(gate.GateError):
            gate.validate_runtime_extents(2_079, unaligned, stream=1)
        with self.assertRaises(gate.GateError):
            gate.validate_runtime_extents(2_079, valid_buffers(2_079), stream=0)
        overflow = valid_buffers(2_079)
        overflow["output_bf16"]["address"] = gate.UINT64_MAX - 15
        with self.assertRaisesRegex(gate.GateError, "overflow"):
            gate.validate_runtime_extents(2_079, overflow, stream=1)

    def test_missing_artifact_fails_closed(self) -> None:
        with self.assertRaisesRegex(gate.GateError, "unavailable regular file"):
            gate._stable_read(
                Path("/home/flocka/.cache/sglang/triton/DOES_NOT_EXIST/missing.cubin"),
                "missing artifact",
            )

    def test_manifest_hash_and_artifact_hash_drift_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
            manifest["kernels"][0]["files"]["cubin"][0] = "0" * 64
            path = write_manifest(manifest, directory)
            with self.assertRaisesRegex(gate.GateError, "SHA256 mismatch"):
                gate._attest_manifest(path, expected_manifest_sha256=None)
            with self.assertRaisesRegex(gate.GateError, "manifest: SHA256 mismatch"):
                gate.attest_manifest(path)

    def test_manifest_abi_extent_and_shape_drift_fail_closed(self) -> None:
        original = json.loads(MANIFEST.read_text(encoding="utf-8"))
        abi = copy.deepcopy(original)
        abi["kernels"][0]["driver_params"][-1][1] = "u32"
        extent = copy.deepcopy(original)
        extent["workspace_layouts"]["2079"]["total_bytes"] += 1
        shape = copy.deepcopy(original)
        shape["execution_policy"]["supported_m"].append(32_768)
        for index, (manifest, message) in enumerate(
            (
                (abi, "driver ABI"),
                (extent, "checked layout drift"),
                (shape, "execution policy drift"),
            )
        ):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(gate.GateError, message):
                    gate._attest_manifest(
                        write_manifest(manifest, directory),
                        expected_manifest_sha256=None,
                    )

    def test_all_source_modules_have_no_gpu_or_process_path(self) -> None:
        modules = set()
        sources = []
        root = Path(gate.__file__).parent
        for path in sorted(root.glob("frozen_triton_c143_*.py")):
            source = path.read_text(encoding="utf-8")
            sources.append(source)
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    modules.add(node.module.split(".")[0])
        self.assertTrue(
            {"torch", "triton", "ctypes", "subprocess", "cuda"}.isdisjoint(modules)
        )
        self.assertNotIn("cuModuleLoadData", "".join(sources))
        self.assertNotIn("cuLaunchKernel", "".join(sources))
        self.assertEqual(
            {root / name for name in gate.SOURCE_MODULES},
            {
                path
                for path in root.glob("frozen_triton_c143_*.py")
                if "test_support" not in path.name
            },
        )
        for path in (
            *root.glob("frozen_triton_c143_*.py"),
            *root.glob("test_frozen_triton_c143_*.py"),
        ):
            self.assertLessEqual(
                len(path.read_text(encoding="utf-8").splitlines()), 250
            )

    def test_gpu_authorization_requires_environment_flag_nonce_and_reservation(
        self,
    ) -> None:
        nonce = "a" * 64
        with tempfile.TemporaryDirectory() as directory:
            receipt = reservation(self.attestation, directory, nonce)
            with (
                mock.patch.dict(os.environ, {}, clear=True),
                self.assertRaises(gate.GateError),
            ):
                gate.authorize_gpu_execution(
                    explicit_flag="I_UNDERSTAND_THIS_RUNS_CUDA",
                    nonce=nonce,
                    reservation_path=receipt,
                    manifest_sha256=self.attestation["manifest_sha256"],
                )
            with mock.patch.dict(
                os.environ, {"ATLAS_GDN_C143_TRITON_RAW_GPU": "1"}, clear=True
            ):
                with self.assertRaises(gate.GateError):
                    gate.authorize_gpu_execution(
                        explicit_flag="wrong",
                        nonce=nonce,
                        reservation_path=receipt,
                        manifest_sha256=self.attestation["manifest_sha256"],
                    )
                with self.assertRaises(gate.GateError):
                    gate.authorize_gpu_execution(
                        explicit_flag="I_UNDERSTAND_THIS_RUNS_CUDA",
                        nonce="b" * 64,
                        reservation_path=receipt,
                        manifest_sha256=self.attestation["manifest_sha256"],
                    )
                authorized = gate.authorize_gpu_execution(
                    explicit_flag="I_UNDERSTAND_THIS_RUNS_CUDA",
                    nonce=nonce,
                    reservation_path=receipt,
                    manifest_sha256=self.attestation["manifest_sha256"],
                )
                self.assertEqual(authorized["owner"], "/root/future-independent-owner")

    def test_gpu_executor_remains_absent_even_after_valid_authorization(self) -> None:
        nonce = "c" * 64
        with tempfile.TemporaryDirectory() as directory:
            receipt = reservation(self.attestation, directory, nonce)
            stderr = io.StringIO()
            with (
                mock.patch.dict(
                    os.environ, {"ATLAS_GDN_C143_TRITON_RAW_GPU": "1"}, clear=True
                ),
                contextlib.redirect_stderr(stderr),
            ):
                rc = gate.main(
                    [
                        "--execute-gpu",
                        "--gpu-execution-flag",
                        "I_UNDERSTAND_THIS_RUNS_CUDA",
                        "--reservation-nonce",
                        nonce,
                        "--reservation",
                        str(receipt),
                    ]
                )
            self.assertEqual(rc, 2)
            self.assertIn("GPU executor is intentionally absent", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
