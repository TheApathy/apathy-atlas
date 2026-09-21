# SPDX-License-Identifier: AGPL-3.0-only
"""CPU/static contract tests; imports no CUDA or Atlas runtime."""

import json
import importlib
import re
import struct
import sys
import unittest
from pathlib import Path

from test_support import (
    EXECUTOR_SOURCES,
    GATE_SOURCES,
    HERE,
    MANIFEST,
    RUST,
    U64_MAX,
    checked_mul,
    layout,
    rust_constant,
    sha,
    source_bundle,
    valid_regions,
    validate_kernel_census,
    validate_regions,
)


class StaticContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest_raw = MANIFEST.read_bytes()
        cls.manifest = json.loads(cls.manifest_raw)
        cls.constants = (HERE / "manifest.rs").read_text()

    def test_isolated_spdx_caps_and_no_registration(self) -> None:
        for name in (*RUST, "test_static.py", "test_support.py"):
            text = (HERE / name).read_text()
            self.assertTrue(
                text.startswith("// SPDX-License-Identifier:")
                or text.startswith("# SPDX-License-Identifier:")
            )
            self.assertLessEqual(len(text.splitlines()), 250, name)
            self.assertNotRegex(text, r"(?m)[ \t]+$")
        ops = (HERE.parent / "../ops.rs").resolve().read_text()
        self.assertNotIn("gdn_triton_sm121", ops)

    def test_manifest_and_approved_source_seals(self) -> None:
        self.assertEqual(
            sha(self.manifest_raw), rust_constant(self.constants, "MANIFEST_SHA256")
        )
        self.assertEqual(len(self.manifest_raw), 15_331)
        self.assertEqual(
            source_bundle(GATE_SOURCES),
            rust_constant(self.constants, "GATE_SOURCE_BUNDLE_SHA256"),
        )
        self.assertEqual(
            source_bundle(EXECUTOR_SOURCES),
            rust_constant(self.constants, "EXECUTOR_SOURCE_BUNDLE_SHA256"),
        )
        sys.path.insert(0, str(MANIFEST.parent))
        try:
            attest = importlib.import_module(
                "frozen_triton_c143_attest"
            ).attest_manifest()
        finally:
            sys.path.pop(0)
        self.assertEqual(
            attest["artifact_attestation_sha256"],
            rust_constant(self.constants, "ARTIFACT_ATTESTATION_SHA256"),
        )
        self.assertEqual(
            self.manifest["provenance"]["sglang_commit"],
            rust_constant(self.constants, "SGLANG_COMMIT"),
        )
        self.assertEqual(self.manifest["provenance"]["sglang_license"], "Apache-2.0")
        self.assertEqual(self.manifest["provenance"]["triton_license"], "MIT")
        mutated = bytearray(self.manifest_raw)
        mutated[-1] ^= 1
        self.assertNotEqual(
            sha(mutated), rust_constant(self.constants, "MANIFEST_SHA256")
        )

    def test_exact_five_kernel_contracts_and_bytes(self) -> None:
        blocks = re.findall(r"KernelSpec \{(.*?)\n    \},", self.constants, re.S)
        self.assertEqual(len(blocks), 5)
        grid_map = {
            "Nt48": ["NT", 48, 1],
            "Fixed4x48": [4, 48, 1],
            "Output": [2, "NT", 48],
        }
        for block, kernel in zip(blocks, self.manifest["kernels"], strict=True):
            for field, value in (
                ("role", kernel["role"]),
                ("function", kernel["function"]),
            ):
                self.assertIn(f'{field}: "{value}"', block)
            self.assertIn(f'cache_dir: "{Path(kernel["artifact_dir"]).name}"', block)
            digest, size = kernel["files"]["cubin"]
            self.assertIn(f'cubin_sha256: "{digest}"', block)
            self.assertIn(f"cubin_bytes: {size:,}".replace(",", "_"), block)
            grid_name = re.search(r"grid: Grid::(\w+)", block).group(1)
            self.assertEqual(grid_map[grid_name], kernel["grid"])
            for field, value in (
                ("block_x", kernel["block"][0]),
                ("dynamic_shared", kernel["dynamic_shared_bytes"]),
                ("static_shared", 1_024),
                ("registers", kernel["registers_per_thread"]),
                ("local_bytes", 0),
                ("stack_bytes", 0),
            ):
                self.assertIn(f"{field}: {value:,}".replace(",", "_"), block)
            path = Path(kernel["artifact_dir"]) / f"{kernel['function']}.cubin"
            before = path.stat()
            raw = path.read_bytes()
            after = path.stat()

            def identity(value):
                return (
                    value.st_dev,
                    value.st_ino,
                    value.st_size,
                    value.st_mtime_ns,
                    value.st_ctime_ns,
                )

            self.assertEqual(identity(before), identity(after))
            self.assertEqual((sha(raw), len(raw)), (digest, size))
            changed = bytearray(raw)
            changed[0] ^= 1
            self.assertNotEqual(sha(changed), digest)
        validate_kernel_census(self.manifest["kernels"])
        with self.assertRaises(ValueError):
            validate_kernel_census(self.manifest["kernels"][:-1])
        duplicate = [dict(kernel) for kernel in self.manifest["kernels"]]
        duplicate[-1]["function"] = duplicate[0]["function"]
        with self.assertRaises(ValueError):
            validate_kernel_census(duplicate)

    def test_abi_grids_and_null_scratch_are_exact(self) -> None:
        abi_defs = dict(
            re.findall(r"const (ABI_\w+): &\[Abi\] = &\[(.*?)\];", self.constants, re.S)
        )
        blocks = re.findall(r"KernelSpec \{(.*?)\n    \},", self.constants, re.S)
        for block, kernel in zip(blocks, self.manifest["kernels"], strict=True):
            abi_name = re.search(r"abi: (ABI_\w+)", block).group(1)
            actual = [
                item.strip().lower()
                for item in abi_defs[abi_name].split(",")
                if item.strip()
            ]
            self.assertEqual(actual, [entry[1] for entry in kernel["driver_params"]])
            self.assertEqual(actual[-2:], ["u64", "u64"])
        launch = (HERE / "launch.rs").read_text()
        self.assertEqual(launch.count("zero_scratch()[0]"), 5)
        self.assertEqual(launch.count("zero_scratch()[1]"), 5)
        self.assertEqual(launch.count("f32::from_bits(0x3db5_04f3)"), 2)
        scalar = self.manifest["runtime_scalar_contracts"]["scale_f32"]
        self.assertEqual(struct.unpack("<I", struct.pack("<f", scalar))[0], 0x3DB504F3)
        self.assertIn(
            "call.args.ends_with(&[ArgValue::U64(0), ArgValue::U64(0)])", launch
        )
        sequence = re.search(
            r"pub const SAME_STREAM_SEQUENCE: &\[&str\] = &\[(.*?)\];",
            self.constants,
            re.S,
        ).group(1)
        self.assertEqual(
            re.findall(r'"([^"]+)"', sequence), self.manifest["same_stream_sequence"]
        )

    def test_workspace_layouts_and_hostile_arithmetic(self) -> None:
        for m, total in ((2_079, 188_241_152), (8_192, 729_286_400)):
            expected = layout(m)
            self.assertEqual(expected, self.manifest["workspace_layouts"][str(m)])
            self.assertEqual(expected["total_bytes"], total)
        with self.assertRaises(ValueError):
            layout(32_768)
        with self.assertRaises(OverflowError):
            checked_mul(U64_MAX, 2)
        records = valid_regions(2_079)
        validate_regions(2_079, records, 1)
        for name, field, value, error in (
            ("atlas_qkv_bf16", 0, 0, ValueError),
            (
                "atlas_gate_beta_f32",
                0,
                records["atlas_gate_beta_f32"][0] + 1,
                ValueError,
            ),
            ("workspace", 1, records["workspace"][1] - 1, ValueError),
            ("output_bf16", 0, records["atlas_qkv_bf16"][0], ValueError),
            ("atlas_state_hkv_f32", 0, U64_MAX - 15, OverflowError),
        ):
            hostile = {key: item.copy() for key, item in records.items()}
            hostile[name][field] = value
            with self.assertRaises(error):
                validate_regions(2_079, hostile, 1)
        with self.assertRaises(ValueError):
            validate_regions(2_079, records, 0)
        types = (HERE / "types.rs").read_text()
        for proof in (
            "checked_mul",
            "checked_add",
            "null device pointer",
            "unaligned pointer",
            "exact extent required",
            "buffers alias",
            "nondefault CUDA stream",
        ):
            self.assertIn(proof, types)

    def test_binary_safe_loader_and_fail_clean_lifetime(self) -> None:
        loader = (HERE / "loader.rs").read_text()
        ffi = (HERE / "ffi.rs").read_text()
        combined = loader + ffi
        self.assertIn("cuModuleLoadData", combined)
        self.assertNotRegex(combined, r"\bcuModuleLoad\s*\(")
        for proof in (
            "recheck_all()?",
            "held.recheck()?",
            "duplicate kernel function",
            "duplicate CUDA function handle",
            "cuModuleGetFunction returned null",
            "ContextLease",
            "impl Drop for LoadedKernel",
            "impl Drop for ModuleGuard",
        ):
            self.assertIn(proof, combined)
        self.assertNotIn("include_bytes!", combined)
        self.assertIn("exact GB10 SM12.1/48SM", ffi)


if __name__ == "__main__":
    unittest.main()
