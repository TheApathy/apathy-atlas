# SPDX-License-Identifier: AGPL-3.0-only

import os
import pathlib
import shutil
import subprocess
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[2]
BUILD_SCRIPT = ROOT / "scripts" / "build-exl3-prefill-max.sh"


class Exl3PrefillBuildReceiptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.repo = pathlib.Path(self.tempdir.name) / "repo"
        (self.repo / "scripts").mkdir(parents=True)
        shutil.copy2(BUILD_SCRIPT, self.repo / "scripts" / BUILD_SCRIPT.name)
        for relative in [
            "Cargo.toml",
            "Cargo.lock",
            "rust-toolchain.toml",
            ".cargo/config.toml",
            "crates/example/src/lib.rs",
            "kernels/gb10/example.cu",
            "vendor/cudarc/src/lib.rs",
        ]:
            path = self.repo / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"fixture {relative}\n", encoding="utf-8")

        subprocess.run(["git", "init", "-q"], cwd=self.repo, check=True)
        subprocess.run(
            ["git", "config", "user.email", "prefill-test@example.invalid"],
            cwd=self.repo,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Prefill Test"],
            cwd=self.repo,
            check=True,
        )
        subprocess.run(["git", "add", "."], cwd=self.repo, check=True)
        subprocess.run(
            ["git", "commit", "-qm", "fixture"], cwd=self.repo, check=True
        )

        fake_bin = pathlib.Path(self.tempdir.name) / "bin"
        fake_bin.mkdir()
        cargo = fake_bin / "cargo"
        cargo.write_text(
            """#!/usr/bin/env python3
import os
from pathlib import Path
for name in ("ATLAS_SKIP_BUILD", "SKIP_ATLAS_BUILD", "ATLAS_EXTRA_NVCC_FLAGS",
             "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"):
    if name in os.environ:
        raise SystemExit(f"unsafe inherited build variable survived: {name}")
expected = {"ATLAS_TARGET_HW": "gb10", "ATLAS_TARGET_MODEL": "deepseek-v4-flash",
            "ATLAS_TARGET_QUANT": "nvfp4", "CUDARC_CUDA_VERSION": "13000"}
for name, value in expected.items():
    if os.environ.get(name) != value:
        raise SystemExit(f"wrong {name}")
binary = Path(os.environ["CARGO_TARGET_DIR"]) / "release" / "spark"
binary.parent.mkdir(parents=True, exist_ok=True)
binary.write_text("#!/bin/sh\\nprintf '%s\\n' 'Omitting the option keeps the MODEL.toml default.'\\n")
binary.chmod(0o755)
""",
            encoding="utf-8",
        )
        cargo.chmod(0o755)
        self.env = os.environ.copy()
        self.env["PATH"] = f"{fake_bin}:{self.env['PATH']}"
        self.env["ATLAS_SKIP_BUILD"] = "1"
        self.env["ATLAS_EXTRA_NVCC_FLAGS"] = "--unsafe-fixture"

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    @property
    def script(self) -> pathlib.Path:
        return self.repo / "scripts" / BUILD_SCRIPT.name

    def run_script(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(
            [str(self.script), *args],
            cwd=self.repo,
            env=self.env,
            check=check,
            text=True,
            capture_output=True,
        )

    def build(self) -> None:
        self.run_script()
        self.run_script("--verify-only")

    def test_receipt_binds_target_source_and_binary(self) -> None:
        self.build()
        receipt = (self.repo / "target/release/spark.prefill-max-build-receipt").read_text()
        self.assertIn("schema atlas-prefill-max-build-v1\n", receipt)
        self.assertIn("target_hw gb10\n", receipt)
        self.assertIn("target_model deepseek-v4-flash\n", receipt)
        self.assertIn("target_quant nvfp4\n", receipt)

        binary = self.repo / "target/release/spark"
        binary.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        binary.chmod(0o755)
        result = self.run_script("--verify-only", check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not match its build receipt", result.stderr)

    def test_tracked_and_untracked_vendor_drift_fail_closed(self) -> None:
        self.build()
        vendor = self.repo / "vendor/cudarc/src/lib.rs"
        vendor.write_text("changed tracked vendor input\n", encoding="utf-8")
        result = self.run_script("--verify-only", check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("build inputs changed", result.stderr)

        self.run_script()
        (self.repo / "vendor/cudarc/src/untracked.rs").write_text(
            "untracked vendor input\n", encoding="utf-8"
        )
        result = self.run_script("--verify-only", check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("build inputs changed", result.stderr)

    def test_receipt_symlink_is_rejected(self) -> None:
        self.build()
        receipt = self.repo / "target/release/spark.prefill-max-build-receipt"
        saved = self.repo / "saved-receipt"
        receipt.rename(saved)
        receipt.symlink_to(saved)
        result = self.run_script("--verify-only", check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("receipt is missing or unsafe", result.stderr)


if __name__ == "__main__":
    unittest.main()
