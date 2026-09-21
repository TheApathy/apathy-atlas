# SPDX-License-Identifier: AGPL-3.0-only
"""Static and pure tests for the fail-closed one-shot build authority."""

from __future__ import annotations

import importlib.util
import hashlib
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "build_provenance.py"
LAUNCHER = ROOT / "build.sh"
EXPECTED_SOURCE_SHA256 = "cdeab5041339611ffe961a214fff9cbc57043fe14196c349ac4514a0092aa530"
EXPECTED_LAUNCHER_SHA256 = "b457e83ef8ca459f5c19d96a047f6dc6cc5928a8e34db61944218fe54c0701b4"


def load_module():
    spec = importlib.util.spec_from_file_location("q38_build_provenance", SOURCE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def contract(source: str) -> list[str]:
    required = {
        "no default output": 'parser.add_argument("output_dir")',
        "exclusive directory": "os.mkdir(output_dir, 0o700)",
        "nofollow held inputs": "os.O_RDONLY | os.O_NOFOLLOW",
        "exclusive copies": "os.O_WRONLY | os.O_CREAT | os.O_EXCL",
        "immutable copies": "os.chmod(destination, 0o400)",
        "held copied inputs": "held_copies.append(held_copy)",
        "held pre post": "held.verify_unchanged()",
        "fixed nvcc": 'Path("/usr/local/cuda-13.0/bin/nvcc")',
        "fixed cxx": 'Path("/usr/bin/c++").resolve()',
        "nvcc pin": "PINNED_NVCC_SHA256",
        "cxx pin": "PINNED_CXX_SHA256",
        "sanitized env": 'BUILD_ENV = {"PATH": "/usr/bin:/bin", "LC_ALL": "C"}',
        "one shot outputs": "ensure_absent(object_file, library, receipt)",
        "exclusive receipt": 'receipt.open("x", encoding="utf-8")',
        "receipt argv": '"commands": commands',
        "receipt environment": '"environment": BUILD_ENV',
        "receipt inputs": '"inputs": copied',
        "receipt tools": '"tools": tool_receipts',
        "external scheduler manifest": 'parser.add_argument("scheduler_manifest")',
        "external manifest digest": 'parser.add_argument("scheduler_manifest_sha256")',
        "manifest held by digest": "Held(Path(args.scheduler_manifest), args.scheduler_manifest_sha256)",
        "manifest immutable": "authority_stat.st_nlink != 1 or authority_stat.st_mode & 0o222",
        "manifest exact schema": "validate_authority(json.loads(raw_authority))",
        "manifest receipt": '"scheduler_manifest": file_receipt(authority_copy)',
    }
    failures = [name for name, token in required.items() if token not in source]
    if hashlib.sha256(source.encode()).hexdigest() != EXPECTED_SOURCE_SHA256:
        failures.append("exact builder bytes")
    return failures


class BuildProvenanceTests(unittest.TestCase):
    def test_build_authority_contract(self) -> None:
        self.assertEqual(contract(SOURCE.read_text()), [])
        shell = (ROOT / "build.sh").read_text()
        self.assertNotIn("BUILD_DIR", shell)
        self.assertIn("exec /usr/bin/env -i PATH=/usr/bin:/bin LC_ALL=C", shell)
        self.assertIn('/usr/bin/python3 "${native_dir}/build_provenance.py" "$@"', shell)
        self.assertEqual(hashlib.sha256(shell.encode()).hexdigest(), EXPECTED_LAUNCHER_SHA256)

    def test_tool_pins_are_exact_lowercase_sha256(self) -> None:
        module = load_module()
        for pin in [
            module.PINNED_NVCC_SHA256,
            module.PINNED_CXX_SHA256,
            module.PINNED_CUDART_SHA256,
            module.PINNED_CUDA_DRIVER_SHA256,
            *module.TOOL_PINS.values(),
        ]:
            self.assertTrue(module.valid_sha256(pin))
        for path, expected in [
            (module.NVCC, module.PINNED_NVCC_SHA256),
            (module.CXX, module.PINNED_CXX_SHA256),
            (module.CUDART, module.PINNED_CUDART_SHA256),
            (module.CUDA_DRIVER, module.PINNED_CUDA_DRIVER_SHA256),
        ]:
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)

    def test_scheduler_manifest_is_exact_and_authorizes_builder(self) -> None:
        module = load_module()
        inputs = {relative: "0" * 64 for relative in module.REQUIRED_INPUTS}
        valid = {
            "schema": "qwen38-ssm-residual-scheduler-v1",
            "inputs": inputs,
            "tools": module.TOOL_PINS,
            "environment": module.BUILD_ENV,
        }
        self.assertIs(module.validate_authority(valid), valid)
        bad_inputs = {key: value for key, value in inputs.items() if key != "build_provenance.py"}
        for candidate in [
            valid | {"schema": "wrong"},
            valid | {"inputs": bad_inputs},
            valid | {"inputs": inputs | {"build_provenance.py": "g" * 64}},
            valid | {"tools": {}},
            valid | {"environment": {"PATH": "/tmp"}},
            valid | {"extra": None},
        ]:
            with self.assertRaises(RuntimeError):
                module.validate_authority(candidate)

    def test_build_mutants_reject(self) -> None:
        source = SOURCE.read_text()
        for old, new in [
            (
                "os.mkdir(output_dir, 0o700)",
                "output_dir.mkdir(parents=True, exist_ok=True)",
            ),
            ("os.O_RDONLY | os.O_NOFOLLOW", "os.O_RDONLY"),
            ("os.O_WRONLY | os.O_CREAT | os.O_EXCL", "os.O_WRONLY | os.O_CREAT"),
            ("os.chmod(destination, 0o400)", "pass"),
            ("held_copies.append(held_copy)", "pass"),
            ("held.verify_unchanged()", "pass"),
            (
                'receipt.open("x", encoding="utf-8")',
                'receipt.open("w", encoding="utf-8")',
            ),
            ('"commands": commands', '"commands": []'),
            ("Held(NVCC, TOOL_PINS[str(NVCC)])", "Held(NVCC)"),
            ("env=BUILD_ENV, check=True", "env=os.environ, check=True"),
            ("env=BUILD_ENV, check=True", "env=BUILD_ENV, check=False"),
            (
                "authority_stat.st_nlink != 1 or authority_stat.st_mode & 0o222",
                "False",
            ),
            (
                "for held in [authority, *held_inputs, *held_copies, *tools]:\n            held.verify_unchanged()",
                "for held in held_inputs[:1]:\n            held.verify_unchanged()",
            ),
            (
                "if stamp_fd(self.fd) != self.initial:",
                "if False and stamp_fd(self.fd) != self.initial:",
            ),
        ]:
            self.assertEqual(source.count(old), 1, old)
            self.assertTrue(contract(source.replace(old, new, 1)), old)


if __name__ == "__main__":
    unittest.main()
