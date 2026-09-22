#!/usr/bin/env python3
"""Hostile CPU-only tests for the DFlash2 checkpoint admission boundary."""

import importlib.util
import hashlib
import json
import math
import os
import pathlib
import struct
import tempfile
import unittest
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "dflash2_checkpoint_admission", HERE / "dflash2_checkpoint_admission.py"
)
assert SPEC and SPEC.loader
ad = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ad)


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def write_safe(path, entries, payload, *, suffix=b""):
    if path.exists():
        path.chmod(0o644)
    header = canonical(entries).rstrip(b"\n")
    header += b" " * ((8 - len(header) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload + suffix)
    path.chmod(0o444)


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.root_obj = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.root_obj.name)

    def tearDown(self):
        for path in self.root.rglob("*"):
            if not path.is_symlink():
                path.chmod(0o644 if path.is_file() else 0o755)
        self.root_obj.cleanup()

    def model(self, entries=None, payload=b"\0\0\x80?", suffix=b""):
        path = self.root / "model.safetensors"
        entries = entries or {"a": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}}
        write_safe(path, entries, payload, suffix=suffix)
        return path

    def test_production_manifest_arithmetic(self):
        specs = ad.expected_specs()
        self.assertEqual((len(specs), sum(math.prod(v) for v in specs.values())), (96, 695497216))
        self.assertEqual(sum(math.prod(v) * 2 for v in specs.values()), 1390994432)
        self.assertEqual(ad.inventory_sha(), "76fc720446562f1ffd02007e972fc955893d5aa92c2fb25a0eca039061ccb4be")

    def test_tiny_header_and_full_payload_qualifications(self):
        path = self.model()
        header = ad.inspect_model(path, {"a": (2,)}, 2, full_payload=False)
        full = ad.inspect_model(path, {"a": (2,)}, 2, full_payload=True)
        self.assertEqual(header["qualification"], "HEADER_ONLY_NO_PAYLOAD_FINITE_SCAN")
        self.assertEqual(full["qualification"], "FULL_PAYLOAD_BF16_FINITE")

    def test_rejects_overlap_gap_and_trailing_bytes(self):
        cases = [
            ({"a": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]},
              "b": {"dtype": "BF16", "shape": [1], "data_offsets": [1, 3]}}, b"\0" * 3, b""),
            ({"a": {"dtype": "BF16", "shape": [1], "data_offsets": [2, 4]}}, b"\0" * 4, b""),
            ({"a": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}}, b"\0\0", b"x"),
        ]
        for index, (entries, payload, suffix) in enumerate(cases):
            with self.subTest(index=index), self.assertRaises(ad.AdmissionError):
                ad.inspect_model(self.model(entries, payload, suffix), {"a": (1)}, 1, False)

    def test_rejects_duplicate_json_name(self):
        path = self.root / "model.safetensors"
        header = b'{"a":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]},"a":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}'
        header += b" " * ((8 - len(header) % 8) % 8)
        path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0")
        path.chmod(0o444)
        with self.assertRaisesRegex(ad.AdmissionError, "duplicate"):
            ad.inspect_model(path, {"a": (1)}, 1, False)

    def test_rejects_shape_dtype_and_nonfinite_bf16(self):
        for entry in (
            {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]},
            {"dtype": "BF16", "shape": [True], "data_offsets": [0, 2]},
        ):
            with self.assertRaises(ad.AdmissionError):
                ad.inspect_model(self.model({"a": entry}, b"\0\0"), {"a": (1,)}, 1, False)
        for word in (b"\x80\x7f", b"\x80\xff", b"\x81\x7f"):
            with self.subTest(word=word), self.assertRaisesRegex(ad.AdmissionError, "nonfinite"):
                ad.inspect_model(self.model(payload=word + b"\0\0"), {"a": (2,)}, 2, True)
        ad.inspect_model(self.model(payload=b"\0\x7f\0\0"), {"a": (2,)}, 2, True)

    def test_rejects_writable_symlink_and_hardlink(self):
        path = self.model()
        path.chmod(0o644)
        with self.assertRaisesRegex(ad.AdmissionError, "immutable"):
            ad.secure_stat(path)
        path.chmod(0o444)
        link = self.root / "hard"
        os.link(path, link)
        with self.assertRaisesRegex(ad.AdmissionError, "one link"):
            ad.secure_stat(path)
        link.unlink()
        sym = self.root / "sym"
        sym.symlink_to(path)
        with self.assertRaises(ad.AdmissionError):
            ad.secure_stat(sym)

    def test_output_must_stay_outside_checkpoint_without_mutation(self):
        checkpoint = self.root / "checkpoint"
        checkpoint.mkdir()
        source = b"# identical source alias\n"
        source_sha = hashlib.sha256(source).hexdigest()
        specs = {"fc.weight": (2,)}
        config = {"schema": "tiny-dflash2"}
        with (
            mock.patch.multiple(ad, COUNT=1, PARAMS=2, RAW=4, SOURCE_SHA=source_sha),
            mock.patch.object(ad, "expected_specs", return_value=specs),
            mock.patch.object(ad, "expected_config", return_value=config),
        ):
            config_path = checkpoint / "config.json"
            config_path.write_bytes(canonical(config))
            for name in ("dflash.py", "dflash2.py"):
                (checkpoint / name).write_bytes(source)
            write_safe(
                checkpoint / "model.safetensors",
                {"fc.weight": {"dtype": "BF16", "shape": [2], "data_offsets": [0, 4]}},
                b"\0" * 4,
            )
            model_sha = hashlib.sha256((checkpoint / "model.safetensors").read_bytes()).hexdigest()
            config_sha = hashlib.sha256(config_path.read_bytes()).hexdigest()
            export = checkpoint / "dflash2_export_receipt.json"
            export.write_bytes(canonical(ad.expected_export(model_sha, config_sha)))
            for path in checkpoint.iterdir():
                path.chmod(0o444)
            checkpoint.chmod(0o555)
            alias = self.root / "checkpoint-alias"
            alias.symlink_to(checkpoint, target_is_directory=True)
            before = checkpoint.lstat()
            directory_before = (before.st_dev, before.st_ino, before.st_mode, before.st_mtime_ns)
            census = {path.name for path in checkpoint.iterdir()}

            def args(output):
                return mock.Mock(
                    checkpoint=checkpoint,
                    output=output,
                    full_payload=True,
                    expected_v3_shared_state_digest=None,
                    training_lineage=None,
                    expected_training_lineage_sha256=None,
                )

            outputs = (
                checkpoint,
                checkpoint / "direct.json",
                checkpoint / "missing" / ".." / "lexical.json",
                alias / "symlink-parent.json",
            )
            for output in outputs:
                with self.subTest(output=output), self.assertRaisesRegex(
                    ad.AdmissionError, "outside checkpoint"
                ):
                    ad.admit(args(output))
                after = checkpoint.lstat()
                self.assertEqual(
                    (after.st_dev, after.st_ino, after.st_mode, after.st_mtime_ns),
                    directory_before,
                )
                self.assertEqual({path.name for path in checkpoint.iterdir()}, census)

            receipt = ad.admit(args(self.root / "outside-admission.json"))
            self.assertEqual(receipt["directory_identity_pre"], receipt["directory_identity_post"])
            after = checkpoint.lstat()
            self.assertEqual(
                (after.st_dev, after.st_ino, after.st_mode, after.st_mtime_ns),
                directory_before,
            )
            self.assertEqual(
                receipt["directory_identity_post"],
                {
                    "dev": after.st_dev,
                    "inode": after.st_ino,
                    "mode": after.st_mode & 0o7777,
                    "mtime_ns": after.st_mtime_ns,
                },
            )
            self.assertEqual({path.name for path in checkpoint.iterdir()}, set(ad.FILES))

    def test_config_is_exact_and_canonical(self):
        path = self.root / "config.json"
        path.write_bytes(canonical(ad.expected_config()))
        path.chmod(0o444)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), "53765f42507beabd84e454a1ef35241767495eb7a2c2cdd9a42816c09bac3356")
        self.assertEqual(ad.read_canonical_json(path), ad.expected_config())
        changed = ad.expected_config()
        changed["is_causal"] = True
        with self.assertRaisesRegex(ad.AdmissionError, "config"):
            ad.validate_config(changed)

    def test_export_receipt_closes_hashes_and_sources(self):
        receipt = json.loads(canonical(ad.expected_export("1" * 64, "2" * 64)))
        ad.validate_export(receipt, "1" * 64, "2" * 64)
        receipt["upstream"]["commit"] = "0" * 40
        with self.assertRaisesRegex(ad.AdmissionError, "receipt"):
            ad.validate_export(receipt, "1" * 64, "2" * 64)

    def test_create_new_receipt_payload_digest_and_no_overwrite(self):
        path = self.root / "admission.json"
        value = ad.write_receipt(path, {"schema": "test", "value": 1})
        base = {key: item for key, item in value.items() if key != "receipt_payload_sha256"}
        self.assertEqual(value["receipt_payload_sha256"], hashlib.sha256(canonical(base)).hexdigest())
        self.assertEqual(path.stat().st_mode & 0o777, 0o444)
        with self.assertRaises(FileExistsError):
            ad.write_receipt(path, {"schema": "test", "value": 1})


if __name__ == "__main__":
    unittest.main()
