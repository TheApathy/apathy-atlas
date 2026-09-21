# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import b63_model_manifest_producer as producer


def write(path: Path, data: bytes) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return len(data), hashlib.sha256(data).hexdigest()


def fixture(root: Path, count: int = 3) -> dict[str, str]:
    shard_names = [
        f"model-{index:05d}-of-{count:05d}.safetensors" for index in range(1, count + 1)
    ]
    for index, name in enumerate(shard_names):
        write(root / name, bytes([index + 1]) * 19)
    weight_map = {f"tensor.{index}": name for index, name in enumerate(shard_names)}
    index_data = json.dumps({"weight_map": weight_map}).encode()
    write(root / "model.safetensors.index.json", index_data)
    ple_entries = []
    for index in range(2):
        name = f"ple-{index}.bin"
        size, digest = write(root / "ple-offload" / name, bytes([11 + index]) * 23)
        ple_entries.append({"file": name, "bytes": size, "sha256": digest})
    ple_data = json.dumps({"entries": ple_entries}).encode()
    write(root / "ple-offload" / "manifest.json", ple_data)
    write(root / "config.json", b"{}")
    write(root / "tokenizer.json", b'{"version":"1"}')
    write(root / "tokenizer_config.json", b'{"model_max_length":1000000}')
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in producer.META_HASHES
    }


def fail_n(real, target: int, *, complete: bool = False):
    calls = 0

    def wrapped(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == target:
            if complete:
                real(*args, **kwargs)
            raise OSError(f"injected failure {target}")
        return real(*args, **kwargs)

    return wrapped


class ManifestProducerTests(unittest.TestCase):
    def test_cli_is_inert_without_exact_literal(self) -> None:
        with mock.patch("sys.argv", ["producer", "--execute", "WRONG"]):
            with mock.patch.object(producer, "produce") as produce:
                with self.assertRaisesRegex(SystemExit, "authorization"):
                    producer.main()
                produce.assert_not_called()

    def test_document_census_content_root_and_two_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            meta = fixture(root)
            calls: list[str] = []
            real = producer._scan

            def tracked(*args, **kwargs):
                calls.append(args[1])
                return real(*args, **kwargs)

            with mock.patch.object(producer, "_scan", side_effect=tracked):
                document = producer.build_document(root, meta, 3)
            self.assertEqual(document["schema"], producer.SCHEMA)
            self.assertEqual(len(document["metadata"]), 5)
            self.assertEqual(len(producer._expected_main_shards(197)), 197)
            self.assertEqual(len(document["ple_sidecars"]), 2)
            for item in (
                document["metadata"]
                + document["main_shards"]
                + document["ple_sidecars"]
            ):
                self.assertGreaterEqual(calls.count(item["relative"]), 2)
            records = [
                {key: item[key] for key in ("relative", "sha256", "size")}
                for group in ("metadata", "main_shards", "ple_sidecars")
                for item in document[group]
            ]
            self.assertEqual(
                document["content_root_sha256"],
                hashlib.sha256(producer.canonical_bytes(records)).hexdigest(),
            )

    def test_create_new_mode_0444_and_no_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            meta = fixture(root)
            output = root.parent / f"{root.name}-manifest.json"
            try:
                receipt = producer.produce(root, output, meta, 3)
                self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o444)
                self.assertEqual(
                    receipt["manifest_sha256"],
                    hashlib.sha256(output.read_bytes()).hexdigest(),
                )
                with self.assertRaises(FileExistsError):
                    producer.produce(root, output, meta, 3)
            finally:
                output.chmod(0o600) if output.exists() else None
                output.unlink(missing_ok=True)

    def test_metadata_hash_shard_census_and_ple_identity_reject(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            meta = fixture(root)
            broken = dict(meta)
            broken["config.json"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "metadata hash"):
                producer.build_document(root, broken, 3)
            with self.assertRaisesRegex(RuntimeError, "shard census"):
                producer.build_document(root, meta, 4)
            (root / "ple-offload" / "ple-0.bin").write_bytes(b"mutated")
            with self.assertRaisesRegex(RuntimeError, "differs"):
                producer.build_document(root, meta, 3)

    def test_symlink_and_second_pass_drift_reject(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            meta = fixture(root)
            shard = root / "model-00001-of-00003.safetensors"
            target = root / "target.bin"
            target.write_bytes(shard.read_bytes())
            shard.unlink()
            shard.symlink_to(target)
            with self.assertRaisesRegex(RuntimeError, "non-canonical"):
                producer.build_document(root, meta, 3)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            meta = fixture(root)
            real = producer._scan
            seen: dict[str, int] = {}

            def drift(*args, **kwargs):
                record, raw = real(*args, **kwargs)
                relative = args[1]
                seen[relative] = seen.get(relative, 0) + 1
                if (
                    relative == "model-00001-of-00003.safetensors"
                    and seen[relative] == 2
                ):
                    record = {**record, "sha256": "f" * 64}
                return record, raw

            with mock.patch.object(producer, "_scan", side_effect=drift):
                with self.assertRaisesRegex(RuntimeError, "between full hash passes"):
                    producer.build_document(root, meta, 3)

    def test_every_publication_stage_failure_removes_owned_output(self) -> None:
        cases = (
            ("initial-fstat", "fstat", 1, False),
            ("write", "write", 1, False),
            ("file-fsync-1", "fsync", 1, False),
            ("fchmod", "fchmod", 1, False),
            ("file-fsync-2", "fsync", 2, False),
            ("guard-open", "open", 2, False),
            ("file-close", "close", 1, True),
            ("parent-open", "open", 3, False),
            ("parent-fsync", "fsync", 3, False),
            ("parent-close", "close", 2, True),
            ("final-scan-close", "close", 3, True),
            ("guard-close", "close", 4, True),
        )
        for label, name, target, complete in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temp:
                descriptors = set(os.listdir("/proc/self/fd"))
                base = Path(temp).resolve()
                root, output = base / "model", base / "manifest.json"
                root.mkdir()
                meta = fixture(root)
                document = producer.build_document(root, meta, 3)
                real = getattr(os, name)
                real_stat = os.stat
                if label == "initial-fstat":

                    def stat_effect(path, *args, **kwargs):
                        if isinstance(path, int):
                            raise OSError("persistent descriptor-stat failure")
                        return real_stat(path, *args, **kwargs)
                else:
                    stat_effect = real_stat
                with (
                    mock.patch.object(
                        producer, "build_document", return_value=document
                    ),
                    mock.patch.object(producer.os, "stat", side_effect=stat_effect),
                    mock.patch.object(
                        producer.os,
                        name,
                        side_effect=fail_n(real, target, complete=complete),
                    ),
                    self.assertRaises(OSError),
                ):
                    producer.produce(root, output, meta, 3)
                self.assertFalse(output.exists())
                self.assertEqual(descriptors, set(os.listdir("/proc/self/fd")))

    def test_final_mutation_is_removed_but_foreign_replacement_survives(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp).resolve()
            root, output = base / "model", base / "manifest.json"
            root.mkdir()
            meta = fixture(root)
            document = producer.build_document(root, meta, 3)
            payload = producer.canonical_bytes(document) + b"\n"
            real_scan = producer._scan

            def mutate(*args, **kwargs):
                output.chmod(0o600)
                output.write_bytes(b"X" * len(payload))
                output.chmod(0o444)
                return real_scan(*args, **kwargs)

            with mock.patch.object(producer, "build_document", return_value=document):
                with mock.patch.object(producer, "_scan", side_effect=mutate):
                    with self.assertRaisesRegex(RuntimeError, "identity/hash"):
                        producer.produce(root, output, meta, 3)
            self.assertFalse(output.exists())

            foreign = b"foreign replacement"

            def replace(*_args, **_kwargs):
                output.unlink()
                output.write_bytes(foreign)
                output.chmod(0o444)
                raise RuntimeError("final validation failed")

            with mock.patch.object(producer, "build_document", return_value=document):
                with mock.patch.object(producer, "_scan", side_effect=replace):
                    with self.assertRaisesRegex(RuntimeError, "validation failed"):
                        producer.produce(root, output, meta, 3)
            self.assertEqual(output.read_bytes(), foreign)
