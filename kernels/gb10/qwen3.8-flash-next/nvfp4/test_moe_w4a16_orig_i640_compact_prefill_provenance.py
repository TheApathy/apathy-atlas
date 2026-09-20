# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from moe_w4a16_orig_i640_compact_prefill_capture_support import (
    encode_kv,
    identity,
)
from moe_w4a16_orig_i640_compact_prefill_provenance import (
    B62,
    BASE,
    IDENTITY_KEYS,
    PREFIX,
    SOURCE_NAMES,
    hook_receipt,
    server_receipt,
    source_receipt,
)


class ProvenanceHostiles(unittest.TestCase):
    def immutable(self, path: Path, body: bytes) -> None:
        path.write_bytes(body)
        path.chmod(0o444)

    def sources(self) -> dict[str, str]:
        here = Path(__file__).resolve().parent
        return source_receipt(here / f"{BASE}_capture.py")

    def test_running_producer_bundle_is_recomputed_not_plan_supplied(self) -> None:
        fields = self.sources()
        self.assertEqual(fields["source.count"], str(len(SOURCE_NAMES)))
        self.assertEqual(fields["producer.count"], "4")
        self.assertEqual(
            fields["producer.entry.path"], str(PREFIX / f"{BASE}_capture.py")
        )
        self.assertNotIn("plan", " ".join(fields))

    def test_copied_entry_source_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            copied = Path(directory) / f"{BASE}_capture.py"
            shutil.copy2(Path(__file__).resolve().parent / copied.name, copied)
            with self.assertRaisesRegex(ValueError, "canonical source tree"):
                source_receipt(copied)

    def hook_fields(self, hook: Path, sources: dict[str, str]) -> dict[str, str]:
        compiler = Path("/usr/bin/cc").resolve(strict=True)
        compiler_id = identity(compiler)
        binary_id = identity(hook, 0o555)
        source_number = SOURCE_NAMES.index(f"{BASE}_capture_preload.c")
        source = Path(__file__).resolve().parent / f"{BASE}_capture_preload.c"
        argv = [
            str(compiler),
            "-std=c11",
            "-O2",
            "-DNDEBUG",
            "-fPIC",
            "-shared",
            "-Wl,-z,relro,-z,now",
            str(source),
            "-o",
            str(hook.resolve()),
            "-Wl,--build-id=sha1",
            "-ldl",
        ]
        fields = {
            "schema": "oi640-hook-compile-v1",
            "profile": "release",
            "target": "aarch64-linux-gnu",
            "source.bundle_sha256": sources["source.bundle_sha256"],
            "source.path": str(PREFIX / source.name),
            "source.sha256": sources[f"source.{source_number:02}.sha256"],
            "compiler.version_sha256": "a" * 64,
            "compile.argc": "12",
            "compile.env_sha256": "b" * 64,
            "build.nonce": "0123456789abcdef" * 4,
        }
        fields.update({f"compiler.{key}": value for key, value in compiler_id.items()})
        fields.update({f"binary.{key}": value for key, value in binary_id.items()})
        fields.update(
            {f"compile.argv.{number:02}": value for number, value in enumerate(argv)}
        )
        return fields

    def test_fake_mode0555_elf_cannot_self_label_as_hook(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root / "hook.so"
            shutil.copy2("/bin/true", fake)
            fake.chmod(0o555)
            receipt = root / "hook.kv"
            sources = self.sources()
            self.immutable(receipt, encode_kv(self.hook_fields(fake, sources)))
            with self.assertRaisesRegex(ValueError, "hook ELF export"):
                hook_receipt(receipt, fake, sources)

    def test_hook_source_or_bundle_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fake = root / "hook.so"
            shutil.copy2("/bin/true", fake)
            fake.chmod(0o555)
            sources = self.sources()
            fields = self.hook_fields(fake, sources)
            fields["source.bundle_sha256"] = "f" * 64
            receipt = root / "hook.kv"
            self.immutable(receipt, encode_kv(fields))
            with self.assertRaisesRegex(ValueError, "hook compile receipt"):
                hook_receipt(receipt, fake, sources)

    def server_fields(self) -> dict[str, str]:
        fields = {
            "schema": "oi640-server-build-v1",
            "target.signature": "gb10|qwen3.8-flash-next|nvfp4|sm_121f",
            "kernel.ptx_count": "154",
            "kernel.override_count": "9",
            "kernel.target_ptx_set_count": "1",
            "kernel.v2_count": "0",
            "source.count": "1839",
            "selected.count": "25",
            "ptx.count": "154",
            "source.manifest.path": "/sealed/source.sha256",
            "selected.manifest.path": "/sealed/selected.sha256",
            "ptx.manifest.path": "/sealed/ptx.sha256",
            "upstream.receipt.path": "/sealed/BUILD_RECEIPT.md",
        }
        fields.update(B62)
        identity = {f"binary.{key}" for key in IDENTITY_KEYS}
        self.assertEqual(set(B62) & identity, identity)
        return fields

    def test_opaque_or_mismatched_server_receipt_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            opaque = root / "opaque.kv"
            self.immutable(opaque, b"unparsed-build-bytes")
            with self.assertRaises(ValueError):
                server_receipt(opaque, Path("/bin/true"))
            mismatched = root / "server.kv"
            self.immutable(mismatched, encode_kv(self.server_fields()))
            fake = root / "spark"
            shutil.copy2("/bin/true", fake)
            fake.chmod(0o555)
            with self.assertRaisesRegex(ValueError, "binary identity mismatch"):
                server_receipt(mismatched, fake)

    def test_server_target_census_and_pinned_hashes_are_not_optional(self) -> None:
        fields = self.server_fields()
        for key in (
            "target.signature",
            "kernel.ptx_count",
            "kernel.override_count",
            "ptx.manifest.sha256",
        ):
            hostile = dict(fields)
            hostile[key] = "0"
            with tempfile.TemporaryDirectory() as directory:
                receipt = Path(directory) / "server.kv"
                self.immutable(receipt, encode_kv(hostile))
                with self.assertRaisesRegex(ValueError, "target/census"):
                    server_receipt(receipt, Path("/bin/true"))

    def test_raw_consumer_reparses_every_provenance_receipt(self) -> None:
        source = (
            Path(__file__).resolve().parent / f"{BASE}_microgate_provenance.cuh"
        ).read_text()
        for token in (
            'manifest_artifact(proof.capture, "producer.receipt"',
            'manifest_artifact(proof.capture, "hook.compile_receipt"',
            'manifest_artifact(proof.capture, "server.build_receipt"',
            "producer.fields.size() != 58",
            "hook.fields.size() != 36",
            "server.fields.size() == 24",
            'sha_manifest_artifact(server, "ptx", 154, true)',
            'identity_fields(server, "binary", binary)',
            "load_provenance(current)",
        ):
            self.assertIn(token, source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
