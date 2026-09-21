# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import hashlib
import json
import shutil
import struct
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from moe_w4a16_orig_i640_compact_prefill_capture import validate_plan
from moe_w4a16_orig_i640_compact_prefill_capture_support import (
    EVENT,
    EVENT_MAGIC,
    decode_kv,
    encode_kv,
    identity,
    model_manifest,
    parse_event,
    parse_offsets,
    sha,
    source_bundle,
    validate_model,
)
from moe_w4a16_orig_i640_compact_prefill_manifest import create_build


class CaptureTests(unittest.TestCase):
    def immutable(self, path: Path, data: bytes) -> None:
        path.write_bytes(data)
        path.chmod(0o444)

    def test_canonical_manifest_rejects_hostile_framing(self) -> None:
        expected = b"a=1\nb=2\n"
        self.assertEqual(encode_kv({"b": "2", "a": "1"}), expected)
        self.assertEqual(decode_kv(expected), {"a": "1", "b": "2"})
        for bad in (b"b=2\na=1\n", b"a=1\na=2\n", b"a=1", b"a=1\n\0"):
            with self.assertRaises(ValueError):
                decode_kv(bad)

    def test_offsets_and_event_are_exact_bounded_readonly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = [0] + [20130] * 512
            offsets = struct.pack("<513I", *values)
            self.immutable(root / "offsets", offsets)
            observed, digest = parse_offsets(root / "offsets", 2013)
            self.assertEqual(observed, values)
            self.assertEqual(digest, hashlib.sha256(offsets).hexdigest())
            nonce = "0123456789abcdef" * 4
            event = EVENT.pack(
                EVENT_MAGIC, 1, 77, 20130, 2052, 0x1234, 100, 9, nonce.encode()
            )
            self.immutable(root / "event", event)
            self.assertEqual(
                parse_event(root / "event", 2013, nonce, 77)["source"], "4660"
            )
            (root / "offsets").chmod(0o644)
            with self.assertRaises(ValueError):
                parse_offsets(root / "offsets", 2013)

    def test_offset_hostiles_reject(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "offsets"
            for values in ([0] * 513, [1] + [20130] * 512, [0, 2, 1] + [20130] * 510):
                path.write_bytes(struct.pack("<513I", *values))
                path.chmod(0o444)
                with self.assertRaises(ValueError):
                    parse_offsets(path, 2013)
                path.unlink()

    def test_model_manifest_crosslinks_every_indexed_shard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text("{}")
            (root / "a.safetensors").write_bytes(b"a")
            (root / "b.safetensors").write_bytes(b"bb")
            index = {"weight_map": {"x": "a.safetensors", "y": "b.safetensors"}}
            (root / "model.safetensors.index.json").write_text(json.dumps(index))
            fields = model_manifest(root)
            self.assertEqual(fields["shard.count"], "2")
            validate_model(fields, root)
            omitted = dict(fields)
            omitted["shard.count"] = "1"
            for suffix in ("path", "sha256", "size"):
                del omitted[f"shard.001.{suffix}"]
            with self.assertRaises(ValueError):
                validate_model(omitted, root)
            (root / "b.safetensors").write_bytes(b"drift")
            with self.assertRaises(ValueError):
                validate_model(fields, root)

    def test_capture_plan_requires_exact_target_route(self) -> None:
        server = "/sealed/spark"
        plan = {
            "schema": "oi640-capture-plan-v1",
            "model": "/model",
            "model_manifest": "/model.kv",
            "server": server,
            "server_receipt": "/server.kv",
            "hook": "/hook.so",
            "hook_compile_receipt": "/hook.kv",
            "output": "/out",
            "health_url": "http://127.0.0.1:8998/health",
            "request_url": "http://127.0.0.1:8998/v1/chat/completions",
            "base_argv": [
                server,
                "serve",
                "--kernel-target",
                "qwen3.8-flash-next",
                "--model-from-path",
                "/model",
                "--max-num-seqs",
                "1",
                "--port",
                "8998",
            ],
            "base_env": {
                "ATLAS_DUMP_EXPERT_IDS": "1",
                "ATLAS_QWEN4_ATTN_PREFILL_BATCH": "1",
                "ATLAS_QWEN4_SSM_PREFILL_BATCH": "0",
            },
            "nonce": "0123456789abcdef" * 4,
            "shapes": {
                str(m): {"request": f"/request-{m}.json", "request_sha256": "b" * 64}
                for m in (2013, 8192)
            },
        }
        validate_plan(plan)
        for mutate in (
            lambda p: p["shapes"].pop("8192"),
            lambda p: p["base_env"].update({"ATLAS_DUMP_EXPERT_IDS": "0"}),
            lambda p: p["base_argv"].remove("qwen3.8-flash-next"),
            lambda p: p["base_argv"].extend(["--port", "8998"]),
        ):
            hostile = json.loads(json.dumps(plan))
            mutate(hostile)
            with self.assertRaises((ValueError, IndexError)):
                validate_plan(hostile)

    def test_build_manifest_binds_sources_model_and_final_binary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}")
            (model / "a.safetensors").write_bytes(b"a")
            (model / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"x": "a.safetensors"}})
            )
            model_path = root / "model.kv"
            self.immutable(model_path, encode_kv(model_manifest(model)))
            sources = []
            for number in range(22):
                source = root / f"source-{number:02}.cc"
                source.write_text(f"source {number}\n")
                sources.append(source)
            binary = root / "gate"
            shutil.copy2("/bin/true", binary)
            binary.chmod(0o555)
            binary_id = identity(binary, 0o555)
            compile_fields = {
                "schema": "oi640-compile-v1",
                "profile": "release",
                "target": "sm_121a",
                "source.count": "22",
                "source.bundle_sha256": source_bundle(sources, root),
                "model.manifest.sha256": sha(model_path),
                "compile.argv_sha256": "a" * 64,
                "compile.env_sha256": "b" * 64,
                "toolchain.sha256": "c" * 64,
                "build.nonce": "0123456789abcdef" * 4,
            }
            compile_fields.update(
                {f"binary.{key}": value for key, value in binary_id.items()}
            )
            compile_path = root / "compile.kv"
            self.immutable(compile_path, encode_kv(compile_fields))
            args = SimpleNamespace(
                root=root,
                source=sources,
                model=model,
                model_manifest=model_path,
                compile_receipt=compile_path,
                binary=binary,
                output=root / "build.kv",
            )
            create_build(args)
            self.assertEqual(
                decode_kv((root / "build.kv").read_bytes())["source.count"], "22"
            )
            binary.chmod(0o755)
            binary.write_bytes(binary.read_bytes() + b"drift")
            binary.chmod(0o555)
            args.output = root / "hostile.kv"
            with self.assertRaises(ValueError):
                create_build(args)

    def test_source_requires_real_copy_receipts_and_release_binding(self) -> None:
        here = Path(__file__).resolve().parent
        hook = (
            here / "moe_w4a16_orig_i640_compact_prefill_capture_preload.c"
        ).read_text()
        command = (here / "moe_w4a16_orig_i640_compact_prefill_capture.py").read_text()
        runner = (
            command
            + (here / "moe_w4a16_orig_i640_compact_prefill_capture_run.py").read_text()
        )
        consumer = (
            here / "moe_w4a16_orig_i640_compact_prefill_microgate_manifest.cuh"
        ).read_text() + (
            here / "moe_w4a16_orig_i640_compact_prefill_microgate_provenance.cuh"
        ).read_text()
        for token in (
            "cuMemcpyDtoH_v2",
            "OI640_BYTES",
            "O_EXCL",
            "real-target-route-v1",
        ):
            self.assertIn(token, hook)
        for token in (
            "prompt_tokens",
            'process_bytes(process.pid, "maps")',
            "listener survived teardown",
            "QWEN4_PREFILL_SELECTOR_RECEIPT",
            "listener_inode",
            "validate_model",
        ):
            self.assertIn(token, runner)
        for token in (
            "identity_fields",
            'field(proof.build, "profile", "release")',
            "verify_model_manifest",
            "provenance_stable",
            "compile_receipt.binary",
            "producer.receipt",
            "hook.compile_receipt",
            "server.build_receipt",
            "OI640_MODEL_MANIFEST_SHA256",
        ):
            self.assertIn(token, consumer)


if __name__ == "__main__":
    unittest.main(verbosity=2)
