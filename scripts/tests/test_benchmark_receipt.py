# SPDX-License-Identifier: AGPL-3.0-only

"""Offline contracts for immutable benchmark provenance receipts."""

import importlib.util
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "benchmark_receipt.py"
BOOT_ID_A = "abcdef01-2222-4333-8444-555555555555"
BOOT_ID_B = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"


class ModelsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/v1/models":
            self.send_error(404)
            return
        payload = json.dumps({"data": [{"id": "unit-model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass


def load_receipt():
    spec = importlib.util.spec_from_file_location("atlas_benchmark_receipt", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class BenchmarkReceiptTests(unittest.TestCase):
    def test_listener_address_must_match_exact_requested_loopback(self):
        ipv4 = self.receipt.proc_listen_address("127.0.0.1")
        ipv6 = self.receipt.proc_listen_address("::1")
        self.assertEqual(ipv4[0], "tcp")
        self.assertEqual(ipv6[0], "tcp6")
        with self.assertRaisesRegex(ValueError, "literal loopback"):
            self.receipt.proc_listen_address("localhost")
        self.assertIsNone(
            self.receipt.RejectRedirects().redirect_request(
                None, None, 302, "redirect", {}, "http://example.invalid"
            )
        )

    def test_activation_checks_listener_before_expensive_repo_and_model_hashes(self):
        planned = self.receipt.build_receipt(
            repo=self.repo,
            binary=Path(sys.executable),
            model=self.model,
            drafter=None,
            argv=self.receipt.process_argv(os.getpid()),
            environment={},
            required_binary_strings=[],
        )
        with mock.patch.object(
            self.receipt, "process_owns_listen_address", return_value=False
        ), mock.patch.object(
            self.receipt,
            "git_identity",
            side_effect=AssertionError("expensive repository hash ran before readiness"),
        ):
            with self.assertRaisesRegex(ValueError, "does not own listening port"):
                self.receipt.current_activation(
                    planned, os.getpid(), "http://127.0.0.1:65534", 65534
                )

    def test_full_environment_digest_is_order_independent_and_value_sensitive(self):
        first = self.receipt.environment_sha256({"B": "2", "A": "1"})
        second = self.receipt.environment_sha256({"A": "1", "B": "2"})
        changed = self.receipt.environment_sha256({"A": "1", "B": "3"})
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)

    def test_planned_digest_uses_kernel_environment_not_mutable_python_view(self):
        kernel_environment = self.receipt.read_process_environment(os.getpid())
        mutable_view = dict(os.environ)
        mutable_view["_"] = "/synthetic/planner-helper"
        with mock.patch.object(self.receipt.os, "environ", mutable_view):
            envelope = self.build(environment={})
        self.assertEqual(
            envelope["manifest"]["full_environment_sha256"],
            self.receipt.environment_sha256(kernel_environment),
        )

    def setUp(self):
        self.receipt = load_receipt()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        subprocess.run(["git", "init", "-q"], cwd=self.repo, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.invalid"],
            cwd=self.repo,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test"], cwd=self.repo, check=True
        )
        (self.repo / "tracked.txt").write_text("base\n", encoding="utf-8")
        subprocess.run(["git", "add", "tracked.txt"], cwd=self.repo, check=True)
        subprocess.run(["git", "commit", "-qm", "base"], cwd=self.repo, check=True)

        self.binary = self.root / "spark"
        self.binary.write_bytes(b"deepseek_v4\0ATLAS_EXL3_PREFILL_CHUNK\0")
        self.binary.chmod(0o755)
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "config.json").write_text(
            json.dumps({"model_type": "deepseek_v4"}), encoding="utf-8"
        )
        (self.model / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"x": "model-1.safetensors"}}), encoding="utf-8"
        )
        (self.model / "model-1.safetensors").write_bytes(b"weight-shard-a")
        (self.model / "tokenizer.json").write_text("{}\n", encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def build(self, **overrides):
        args = {
            "repo": self.repo,
            "binary": self.binary,
            "model": self.model,
            "drafter": None,
            "argv": [str(self.binary), "serve", str(self.model), "--port", "8977"],
            "environment": {"ATLAS_EXL3_PREFILL_CHUNK": "1"},
            "required_binary_strings": ["deepseek_v4", "ATLAS_EXL3_PREFILL_CHUNK"],
        }
        args.update(overrides)
        return self.receipt.build_receipt(**args)

    def test_receipt_binds_git_binary_model_tokenizer_argv_and_environment(self):
        envelope = self.build()
        manifest = envelope["manifest"]
        self.assertEqual(manifest["schema"], "atlas-benchmark-receipt-v2")
        self.assertEqual(manifest["receipt_state"], "PLANNED")
        self.assertFalse(manifest["git"]["dirty"])
        self.assertEqual(len(manifest["git"]["commit"]), 40)
        self.assertEqual(manifest["model"]["model_type"], "deepseek_v4")
        self.assertIn("config.json", manifest["model"]["files"])
        self.assertIn("model.safetensors.index.json", manifest["model"]["files"])
        self.assertIn("tokenizer.json", manifest["model"]["files"])
        self.assertEqual(
            manifest["model"]["weight_shards"]["model-1.safetensors"]["size"],
            len(b"weight-shard-a"),
        )
        self.assertEqual(manifest["argv"][-2:], ["--port", "8977"])
        self.assertEqual(manifest["environment"], {"ATLAS_EXL3_PREFILL_CHUNK": "1"})
        self.assertEqual(len(manifest["full_environment_sha256"]), 64)
        self.assertTrue(self.receipt.verify_envelope(envelope))

    def test_dirty_and_untracked_content_changes_tree_identity(self):
        clean = self.build()["manifest"]["git"]["tree_sha256"]
        (self.repo / "tracked.txt").write_text("changed\n", encoding="utf-8")
        (self.repo / "new.txt").write_text("untracked\n", encoding="utf-8")
        dirty = self.build()["manifest"]["git"]
        self.assertTrue(dirty["dirty"])
        self.assertNotEqual(dirty["tree_sha256"], clean)

    def test_missing_identity_files_and_binary_gate_fail_closed(self):
        self.binary.unlink()
        with self.assertRaisesRegex(ValueError, "binary"):
            self.build()
        self.binary.write_bytes(b"deepseek_v4 only")
        self.binary.chmod(0o755)
        with self.assertRaisesRegex(ValueError, "ATLAS_EXL3_PREFILL_CHUNK"):
            self.build()
        (self.model / "model.safetensors.index.json").unlink()
        with self.assertRaisesRegex(ValueError, "index"):
            self.build(required_binary_strings=[])

    def test_weight_shards_are_required_hashed_and_confined(self):
        before = self.build()["manifest"]["model"]["weight_shards"]
        (self.model / "model-1.safetensors").write_bytes(b"weight-shard-b")
        after = self.build()["manifest"]["model"]["weight_shards"]
        self.assertNotEqual(before, after)
        (self.model / "model-1.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "shard is missing"):
            self.build()

        outside = self.root / "outside.safetensors"
        outside.write_bytes(b"outside")
        (self.model / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"x": "../outside.safetensors"}}),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "escapes checkpoint"):
            self.build()

        link = self.model / "link.safetensors"
        link.symlink_to(outside)
        (self.model / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"x": "link.safetensors"}}), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "escapes checkpoint"):
            self.build()

    def test_secret_shaped_environment_keys_are_rejected(self):
        for key in ["API_TOKEN", "HF_TOKEN", "DATABASE_PASSWORD", "CLIENT_SECRET"]:
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "secret"):
                self.build(environment={key: "must-not-land"})

    def test_secret_bearing_argv_is_rejected(self):
        forbidden = [
            [str(self.binary), "serve", "--api-key", "must-not-land"],
            [str(self.binary), "serve", "--token=must-not-land"],
            [str(self.binary), "serve", "https://user:pass@example.invalid/model"],
            [str(self.binary), "serve", "https://example.invalid/model?api_key=hidden"],
        ]
        for argv in forbidden:
            with self.subTest(argv=argv), self.assertRaisesRegex(
                ValueError, "forbidden"
            ):
                self.build(argv=argv)

    def test_manifest_digest_detects_mutation(self):
        envelope = self.build()
        receipt_path = self.root / "receipt.json"
        receipt_path.write_text(json.dumps(envelope), encoding="utf-8")
        self.assertEqual(self.receipt.read_envelope(receipt_path), envelope)
        envelope["manifest"]["argv"].append("--changed")
        self.assertFalse(self.receipt.verify_envelope(envelope))
        receipt_path.write_text(json.dumps(envelope), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "digest"):
            self.receipt.read_envelope(receipt_path)

    def test_v1_receipts_are_rejected_after_boot_binding_schema_upgrade(self):
        envelope = self.build()
        envelope["manifest"]["schema"] = "atlas-benchmark-receipt-v1"
        envelope["manifest_sha256"] = self.receipt.hashlib.sha256(
            self.receipt.canonical_json(envelope["manifest"])
        ).hexdigest()
        receipt_path = self.root / "legacy.receipt.json"
        receipt_path.write_text(json.dumps(envelope), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "schema"):
            self.receipt.read_envelope(receipt_path)

    def test_boot_id_parser_requires_canonical_uuid(self):
        self.assertEqual(self.receipt.validate_boot_id(BOOT_ID_A), BOOT_ID_A)
        for value in (
            BOOT_ID_A.upper(),
            "abcdef01222243338444555555555555",
            "not-a-boot-id",
            "",
            None,
        ):
            with self.subTest(value=value), self.assertRaisesRegex(
                ValueError, "boot ID"
            ):
                self.receipt.validate_boot_id(value)

    def test_receipt_output_is_exclusive(self):
        path = self.root / "immutable.receipt.json"
        self.receipt.write_receipt(path, "first\n")
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.receipt.write_receipt(path, "second\n")
        self.assertEqual(path.read_text(encoding="utf-8"), "first\n")

    def test_gpu_snapshot_parses_identity_and_allows_unavailable_dynamic_fields(self):
        output = (
            "GPU-unit, NVIDIA GB10, 580.126.09, P0, 49, 12.15, 2398, "
            "[N/A], [N/A], [N/A], [N/A]\n"
        )
        result = subprocess.CompletedProcess([], 0, stdout=output, stderr="")
        with mock.patch.object(self.receipt.subprocess, "run", return_value=result):
            snapshot = self.receipt.gpu_snapshot()
        self.assertEqual(snapshot["identity"]["uuid"], "GPU-unit")
        self.assertEqual(snapshot["identity"]["name"], "NVIDIA GB10")
        self.assertEqual(snapshot["state"]["clocks.sm"], "2398")
        self.assertEqual(snapshot["state"]["memory.total"], "[N/A]")

    def test_active_receipt_binds_live_pid_port_model_and_checkpoint(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), ModelsHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        port = server.server_address[1]
        base_url = f"http://127.0.0.1:{port}"
        planned = self.receipt.build_receipt(
            repo=self.repo,
            binary=Path(sys.executable),
            model=self.model,
            drafter=None,
            argv=self.receipt.process_argv(os.getpid()),
            environment={},
            required_binary_strings=[],
        )
        gpu = {
            "identity": {
                "uuid": "GPU-unit",
                "name": "NVIDIA GB10",
                "driver_version": "580.126.09",
            },
            "state": {
                "pstate": "P0",
                "temperature.gpu": "49",
                "power.draw": "12.15",
                "clocks.sm": "2398",
                "clocks.mem": "[N/A]",
                "memory.total": "[N/A]",
                "memory.used": "[N/A]",
                "memory.free": "[N/A]",
            },
        }
        try:
            with mock.patch.object(
                self.receipt, "gpu_snapshot", return_value=gpu
            ), mock.patch.object(
                self.receipt, "host_boot_id", return_value=BOOT_ID_A
            ) as boot_mock:
                active = self.receipt.activate_envelope(
                    planned, os.getpid(), base_url, port
                )
                self.assertTrue(self.receipt.verify_envelope(active))
                self.assertEqual(
                    self.receipt.receipt_state(active, base_url), "ACTIVE_VERIFIED"
                )
                self.assertEqual(active["activation"]["model_id"], "unit-model")
                self.assertEqual(active["activation"]["host_boot_id"], BOOT_ID_A)
                self.assertEqual(
                    self.receipt.receipt_digest(active), active["activation_sha256"]
                )
                missing_boot = json.loads(json.dumps(active))
                del missing_boot["activation"]["host_boot_id"]
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    self.receipt.validate_activation(missing_boot)
                malformed_boot = json.loads(json.dumps(active))
                malformed_boot["activation"]["host_boot_id"] = "not-canonical"
                with self.assertRaisesRegex(ValueError, "boot ID"):
                    self.receipt.validate_activation(malformed_boot)
                boot_mock.return_value = BOOT_ID_B
                with self.assertRaisesRegex(ValueError, "runtime identity changed"):
                    self.receipt.verify_active_envelope(active, base_url)
                boot_mock.return_value = BOOT_ID_A
                with self.assertRaisesRegex(ValueError, "listening port"):
                    self.receipt.activate_envelope(
                        planned, os.getpid(), base_url, port + 1
                    )
                (self.repo / "tracked.txt").write_text(
                    "mutated-after-planning\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, "repository changed"):
                    self.receipt.verify_active_envelope(active, base_url)
                (self.repo / "tracked.txt").write_text("base\n", encoding="utf-8")
                (self.model / "model-1.safetensors").write_bytes(
                    b"mutated-after-launch"
                )
                with self.assertRaisesRegex(ValueError, "checkpoint changed"):
                    self.receipt.verify_active_envelope(active, base_url)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
