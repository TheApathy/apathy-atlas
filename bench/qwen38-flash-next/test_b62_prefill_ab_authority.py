# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import hashlib
import shutil
import sys
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path
from unittest import mock

import b62_prefill_ab_authority as authority
import b62_prefill_ab as launcher
import b62_prefill_ab_contract as contract
import b62_prefill_ab_http as http_io
import b62_prefill_ab_model as model_identity

HERE = Path(__file__).resolve().parent
HEX_A = "a" * 64
HEX_B = "b" * 64
HEX_C = "c" * 64
GPU_UUID = "GPU-00000000-0000-0000-0000-000000000000"


class FakeResponse:
    status = 200

    def __init__(self) -> None:
        self.read_size = None

    def read(self, size: int) -> bytes:
        self.read_size = size
        return b"x" * size

    def getheader(self, name: str) -> str:
        return "application/json"


class FakeConnection:
    response = FakeResponse()

    def __init__(self, *args, **kwargs) -> None:
        pass

    def request(self, *args, **kwargs) -> None:
        pass

    def getresponse(self) -> FakeResponse:
        return self.response

    def close(self) -> None:
        pass


class AuthorityTests(unittest.TestCase):
    def test_repaired_bundle_is_exact_and_mutation_rejects(self) -> None:
        paths = [HERE / name for name in authority.HARNESS_NAMES]
        result = authority.attest_harness(paths)
        self.assertEqual(result["bundle_sha256"], authority.HARNESS_BUNDLE_SHA256)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            copies = []
            for source in paths:
                target = root / source.name
                shutil.copyfile(source, target)
                copies.append(target)
            authority.attest_harness(copies)
            target = root / "b62_prefill_ab_contract.py"
            target.write_bytes(target.read_bytes() + b"# drift\n")
            with self.assertRaisesRegex(RuntimeError, "bundle identity drift"):
                authority.attest_harness(copies)

    def test_authoritative_model_placeholder_fails_before_access(self) -> None:
        self.assertFalse(
            model_identity.identity.HEX64.fullmatch(contract.MODEL_MANIFEST_SHA256)
        )
        with self.assertRaisesRegex(RuntimeError, "not pinned"):
            model_identity.load_manifest(
                contract.MODEL_MANIFEST, contract.MODEL_MANIFEST_SHA256
            )

    def test_launcher_placeholder_fails_before_gpu_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"
            argv = [
                "b62_prefill_ab.py",
                "--output-dir",
                str(output),
                "--execute",
                contract.EXECUTE,
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(
                    launcher.inventory, "attest_inventory"
                ) as gpu_inventory,
            ):
                self.assertEqual(launcher.main(), 2)
            gpu_inventory.assert_not_called()
            self.assertEqual([path.name for path in output.iterdir()], ["failure.json"])
            self.assertIn("not pinned", (output / "failure.json").read_text())

    def test_root_team_authorization_mode_time_release_and_boot(self) -> None:
        issued = 1_700_000_000
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reservation, team, boot = (
                root / name for name in ("auth.json", "TEAM", "boot")
            )
            boot.write_text("test-boot-id\n")
            boot_hash = hashlib.sha256(boot.read_bytes()).hexdigest()
            patchers = (
                mock.patch.object(authority, "RESERVATION", reservation),
                mock.patch.object(authority, "TEAM_INBOX", team),
                mock.patch.object(authority, "BOOT_ID", boot),
                mock.patch.object(contract, "MODEL_MANIFEST_SHA256", HEX_A),
                mock.patch.object(contract, "MODEL_CONTENT_ROOT_SHA256", HEX_B),
            )
            with patchers[0], patchers[1], patchers[2], patchers[3], patchers[4]:
                document = {
                    "schema": authority.SCHEMA,
                    "lane": authority.LANE,
                    "issuer": "/root",
                    "resource": "local-gb10-exclusive",
                    "port": contract.PORT,
                    "binary_sha256": contract.BINARY_SHA256,
                    "harness_bundle_sha256": authority.HARNESS_BUNDLE_SHA256,
                    "model_manifest_sha256": HEX_A,
                    "model_content_root_sha256": HEX_B,
                    "nonce_sha256": HEX_C,
                    "boot_id_sha256": boot_hash,
                    "gpu_uuid": GPU_UUID,
                    "nvidia_smi_sha256": "d" * 64,
                    "issued_at_unix": issued,
                    "expires_at_unix": issued + 600,
                    "issued_at_utc": datetime.fromtimestamp(issued, UTC).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                    "authorization": "b62-prefill-ab-only",
                }
                raw = contract.canonical_bytes(document) + b"\n"
                reservation.write_bytes(raw)
                reservation.chmod(0o444)
                team.write_text(authority._claim_line(document) + "\n")
                result = authority.load_reservation(now=issued + 1)
                self.assertEqual(result["document"], document)
                with team.open("a") as stream:
                    stream.write("unrelated append-only coordination\n")
                self.assertEqual(authority.load_reservation(now=issued + 1), result)
                reservation.chmod(0o400)
                with self.assertRaisesRegex(RuntimeError, "mode drift"):
                    authority.load_reservation(now=issued + 1)
                reservation.chmod(0o444)
                with self.assertRaisesRegex(RuntimeError, "stale"):
                    authority.load_reservation(now=issued + 601)
                with team.open("a") as stream:
                    stream.write(
                        f"x /root RELEASE-GPU: {authority.LANE} nonce_sha256={HEX_C}\n"
                    )
                with self.assertRaisesRegex(RuntimeError, "released"):
                    authority.load_reservation(now=issued + 1)

    def test_failure_artifacts_are_purged_and_semantic_is_timing_free(self) -> None:
        semantic = {
            "content_sha256": HEX_A,
            "ttft_ms": 2.0,
            "prompt_tokens_per_second": 3.0,
        }
        self.assertEqual(
            http_io.persistent_semantic(semantic), {"content_sha256": HEX_A}
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for arm in contract.ARM_ORDER:
                for suffix in ("server.log", "http.jsonl"):
                    path = root / f"{arm}-{suffix}"
                    path.write_text("prompt_tokens_per_second=9999")
                    path.chmod(0o444)
            authority.purge_unqualified(root)
            self.assertEqual(list(root.iterdir()), [])

    def test_http_body_read_is_bounded_before_size_rejection(self) -> None:
        FakeConnection.response = FakeResponse()
        with mock.patch.object(http_io.http.client, "HTTPConnection", FakeConnection):
            with self.assertRaisesRegex(RuntimeError, "bounded parser"):
                http_io.http_json("GET", "/v1/models")
        self.assertEqual(FakeConnection.response.read_size, (64 << 20) + 1)

    def test_ple_intermediate_symlink_rejects(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            real = root / "real"
            real.mkdir()
            item = real / "x.bin"
            item.write_bytes(b"x")
            (root / "ple-offload").symlink_to(real, target_is_directory=True)
            record = {
                "relative": "ple-offload/x.bin",
                "sha256": hashlib.sha256(b"x").hexdigest(),
                "size": 1,
                "mode": 0o644,
                "device": item.stat().st_dev,
                "inode": item.stat().st_ino,
                "mtime_ns": item.stat().st_mtime_ns,
            }
            with mock.patch.object(contract, "MODEL", root):
                with self.assertRaisesRegex(RuntimeError, "not canonical"):
                    model_identity._attest_record(record, hash_content=True)

    def test_all_sources_have_spdx_and_cap(self) -> None:
        for name in authority.HARNESS_NAMES:
            lines = (HERE / name).read_text().splitlines()
            self.assertEqual(lines[0], "# SPDX-License-Identifier: AGPL-3.0-only")
            self.assertLessEqual(len(lines), 250, f"{name}: {len(lines)}")


if __name__ == "__main__":
    unittest.main()
