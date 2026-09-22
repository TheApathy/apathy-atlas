# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import b63_ple_prefill_ab_inventory as inventory
import b63_ple_prefill_ab_process as process_identity

GPU_UUID = "GPU-00000000-0000-0000-0000-000000000000"
RESERVATION = {
    "document": {
        "gpu_uuid": GPU_UUID,
        "nvidia_smi_sha256": "a" * 64,
    }
}
EVIDENCE = {"tool": {}, "argv_sha256": "b" * 64, "stdout_sha256": "c" * 64}


class InventoryTests(unittest.TestCase):
    def test_exact_gb10_and_compute_client_set(self) -> None:
        results = [
            ([f"{GPU_UUID}, NVIDIA GB10"], EVIDENCE),
            ([f"123, {GPU_UUID}"], EVIDENCE),
        ]
        with mock.patch.object(inventory, "_query", side_effect=results):
            result = inventory.attest_inventory(RESERVATION, {123})
        self.assertEqual(
            result["compute_clients"], [{"pid": 123, "gpu_uuid": GPU_UUID}]
        )

    def test_foreign_duplicate_and_wrong_gpu_reject(self) -> None:
        cases = (
            ([f"{GPU_UUID}, NVIDIA GB10"], [f"999, {GPU_UUID}"], {123}),
            (
                [f"{GPU_UUID}, NVIDIA GB10"],
                [f"123, {GPU_UUID}", f"123, {GPU_UUID}"],
                {123},
            ),
            ([f"{GPU_UUID}, Other GPU"], [], set()),
        )
        for gpu_lines, app_lines, expected in cases:
            with (
                self.subTest(lines=app_lines),
                mock.patch.object(
                    inventory,
                    "_query",
                    side_effect=[(gpu_lines, EVIDENCE), (app_lines, EVIDENCE)],
                ),
                self.assertRaises(RuntimeError),
            ):
                inventory.attest_inventory(RESERVATION, expected)

    def test_nvidia_smi_tool_hash_and_mode_are_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tool = Path(directory) / "nvidia-smi"
            tool.write_bytes(b"tool")
            tool.chmod(0o555)
            import hashlib

            digest = hashlib.sha256(b"tool").hexdigest()
            with mock.patch.object(inventory, "NVIDIA_SMI", tool):
                self.assertEqual(inventory._tool(digest)["mode"], 0o555)
                tool.chmod(0o755)
                with self.assertRaisesRegex(RuntimeError, "mode/content"):
                    inventory._tool(digest)

    def test_owned_session_descendants_are_group_killed(self) -> None:
        actions: list[str] = []
        with (
            mock.patch.object(
                process_identity, "session_members", side_effect=[{8}, set(), set()]
            ),
            mock.patch.object(process_identity.os, "killpg") as killpg,
        ):
            process_identity.drain_session(7, actions)
        killpg.assert_called_once_with(7, process_identity.signal.SIGKILL)
        self.assertEqual(actions, ["session-sigkill"])

    def test_launch_and_signal_source_pin_new_session_and_exact_elf(self) -> None:
        root = Path(__file__).resolve().parent
        launcher = (root / "b63_ple_prefill_ab.py").read_text()
        process = (root / "b63_ple_prefill_ab_process.py").read_text()
        self.assertIn("start_new_session=True", launcher)
        self.assertIn("!= contract.BINARY_SHA256", process)
        self.assertIn("os.killpg", process)


if __name__ == "__main__":
    unittest.main()
