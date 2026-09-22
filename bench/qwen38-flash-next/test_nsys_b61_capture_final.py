# SPDX-License-Identifier: AGPL-3.0-only
"""Final hostile tests for bounded control, cleanup, and trace agreement."""

from __future__ import annotations

import signal
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nsys_b61_capture_cleanup as cleanup_support
import nsys_b61_capture_control as control_support
import nsys_b61_capture_process as process_support
import nsys_b61_capture_trace as trace_support


def valid_trace(root: Path) -> tuple[Path, Path]:
    csv_path, db_path = root / "trace.csv", root / "trace.sqlite"
    names = [
        ("paged_decode_attn", 7),
        ("gated_delta_rule", 7),
        ("moe_gate", 7),
        ("qwen4_ple_rows", 0),
        ("argmax_bf16", 0),
    ]
    csv_path.write_text(
        "Duration (ns),Name\n"
        + "".join(f"5,{name}\n" for name, _ in names for _ in range(400))
    )
    with sqlite3.connect(db_path) as db:
        db.execute("CREATE TABLE StringIds (id INTEGER, value TEXT)")
        db.execute(
            "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
            "(graphNodeId INTEGER, start INTEGER, end INTEGER, demangledName INTEGER)"
        )
        for index, (name, graph) in enumerate(names, 1):
            db.execute("INSERT INTO StringIds VALUES (?, ?)", (index, name))
            db.executemany(
                "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?)",
                [(graph, row * 10, row * 10 + 5, index) for row in range(400)],
            )
    return csv_path, db_path


class FinalHostileTests(unittest.TestCase):
    def test_duration_and_unrelated_row_drift_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            csv_path, db_path = valid_trace(Path(directory))
            csv_path.write_text(csv_path.read_text().replace("5,paged", "999,paged", 1))
            with self.assertRaisesRegex(RuntimeError, "duration mismatch"):
                trace_support.validate_trace(csv_path, db_path)
            second = Path(directory) / "second"
            second.mkdir()
            csv_path, db_path = valid_trace(second)
            with csv_path.open("a") as stream:
                stream.write("5,unrelated_kernel\n")
            with self.assertRaisesRegex(RuntimeError, "total row/duration mismatch"):
                trace_support.validate_trace(csv_path, db_path)

    def test_control_timeout_terminates_kills_and_reaps(self) -> None:
        child = mock.Mock()
        child.wait.side_effect = [
            subprocess.TimeoutExpired("nsys", 1),
            subprocess.TimeoutExpired("nsys", 5),
            -9,
        ]
        with mock.patch.object(control_support.subprocess, "Popen", return_value=child):
            got = control_support.run_bounded(
                ["nsys", "stop"],
                cwd=Path("/tmp"),
                env={},
                output=mock.Mock(),
                timeout_seconds=1,
            )
        self.assertTrue(got["timed_out"])
        self.assertEqual(got["actions"], ["terminate", "kill"])
        child.terminate.assert_called_once_with()
        child.kill.assert_called_once_with()

    def test_surviving_owned_descendant_is_exactly_escalated(self) -> None:
        identity = {"pid": 123, "starttime_ticks": 11}
        with (
            mock.patch.object(
                process_support,
                "original_pid_alive",
                side_effect=[True, True, False],
            ),
            mock.patch.object(process_support, "starttime_ticks", return_value=11),
            mock.patch.object(cleanup_support.os, "kill") as kill,
            mock.patch.object(
                cleanup_support.time, "monotonic", side_effect=[0, 6, 6, 12]
            ),
        ):
            got = cleanup_support._terminate_owned(identity)
        self.assertEqual(got, ["sigterm", "sigkill"])
        self.assertEqual(
            kill.call_args_list,
            [mock.call(123, signal.SIGTERM), mock.call(123, signal.SIGKILL)],
        )


if __name__ == "__main__":
    unittest.main()
