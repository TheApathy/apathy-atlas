# SPDX-License-Identifier: AGPL-3.0-only
"""Hostile CPU-only tests for b61 capture provenance and cleanup."""

from __future__ import annotations

import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nsys_b61_capture_cleanup as cleanup_support
import nsys_b61_capture_logs as log_support
import nsys_b61_capture_model as model_support
import nsys_b61_capture_process as process_support
import nsys_b61_capture_support as support
import nsys_b61_capture_trace as trace_support


def valid_server_log() -> str:
    per_request = "\n".join(
        [
            "Session abc: 38 prompt tokens, tools=false (0 defined)",
            "Prefill first token: 71093",
            "Qwen4 PLE segmented graph captured for slot=0 (layers 1..48 + head)",
        ]
    )
    return "\n".join(
        [
            "Qwen4 PLE: discovered sparse offload manifest /model/ple-offload/manifest.json",
            "Prefix caching: disabled",
            "WARN Skipping MoE weight transposition (NVFP4). Prefill will use fallback grouped GEMM.",
            "Qwen3.5 weight loader: 48 layers (12 attention, 36 linear_attn)",
            "Qwen4 PLE sparse NVFP4 offload enabled cache_mb=0 io_mode=Direct",
            "No MTP weights found — speculative decoding disabled",
            "SSM snapshot pool: Marconi 0 slots",
            "Scheduler started (batched mode, max_batch=1, mtp=false, ngram=false, num_drafts=0, "
            "policy=fifo, chunked_prefill=true, max_prefill_tokens=512)",
            "Listening on 127.0.0.1:8998",
            "Request: model=qwen3.8-flash-next, stream=false, temp=Some(0.0), max_tokens=64",
            per_request,
            "Chunked prefill start: 38 prompt tokens, chunk_size=38, max_tokens=64",
            "Prefilled (single chunk): seq_len=38, remaining=63",
            "Done: 64 tokens (length)",
            "Request: model=qwen3.8-flash-next, stream=false, temp=Some(0.0), max_tokens=400",
            per_request,
            "Chunked prefill start: 38 prompt tokens, chunk_size=38, max_tokens=400",
            "Prefilled (single chunk): seq_len=38, remaining=399",
            "Done: 400 tokens (length)",
        ]
    )


class LogCensusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / "server.log"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_exact_server_census_and_route_rejection(self) -> None:
        self.path.write_text(valid_server_log())
        got = log_support.validate_server_log(self.path)
        self.assertEqual((got["request_64"], got["request_400"]), (1, 1))
        self.path.write_text(valid_server_log() + "\nDFlash propose engaged")
        with self.assertRaisesRegex(RuntimeError, "selector/route"):
            log_support.validate_server_log(self.path)

    def test_unknown_warning_fails(self) -> None:
        self.path.write_text(valid_server_log() + "\nWARN surprise fallback")
        with self.assertRaisesRegex(RuntimeError, "warning/error/fallback"):
            log_support.validate_server_log(self.path)

    def test_control_sequence_and_status_are_exact(self) -> None:
        labels = ["start", "stop", "cuda_gpu_trace", "cuda_gpu_kern_gb_sum", "shutdown"]
        events = [{"label": label, "returncode": 0} for label in labels]
        text = "\n".join(
            f'CONTROL {{"label":"{label}"}}\nCONTROL_RESULT {{"label":"{label}"}}'
            for label in labels
        )
        self.path.write_text(text)
        self.assertEqual(
            log_support.validate_control_log(self.path, events)["event_count"], 5
        )
        events[-1]["returncode"] = 1
        with self.assertRaisesRegex(RuntimeError, "sequence/status"):
            log_support.validate_control_log(self.path, events)


class TraceHostileTests(unittest.TestCase):
    def test_compound_kernel_name_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path, db_path = root / "trace.csv", root / "trace.sqlite"
            names = [
                ("paged_decode_attn_moe_expert", 7),
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
            with self.assertRaisesRegex(RuntimeError, "compound kernel-family"):
                trace_support.validate_trace(csv_path, db_path)


class CleanupAndAdmissionTests(unittest.TestCase):
    def test_cleanup_continues_after_snapshot_and_target_failures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            server_log = (root / "server.log").open("xb")
            control_log = (root / "nsys-control.log").open("xb")
            launcher = mock.Mock(pid=12)
            receipt = {}
            with (
                mock.patch.object(
                    cleanup_support, "_snapshot_owned", side_effect=OSError("race")
                ),
                mock.patch.object(cleanup_support, "_control"),
                mock.patch.object(
                    process_support,
                    "terminate_exact_target",
                    side_effect=OSError("drift"),
                ),
                mock.patch.object(
                    process_support, "reap_launcher", return_value={"exit_code": 0}
                ),
                mock.patch.object(process_support, "assert_cleanup"),
                mock.patch.object(log_support, "write_json"),
            ):
                qualified = cleanup_support.finalize(
                    output=root,
                    receipt=receipt,
                    candidate=False,
                    launcher=launcher,
                    target_pid=22,
                    before={"pid": 22, "starttime_ticks": 9},
                    collection=True,
                    session="unit",
                    server_log=server_log,
                    control_log=control_log,
                    events=[],
                    owned_descendants=[],
                )
        self.assertFalse(qualified)
        self.assertTrue(server_log.closed and control_log.closed)
        self.assertTrue(
            any("descendant snapshot" in item for item in receipt["cleanup_errors"])
        )
        self.assertTrue(any("target" in item for item in receipt["cleanup_errors"]))

    def test_launcher_uses_terminate_then_kill(self) -> None:
        launcher = mock.Mock()
        launcher.wait.side_effect = [
            subprocess.TimeoutExpired("nsys", 10),
            subprocess.TimeoutExpired("nsys", 5),
            -9,
        ]
        got = process_support.reap_launcher(launcher)
        self.assertEqual(got, {"exit_code": -9, "actions": ["terminate", "kill"]})
        launcher.terminate.assert_called_once_with()
        launcher.kill.assert_called_once_with()

    def test_target_identity_drift_is_never_signaled(self) -> None:
        expected = {"starttime_ticks": 11, "pid": 123}
        with (
            mock.patch.object(process_support, "original_pid_alive", return_value=True),
            mock.patch.object(process_support.os, "kill") as kill,
        ):
            with self.assertRaisesRegex(RuntimeError, "reused/drifted"):
                process_support.terminate_exact_target(
                    123, expected, lambda _: {"drift": True}
                )
        kill.assert_not_called()

    def test_candidate_capture_rejects_before_model_read(self) -> None:
        with mock.patch.object(model_support.support, "require_regular") as regular:
            with self.assertRaisesRegex(RuntimeError, "authorization missing"):
                model_support.capture_candidate(Path("unused"), "wrong")
        regular.assert_not_called()

    @unittest.skipUnless(support.NSYS.is_file(), "pinned Nsight CLI unavailable")
    def test_real_nsys_capability_contract(self) -> None:
        got = support.nsys_capabilities(support.NSYS)
        self.assertEqual(
            set(got),
            {
                "version",
                "launch",
                "start",
                "stop",
                "cancel",
                "shutdown",
                "reports",
                "report_catalog",
            },
        )
        self.assertTrue(all(len(digest) == 64 for digest in got.values()))


if __name__ == "__main__":
    unittest.main()
