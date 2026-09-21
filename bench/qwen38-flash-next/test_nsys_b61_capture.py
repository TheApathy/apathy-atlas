# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only contract tests for the source-free b61 capture harness."""

from __future__ import annotations

import csv
import hashlib
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nsys_b61_capture as capture
import nsys_b61_capture_cleanup as cleanup_support
import nsys_b61_capture_control as control_support
import nsys_b61_capture_logs as log_support
import nsys_b61_capture_model as model_support
import nsys_b61_capture_process as process_support
import nsys_b61_capture_support as support
import nsys_b61_capture_trace as trace_support


class RequestContractTests(unittest.TestCase):
    def test_canonical_request_hashes(self) -> None:
        self.assertEqual(
            hashlib.sha256(support.PROMPT.encode()).hexdigest(),
            "202498f2a136366a7320e9fd95255de1d1ae999ea9c41301903728e6f02759f2",
        )
        expected = {
            64: "629f7d3ec954887c749e8e11f0a83c168dda87b67abc819287378d286ddf59ed",
            400: "607f2a8dd031605b9a2baeae8cd1764a7eb97634f1bc297bf383be59196fccdd",
        }
        for count, digest in expected.items():
            self.assertEqual(
                hashlib.sha256(
                    support.canonical_bytes(support.request_body(count))
                ).hexdigest(),
                digest,
            )

    def test_phase_guard_requires_warm_then_one_request(self) -> None:
        semantic = {"completion_tokens": 64}
        guard = process_support.CaptureGuard()
        with self.assertRaises(RuntimeError):
            guard.collection_started()
        guard.warmup_done(support.request_body(64), semantic)
        guard.collection_started()
        guard.traced_request(support.request_body(400))
        with self.assertRaises(RuntimeError):
            guard.traced_request(support.request_body(400))
        guard.collection_stopped()
        self.assertEqual((guard.phase, guard.traced_requests), ("stopped", 1))

    def test_commands_are_split_and_environment_is_closed(self) -> None:
        launch = capture.launch_command("unit_session")
        start = capture.start_command("unit_session", Path("/tmp/unit-trace"))
        self.assertIn("--cuda-graph-trace=node", launch)
        self.assertIn("--inherit-environment=false", launch)
        self.assertIn("--trace=cuda,nvtx", launch)
        self.assertNotIn("--export=sqlite", launch)
        self.assertEqual(launch[-len(capture.server_argv()) :], capture.server_argv())
        self.assertIn("--export=sqlite", start)
        self.assertIn("--output=/tmp/unit-trace", start)

    def test_no_wall_or_server_rate_is_recorded(self) -> None:
        modules = (
            capture,
            cleanup_support,
            control_support,
            log_support,
            model_support,
            process_support,
            support,
            trace_support,
        )
        text = "".join(Path(module.__file__).read_text() for module in modules)
        self.assertNotIn("wall_seconds", text)
        self.assertNotIn("response_token/s", text)
        self.assertNotIn("tokens_per_second", text)

    def test_wrong_output_fails_semantic_oracle(self) -> None:
        response = {
            "choices": [{"finish_reason": "length", "message": {"content": "wrong"}}],
            "usage": {"prompt_tokens": 38, "completion_tokens": 400},
        }
        with self.assertRaisesRegex(RuntimeError, "semantic oracle"):
            log_support.validate_response(response, 400)

    def test_sources_obey_spdx_and_file_cap(self) -> None:
        modules = (
            capture,
            cleanup_support,
            log_support,
            model_support,
            process_support,
            support,
            trace_support,
        )
        tests = (
            Path(__file__),
            Path(__file__).with_name("test_nsys_b61_capture_hostile.py"),
            Path(__file__).with_name("test_nsys_b61_capture_final.py"),
        )
        for path in (*(Path(module.__file__) for module in modules), *tests):
            lines = path.read_text().splitlines()
            self.assertLessEqual(len(lines), 250)
            self.assertIn(
                "SPDX-License-Identifier: AGPL-3.0-only", "\n".join(lines[:2])
            )

    def test_unpinned_reference_rejects_before_model_access(self) -> None:
        with mock.patch.object(model_support.support, "require_regular") as regular:
            with self.assertRaisesRegex(RuntimeError, "not pinned"):
                model_support.attest_model()
        regular.assert_not_called()


class TraceContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def make_trace(
        self, *, omit: str | None = None, graph_column: bool = True
    ) -> tuple[Path, Path]:
        csv_path, sqlite_path = self.root / "trace.csv", self.root / "trace.sqlite"
        names = {
            "attention": ("paged_decode_attn_turbo4", 7),
            "gdn": ("gated_delta_rule_wy17", 7),
            "moe": ("moe_expert_gate_up_shared", 7),
            "ple": ("qwen4_ple_dequant_rows", 0),
            "terminal": ("argmax_bf16", 0),
        }
        with csv_path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["Duration (ns)", "Name"])
            writer.writeheader()
            for family, (name, _) in names.items():
                if family != omit:
                    writer.writerows(
                        {"Duration (ns)": "5", "Name": name} for _ in range(400)
                    )
        with sqlite3.connect(sqlite_path) as db:
            column = "graphNodeId INTEGER" if graph_column else "correlationId INTEGER"
            db.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
            db.execute(
                f"CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL "
                f"({column}, start INTEGER, end INTEGER, demangledName INTEGER)"
            )
            for index, (family, (name, graph)) in enumerate(names.items(), 1):
                if family == omit:
                    continue
                db.execute("INSERT INTO StringIds VALUES (?, ?)", (index, name))
                db.executemany(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?)",
                    [
                        (graph, offset * 10, offset * 10 + 5, index)
                        for offset in range(400)
                    ],
                )
        return csv_path, sqlite_path

    def test_requires_families_and_segmented_graph_nodes(self) -> None:
        csv_path, sqlite_path = self.make_trace()
        got = trace_support.validate_trace(csv_path, sqlite_path)
        self.assertEqual(got["location_rows"], {"graph": 1200, "eager": 800})
        self.assertEqual(got["cuda_rows"], {"csv": 2000, "sqlite_kernels": 2000})
        self.assertEqual(got["cuda_duration_ns"], {"csv": 10000, "sqlite": 10000})
        self.assertEqual(got["duration_unit"], "ns")
        self.assertEqual(set(got["family_rows"].values()), {400})
        self.assertEqual(got["family_by_location"]["moe"]["graph"]["duration_ns"], 2000)

    def test_missing_family_fails_closed(self) -> None:
        csv_path, sqlite_path = self.make_trace(omit="moe")
        with self.assertRaisesRegex(RuntimeError, "kernel-family"):
            trace_support.validate_trace(csv_path, sqlite_path)

    def test_missing_graph_column_fails_closed(self) -> None:
        csv_path, sqlite_path = self.make_trace(graph_column=False)
        with self.assertRaisesRegex(RuntimeError, "graphNodeId"):
            trace_support.validate_trace(csv_path, sqlite_path)


class CapabilityTests(unittest.TestCase):
    def test_missing_node_capability_fails_closed(self) -> None:
        with tempfile.NamedTemporaryFile() as binary:
            outputs = [
                "2025.3",
                "--session-new= --cuda-graph-trace= --inherit-environment= --trace= "
                "--sample= --cpuctxsw=",
                "--session= --export= --force-overwrite= --output=",
                "--session=",
                "--session=",
                "--session= --kill=",
                "cuda_gpu_trace",
                "cuda_gpu_trace cuda_gpu_kern_gb_sum",
            ]
            completed = [
                subprocess.CompletedProcess([], int(index == 7), stdout)
                for index, stdout in enumerate(outputs)
            ]
            with mock.patch.object(support.subprocess, "run", side_effect=completed):
                with self.assertRaisesRegex(RuntimeError, "launch"):
                    support.nsys_capabilities(Path(binary.name))


if __name__ == "__main__":
    unittest.main()
