# SPDX-License-Identifier: AGPL-3.0-only

"""Offline contracts for the decode A/B measurement probe."""

import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "decode_ab_probe.py"


def load_probe():
    spec = importlib.util.spec_from_file_location("atlas_decode_ab_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.object(sys, "argv", [str(SCRIPT)]):
        spec.loader.exec_module(module)
    return module


class DecodeProbeTests(unittest.TestCase):
    def setUp(self):
        self.probe = load_probe()

    def response(
        self,
        *,
        finish_reason="stop",
        model="model",
        reasoning="",
        include_tps=True,
        include_engine=True,
    ):
        payload = {
            "model": model,
            "choices": [
                {
                    "finish_reason": finish_reason,
                    "message": {
                        "content": "full deterministic completion",
                        "reasoning_content": reasoning,
                    },
                }
            ],
            "usage": {
                "prompt_tokens": 17,
                "completion_tokens": 23,
                "total_tokens": 40,
                "response_token/s": 46.5,
            },
        }
        if not include_tps:
            payload["usage"].pop("response_token/s")
        if include_engine:
            payload["usage"]["atlas_engine"] = {
                "classification": "SERIAL",
                "speculative_steps": 0,
                "serial_tokens": 23,
                "low_gear_steps": 0,
            }
        return io.BytesIO(json.dumps(payload).encode())

    def test_run_one_records_full_identity_usage_finish_and_hash(self):
        with mock.patch.object(
            self.probe.urllib.request, "urlopen", return_value=self.response()
        ), mock.patch.object(self.probe.time, "perf_counter", side_effect=[10.0, 11.0]):
            result = self.probe.run_one("model", "prompt")
        self.assertEqual(result["response_model"], "model")
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(result["prompt_tokens"], 17)
        self.assertEqual(result["completion_tokens"], 23)
        self.assertEqual(result["total_tokens"], 40)
        self.assertEqual(result["tok_s"], 46.5)
        self.assertEqual(result["metric_source"], "usage.response_token/s")
        self.assertEqual(result["engine"]["classification"], "SERIAL")
        self.assertEqual(len(result["sha256"]), 64)

    def test_run_one_rejects_unknown_finish_reason_and_model_mismatch(self):
        with mock.patch.object(
            self.probe.urllib.request,
            "urlopen",
            return_value=self.response(finish_reason="tool_calls"),
        ), mock.patch.object(self.probe.time, "perf_counter", side_effect=[10.0, 11.0]):
            with self.assertRaisesRegex(ValueError, "finish_reason"):
                self.probe.run_one("model", "prompt")
        with mock.patch.object(
            self.probe.urllib.request,
            "urlopen",
            return_value=self.response(model="wrong"),
        ), mock.patch.object(self.probe.time, "perf_counter", side_effect=[10.0, 11.0]):
            with self.assertRaisesRegex(ValueError, "model"):
                self.probe.run_one("model", "prompt")

    def test_run_one_requires_server_metric_and_hashes_reasoning(self):
        with mock.patch.object(
            self.probe.urllib.request,
            "urlopen",
            return_value=self.response(include_tps=False),
        ), mock.patch.object(self.probe.time, "perf_counter", side_effect=[10.0, 11.0]):
            with self.assertRaisesRegex(ValueError, "server-side"):
                self.probe.run_one("model", "prompt")
        with mock.patch.object(
            self.probe.urllib.request,
            "urlopen",
            side_effect=[
                self.response(reasoning="first"),
                self.response(reasoning="second"),
            ],
        ), mock.patch.object(
            self.probe.time, "perf_counter", side_effect=[10.0, 11.0, 20.0, 21.0]
        ):
            first = self.probe.run_one("model", "prompt")
            second = self.probe.run_one("model", "prompt")
        self.assertNotEqual(first["sha256"], second["sha256"])

    def test_engine_usage_fails_closed_on_missing_or_inconsistent_evidence(self):
        missing = self.probe.parse_engine_usage({})
        self.assertEqual(missing["classification"], "UNVERIFIED")
        inconsistent = {
            "atlas_engine": {
                "classification": "SPECULATIVE",
                "speculative_steps": 0,
                "serial_tokens": 5,
                "low_gear_steps": 0,
            }
        }
        with self.assertRaisesRegex(ValueError, "disagrees"):
            self.probe.parse_engine_usage(inconsistent)

    def test_main_excludes_warmup_and_reports_five_run_medians(self):
        def record(rate):
            return {
                "tok_s": rate,
                "metric_source": "usage.response_token/s",
                "prompt_tokens": 17,
                "completion_tokens": 23,
                "total_tokens": 40,
                "wall_s": 1.0,
                "response_model": "model",
                "finish_reason": "stop",
                "sha256": "a" * 64,
                "engine": {
                    "classification": "SERIAL",
                    "speculative_steps": 0,
                    "serial_tokens": 23,
                    "low_gear_steps": 0,
                },
                "prefix": "ok",
            }

        side_effect = []
        for run in range(6):
            for workload in range(4):
                side_effect.append(record(10.0 + run * 10.0 + workload))
        stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(
            self.probe, "model_id", return_value="model"
        ), mock.patch.object(
            self.probe, "run_one", side_effect=side_effect
        ) as run_mock, mock.patch.dict(
            self.probe.os.environ, {}, clear=True
        ), mock.patch(
            "sys.stdout", stdout
        ):
            old_cwd = os.getcwd()
            os.chdir(temp)
            try:
                self.probe.main()
            finally:
                os.chdir(old_cwd)
            report = json.loads(
                (Path(temp) / "probe-probe.json").read_text(encoding="utf-8")
            )
        self.assertEqual(run_mock.call_count, 24)
        self.assertEqual(report["warmup_runs"], 1)
        self.assertEqual(report["measured_runs"], 5)
        self.assertEqual(len(report["warmups"]), 1)
        self.assertEqual(len(report["runs"]), 5)
        self.assertEqual(report["median_tok_s"]["code"], 40.0)
        self.assertEqual(report["median_tok_s"]["prose"], 43.0)
        self.assertEqual(report["classification"], "UNGRADED")
        self.assertEqual(report["engine_classification"], "SERIAL")
        self.assertEqual(report["median_tok_s_by_engine"]["code"], {"SERIAL": 40.0})
        self.assertIsNone(report["benchmark_receipt_sha256"])
        self.assertIn("5-run median", stdout.getvalue())

    def test_main_revalidates_active_receipt_before_writing_report(self):
        record = {
            "tok_s": 50.0,
            "metric_source": "usage.response_token/s",
            "prompt_tokens": 17,
            "completion_tokens": 23,
            "total_tokens": 40,
            "wall_s": 1.0,
            "response_model": "model",
            "finish_reason": "stop",
            "truncated": False,
            "sha256": "a" * 64,
            "engine": {
                "classification": "SERIAL",
                "speculative_steps": 0,
                "serial_tokens": 23,
                "low_gear_steps": 0,
            },
            "prefix": "ok",
        }
        binding = {
            "classification": "ACTIVE_VERIFIED",
            "benchmark_receipt_sha256": "a" * 64,
            "gpu_identity": {"uuid": "GPU-unit"},
            "gpu_activation_state": {"pstate": "P0"},
        }
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(
            self.probe, "model_id", return_value="model"
        ), mock.patch.object(
            self.probe, "run_one", return_value=record
        ), mock.patch.object(
            self.probe,
            "load_receipt_binding",
            side_effect=[binding, ValueError("runtime drift")],
            create=True,
        ) as load_mock, mock.patch.dict(
            self.probe.os.environ, {"ATLAS_BENCH_RECEIPT": "/tmp/active.json"}, clear=True
        ), mock.patch(
            "sys.stdout", io.StringIO()
        ):
            old_cwd = os.getcwd()
            os.chdir(temp)
            try:
                with self.assertRaisesRegex(ValueError, "runtime drift"):
                    self.probe.main()
            finally:
                os.chdir(old_cwd)
            self.assertFalse((Path(temp) / f"probe-{self.probe.TAG}.json").exists())
        self.assertEqual(load_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
