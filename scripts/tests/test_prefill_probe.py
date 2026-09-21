# SPDX-License-Identifier: AGPL-3.0-only

"""Offline contracts for the uncached prefill throughput probe."""

import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / "prefill_probe.py"


def load_probe():
    spec = importlib.util.spec_from_file_location("atlas_prefill_probe", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.object(sys, "argv", [str(SCRIPT)]):
        spec.loader.exec_module(module)
    return module


def event(payload):
    return f"data: {json.dumps(payload)}\n".encode()


class SseResponse:
    def __init__(self, lines):
        self.lines = lines

    def __iter__(self):
        return iter(self.lines)


class PrefillProbeTests(unittest.TestCase):
    def setUp(self):
        self.probe = load_probe()

    def run_stream(self, lines):
        response = SseResponse(lines + [b"data: [DONE]\n"])
        with mock.patch.object(
            self.probe.urllib.request, "urlopen", return_value=response
        ), mock.patch.object(
            self.probe.time, "perf_counter", side_effect=[100.0, 101.25, 104.0]
        ):
            return self.probe.run("model", "unique prompt")

    @staticmethod
    def content_event():
        return event({"choices": [{"delta": {"content": "OK"}}]})

    @staticmethod
    def usage_event(
        prompt_tokens, details_marker=True, cached_marker=True, cached_tokens=0
    ):
        usage = {"prompt_tokens": prompt_tokens}
        if details_marker:
            details = {}
            if cached_marker:
                details["cached_tokens"] = cached_tokens
            usage["prompt_tokens_details"] = details
        return event({"choices": [], "usage": usage})

    def test_run_uses_final_verified_usage_and_returns_total_and_fresh_counts(self):
        intermediate = event(
            {
                "choices": [{"delta": {}}],
                "usage": {
                    "prompt_tokens": 7,
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
            }
        )
        total, fresh, ttft, wall = self.run_stream(
            [self.content_event(), intermediate, self.usage_event(123)]
        )
        self.assertEqual((total, fresh), (123, 123))
        self.assertEqual(ttft, 1.25)
        self.assertEqual(wall, 4.0)

    def test_cached_prompt_is_rejected_before_any_rate_can_be_reported(self):
        with self.assertRaisesRegex(ValueError, "cached"):
            self.run_stream(
                [self.content_event(), self.usage_event(123, cached_tokens=1)]
            )

    def test_missing_final_usage_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "usage"):
            self.run_stream([self.content_event()])

    def test_nonfinal_usage_does_not_satisfy_the_contract(self):
        nonfinal = event(
            {
                "choices": [{"delta": {"content": "still streaming"}}],
                "usage": {
                    "prompt_tokens": 123,
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
            }
        )
        with self.assertRaisesRegex(ValueError, "final.*usage|usage.*final"):
            self.run_stream([self.content_event(), nonfinal])

    def test_missing_prompt_token_details_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "prompt_tokens_details"):
            self.run_stream(
                [self.content_event(), self.usage_event(123, details_marker=False)]
            )

    def test_missing_cached_token_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "cached_tokens"):
            self.run_stream(
                [self.content_event(), self.usage_event(123, cached_marker=False)]
            )

    def test_missing_first_output_timestamp_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "first streamed output"):
            self.run_stream([self.usage_event(123)])

    def test_reasoning_delta_starts_ttft_clock(self):
        reasoning = event({"choices": [{"delta": {"reasoning_content": "thinking"}}]})
        total, fresh, ttft, wall = self.run_stream([reasoning, self.usage_event(123)])
        self.assertEqual((total, fresh), (123, 123))
        self.assertEqual(ttft, 1.25)
        self.assertEqual(wall, 4.0)

    def test_nonce_changes_the_prefix_without_changing_requested_body(self):
        first = self.probe.make_prompt(72, nonce="probe-a")
        second = self.probe.make_prompt(72, nonce="probe-b")
        self.assertNotEqual(
            first[: first.index("Fact 0:")], second[: second.index("Fact 0:")]
        )
        self.assertEqual(
            first[first.index("Fact 0:") :], second[second.index("Fact 0:") :]
        )
        self.assertIn("Fact 3:", first)

    def test_main_reports_five_run_median_range_and_raw_json(self):
        # Presentation remains auditable even if a future caller supplies a
        # partially cached measurement: never label total tokens as fresh work.
        per_length = [
            (50, 50, 1.0, 1.1),  # warmup
            (120, 80, 2.0, 2.1),
            (120, 80, 1.6, 1.7),
            (120, 80, 4.0 / 3.0, 1.5),
            (120, 80, 8.0 / 7.0, 1.3),
            (120, 80, 1.0, 1.2),
        ]
        results = per_length * 3
        stdout = io.StringIO()
        with mock.patch.object(
            self.probe, "model_id", return_value="model"
        ), mock.patch.object(
            self.probe, "make_prompt", return_value="prompt"
        ), mock.patch.object(
            self.probe, "run", side_effect=results
        ) as run_mock, mock.patch.dict(
            self.probe.os.environ, {}, clear=True
        ), mock.patch(
            "sys.stdout", stdout
        ):
            self.probe.main()
        output = stdout.getvalue()
        self.assertEqual(run_mock.call_count, 18)
        self.assertIn("samples=5", output)
        self.assertIn("median=   60.0 tok/s", output)
        self.assertIn("range=40.0..80.0", output)
        raw_line = next(
            line for line in output.splitlines() if line.startswith("RESULT_JSON=")
        )
        report = json.loads(raw_line.removeprefix("RESULT_JSON="))
        self.assertEqual(report["schema"], "atlas-prefill-probe-v1")
        self.assertEqual(report["classification"], "UNGRADED")
        self.assertIsNone(report["benchmark_receipt_sha256"])
        self.assertEqual(report["model"], "model")
        self.assertEqual(report["measured_runs"], 5)
        self.assertEqual(len(report["lengths"]), 3)
        self.assertEqual(len(report["lengths"][0]["samples"]), 5)
        self.assertEqual(report["lengths"][0]["median_prefill_tok_s"], 60.0)
        self.assertEqual(report["lengths"][0]["min_prefill_tok_s"], 40.0)
        self.assertEqual(report["lengths"][0]["max_prefill_tok_s"], 80.0)
        self.assertTrue(
            all(
                sample["fresh_tokens"] == 80
                for sample in report["lengths"][0]["samples"]
            )
        )

    def test_receipt_binding_is_verified_and_mutation_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text(
                json.dumps({"model_type": "deepseek_v4"}), encoding="utf-8"
            )
            (model / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"x": "model.safetensors"}}), encoding="utf-8"
            )
            (model / "model.safetensors").write_bytes(b"weights")
            (model / "tokenizer.json").write_text("{}\n", encoding="utf-8")
            binary = root / "spark"
            binary.write_bytes(b"deepseek_v4")
            binary.chmod(0o755)
            envelope = self.probe.benchmark_receipt.build_receipt(
                repo=Path(__file__).resolve().parents[2],
                binary=binary,
                model=model,
                drafter=None,
                argv=[str(binary), "serve", str(model)],
                environment={},
                required_binary_strings=["deepseek_v4"],
            )
            path = Path(temp) / "receipt.json"
            path.write_text(json.dumps(envelope), encoding="utf-8")
            binding = self.probe.load_receipt_binding(str(path))
            self.assertEqual(binding["classification"], "PLANNED")
            self.assertEqual(
                binding["benchmark_receipt_sha256"], envelope["manifest_sha256"]
            )
            envelope["manifest"]["argv"].append("--changed")
            path.write_text(json.dumps(envelope), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "digest"):
                self.probe.load_receipt_binding(str(path))

    def test_synthetic_receipt_cannot_claim_planned_provenance(self):
        manifest = {"schema": "atlas-benchmark-receipt-v1", "receipt_state": "PLANNED"}
        envelope = {
            "manifest": manifest,
            "manifest_sha256": self.probe.benchmark_receipt.hashlib.sha256(
                self.probe.benchmark_receipt.canonical_json(manifest)
            ).hexdigest(),
        }
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "receipt.json"
            path.write_text(json.dumps(envelope), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                self.probe.load_receipt_binding(str(path))

    def test_main_revalidates_active_receipt_after_measurement(self):
        binding = {
            "classification": "ACTIVE_VERIFIED",
            "benchmark_receipt_sha256": "a" * 64,
            "gpu_identity": {"uuid": "GPU-unit"},
            "gpu_activation_state": {"pstate": "P0"},
        }
        with mock.patch.object(
            self.probe, "model_id", return_value="model"
        ), mock.patch.object(
            self.probe, "make_prompt", return_value="prompt"
        ), mock.patch.object(
            self.probe, "run", return_value=(80, 80, 1.0, 1.1)
        ), mock.patch.object(
            self.probe,
            "load_receipt_binding",
            side_effect=[binding, ValueError("runtime drift")],
        ) as load_mock, mock.patch.dict(
            self.probe.os.environ, {"ATLAS_BENCH_RECEIPT": "/tmp/active.json"}, clear=True
        ), mock.patch(
            "sys.stdout", io.StringIO()
        ):
            with self.assertRaisesRegex(ValueError, "runtime drift"):
                self.probe.main()
        self.assertEqual(load_mock.call_count, 2)


if __name__ == "__main__":
    unittest.main()
