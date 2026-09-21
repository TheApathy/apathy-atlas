# SPDX-License-Identifier: AGPL-3.0-only

"""Offline contracts for the exact DeepSeek V4 max-prefill qualifier."""

import hashlib
import importlib.util
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "qualify-exl3-prefill-max.py"


def load_qualifier():
    spec = importlib.util.spec_from_file_location("atlas_max_prefill_qualifier", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeAtlasHandler(BaseHTTPRequestHandler):
    prompts = []
    event_mode = "valid"

    def log_message(self, _format, *_args):
        pass

    def _read_json(self):
        length = int(self.headers["Content-Length"])
        return json.loads(self.rfile.read(length))

    def do_POST(self):
        payload = self._read_json()
        if self.path == "/tokenize":
            prompt = payload["prompt"]
            # The test oracle only needs stable, nonempty token IDs. Production
            # takes these IDs from the live Atlas tokenizer.
            tokens = [1000 + byte for byte in prompt.encode("utf-8")]
            body = json.dumps({"tokens": tokens, "count": len(tokens)}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/v1/completions":
            prompt = payload["prompt"]
            self.__class__.prompts.append(prompt)
            usage = {
                "choices": [],
                "usage": {
                    "prompt_tokens": len(prompt),
                    "completion_tokens": 4,
                    "prompt_tokens_details": {"cached_tokens": 0},
                },
            }
            chunks = [
                {"choices": [{"text": "ok"}]},
                {"choices": [{"text": "", "finish_reason": "length"}]},
                usage,
            ]
            if self.__class__.event_mode == "content_after_terminal":
                chunks.insert(2, {"choices": [{"text": "late"}]})
            elif self.__class__.event_mode == "duplicate_usage":
                chunks.append(usage)
            body = (
                b"".join(f"data: {json.dumps(chunk)}\n\n".encode() for chunk in chunks)
                + b"data: [DONE]\n\n"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_error(404)


class FakeAtlasServer:
    def __enter__(self):
        FakeAtlasHandler.prompts = []
        FakeAtlasHandler.event_mode = "valid"
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeAtlasHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.base_url = f"http://{host}:{port}"
        return self

    def __exit__(self, *_args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


class MaxPrefillQualifierTests(unittest.TestCase):
    def setUp(self):
        self.qualifier = load_qualifier()

    def test_live_tokenizer_builds_unique_exact_token_vectors(self):
        with FakeAtlasServer() as server:
            first = self.qualifier.build_exact_prompt(server.base_url, "a" * 64, 2410)
            second = self.qualifier.build_exact_prompt(server.base_url, "b" * 64, 2410)
        self.assertEqual(len(first), 2410)
        self.assertEqual(len(second), 2410)
        self.assertNotEqual(first, second)

    def test_receipt_verifier_uses_selected_port_and_restores_generic_probe(self):
        original = self.qualifier.prefill_probe.BASE
        with mock.patch.object(
            self.qualifier.prefill_probe,
            "load_receipt_binding",
            side_effect=lambda _path: {"base": self.qualifier.prefill_probe.BASE},
        ):
            binding = self.qualifier.load_receipt_binding(
                Path("active.json"), "http://127.0.0.1:9123"
            )
        self.assertEqual(binding["base"], "http://127.0.0.1:9123")
        self.assertEqual(self.qualifier.prefill_probe.BASE, original)

    def test_fake_server_measurement_uses_exact_uncached_prompts(self):
        with FakeAtlasServer() as server:
            report = self.qualifier.measure_exact_prefill(
                server.base_url, "deepseek", measured_runs=2
            )
        self.assertEqual(len(FakeAtlasHandler.prompts), 3)
        self.assertTrue(all(len(prompt) == 2410 for prompt in FakeAtlasHandler.prompts))
        self.assertEqual(len({tuple(prompt) for prompt in FakeAtlasHandler.prompts}), 3)
        self.assertEqual(report["warmup"]["total_tokens"], 2410)
        self.assertEqual(
            report["warmup"]["output_sha256"], hashlib.sha256(b"ok").hexdigest()
        )
        self.assertEqual(report["warmup"]["finish_reason"], "length")
        self.assertEqual(report["warmup"]["completion_tokens"], 4)
        self.assertEqual(len(report["samples"]), 2)
        self.assertTrue(
            all(sample["fresh_tokens"] == 2410 for sample in report["samples"])
        )
        encoded = json.dumps(report)
        self.assertNotIn('"prompt":', encoded)
        self.assertNotIn("nonce=", encoded)

    def test_stream_protocol_rejects_content_after_terminal_and_repeated_usage(self):
        for mode, pattern in (
            ("content_after_terminal", "content after terminal"),
            ("duplicate_usage", "repeated usage|data after"),
        ):
            with self.subTest(mode=mode), FakeAtlasServer() as server:
                FakeAtlasHandler.event_mode = mode
                prompt = self.qualifier.build_exact_prompt(
                    server.base_url, "a" * 64, 2410
                )
                with self.assertRaisesRegex(ValueError, pattern):
                    self.qualifier.run_exact_completion(
                        server.base_url, "deepseek", prompt
                    )

    def test_cached_or_wrong_shape_usage_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "exactly 2410"):
            self.qualifier.validate_usage(
                {"prompt_tokens": 2409, "prompt_tokens_details": {"cached_tokens": 0}}
            )
        with self.assertRaisesRegex(ValueError, "zero cached"):
            self.qualifier.validate_usage(
                {"prompt_tokens": 2410, "prompt_tokens_details": {"cached_tokens": 1}}
            )
        with self.assertRaisesRegex(ValueError, "finish_reason"):
            self.qualifier.validate_terminal_usage({"completion_tokens": 4}, "other")
        with self.assertRaisesRegex(ValueError, "four"):
            self.qualifier.validate_terminal_usage({"completion_tokens": 3}, "length")

    def test_profile_is_derived_from_config_and_process_must_match_exactly(self):
        required = self.qualifier.REQUIRED_PROFILE_VALUES
        output = (
            "env  : "
            + " ".join(f"{key}={value}" for key, value in sorted(required.items()))
            + "\n"
        )
        expected = self.qualifier.parse_profile_config(output)
        self.assertEqual(expected, required)

        for key, value in sorted(required.items()):
            assignments = [
                f"{current}={current_value}"
                for current, current_value in sorted(required.items())
                if current != key
            ]
            with (
                self.subTest(key=key, failure="omitted"),
                self.assertRaisesRegex(ValueError, f"omits required value: {key}"),
            ):
                self.qualifier.parse_profile_config(
                    "env  : " + " ".join(assignments) + "\n"
                )

            wrong_value = "0" if value != "0" else "1"
            assignments = [
                f"{current}={wrong_value if current == key else current_value}"
                for current, current_value in sorted(required.items())
            ]
            with (
                self.subTest(key=key, failure="wrong"),
                self.assertRaisesRegex(ValueError, f"wrong value for {key}"),
            ):
                self.qualifier.parse_profile_config(
                    "env  : " + " ".join(assignments) + "\n"
                )

        with self.assertRaisesRegex(ValueError, "unexpected value"):
            self.qualifier.parse_profile_config(
                output.rstrip() + " ATLAS_UNREVIEWED_PROFILE_VALUE=1\n"
            )
        with self.assertRaisesRegex(ValueError, "repeats"):
            self.qualifier.parse_profile_config(
                output.rstrip() + " ATLAS_KV_OVERCOMMIT=0\n"
            )
        actual = {**expected, "PATH": "/usr/bin"}
        self.qualifier.validate_process_profile(expected, actual)
        with self.assertRaisesRegex(ValueError, "environment mismatch"):
            self.qualifier.validate_process_profile(
                expected, {**actual, "ATLAS_UNRECEIPTED_EXPERIMENT": "1"}
            )
        broken = dict(actual)
        broken.pop(next(iter(required)))
        with self.assertRaisesRegex(ValueError, "environment mismatch"):
            self.qualifier.validate_process_profile(expected, broken)

    def test_real_launcher_exports_exact_required_profile(self):
        with tempfile.TemporaryDirectory() as temp:
            profile = self.qualifier.run_config_preflight(Path(temp), 9123)
        self.assertEqual(profile, self.qualifier.REQUIRED_PROFILE_VALUES)

    def test_fake_process_environment_is_read_from_proc_and_checked(self):
        expected = dict(self.qualifier.REQUIRED_PROFILE_VALUES)
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("ATLAS_")
            and key not in self.qualifier.FORBIDDEN_PROCESS_ENV
        }
        environment.update(expected)
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"], env=environment
        )
        try:
            deadline = time.monotonic() + 2
            while True:
                actual = self.qualifier.benchmark_receipt.read_process_environment(
                    process.pid
                )
                if all(actual.get(key) == value for key, value in expected.items()):
                    break
                if time.monotonic() >= deadline:
                    self.fail("fake process did not publish its exec environment")
                time.sleep(0.01)
            self.qualifier.validate_process_profile(expected, actual)
        finally:
            process.terminate()
            process.wait(timeout=5)

    def test_all_strict_engagement_markers_are_required_at_exact_shape(self):
        good = "\n".join(
            [
                "ATLAS_PREFILL_MAX_ARMS_RECEIPT core=w2a8 fused_gu=n128 down=n256 n_tokens=2410 total_expanded=14460 experts=256 top_k=6",
                "ATLAS_PREFILL_MAX_ARMS_RECEIPT tail=fused_unpermute n_tokens=2410 hidden=4096 top_k=6",
                "V4_PREFILL_MAX_ARM_ENGAGED arm=hc_pre_finish_rms_fused site=attention layer=0 n=2410",
                "V4_PREFILL_MAX_ARM_ENGAGED arm=hc_pre_finish_rms_fused site=ffn layer=0 n=2410",
                "V4_PREFILL_MAX_ARM_ENGAGED arm=kv_alias layer=0 n=2410 nq=64 nkv=1 hd_mla=512",
                "V4_PREFILL_MAX_ARM_ENGAGED arm=inverse_rope layer=0 n=2410 nq=64 nkv=1 hd_mla=512",
            ]
        )
        counts = self.qualifier.validate_engagement_text(good)
        self.assertEqual(set(counts), set(self.qualifier.ENGAGEMENT_PATTERNS))
        with self.assertRaisesRegex(ValueError, "inverse_rope"):
            self.qualifier.validate_engagement_text(
                good.replace("arm=inverse_rope", "arm=wrong")
            )
        with self.assertRaisesRegex(ValueError, "w2a8_core"):
            self.qualifier.validate_engagement_text(
                good.replace("n_tokens=2410", "n_tokens=2409")
            )

    def test_release_report_requires_twenty_exact_samples_and_active_receipt(self):
        sample = {
            "total_tokens": 2410,
            "fresh_tokens": 2410,
            "ttft_seconds": 2.0,
            "wall_seconds": 2.1,
            "prefill_tok_s": 1205.0,
            "prompt_sha256": hashlib.sha256(b"unique").hexdigest(),
            "output_sha256": hashlib.sha256(b"output").hexdigest(),
            "finish_reason": "length",
            "completion_tokens": 4,
        }
        measurement = {"warmup": sample, "samples": [dict(sample) for _ in range(20)]}
        for index, current in enumerate(measurement["samples"]):
            current["prompt_sha256"] = hashlib.sha256(str(index).encode()).hexdigest()
        binding = {
            "classification": "ACTIVE_VERIFIED",
            "benchmark_receipt_sha256": "a" * 64,
            "gpu_identity": {"uuid": "GPU-unit"},
            "gpu_activation_state": {"pstate": "P0"},
        }
        report = self.qualifier.build_report(
            measurement=measurement,
            receipt_binding=binding,
            engagement_counts={key: 1 for key in self.qualifier.ENGAGEMENT_PATTERNS},
            build_receipt_sha256="b" * 64,
            model_preflight_sha256="c" * 64,
        )
        self.assertEqual(report["schema"], "atlas-exl3-prefill-max-qualification-v1")
        self.assertEqual(report["classification"], "ACTIVE_VERIFIED")
        self.assertEqual(report["measured_runs"], 20)
        self.assertFalse(report["target_2000_tok_s_met"])
        with self.assertRaisesRegex(ValueError, "20"):
            self.qualifier.build_report(
                measurement={**measurement, "samples": measurement["samples"][:-1]},
                receipt_binding=binding,
                engagement_counts={
                    key: 1 for key in self.qualifier.ENGAGEMENT_PATTERNS
                },
                build_receipt_sha256="b" * 64,
                model_preflight_sha256="c" * 64,
            )
        with self.assertRaisesRegex(ValueError, "20"):
            self.qualifier.build_report(
                measurement={
                    **measurement,
                    "samples": [*measurement["samples"], dict(sample)],
                },
                receipt_binding=binding,
                engagement_counts={
                    key: 1 for key in self.qualifier.ENGAGEMENT_PATTERNS
                },
                build_receipt_sha256="b" * 64,
                model_preflight_sha256="c" * 64,
            )
        with self.assertRaisesRegex(ValueError, "ACTIVE_VERIFIED"):
            self.qualifier.build_report(
                measurement=measurement,
                receipt_binding={**binding, "classification": "PLANNED"},
                engagement_counts={
                    key: 1 for key in self.qualifier.ENGAGEMENT_PATTERNS
                },
                build_receipt_sha256="b" * 64,
                model_preflight_sha256="c" * 64,
            )

    def test_output_artifacts_cannot_self_contaminate_repo_identity(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            self.qualifier.create_output_dir(
                self.qualifier.REPO / "unit-qualification-output"
            )

    def test_build_receipt_is_regular_bounded_and_unchanged(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "build-receipt"
            path.write_text("schema atlas-prefill-max-build-v1\n", encoding="utf-8")
            digest = self.qualifier.build_receipt_digest(path)
            self.qualifier.require_unchanged_build_receipt(path, digest)
            path.write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "changed"):
                self.qualifier.require_unchanged_build_receipt(path, digest)

    def test_stop_server_returns_only_after_original_process_is_gone(self):
        with (
            mock.patch.object(
                self.qualifier,
                "original_process_running",
                side_effect=[True, False],
            ),
            mock.patch.object(self.qualifier.os, "kill") as kill,
        ):
            self.qualifier.stop_server(123, 456, graceful_timeout=1.0)
        kill.assert_called_once_with(123, self.qualifier.signal.SIGTERM)

    def test_stop_server_fails_if_process_survives_sigkill(self):
        with (
            mock.patch.object(
                self.qualifier, "original_process_running", return_value=True
            ),
            mock.patch.object(self.qualifier.os, "kill") as kill,
        ):
            with self.assertRaisesRegex(ValueError, "remained live after SIGKILL"):
                self.qualifier.stop_server(
                    123,
                    456,
                    graceful_timeout=0.0,
                    kill_timeout=0.0,
                )
        self.assertEqual(
            kill.call_args_list,
            [
                mock.call(123, self.qualifier.signal.SIGTERM),
                mock.call(123, self.qualifier.signal.SIGKILL),
            ],
        )

    def test_live_pid_with_unavailable_identity_is_not_reported_stopped(self):
        with (
            mock.patch.object(
                self.qualifier.benchmark_receipt,
                "process_start_ticks",
                side_effect=ValueError("process start time unavailable"),
            ),
            mock.patch.object(self.qualifier.os, "kill") as kill,
        ):
            with self.assertRaisesRegex(ValueError, "PID remains live"):
                self.qualifier.original_process_running(123, 456)
        kill.assert_called_once_with(123, 0)


if __name__ == "__main__":
    unittest.main()
