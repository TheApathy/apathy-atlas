import importlib.util
import json
import math
import pathlib
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

MODULE_PATH = pathlib.Path(__file__).with_name("dflash_release_gate.py")
SPEC = importlib.util.spec_from_file_location("dflash_release_gate", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
BOOT_ID_A = "abcdef01-2222-4333-8444-555555555555"
BOOT_ID_B = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
GPU_ID = {
    "uuid": "GPU-unit",
    "name": "NVIDIA GB10",
    "driver_version": "580.126.09",
}


class StreamResponse:
    status = 200

    def __init__(self, events):
        self.events = events

    def __iter__(self):
        return iter(self.events)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def result_cases(*, finish_reason="stop", completion_tokens=2):
    return {
        name: {
            "prompt_sha256": MODULE.hashlib.sha256(name.encode()).hexdigest(),
            "runs": [
                {
                    "output_sha256": MODULE.hashlib.sha256(
                        f"{name}-output".encode()
                    ).hexdigest(),
                    "finish_reason": finish_reason,
                    "truncated": finish_reason == "length",
                    "completion_tokens": completion_tokens,
                    "decode_tok_s": 1.0,
                }
            ],
        }
        for name in MODULE.PROMPTS
    }


def provenance(
    *,
    model_identity="a" * 64,
    implementation_identity="b" * 64,
    receipt_digest="c" * 64,
    boot_id=BOOT_ID_A,
    gpu_identity=GPU_ID,
):
    envelope = active_envelope(
        implementation_marker=implementation_identity,
        model_marker=model_identity,
        boot_id=boot_id,
        gpu_identity=gpu_identity,
    )
    return {
        "model_identity": MODULE.hashlib.sha256(
            MODULE.benchmark_receipt.canonical_json(envelope["manifest"]["model"])
        ).hexdigest(),
        "implementation_identity": envelope["manifest_sha256"],
        "benchmark_receipt_sha256": envelope["activation_sha256"],
        "host_boot_id": boot_id,
        "gpu_identity": gpu_identity,
        "benchmark_receipt": envelope,
    }


def active_envelope(
    *,
    implementation_marker="base",
    model_marker="model",
    boot_id=BOOT_ID_A,
    gpu_identity=GPU_ID,
):
    manifest = {
        "schema": "atlas-benchmark-receipt-v2",
        "receipt_state": "PLANNED",
        "git": {"marker": implementation_marker},
        "binary": {"marker": implementation_marker},
        "model": {"path": "/model", "files": {"config.json": model_marker}},
        "drafter": None,
        "argv": ["spark", "serve"],
        "environment": {},
        "full_environment_sha256": "0" * 64,
    }
    manifest_sha256 = MODULE.hashlib.sha256(
        MODULE.benchmark_receipt.canonical_json(manifest)
    ).hexdigest()
    activation = {
        "state": "ACTIVE_VERIFIED",
        "manifest_sha256": manifest_sha256,
        "pid": 1,
        "process_start_ticks": 1,
        "process_exe_path": "/spark",
        "process_exe_sha256": "1" * 64,
        "process_argv": ["spark", "serve"],
        "environment": {},
        "listen_port": 8977,
        "base_url": "http://127.0.0.1:8977",
        "model_id": "model",
        "host_boot_id": boot_id,
        "gpu_identity": gpu_identity,
        "gpu_state": {
            "pstate": "P0",
            "temperature.gpu": "1",
            "power.draw": "1",
            "clocks.sm": "1",
            "clocks.mem": "1",
            "memory.total": "1",
            "memory.used": "1",
            "memory.free": "1",
        },
    }
    return {
        "manifest": manifest,
        "manifest_sha256": manifest_sha256,
        "activation": activation,
        "activation_sha256": MODULE.hashlib.sha256(
            MODULE.benchmark_receipt.canonical_json(activation)
        ).hexdigest(),
    }


def measured_run():
    return {
        "completion_tokens": 2,
        "finish_reason": "stop",
        "truncated": False,
        "ttft_seconds": 0.1,
        "decode_seconds": 1.0,
        "decode_tok_s": 1.0,
        "reasoning_sha256": "d" * 64,
        "content_sha256": "e" * 64,
        "output_sha256": "f" * 64,
    }


class DflashReleaseGateTests(unittest.TestCase):
    def test_compare_rejects_nonfinite_metrics_and_malformed_records(self):
        baseline = {
            **provenance(),
            "contract": {},
            "median_decode_tok_s": 22.0,
            "cases": result_cases(),
        }
        candidate = {
            **baseline,
            **provenance(implementation_identity="d" * 64, receipt_digest="e" * 64),
            "median_decode_tok_s": math.nan,
            "acceptance": {"committed_tokens_per_step": math.nan},
        }
        report = MODULE.compare_results(baseline, candidate, 85.0, 3.0)
        self.assertEqual(report["status"], "fail")
        self.assertTrue(any("finite" in failure for failure in report["failures"]))
        malformed = MODULE.compare_results({}, [], 85.0, 3.0)
        self.assertEqual(malformed["status"], "fail")
        with self.assertRaisesRegex(RuntimeError, "finite"):
            MODULE.compare(SimpleNamespace(min_tok_s=math.nan, min_tok_step=3.0))

    def test_atomic_writer_never_replaces_without_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = pathlib.Path(tmp) / "result.json"
            output.write_text("original\n")
            with self.assertRaisesRegex(RuntimeError, "exists"):
                MODULE.write_json_atomic(output, {"replacement": True})
            self.assertEqual(output.read_text(), "original\n")
            MODULE.write_json_atomic(output, {"replacement": True}, overwrite=True)
            self.assertEqual(json.loads(output.read_text()), {"replacement": True})

    def test_active_receipt_derives_provenance_and_rejects_planned_or_wrong_model(self):
        envelope = active_envelope()
        with mock.patch.object(
            MODULE.benchmark_receipt, "read_envelope", return_value=envelope
        ), mock.patch.object(
            MODULE.benchmark_receipt,
            "receipt_state",
            return_value="ACTIVE_VERIFIED",
        ):
            binding = MODULE.load_receipt_binding(
                pathlib.Path("receipt.json"), "http://127.0.0.1:8977", "model"
            )
        expected_model = MODULE.hashlib.sha256(
            MODULE.benchmark_receipt.canonical_json(envelope["manifest"]["model"])
        ).hexdigest()
        self.assertEqual(binding["model_identity"], expected_model)
        self.assertEqual(
            binding["implementation_identity"], envelope["manifest_sha256"]
        )
        self.assertEqual(
            binding["benchmark_receipt_sha256"], envelope["activation_sha256"]
        )
        self.assertEqual(binding["host_boot_id"], BOOT_ID_A)
        self.assertEqual(binding["gpu_identity"], GPU_ID)

        with mock.patch.object(
            MODULE.benchmark_receipt, "read_envelope", return_value=envelope
        ), mock.patch.object(
            MODULE.benchmark_receipt, "receipt_state", return_value="PLANNED"
        ), self.assertRaisesRegex(
            RuntimeError, "ACTIVE_VERIFIED"
        ):
            MODULE.load_receipt_binding(
                pathlib.Path("receipt.json"), "http://127.0.0.1:8977", "model"
            )
        with mock.patch.object(
            MODULE.benchmark_receipt, "read_envelope", return_value=envelope
        ), mock.patch.object(
            MODULE.benchmark_receipt,
            "receipt_state",
            return_value="ACTIVE_VERIFIED",
        ), self.assertRaisesRegex(
            RuntimeError, "model"
        ):
            MODULE.load_receipt_binding(
                pathlib.Path("receipt.json"), "http://127.0.0.1:8977", "wrong"
            )

    def test_run_revalidates_same_active_receipt_before_publishing(self):
        binding = provenance()
        with tempfile.TemporaryDirectory() as tmp:
            output = pathlib.Path(tmp) / "result.json"
            args = SimpleNamespace(
                url="http://127.0.0.1:8977",
                model="model",
                label="candidate",
                receipt=pathlib.Path("receipt.json"),
                max_tokens=2,
                reps=1,
                server_log=None,
                output=output,
                overwrite=False,
            )
            with mock.patch.object(
                MODULE, "load_receipt_binding", side_effect=[binding, binding]
            ) as load_mock, mock.patch.object(
                MODULE, "stream_once", return_value=measured_run()
            ):
                MODULE.run(args)
            report = json.loads(output.read_text())
            self.assertEqual(load_mock.call_count, 2)
            for key, value in binding.items():
                self.assertEqual(report[key], value)

        with tempfile.TemporaryDirectory() as tmp:
            output = pathlib.Path(tmp) / "result.json"
            args.output = output
            changed = {**binding, "host_boot_id": BOOT_ID_B}
            with mock.patch.object(
                MODULE, "load_receipt_binding", side_effect=[binding, changed]
            ), mock.patch.object(
                MODULE, "stream_once", return_value=measured_run()
            ), self.assertRaisesRegex(
                RuntimeError, "receipt binding changed"
            ):
                MODULE.run(args)
            self.assertFalse(output.exists())

    def test_parses_last_acceptance_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = pathlib.Path(tmp) / "server.log"
            log.write_text(
                "DSPARK accept: 2.50 tok/step over 64 steps | draft accept 30.0%\n"
                "DSPARK accept: 3.25 tok/step over 128 steps | draft accept 45.5%\n"
            )
            result = MODULE.parse_accept_log(log)
        self.assertEqual(result["committed_tokens_per_step"], 3.25)
        self.assertEqual(result["steps"], 128)
        self.assertEqual(result["draft_accept_percent"], 45.5)

    def test_missing_acceptance_summary_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = pathlib.Path(tmp) / "server.log"
            log.write_text("ordinary log line\n")
            with self.assertRaisesRegex(RuntimeError, "no DSPARK accept summary"):
                MODULE.parse_accept_log(log)

    def test_acceptance_parser_can_ignore_stale_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = pathlib.Path(tmp) / "server.log"
            stale = "DSPARK accept: 9.00 tok/step over 1 steps | draft accept 99.0%\n"
            log.write_text(stale)
            offset = log.stat().st_size
            with log.open("a") as handle:
                handle.write(
                    "DSPARK accept: 3.50 tok/step over 128 steps | draft accept 50.0%\n"
                )
            result = MODULE.parse_accept_log(log, offset)
        self.assertEqual(result["committed_tokens_per_step"], 3.5)

    def test_acceptance_parser_rejects_log_rotation(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = pathlib.Path(tmp) / "server.log"
            log.write_text("old\n")
            stat = log.stat()
            identity = (stat.st_dev, stat.st_ino)
            prefix_hash = MODULE.hashlib.sha256(log.read_bytes()).hexdigest()
            log.unlink()
            log.write_text(
                "DSPARK accept: 9.00 tok/step over 1 steps | draft accept 99.0%\n"
            )
            with self.assertRaisesRegex(RuntimeError, "replaced|prefix changed"):
                MODULE.parse_accept_log(log, stat.st_size, identity, prefix_hash)

    def test_reasoning_tokens_start_decode_clock_and_are_hashed(self):
        events = [
            b'data: {"choices":[{"delta":{"reasoning_content":"think"}}]}\n',
            b'data: {"choices":[{"delta":{"content":"answer"}}]}\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n',
            b'data: {"usage":{"completion_tokens":2},"choices":[]}\n',
            b"data: [DONE]\n",
        ]
        with mock.patch.object(
            MODULE.benchmark_receipt,
            "urlopen_no_redirect",
            return_value=StreamResponse(events),
        ):
            result = MODULE.stream_once("http://unused", "model", "prompt", 2)
        self.assertEqual(result["completion_tokens"], 2)
        self.assertEqual(result["finish_reason"], "stop")
        self.assertFalse(result["truncated"])
        self.assertNotEqual(result["reasoning_sha256"], result["content_sha256"])
        self.assertGreater(result["decode_tok_s"], 0)

        length_events = list(events)
        length_events[2] = (
            b'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n'
        )
        with mock.patch.object(
            MODULE.benchmark_receipt,
            "urlopen_no_redirect",
            return_value=StreamResponse(length_events),
        ):
            truncated = MODULE.stream_once("http://unused", "model", "prompt", 2)
        self.assertEqual(truncated["finish_reason"], "length")
        self.assertTrue(truncated["truncated"])

    def test_stream_rejects_missing_duplicate_or_unknown_terminal_reason(self):
        variants = {
            "missing": [],
            "duplicate": [
                b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n',
                b'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n',
            ],
            "unknown": [
                b'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}\n'
            ],
        }
        for name, terminals in variants.items():
            events = [
                b'data: {"choices":[{"delta":{"content":"answer"}}]}\n',
                *terminals,
                b'data: {"usage":{"completion_tokens":2},"choices":[]}\n',
                b"data: [DONE]\n",
            ]
            with self.subTest(name=name), mock.patch.object(
                MODULE.benchmark_receipt,
                "urlopen_no_redirect",
                return_value=StreamResponse(events),
            ), self.assertRaisesRegex(RuntimeError, "finish_reason"):
                MODULE.stream_once("http://unused", "model", "prompt", 2)

    def test_stream_requires_done_and_exact_positive_completion_tokens(self):
        terminal = b'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n'
        usage_template = 'data: {"usage":{"completion_tokens":%s},"choices":[]}\n'
        for label, encoded in (
            ("bool", "true"),
            ("string", '"2"'),
            ("zero", "0"),
        ):
            events = [
                b'data: {"choices":[{"delta":{"content":"answer"}}]}\n',
                terminal,
                (usage_template % encoded).encode(),
                b"data: [DONE]\n",
            ]
            with self.subTest(label=label), mock.patch.object(
                MODULE.benchmark_receipt,
                "urlopen_no_redirect",
                return_value=StreamResponse(events),
            ), self.assertRaisesRegex(RuntimeError, "completion_tokens"):
                MODULE.stream_once("http://unused", "model", "prompt", 2)

        missing_done = [
            b'data: {"choices":[{"delta":{"content":"answer"}}]}\n',
            terminal,
            b'data: {"usage":{"completion_tokens":2},"choices":[]}\n',
        ]
        with mock.patch.object(
            MODULE.benchmark_receipt,
            "urlopen_no_redirect",
            return_value=StreamResponse(missing_done),
        ), self.assertRaisesRegex(RuntimeError, "DONE"):
            MODULE.stream_once("http://unused", "model", "prompt", 2)

    def test_stream_rejects_content_after_terminal_reason(self):
        events = [
            b'data: {"choices":[{"delta":{"content":"answer"}}]}\n',
            b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n',
            b'data: {"choices":[{"delta":{"content":"late"}}]}\n',
            b'data: {"usage":{"completion_tokens":2},"choices":[]}\n',
            b"data: [DONE]\n",
        ]
        with mock.patch.object(
            MODULE.benchmark_receipt,
            "urlopen_no_redirect",
            return_value=StreamResponse(events),
        ), self.assertRaisesRegex(RuntimeError, "after finish_reason"):
            MODULE.stream_once("http://unused", "model", "prompt", 2)

    def test_compare_rejects_checkpoint_identity_drift(self):
        case = result_cases()
        baseline = {
            **provenance(model_identity="a" * 64),
            "contract": {"temperature": 0},
            "aggregate_decode_tok_s": 22.0,
            "median_decode_tok_s": 22.0,
            "cases": case,
        }
        candidate = {
            **baseline,
            **provenance(
                model_identity="b" * 64,
                implementation_identity="d" * 64,
                receipt_digest="e" * 64,
            ),
            "aggregate_decode_tok_s": 70.0,
            "median_decode_tok_s": 70.0,
            "acceptance": {"committed_tokens_per_step": 4.0},
        }
        report = MODULE.compare_results(baseline, candidate, 65.0, 3.0)
        self.assertIn("model identities differ", report["failures"])

    def test_compare_uses_median_and_requires_distinct_implementation(self):
        case = result_cases()
        baseline = {
            **provenance(),
            "contract": {"temperature": 0},
            "aggregate_decode_tok_s": 100.0,
            "median_decode_tok_s": 20.0,
            "cases": case,
        }
        candidate = {
            **baseline,
            "aggregate_decode_tok_s": 100.0,
            "median_decode_tok_s": 60.0,
            "acceptance": {"committed_tokens_per_step": 4.0},
        }
        report = MODULE.compare_results(baseline, candidate, 65.0, 3.0)
        self.assertTrue(
            any(
                "implementation identities are identical" in item
                for item in report["failures"]
            )
        )
        self.assertTrue(any("below 65.00" in item for item in report["failures"]))

    def test_compare_rejects_per_run_finish_and_completion_count_mismatch(self):
        baseline = {
            **provenance(),
            "contract": {"temperature": 0},
            "aggregate_decode_tok_s": 22.0,
            "median_decode_tok_s": 22.0,
            "cases": result_cases(),
        }
        candidate = {
            **baseline,
            **provenance(implementation_identity="d" * 64, receipt_digest="e" * 64),
            "median_decode_tok_s": 85.0,
            "acceptance": {"committed_tokens_per_step": 4.0},
            "cases": result_cases(finish_reason="length", completion_tokens=3),
        }
        report = MODULE.compare_results(baseline, candidate, 85.0, 3.0)
        self.assertTrue(
            any("finish_reason differs" in item for item in report["failures"])
        )
        self.assertTrue(
            any("completion_tokens differs" in item for item in report["failures"])
        )

    def test_compare_rejects_truncated_marker_inconsistent_with_finish_reason(self):
        candidate_cases = result_cases()
        candidate_cases["code"]["runs"][0]["truncated"] = True
        baseline = {
            **provenance(),
            "contract": {"temperature": 0},
            "median_decode_tok_s": 22.0,
            "aggregate_decode_tok_s": 22.0,
            "cases": result_cases(),
        }
        candidate = {
            **baseline,
            **provenance(implementation_identity="d" * 64, receipt_digest="e" * 64),
            "median_decode_tok_s": 85.0,
            "acceptance": {"committed_tokens_per_step": 4.0},
            "cases": candidate_cases,
        }
        report = MODULE.compare_results(baseline, candidate, 85.0, 3.0)
        self.assertTrue(
            any("truncated marker is malformed" in item for item in report["failures"])
        )

    def test_compare_rejects_malformed_prompt_and_output_hashes(self):
        baseline = {
            **provenance(),
            "contract": {"temperature": 0},
            "median_decode_tok_s": 22.0,
            "aggregate_decode_tok_s": 22.0,
            "cases": result_cases(),
        }
        candidate_cases = result_cases()
        candidate_cases["code"]["prompt_sha256"] = "not-a-hash"
        candidate_cases["math"]["runs"][0]["output_sha256"] = "also-not-a-hash"
        candidate = {
            **baseline,
            **provenance(implementation_identity="d" * 64, receipt_digest="e" * 64),
            "median_decode_tok_s": 85.0,
            "acceptance": {"committed_tokens_per_step": 4.0},
            "cases": candidate_cases,
        }
        report = MODULE.compare_results(baseline, candidate, 85.0, 3.0)
        self.assertTrue(
            any("prompt_sha256 is malformed" in item for item in report["failures"])
        )
        self.assertTrue(
            any("output_sha256 is malformed" in item for item in report["failures"])
        )

    def test_compare_binds_same_gpu_and_os_boot_but_not_same_model_load(self):
        baseline = {
            **provenance(),
            "contract": {"temperature": 0},
            "median_decode_tok_s": 22.0,
            "aggregate_decode_tok_s": 22.0,
            "cases": result_cases(),
        }
        candidate = {
            **baseline,
            **provenance(implementation_identity="d" * 64, receipt_digest="e" * 64),
            "median_decode_tok_s": 85.0,
            "acceptance": {"committed_tokens_per_step": 4.0},
        }
        report = MODULE.compare_results(baseline, candidate, 85.0, 3.0)
        self.assertEqual(report["status"], "pass")

        legacy = {**candidate, "model_identity": "free-form-checkpoint-name"}
        report = MODULE.compare_results(baseline, legacy, 85.0, 3.0)
        self.assertTrue(
            any(
                "model_identity is malformed" in failure
                for failure in report["failures"]
            )
        )
        tampered = json.loads(json.dumps(candidate))
        tampered["benchmark_receipt"]["activation"]["pid"] = 999
        report = MODULE.compare_results(baseline, tampered, 85.0, 3.0)
        self.assertTrue(
            any(
                "embedded benchmark receipt is malformed" in failure
                for failure in report["failures"]
            )
        )

        # A Linux boot cohort does not imply a shared model-load/autotune instance;
        # distinct manifest identities remain mandatory even on the same boot.
        same_manifest = {**candidate, **provenance()}
        report = MODULE.compare_results(baseline, same_manifest, 85.0, 3.0)
        self.assertTrue(
            any(
                "implementation identities are identical" in failure
                for failure in report["failures"]
            )
        )
        for key, value in (
            ("host_boot_id", BOOT_ID_B),
            ("gpu_identity", {**GPU_ID, "uuid": "other"}),
        ):
            drifted = {**candidate, key: value}
            with self.subTest(key=key):
                report = MODULE.compare_results(baseline, drifted, 85.0, 3.0)
                self.assertTrue(
                    any(
                        key.replace("host_", "") in failure
                        for failure in report["failures"]
                    )
                )

    def test_atomic_result_writer_publishes_complete_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = pathlib.Path(tmp) / "result.json"
            MODULE.write_json_atomic(output, {"status": "complete"})
            self.assertEqual(json.loads(output.read_text()), {"status": "complete"})


if __name__ == "__main__":
    unittest.main()
