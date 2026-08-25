import importlib.util
import json
import pathlib
import tempfile
import unittest
from unittest import mock


MODULE_PATH = pathlib.Path(__file__).with_name("dflash_release_gate.py")
SPEC = importlib.util.spec_from_file_location("dflash_release_gate", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class DflashReleaseGateTests(unittest.TestCase):
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
            b'data: {"usage":{"completion_tokens":2},"choices":[]}\n',
            b"data: [DONE]\n",
        ]
        class Response:
            status = 200

            def __iter__(self):
                return iter(events)

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

        with mock.patch.object(MODULE.urllib.request, "urlopen", return_value=Response()):
            result = MODULE.stream_once("http://unused", "model", "prompt", 2)
        self.assertEqual(result["completion_tokens"], 2)
        self.assertNotEqual(result["reasoning_sha256"], result["content_sha256"])
        self.assertGreater(result["decode_tok_s"], 0)

    def test_compare_rejects_checkpoint_identity_drift(self):
        case = {
            name: {
                "prompt_sha256": name,
                "runs": [{"output_sha256": f"{name}-output"}],
            }
            for name in MODULE.PROMPTS
        }
        baseline = {
            "model_identity": "checkpoint-a",
            "implementation_identity": "plain-build",
            "contract": {"temperature": 0},
            "aggregate_decode_tok_s": 22.0,
            "median_decode_tok_s": 22.0,
            "cases": case,
        }
        candidate = {
            **baseline,
            "model_identity": "checkpoint-b",
            "implementation_identity": "dflash-build",
            "aggregate_decode_tok_s": 70.0,
            "median_decode_tok_s": 70.0,
            "acceptance": {"committed_tokens_per_step": 4.0},
        }
        report = MODULE.compare_results(baseline, candidate, 65.0, 3.0)
        self.assertIn("model identities differ", report["failures"])

    def test_compare_uses_median_and_requires_distinct_implementation(self):
        case = {
            name: {
                "prompt_sha256": name,
                "runs": [{"output_sha256": f"{name}-output"}],
            }
            for name in MODULE.PROMPTS
        }
        baseline = {
            "model_identity": "checkpoint",
            "implementation_identity": "same-build",
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
            any("implementation identities are identical" in item for item in report["failures"])
        )
        self.assertTrue(any("below 65.00" in item for item in report["failures"]))

    def test_atomic_result_writer_publishes_complete_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = pathlib.Path(tmp) / "result.json"
            MODULE.write_json_atomic(output, {"status": "complete"})
            self.assertEqual(json.loads(output.read_text()), {"status": "complete"})


if __name__ == "__main__":
    unittest.main()
