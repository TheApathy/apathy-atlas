# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "repro/qwen38-coding/run.py"


def load_module():
    spec = importlib.util.spec_from_file_location("qwen38_coding_run", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Qwen38CodingReproTests(unittest.TestCase):
    def test_report_omits_generated_content(self) -> None:
        module = load_module()
        response = {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": "public synthetic response"},
                }
            ],
            "usage": {
                "completion_tokens": 1500,
                "prompt_tokens": 38,
                "response_token/s": 40.0,
            },
        }

        def fake_get_json(url, body=None):
            if url.endswith("/models"):
                return {"data": [{"id": "synthetic-qwen38"}]}
            self.assertEqual(body["reasoning_effort"], "none")
            self.assertEqual(body["temperature"], 0.0)
            return response

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            argv = [
                "run.py",
                "--output",
                str(output),
                "--repetitions",
                "3",
                "--no-expected-gate",
            ]
            with mock.patch.object(module, "get_json", side_effect=fake_get_json):
                with mock.patch.object(sys, "argv", argv):
                    self.assertEqual(module.main(), 0)

            report = json.loads(output.read_text())
            self.assertTrue(report["deterministic"])
            self.assertTrue(report["gate"]["pass"])
            self.assertEqual(len(report["runs"]), 3)
            for run in report["runs"]:
                self.assertNotIn("content", run)
                self.assertNotIn("reasoning_content", run)
                self.assertEqual(run["completion_tokens"], 1500)


if __name__ == "__main__":
    unittest.main()
