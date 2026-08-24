#!/usr/bin/env python3

import hashlib
import importlib.util
import pathlib
import unittest
from unittest import mock


MODULE_PATH = pathlib.Path(__file__).with_name("mia_decode_sweep.py")
SPEC = importlib.util.spec_from_file_location("mia_decode_sweep", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FakeResponse:
    def __init__(self, lines):
        self.lines = lines

    def __enter__(self):
        return iter(self.lines)

    def __exit__(self, *_args):
        return False


class StreamingMeasurementTests(unittest.TestCase):
    def test_decode_clock_excludes_ttft_and_uses_usage_tokens(self):
        response = FakeResponse(
            [
                b'data: {"choices":[{"delta":{"content":"hel"}}]}\n',
                b'data: {"choices":[{"delta":{"content":"lo"}}]}\n',
                b'data: {"choices":[],"usage":{"completion_tokens":3}}\n',
                b"data: [DONE]\n",
            ]
        )
        with (
            mock.patch.object(MODULE.urllib.request, "urlopen", return_value=response),
            mock.patch.object(
                MODULE.time,
                "perf_counter",
                side_effect=[0.0, 1.0, 2.0, 3.0],
            ),
        ):
            result = MODULE.stream_once("http://server", "model", "prompt", 16)

        self.assertEqual(result["completion_tokens"], 3)
        self.assertEqual(result["ttft_seconds"], 1.0)
        self.assertEqual(result["decode_seconds"], 1.0)
        self.assertEqual(result["decode_tok_s"], 2.0)
        self.assertEqual(
            result["output_sha256"], hashlib.sha256(b"hello").hexdigest()
        )


if __name__ == "__main__":
    unittest.main()
