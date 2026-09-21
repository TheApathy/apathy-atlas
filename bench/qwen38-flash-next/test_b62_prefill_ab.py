# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import b62_prefill_ab_contract as contract
import b62_prefill_ab_http as http_io
import b62_prefill_ab_validate as validate


def response_for(name: str) -> dict:
    spec = contract.request_specs()[name]
    content = contract.M2013_ANSWER if name == "m2013" else "x" * spec.content_bytes
    if name == "canary":
        # The known canary text is not retained; tests override its hash contract.
        content = "fixture"
    return {
        "id": "chatcmpl-fixture",
        "object": "chat.completion",
        "created": 1,
        "model": contract.MODEL_NAME,
        "system_fingerprint": "fp_atlas",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": spec.finish_reason,
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": spec.prompt_tokens,
            "completion_tokens": spec.completion_tokens,
            "total_tokens": spec.prompt_tokens + spec.completion_tokens,
            "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
            "completion_tokens_details": {
                "reasoning_tokens": 0,
                "audio_tokens": 0,
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
            "time_to_first_token_ms": 50_000.0,
            "response_token/s": 40.0,
        },
    }


def receipt_line(arm: str, m: int) -> str:
    fields = {
        **validate.COMMON_RECEIPT,
        **validate.FAMILY_RECEIPT[arm],
        "M": str(m),
    }
    return validate.RECEIPT_MARKER + " ".join(
        f"{key}={value}" for key, value in fields.items()
    )


def server_log(arm: str) -> str:
    lines = [
        "Listening on 127.0.0.1:8998",
        "Prefix caching: disabled",
        "SSM snapshot pool: Marconi 0 slots",
        "Scheduler started max_batch=1, mtp=false, ngram=false, num_drafts=0, "
        "policy=fifo, chunked_prefill=true, max_prefill_tokens=2048",
        "Session a: 38 prompt tokens, tools=false (0 defined)",
        "Request: temp=Some(0.0), max_tokens=400",
        "Chunked prefill start: 38 prompt tokens, chunk_size=38, max_tokens=400",
        "Done: 400 tokens (length)",
    ]
    if arm != "control":
        lines.append(receipt_line(arm, 38))
    for index in range(5):
        lines += [
            f"Session m{index}: 2013 prompt tokens, tools=false (0 defined)",
            "Request: temp=Some(0.0), max_tokens=32",
            "Chunked prefill start: 2013 prompt tokens, chunk_size=2013, max_tokens=32",
            "Done: 27 tokens (stop)",
        ]
        if arm != "control":
            lines.append(receipt_line(arm, 2013))
    return "\n".join(lines) + "\n"


class ContractTests(unittest.TestCase):
    def test_exact_request_identities(self) -> None:
        specs = contract.request_specs()
        self.assertEqual(len(specs["canary"].wire), 268)
        self.assertEqual(len(specs["m2013"].wire), 9_377)
        self.assertEqual(specs["m2013"].prompt_tokens, 2_013)
        self.assertEqual(specs["m2013"].completion_tokens, 27)
        self.assertEqual(
            specs["m2013"].content_sha256,
            "e331cbae40fd262c53717fe8dafd3f98ef409a59a7b18fb7f736428d7e97bfa1",
        )

    def test_m2013_response_and_metrics(self) -> None:
        response = response_for("m2013")
        semantic = http_io.validate_response(
            response, contract.request_specs()["m2013"]
        )
        self.assertEqual(
            semantic["stable_sha256"], contract.request_specs()["m2013"].stable_sha256
        )
        self.assertEqual(semantic["semantic_sha256"], contract.M2013_SEMANTIC_SHA256)
        self.assertAlmostEqual(semantic["prompt_tokens_per_second"], 40.26)
        runs = [
            {"kind": "m2013", "semantic": {**semantic, "ttft_ms": float(value)}}
            for value in (1, 2, 3, 4, 5)
        ]
        report = validate.metrics(runs)
        self.assertEqual(report["ttft_ms"]["median"], 3.0)
        self.assertEqual(report["ttft_ms"]["p90"], 5.0)

    def test_log_census_all_arms(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for arm in contract.ARM_ORDER:
                path = Path(directory) / arm
                path.write_text(server_log(arm))
                report = validate.validate_server_log(path, arm)
                self.assertEqual(
                    report["selector_receipts"], 0 if arm == "control" else 6
                )

    def test_percentile_edges(self) -> None:
        self.assertEqual(contract.percentile([2.0], 0.9), 2.0)
        self.assertEqual(contract.percentile([1.0, 3.0], 0.5), 1.0)
        with self.assertRaises(ValueError):
            contract.percentile([math.inf], 0.9)


if __name__ == "__main__":
    unittest.main()
