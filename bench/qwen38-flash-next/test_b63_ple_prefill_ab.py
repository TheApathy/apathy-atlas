# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import math
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import b63_ple_prefill_ab_contract as contract
import b63_ple_prefill_ab_http as http_io
import b63_ple_prefill_ab_validate as validate


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
    if arm != "candidate":
        raise ValueError("PLE receipt exists only for the candidate arm")
    fields = {**validate.COMMON_RECEIPT, "num_tokens": str(m)}
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
    if arm == "candidate":
        lines.append(receipt_line(arm, 38))
    for index in range(5):
        lines += [
            f"Session m{index}: 2013 prompt tokens, tools=false (0 defined)",
            "Request: temp=Some(0.0), max_tokens=32",
            "Chunked prefill start: 2013 prompt tokens, chunk_size=2013, max_tokens=32",
            "Done: 27 tokens (stop)",
        ]
        if arm == "candidate":
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
        self.assertEqual(
            contract.BINARY_SHA256,
            "23db269e1fdefaaa54885df6dff9a8297edac67521332dec5fcda076b0cfc644",
        )
        self.assertEqual(contract.BINARY_SIZE, 38_862_088)

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
        runs = [{"kind": "m2013", "semantic": semantic}] * 5
        report = validate.metrics(runs)
        self.assertEqual(report["ttft_ms"]["median"], 50_000.0)
        self.assertEqual(report["ttft_ms"]["p90"], 50_000.0)

    def test_log_census_all_arms(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for arm in contract.ARM_ORDER:
                path = Path(directory) / arm
                path.write_text(server_log(arm))
                report = validate.validate_server_log(path, arm)
                self.assertEqual(
                    report["selector_receipts"], 6 if arm == "candidate" else 0
                )

    def test_percentile_edges(self) -> None:
        self.assertEqual(contract.percentile([2.0], 0.9), 2.0)
        self.assertEqual(contract.percentile([1.0, 3.0], 0.5), 1.0)
        with self.assertRaises(ValueError):
            contract.percentile([math.inf], 0.9)

    def test_exact_raw_parity_receipt_admission(self) -> None:
        model_manifest, model_root = "a" * 64, "b" * 64
        document = {
            "schema": validate.RAW_PARITY_SCHEMA,
            "qualified": True,
            "performance_claim_allowed": False,
            "binary_sha256": contract.BINARY_SHA256,
            "build_files": contract.BUILD_FILES,
            "model_manifest_sha256": model_manifest,
            "model_content_root_sha256": model_root,
            "control_selector": 0,
            "candidate_selector": 1,
            "prompt_tokens": [38, 2013],
            "projection": "cublaslt_bf16_non_bit_exact",
            "hidden_equal": True,
            "live_state_equal": True,
            "checkpoint_state_equal": True,
            "canonical_output_equal": True,
            "capture_source_bundle_sha256": "c" * 64,
            "capture_artifacts_root_sha256": "d" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parity.json"
            raw = contract.canonical_bytes(document) + b"\n"
            path.write_bytes(raw)
            path.chmod(0o444)
            with (
                mock.patch.object(contract, "RAW_PARITY_RECEIPT", path),
                mock.patch.object(
                    contract,
                    "RAW_PARITY_RECEIPT_SHA256",
                    hashlib.sha256(raw).hexdigest(),
                ),
                mock.patch.object(contract, "MODEL_MANIFEST_SHA256", model_manifest),
                mock.patch.object(contract, "MODEL_CONTENT_ROOT_SHA256", model_root),
            ):
                self.assertEqual(validate.attest_raw_parity()["document"], document)
                mutations = {k: int(v) for k, v in document.items() if type(v) is bool}
                mutations["candidate_selector"] = True
                for key, replacement in mutations.items():
                    path.chmod(0o644)
                    mutated = {**document, key: replacement}
                    raw = contract.canonical_bytes(mutated) + b"\n"
                    path.write_bytes(raw)
                    path.chmod(0o444)
                    with (
                        mock.patch.object(
                            contract,
                            "RAW_PARITY_RECEIPT_SHA256",
                            hashlib.sha256(raw).hexdigest(),
                        ),
                        self.assertRaises(RuntimeError),
                    ):
                        validate.attest_raw_parity()

    def test_campaign_performance_and_hostile_gates(self) -> None:
        specs = contract.request_specs()

        def arm(name: str, pid: int, ttfts: list[float]) -> dict:
            runs = [
                {
                    "kind": "canary",
                    "semantic": {
                        "stable_sha256": specs["canary"].stable_sha256,
                    },
                }
            ] + [
                {
                    "kind": "m2013",
                    "semantic": {
                        "stable_sha256": specs["m2013"].stable_sha256,
                        "prompt_tokens": 2_013,
                        "ttft_ms": ttft,
                        "prompt_tokens_per_second": 2_013_000.0 / ttft,
                    },
                }
                for ttft in ttfts
            ]
            return {
                "arm": name,
                "process": {"pid": pid, "starttime_ticks": pid},
                "runs": runs,
                "metrics": validate.metrics(runs[1:]),
            }

        arms = [arm("control", 11, [900.0] * 5), arm("candidate", 12, [800.0] * 5)]
        gate = validate.campaign_gate(arms)
        self.assertEqual(gate["minimum_median_prompt_tokens_per_second"], 2_000.0)
        self.assertEqual(gate["maximum_p90_ttft_ms"], 1_006.5)
        passed = [
            value
            for key, value in gate.items()
            if key.endswith(("_pass", "_win", "_non_regression"))
        ]
        self.assertEqual(passed, [True] * 4)
        arms[1]["runs"][1]["semantic"]["prompt_tokens"] = 2_012
        with self.assertRaisesRegex(RuntimeError, "canonical output/usage"):
            validate.campaign_gate(arms)
        hostiles = (
            ([1_200.0] * 5, [1_100.0] * 5, "target_median"),
            ([900.0] * 3 + [1_200.0] * 2, [800.0] * 3 + [1_100.0] * 2, "target_p90"),
            ([700.0] * 5, [800.0] * 5, "candidate_median"),
        )
        for control, candidate, failure in hostiles:
            with self.assertRaisesRegex(RuntimeError, failure):
                pair = [arm("control", 21, control), arm("candidate", 22, candidate)]
                validate.campaign_gate(pair)


if __name__ == "__main__":
    unittest.main()
