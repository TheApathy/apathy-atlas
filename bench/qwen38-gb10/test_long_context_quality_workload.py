#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only

import copy
import unittest

import long_context_quality_contract as contract
import long_context_quality_receipt as receipt
import long_context_quality_workload as workload


def fake_tokens(target: int) -> dict[str, list[int]]:
    return {
        "prefix": [10, 11, 12],
        "filler": [20, 21, 22, 23, 24],
        "early": [30, target % 1000, 31],
        "middle": [40, target % 997, 41],
        "late": [50, target % 991, 51],
        "query": [60, 61, 62, 63],
    }


class WorkloadTests(unittest.TestCase):
    def test_harness_identity_covers_every_execution_module(self) -> None:
        self.assertEqual(
            set(receipt.harness_identity()),
            {"contract", "io", "workload", "runner"},
        )

    def test_exact_ladder_and_unique_codes(self) -> None:
        self.assertEqual(
            workload.TARGET_TOKENS,
            (262_143, 262_144, 262_145, 500_000, 750_000, 999_000),
        )
        seen = set()
        for target in workload.TARGET_TOKENS:
            texts, codes = workload.component_texts(target)
            self.assertEqual(set(texts), set(workload.COMPONENT_KEYS))
            self.assertEqual(set(codes), {"early", "middle", "late"})
            self.assertNotIn(contract.expected_answer(codes), texts["query"])
            self.assertTrue(set(codes.values()).isdisjoint(seen))
            seen.update(codes.values())

    def test_prompt_is_exact_length_with_ordered_spread_needles(self) -> None:
        for target in workload.TARGET_TOKENS:
            prompt, positions = workload.assemble_prompt(target, fake_tokens(target))
            self.assertEqual(len(prompt), target)
            self.assertEqual(list(positions), ["early", "middle", "late"])
            self.assertLess(positions["early"], target // 8)
            self.assertTrue(target * 2 // 5 < positions["middle"] < target * 3 // 5)
            self.assertGreater(positions["late"], target * 7 // 8)
            tokens = fake_tokens(target)
            for label in positions:
                start = positions[label]
                self.assertEqual(
                    prompt[start : start + len(tokens[label])], tokens[label]
                )

    def test_prompt_rejects_missing_empty_boolean_and_oversized_components(
        self,
    ) -> None:
        target = workload.TARGET_TOKENS[0]
        cases = []
        missing = fake_tokens(target)
        missing.pop("late")
        cases.append(missing)
        empty = fake_tokens(target)
        empty["filler"] = []
        cases.append(empty)
        boolean = fake_tokens(target)
        boolean["query"] = [True]
        cases.append(boolean)
        oversized = fake_tokens(target)
        oversized["prefix"] = [1] * target
        cases.append(oversized)
        for tokens in cases:
            with self.assertRaises(ValueError):
                workload.assemble_prompt(target, tokens)

    def test_request_is_exact_deterministic_completion_bypass(self) -> None:
        prompt, _ = workload.assemble_prompt(262_143, fake_tokens(262_143))
        body = workload.completion_request("qwen38", prompt)
        self.assertEqual(body["prompt_token_ids"], prompt)
        self.assertEqual(body["prompt"], "")
        self.assertEqual(body["temperature"], 0.0)
        self.assertEqual(body["top_k"], 1)
        self.assertEqual(body["seed"], workload.SEED)
        self.assertEqual(body["max_tokens"], workload.OUTPUT_TOKENS)
        mutant = copy.deepcopy(body)
        mutant["prompt_token_ids"].pop()
        self.assertNotEqual(
            contract.canonical_bytes(mutant), contract.canonical_bytes(body)
        )

    def test_runner_binds_tokenize_prompt_accounting_and_repetitions(self) -> None:
        target = workload.TARGET_TOKENS[0]
        texts, codes = workload.component_texts(target)
        token_map = {
            text: [100 + index, 200 + index]
            for index, text in enumerate(texts.values())
        }
        calls = []

        def post(_base, route, body, _timeout):
            calls.append((route, body))
            if route == "/tokenize":
                tokens = token_map[body["prompt"]]
                value = {"tokens": tokens, "count": len(tokens)}
            else:
                prompt_tokens = len(body["prompt_token_ids"])
                answer = contract.expected_answer(codes)
                value = {
                    "id": f"cmpl-{len(calls)}",
                    "object": "text_completion",
                    "created": len(calls),
                    "model": "qwen38",
                    "choices": [{"index": 0, "text": answer, "finish_reason": "stop"}],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": 12,
                        "total_tokens": prompt_tokens + 12,
                        "prompt_tokens_details": {
                            "cached_tokens": 0,
                            "audio_tokens": 0,
                        },
                        "completion_tokens_details": {
                            "reasoning_tokens": 0,
                            "audio_tokens": 0,
                            "accepted_prediction_tokens": 0,
                            "rejected_prediction_tokens": 0,
                        },
                        "time_to_first_token_ms": 1000.0,
                        "response_token/s": 85.0,
                    },
                }
            return value, contract.canonical_bytes(value)

        row = receipt.run_target(
            "http://127.0.0.1:8888", {"model": "qwen38"}, target, 1, post
        )
        self.assertEqual(row["target_prompt_tokens"], target)
        self.assertEqual(len(row["observations"]), receipt.REPETITIONS)
        self.assertTrue(row["every_repetition_at_least_2000_prefill_tps"])
        self.assertEqual([route for route, _ in calls].count("/tokenize"), 6)
        self.assertEqual([route for route, _ in calls].count("/v1/completions"), 2)

    def test_runner_rejects_nondeterministic_completion_accounting(self) -> None:
        target = workload.TARGET_TOKENS[0]
        _, codes = workload.component_texts(target)
        completions = 0

        def post(_base, route, body, _timeout):
            nonlocal completions
            if route == "/tokenize":
                value = {"tokens": [1, 2], "count": 2}
            else:
                completions += 1
                count = 11 + completions
                value = {
                    "id": str(completions),
                    "object": "text_completion",
                    "created": completions,
                    "model": "qwen38",
                    "choices": [
                        {
                            "index": 0,
                            "text": contract.expected_answer(codes),
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": len(body["prompt_token_ids"]),
                        "completion_tokens": count,
                        "total_tokens": len(body["prompt_token_ids"]) + count,
                        "prompt_tokens_details": {
                            "cached_tokens": 0,
                            "audio_tokens": 0,
                        },
                        "completion_tokens_details": {
                            "reasoning_tokens": 0,
                            "audio_tokens": 0,
                            "accepted_prediction_tokens": 0,
                            "rejected_prediction_tokens": 0,
                        },
                        "time_to_first_token_ms": 1000.0,
                        "response_token/s": 85.0,
                    },
                }
            return value, contract.canonical_bytes(value)

        with self.assertRaisesRegex(ValueError, "nondeterministic"):
            receipt.run_target(
                "http://127.0.0.1:8888", {"model": "qwen38"}, target, 1, post
            )


if __name__ == "__main__":
    unittest.main()
