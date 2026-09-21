#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only

import json
import os
import pathlib
import stat
import tempfile
import unittest

import long_context_quality_contract as contract
import long_context_quality_io as quality_io
import long_context_quality_receipt as receipt


def provenance() -> dict:
    value = {
        "schema": contract.PROVENANCE_SCHEMA,
        "source_revision": "1" * 40,
        "binary_sha256": "2" * 64,
        "build_receipt_sha256": "3" * 64,
        "kernel_bundle_sha256": "4" * 64,
        "model_config_sha256": "5" * 64,
        "model_index_sha256": "6" * 64,
        "tokenizer_sha256": "7" * 64,
        "effective_environment_sha256": "8" * 64,
        "server_receipt_sha256": "9" * 64,
        "server_run_nonce": "a" * 64,
        "model": "qwen38",
        "runtime": dict(contract.REQUIRED_RUNTIME),
    }
    return value


def completion(target: int, answer: str) -> dict:
    return {
        "id": "cmpl-varying",
        "object": "text_completion",
        "created": 1,
        "model": "qwen38",
        "choices": [{"index": 0, "text": answer, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": target,
            "completion_tokens": 12,
            "total_tokens": target + 12,
            "prompt_tokens_details": {"cached_tokens": 0, "audio_tokens": 0},
            "completion_tokens_details": {
                "reasoning_tokens": 0,
                "audio_tokens": 0,
                "accepted_prediction_tokens": 0,
                "rejected_prediction_tokens": 0,
            },
            "time_to_first_token_ms": 500.0,
            "response_token/s": 85.0,
        },
    }


class ContractTests(unittest.TestCase):
    def test_provenance_is_exact_and_target_only_bf16(self) -> None:
        self.assertEqual(contract.validate_provenance(provenance()), provenance())
        for mutation in [
            ("schema", "wrong"),
            ("binary_sha256", "g" * 64),
            ("source_revision", "1" * 39),
        ]:
            value = provenance()
            value[mutation[0]] = mutation[1]
            with self.assertRaises(ValueError):
                contract.validate_provenance(value)
        for key, bad in [
            ("max_seq_len", 1_048_576),
            ("kv_cache_dtype", "nvfp4"),
            ("runtime_mode", "dflash-v3"),
            ("dflash", True),
            ("high_speed_swap", True),
            ("prefix_caching", True),
        ]:
            value = provenance()
            value["runtime"][key] = bad
            with self.assertRaises(ValueError):
                contract.validate_provenance(value)

    def test_tokenize_contract_rejects_shape_count_and_token_domain(self) -> None:
        self.assertEqual(
            contract.validate_tokenize({"tokens": [1, 2], "count": 2}), [1, 2]
        )
        for bad in [
            {"tokens": [], "count": 0},
            {"tokens": [1], "count": 2},
            {"tokens": [-1], "count": 1},
            {"tokens": [True], "count": 1},
            {"tokens": [1], "count": 1, "extra": 0},
        ]:
            with self.assertRaises(ValueError):
                contract.validate_tokenize(bad)

    def test_completion_requires_exact_answer_and_uncached_accounting(self) -> None:
        target = 262_145
        codes = {"early": "AAA1", "middle": "BBB2", "late": "CCC3"}
        expected = contract.expected_answer(codes)
        row = contract.validate_completion(completion(target, expected), target, codes)
        self.assertEqual(row["prompt_tokens"], target)
        self.assertEqual(row["cached_tokens"], 0)
        self.assertEqual(row["effective_prefill_tokens_per_second"], 524_290.0)
        for mutate in (
            lambda value: value["usage"].update(prompt_tokens=target - 1),
            lambda value: value["usage"]["prompt_tokens_details"].update(
                cached_tokens=1
            ),
            lambda value: value["choices"][0].update(text=expected + " extra"),
            lambda value: value["choices"][0].update(finish_reason="length"),
            lambda value: value["usage"].update(time_to_first_token_ms=0.0),
        ):
            value = completion(target, expected)
            mutate(value)
            with self.assertRaises(ValueError):
                contract.validate_completion(value, target, codes)

    def test_receipt_seal_is_create_new_read_only_and_canonical(self) -> None:
        receipt = {"schema": contract.RECEIPT_SCHEMA, "rows": []}
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "receipt.json"
            digest = contract.seal_receipt(path, receipt)
            self.assertEqual(digest, contract.sha256(contract.canonical_bytes(receipt)))
            self.assertEqual(json.loads(path.read_bytes()), receipt)
            info = path.stat()
            self.assertTrue(stat.S_ISREG(info.st_mode))
            self.assertEqual(info.st_nlink, 1)
            self.assertEqual(info.st_mode & 0o222, 0)
            with self.assertRaises(FileExistsError):
                contract.seal_receipt(path, receipt)

    def test_only_explicit_local_http_origins_are_admitted(self) -> None:
        for valid in [
            "http://127.0.0.1:8888",
            "http://localhost:80",
            "http://[::1]:9/",
        ]:
            self.assertEqual(quality_io.local_base_url(valid), valid.rstrip("/"))
        for invalid in [
            "https://127.0.0.1:8888",
            "http://example.com:8888",
            "http://user@localhost:80",
            "http://localhost",
            "http://localhost:80/path",
        ]:
            with self.assertRaises(ValueError):
                quality_io.local_base_url(invalid)

    def test_provenance_and_artifact_reads_require_stable_immutable_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            value = provenance()
            paths = {}
            for index, (key, argument) in enumerate(receipt.ARTIFACT_ARGUMENTS.items()):
                path = root / argument
                path.write_bytes(f"artifact-{index}".encode())
                path.chmod(0o555 if key == "binary_sha256" else 0o444)
                value[key] = contract.sha256(path.read_bytes())
                paths[key] = path
            self.assertEqual(
                receipt.verify_artifacts(value, paths),
                {key: value[key] for key in paths},
            )

            provenance_path = root / "provenance.json"
            provenance_path.write_bytes(contract.canonical_bytes(value))
            provenance_path.chmod(0o444)
            self.assertEqual(quality_io.load_provenance(provenance_path), value)

            paths["model_config_sha256"].chmod(0o644)
            with self.assertRaisesRegex(ValueError, "writable"):
                receipt.verify_artifacts(value, paths)
            paths["model_config_sha256"].chmod(0o444)
            link = root / "linked-provenance.json"
            os.symlink(provenance_path, link)
            with self.assertRaises(OSError):
                quality_io.load_provenance(link)


if __name__ == "__main__":
    unittest.main()
