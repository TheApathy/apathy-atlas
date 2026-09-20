#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only

import os
import pathlib
import tempfile
import unittest
from unittest import mock

import capture_semantic_oracle as capture
import prefill_decode_repro as harness
import semantic_oracle_capture_io as capture_io


class CaptureSemanticOracleTests(unittest.TestCase):
    def provenance(self):
        environment = {"ATLAS_CAPTURE_TEST": "1"}
        return harness.validate_provenance(
            {
                "runtime_mode": "no-spec",
                "source_revision": "1" * 40,
                "binary_sha256": "2" * 64,
                "kernel_bundle_sha256": "3" * 64,
                "model_index_sha256": "4" * 64,
                "tokenizer_sha256": "5" * 64,
                "effective_environment": environment,
                "effective_environment_sha256": harness.sha256(
                    harness.canonical_bytes(environment)
                ),
                "full_environment_sha256": "6" * 64,
                "command_line_sha256": "7" * 64,
                "server_run_nonce": "8" * 64,
                "same_mode_environment_delta": {},
                "required_route_markers": {"ENGAGED TEST_CAPTURE: route": 1},
            },
            "no-spec",
        )

    def abi(self):
        provenance = self.provenance()
        return {
            "schema": capture.schema.ABI_ATTESTATION_SCHEMA,
            "source_revision": provenance["source_revision"],
            "binary_sha256": provenance["binary_sha256"],
            "kernel_bundle_sha256": provenance["kernel_bundle_sha256"],
            "ptx_entry": "inferspark_prefill",
            "host_argument_count": 13,
            "ptx_parameter_count": 13,
            "ptx_sha256": harness.CORRECTED_ATTN_PTX_SHA256,
        }

    @staticmethod
    def prefill_response(target, text="atlas "):
        return {
            "choices": [{"text": text, "finish_reason": "length"}],
            "usage": {
                "prompt_tokens": {2048: 2079, 8192: 8223, 32768: 32800}[target],
                "completion_tokens": 32,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        }

    @staticmethod
    def decode_response(content="answer"):
        return {
            "choices": [
                {
                    "message": {"content": content, "reasoning_content": ""},
                    "finish_reason": "length",
                }
            ],
            "usage": {
                "prompt_tokens": 38,
                "completion_tokens": 400,
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        }

    def test_abi_requires_exact_corrected_module_and_provenance(self):
        provenance = self.provenance()
        self.assertEqual(
            capture.validate_abi_attestation(self.abi(), provenance)[
                "ptx_parameter_count"
            ],
            13,
        )
        for field, value in (
            ("host_argument_count", 11),
            ("ptx_parameter_count", 11),
            ("ptx_sha256", "9" * 64),
            ("binary_sha256", "a" * 64),
            ("kernel_bundle_sha256", "b" * 64),
        ):
            bad = self.abi()
            bad[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                capture.validate_abi_attestation(bad, provenance)

    def test_prefill_capture_is_exact_zero_cache_and_repeated(self):
        calls = []

        def post(_endpoint, body):
            calls.append(body)
            target = int(body["prompt"].split()[3].split("-")[0])
            return self.prefill_response(target)

        rows = capture.capture_prefill_rows("http://127.0.0.1:8896", "model", post)
        self.assertEqual(len(rows), 15)
        self.assertEqual(len(calls), 30)
        self.assertEqual(len({row["request_body_sha256"] for row in rows}), 15)
        self.assertTrue(all(len(row["observations"]) == 2 for row in rows))
        self.assertTrue(
            all(
                observation["cached_prompt_tokens"] == 0
                for row in rows
                for observation in row["observations"]
            )
        )

    def test_prefill_rejects_repeat_drift_cache_and_bad_finish(self):
        for mutation in ("text", "cache", "finish", "bool_count"):
            count = 0

            def post(_endpoint, body):
                nonlocal count
                count += 1
                target = int(body["prompt"].split()[3].split("-")[0])
                response = self.prefill_response(target)
                if count == 2:
                    if mutation == "text":
                        response["choices"][0]["text"] = "different"
                    elif mutation == "cache":
                        response["usage"]["prompt_tokens_details"]["cached_tokens"] = 1
                    elif mutation == "finish":
                        response["choices"][0]["finish_reason"] = "stop"
                    else:
                        response["usage"]["completion_tokens"] = True
                return response

            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                capture.capture_prefill_rows("http://127.0.0.1:8896", "model", post)

    def test_decode_capture_requires_five_identical_uncached_outputs(self):
        rows = capture.capture_decode_rows(
            "http://127.0.0.1:8896", "model", lambda *_: self.decode_response()
        )
        self.assertEqual(len(rows), 5)
        self.assertEqual({row["prompt_tokens"] for row in rows}, {38})
        self.assertEqual(len({row["stable_output_sha256"] for row in rows}), 1)
        count = 0

        def drift(*_args):
            nonlocal count
            count += 1
            return self.decode_response("changed" if count == 5 else "answer")

        with self.assertRaises(ValueError):
            capture.capture_decode_rows("http://127.0.0.1:8896", "model", drift)
        for field, value in (
            ("cached_tokens", 1),
            ("cached_tokens", False),
            ("finish_reason", "stop"),
            ("completion_tokens", True),
        ):
            response = self.decode_response()
            target = response["usage"]["prompt_tokens_details"]
            if field == "finish_reason":
                target = response["choices"][0]
            elif field == "completion_tokens":
                target = response["usage"]
            target[field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                capture.capture_decode_rows(
                    "http://127.0.0.1:8896", "model", lambda *_: response
                )

    def test_candidate_is_provisional_and_production_rejects_it(self):
        prefill = capture.capture_prefill_rows(
            "http://127.0.0.1:8896",
            "model",
            lambda _endpoint, body: self.prefill_response(
                int(body["prompt"].split()[3].split("-")[0])
            ),
        )
        decode = capture.capture_decode_rows(
            "http://127.0.0.1:8896", "model", lambda *_: self.decode_response()
        )
        route = {"route_log_sha256": "c" * 64}
        with mock.patch.object(
            capture.harness, "validate_route_evidence_record", return_value=route
        ):
            candidate = capture.build_candidate(
                "model",
                self.provenance(),
                "a" * 64,
                self.abi(),
                "b" * 64,
                route,
                prefill,
                decode,
                capture.capture_identity(),
            )
        self.assertEqual(candidate["status"], "provisional-unapproved")
        self.assertFalse(candidate["performance_claim"])
        self.assertFalse(candidate["approval_boundary"]["accepted_by_harness"])
        markers = ("ENGAGED dflash", "ENGAGED SpEc")
        self.assertTrue(all(map(capture_io._is_speculative_route_marker, markers)))
        with self.assertRaises(ValueError):
            harness.validate_semantic_oracle(candidate)

    def test_retained_json_and_atomic_output_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            source = root / "source.json"
            raw = harness.canonical_bytes(self.abi()) + b"\n"
            source.write_bytes(raw)
            source.chmod(0o444)
            value, digest, descriptor, frozen = capture_io.open_retained_json(
                source, harness.sha256(raw), "ABI attestation"
            )
            self.assertEqual(value, self.abi())
            capture_io.assert_retained_unchanged(source, descriptor, digest, frozen)
            source.chmod(0o644)
            with self.assertRaises(ValueError):
                capture_io.assert_retained_unchanged(source, descriptor, digest, frozen)
            os.close(descriptor)
            with self.assertRaises(ValueError):
                capture_io.open_retained_json(
                    source, harness.sha256(raw), "ABI attestation"
                )
            source.chmod(0o444)
            with self.assertRaises(ValueError):
                capture_io.open_retained_json(source, "0" * 64, "ABI attestation")
            alias = root / "alias.json"
            alias.symlink_to(source)
            with self.assertRaises(OSError):
                capture_io.open_retained_json(
                    alias, harness.sha256(raw), "ABI attestation"
                )
            output = root / "candidate.json"
            capture_io.atomic_exclusive_write(output, b"candidate\n")
            self.assertEqual(output.read_bytes(), b"candidate\n")
            self.assertEqual(output.stat().st_mode & 0o777, 0o444)
            self.assertEqual(output.stat().st_nlink, 1)
            with self.assertRaises(FileExistsError):
                capture_io.atomic_exclusive_write(output, b"replacement\n")


if __name__ == "__main__":
    unittest.main()
