#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only

import copy
import importlib.util
import json
import os
import pathlib
import tempfile
import unittest
from unittest import mock

MODULE_PATH = pathlib.Path(__file__).with_name("prefill_decode_repro.py")
SPEC = importlib.util.spec_from_file_location("prefill_decode_repro", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class QualificationTests(unittest.TestCase):
    TEST_MARKER = "ENGAGED TEST_ROUTE: active"
    ORACLE_SHA = "0" * 64
    DELTA = {"ATLAS_PREFILL_QKNORM_ROPE": {"control": "0", "candidate": "1"}}

    @staticmethod
    def provenance(
        mode: str,
        *,
        candidate: bool = False,
        same_mode: bool = False,
        nonce_character: str = "a",
        binary_sha256: str = "2" * 64,
        full_environment_sha256: str | None = None,
    ) -> dict:
        environment = {
            "ATLAS_BASE_TEST_FLAG": "1",
            "ATLAS_PREFILL_QKNORM_ROPE": "1" if candidate else "0",
        }
        value = {
            "runtime_mode": mode,
            "source_revision": "1" * 40,
            "binary_sha256": binary_sha256,
            "kernel_bundle_sha256": "3" * 64,
            "model_index_sha256": "4" * 64,
            "tokenizer_sha256": "5" * 64,
            "effective_environment": environment,
            "effective_environment_sha256": MODULE.sha256(
                MODULE.canonical_bytes(environment)
            ),
            "full_environment_sha256": full_environment_sha256
            or ("7" if candidate else "8") * 64,
            "command_line_sha256": "c" * 64,
            "server_run_nonce": nonce_character * 64,
            "same_mode_environment_delta": (
                QualificationTests.DELTA if same_mode else {}
            ),
            "required_route_markers": {QualificationTests.TEST_MARKER: 1},
        }
        if mode == "dflash-v3":
            value["draft_index_sha256"] = "6" * 64
        return value

    @staticmethod
    def route_evidence(provenance: dict, *, base_hash: str = "9" * 64) -> dict:
        markers = provenance["required_route_markers"]
        binding = {
            "endpoint_port": 8896,
            "listener_socket_inode": 12345,
            "listener_pid": 4242,
            "process_start_time_ticks": 123456,
            "executable_sha256": provenance["binary_sha256"],
            "executable_device": 0,
            "executable_inode": 22,
            "cmdline_sha256": "c" * 64,
            "cmdline_bytes": 64,
            "effective_environment_sha256": provenance["effective_environment_sha256"],
            "full_environment_sha256": provenance["full_environment_sha256"],
            "base_environment_sha256": base_hash,
            "target_environment_sha256": "e" * 64,
            "target_command_line_sha256": "f" * 64,
            "route_log_device": 0,
            "route_log_inode": 33,
            "server_run_nonce": provenance["server_run_nonce"],
        }
        return {
            "route_log_sha256": "a" * 64,
            "route_log_bytes": 128,
            "engaged_lines_sha256": "b" * 64,
            "engaged_line_count": sum(markers.values()),
            "observed_route_markers": markers,
            "process_binding": binding,
        }

    @staticmethod
    def prefill_row(
        *,
        target: int = 2_048,
        index: int = 0,
        ttft_ms: float = 900.0,
        output_character: str = "d",
    ) -> dict:
        return {
            "target_prompt_tokens": target,
            "index": index,
            "prompt_tokens": target,
            "server_ttft_ms": ttft_ms,
            "effective_prefill_tokens_per_second": target * 1_000.0 / ttft_ms,
            "request_body_sha256": f"{target:064x}"[:-1] + f"{index:x}",
            "stable_output_sha256": output_character * 64,
            "cached_prompt_tokens": 0,
            "cache_accounting_present": True,
            "requested_continuation_tokens": 32,
            "completion_tokens": 32,
            "finish_reason": "length",
            "performance_passes_target": target * 1_000.0 / ttft_ms >= 2_000.0,
        }

    @classmethod
    def reference(cls, provenance: dict, rows: list[dict], comparison: str) -> dict:
        return {
            "schema": MODULE.REFERENCE_SCHEMA,
            "comparison_kind": comparison,
            "provenance": provenance,
            "route_evidence": cls.route_evidence(provenance),
            "semantic_oracle_sha256": cls.ORACLE_SHA,
            "rows": [
                {
                    key: row[key]
                    for key in (
                        "target_prompt_tokens",
                        "index",
                        "prompt_tokens",
                        "server_ttft_ms",
                        "effective_prefill_tokens_per_second",
                        "request_body_sha256",
                        "stable_output_sha256",
                        "cached_prompt_tokens",
                        "cache_accounting_present",
                        "requested_continuation_tokens",
                        "completion_tokens",
                        "finish_reason",
                    )
                }
                for row in rows
            ],
        }

    @staticmethod
    def prefill_response(details):
        return {
            "choices": [{"text": "atlas " * 32, "finish_reason": "length"}],
            "usage": {
                "prompt_tokens": 2_048,
                "completion_tokens": 32,
                "time_to_first_token_ms": 1_000.0,
                **({"prompt_tokens_details": details} if details is not None else {}),
            },
        }

    @staticmethod
    def decode_response(
        completion_tokens: int = 400,
        finish_reason: str = "length",
        content: str = "answer",
    ) -> dict:
        return {
            "choices": [
                {
                    "message": {"content": content, "reasoning_content": ""},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": 64,
                "completion_tokens": completion_tokens,
                "response_token/s": 85.0,
            },
        }

    def decode_row(
        self,
        *,
        index: int = 0,
        completion_tokens: int = 400,
        finish_reason: str = "length",
        content: str = "answer",
    ) -> dict:
        with mock.patch.object(
            MODULE,
            "post_json",
            return_value=self.decode_response(
                completion_tokens, finish_reason, content
            ),
        ):
            return MODULE.measure_decode("http://unused", "qwen38", index, 400, 85.0)

    @classmethod
    def semantic_oracle(
        cls,
        prefill: list[dict],
        decode: list[dict],
        *,
        model: str = "qwen38",
    ) -> dict:
        return {
            "schema": MODULE.SEMANTIC_ORACLE_SCHEMA,
            "model": model,
            "model_index_sha256": "4" * 64,
            "tokenizer_sha256": "5" * 64,
            "corrected_attention_abi": {
                "host_argument_count": 13,
                "ptx_parameter_count": 13,
                "ptx_sha256": MODULE.CORRECTED_ATTN_PTX_SHA256,
            },
            "prefill_rows": [
                {
                    key: row[key]
                    for key in MODULE.PREFILL_ORACLE_ROW_KEYS
                }
                for row in prefill
            ],
            "decode_rows": [
                {
                    key: row[key]
                    for key in MODULE.DECODE_ORACLE_ROW_KEYS
                }
                for row in decode
            ],
        }

    def canonical_semantic_workload(self) -> tuple[list[dict], list[dict]]:
        prefill = [
            self.prefill_row(target=target, index=index)
            for target in MODULE.DEFAULT_INPUTS
            for index in range(5)
        ]
        decode = [self.decode_row(index=index) for index in range(5)]
        return prefill, decode

    def test_two_thousand_tps_budgets(self):
        self.assertEqual(MODULE.target_ttft_ms(2_048), 1_024.0)
        self.assertEqual(MODULE.target_ttft_ms(8_192), 4_096.0)
        self.assertEqual(MODULE.target_ttft_ms(32_768), 16_384.0)
        self.assertEqual(MODULE.effective_prefill_tps(8_192, 4_096.0), 2_000.0)

    def test_workload_and_measurement_validation(self):
        MODULE.validate_workload([1, 2], 1, 1)
        for inputs, repetitions, decode in (
            ([], 1, 1),
            ([0], 1, 1),
            ([1, 1], 1, 1),
            ([1], 0, 1),
            ([1], 1, 0),
        ):
            with self.assertRaises(ValueError):
                MODULE.validate_workload(inputs, repetitions, decode)
        for tokens, milliseconds in ((0, 1.0), (1, 0.0), (-1, 1.0)):
            with self.assertRaises(ValueError):
                MODULE.effective_prefill_tps(tokens, milliseconds)

    def test_live_qualification_requires_canonical_full_gate(self):
        MODULE.validate_qualification_workload(
            [2_048, 8_192, 32_768], 5, 400, 32, 2_000.0, 85.0
        )
        for arguments in (
            ([2_048, 8_192], 5, 400, 32, 2_000.0, 85.0),
            ([2_048, 8_192, 32_768], 1, 400, 32, 2_000.0, 85.0),
            ([2_048, 8_192, 32_768], 6, 400, 32, 2_000.0, 85.0),
            ([2_048, 8_192, 32_768], 5, 399, 32, 2_000.0, 85.0),
            ([2_048, 8_192, 32_768], 5, 400, 31, 2_000.0, 85.0),
            ([2_048, 8_192, 32_768], 5, 400, 32, 1_999.0, 85.0),
            ([2_048, 8_192, 32_768], 5, 400, 32, 2_000.0, 84.9),
        ):
            with self.assertRaises(ValueError):
                MODULE.validate_qualification_workload(*arguments)

    def test_unique_prefix_changes_request_identity(self):
        prompts = {
            MODULE.build_prefill_prompt(2_048, 0, 32),
            MODULE.build_prefill_prompt(2_048, 1, 32),
            MODULE.build_prefill_prompt(8_192, 0, 32),
        }
        self.assertEqual(len(prompts), 3)

    def test_prefill_requires_zero_cache_and_complete_continuation(self):
        for details, expected in (
            (None, False),
            ({"cached_tokens": 1}, False),
            ({"cached_tokens": 0}, True),
        ):
            with mock.patch.object(
                MODULE, "post_json", return_value=self.prefill_response(details)
            ):
                row = MODULE.measure_prefill(
                    "http://unused", "qwen38", 2_048, 0, 2_000.0, 32
                )
            self.assertEqual(row["performance_passes_target"], expected)
        response = self.prefill_response({"cached_tokens": 0})
        response["usage"]["completion_tokens"] = 31
        with mock.patch.object(MODULE, "post_json", return_value=response):
            row = MODULE.measure_prefill(
                "http://unused", "qwen38", 2_048, 0, 2_000.0, 32
            )
        self.assertFalse(row["performance_passes_target"])
        response = self.prefill_response({"cached_tokens": 0})
        response["choices"][0]["finish_reason"] = "stop"
        with mock.patch.object(MODULE, "post_json", return_value=response):
            row = MODULE.measure_prefill(
                "http://unused", "qwen38", 2_048, 0, 2_000.0, 32
            )
        self.assertFalse(row["performance_passes_target"])

    def test_prefill_suppresses_only_pinned_qwen_stop_tokens(self):
        captured = {}

        def capture_request(_endpoint, body):
            captured.update(body)
            return self.prefill_response({"cached_tokens": 0})

        with mock.patch.object(MODULE, "post_json", side_effect=capture_request):
            MODULE.measure_prefill(
                "http://unused", "qwen38", 2_048, 0, 2_000.0, 32
            )
        self.assertEqual(
            captured["logit_bias"],
            {"248044": -100.0, "248045": -100.0, "248046": -100.0},
        )

    def test_provenance_binds_environment_nonce_and_known_delta(self):
        candidate = MODULE.validate_provenance(
            self.provenance(
                "dflash-v3", candidate=True, same_mode=True, nonce_character="b"
            ),
            "dflash-v3",
        )
        self.assertEqual(candidate["draft_index_sha256"], "6" * 64)
        self.assertEqual(candidate["same_mode_environment_delta"], self.DELTA)
        for mutation in (
            {"effective_environment_sha256": "0" * 64},
            {"server_run_nonce": "short"},
            {"full_environment_sha256": "short"},
            {
                "same_mode_environment_delta": {
                    "ATLAS_PREFILL_QKNORM_ROPE": {
                        "control": "1",
                        "candidate": "0",
                    }
                }
            },
            {
                "same_mode_environment_delta": {
                    "ATLAS_PROFILE": {"control": "0", "candidate": "1"}
                }
            },
        ):
            malformed = self.provenance("dflash-v3")
            malformed.update(mutation)
            with self.assertRaises(ValueError):
                MODULE.validate_provenance(malformed, "dflash-v3")

    def test_runtime_mode_and_endpoint_are_exact(self):
        self.assertEqual(
            MODULE.resolve_runtime_mode(MODULE.TARGET_VS_DFLASH, True, None),
            "no-spec",
        )
        self.assertEqual(
            MODULE.resolve_runtime_mode(MODULE.SAME_MODE, False, "dflash-v3"),
            "dflash-v3",
        )
        self.assertEqual(
            MODULE.endpoint_port("http://127.0.0.1:8896/v1/completions"), 8896
        )
        for endpoint in (
            "http://localhost:8896/v1/completions",
            "http://127.0.0.1:8896/v1/chat/completions",
            "http://127.0.0.1:8896/v1/completions?x=1",
            "https://127.0.0.1:8896/v1/completions",
        ):
            with self.assertRaises(ValueError):
                MODULE.endpoint_port(endpoint)

    def test_target_command_normalization_removes_only_canonical_dflash_tuple(self):
        control = b"/spark\x00serve\x00--model-name\x00qwen38\x00"
        candidate = (
            b"/spark\x00serve\x00--dflash\x00--draft-model\x00/draft\x00"
            b"--dflash-gamma\x0015\x00--dflash-quantization\x00nvfp4\x00"
            b"--model-name\x00qwen38\x00"
        )
        self.assertEqual(
            MODULE._normalized_target_command_line_sha256(control, "no-spec"),
            MODULE._normalized_target_command_line_sha256(candidate, "dflash-v3"),
        )
        with self.assertRaises(ValueError):
            MODULE._normalized_target_command_line_sha256(
                control + b"--speculative\x00", "no-spec"
            )
        with self.assertRaises(ValueError):
            MODULE._normalized_target_command_line_sha256(
                candidate + b"--self-speculative\x00", "dflash-v3"
            )

    def test_route_evidence_requires_exact_counts_and_retained_file(self):
        provenance = self.provenance("no-spec")
        binding = self.route_evidence(provenance)["process_binding"]
        required = {self.TEST_MARKER: 1}
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "server.log"
            path.write_text(
                "startup async_engaged=0\n"
                "spec-policy accept: raw-BF16 masked-argmax fast path engaged\n"
                f"INFO {self.TEST_MARKER}\n"
            )
            path.chmod(0o600)
            evidence = MODULE.capture_route_evidence(path, required, binding)
            self.assertEqual(evidence["observed_route_markers"], required)
            path.write_text(f"INFO {self.TEST_MARKER}\nINFO {self.TEST_MARKER}\n")
            with self.assertRaises(ValueError):
                MODULE.capture_route_evidence(path, required, binding)
            path.write_text(f"INFO {self.TEST_MARKER}\nINFO ENGAGED UNKNOWN: route\n")
            with self.assertRaises(ValueError):
                MODULE.capture_route_evidence(path, required, binding)
            path.write_text(
                f"INFO {self.TEST_MARKER}\n"
                "INFO DFLASH_UNKNOWN engaged for an undeclared route\n"
            )
            with self.assertRaises(ValueError):
                MODULE.capture_route_evidence(path, required, binding)
            link = pathlib.Path(directory) / "linked.log"
            link.symlink_to(path)
            with self.assertRaises(ValueError):
                MODULE.capture_route_evidence(link, required, binding)

    def _fake_proc(self, directory: pathlib.Path):
        proc = directory / "proc"
        process = proc / "4242"
        (proc / "net").mkdir(parents=True)
        (process / "fd").mkdir(parents=True)
        (process / "fdinfo").mkdir()
        binary = directory / "spark"
        binary.write_bytes(b"qualified spark binary")
        binary_hash = MODULE.sha256(binary.read_bytes())
        nonce = "a" * 64
        atlas_environment = {
            "ATLAS_BASE_TEST_FLAG": "1",
            "ATLAS_PREFILL_QKNORM_ROPE": "0",
        }
        environment_entries = [
            b"PATH=/usr/bin",
            b"CUDA_VISIBLE_DEVICES=0",
            *[f"{key}={value}".encode() for key, value in atlas_environment.items()],
            f"{MODULE.QUALIFICATION_NONCE_ENV}={nonce}".encode(),
        ]
        raw_environment = b"\x00".join(environment_entries) + b"\x00"
        provenance = self.provenance(
            "no-spec",
            binary_sha256=binary_hash,
            full_environment_sha256=MODULE._normalized_environment_sha256(
                raw_environment
            ),
        )
        provenance["effective_environment"] = atlas_environment
        provenance["effective_environment_sha256"] = MODULE.sha256(
            MODULE.canonical_bytes(atlas_environment)
        )
        provenance = MODULE.validate_provenance(provenance, "no-spec")
        route_log = directory / "server.log"
        route_log.write_text(
            f"{MODULE.QUALIFICATION_LOG_PREFIX} pid=4242 nonce={nonce}\n"
            f"INFO {self.TEST_MARKER}\n"
        )
        route_log.chmod(0o600)
        (proc / "net/tcp").write_text(
            "sl local_address rem_address st tx rx tr tm retr uid timeout inode\n"
            "0: 0100007F:22C0 00000000:0000 0A 00000000:00000000 "
            "00:00000000 00000000 1000 0 12345\n"
        )
        (process / "stat").write_text(
            "4242 (spark server) S " + " ".join(["0"] * 18 + ["123456"] + ["0"] * 10)
        )
        raw_cmdline = b"/qualified/spark\x00serve\x00--port\x008896\x00"
        (process / "cmdline").write_bytes(raw_cmdline)
        provenance["command_line_sha256"] = MODULE.sha256(raw_cmdline)
        provenance = MODULE.validate_provenance(provenance, "no-spec")
        (process / "environ").write_bytes(raw_environment)
        (process / "exe").symlink_to(binary)
        (process / "fd/1").symlink_to(route_log)
        (process / "fd/2").symlink_to(route_log)
        (process / "fd/3").symlink_to("socket:[12345]")
        (process / "fdinfo/1").write_text("flags:\t0100001\n")
        (process / "fdinfo/2").write_text("flags:\t0100001\n")
        return proc, route_log, provenance

    def test_live_binding_resolves_listener_and_attests_process(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = pathlib.Path(directory_name)
            proc, route_log, provenance = self._fake_proc(directory)
            descriptor = MODULE.open_route_log(route_log)
            try:
                binding = MODULE.bind_live_server(
                    "http://127.0.0.1:8896/v1/completions",
                    route_log,
                    descriptor,
                    provenance,
                    proc,
                )
            finally:
                os.close(descriptor)
        self.assertEqual(binding["listener_pid"], 4242)
        self.assertEqual(binding["listener_socket_inode"], 12345)
        self.assertEqual(binding["executable_sha256"], provenance["binary_sha256"])

    def test_listener_resolution_skips_unrelated_opaque_same_uid_process(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = pathlib.Path(directory_name)
            proc, _, _ = self._fake_proc(directory)
            opaque_fd = proc / "2557/fd"
            opaque_fd.mkdir(parents=True)
            (opaque_fd / "0").write_text("opaque")
            real_readlink = os.readlink

            def selective_readlink(path):
                if pathlib.Path(path) == opaque_fd / "0":
                    raise PermissionError(13, "permission denied", path)
                return real_readlink(path)

            with mock.patch.object(
                MODULE.os, "readlink", side_effect=selective_readlink
            ):
                self.assertEqual(MODULE._listener_pid(proc, {"12345"}), 4242)

                def listener_is_opaque(path):
                    if pathlib.Path(path) == proc / "4242/fd/3":
                        raise PermissionError(13, "permission denied", path)
                    return selective_readlink(path)

                with mock.patch.object(
                    MODULE.os, "readlink", side_effect=listener_is_opaque
                ), self.assertRaises(ValueError):
                    MODULE._listener_pid(proc, {"12345"})

    def test_route_writer_resolution_skips_unrelated_opaque_descriptor(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = pathlib.Path(directory_name)
            proc, route_log, _ = self._fake_proc(directory)
            opaque_fd = proc / "2557/fd/0"
            opaque_fd.parent.mkdir(parents=True)
            opaque_fd.write_text("opaque")
            route_stat = route_log.stat()
            route_identity = (route_stat.st_dev, route_stat.st_ino)
            real_stat = pathlib.Path.stat

            def selective_stat(path, *args, **kwargs):
                if pathlib.Path(path) == opaque_fd:
                    raise PermissionError(13, "permission denied", path)
                return real_stat(path, *args, **kwargs)

            with mock.patch.object(
                pathlib.Path, "stat", autospec=True, side_effect=selective_stat
            ):
                self.assertEqual(
                    MODULE._writable_route_log_descriptors(proc, route_identity),
                    {(4242, 1), (4242, 2)},
                )

            def listener_is_opaque(path, *args, **kwargs):
                if pathlib.Path(path) in {
                    opaque_fd,
                    proc / "4242/fd/1",
                }:
                    raise PermissionError(13, "permission denied", path)
                return real_stat(path, *args, **kwargs)

            with mock.patch.object(
                pathlib.Path,
                "stat",
                autospec=True,
                side_effect=listener_is_opaque,
            ):
                self.assertNotEqual(
                    MODULE._writable_route_log_descriptors(proc, route_identity),
                    {(4242, 1), (4242, 2)},
                )

    def test_process_attestation_emits_only_safe_manifest_fields(self):
        with tempfile.TemporaryDirectory() as directory_name:
            directory = pathlib.Path(directory_name)
            proc, _, provenance = self._fake_proc(directory)
            attestation = MODULE.process_attestation(4242, proc)
            waited = MODULE.wait_process_attestation(
                4242,
                pathlib.Path(os.readlink(proc / "4242/exe")),
                "http://127.0.0.1:8896/v1/completions",
                0.1,
                proc,
            )
        self.assertEqual(attestation["binary_sha256"], provenance["binary_sha256"])
        self.assertEqual(
            attestation["effective_environment"], provenance["effective_environment"]
        )
        self.assertNotIn("PATH", attestation["effective_environment"])
        self.assertNotIn(
            MODULE.QUALIFICATION_NONCE_ENV, attestation["effective_environment"]
        )
        self.assertEqual(waited, attestation)

    def test_live_binding_rejects_wrong_log_environment_and_marker_order(self):
        for mutation in (
            "stderr",
            "environment",
            "marker-order",
            "listener",
            "extra-writer",
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as name:
                directory = pathlib.Path(name)
                proc, route_log, provenance = self._fake_proc(directory)
                if mutation == "stderr":
                    other = directory / "other.log"
                    other.write_text("other")
                    (proc / "4242/fd/2").unlink()
                    (proc / "4242/fd/2").symlink_to(other)
                elif mutation == "environment":
                    (proc / "4242/environ").write_bytes(b"ATLAS_OTHER=1\x00")
                elif mutation == "marker-order":
                    route_log.write_text(
                        f"INFO {self.TEST_MARKER}\n"
                        f"{MODULE.QUALIFICATION_LOG_PREFIX} pid=4242 nonce={'a' * 64}\n"
                    )
                elif mutation == "listener":
                    tcp = proc / "net/tcp"
                    tcp.write_text(tcp.read_text().replace("0100007F", "00000000"))
                else:
                    helper = proc / "4343"
                    (helper / "fd").mkdir(parents=True)
                    (helper / "fdinfo").mkdir()
                    (helper / "fd/4").symlink_to(route_log)
                    (helper / "fdinfo/4").write_text("flags:\t0100001\n")
                descriptor = MODULE.open_route_log(route_log)
                try:
                    with self.assertRaises(ValueError):
                        MODULE.bind_live_server(
                            "http://127.0.0.1:8896/v1/completions",
                            route_log,
                            descriptor,
                            provenance,
                            proc,
                        )
                finally:
                    os.close(descriptor)

    def test_target_vs_dflash_binds_output_and_processes(self):
        control_provenance = self.provenance("no-spec", nonce_character="a")
        candidate_provenance = self.provenance("dflash-v3", nonce_character="b")
        candidate = MODULE.validate_provenance(candidate_provenance, "dflash-v3")
        rows = [self.prefill_row()]
        reference = self.reference(
            control_provenance,
            [self.prefill_row(ttft_ms=1_000.0)],
            MODULE.TARGET_VS_DFLASH,
        )
        MODULE.bind_prefill_reference(
            rows,
            reference,
            candidate,
            self.route_evidence(candidate_provenance),
            self.ORACLE_SHA,
        )
        self.assertTrue(rows[0]["matches_no_spec_reference"])
        self.assertTrue(rows[0]["passes_target"])
        with self.assertRaisesRegex(ValueError, "different semantic oracle"):
            MODULE.bind_prefill_reference(
                [self.prefill_row()],
                reference,
                candidate,
                self.route_evidence(candidate_provenance),
                "1" * 64,
            )
        rows[0]["stable_output_sha256"] = "e" * 64
        MODULE.bind_prefill_reference(
            rows,
            reference,
            candidate,
            self.route_evidence(candidate_provenance),
            self.ORACLE_SHA,
        )
        self.assertFalse(rows[0]["passes_target"])

        contaminated = self.provenance("dflash-v3", candidate=True, nonce_character="b")
        contaminated_candidate = MODULE.validate_provenance(contaminated, "dflash-v3")
        with self.assertRaises(ValueError):
            MODULE.bind_prefill_reference(
                [self.prefill_row()],
                reference,
                contaminated_candidate,
                self.route_evidence(contaminated),
                self.ORACLE_SHA,
            )

        for binding_key in (
            "target_command_line_sha256",
            "target_environment_sha256",
        ):
            drifted_evidence = self.route_evidence(candidate_provenance)
            drifted_evidence["process_binding"][binding_key] = "0" * 64
            with self.assertRaises(ValueError):
                MODULE.bind_prefill_reference(
                    [self.prefill_row()],
                    reference,
                    candidate,
                    drifted_evidence,
                    self.ORACLE_SHA,
                )

    def test_same_mode_allows_only_one_exact_flag_delta(self):
        control_provenance = self.provenance(
            "no-spec", same_mode=True, nonce_character="a"
        )
        candidate_provenance = self.provenance(
            "no-spec", candidate=True, same_mode=True, nonce_character="b"
        )
        candidate = MODULE.validate_provenance(candidate_provenance, "no-spec")
        rows = [self.prefill_row()]
        reference = self.reference(
            control_provenance, [self.prefill_row(ttft_ms=1_000.0)], MODULE.SAME_MODE
        )
        MODULE.bind_prefill_reference(
            rows,
            reference,
            candidate,
            self.route_evidence(candidate_provenance),
            self.ORACLE_SHA,
            MODULE.SAME_MODE,
        )
        self.assertTrue(rows[0]["matches_reference"])
        candidate_evidence = self.route_evidence(candidate_provenance)
        candidate_evidence["process_binding"]["base_environment_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            MODULE.bind_prefill_reference(
                rows,
                reference,
                candidate,
                candidate_evidence,
                self.ORACLE_SHA,
                MODULE.SAME_MODE,
            )

    def test_flashinfer_projection_is_an_exact_attributable_delta(self):
        flag = "ATLAS_PREFILL_PROJ_FLASHINFER"
        self.assertEqual(
            MODULE.validate_same_mode_environment_delta(
                {flag: {"control": "0", "candidate": "1"}}
            ),
            {flag: {"control": "0", "candidate": "1"}},
        )
        for malformed in (
            {flag: {"control": "1", "candidate": "0"}},
            {flag: {"control": "0", "candidate": "true"}},
        ):
            with self.assertRaises(ValueError):
                MODULE.validate_same_mode_environment_delta(malformed)

    def test_flashinfer_ssm_projection_is_a_separate_attributable_delta(self):
        attention = "ATLAS_PREFILL_PROJ_FLASHINFER"
        ssm = "ATLAS_PREFILL_SSM_FLASHINFER"
        delta = {ssm: {"control": "0", "candidate": "1"}}
        self.assertEqual(MODULE.validate_same_mode_environment_delta(delta), delta)
        for malformed in (
            {ssm: {"control": "1", "candidate": "0"}},
            {ssm: {"control": "0", "candidate": "true"}},
            {
                attention: {"control": "0", "candidate": "1"},
                ssm: {"control": "0", "candidate": "1"},
            },
        ):
            with self.assertRaises(ValueError):
                MODULE.validate_same_mode_environment_delta(malformed)

    def test_gdn_c143_is_an_exact_attributable_delta(self):
        flag = "ATLAS_GDN_C143_PREFILL"
        delta = {flag: {"control": "0", "candidate": "1"}}
        self.assertEqual(MODULE.validate_same_mode_environment_delta(delta), delta)
        for malformed in (
            {flag: {"control": "1", "candidate": "0"}},
            {flag: {"control": "0", "candidate": "true"}},
            {
                flag: {"control": "0", "candidate": "1"},
                "ATLAS_GDN_PREFILL_GATECACHE": {
                    "control": "0",
                    "candidate": "1",
                },
            },
        ):
            with self.assertRaises(ValueError):
                MODULE.validate_same_mode_environment_delta(malformed)

    def test_attention_gate_fusion_is_an_exact_attributable_delta(self):
        flag = "ATLAS_PREFILL_ATTN_GATE_FUSED"
        delta = {flag: {"control": "0", "candidate": "1"}}

        def manifest(candidate: bool, nonce: str) -> dict:
            value = self.provenance(
                "dflash-v3",
                candidate=candidate,
                same_mode=True,
                nonce_character=nonce,
            )
            environment = value["effective_environment"]
            del environment["ATLAS_PREFILL_QKNORM_ROPE"]
            environment[flag] = "1" if candidate else "0"
            value["effective_environment_sha256"] = MODULE.sha256(
                MODULE.canonical_bytes(environment)
            )
            value["same_mode_environment_delta"] = delta
            return value

        control = manifest(False, "a")
        candidate = manifest(True, "b")
        validated = MODULE.validate_provenance(candidate, "dflash-v3")
        rows = [self.prefill_row()]
        reference = self.reference(
            control, [self.prefill_row(ttft_ms=1_000.0)], MODULE.SAME_MODE
        )
        MODULE.bind_prefill_reference(
            rows,
            reference,
            validated,
            self.route_evidence(candidate),
            self.ORACLE_SHA,
            MODULE.SAME_MODE,
        )
        self.assertTrue(rows[0]["matches_reference"])

        for malformed in (
            {flag: {"control": "1", "candidate": "0"}},
            {flag: {"control": "0", "candidate": "2"}},
            {
                flag: {"control": "0", "candidate": "1"},
                "ATLAS_PREFILL_QKNORM_ROPE": {
                    "control": "0",
                    "candidate": "1",
                },
            },
            {"ATLAS_UNKNOWN_KERNEL": {"control": "0", "candidate": "1"}},
        ):
            with self.assertRaises(ValueError):
                MODULE.validate_same_mode_environment_delta(malformed)

        changed_base = self.route_evidence(candidate)
        changed_base["process_binding"]["base_environment_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            MODULE.bind_prefill_reference(
                [self.prefill_row()],
                reference,
                validated,
                changed_base,
                self.ORACLE_SHA,
                MODULE.SAME_MODE,
            )

    def test_semantic_oracle_binds_every_request_and_rejects_old_trajectory(self):
        prefill, decode = self.canonical_semantic_workload()
        oracle = self.semantic_oracle(prefill, decode)
        provenance = MODULE.validate_provenance(self.provenance("no-spec"), "no-spec")
        measured_prefill = copy.deepcopy(prefill)
        measured_decode = copy.deepcopy(decode)
        MODULE.bind_semantic_oracle(
            measured_prefill,
            measured_decode,
            oracle,
            "qwen38",
            provenance,
        )
        self.assertTrue(
            all(row["matches_semantic_oracle"] for row in measured_prefill)
        )
        self.assertTrue(
            all(row["matches_semantic_oracle"] for row in measured_decode)
        )

        # Concrete regression class: a deterministic old f51d/first-token
        # trajectory may equal its own control, but it must not equal the
        # independently frozen corrected-attention output.
        broken = copy.deepcopy(prefill)
        broken[0]["stable_output_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "stable_output_sha256"):
            MODULE.bind_semantic_oracle(
                broken, copy.deepcopy(decode), oracle, "qwen38", provenance
            )

        for field, replacement in (
            ("request_body_sha256", "e" * 64),
            ("prompt_tokens", prefill[0]["prompt_tokens"] + 1),
            ("completion_tokens", 31),
            ("finish_reason", "stop"),
        ):
            with self.subTest(field=field):
                mutated = copy.deepcopy(prefill)
                mutated[0][field] = replacement
                with self.assertRaisesRegex(ValueError, field):
                    MODULE.bind_semantic_oracle(
                        mutated,
                        copy.deepcopy(decode),
                        oracle,
                        "qwen38",
                        provenance,
                    )
        for field, replacement in (
            ("request_body_sha256", "e" * 64),
            ("prompt_tokens", decode[0]["prompt_tokens"] + 1),
            ("stable_output_sha256", "e" * 64),
            ("completion_tokens", 399),
            ("finish_reason", "stop"),
        ):
            with self.subTest(decode_field=field):
                mutated = copy.deepcopy(decode)
                mutated[0][field] = replacement
                with self.assertRaisesRegex(ValueError, field):
                    MODULE.bind_semantic_oracle(
                        copy.deepcopy(prefill),
                        mutated,
                        oracle,
                        "qwen38",
                        provenance,
                    )

    def test_semantic_oracle_schema_and_corrected_abi_are_exact(self):
        prefill, decode = self.canonical_semantic_workload()
        oracle = self.semantic_oracle(prefill, decode)
        validated = MODULE.validate_semantic_oracle(oracle)
        self.assertEqual(
            validated["corrected_attention_abi"],
            {
                "host_argument_count": 13,
                "ptx_parameter_count": 13,
                "ptx_sha256": MODULE.CORRECTED_ATTN_PTX_SHA256,
            },
        )
        mutations = []
        extra = copy.deepcopy(oracle)
        extra["unknown"] = True
        mutations.append(extra)
        wrong_schema = copy.deepcopy(oracle)
        wrong_schema["schema"] = "qwen38-semantic-oracle-v0"
        mutations.append(wrong_schema)
        missing_row = copy.deepcopy(oracle)
        missing_row["prefill_rows"].pop()
        mutations.append(missing_row)
        repeated_row = copy.deepcopy(oracle)
        repeated_row["prefill_rows"][-1] = copy.deepcopy(
            repeated_row["prefill_rows"][0]
        )
        mutations.append(repeated_row)
        for key, value in (
            ("host_argument_count", 11),
            ("ptx_parameter_count", 11),
            ("ptx_sha256", "0" * 64),
        ):
            invalid_abi = copy.deepcopy(oracle)
            invalid_abi["corrected_attention_abi"][key] = value
            mutations.append(invalid_abi)
        for mutation in mutations:
            with self.assertRaises(ValueError):
                MODULE.validate_semantic_oracle(mutation)

    def test_semantic_oracle_file_is_hash_locked_retained_and_immutable(self):
        prefill, decode = self.canonical_semantic_workload()
        raw = MODULE.canonical_bytes(self.semantic_oracle(prefill, decode)) + b"\n"
        expected = MODULE.sha256(raw)
        with tempfile.TemporaryDirectory() as directory_name:
            directory = pathlib.Path(directory_name)
            path = directory / "oracle.json"
            path.write_bytes(raw)
            path.chmod(0o444)
            oracle, actual, descriptor, frozen = MODULE.open_semantic_oracle(
                path, expected
            )
            try:
                self.assertEqual(actual, expected)
                self.assertEqual(oracle["schema"], MODULE.SEMANTIC_ORACLE_SCHEMA)
                MODULE.assert_semantic_oracle_unchanged(
                    path, descriptor, expected, frozen
                )
                retained = directory / "retained.json"
                path.rename(retained)
                path.write_bytes(raw)
                path.chmod(0o444)
                with self.assertRaises(ValueError):
                    MODULE.assert_semantic_oracle_unchanged(
                        path, descriptor, expected, frozen
                    )
            finally:
                os.close(descriptor)

            mutable = directory / "mutable.json"
            mutable.write_bytes(raw)
            mutable.chmod(0o444)
            _, _, descriptor, frozen = MODULE.open_semantic_oracle(mutable, expected)
            try:
                mutable.chmod(0o644)
                mutable.write_bytes(raw + b" ")
                mutable.chmod(0o444)
                with self.assertRaises(ValueError):
                    MODULE.assert_semantic_oracle_unchanged(
                        mutable, descriptor, expected, frozen
                    )
            finally:
                os.close(descriptor)

            wrong_hash = directory / "wrong-hash.json"
            wrong_hash.write_bytes(raw)
            wrong_hash.chmod(0o444)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                MODULE.open_semantic_oracle(wrong_hash, "1" * 64)

    def test_relative_prefill_requires_strict_median_and_p90_wins(self):
        rows = []
        for target, control in (
            (2_048, 1_000.0),
            (8_192, 4_000.0),
            (32_768, 16_000.0),
        ):
            for index in range(5):
                rows.append(
                    {
                        "target_prompt_tokens": target,
                        "server_ttft_ms": control - 100.0,
                        "reference_server_ttft_ms": control,
                    }
                )
        summary = MODULE.summarize_relative_prefill(rows)
        self.assertTrue(summary["passes"])
        rows[-1]["server_ttft_ms"] = 16_000.0
        self.assertFalse(MODULE.summarize_relative_prefill(rows)["passes"])
        rows[-1]["server_ttft_ms"] = 15_900.0
        rows[-2]["server_ttft_ms"] = 17_000.0
        self.assertFalse(MODULE.summarize_relative_prefill(rows)["passes"])
        with self.assertRaises(ValueError):
            MODULE.summarize_relative_prefill(rows[:10])

    def test_reference_writer_includes_control_ttft(self):
        prefill = [self.prefill_row()]
        decode = [self.decode_row()]
        provenance = self.provenance("no-spec")
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "reference.json"
            MODULE.write_prefill_reference(
                path,
                "qwen38",
                MODULE.TARGET_VS_DFLASH,
                provenance,
                "f" * 64,
                self.route_evidence(provenance),
                self.ORACLE_SHA,
                prefill,
                decode,
            )
            reference = json.loads(path.read_bytes())
        self.assertEqual(reference["schema"], MODULE.REFERENCE_SCHEMA)
        self.assertEqual(reference["rows"][0]["server_ttft_ms"], 900.0)

    def test_decode_gate_and_reference_are_fail_closed(self):
        complete = self.decode_row()
        self.assertTrue(complete["performance_passes_target"])
        self.assertFalse(
            self.decode_row(completion_tokens=399)["performance_passes_target"]
        )
        self.assertFalse(
            self.decode_row(finish_reason="stop")["performance_passes_target"]
        )
        reference = {
            "schema": MODULE.REFERENCE_SCHEMA,
            "decode_rows": [
                {
                    "index": 0,
                    "request_body_sha256": complete["request_body_sha256"],
                    "prompt_tokens": complete["prompt_tokens"],
                    "stable_output_sha256": complete["stable_output_sha256"],
                    "requested_completion_tokens": 400,
                    "completion_tokens": 400,
                    "finish_reason": "length",
                }
            ],
        }
        candidate = dict(complete)
        MODULE.bind_decode_reference([candidate], reference)
        self.assertTrue(candidate["passes_target"])
        candidate = dict(complete)
        candidate["stable_output_sha256"] = "0" * 64
        MODULE.bind_decode_reference([candidate], reference)
        self.assertFalse(candidate["passes_target"])
        self.assertTrue(MODULE.decode_reference_is_valid([complete]))
        self.assertFalse(MODULE.decode_reference_is_valid([]))


if __name__ == "__main__":
    unittest.main()
