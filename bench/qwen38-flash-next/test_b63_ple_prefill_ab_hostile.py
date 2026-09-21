# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

import b63_ple_prefill_ab_contract as contract
import b63_ple_prefill_ab_http as http_io
import b63_ple_prefill_ab_identity as identity
import b63_ple_prefill_ab_model as model_identity
import b63_ple_prefill_ab_process as process_identity
import b63_ple_prefill_ab_validate as validate
from test_b63_ple_prefill_ab import receipt_line, response_for, server_log

HERE = Path(__file__).resolve().parent


class HostileResponseTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = contract.request_specs()["m2013"]
        self.response = response_for("m2013")

    def rejects(self, mutate) -> None:
        value = copy.deepcopy(self.response)
        mutate(value)
        with self.assertRaises(RuntimeError):
            http_io.validate_response(value, self.spec)

    def test_schema_and_oracle_mutations_reject(self) -> None:
        mutations = (
            lambda value: value.__setitem__("extra", 1),
            lambda value: value["choices"].append(copy.deepcopy(value["choices"][0])),
            lambda value: value["choices"][0]["message"].__setitem__(
                "reasoning_content", ""
            ),
            lambda value: value["choices"][0]["message"].__setitem__(
                "content", "ORCHID-7391"
            ),
            lambda value: value["usage"].__setitem__("prompt_tokens", 2_012),
            lambda value: value["usage"].__setitem__("completion_tokens", 27.0),
            lambda value: value["usage"]["prompt_tokens_details"].__setitem__(
                "cached_tokens", 1
            ),
            lambda value: value["usage"]["prompt_tokens_details"].__setitem__(
                "cached_tokens", 0.0
            ),
            lambda value: value["usage"].__setitem__(
                "time_to_first_token_ms", float("nan")
            ),
            lambda value: value["usage"].__setitem__("response_token/s", 0.0),
        )
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                self.rejects(mutate)


class HostileLogTests(unittest.TestCase):
    def check_rejects(self, text: str, arm: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.log"
            path.write_text(text)
            with self.assertRaises(RuntimeError):
                validate.validate_server_log(path, arm)

    def test_receipt_and_route_mutations_reject(self) -> None:
        valid = server_log("candidate")
        mutations = (
            valid.replace("heads=8", "heads=7", 1),
            valid.replace("num_tokens=38", "num_tokens=2013", 1),
            valid.replace(
                "projection_parity_required=true", "projection_parity_required=false", 1
            ),
            valid.replace(
                "performance_claim_allowed=false", "performance_claim_allowed=true", 1
            ),
            valid + receipt_line("candidate", 2013) + "\n",
            valid.replace("Done: 27 tokens (stop)\n", "", 1),
            valid + "ERROR async launch failed\n",
            valid + "prefix cache hit\n",
        )
        for text in mutations:
            with self.subTest():
                self.check_rejects(text, "candidate")

    def test_control_and_cross_family_receipts_reject(self) -> None:
        self.check_rejects(
            server_log("control") + receipt_line("candidate", 38) + "\n", "control"
        )
        self.check_rejects(
            server_log("candidate").replace("family=ple", "family=attention", 1),
            "candidate",
        )

    def test_duplicate_receipt_field_rejects(self) -> None:
        text = server_log("candidate").replace("family=ple", "family=ple family=ple", 1)
        self.check_rejects(text, "candidate")

    def test_malformed_quoted_receipt_rejects(self) -> None:
        text = server_log("candidate").replace("family=ple", 'family="ple', 1)
        self.check_rejects(text, "candidate")


class HostileIdentityTests(unittest.TestCase):
    @staticmethod
    def record(relative: str, digest: str = "b" * 64) -> dict:
        return {
            "relative": relative,
            "sha256": digest,
            "size": 1,
            "mode": 0o444,
            "device": 1,
            "inode": 1,
            "mtime_ns": 1,
        }

    def test_strict_json_rejects_duplicates_and_nonfinite(self) -> None:
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                identity.strict_json(raw)

    def test_sealed_file_rejects_hash_mode_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            raw = b'{"ok":true}'
            digest = hashlib.sha256(raw).hexdigest()
            path.write_bytes(raw)
            path.chmod(0o444)
            got, evidence = identity.stable_bytes(
                path, expected_sha256=digest, readonly=True
            )
            self.assertEqual(got, raw)
            self.assertEqual(evidence["mode"], 0o444)
            with self.assertRaises(RuntimeError):
                identity.stable_bytes(path, expected_sha256="0" * 64, readonly=True)
            path.chmod(0o644)
            with self.assertRaises(RuntimeError):
                identity.stable_bytes(path, expected_sha256=digest, readonly=True)
            link = Path(directory) / "link"
            link.symlink_to(path)
            with self.assertRaises((OSError, RuntimeError)):
                identity.stable_bytes(link, expected_sha256=digest)

    def test_model_manifest_requires_exact_census_metadata_and_root(self) -> None:
        metadata = [
            self.record(name, digest)
            for name, digest in sorted(model_identity.META_HASHES.items())
        ]
        main = [
            self.record(f"model-{index:05d}-of-00197.safetensors")
            for index in range(197)
        ]
        document = {
            "schema": model_identity.SCHEMA,
            "model_path": str(contract.MODEL),
            "content_root_sha256": "",
            "metadata": metadata,
            "main_shards": main,
            "ple_sidecars": [self.record("ple-offload/ple-00000.bin")],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            for candidate, accepted in (
                (document, True),
                ({**document, "main_shards": main[:-1]}, False),
            ):
                records = [
                    {key: item[key] for key in ("relative", "sha256", "size")}
                    for group in (
                        candidate["metadata"],
                        candidate["main_shards"],
                        candidate["ple_sidecars"],
                    )
                    for item in group
                ]
                root = hashlib.sha256(contract.canonical_bytes(records)).hexdigest()
                candidate = {**candidate, "content_root_sha256": root}
                path.chmod(0o644) if path.exists() else None
                raw = contract.canonical_bytes(candidate) + b"\n"
                path.write_bytes(raw)
                path.chmod(0o444)
                if accepted:
                    model_identity._load_manifest_document(
                        path, hashlib.sha256(raw).hexdigest(), root
                    )
                else:
                    with self.assertRaises(RuntimeError):
                        model_identity._load_manifest_document(
                            path, hashlib.sha256(raw).hexdigest(), root
                        )


class StaticContractTests(unittest.TestCase):
    def test_arm_environment_and_launch_are_isolated(self) -> None:
        self.assertEqual(contract.ARM_ORDER, ("control", "candidate"))
        for arm, expected in contract.ARM_SELECTORS.items():
            environment = process_identity.arm_environment(arm)
            self.assertEqual(environment["ATLAS_QWEN4_PLE_PREFILL_BATCH"], expected)
            self.assertTrue(all(environment[key] == "0" for key in contract.ROUTE_ENV))
            self.assertEqual(
                set(environment),
                set(contract.ROUTE_ENV)
                | {
                    "ATLAS_QWEN4_PLE_PREFILL_BATCH",
                    "LANG",
                    "LD_LIBRARY_PATH",
                    "PATH",
                    "RUST_LOG",
                },
            )
        argv = process_identity.server_argv(contract.MODEL)
        self.assertEqual(argv[argv.index("--max-num-seqs") + 1], "1")
        self.assertEqual(argv[argv.index("--max-prefill-tokens") + 1], "2048")
        self.assertNotIn("--enable-prefix-caching", argv)

    def test_listener_must_be_one_exact_ipv4_loopback(self) -> None:
        good = [{"table": "tcp", "address": "0100007F", "inode": "7"}]
        self.assertEqual(process_identity.validate_listener_records(good), good[0])
        for bad in (
            [],
            good * 2,
            [{**good[0], "address": "00000000"}],
            [{**good[0], "table": "tcp6"}],
        ):
            with self.subTest(bad=bad), self.assertRaises(RuntimeError):
                process_identity.validate_listener_records(bad)

    def test_failure_schema_has_no_performance_fields(self) -> None:
        source = (HERE / "b63_ple_prefill_ab.py").read_text()
        qualification = source.split(
            '"schema": "atlas-b63-ple-prefill-ab-qualification-v3"', 1
        )[1]
        self.assertIn('"performance_gate": performance_gate', qualification)
        failure = source.split('"schema": "atlas-b63-ple-prefill-ab-failure-v2"', 1)[1]
        failure = failure.split("result = output /", 1)[0]
        self.assertNotIn('"arms"', failure)
        self.assertNotIn('"metrics"', failure)
        self.assertNotIn('"performance_gate"', failure)
        self.assertIn('"performance_claim_allowed": False', failure)


if __name__ == "__main__":
    unittest.main()
