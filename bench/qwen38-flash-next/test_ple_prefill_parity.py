# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import hashlib
import itertools
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest import mock

import ple_prefill_parity as parity
import ple_prefill_parity_authority as root_authority
import ple_prefill_parity_capture as capture
import ple_prefill_parity_contract as contract
import ple_prefill_parity_log as log_runtime
import ple_prefill_parity_runtime as runtime
import ple_prefill_parity_validate as validate

HEX = "a" * 64


def lifecycle(*, forced: bool = False, terminated: bool = False) -> dict[str, object]:
    if forced and terminated:
        raise ValueError("lifecycle cannot be both forced and SIGTERM-terminated")
    return {
        "actions": ["sigterm", "sigkill"] if forced else ["sigterm"],
        "timeout_after_sigterm_seconds": 15 if forced else None,
        "returncode": -9 if forced else (-15 if terminated else 0),
        "clean_exit": not forced and not terminated,
        "sigterm_identity_rechecked": True,
        "sigkill_identity_rechecked": forced,
        "session_drain_signal_count": 0,
        "final_session_empty": True,
        "final_listener_empty": True,
        "final_gpu_inventory_empty": True,
        "build_model_authority_stable": True,
    }


def seal(path: Path, raw: bytes, mode: int = 0o444) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.chmod(0o644)
    path.write_bytes(raw)
    path.chmod(mode)
    return hashlib.sha256(raw).hexdigest()


def manifest(records: dict[str, str]) -> bytes:
    return "".join(
        f"{digest}  {name}\n" for name, digest in sorted(records.items())
    ).encode()


def record(name: str, index: int) -> dict[str, object]:
    return {
        "relative": name,
        "sha256": f"{index:064x}",
        "size": index + 1,
        "mode": 0o444,
        "device": 1,
        "inode": index + 1,
        "mtime_ns": index + 1,
    }


def materialize_record(model: Path, item: dict[str, object]) -> None:
    path = model / str(item["relative"])
    seal(path, b"\0" * int(item["size"]))
    info = path.lstat()
    item.update(
        {
            "mode": info.st_mode & 0o777,
            "device": info.st_dev,
            "inode": info.st_ino,
            "mtime_ns": info.st_mtime_ns,
        }
    )


@contextmanager
def authority_fixture(base: Path):
    build, model = base / "build", base / "model"
    (build / "release").mkdir(parents=True)
    model.mkdir()
    elf = bytearray(64)
    elf[:7] = b"\x7fELF\x02\x01\x01"
    elf[16:18] = (3).to_bytes(2, "little")
    elf[18:20] = (183).to_bytes(2, "little")
    elf[20:24] = (1).to_bytes(4, "little")
    elf[52:54] = (64).to_bytes(2, "little")
    elf_path = build / "release/spark"
    elf_hash = seal(elf_path, bytes(elf), 0o555)
    source_hash = "b" * 64
    qwen = "./crates/spark-model/src/layers/qwen4_ple.rs"
    callsite = "crates/spark-model/src/model/trait_impl/prefill_b/forward_layers.rs"
    callsite_hash = "c" * 64
    maps = {
        "source-manifest.sha256": {qwen: source_hash, "./z": HEX},
        "selected-source-manifest.sha256": {
            callsite: callsite_hash,
            **{f"selected/{i:02}.rs": HEX for i in range(24)},
        },
        "ptx-manifest.sha256": {f"./ptx/{i:03}.ptx": HEX for i in range(154)},
        "artifact-manifest.sha256": {
            "release/spark": elf_hash,
            **{f"artifact/{i:02}": HEX for i in range(13)},
        },
    }
    pins = {key: HEX for key in contract.PINS}
    pins.update(
        {
            "elf": elf_hash,
            "capture_source": source_hash,
            "capture_callsite": callsite_hash,
        }
    )
    for name, records in maps.items():
        raw = manifest(records)
        pin = contract.BUILD_FILES[name][0]
        pins[pin] = seal(build / name, raw)
    metadata = [
        record(name, i + 1)
        for i, name in enumerate(
            sorted(
                {
                    "config.json",
                    "model.safetensors.index.json",
                    "ple-offload/manifest.json",
                    "tokenizer.json",
                    "tokenizer_config.json",
                }
            )
        )
    ]
    main = [record(f"model-{i:05}.safetensors", i + 10) for i in range(197)]
    ple = [record(f"ple-offload/sidecar-{i:05}.bin", i + 300) for i in range(128)]
    for item in (*metadata, *main, *ple):
        materialize_record(model, item)
    content = [
        {key: item[key] for key in ("relative", "sha256", "size")}
        for group in (metadata, main, ple)
        for item in group
    ]
    content_root = hashlib.sha256(contract.canonical_bytes(content)).hexdigest()
    model_document = {
        "schema": parity.MODEL_SCHEMA,
        "model_path": str(model),
        "content_root_sha256": content_root,
        "metadata": metadata,
        "main_shards": main,
        "ple_sidecars": ple,
    }
    model_manifest = base / "model-manifest.json"
    pins["model_manifest"] = seal(
        model_manifest, contract.canonical_bytes(model_document) + b"\n"
    )
    pins["model_content_root"] = content_root
    elf_stat = elf_path.lstat()
    receipt = "\n".join(
        (
            "# Capture build",
            f"- Target signature: `{contract.TARGET}`",
            "- Kernel census: 154 PTX modules, 9 model-specific overrides",
            f"- Source manifest SHA256: `{pins['source_manifest']}` (2 files)",
            f"- Selected-source manifest SHA256: `{pins['selected_manifest']}` (25 files)",
            f"- PTX manifest SHA256: `{pins['ptx_manifest']}` (154 files)",
            f"- ELF identity: path=release/spark sha256={elf_hash} size=64 "
            f"device={elf_stat.st_dev} inode={elf_stat.st_ino} mtime_ns={elf_stat.st_mtime_ns} "
            f"nlink=1 mode=0555 build_id={'c' * 40}",
            "",
        )
    ).encode()
    pins["build_receipt"] = seal(build / "BUILD_RECEIPT.md", receipt)
    maps["artifact-manifest.sha256"].update(
        {
            name: pins[pin]
            for name, (pin, _) in contract.BUILD_FILES.items()
            if name != "artifact-manifest.sha256"
        }
    )
    raw = manifest(maps["artifact-manifest.sha256"])
    pins["artifact_manifest"] = seal(build / "artifact-manifest.sha256", raw)
    token_pins = {key: tuple(HEX for _ in range(4)) for key in contract.TOKEN_PINS}
    patches = (
        mock.patch.object(contract, "BUILD", build),
        mock.patch.object(contract, "ELF", elf_path),
        mock.patch.object(contract, "MODEL", model),
        mock.patch.object(contract, "MODEL_MANIFEST", model_manifest),
        mock.patch.object(contract, "ELF_SIZE", 64),
        mock.patch.object(contract, "SOURCE_FILES", 2),
        mock.patch.object(contract, "BUILD_ID", "c" * 40),
        mock.patch.dict(contract.PINS, pins, clear=True),
        mock.patch.dict(contract.TOKEN_PINS, token_pins, clear=True),
    )
    with ExitStack() as stack:
        for patch in patches:
            stack.enter_context(patch)
        yield build, pins


def artifact(frame: Path, name: str, raw: bytes) -> dict[str, object]:
    path = frame / name
    digest = seal(path, raw)
    info = path.lstat()
    return {
        "file": name,
        "dtype": "bf16le",
        "shape": [],
        "bytes": len(raw),
        "sha256": digest,
        "dev": info.st_dev,
        "ino": info.st_ino,
        "mode": "0444",
    }


def synthetic_runs() -> list[dict[str, object]]:
    runs = []
    for index, (name, arm) in enumerate(contract.SCHEDULE):
        nonce = f"{index + 1:064x}"
        root = Path(f"/tmp/atlas-ple-parity-synthetic-{index}")
        frames = []
        prompt_tokens = contract.request_specs()[name].prompt_tokens
        for start, count, reset in contract.FRAME_LAYOUT[prompt_tokens]:
            pins = contract.TOKEN_PINS[(name, start)]
            frames.append(
                {
                    "receipt": {
                        "frame_commit_mode": "0500",
                        "performance_claim_allowed": False,
                        "producer_stream_synchronized": True,
                        "request_tokens_sha256": pins[0],
                        "ple_prior_m": start,
                        "ple_prior_tokens_sha256": pins[1],
                        "ple_ordered_m": start + count,
                        "ple_ordered_tokens_sha256": pins[2],
                        "chunk_start": start,
                        "chunk_m": count,
                        "chunk_tokens_sha256": pins[3],
                        "reset_state": reset,
                        "continuation": not reset,
                        "slot_idx": 0,
                    },
                    "receipt_identity": {
                        "sha256": HEX,
                        "bytes": 1,
                        "dev": 1,
                        "ino": index + start + 1,
                        "mode": 0o444,
                        "nlink": 1,
                        "mtime_ns": 1,
                    },
                    "hidden": b"\0\0",
                    "live": b"\0\0",
                    "checkpoint": b"\0\0",
                }
            )
        runs.append(
            {
                "name": name,
                "arm": arm,
                "nonce": nonce,
                "capture_root": root,
                "process": {
                    "pid": 100 + index,
                    "starttime_ticks": 1_000 + index,
                    "elf_sha256": contract.PINS["elf"],
                    "argv": contract.server_argv(),
                    "environment": contract.arm_environment(arm, nonce, root),
                    "lifecycle": lifecycle(),
                },
                "response": {"synthetic": name},
                "frames": frames,
            }
        )
    return runs


class PleParityTests(unittest.TestCase):
    def test_bounded_server_log_retains_head_tail_and_seals(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp) / "campaign"
            parent.mkdir(mode=0o700)
            root = parent / "capture-0-m38-serial0"
            root.mkdir(mode=0o700)
            sink = log_runtime.BoundedServerLog(root)
            writer = log_runtime.os.dup(sink.stdout_fd)
            sink.start_before_spawn()
            sink.after_spawn()
            raw = b"HEAD" + b"x" * log_runtime.SERVER_LOG_LIMIT + b"TAIL"
            for start in range(0, len(raw), 64 << 10):
                log_runtime.write_all(writer, raw[start : start + (64 << 10)])
            log_runtime.os.close(writer)
            receipt = sink.finish()
            stored = Path(receipt["path"]).read_bytes()
            self.assertTrue(receipt["truncated"])
            self.assertEqual(receipt["raw_bytes"], len(raw))
            self.assertEqual(receipt["raw_sha256"], hashlib.sha256(raw).hexdigest())
            self.assertLessEqual(len(stored), log_runtime.SERVER_LOG_LIMIT)
            self.assertTrue(stored.startswith(b"HEAD"))
            self.assertTrue(stored.endswith(b"TAIL"))
            self.assertIn(log_runtime.SERVER_LOG_MARKER, stored)
            self.assertEqual(Path(receipt["path"]).stat().st_mode & 0o777, 0o444)

    def test_bounded_server_log_rejects_existing_symlink(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp) / "campaign"
            parent.mkdir(mode=0o700)
            root = parent / "capture"
            root.mkdir(mode=0o700)
            log = parent / ".capture.server.log"
            log.symlink_to(parent / "missing")
            with self.assertRaises(FileExistsError):
                log_runtime.BoundedServerLog(root)

    def test_bounded_server_log_pipe_failure_unlinks_created_file(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp) / "campaign"
            parent.mkdir(mode=0o700)
            root = parent / "capture"
            root.mkdir(mode=0o700)
            log = parent / ".capture.server.log"
            with (
                mock.patch.object(
                    log_runtime.os, "pipe2", side_effect=OSError("pipe failure")
                ),
                self.assertRaisesRegex(OSError, "pipe failure"),
            ):
                log_runtime.BoundedServerLog(root)
            self.assertFalse(log.exists())

    def test_bounded_server_log_thread_start_failure_is_pre_effect(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp) / "campaign"
            parent.mkdir(mode=0o700)
            root = parent / "capture"
            root.mkdir(mode=0o700)
            sink = log_runtime.BoundedServerLog(root)
            with (
                mock.patch.object(
                    log_runtime.threading.Thread,
                    "start",
                    side_effect=RuntimeError("thread start"),
                ),
                self.assertRaisesRegex(RuntimeError, "thread start"),
            ):
                sink.start_before_spawn()
            self.assertFalse(sink.path.exists())

    def test_bounded_server_log_start_then_interrupt_has_single_fd_owner(self):
        real_start = log_runtime.threading.Thread.start

        def start_then_interrupt(thread):
            real_start(thread)
            raise KeyboardInterrupt

        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp) / "campaign"
            parent.mkdir(mode=0o700)
            root = parent / "capture"
            root.mkdir(mode=0o700)
            sink = log_runtime.BoundedServerLog(root)
            with (
                mock.patch.object(
                    log_runtime.threading.Thread,
                    "start",
                    new=start_then_interrupt,
                ),
                self.assertRaises(KeyboardInterrupt),
            ):
                sink.start_before_spawn()
            self.assertFalse(sink.path.exists())
            read_fd, write_fd = log_runtime.os.pipe2(log_runtime.os.O_CLOEXEC)
            try:
                log_runtime.write_all(write_fd, b"fd-reuse-canary")
                self.assertEqual(log_runtime.os.read(read_fd, 15), b"fd-reuse-canary")
            finally:
                log_runtime.os.close(read_fd)
                log_runtime.os.close(write_fd)

    def test_bounded_server_log_propagates_drain_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp) / "campaign"
            parent.mkdir(mode=0o700)
            root = parent / "capture"
            root.mkdir(mode=0o700)
            sink = log_runtime.BoundedServerLog(root)
            with mock.patch.object(
                log_runtime.os, "read", side_effect=OSError("drain")
            ):
                sink.start_before_spawn()
                sink.after_spawn()
                with self.assertRaisesRegex(RuntimeError, "drain failed"):
                    sink.finish()

    def test_bounded_server_log_rejects_post_lstat_oversize_swap(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp) / "campaign"
            parent.mkdir(mode=0o700)
            root = parent / "capture"
            root.mkdir(mode=0o700)
            sink = log_runtime.BoundedServerLog(root)
            writer = log_runtime.os.dup(sink.stdout_fd)
            sink.start_before_spawn()
            sink.after_spawn()
            log_runtime.write_all(writer, b"original")
            log_runtime.os.close(writer)
            assert sink._thread is not None
            sink._thread.join(timeout=2)
            replacement = parent / "replacement"
            replacement.write_bytes(b"z" * (log_runtime.SERVER_LOG_LIMIT + 1))
            replacement.chmod(0o444)
            real_open = log_runtime.os.open

            def swap_then_open(path, flags, *args):
                if (
                    Path(path) == sink.path
                    and flags & log_runtime.os.O_ACCMODE == log_runtime.os.O_RDONLY
                ):
                    sink.path.unlink()
                    replacement.rename(sink.path)
                return real_open(path, flags, *args)

            with (
                mock.patch.object(log_runtime.os, "open", side_effect=swap_then_open),
                self.assertRaisesRegex(RuntimeError, "open identity drift"),
            ):
                sink.finish()

    def test_bounded_server_log_rejects_same_inode_chmod_after_open(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp) / "campaign"
            parent.mkdir(mode=0o700)
            root = parent / "capture"
            root.mkdir(mode=0o700)
            sink = log_runtime.BoundedServerLog(root)
            writer = log_runtime.os.dup(sink.stdout_fd)
            sink.start_before_spawn()
            sink.after_spawn()
            log_runtime.write_all(writer, b"original")
            log_runtime.os.close(writer)
            assert sink._thread is not None
            sink._thread.join(timeout=2)
            real_read = log_runtime.os.read
            changed = False

            def chmod_then_read(fd, size):
                nonlocal changed
                if not changed:
                    changed = True
                    sink.path.chmod(0o644)
                return real_read(fd, size)

            with (
                mock.patch.object(log_runtime.os, "read", side_effect=chmod_then_read),
                self.assertRaisesRegex(RuntimeError, "identity drift"),
            ):
                sink.finish()

    def test_placeholders_stop_before_filesystem(self):
        key = ("m38", 0)
        blocked = ("UNRELEASED", *contract.TOKEN_PINS[key][1:])
        with (
            mock.patch.dict(contract.TOKEN_PINS, {key: blocked}),
            mock.patch.object(parity, "_build_authority", side_effect=AssertionError),
            self.assertRaisesRegex(RuntimeError, "placeholders"),
        ):
            parity.preflight_authority()

    def test_authority_parses_every_cross_link(self):
        with (
            tempfile.TemporaryDirectory() as temp,
            authority_fixture(Path(temp)) as (build, pins),
        ):
            admitted = parity.preflight_authority()
            self.assertEqual(admitted["build"]["elf"]["sha256"], pins["elf"])
            receipt = build / "BUILD_RECEIPT.md"
            raw = receipt.read_bytes().replace(b"nlink=1", b"nlink=2")
            prior_receipt_hash = pins["build_receipt"]
            pins["build_receipt"] = seal(receipt, raw)
            contract.PINS["build_receipt"] = pins["build_receipt"]
            artifact_manifest = build / "artifact-manifest.sha256"
            raw = artifact_manifest.read_bytes().replace(
                prior_receipt_hash.encode(), pins["build_receipt"].encode()
            )
            pins["artifact_manifest"] = seal(artifact_manifest, raw)
            contract.PINS["artifact_manifest"] = pins["artifact_manifest"]
            with self.assertRaisesRegex(RuntimeError, "ELF identity"):
                parity.preflight_authority()

    def test_model_authority_rejects_same_path_inode_replacement(self):
        with (
            tempfile.TemporaryDirectory() as temp,
            authority_fixture(Path(temp)),
        ):
            target = contract.MODEL / "config.json"
            raw = target.read_bytes()
            target.rename(contract.MODEL / "config.json.prior")
            seal(target, raw)
            with self.assertRaisesRegex(RuntimeError, "model record identity"):
                parity._model_authority()

    def test_stable_reader_rejects_symlink_mode_hash_and_growth(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "value"
            digest = seal(path, b"value")
            self.assertEqual(
                capture.stable_bytes(
                    path,
                    expected_sha256=digest,
                    expected_size=5,
                    expected_mode=0o444,
                    limit=5,
                )[0],
                b"value",
            )
            path.chmod(0o555)
            with self.assertRaises(RuntimeError):
                capture.stable_bytes(
                    path,
                    expected_sha256=digest,
                    expected_size=5,
                    expected_mode=0o444,
                    limit=5,
                )
            link = root / "link"
            link.symlink_to(path)
            with self.assertRaises(RuntimeError):
                capture.stable_bytes(
                    link,
                    expected_sha256=None,
                    expected_size=None,
                    expected_mode=0o555,
                    limit=5,
                )

    def test_frame_receipt_exact_types_census_and_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "capture"
            root.mkdir(mode=0o700)
            nonce, arm, request_m = "d" * 64, "serial0", 38
            geometry = (0, 1, True)
            name = f"frame-{nonce}-{arm}-m38-s0-n1-reset"
            frame = root / name
            frame.mkdir(mode=0o700)
            hidden = artifact(frame, "post_ple_hidden.bf16le", b"\0" * 20_480)
            live = artifact(frame, "ple_live.bf16le", b"\0" * 184_320)
            checkpoint = artifact(frame, "ple_checkpoint.bf16le", b"\0" * 184_320)
            hidden["shape"], live["shape"], checkpoint["shape"] = (
                [1, 10_240],
                [10_240, 9],
                [10_240, 9],
            )
            rs, fs = root.lstat(), frame.lstat()
            receipt = {key: None for key in capture.RECEIPT_KEYS}
            receipt.update(
                {
                    "schema": contract.SCHEMA,
                    "boundary": "post_ple_pre_layer1",
                    "performance_claim_allowed": False,
                    "producer_stream_synchronized": True,
                    "producer_stream": 9,
                    "pid": 123,
                    "nonce": nonce,
                    "capture_root": str(root),
                    "capture_root_dev": rs.st_dev,
                    "capture_root_ino": rs.st_ino,
                    "frame": name,
                    "frame_dev": fs.st_dev,
                    "frame_ino": fs.st_ino,
                    "frame_commit_mode": "0500",
                    "arm": arm,
                    "selector": 0,
                    "request_m": 38,
                    "request_tokens_encoding": "u32le",
                    "request_tokens_sha256": HEX,
                    "ple_prior_m": 0,
                    "ple_prior_tokens_sha256": HEX,
                    "ple_ordered_m": 1,
                    "ple_ordered_tokens_sha256": HEX,
                    "chunk_start": 0,
                    "chunk_m": 1,
                    "chunk_tokens_encoding": "u32le",
                    "chunk_tokens_sha256": HEX,
                    "slot_idx": 0,
                    "reset_state": True,
                    "continuation": False,
                    "artifacts": {
                        "post_ple_hidden": hidden,
                        "ple_live": live,
                        "ple_checkpoint": checkpoint,
                    },
                }
            )
            seal(frame / "receipt.json", contract.canonical_bytes(receipt) + b"\n")
            frame.chmod(0o500)
            with mock.patch.dict(contract.FRAME_LAYOUT, {38: (geometry,)}, clear=True):
                self.assertEqual(
                    len(capture.load_arm(root, arm, request_m, nonce, 123)), 1
                )
                (root / "extra").write_bytes(b"")
                with self.assertRaisesRegex(RuntimeError, "census"):
                    capture.load_arm(root, arm, request_m, nonce, 123)
                (root / "extra").unlink()
                frame.chmod(0o700)
                with self.assertRaisesRegex(RuntimeError, "committed"):
                    capture.load_arm(root, arm, request_m, nonce, 123)
                frame.chmod(0o500)
                (frame / "receipt.json").chmod(0o644)
                with self.assertRaisesRegex(RuntimeError, "mode"):
                    capture.load_arm(root, arm, request_m, nonce, 123)

    def test_campaign_requires_bit_parity_before_any_diagnostics(self):
        runs = synthetic_runs()
        with mock.patch.object(
            validate, "canonical_response", return_value={"ok": True}
        ):
            result = validate.validate_campaign(runs)
            self.assertTrue(result["raw_bit_equal"])
            self.assertFalse(result["performance_claim_allowed"])
            self.assertFalse(result["timing_allowed"])
            runs[1]["frames"][0]["hidden"] = b"\x00\x01"
            with self.assertRaises(validate.ParityMismatch) as caught:
                validate.validate_campaign(runs)
        self.assertEqual(caught.exception.diagnostics[0]["label"], "m38.frame0.hidden")
        self.assertNotIn("time", caught.exception.diagnostics[0])

    def test_campaign_rejects_token_pin_and_process_reuse(self):
        runs = synthetic_runs()
        runs[0]["frames"][0]["receipt"]["chunk_tokens_sha256"] = "f" * 64
        with self.assertRaisesRegex(RuntimeError, "canonical token"):
            validate.validate_campaign(runs)
        runs = synthetic_runs()
        runs[1]["process"]["pid"] = runs[0]["process"]["pid"]
        runs[1]["process"]["starttime_ticks"] = runs[0]["process"]["starttime_ticks"]
        with self.assertRaisesRegex(RuntimeError, "reused"):
            validate.validate_campaign(runs)

    def test_campaign_admits_exact_graceful_terminated_or_forced_lifecycle(self):
        runs = synthetic_runs()
        runs[1]["process"]["lifecycle"] = lifecycle(terminated=True)
        runs[2]["process"]["lifecycle"] = lifecycle(forced=True)
        with mock.patch.object(
            validate, "canonical_response", return_value={"ok": True}
        ):
            self.assertTrue(validate.validate_campaign(runs)["qualified"])
            mutations = (
                ("actions", ["sigkill"]),
                ("timeout_after_sigterm_seconds", 14),
                ("returncode", -15),
                ("clean_exit", True),
                ("sigkill_identity_rechecked", False),
                ("session_drain_signal_count", 1),
                ("final_session_empty", False),
                ("final_listener_empty", False),
                ("final_gpu_inventory_empty", False),
                ("build_model_authority_stable", False),
            )
            for key, value in mutations:
                hostile = synthetic_runs()
                hostile[1]["process"]["lifecycle"] = lifecycle(forced=True)
                hostile[1]["process"]["lifecycle"][key] = value
                with (
                    self.subTest(key=key),
                    self.assertRaisesRegex(RuntimeError, "lifecycle"),
                ):
                    validate.validate_campaign(hostile)

    def test_lifecycle_cartesian_admits_only_three_exact_signal_outcomes(self):
        signal_names = ("sigterm", "sigkill", "sigint")
        action_paths = [
            list(path)
            for length in range(3)
            for path in itertools.product(signal_names, repeat=length)
        ]
        admitted = []
        for (
            actions,
            timeout,
            returncode,
            clean,
            term_check,
            kill_check,
        ) in itertools.product(
            action_paths,
            (None, 0, 14, 15, 16),
            (-15, -9, -2, 0, 1, 15),
            (False, True),
            (False, True),
            (False, True),
        ):
            candidate = lifecycle()
            candidate.update(
                {
                    "actions": actions,
                    "timeout_after_sigterm_seconds": timeout,
                    "returncode": returncode,
                    "clean_exit": clean,
                    "sigterm_identity_rechecked": term_check,
                    "sigkill_identity_rechecked": kill_check,
                }
            )
            exact = (actions, timeout, returncode, clean, term_check, kill_check) in (
                (["sigterm"], None, 0, True, True, False),
                (["sigterm"], None, -15, False, True, False),
                (["sigterm", "sigkill"], 15, -9, False, True, True),
            )
            if exact:
                self.assertIs(validate.admit_lifecycle(candidate), candidate)
                admitted.append((tuple(actions), timeout, returncode))
            else:
                with self.assertRaisesRegex(RuntimeError, "lifecycle_tuple="):
                    validate.admit_lifecycle(candidate)
        self.assertEqual(
            admitted,
            [
                (("sigterm",), None, -15),
                (("sigterm",), None, 0),
                (("sigterm", "sigkill"), 15, -9),
            ],
        )

        for key in (
            "session_drain_signal_count",
            "final_session_empty",
            "final_listener_empty",
            "final_gpu_inventory_empty",
            "build_model_authority_stable",
        ):
            candidate = lifecycle(terminated=True)
            candidate[key] = 1 if key == "session_drain_signal_count" else False
            with self.assertRaisesRegex(RuntimeError, "lifecycle_tuple="):
                validate.admit_lifecycle(candidate)

    def test_lifecycle_rejection_diagnostic_is_canonical_and_bounded(self):
        candidate = lifecycle()
        candidate["returncode"] = -2
        with self.assertRaisesRegex(RuntimeError, "lifecycle_tuple=") as caught:
            validate.admit_lifecycle(candidate)
        self.assertIn(
            '[["actions",["sigterm"]],["timeout_after_sigterm_seconds",null],["returncode",-2]',
            str(caught.exception),
        )

        candidate["actions"] = ["x" * 100_000] * 100
        candidate["returncode"] = 10**100_000
        with self.assertRaises(RuntimeError) as caught:
            validate.admit_lifecycle(candidate)
        self.assertLess(len(str(caught.exception)), 700)
        self.assertNotIn("x" * 100, str(caught.exception))

    def test_campaign_rejects_partial_or_unsealed_loaded_frame(self):
        with mock.patch.object(
            validate, "canonical_response", return_value={"ok": True}
        ):
            partial = synthetic_runs()
            del partial[0]["frames"][0]["receipt_identity"]
            with self.assertRaisesRegex(RuntimeError, "sealed frame"):
                validate.validate_campaign(partial)
            unsealed = synthetic_runs()
            unsealed[0]["frames"][0]["receipt_identity"]["mode"] = 0o644
            with self.assertRaisesRegex(RuntimeError, "sealed frame"):
                validate.validate_campaign(unsealed)

    def test_runtime_adapter_uses_fresh_process_and_internal_authority(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "capture"
            root.mkdir(mode=0o700)
            spec = contract.request_specs()["m38"]
            nonce = "d" * 64
            identity = {
                "pid": 1234,
                "starttime_ticks": 77,
                "elf_sha256": contract.PINS["elf"],
                "argv": contract.server_argv(),
                "environment": contract.arm_environment("serial0", nonce, root),
            }
            response = {"wire": "only-in-memory"}
            inventories: list[set[int]] = []
            admitted = {"build": "exact", "model": "exact"}
            authorized = mock.Mock()
            authorized.inventory.side_effect = lambda expected: (
                inventories.append(expected) or {"expected": sorted(expected)}
            )
            process = mock.Mock(pid=1234)
            with (
                mock.patch.object(
                    runtime.authority, "Authority", return_value=authorized
                ),
                mock.patch.object(
                    runtime.parity, "preflight_authority", return_value=admitted
                ) as preflight,
                mock.patch.object(runtime.subprocess, "Popen", return_value=process),
                mock.patch.object(runtime, "_starttime_ticks", return_value=77),
                mock.patch.object(runtime, "_wait_ready", return_value=identity),
                mock.patch.object(runtime, "_http_json", return_value=response),
                mock.patch.object(runtime, "_attest", return_value=identity),
                mock.patch.object(
                    runtime, "_terminate", return_value=lifecycle()
                ) as terminate,
                mock.patch.object(runtime, "_assert_port_free"),
            ):
                result = runtime.RuntimeAdapter()("m38", "serial0", spec, nonce, root)
            self.assertEqual(result["response"], response)
            self.assertEqual(result["process"], {**identity, "lifecycle": lifecycle()})
            self.assertEqual(inventories, [set(), {1234}, {1234}, set()])
            self.assertEqual(preflight.call_count, 3)
            terminate.assert_called_once_with(process, 77)

    def test_runtime_popen_failure_closes_and_unlinks_log(self):
        authorized = mock.Mock()
        admitted = {"authority": "exact"}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "capture"
            root.mkdir(mode=0o700)
            with (
                mock.patch.object(
                    runtime.authority, "Authority", return_value=authorized
                ),
                mock.patch.object(
                    runtime.parity, "preflight_authority", return_value=admitted
                ),
                mock.patch.object(
                    runtime.subprocess, "Popen", side_effect=OSError("spawn failed")
                ),
                mock.patch.object(runtime, "_assert_port_free"),
                self.assertRaisesRegex(OSError, "spawn failed"),
            ):
                runtime.RuntimeAdapter()(
                    "m38",
                    "serial0",
                    contract.request_specs()["m38"],
                    "d" * 64,
                    root,
                )
            self.assertFalse((root.parent / ".capture.server.log").exists())
            authorized.inventory.assert_called_once_with(set())

    def test_runtime_post_spawn_log_failure_owns_cleanup(self):
        authorized = mock.Mock()
        admitted = {"authority": "exact"}
        process = mock.Mock(pid=1234)
        server_log = mock.Mock(stdout_fd=8)
        server_log.after_spawn.side_effect = RuntimeError("close failed")
        server_log.finish.return_value = {"excerpt": "bounded"}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "capture"
            root.mkdir(mode=0o700)
            with (
                mock.patch.object(
                    runtime.authority, "Authority", return_value=authorized
                ),
                mock.patch.object(
                    runtime.parity, "preflight_authority", return_value=admitted
                ),
                mock.patch.object(
                    runtime, "_BoundedServerLog", return_value=server_log
                ),
                mock.patch.object(runtime.subprocess, "Popen", return_value=process),
                mock.patch.object(runtime, "_terminate_unattested") as terminate,
                mock.patch.object(runtime, "_assert_port_free"),
                self.assertRaisesRegex(RuntimeError, "close failed"),
            ):
                runtime.RuntimeAdapter()(
                    "m38",
                    "serial0",
                    contract.request_specs()["m38"],
                    "d" * 64,
                    root,
                )
        terminate.assert_called_once_with(process)
        self.assertEqual(
            authorized.inventory.call_args_list, [mock.call(set()), mock.call(set())]
        )
        self.assertEqual(server_log.after_spawn.call_count, 2)
        server_log.finish.assert_called_once_with()

    def test_runtime_keyboard_interrupt_after_spawn_owns_cleanup(self):
        authorized = mock.Mock()
        admitted = {"authority": "exact"}
        process = mock.Mock(pid=1234)
        server_log = mock.Mock(stdout_fd=8)
        server_log.finish.return_value = {"excerpt": "bounded"}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "capture"
            root.mkdir(mode=0o700)
            with (
                mock.patch.object(
                    runtime.authority, "Authority", return_value=authorized
                ),
                mock.patch.object(
                    runtime.parity, "preflight_authority", return_value=admitted
                ),
                mock.patch.object(
                    runtime, "_BoundedServerLog", return_value=server_log
                ),
                mock.patch.object(runtime.subprocess, "Popen", return_value=process),
                mock.patch.object(
                    runtime, "_starttime_ticks", side_effect=KeyboardInterrupt
                ),
                mock.patch.object(runtime, "_terminate_unattested") as terminate,
                mock.patch.object(runtime, "_assert_port_free"),
                self.assertRaises(KeyboardInterrupt),
            ):
                runtime.RuntimeAdapter()(
                    "m38",
                    "serial0",
                    contract.request_specs()["m38"],
                    "d" * 64,
                    root,
                )
        terminate.assert_called_once_with(process)
        self.assertEqual(
            authorized.inventory.call_args_list, [mock.call(set()), mock.call(set())]
        )
        server_log.finish.assert_called_once_with()

    def test_runtime_spawn_return_then_sigint_unmask_owns_cleanup(self):
        authorized = mock.Mock()
        admitted = {"authority": "exact"}
        process = mock.Mock(pid=1234)
        server_log = mock.Mock(stdout_fd=8)
        server_log.finish.return_value = {"excerpt": "bounded"}
        primary = KeyboardInterrupt("pending SIGINT after Popen return")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "capture"
            root.mkdir(mode=0o700)
            with (
                mock.patch.object(
                    runtime.authority, "Authority", return_value=authorized
                ),
                mock.patch.object(
                    runtime.parity, "preflight_authority", return_value=admitted
                ) as preflight,
                mock.patch.object(
                    runtime, "_BoundedServerLog", return_value=server_log
                ),
                mock.patch.object(runtime.subprocess, "Popen", return_value=process),
                mock.patch.object(
                    runtime.signal,
                    "pthread_sigmask",
                    side_effect=[set(), primary],
                ),
                mock.patch.object(runtime, "_terminate_unattested") as terminate,
                mock.patch.object(runtime, "_assert_port_free"),
                self.assertRaises(KeyboardInterrupt) as caught,
            ):
                runtime.RuntimeAdapter()(
                    "m38",
                    "serial0",
                    contract.request_specs()["m38"],
                    "d" * 64,
                    root,
                )
        self.assertIs(caught.exception, primary)
        terminate.assert_called_once_with(process)
        self.assertEqual(preflight.call_count, 3)
        server_log.after_spawn.assert_called_once_with()
        server_log.finish.assert_called_once_with()

    def test_runtime_system_exit_after_spawn_owns_cleanup(self):
        authorized = mock.Mock()
        process = mock.Mock(pid=1234)
        server_log = mock.Mock(stdout_fd=8)
        server_log.finish.return_value = {"excerpt": "bounded"}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "capture"
            root.mkdir(mode=0o700)
            with (
                mock.patch.object(
                    runtime.authority, "Authority", return_value=authorized
                ),
                mock.patch.object(
                    runtime.parity,
                    "preflight_authority",
                    return_value={"authority": "exact"},
                ),
                mock.patch.object(
                    runtime, "_BoundedServerLog", return_value=server_log
                ),
                mock.patch.object(runtime.subprocess, "Popen", return_value=process),
                mock.patch.object(
                    runtime, "_starttime_ticks", side_effect=SystemExit("stop")
                ),
                mock.patch.object(runtime, "_terminate_unattested") as terminate,
                mock.patch.object(runtime, "_assert_port_free"),
                self.assertRaisesRegex(SystemExit, "stop"),
            ):
                runtime.RuntimeAdapter()(
                    "m38",
                    "serial0",
                    contract.request_specs()["m38"],
                    "d" * 64,
                    root,
                )
        terminate.assert_called_once_with(process)
        self.assertEqual(
            authorized.inventory.call_args_list, [mock.call(set()), mock.call(set())]
        )
        server_log.finish.assert_called_once_with()

    def test_runtime_preserves_interrupt_and_attempts_every_finalizer(self):
        authorized = mock.Mock()
        authorized.inventory.side_effect = [{}, RuntimeError("gpu cleanup")]
        admitted = {"authority": "exact"}
        process = mock.Mock(pid=1234)
        server_log = mock.Mock(stdout_fd=8)
        server_log.finish.side_effect = RuntimeError("log cleanup")
        primary = KeyboardInterrupt("primary")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "capture"
            root.mkdir(mode=0o700)
            with (
                mock.patch.object(
                    runtime.authority, "Authority", return_value=authorized
                ),
                mock.patch.object(
                    runtime.parity, "preflight_authority", return_value=admitted
                ) as preflight,
                mock.patch.object(
                    runtime, "_BoundedServerLog", return_value=server_log
                ),
                mock.patch.object(runtime.subprocess, "Popen", return_value=process),
                mock.patch.object(runtime, "_starttime_ticks", side_effect=primary),
                mock.patch.object(
                    runtime,
                    "_terminate_unattested",
                    side_effect=RuntimeError("process cleanup"),
                ) as terminate,
                mock.patch.object(runtime, "_assert_port_free"),
                self.assertRaises(KeyboardInterrupt) as caught,
            ):
                runtime.RuntimeAdapter()(
                    "m38",
                    "serial0",
                    contract.request_specs()["m38"],
                    "d" * 64,
                    root,
                )
        self.assertIs(caught.exception, primary)
        terminate.assert_called_once_with(process)
        self.assertEqual(preflight.call_count, 3)
        server_log.finish.assert_called_once_with()
        self.assertIn("process cleanup", "\n".join(primary.__notes__))
        self.assertIn("gpu cleanup", "\n".join(primary.__notes__))
        self.assertIn("log cleanup", "\n".join(primary.__notes__))

    def test_runtime_inventory_failure_does_not_skip_authority_or_log_finish(self):
        admitted = {"authority": "exact"}
        authorized = mock.Mock()
        authorized.inventory.side_effect = [
            {},
            {},
            {},
            RuntimeError("final GPU inventory"),
        ]
        process = mock.Mock(pid=1234)
        identity = {
            "pid": 1234,
            "starttime_ticks": 77,
            "elf_sha256": contract.PINS["elf"],
            "argv": contract.server_argv(),
            "environment": {},
        }
        server_log = mock.Mock(stdout_fd=8)
        server_log.finish.return_value = {"excerpt": "bounded"}
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "capture"
            root.mkdir(mode=0o700)
            with (
                mock.patch.object(
                    runtime.authority, "Authority", return_value=authorized
                ),
                mock.patch.object(
                    runtime.parity, "preflight_authority", return_value=admitted
                ) as preflight,
                mock.patch.object(
                    runtime, "_BoundedServerLog", return_value=server_log
                ),
                mock.patch.object(runtime.subprocess, "Popen", return_value=process),
                mock.patch.object(runtime, "_starttime_ticks", return_value=77),
                mock.patch.object(runtime, "_wait_ready", return_value=identity),
                mock.patch.object(runtime, "_http_json", return_value={}),
                mock.patch.object(runtime, "_attest", return_value=identity),
                mock.patch.object(runtime, "_terminate", return_value=lifecycle()),
                mock.patch.object(runtime, "_assert_port_free"),
                self.assertRaisesRegex(RuntimeError, "final GPU inventory"),
            ):
                runtime.RuntimeAdapter()(
                    "m38",
                    "serial0",
                    contract.request_specs()["m38"],
                    "d" * 64,
                    root,
                )
        self.assertEqual(preflight.call_count, 3)
        server_log.finish.assert_called_once_with()

    def test_runtime_rejects_stale_authority_after_cleanup(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "capture"
            root.mkdir(mode=0o700)
            spec = contract.request_specs()["m38"]
            nonce = "d" * 64
            identity = {
                "pid": 1234,
                "starttime_ticks": 77,
                "elf_sha256": contract.PINS["elf"],
                "argv": contract.server_argv(),
                "environment": contract.arm_environment("serial0", nonce, root),
            }
            admitted = {"authority": "exact"}
            authorized = mock.Mock()
            process = mock.Mock(pid=1234)
            with (
                mock.patch.object(
                    runtime.authority, "Authority", return_value=authorized
                ),
                mock.patch.object(
                    runtime.parity,
                    "preflight_authority",
                    side_effect=[admitted, admitted, {"authority": "stale"}],
                ),
                mock.patch.object(runtime.subprocess, "Popen", return_value=process),
                mock.patch.object(runtime, "_starttime_ticks", return_value=77),
                mock.patch.object(runtime, "_wait_ready", return_value=identity),
                mock.patch.object(runtime, "_http_json", return_value={}),
                mock.patch.object(runtime, "_attest", return_value=identity),
                mock.patch.object(runtime, "_terminate", return_value=lifecycle()),
                mock.patch.object(runtime, "_assert_port_free"),
                self.assertRaisesRegex(RuntimeError, "authority changed"),
            ):
                runtime.RuntimeAdapter()("m38", "serial0", spec, nonce, root)

    def test_runtime_rechecks_empty_inventory_after_cleanup_error(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "capture"
            root.mkdir(mode=0o700)
            spec = contract.request_specs()["m38"]
            nonce = "e" * 64
            identity = {
                "pid": 1234,
                "starttime_ticks": 77,
                "elf_sha256": contract.PINS["elf"],
                "argv": contract.server_argv(),
                "environment": contract.arm_environment("serial0", nonce, root),
            }
            inventories: list[set[int]] = []
            admitted = {"build": "exact", "model": "exact"}
            authorized = mock.Mock()
            authorized.inventory.side_effect = lambda expected: (
                inventories.append(expected) or {}
            )
            process = mock.Mock(pid=1234)
            with (
                mock.patch.object(
                    runtime.authority, "Authority", return_value=authorized
                ),
                mock.patch.object(
                    runtime.parity, "preflight_authority", return_value=admitted
                ),
                mock.patch.object(runtime.subprocess, "Popen", return_value=process),
                mock.patch.object(runtime, "_starttime_ticks", return_value=77),
                mock.patch.object(runtime, "_wait_ready", return_value=identity),
                mock.patch.object(
                    runtime, "_http_json", side_effect=RuntimeError("HTTP")
                ),
                mock.patch.object(
                    runtime, "_terminate", side_effect=RuntimeError("cleanup")
                ),
                mock.patch.object(runtime, "_assert_port_free"),
            ):
                with self.assertRaisesRegex(RuntimeError, "HTTP"):
                    runtime.RuntimeAdapter()("m38", "serial0", spec, nonce, root)
            self.assertEqual(inventories[-1], set())

    def test_terminate_constructs_exact_graceful_and_forced_receipts(self):
        process = mock.Mock(pid=1234)
        process.wait.return_value = 0
        with (
            mock.patch.object(runtime, "_alive", return_value=True),
            mock.patch.object(runtime, "_require_signal_identity") as identity,
            mock.patch.object(runtime, "_drain_session", return_value=0),
            mock.patch.object(runtime, "_session_members", return_value=set()),
            mock.patch.object(runtime, "_listeners", return_value=[]),
            mock.patch.object(runtime.os, "killpg") as killpg,
        ):
            graceful = runtime._terminate(process, 77)
        self.assertEqual(graceful["actions"], ["sigterm"])
        self.assertEqual(graceful["returncode"], 0)
        self.assertTrue(graceful["clean_exit"])
        identity.assert_called_once_with(1234, 77)
        killpg.assert_called_once_with(1234, runtime.signal.SIGTERM)

        process.wait.return_value = -15
        with (
            mock.patch.object(runtime, "_alive", return_value=True),
            mock.patch.object(runtime, "_require_signal_identity") as identity,
            mock.patch.object(runtime, "_drain_session", return_value=0),
            mock.patch.object(runtime, "_session_members", return_value=set()),
            mock.patch.object(runtime, "_listeners", return_value=[]),
            mock.patch.object(runtime.os, "killpg") as killpg,
        ):
            terminated = runtime._terminate(process, 77)
        self.assertEqual(terminated["actions"], ["sigterm"])
        self.assertIsNone(terminated["timeout_after_sigterm_seconds"])
        self.assertEqual(terminated["returncode"], -15)
        self.assertFalse(terminated["clean_exit"])
        self.assertFalse(terminated["sigkill_identity_rechecked"])
        identity.assert_called_once_with(1234, 77)
        killpg.assert_called_once_with(1234, runtime.signal.SIGTERM)

        process.wait.side_effect = [
            runtime.subprocess.TimeoutExpired(cmd="spark", timeout=15),
            -9,
        ]
        with (
            mock.patch.object(runtime, "_alive", return_value=True),
            mock.patch.object(runtime, "_require_signal_identity") as identity,
            mock.patch.object(runtime, "_drain_session", return_value=0),
            mock.patch.object(runtime, "_session_members", return_value=set()),
            mock.patch.object(runtime, "_listeners", return_value=[]),
            mock.patch.object(runtime.os, "killpg") as killpg,
        ):
            forced = runtime._terminate(process, 77)
        self.assertEqual(forced["actions"], ["sigterm", "sigkill"])
        self.assertEqual(forced["timeout_after_sigterm_seconds"], 15)
        self.assertEqual(forced["returncode"], -9)
        self.assertFalse(forced["clean_exit"])
        self.assertEqual(identity.call_count, 2)
        self.assertEqual(
            killpg.call_args_list,
            [
                mock.call(1234, runtime.signal.SIGTERM),
                mock.call(1234, runtime.signal.SIGKILL),
            ],
        )

    def test_terminate_dead_leader_still_drains_descendants_and_listener(self):
        process = mock.Mock(pid=1234)
        process.poll.return_value = -6
        with (
            mock.patch.object(runtime, "_alive", return_value=False),
            mock.patch.object(runtime, "_drain_session", return_value=1) as drain,
            mock.patch.object(runtime, "_session_members", return_value=set()),
            mock.patch.object(runtime, "_listeners", return_value=[]) as listeners,
            self.assertRaisesRegex(RuntimeError, "before SIGTERM rc=-6"),
        ):
            runtime._terminate(process, 77)
        drain.assert_called_once_with(1234)
        self.assertGreaterEqual(listeners.call_count, 1)

    def test_terminate_poll_interrupt_still_drains_owned_session(self):
        process = mock.Mock(pid=1234)
        process.poll.side_effect = [KeyboardInterrupt, -9]
        with (
            mock.patch.object(runtime, "_drain_session", return_value=1) as drain,
            mock.patch.object(runtime, "_session_members", return_value=set()),
            mock.patch.object(runtime, "_listeners", return_value=[]),
            self.assertRaises(KeyboardInterrupt),
        ):
            runtime._terminate(process, 77)
        drain.assert_called_once_with(1234)

    def test_terminate_unattested_interrupts_still_drain_reap_and_census(self):
        cases = ("poll", "terminate", "wait")
        for point in cases:
            with self.subTest(point=point):
                process = mock.Mock(pid=1234)
                process.poll.return_value = None
                process.wait.return_value = 0
                primary = KeyboardInterrupt(point)
                if point == "poll":
                    process.poll.side_effect = primary
                elif point == "terminate":
                    process.terminate.side_effect = primary
                else:
                    process.wait.side_effect = [primary, 0]
                with (
                    mock.patch.object(
                        runtime, "_drain_session", return_value=1
                    ) as drain,
                    mock.patch.object(runtime, "_assert_port_free") as port,
                    self.assertRaises(KeyboardInterrupt) as caught,
                ):
                    runtime._terminate_unattested(process)
                self.assertIs(caught.exception, primary)
                drain.assert_called_once_with(1234)
                port.assert_called_once_with()
                self.assertGreaterEqual(process.wait.call_count, 1)

    def test_terminate_refuses_sigkill_after_identity_drift(self):
        process = mock.Mock(pid=1234)
        process.wait.side_effect = runtime.subprocess.TimeoutExpired(
            cmd="spark", timeout=15
        )
        with (
            mock.patch.object(runtime, "_alive", return_value=True),
            mock.patch.object(
                runtime,
                "_require_signal_identity",
                side_effect=[None, RuntimeError("identity drift")],
            ),
            mock.patch.object(runtime, "_drain_session", return_value=0),
            mock.patch.object(runtime, "_session_members", return_value=set()),
            mock.patch.object(runtime, "_listeners", return_value=[]),
            mock.patch.object(runtime.os, "killpg") as killpg,
            self.assertRaisesRegex(RuntimeError, "identity drift"),
        ):
            runtime._terminate(process, 77)
        killpg.assert_called_once_with(1234, runtime.signal.SIGTERM)

    def test_runtime_cannot_bypass_absent_internal_authority(self):
        with tempfile.TemporaryDirectory() as temp:
            absent = Path(temp) / "absent-reservation.json"
            with (
                mock.patch.object(root_authority, "RESERVATION", absent),
                mock.patch.object(runtime.subprocess, "Popen") as popen,
                self.assertRaises(FileNotFoundError),
            ):
                runtime.RuntimeAdapter()
        popen.assert_not_called()

    def test_authority_bundle_self_attestation(self):
        self.assertEqual(
            root_authority.attest_bundle()["sha256"],
            root_authority.HARNESS_BUNDLE_SHA256,
        )
        with (
            mock.patch.object(root_authority, "HARNESS_BUNDLE_SHA256", "0" * 64),
            self.assertRaisesRegex(RuntimeError, "bundle is unreleased"),
        ):
            root_authority.attest_bundle()

    def test_server_contract_keeps_qsa_off_and_m2013_fits(self):
        argv = contract.server_argv()
        self.assertNotIn("--qwen4-qsa", argv)
        self.assertEqual(argv[argv.index("--max-seq-len") + 1], "2048")
        spec = contract.request_specs()["m2013"]
        self.assertLessEqual(spec.prompt_tokens + spec.completion_tokens, 2048)
        self.assertEqual(
            contract.FRAME_LAYOUT[spec.prompt_tokens],
            ((0, 2_000, True), (2_000, 13, False)),
        )

    def test_readiness_loop_rechecks_root_authority_each_interval(self):
        process = mock.Mock(pid=1234)
        process.poll.return_value = None
        authorized = mock.Mock()
        with (
            mock.patch.object(runtime.time, "monotonic", side_effect=[0.0, 1.0, 300.0]),
            mock.patch.object(runtime.time, "sleep"),
            mock.patch.object(
                runtime, "_attest", side_effect=RuntimeError("not ready")
            ),
            self.assertRaisesRegex(RuntimeError, "did not become ready"),
        ):
            runtime._wait_ready(
                process, "serial0", "a" * 64, Path("/tmp/capture"), authorized
            )
        authorized.inventory_subset.assert_called_once_with({1234})

    def test_readiness_loop_does_not_swallow_foreign_gpu_client(self):
        process = mock.Mock(pid=1234)
        process.poll.return_value = None
        authorized = mock.Mock()
        authorized.inventory_subset.side_effect = RuntimeError("foreign GPU")
        with (
            mock.patch.object(runtime.time, "monotonic", side_effect=[0.0, 1.0]),
            self.assertRaisesRegex(RuntimeError, "foreign GPU"),
        ):
            runtime._wait_ready(
                process, "serial0", "a" * 64, Path("/tmp/capture"), authorized
            )

    def test_boot_identity_reads_procfs_to_eof_not_reported_size(self):
        digest = root_authority._boot_hash()
        self.assertRegex(digest, r"^[0-9a-f]{64}$")
        self.assertNotEqual(digest, hashlib.sha256(b"").hexdigest())
        with mock.patch.object(root_authority, "_stable_proc", return_value=b""):
            with self.assertRaisesRegex(RuntimeError, "boot ID wire"):
                root_authority._boot_hash()

    def test_session_drain_kills_different_process_group_descendant(self):
        with (
            mock.patch.object(
                runtime, "_session_members", side_effect=[{222}, {222}, set()]
            ),
            mock.patch.object(runtime, "_starttime_ticks", return_value=55),
            mock.patch.object(runtime, "_alive", return_value=True),
            mock.patch.object(runtime.os, "kill") as kill,
            mock.patch.object(runtime.time, "sleep"),
        ):
            runtime._drain_session(111)
        kill.assert_called_once_with(222, runtime.signal.SIGKILL)

    def test_early_start_failure_still_cleans_and_rechecks_inventory(self):
        inventories: list[set[int]] = []
        authorized = mock.Mock()
        authorized.inventory.side_effect = lambda expected: (
            inventories.append(expected) or {}
        )
        process = mock.Mock(pid=1234)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "capture"
            root.mkdir(mode=0o700)
            with (
                mock.patch.object(
                    runtime.authority, "Authority", return_value=authorized
                ),
                mock.patch.object(
                    runtime.parity,
                    "preflight_authority",
                    return_value={"authority": "exact"},
                ),
                mock.patch.object(runtime.subprocess, "Popen", return_value=process),
                mock.patch.object(
                    runtime, "_starttime_ticks", side_effect=RuntimeError("start")
                ),
                mock.patch.object(runtime, "_terminate_unattested") as terminate,
                mock.patch.object(runtime, "_assert_port_free"),
            ):
                with self.assertRaisesRegex(RuntimeError, "start"):
                    runtime.RuntimeAdapter()(
                        "m38",
                        "serial0",
                        contract.request_specs()["m38"],
                        "f" * 64,
                        root,
                    )
            terminate.assert_called_once_with(process)
            self.assertEqual(inventories, [set(), set()])


if __name__ == "__main__":
    unittest.main()
