#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Strict retained-input and atomic-output helpers for provisional capture."""

from __future__ import annotations

import json
import os
import select
import stat
import tempfile
from pathlib import Path
from typing import Callable

import prefill_decode_repro as harness


def open_retained_json(path: Path, expected_sha256: str, label: str):
    if not isinstance(expected_sha256, str) or not harness._is_hex(expected_sha256, 64):
        raise ValueError(f"expected {label} hash must be a lowercase SHA-256")
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        raw, opened = harness._semantic_oracle_snapshot(path, descriptor)
        actual = harness.sha256(raw)
        if actual != expected_sha256:
            raise ValueError(
                f"{label} SHA-256 mismatch: expected {expected_sha256}, got {actual}"
            )
        value = json.loads(raw, object_pairs_hook=harness._reject_duplicate_json_keys)
        frozen = tuple(
            getattr(opened, key)
            for key in (
                "st_dev",
                "st_ino",
                "st_mode",
                "st_nlink",
                "st_uid",
                "st_size",
                "st_mtime_ns",
                "st_ctime_ns",
            )
        )
        return value, actual, descriptor, frozen
    except Exception:
        os.close(descriptor)
        raise


def assert_retained_unchanged(
    path: Path, descriptor: int, digest: str, frozen: tuple[int, ...]
) -> None:
    harness.assert_semantic_oracle_unchanged(path, descriptor, digest, frozen)


def atomic_exclusive_write(path: Path, raw: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
            os.fchmod(output.fileno(), 0o444)
        prepared = os.stat(temporary, follow_symlinks=False)
        os.link(temporary, path, follow_symlinks=False)
        os.unlink(temporary)
        sealed = os.stat(path, follow_symlinks=False)
        if (
            not stat.S_ISREG(sealed.st_mode)
            or (sealed.st_dev, sealed.st_ino) != (prepared.st_dev, prepared.st_ino)
            or sealed.st_uid != os.getuid()
            or sealed.st_nlink != 1
            or stat.S_IMODE(sealed.st_mode) != 0o444
        ):
            os.unlink(path)
            raise ValueError(
                "candidate output did not seal as an immutable single-link file"
            )
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _discover_model(endpoint: str) -> str:
    response = harness.get_json(endpoint.rsplit("/completions", 1)[0] + "/models")
    data = response.get("data") if isinstance(response, dict) else None
    if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict):
        raise ValueError("model discovery must return exactly one model")
    model = data[0].get("id")
    if (
        not isinstance(model, str)
        or not model
        or any(char in model for char in "\x00\r\n")
    ):
        raise ValueError("model discovery returned a malformed identifier")
    return model


def _is_speculative_route_marker(marker: str) -> bool:
    normalized = marker.upper()
    return "DFLASH" in normalized or "SPEC" in normalized


def run_capture(
    args,
    validate_abi: Callable,
    capture_prefill: Callable,
    capture_decode: Callable,
    build_candidate: Callable,
    capture_identity_provider: Callable,
    capture_identity: dict,
) -> tuple[Path, str]:
    harness.endpoint_port(args.endpoint)
    provenance_value, provenance_sha, provenance_fd, provenance_stat = (
        open_retained_json(
            args.provenance_manifest,
            args.expected_provenance_sha256,
            "provenance manifest",
        )
    )
    abi_fd = route_fd = executable_fd = pid_fd = None
    try:
        abi_value, abi_sha, abi_fd, abi_stat = open_retained_json(
            args.abi_attestation,
            args.expected_abi_attestation_sha256,
            "ABI attestation",
        )
        provenance = harness.validate_provenance(provenance_value, "no-spec")
        validate_abi(abi_value, provenance)
        if provenance["same_mode_environment_delta"] or any(
            _is_speculative_route_marker(marker)
            for marker in provenance["required_route_markers"]
        ):
            raise ValueError(
                "capture requires a non-speculative, non-comparison route manifest"
            )
        route_fd = harness.open_route_log(args.route_log)
        binding = harness.bind_live_server(
            args.endpoint, args.route_log, route_fd, provenance
        )
        route_prefix, _ = harness._route_log_snapshot(args.route_log, route_fd)
        if any(
            harness.is_route_engagement_line(line) for line in route_prefix.splitlines()
        ):
            raise ValueError(
                "fresh capture route log already contains engagement evidence"
            )
        executable_fd = os.open(
            Path("/proc") / str(binding["listener_pid"]) / "exe",
            os.O_RDONLY | os.O_CLOEXEC,
        )
        executable_stat = os.fstat(executable_fd)
        if (
            executable_stat.st_dev,
            executable_stat.st_ino,
            harness._sha256_fd(executable_fd),
        ) != (
            binding["executable_device"],
            binding["executable_inode"],
            provenance["binary_sha256"],
        ):
            raise ValueError("live executable does not match retained provenance")
        pid_fd = os.pidfd_open(binding["listener_pid"])

        def check(stage: str):
            nonlocal route_prefix
            poller = select.poll()
            poller.register(pid_fd, select.POLLIN)
            if poller.poll(0):
                raise ValueError(f"listener exited during {stage}")
            current = harness.bind_live_server(
                args.endpoint,
                args.route_log,
                route_fd,
                provenance,
                expected_binding=binding,
                retained_executable_fd=executable_fd,
            )
            if current != binding:
                raise ValueError(f"listener identity changed during {stage}")
            log, _ = harness._route_log_snapshot(args.route_log, route_fd)
            if not log.startswith(route_prefix):
                raise ValueError(f"route log was rewritten during {stage}")
            route_prefix = log
            return current

        check("model discovery start")
        model = _discover_model(args.endpoint)
        check("model discovery end")
        prefill = capture_prefill(args.endpoint, model)
        check("prefill capture")
        decode = capture_decode(
            args.endpoint.replace("/completions", "/chat/completions"), model
        )
        final_binding = check("decode capture")
        route_evidence = harness.capture_route_evidence(
            args.route_log,
            provenance["required_route_markers"],
            final_binding,
            route_fd,
        )
        check("evidence seal")
        final_log, _ = harness._route_log_snapshot(args.route_log, route_fd)
        if harness.sha256(final_log) != route_evidence["route_log_sha256"]:
            raise ValueError("route log changed after evidence capture")
        if harness._sha256_fd(executable_fd) != provenance["binary_sha256"]:
            raise ValueError("server executable changed during capture")
        assert_retained_unchanged(
            args.provenance_manifest, provenance_fd, provenance_sha, provenance_stat
        )
        assert_retained_unchanged(args.abi_attestation, abi_fd, abi_sha, abi_stat)
        if capture_identity_provider() != capture_identity:
            raise ValueError("capture components changed during capture")
        candidate = build_candidate(
            model,
            provenance,
            provenance_sha,
            abi_value,
            abi_sha,
            route_evidence,
            prefill,
            decode,
            capture_identity,
        )
        raw = harness.canonical_bytes(candidate) + b"\n"
        atomic_exclusive_write(args.output, raw)
        return args.output, harness.sha256(raw)
    finally:
        for descriptor in (pid_fd, executable_fd, route_fd, abi_fd, provenance_fd):
            if descriptor is not None:
                os.close(descriptor)
