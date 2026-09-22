# SPDX-License-Identifier: AGPL-3.0-only
"""Stable immutable-file, sealed-build, and reservation admission."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any

import b63_ple_prefill_ab_contract as contract

HEX64 = re.compile(r"[0-9a-f]{64}")


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number: {value}")


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json(raw: bytes) -> Any:
    return json.loads(
        raw,
        object_pairs_hook=_object,
        parse_constant=_reject_constant,
        parse_int=int,
        parse_float=float,
    )


def _summary(info: os.stat_result, digest: str) -> dict[str, Any]:
    return {
        "device": info.st_dev,
        "inode": info.st_ino,
        "size": info.st_size,
        "mode": stat.S_IMODE(info.st_mode),
        "mtime_ns": info.st_mtime_ns,
        "sha256": digest,
    }


def stable_bytes(
    path: Path,
    *,
    expected_sha256: str | None = None,
    max_bytes: int | None = None,
    readonly: bool = False,
    exact_mode: int | None = None,
) -> tuple[bytes, dict[str, Any]]:
    if not path.is_absolute() or not HEX64.fullmatch(expected_sha256 or "0" * 64):
        raise RuntimeError("path must be absolute and expected hash valid")
    if path.resolve(strict=True) != path:
        raise RuntimeError(f"path is not canonical: {path}")
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"not a regular file: {path}")
        if readonly and stat.S_IMODE(before.st_mode) & 0o222:
            raise RuntimeError(f"file is writable: {path}")
        if exact_mode is not None and stat.S_IMODE(before.st_mode) != exact_mode:
            raise RuntimeError(f"file mode drift: {path}")
        if max_bytes is not None and before.st_size > max_bytes:
            raise RuntimeError(f"file exceeds bounded read: {path}")
        chunks = []
        remaining = before.st_size
        while remaining:
            block = os.read(fd, min(8 << 20, remaining))
            if not block:
                raise RuntimeError(f"short read: {path}")
            chunks.append(block)
            remaining -= len(block)
        raw = b"".join(chunks)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    final_path = path.lstat()
    if _summary(before, "") != _summary(final_path, ""):
        raise RuntimeError(f"pathname replacement during read: {path}")
    if _summary(before, "") != _summary(after, ""):
        raise RuntimeError(f"file drift during read: {path}")
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise RuntimeError(f"file hash drift: {path}")
    return raw, _summary(before, digest)


def load_sealed_json(path: Path, expected_sha256: str) -> tuple[Any, dict[str, Any]]:
    raw, evidence = stable_bytes(
        path,
        expected_sha256=expected_sha256,
        max_bytes=128 << 20,
        readonly=True,
        exact_mode=0o444,
    )
    return strict_json(raw), evidence


def attest_build() -> dict[str, Any]:
    root = contract.BINARY.parent.parent
    binary, binary_evidence = stable_bytes(
        contract.BINARY, expected_sha256=contract.BINARY_SHA256, readonly=True
    )
    if (
        binary_evidence["size"] != contract.BINARY_SIZE
        or binary_evidence["mode"] != 0o555
        or binary.count(bytes.fromhex(contract.BUILD_ID)) != 1
    ):
        raise RuntimeError("sealed b63 ELF identity drift")
    files = {}
    for name, expected in contract.BUILD_FILES.items():
        raw, evidence = stable_bytes(
            root / name, expected_sha256=expected, max_bytes=64 << 20, readonly=True
        )
        if evidence["mode"] != 0o444:
            raise RuntimeError(f"sealed b63 sidecar mode drift: {name}")
        if name in contract.MANIFEST_LINES:
            lines = len(raw.splitlines())
            if lines != contract.MANIFEST_LINES[name]:
                raise RuntimeError(f"sealed b63 manifest census drift: {name}")
            evidence["lines"] = lines
        files[name] = evidence
    return {
        "binary": binary_evidence,
        "build_id": contract.BUILD_ID,
        "files": files,
    }


def write_json_line(stream: Any, value: object) -> None:
    stream.write(contract.canonical_bytes(value) + b"\n")
    stream.flush()
    os.fsync(stream.fileno())


def freeze_file(path: Path) -> dict[str, Any]:
    path.chmod(0o444)
    info = path.lstat()
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size": info.st_size,
        "mode": stat.S_IMODE(info.st_mode),
    }


def publish_json(path: Path, document: dict[str, Any]) -> None:
    with path.open("xb") as stream:
        write_json_line(stream, document)
    path.chmod(0o444)
