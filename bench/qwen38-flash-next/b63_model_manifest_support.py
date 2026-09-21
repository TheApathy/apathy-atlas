# SPDX-License-Identifier: AGPL-3.0-only
"""Strict JSON and stable model-file reads for the manifest producer."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path, PurePosixPath
from typing import Any


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RuntimeError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(raw: bytes) -> Any:
    return json.loads(
        raw,
        object_pairs_hook=_object,
        parse_constant=lambda value: (_ for _ in ()).throw(
            RuntimeError(f"non-finite JSON value: {value}")
        ),
    )


def relative_path(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise RuntimeError("invalid relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or str(path) != value:
        raise RuntimeError("invalid relative path")
    return value


def _summary(info: os.stat_result, relative: str, digest: str) -> dict[str, Any]:
    return {
        "relative": relative,
        "sha256": digest,
        "size": info.st_size,
        "mode": stat.S_IMODE(info.st_mode),
        "device": info.st_dev,
        "inode": info.st_ino,
        "mtime_ns": info.st_mtime_ns,
    }


def scan(
    root: Path, relative: str, *, capture: bool = False
) -> tuple[dict, bytes | None]:
    relative = relative_path(relative)
    path = root / relative
    if root.resolve(strict=True) != root or path.resolve(strict=True) != path:
        raise RuntimeError(f"non-canonical model path: {path}")
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise RuntimeError(f"model entry is not a nonempty regular file: {path}")
        if capture and before.st_size > 128 << 20:
            raise RuntimeError(f"captured model metadata exceeds 128 MiB: {path}")
        remaining = before.st_size
        while remaining:
            block = os.read(fd, min(8 << 20, remaining))
            if not block:
                raise RuntimeError(f"short model read: {path}")
            digest.update(block)
            if capture:
                chunks.append(block)
            remaining -= len(block)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    final = path.lstat()
    empty = "0" * 64
    if _summary(before, relative, empty) != _summary(after, relative, empty):
        raise RuntimeError(f"model entry changed during read: {path}")
    if _summary(before, relative, empty) != _summary(final, relative, empty):
        raise RuntimeError(f"model pathname changed during read: {path}")
    raw = b"".join(chunks) if capture else None
    return _summary(before, relative, digest.hexdigest()), raw


def expected_main_shards(count: int) -> list[str]:
    if count != 197:
        return [
            f"model-{index:05d}-of-{count:05d}.safetensors"
            for index in range(1, count + 1)
        ]
    layers = [
        f"layer-{layer:05d}-experts-{start:04d}-{start + 127:04d}.safetensors"
        for layer in range(48)
        for start in range(0, 512, 128)
    ]
    dense = [f"model-bf16-{index:05d}.safetensors" for index in (1, 10, 11, 12)]
    return sorted(layers + dense + ["model-plefp8-00009.safetensors"])
