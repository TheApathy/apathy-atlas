#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded stable file and local HTTP I/O for the 1M quality receipt."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import long_context_quality_contract as contract

MAX_JSON_BYTES = 1 << 20


def local_base_url(value: str) -> str:
    parsed = urllib.parse.urlsplit(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "::1", "localhost"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path not in {"", "/"}
        or parsed.port is None
    ):
        raise ValueError("base URL must be an explicit local HTTP origin with port")
    return value.rstrip("/")


def post_json(
    base_url: str, route: str, value: object, timeout: float
) -> tuple[dict, bytes]:
    body = contract.canonical_bytes(value)
    request = urllib.request.Request(
        f"{local_base_url(base_url)}{route}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read(MAX_JSON_BYTES + 1)
        if response.status != 200:
            raise RuntimeError(f"{route} returned HTTP {response.status}")
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError(f"{route} response exceeds {MAX_JSON_BYTES} bytes")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"{route} response must be a JSON object")
    return parsed, raw


def _identity(info: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def stable_file_sha256(
    path: Path, *, executable: bool = False, immutable: bool = True
) -> str:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ValueError(f"artifact is not a regular one-link file: {path}")
        if immutable and before.st_mode & 0o222:
            raise ValueError(f"artifact is writable: {path}")
        if executable and not before.st_mode & 0o111:
            raise ValueError(f"binary is not executable: {path}")
        digest = hashlib.sha256()
        while block := os.read(descriptor, 1 << 20):
            digest.update(block)
        after = os.fstat(descriptor)
        if _identity(before) != _identity(after):
            raise RuntimeError(f"artifact changed while hashing: {path}")
        final = os.lstat(path)
        if _identity(after) != _identity(final):
            raise RuntimeError(f"artifact path identity changed while hashing: {path}")
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def stable_read(path: Path, maximum: int, *, immutable: bool = True) -> bytes:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size > maximum
            or (immutable and before.st_mode & 0o222)
        ):
            raise ValueError(
                f"input is not a bounded immutable regular one-link file: {path}"
            )
        chunks = []
        total = 0
        while block := os.read(descriptor, min(1 << 16, maximum + 1 - total)):
            chunks.append(block)
            total += len(block)
            if total > maximum:
                raise ValueError(f"input exceeds {maximum} bytes: {path}")
        after = os.fstat(descriptor)
        final = os.lstat(path)
        if _identity(before) != _identity(after) or _identity(after) != _identity(
            final
        ):
            raise RuntimeError(f"input identity changed while reading: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def load_provenance(path: Path) -> dict[str, Any]:
    raw = stable_read(path, MAX_JSON_BYTES)
    value = contract.validate_provenance(json.loads(raw))
    if raw != contract.canonical_bytes(value):
        raise ValueError("provenance must be exact canonical JSON bytes")
    return value
