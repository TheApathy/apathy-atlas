"""Strict CPU-only I/O and scalar validation for the frozen oracle."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from pathlib import Path
from typing import Any

from frozen_triton_c143_constants import UINT64_MAX

HEX64 = re.compile(r"[0-9a-f]{64}\Z")


class GateError(ValueError):
    """Fail-closed contract or attestation error."""


def json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GateError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def decode_json(raw: bytes, label: str) -> Any:
    try:
        return json.loads(raw, object_pairs_hook=json_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GateError(f"{label}: invalid JSON") from exc


def require_exact_keys(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise GateError(f"{label}: expected object")
    actual = set(value)
    if actual != keys:
        raise GateError(
            f"{label}: key set mismatch; missing={sorted(keys - actual)} "
            f"extra={sorted(actual - keys)}"
        )
    return value


def require_bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise GateError(f"{label}: expected bool")
    return value


def require_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum or value > UINT64_MAX:
        raise GateError(f"{label}: invalid unsigned integer")
    return value


def require_number(value: Any, label: str, *, positive: bool = False) -> float:
    if type(value) not in (int, float):
        raise GateError(f"{label}: expected finite number")
    number = float(value)
    if not __import__("math").isfinite(number) or (positive and number <= 0.0):
        raise GateError(f"{label}: invalid finite number")
    return number


def require_sha256(value: Any, label: str) -> str:
    if type(value) is not str or HEX64.fullmatch(value) is None:
        raise GateError(f"{label}: expected lowercase SHA256")
    return value


def checked_add(a: int, b: int, label: str) -> int:
    require_int(a, f"{label}.lhs")
    require_int(b, f"{label}.rhs")
    result = a + b
    if result > UINT64_MAX:
        raise GateError(f"{label}: u64 addition overflow")
    return result


def checked_mul(*values: int, label: str) -> int:
    result = 1
    for index, value in enumerate(values):
        require_int(value, f"{label}[{index}]")
        if value and result > UINT64_MAX // value:
            raise GateError(f"{label}: u64 multiplication overflow")
        result *= value
    return result


def align_up(value: int, alignment: int, label: str) -> int:
    if alignment <= 0 or alignment & (alignment - 1):
        raise GateError(f"{label}: alignment must be a power of two")
    return checked_add(value, (-value) % alignment, label)


def canonical_regular(path: Path, label: str) -> tuple[Path, os.stat_result]:
    if not path.is_absolute():
        raise GateError(f"{label}: path must be absolute")
    try:
        before = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise GateError(f"{label}: unavailable regular file: {path}") from exc
    if resolved != path:
        raise GateError(f"{label}: path is not canonical: {path}")
    if not stat.S_ISREG(before.st_mode):
        raise GateError(f"{label}: not a regular file: {path}")
    return resolved, before


def stable_read(
    path: Path,
    label: str,
    *,
    expected_size: int | None = None,
    maximum_size: int = 4 << 20,
) -> tuple[bytes, str, os.stat_result]:
    resolved, path_before = canonical_regular(path, label)
    if path_before.st_size > maximum_size:
        raise GateError(f"{label}: file exceeds size cap")
    if expected_size is not None and path_before.st_size != expected_size:
        raise GateError(
            f"{label}: size mismatch; expected={expected_size} "
            f"actual={path_before.st_size}"
        )
    if not hasattr(os, "O_NOFOLLOW"):
        raise GateError(f"{label}: O_NOFOLLOW is required for stable reads")
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(resolved, flags)
    except OSError as exc:
        raise GateError(f"{label}: stable read failed") from exc
    try:
        descriptor_before = os.fstat(descriptor)
        if not stat.S_ISREG(descriptor_before.st_mode):
            raise GateError(f"{label}: opened object is not a regular file")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1 << 20, maximum_size + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum_size:
                raise GateError(f"{label}: file exceeds size cap during read")
        descriptor_after = os.fstat(descriptor)
        path_after = resolved.lstat()
    except OSError as exc:
        raise GateError(f"{label}: stable read failed") from exc
    finally:
        os.close(descriptor)
    identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    snapshots = (path_before, descriptor_before, descriptor_after, path_after)
    if any(
        getattr(snapshot, field) != getattr(path_before, field)
        for snapshot in snapshots[1:]
        for field in identity
    ):
        raise GateError(f"{label}: file changed during read")
    raw = b"".join(chunks)
    if len(raw) != descriptor_after.st_size:
        raise GateError(f"{label}: short or extended read")
    return raw, hashlib.sha256(raw).hexdigest(), descriptor_after


def hash_file(path: Path, label: str) -> str:
    return stable_read(path, label, maximum_size=2 << 20)[1]
