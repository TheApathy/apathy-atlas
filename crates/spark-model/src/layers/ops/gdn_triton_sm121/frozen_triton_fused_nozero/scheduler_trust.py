# SPDX-License-Identifier: AGPL-3.0-only
"""Held root-owned file primitive for scheduler authority."""

from __future__ import annotations

import hashlib
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from frozen_triton_c143_io import GateError


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_gid,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_fd(descriptor: int, maximum_size: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = os.read(descriptor, min(1 << 16, maximum_size + 1 - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if total > maximum_size:
            raise GateError("root authority file exceeds size cap")
    return b"".join(chunks)


def _check_root_parents(path: Path, label: str) -> None:
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise GateError(f"{label}: path is not canonical absolute")
    current = Path("/")
    for part in path.parent.parts[1:]:
        current /= part
        value = current.lstat()
        if (
            not stat.S_ISDIR(value.st_mode)
            or value.st_uid != 0
            or value.st_mode & 0o022
        ):
            raise GateError(f"{label}: parent is not root-owned and non-writable")


@dataclass
class HeldRootFile:
    path: Path
    label: str
    descriptor: int
    initial: tuple[int, ...]
    raw: bytes
    sha256: str

    @classmethod
    def open(cls, path: Path, label: str, modes: set[int]) -> HeldRootFile:
        _check_root_parents(path, label)
        before = path.lstat()
        mode = stat.S_IMODE(before.st_mode)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != 0
            or before.st_nlink != 1
            or mode not in modes
        ):
            raise GateError(f"{label}: not a root-owned immutable single-link file")
        if not hasattr(os, "O_NOFOLLOW"):
            raise GateError(f"{label}: O_NOFOLLOW is required")
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            opened = os.fstat(descriptor)
            raw = _read_fd(descriptor, 64 << 10)
            after = os.fstat(descriptor)
            if _identity(before) != _identity(opened) or _identity(before) != _identity(
                after
            ):
                raise GateError(f"{label}: identity changed while opening")
            return cls(
                path,
                label,
                descriptor,
                _identity(before),
                raw,
                hashlib.sha256(raw).hexdigest(),
            )
        except Exception:
            os.close(descriptor)
            raise

    def recheck(self) -> None:
        path_now = self.path.lstat()
        descriptor_now = os.fstat(self.descriptor)
        raw_now = _read_fd(self.descriptor, 64 << 10)
        if (
            _identity(path_now) != self.initial
            or _identity(descriptor_now) != self.initial
            or hashlib.sha256(raw_now).hexdigest() != self.sha256
        ):
            raise GateError(f"{self.label}: identity or content changed")

    def close(self) -> None:
        os.close(self.descriptor)
