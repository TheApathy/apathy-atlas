# SPDX-License-Identifier: AGPL-3.0-only
"""Descriptor-backed verified bytes and sealed shared-library loading."""

from __future__ import annotations

import fcntl
import hashlib
import os
import stat
from pathlib import Path
from typing import Callable, TypeVar

from frozen_triton_c143_io import GateError, require_sha256, stable_read

T = TypeVar("T")
IDENTITY = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
REQUIRED_SEALS = (
    fcntl.F_SEAL_WRITE | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL
)


def load_verified_file(
    path: Path,
    label: str,
    expected_sha256: str,
    expected_size: int,
    loader: Callable[[bytes], T],
) -> T:
    require_sha256(expected_sha256, f"{label}.sha256")
    raw, before_sha, before = stable_read(
        path, label, expected_size=expected_size, maximum_size=1 << 20
    )
    if before_sha != expected_sha256:
        raise GateError(f"{label}: verified-byte identity drift")
    result = loader(raw)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise GateError(f"{label}: held bytes changed during load")
    _, after_sha, after = stable_read(
        path, f"{label} after load", expected_size=expected_size, maximum_size=1 << 20
    )
    if after_sha != expected_sha256 or any(
        getattr(before, field) != getattr(after, field) for field in IDENTITY
    ):
        raise GateError(f"{label}: source changed across verified load")
    return result


def _read_fd(fd: int, size: int) -> bytes:
    chunks = []
    offset = 0
    while offset < size:
        chunk = os.pread(fd, min(1 << 20, size - offset), offset)
        if not chunk:
            raise GateError("sealed memfd short read")
        chunks.append(chunk)
        offset += len(chunk)
    return b"".join(chunks)


class SealedMemfd:
    def __init__(self, fd: int, name: str, sha256: str, size: int) -> None:
        self.fd = fd
        self.name = name
        self.sha256 = sha256
        self.size = size

    @classmethod
    def from_file(
        cls, path: Path, expected_sha256: str, expected_size: int, name: str
    ) -> "SealedMemfd":
        def seal(raw: bytes) -> SealedMemfd:
            if not hasattr(os, "memfd_create") or not hasattr(os, "MFD_ALLOW_SEALING"):
                raise GateError("sealed memfd support is required")
            fd = os.memfd_create(name, os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
            try:
                view = memoryview(raw)
                while view:
                    written = os.write(fd, view)
                    if written <= 0:
                        raise OSError("short memfd write")
                    view = view[written:]
                os.fchmod(fd, 0o500)
                fcntl.fcntl(fd, fcntl.F_ADD_SEALS, REQUIRED_SEALS)
                result = cls(fd, name, expected_sha256, expected_size)
                result.verify()
                return result
            except Exception:
                os.close(fd)
                raise

        return load_verified_file(
            path, "Atlas bridge SO", expected_sha256, expected_size, seal
        )

    @property
    def path(self) -> Path:
        return Path(f"/proc/self/fd/{self.fd}")

    def verify(self) -> str:
        current = os.fstat(self.fd)
        seals = fcntl.fcntl(self.fd, fcntl.F_GET_SEALS)
        digest = hashlib.sha256(_read_fd(self.fd, self.size)).hexdigest()
        if (
            not stat.S_ISREG(current.st_mode)
            or stat.S_IMODE(current.st_mode) != 0o500
            or current.st_size != self.size
            or seals != REQUIRED_SEALS
            or digest != self.sha256
        ):
            raise GateError("sealed Atlas bridge identity drift")
        return digest

    def verify_executable_mapping(self) -> None:
        self.verify()
        current = os.fstat(self.fd)
        for line in Path("/proc/self/maps").read_text(encoding="ascii").splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) < 6 or "x" not in fields[1] or self.name not in fields[5]:
                continue
            major, minor = (int(part, 16) for part in fields[3].split(":"))
            if (
                int(fields[4]) == current.st_ino
                and major == os.major(current.st_dev)
                and minor == os.minor(current.st_dev)
            ):
                return
        raise GateError("loaded Atlas bridge is not the sealed memfd mapping")

    def close(self) -> None:
        os.close(self.fd)
