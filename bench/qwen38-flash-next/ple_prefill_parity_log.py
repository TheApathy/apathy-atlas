# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded, continuously drained child-output capture for PLE parity runs."""

from __future__ import annotations

import hashlib
import os
import signal
import stat
import threading
from pathlib import Path
from typing import Any

SERVER_LOG_LIMIT = 1 << 20
SERVER_LOG_MARKER = b"\n[... bounded middle omitted ...]\n"


def write_all(fd: int, raw: bytes) -> None:
    view = memoryview(raw)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise RuntimeError("bounded server log short write")
        view = view[written:]


class BoundedServerLog:
    """Continuously drain child output while retaining a bounded head and tail."""

    def __init__(self, root: Path) -> None:
        parent = root.parent
        parent_stat = parent.lstat()
        root_stat = root.lstat()
        if (
            not parent.is_absolute()
            or not stat.S_ISDIR(parent_stat.st_mode)
            or stat.S_IMODE(parent_stat.st_mode) != 0o700
            or not stat.S_ISDIR(root_stat.st_mode)
            or stat.S_IMODE(root_stat.st_mode) != 0o700
        ):
            raise RuntimeError("bounded server log root identity drift")
        self.path = parent / f".{root.name}.server.log"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
        self._file_fd = os.open(self.path, flags, 0o600)
        try:
            self._read_fd, self.stdout_fd = os.pipe2(os.O_CLOEXEC)
        except BaseException:
            os.close(self._file_fd)
            self.path.unlink(missing_ok=True)
            raise
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self._receipt: dict[str, Any] | None = None

    def start_before_spawn(self) -> None:
        thread = threading.Thread(
            target=self._drain, name="bounded-parity-server-log", daemon=True
        )
        self._thread = thread
        try:
            previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
            try:
                thread.start()
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
        except BaseException as start_error:
            cleanup_error: BaseException | None = None
            try:
                self.after_spawn()
            except BaseException as error:
                cleanup_error = error
            if thread.ident is None:
                for fd in (self._read_fd, self._file_fd):
                    try:
                        os.close(fd)
                    except OSError as error:
                        cleanup_error = cleanup_error or error
            else:
                thread.join(timeout=5)
                if thread.is_alive():
                    cleanup_error = cleanup_error or RuntimeError(
                        "interrupted bounded server log thread did not finish"
                    )
            self._thread = None
            self.path.unlink(missing_ok=True)
            if cleanup_error is not None:
                raise RuntimeError(
                    f"bounded server log start cleanup failed: {cleanup_error}"
                ) from start_error
            raise

    def after_spawn(self) -> None:
        if self.stdout_fd >= 0:
            writer, self.stdout_fd = self.stdout_fd, -1
            os.close(writer)

    def abort_before_runtime(self) -> None:
        self.after_spawn()
        try:
            self.finish()
        finally:
            self.path.unlink(missing_ok=True)

    def _drain(self) -> None:
        head_limit = SERVER_LOG_LIMIT // 2
        tail_limit = SERVER_LOG_LIMIT - head_limit - len(SERVER_LOG_MARKER)
        head, tail = bytearray(), bytearray()
        digest, total = hashlib.sha256(), 0
        try:
            while raw := os.read(self._read_fd, 64 << 10):
                digest.update(raw)
                total += len(raw)
                if len(head) < head_limit:
                    head.extend(raw[: head_limit - len(head)])
                tail.extend(raw)
                if len(tail) > tail_limit:
                    del tail[: len(tail) - tail_limit]
            stored = bytes(head)
            if total > len(head):
                stored += SERVER_LOG_MARKER + bytes(tail)
            write_all(self._file_fd, stored)
            os.fsync(self._file_fd)
            os.fchmod(self._file_fd, 0o444)
            self._receipt = {
                "path": str(self.path),
                "raw_bytes": total,
                "raw_sha256": digest.hexdigest(),
                "stored_bytes": len(stored),
                "stored_sha256": hashlib.sha256(stored).hexdigest(),
                "truncated": total != len(stored),
            }
        except BaseException as error:
            self._error = error
        finally:
            os.close(self._read_fd)
            os.close(self._file_fd)

    def finish(self) -> dict[str, Any]:
        if self._thread is None:
            raise RuntimeError("bounded server log was not started")
        self._thread.join(timeout=5)
        if self._thread.is_alive():
            raise RuntimeError("bounded server log drain did not finish")
        if self._error is not None:
            raise RuntimeError("bounded server log drain failed") from self._error
        receipt = self._receipt
        before = self.path.lstat()
        if (
            receipt is None
            or not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o444
            or before.st_nlink != 1
            or before.st_size != receipt["stored_bytes"]
            or before.st_size > SERVER_LOG_LIMIT
        ):
            raise RuntimeError("bounded server log seal drift")
        fd = os.open(self.path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            opened = os.fstat(fd)
            identity = (
                before.st_dev,
                before.st_ino,
                before.st_mode,
                before.st_nlink,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            if (
                identity
                != (
                    opened.st_dev,
                    opened.st_ino,
                    opened.st_mode,
                    opened.st_nlink,
                    opened.st_size,
                    opened.st_mtime_ns,
                    opened.st_ctime_ns,
                )
                or not stat.S_ISREG(opened.st_mode)
                or stat.S_IMODE(opened.st_mode) != 0o444
                or opened.st_nlink != 1
                or opened.st_size > SERVER_LOG_LIMIT
            ):
                raise RuntimeError("bounded server log open identity drift")
            raw = bytearray()
            while len(raw) < opened.st_size:
                block = os.read(fd, min(64 << 10, opened.st_size - len(raw)))
                if not block:
                    raise RuntimeError("bounded server log short read")
                raw.extend(block)
            after_fd = os.fstat(fd)
        finally:
            os.close(fd)
        after = self.path.lstat()
        if (
            hashlib.sha256(raw).hexdigest() != receipt["stored_sha256"]
            or identity
            != (
                after_fd.st_dev,
                after_fd.st_ino,
                after_fd.st_mode,
                after_fd.st_nlink,
                after_fd.st_size,
                after_fd.st_mtime_ns,
                after_fd.st_ctime_ns,
            )
            or identity
            != (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_nlink,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
        ):
            raise RuntimeError("bounded server log identity drift")
        return {
            **receipt,
            "excerpt": bytes(raw[-4096:]).decode("utf-8", errors="replace"),
        }
