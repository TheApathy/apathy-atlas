# SPDX-License-Identifier: AGPL-3.0-only
"""Strict reader for committed raw post-PLE capture frames."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any

import ple_prefill_parity_contract as contract

RECEIPT_KEYS = {
    "schema",
    "boundary",
    "performance_claim_allowed",
    "producer_stream_synchronized",
    "producer_stream",
    "pid",
    "nonce",
    "capture_root",
    "capture_root_dev",
    "capture_root_ino",
    "frame",
    "frame_dev",
    "frame_ino",
    "frame_commit_mode",
    "arm",
    "selector",
    "request_m",
    "request_tokens_encoding",
    "request_tokens_sha256",
    "ple_prior_m",
    "ple_prior_tokens_sha256",
    "ple_ordered_m",
    "ple_ordered_tokens_sha256",
    "chunk_start",
    "chunk_m",
    "chunk_tokens_encoding",
    "chunk_tokens_sha256",
    "slot_idx",
    "reset_state",
    "continuation",
    "artifacts",
}
ARTIFACT_KEYS = {"file", "dtype", "shape", "bytes", "sha256", "dev", "ino", "mode"}


def _reject_constant(value: str) -> Any:
    raise ValueError(f"invalid JSON constant {value}")


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key}")
        result[key] = value
    return result


def strict_json(raw: bytes) -> Any:
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=_reject_constant)


def _identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        stat.S_IMODE(metadata.st_mode),
        metadata.st_nlink,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def stable_bytes(
    path: Path,
    *,
    expected_sha256: str | None,
    expected_size: int | None,
    expected_mode: int,
    limit: int,
) -> tuple[bytes, dict[str, int | str]]:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise RuntimeError(
            f"capture file is not one regular unlinked authority: {path}"
        )
    if stat.S_IMODE(before.st_mode) != expected_mode:
        raise RuntimeError(f"capture file mode drift: {path}")
    if before.st_size > limit or (
        expected_size is not None and before.st_size != expected_size
    ):
        raise RuntimeError(f"capture file size drift: {path}")
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        if _identity(opened) != _identity(before):
            raise RuntimeError("capture file changed across no-follow open")
        chunks, remaining = [], before.st_size
        while remaining:
            chunk = os.read(fd, min(1 << 20, remaining))
            if not chunk:
                raise RuntimeError("capture file truncated during stable read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise RuntimeError("capture file grew during stable read")
        after_fd = os.fstat(fd)
    finally:
        os.close(fd)
    after_path = path.lstat()
    if _identity(before) != _identity(after_fd) or _identity(before) != _identity(
        after_path
    ):
        raise RuntimeError("capture file identity changed during stable read")
    raw = b"".join(chunks)
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise RuntimeError(f"capture file SHA256 drift: {path}")
    return raw, {
        "sha256": digest,
        "bytes": len(raw),
        "dev": before.st_dev,
        "ino": before.st_ino,
        "mode": stat.S_IMODE(before.st_mode),
        "nlink": before.st_nlink,
        "mtime_ns": before.st_mtime_ns,
    }


def _hex64(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise RuntimeError(f"{label} is not lowercase SHA256")
    return value


def _artifact(
    frame: Path,
    record: object,
    name: str,
    shape: list[int],
    byte_count: int,
) -> tuple[bytes, dict[str, Any]]:
    if not isinstance(record, dict) or set(record) != ARTIFACT_KEYS:
        raise RuntimeError(f"{name} artifact schema drift")
    exact = {
        "file": name,
        "dtype": "bf16le",
        "shape": shape,
        "bytes": byte_count,
        "mode": "0444",
    }
    if any(
        type(record.get(key)) is not type(value) or record.get(key) != value
        for key, value in exact.items()
    ):
        raise RuntimeError(f"{name} artifact contract drift")
    digest = _hex64(record["sha256"], f"{name} SHA256")
    if any(type(record[key]) is not int or record[key] <= 0 for key in ("dev", "ino")):
        raise RuntimeError(f"{name} artifact identity type drift")
    raw, identity = stable_bytes(
        frame / name,
        expected_sha256=digest,
        expected_size=byte_count,
        expected_mode=0o444,
        limit=byte_count,
    )
    if (identity["dev"], identity["ino"]) != (record["dev"], record["ino"]):
        raise RuntimeError(f"{name} artifact inode drift")
    return raw, record


def _frame_name(
    nonce: str, arm: str, request_m: int, start: int, count: int, reset: bool
) -> str:
    phase = "reset" if reset else "continuation"
    return f"frame-{nonce}-{arm}-m{request_m}-s{start}-n{count}-{phase}"


def load_frame(
    root: Path,
    arm: str,
    request_m: int,
    nonce: str,
    pid: int,
    geometry: tuple[int, int, bool],
) -> dict[str, Any]:
    start, count, reset = geometry
    if arm not in contract.ARM_SELECTORS or type(pid) is not int or pid <= 1:
        raise RuntimeError("capture frame arm/PID identity drift")
    name = _frame_name(nonce, arm, request_m, start, count, reset)
    frame = root / name
    frame_stat = frame.lstat()
    if (
        not stat.S_ISDIR(frame_stat.st_mode)
        or stat.S_IMODE(frame_stat.st_mode) != 0o500
    ):
        raise RuntimeError("capture frame is not committed mode0500 directory")
    receipt_raw, receipt_identity = stable_bytes(
        frame / "receipt.json",
        expected_sha256=None,
        expected_size=None,
        expected_mode=0o444,
        limit=1 << 20,
    )
    if not receipt_raw.endswith(b"\n") or receipt_raw.endswith(b"\n\n"):
        raise RuntimeError("capture receipt newline contract drift")
    receipt = strict_json(receipt_raw)
    if not isinstance(receipt, dict) or set(receipt) != RECEIPT_KEYS:
        raise RuntimeError("capture receipt schema drift")
    exact = {
        "schema": contract.SCHEMA,
        "boundary": "post_ple_pre_layer1",
        "performance_claim_allowed": False,
        "producer_stream_synchronized": True,
        "pid": pid,
        "nonce": nonce,
        "capture_root": str(root),
        "capture_root_dev": root.lstat().st_dev,
        "capture_root_ino": root.lstat().st_ino,
        "frame": name,
        "frame_dev": frame_stat.st_dev,
        "frame_ino": frame_stat.st_ino,
        "frame_commit_mode": "0500",
        "arm": arm,
        "selector": int(contract.ARM_SELECTORS[arm]),
        "request_m": request_m,
        "request_tokens_encoding": "u32le",
        "ple_prior_m": start,
        "ple_ordered_m": start + count,
        "chunk_start": start,
        "chunk_m": count,
        "chunk_tokens_encoding": "u32le",
        "slot_idx": 0,
        "reset_state": reset,
        "continuation": not reset,
    }
    if any(
        type(receipt.get(key)) is not type(value) or receipt.get(key) != value
        for key, value in exact.items()
    ):
        raise RuntimeError("capture receipt exact field drift")
    if type(receipt["producer_stream"]) is not int or receipt["producer_stream"] < 0:
        raise RuntimeError("capture producer stream identity drift")
    for key in (
        "request_tokens_sha256",
        "ple_prior_tokens_sha256",
        "ple_ordered_tokens_sha256",
        "chunk_tokens_sha256",
    ):
        _hex64(receipt[key], key)
    artifacts = receipt["artifacts"]
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "post_ple_hidden",
        "ple_live",
        "ple_checkpoint",
    }:
        raise RuntimeError("capture artifact census drift")
    hidden, _ = _artifact(
        frame,
        artifacts["post_ple_hidden"],
        "post_ple_hidden.bf16le",
        [count, 10_240],
        count * 10_240 * 2,
    )
    live, _ = _artifact(
        frame,
        artifacts["ple_live"],
        "ple_live.bf16le",
        [10_240, 9],
        184_320,
    )
    checkpoint, _ = _artifact(
        frame,
        artifacts["ple_checkpoint"],
        "ple_checkpoint.bf16le",
        [10_240, 9],
        184_320,
    )
    if _identity(frame.lstat()) != _identity(frame_stat):
        raise RuntimeError("capture frame identity changed during admission")
    return {
        "receipt": receipt,
        "receipt_identity": receipt_identity,
        "hidden": hidden,
        "live": live,
        "checkpoint": checkpoint,
    }


def load_arm(
    root: Path, arm: str, request_m: int, nonce: str, pid: int
) -> list[dict[str, Any]]:
    root_stat = root.lstat()
    if (
        not root.is_absolute()
        or root.resolve() != root
        or not stat.S_ISDIR(root_stat.st_mode)
    ):
        raise RuntimeError("capture root path/type is not canonical")
    if stat.S_IMODE(root_stat.st_mode) != 0o700:
        raise RuntimeError("capture root mode is not 0700")
    root_identity = _identity(root_stat)
    layout = contract.FRAME_LAYOUT[request_m]
    wanted = {_frame_name(nonce, arm, request_m, *geometry) for geometry in layout}
    if {entry.name for entry in root.iterdir()} != wanted:
        raise RuntimeError("capture frame census drift")
    frames = [
        load_frame(root, arm, request_m, nonce, pid, geometry) for geometry in layout
    ]
    if _identity(root.lstat()) != root_identity:
        raise RuntimeError("capture root identity changed during admission")
    return frames
