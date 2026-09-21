# SPDX-License-Identifier: AGPL-3.0-only
"""Internal root-issued TEAM/GPU authority for raw PLE parity."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import ple_prefill_parity_capture as capture
import ple_prefill_parity_contract as contract

HERE = Path(__file__).resolve().parent
TEAM = Path("/home/flocka/atlas/qwen38/TEAM_INBOX.md")
RESERVATION = Path("/var/tmp/atlas-flash-b64-ple-parity-gpu-reservation.json")
BOOT_ID = Path("/proc/sys/kernel/random/boot_id")
NVIDIA_SMI = Path("/usr/bin/nvidia-smi")
SCHEMA = "atlas-flash-b64-ple-parity-root-authorization-v1"
LANE = "qwen38-flash-next-b64-post-ple-raw-parity"
MAX_WINDOW_SECONDS = 3_600
BUNDLE_NAMES = (
    "ple_prefill_parity.py",
    "ple_prefill_parity_authority.py",
    "ple_prefill_parity_capture.py",
    "ple_prefill_parity_contract.py",
    "ple_prefill_parity_log.py",
    "ple_prefill_parity_runtime.py",
    "ple_prefill_parity_validate.py",
    "test_ple_prefill_parity.py",
)
HARNESS_BUNDLE_SHA256 = (
    "4d5afc94cde37fd8b69bab9b1767174a74200c4cf2f746fb08479132edf7b283"
)
_BUNDLE_LINE = re.compile(rb'(?m)^HARNESS_BUNDLE_SHA256 = \(\n    "[0-9a-f]{64}"\n\)$')


def _stable(
    path: Path, limit: int, expected_mode: int | None = None
) -> tuple[bytes, dict[str, int | str]]:
    before = path.lstat()
    if (
        not path.is_absolute()
        or not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
    ):
        raise RuntimeError("authority file identity drift")
    mode = stat.S_IMODE(before.st_mode)
    if (expected_mode is not None and mode != expected_mode) or before.st_size > limit:
        raise RuntimeError("authority file mode/size drift")
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        if (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns) != (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ):
            raise RuntimeError("authority file changed across open")
        raw = b""
        while len(raw) < before.st_size:
            block = os.read(fd, min(1 << 20, before.st_size - len(raw)))
            if not block:
                raise RuntimeError("authority file short read")
            raw += block
        after_fd = os.fstat(fd)
    finally:
        os.close(fd)
    after = path.lstat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_mode")
    if any(
        getattr(before, key) != getattr(candidate, key)
        for candidate in (after_fd, after)
        for key in fields
    ):
        raise RuntimeError("authority file changed during read")
    return raw, {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "device": before.st_dev,
        "inode": before.st_ino,
        "size": before.st_size,
        "mtime_ns": before.st_mtime_ns,
        "mode": mode,
    }


def _stable_proc(path: Path, limit: int) -> bytes:
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError("proc authority is not a regular file")
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        opened = os.fstat(fd)
        chunks, total = [], 0
        while True:
            block = os.read(fd, min(256, limit + 1 - total))
            if not block:
                break
            chunks.append(block)
            total += len(block)
            if total > limit:
                raise RuntimeError("proc authority exceeds bounded reader")
        after_fd = os.fstat(fd)
    finally:
        os.close(fd)
    after = path.lstat()
    fields = ("st_dev", "st_ino", "st_mode", "st_mtime_ns")
    if any(
        getattr(before, key) != getattr(candidate, key)
        for candidate in (opened, after_fd, after)
        for key in fields
    ):
        raise RuntimeError("proc authority changed during bounded read")
    return b"".join(chunks)


def attest_bundle() -> dict[str, Any]:
    if tuple(sorted(BUNDLE_NAMES)) != BUNDLE_NAMES:
        raise RuntimeError("raw parity bundle order drift")
    files, normalized = {}, []
    for name in BUNDLE_NAMES:
        raw, evidence = _stable(HERE / name, 1 << 20)
        if name == Path(__file__).name:
            raw, count = _BUNDLE_LINE.subn(
                b'HARNESS_BUNDLE_SHA256 = (\n    "' + b"0" * 64 + b'"\n)', raw
            )
            if count != 1:
                raise RuntimeError("raw parity bundle self-pin drift")
        normalized.append(f"{hashlib.sha256(raw).hexdigest()}  {name}\n".encode())
        files[name] = evidence
    root = hashlib.sha256(b"".join(normalized)).hexdigest()
    if root != HARNESS_BUNDLE_SHA256:
        raise RuntimeError("raw parity harness bundle is unreleased")
    return {"sha256": root, "files": files}


def _boot_hash() -> str:
    raw = _stable_proc(BOOT_ID, 64)
    if re.fullmatch(rb"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\n", raw) is None:
        raise RuntimeError("boot ID wire format drift")
    return hashlib.sha256(raw).hexdigest()


def _claim_line(document: dict[str, Any]) -> str:
    fields = (
        f"resource=local-gb10-exclusive port={contract.PORT}",
        f"nonce_sha256={document['nonce_sha256']}",
        f"expires_at_unix={document['expires_at_unix']}",
        f"boot_id_sha256={document['boot_id_sha256']}",
        f"gpu_uuid={document['gpu_uuid']}",
        f"nvidia_smi_sha256={document['nvidia_smi_sha256']}",
        f"elf_sha256={contract.PINS['elf']}",
        f"harness_bundle_sha256={HARNESS_BUNDLE_SHA256}",
        f"model_manifest_sha256={contract.PINS['model_manifest']}",
        f"model_content_root_sha256={contract.PINS['model_content_root']}",
    )
    return (
        f"{document['issued_at_utc']} /root CLAIM-GPU+PORT+MODEL: {LANE} "
        + " ".join(fields)
    )


def load(*, now: int | None = None) -> dict[str, Any]:
    contract.require_released_pins()
    bundle = attest_bundle()
    raw, file_evidence = _stable(RESERVATION, 1 << 20, 0o444)
    document = capture.strict_json(raw)
    required = {
        "schema",
        "lane",
        "issuer",
        "resource",
        "port",
        "elf_sha256",
        "harness_bundle_sha256",
        "model_manifest_sha256",
        "model_content_root_sha256",
        "nonce_sha256",
        "boot_id_sha256",
        "gpu_uuid",
        "nvidia_smi_sha256",
        "issued_at_unix",
        "expires_at_unix",
        "issued_at_utc",
        "authorization",
    }
    if not isinstance(document, dict) or set(document) != required:
        raise RuntimeError("invalid raw parity root authorization schema")
    expected = {
        "schema": SCHEMA,
        "lane": LANE,
        "issuer": "/root",
        "resource": "local-gb10-exclusive",
        "port": contract.PORT,
        "elf_sha256": contract.PINS["elf"],
        "harness_bundle_sha256": HARNESS_BUNDLE_SHA256,
        "model_manifest_sha256": contract.PINS["model_manifest"],
        "model_content_root_sha256": contract.PINS["model_content_root"],
        "authorization": "b64-post-ple-raw-parity-only",
    }
    if any(document.get(key) != value for key, value in expected.items()):
        raise RuntimeError("raw parity root authorization identity mismatch")
    hex64 = re.compile(r"[0-9a-f]{64}")
    if any(
        type(document[key]) is not str or hex64.fullmatch(document[key]) is None
        for key in ("nonce_sha256", "boot_id_sha256", "nvidia_smi_sha256")
    ):
        raise RuntimeError("raw parity root authorization hash drift")
    if (
        type(document["gpu_uuid"]) is not str
        or re.fullmatch(r"GPU-[0-9A-Fa-f-]{36}", document["gpu_uuid"]) is None
    ):
        raise RuntimeError("raw parity authorized GPU UUID drift")
    issued, expires = document["issued_at_unix"], document["expires_at_unix"]
    current = int(time.time()) if now is None else now
    expected_utc = datetime.fromtimestamp(issued, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    if (
        type(issued) is not int
        or type(expires) is not int
        or not 0 < expires - issued <= MAX_WINDOW_SECONDS
        or not issued <= current <= expires
        or document["issued_at_utc"] != expected_utc
        or document["boot_id_sha256"] != _boot_hash()
    ):
        raise RuntimeError("raw parity root authorization time/boot drift")
    team_raw, _ = _stable(TEAM, 64 << 20)
    lines = team_raw.decode("utf-8", errors="strict").splitlines()
    claim = _claim_line(document)
    if lines.count(claim) != 1:
        raise RuntimeError("raw parity TEAM claim is absent or duplicated")
    claim_index = lines.index(claim)
    release = (
        f"/root RELEASE-GPU+PORT+MODEL: {LANE} nonce_sha256={document['nonce_sha256']}"
    )
    if any(release in line for line in lines[claim_index + 1 :]) or any(
        " CLAIM-GPU" in line for line in lines[claim_index + 1 :]
    ):
        raise RuntimeError("raw parity TEAM claim was released or superseded")
    return {"document": document, "file": file_evidence, "bundle": bundle}


def _query(arguments: list[str], expected_sha256: str) -> list[str]:
    raw, evidence = _stable(NVIDIA_SMI, 32 << 20)
    if evidence["sha256"] != expected_sha256 or not evidence["mode"] & 0o111:
        raise RuntimeError("authorized nvidia-smi identity drift")
    completed = subprocess.run(
        [str(NVIDIA_SMI), *arguments],
        capture_output=True,
        check=False,
        env={"LANG": "C.UTF-8", "PATH": "/usr/bin:/bin"},
        text=True,
        timeout=15,
    )
    if completed.returncode != 0 or completed.stderr.strip():
        raise RuntimeError("authorized nvidia-smi query failed")
    if _stable(NVIDIA_SMI, 32 << 20)[1] != evidence:
        raise RuntimeError("nvidia-smi changed across inventory query")
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


class Authority:
    def __init__(self) -> None:
        self.initial = load()

    def recheck(self) -> dict[str, Any]:
        current = load()
        if current != self.initial:
            raise RuntimeError("raw parity root authority changed")
        return current

    def _inventory(self, admitted_pids: set[int], *, exact: bool) -> dict[str, Any]:
        document = self.recheck()["document"]
        tool = document["nvidia_smi_sha256"]
        gpu = _query(["--query-gpu=uuid,name", "--format=csv,noheader"], tool)
        if len(gpu) != 1 or "," not in gpu[0]:
            raise RuntimeError("raw parity single-GPU census drift")
        gpu_uuid, gpu_name = (part.strip() for part in gpu[0].split(",", 1))
        if gpu_uuid != document["gpu_uuid"] or "GB10" not in gpu_name:
            raise RuntimeError("raw parity authorized GB10 drift")
        apps = _query(
            ["--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"],
            tool,
        )
        rows = []
        for line in apps:
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 2 or not parts[0].isdigit() or parts[1] != gpu_uuid:
                raise RuntimeError("malformed or foreign GPU compute client")
            rows.append({"pid": int(parts[0]), "gpu_uuid": parts[1]})
        observed = {row["pid"] for row in rows}
        if len(rows) != len(observed) or not observed <= admitted_pids:
            raise RuntimeError("raw parity GPU compute-client exclusivity drift")
        if exact and observed != admitted_pids:
            raise RuntimeError("raw parity expected GPU compute client is absent")
        return {"gpu_uuid": gpu_uuid, "gpu_name": gpu_name, "compute_clients": rows}

    def inventory(self, expected_pids: set[int]) -> dict[str, Any]:
        return self._inventory(expected_pids, exact=True)

    def inventory_subset(self, admitted_pids: set[int]) -> dict[str, Any]:
        return self._inventory(admitted_pids, exact=False)
