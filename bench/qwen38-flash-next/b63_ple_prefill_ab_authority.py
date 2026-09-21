# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import hashlib
import os
import re
import stat
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import b63_ple_prefill_ab_contract as contract
import b63_ple_prefill_ab_identity as identity

HERE = Path(__file__).resolve().parent
TEAM_INBOX = Path("/home/flocka/atlas/qwen38/TEAM_INBOX.md")
RESERVATION = Path("/var/tmp/atlas-b63-ple-prefill-gpu-reservation.json")
BOOT_ID = Path("/proc/sys/kernel/random/boot_id")
SCHEMA = "atlas-b63-ple-prefill-root-gpu-authorization-v2"
LANE = "qwen38-flash-next-b63-whole-prompt-ple-m2013-ab-runtime"
MAX_WINDOW_SECONDS = 3_600
HARNESS_NAMES = (
    "b63_ple_prefill_ab.py",
    "b63_ple_prefill_ab_authority.py",
    "b63_ple_prefill_ab_contract.py",
    "b63_ple_prefill_ab_http.py",
    "b63_ple_prefill_ab_identity.py",
    "b63_ple_prefill_ab_inventory.py",
    "b63_ple_prefill_ab_model.py",
    "b63_ple_prefill_ab_process.py",
    "b63_ple_prefill_ab_validate.py",
    "test_b63_ple_prefill_ab.py",
    "test_b63_ple_prefill_ab_authority.py",
    "test_b63_ple_prefill_ab_hostile.py",
    "test_b63_ple_prefill_ab_inventory.py",
)
HARNESS_BUNDLE_SHA256 = (
    "9dc80eb422334d9730e5d9b96bdf9b56993d2249f0504c9d051cbfa9188dd8b9"
)
_BUNDLE_LINE = re.compile(rb'(?m)^HARNESS_BUNDLE_SHA256 = \(\n    "[0-9a-f]{64}"\n\)$')


def _stable_current(
    path: Path, max_bytes: int, exact_mode: int | None
) -> tuple[bytes, dict]:
    if not path.is_absolute() or path.resolve(strict=True) != path:
        raise RuntimeError(f"noncanonical authority path: {path}")
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
            raise RuntimeError(f"invalid authority file: {path}")
        mode = stat.S_IMODE(before.st_mode)
        if exact_mode is not None and mode != exact_mode:
            raise RuntimeError(f"authority mode drift: {path}")
        chunks, remaining = [], before.st_size
        while remaining:
            block = os.read(fd, min(8 << 20, remaining))
            if not block:
                raise RuntimeError(f"authority short read: {path}")
            chunks.append(block)
            remaining -= len(block)
        raw = b"".join(chunks)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    current = path.lstat()
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_mode")
    if any(
        getattr(before, key) != getattr(candidate, key)
        for candidate in (after, current)
        for key in fields
    ):
        raise RuntimeError(f"authority file drift: {path}")
    return raw, {
        "device": before.st_dev,
        "inode": before.st_ino,
        "size": before.st_size,
        "mode": mode,
        "mtime_ns": before.st_mtime_ns,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def attest_harness(paths: list[Path]) -> dict[str, Any]:
    if tuple(sorted(path.name for path in paths)) != HARNESS_NAMES:
        raise RuntimeError("harness source name census drift")
    files: dict[str, dict[str, Any]] = {}
    normalized = []
    for path in sorted(paths):
        raw, evidence = _stable_current(path, 1 << 20, None)
        normalized_raw = raw
        if path.name == Path(__file__).name:
            normalized_raw, count = _BUNDLE_LINE.subn(
                b'HARNESS_BUNDLE_SHA256 = (\n    "' + b"0" * 64 + b'"\n)', raw
            )
            if count != 1:
                raise RuntimeError("harness bundle authority line drift")
        normalized.append(
            f"{hashlib.sha256(normalized_raw).hexdigest()}  {path.name}\n".encode()
        )
        files[path.name] = evidence
    root = hashlib.sha256(b"".join(normalized)).hexdigest()
    if root != HARNESS_BUNDLE_SHA256:
        raise RuntimeError("released harness bundle identity drift")
    return {"bundle_sha256": root, "files": files}


def _claim_line(document: dict[str, Any]) -> str:
    fields = (
        f"resource=local-gb10-exclusive port={contract.PORT}",
        f"nonce_sha256={document['nonce_sha256']}",
        f"expires_at_unix={document['expires_at_unix']}",
        f"boot_id_sha256={document['boot_id_sha256']}",
        f"gpu_uuid={document['gpu_uuid']}",
        f"nvidia_smi_sha256={document['nvidia_smi_sha256']}",
        f"binary_sha256={contract.BINARY_SHA256}",
        f"harness_bundle_sha256={HARNESS_BUNDLE_SHA256}",
        f"raw_parity_receipt_sha256={contract.RAW_PARITY_RECEIPT_SHA256}",
        f"model_manifest_sha256={contract.MODEL_MANIFEST_SHA256}",
        f"model_content_root_sha256={contract.MODEL_CONTENT_ROOT_SHA256}",
    )
    return f"{document['issued_at_utc']} /root CLAIM-GPU: {LANE} " + " ".join(fields)


def _boot_hash() -> str:
    return hashlib.sha256(_stable_current(BOOT_ID, 256, None)[0]).hexdigest()


def load_reservation(*, now: int | None = None) -> dict[str, Any]:
    for value in (
        contract.MODEL_MANIFEST_SHA256,
        contract.MODEL_CONTENT_ROOT_SHA256,
        contract.RAW_PARITY_RECEIPT_SHA256,
    ):
        if not isinstance(value, str) or not identity.HEX64.fullmatch(value):
            raise RuntimeError("authoritative model root is not pinned")
    raw, evidence = _stable_current(RESERVATION, 1 << 20, 0o444)
    document = identity.strict_json(raw)
    required = {
        "schema",
        "lane",
        "issuer",
        "resource",
        "port",
        "binary_sha256",
        "harness_bundle_sha256",
        "raw_parity_receipt_sha256",
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
        raise RuntimeError("invalid root GPU authorization schema")
    expected = {
        "schema": SCHEMA,
        "lane": LANE,
        "issuer": "/root",
        "resource": "local-gb10-exclusive",
        "port": contract.PORT,
        "binary_sha256": contract.BINARY_SHA256,
        "harness_bundle_sha256": HARNESS_BUNDLE_SHA256,
        "raw_parity_receipt_sha256": contract.RAW_PARITY_RECEIPT_SHA256,
        "model_manifest_sha256": contract.MODEL_MANIFEST_SHA256,
        "model_content_root_sha256": contract.MODEL_CONTENT_ROOT_SHA256,
        "authorization": "b63-whole-prompt-ple-prefill-ab-only",
    }
    if any(document.get(key) != value for key, value in expected.items()):
        raise RuntimeError("root GPU authorization identity mismatch")
    for key in ("nonce_sha256", "boot_id_sha256", "nvidia_smi_sha256"):
        if not isinstance(document[key], str) or not identity.HEX64.fullmatch(
            document[key]
        ):
            raise RuntimeError(f"invalid authorization field: {key}")
    if not isinstance(document["gpu_uuid"], str) or not re.fullmatch(
        r"GPU-[0-9A-Fa-f-]{36}", document["gpu_uuid"]
    ):
        raise RuntimeError("invalid authorized GPU UUID")
    issued, expires = document["issued_at_unix"], document["expires_at_unix"]
    if (
        type(issued) is not int
        or type(expires) is not int
        or not 0 < expires - issued <= MAX_WINDOW_SECONDS
    ):
        raise RuntimeError("invalid authorization time window")
    expected_utc = datetime.fromtimestamp(issued, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    current = int(time.time()) if now is None else now
    if document["issued_at_utc"] != expected_utc or not issued <= current <= expires:
        raise RuntimeError("root GPU authorization is stale or not yet active")
    if document["boot_id_sha256"] != _boot_hash():
        raise RuntimeError("root GPU authorization boot identity mismatch")
    team_raw, team_evidence = _stable_current(TEAM_INBOX, 64 << 20, None)
    lines = team_raw.decode("utf-8", errors="strict").splitlines()
    claim = _claim_line(document)
    release = f"/root RELEASE-GPU: {LANE} nonce_sha256={document['nonce_sha256']}"
    if lines.count(claim) != 1 or any(release in line for line in lines):
        raise RuntimeError("root TEAM GPU claim is absent, duplicated, or released")
    return {
        "document": document,
        "file": evidence,
        "team_file": {key: team_evidence[key] for key in ("device", "inode", "mode")},
        "team_claim_sha256": hashlib.sha256(claim.encode()).hexdigest(),
    }


def input_recheck(
    manifest: dict[str, Any], parity: dict[str, Any], full: bool
) -> dict[str, Any]:
    import b63_ple_prefill_ab_model as model_identity
    import b63_ple_prefill_ab_validate as validate

    current = model_identity.load_manifest(
        contract.MODEL_MANIFEST, contract.MODEL_MANIFEST_SHA256
    )
    if current != manifest:
        raise RuntimeError("model manifest identity drift")
    current_parity = validate.attest_raw_parity()
    if current_parity != parity:
        raise RuntimeError("raw PLE parity receipt identity drift")
    return {
        "reservation": load_reservation(),
        "build": identity.attest_build(),
        "model_manifest": current["manifest_evidence"],
        "raw_parity": current_parity,
        "model": model_identity.attest_model(current, hash_content=full),
        "harness": attest_harness([HERE / name for name in HARNESS_NAMES]),
    }


def purge_unqualified(output: Path) -> None:
    expected = {
        f"{arm}-{suffix}"
        for arm in contract.ARM_ORDER
        for suffix in ("server.log", "http.jsonl")
    }
    for path in list(output.iterdir()):
        if path.name not in expected or not path.is_file() or path.is_symlink():
            raise RuntimeError(f"unexpected unqualified artifact: {path}")
        path.chmod(0o600)
        path.unlink()
    if list(output.iterdir()):
        raise RuntimeError("unqualified artifact purge incomplete")
