# SPDX-License-Identifier: AGPL-3.0-only
"""Full model-manifest admission and pre/post content identity."""

from __future__ import annotations

import hashlib
import os
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Any

import b62_prefill_ab_contract as contract
import b62_prefill_ab_identity as identity

SCHEMA = "atlas-b62-qwen38-flash-next-model-manifest-v1"
META_HASHES = {
    "config.json": "e765305daba0951974308f4d32c075b52a6a45974730d273f2216718a994d624",
    "model.safetensors.index.json": "c654034a19be39baf2348dc02c818b555d3e0f2dc036346f58fe3623c1bc311d",
    "ple-offload/manifest.json": "01911bc91039510642c8ebec4aa4f777bd533b317a0f448063707072aa3b5520",
    "tokenizer.json": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
    "tokenizer_config.json": "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27",
}
RECORD_KEYS = {"relative", "sha256", "size", "mode", "device", "inode", "mtime_ns"}


def _valid_relative(value: object) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts and str(path) == value


def _records(value: object, *, count: int | None = None) -> list[dict[str, Any]]:
    if not isinstance(value, list) or (count is not None and len(value) != count):
        raise RuntimeError("model manifest record census mismatch")
    names = []
    for record in value:
        if not isinstance(record, dict) or set(record) != RECORD_KEYS:
            raise RuntimeError("invalid model manifest record")
        if not _valid_relative(record["relative"]):
            raise RuntimeError("invalid model manifest relative path")
        if not isinstance(record["sha256"], str) or not re.fullmatch(
            r"[0-9a-f]{64}", record["sha256"]
        ):
            raise RuntimeError("invalid model manifest content hash")
        for key in ("size", "mode", "device", "inode", "mtime_ns"):
            if type(record[key]) is not int or record[key] < 0:
                raise RuntimeError("invalid model manifest stat identity")
        if record["size"] <= 0 or record["mode"] > 0o777:
            raise RuntimeError("invalid model manifest size/mode")
        names.append(record["relative"])
    if names != sorted(set(names)):
        raise RuntimeError("model manifest records must be sorted and unique")
    return value


def load_manifest(path: Path, expected_sha256: str) -> dict[str, Any]:
    if (
        expected_sha256 != contract.MODEL_MANIFEST_SHA256
        or not identity.HEX64.fullmatch(contract.MODEL_MANIFEST_SHA256)
        or not identity.HEX64.fullmatch(contract.MODEL_CONTENT_ROOT_SHA256)
        or path != contract.MODEL_MANIFEST
    ):
        raise RuntimeError("authoritative full-shard model root is not pinned")
    return _load_manifest_document(
        path, expected_sha256, contract.MODEL_CONTENT_ROOT_SHA256
    )


def _load_manifest_document(
    path: Path, expected_sha256: str, expected_content_root: str
) -> dict[str, Any]:
    document, evidence = identity.load_sealed_json(path, expected_sha256)
    if not isinstance(document, dict) or set(document) != {
        "schema",
        "model_path",
        "content_root_sha256",
        "metadata",
        "main_shards",
        "ple_sidecars",
    }:
        raise RuntimeError("invalid model manifest top-level schema")
    if document["schema"] != SCHEMA or document["model_path"] != str(contract.MODEL):
        raise RuntimeError("model manifest target mismatch")
    metadata = _records(document["metadata"], count=len(META_HASHES))
    main = _records(document["main_shards"], count=197)
    ple = _records(document["ple_sidecars"])
    if not ple:
        raise RuntimeError("model manifest has no PLE sidecars")
    actual_meta = {record["relative"]: record["sha256"] for record in metadata}
    if actual_meta != META_HASHES:
        raise RuntimeError("model metadata hash set mismatch")
    if any(PurePosixPath(item["relative"]).name != item["relative"] for item in main):
        raise RuntimeError("main model shards must be flat")
    if any(not item["relative"].startswith("ple-offload/") for item in ple):
        raise RuntimeError("PLE sidecar path escaped its directory")
    all_names = [item["relative"] for group in (metadata, main, ple) for item in group]
    if len(all_names) != len(set(all_names)):
        raise RuntimeError("model manifest duplicates paths across groups")
    content_records = [
        {key: item[key] for key in ("relative", "sha256", "size")}
        for group in (metadata, main, ple)
        for item in group
    ]
    root = hashlib.sha256(contract.canonical_bytes(content_records)).hexdigest()
    if document["content_root_sha256"] != root or root != expected_content_root:
        raise RuntimeError("authoritative model content root mismatch")
    return {**document, "manifest_evidence": evidence}


def _attest_record(record: dict[str, Any], *, hash_content: bool) -> dict[str, Any]:
    path = contract.MODEL / record["relative"]
    if path.resolve(strict=True) != path:
        raise RuntimeError(f"model entry path is not canonical: {path}")
    flags = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        actual = {
            "relative": record["relative"],
            "size": before.st_size,
            "mode": stat.S_IMODE(before.st_mode),
            "device": before.st_dev,
            "inode": before.st_ino,
            "mtime_ns": before.st_mtime_ns,
        }
        if not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"model entry is not regular: {path}")
        if any(actual[key] != record[key] for key in actual if key != "relative"):
            raise RuntimeError(f"model stat identity drift: {path}")
        digest = hashlib.sha256()
        if hash_content:
            for block in iter(lambda: os.read(fd, 8 << 20), b""):
                digest.update(block)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        stat.S_IMODE(before.st_mode),
        before.st_mtime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        stat.S_IMODE(after.st_mode),
        after.st_mtime_ns,
    )
    if before_identity != after_identity:
        raise RuntimeError(f"model entry drift during admission: {path}")
    current = path.lstat()
    current_identity = (
        current.st_dev,
        current.st_ino,
        current.st_size,
        stat.S_IMODE(current.st_mode),
        current.st_mtime_ns,
    )
    if current_identity != before_identity:
        raise RuntimeError(f"model pathname replacement: {path}")
    if hash_content and digest.hexdigest() != record["sha256"]:
        raise RuntimeError(f"model content drift: {path}")
    return {**actual, "sha256": record["sha256"]}


def _load_small(record: dict[str, Any]) -> Any:
    if record["size"] > 128 << 20:
        raise RuntimeError("model metadata exceeds bounded parser")
    path = contract.MODEL / record["relative"]
    raw, evidence = identity.stable_bytes(
        path, expected_sha256=record["sha256"], max_bytes=128 << 20
    )
    for key in ("size", "mode", "device", "inode", "mtime_ns"):
        if evidence[key] != record[key]:
            raise RuntimeError(
                "model metadata identity changed before relationship check"
            )
    return identity.strict_json(raw)


def _crosslink(manifest: dict[str, Any]) -> None:
    metadata = {item["relative"]: item for item in manifest["metadata"]}
    main_names = [item["relative"] for item in manifest["main_shards"]]
    index = _load_small(metadata["model.safetensors.index.json"])
    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
        raise RuntimeError("model index schema mismatch")
    if sorted(set(index["weight_map"].values())) != main_names:
        raise RuntimeError("model index/main-shard cross-link mismatch")
    ple_manifest = _load_small(metadata["ple-offload/manifest.json"])
    entries = ple_manifest.get("entries") if isinstance(ple_manifest, dict) else None
    if not isinstance(entries, list):
        raise RuntimeError("PLE manifest schema mismatch")
    expected = {
        item["relative"].removeprefix("ple-offload/"): (item["size"], item["sha256"])
        for item in manifest["ple_sidecars"]
    }
    actual = {
        item.get("file"): (item.get("bytes"), item.get("sha256")) for item in entries
    }
    if actual != expected or len(actual) != len(entries):
        raise RuntimeError("PLE manifest/sidecar cross-link mismatch")


def attest_model(manifest: dict[str, Any], *, hash_content: bool) -> dict[str, Any]:
    if (
        contract.MODEL.resolve(strict=True) != contract.MODEL
        or not contract.MODEL.is_dir()
    ):
        raise RuntimeError("model root identity drift")
    records = [
        _attest_record(record, hash_content=hash_content)
        for group in ("metadata", "main_shards", "ple_sidecars")
        for record in manifest[group]
    ]
    _crosslink(manifest)
    return {
        "file_count": len(records),
        "main_shards": len(manifest["main_shards"]),
        "ple_sidecars": len(manifest["ple_sidecars"]),
        "aggregate_sha256": hashlib.sha256(
            contract.canonical_bytes(records)
        ).hexdigest(),
    }
