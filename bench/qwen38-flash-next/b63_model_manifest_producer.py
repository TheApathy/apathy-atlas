# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import argparse
import hashlib
import os
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Any

from b63_model_manifest_support import (
    canonical_bytes,
    expected_main_shards as _expected_main_shards,
    parse_json as _json,
    relative_path as _relative,
    scan as _scan,
)

MODEL = Path("/home/flocka/models/Qwen3.8-Flash-Next-NVFP4-Offload")
OUTPUT = Path("/var/tmp/atlas-b62-qwen38-flash-next-model-manifest.json")
SCHEMA = "atlas-b62-qwen38-flash-next-model-manifest-v1"
EXECUTE = "HASH_FULL_UNPRUNED_FLASH_NEXT_MODEL_TWICE_AND_SEAL_MANIFEST"
DIRECTORY_FLAGS = os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_DIRECTORY", 0)
META_HASHES = {
    "config.json": "e765305daba0951974308f4d32c075b52a6a45974730d273f2216718a994d624",
    "model.safetensors.index.json": "c654034a19be39baf2348dc02c818b555d3e0f2dc036346f58fe3623c1bc311d",
    "ple-offload/manifest.json": "01911bc91039510642c8ebec4aa4f777bd533b317a0f448063707072aa3b5520",
    "tokenizer.json": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
    "tokenizer_config.json": "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27",
}


def _names(
    index_raw: bytes, ple_raw: bytes, main_count: int
) -> tuple[list[str], dict[str, tuple]]:
    index = _json(index_raw)
    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError("model index has no weight_map")
    values = list(weight_map.values())
    if any(not isinstance(name, str) for name in values):
        raise RuntimeError("model index contains a non-string shard")
    main = sorted(set(values))
    if main != _expected_main_shards(main_count) or any(
        PurePosixPath(_relative(name)).name != name for name in main
    ):
        raise RuntimeError("model index shard census/path mismatch")
    ple_doc = _json(ple_raw)
    entries = ple_doc.get("entries") if isinstance(ple_doc, dict) else None
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("PLE manifest has no entries")
    expected: dict[str, tuple] = {}
    for item in entries:
        if not isinstance(item, dict):
            raise RuntimeError("invalid PLE manifest entry")
        name = _relative(item.get("file"))
        if PurePosixPath(name).name != name:
            raise RuntimeError("PLE manifest entry is not flat")
        size, digest = item.get("bytes"), item.get("sha256")
        if (
            type(size) is not int
            or size <= 0
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise RuntimeError("invalid PLE manifest content identity")
        if name in expected:
            raise RuntimeError("duplicate PLE manifest entry")
        expected[name] = (size, digest)
    return main, expected


def build_document(
    root: Path, meta_hashes: dict[str, str], main_count: int
) -> dict[str, Any]:
    root_before = root.lstat()
    if root.resolve(strict=True) != root or not stat.S_ISDIR(root_before.st_mode):
        raise RuntimeError("model root is not a canonical directory")
    metadata: list[dict] = []
    raw: dict[str, bytes] = {}
    for relative in sorted(meta_hashes):
        record, content = _scan(root, relative, capture=True)
        if record["sha256"] != meta_hashes[relative]:
            raise RuntimeError(f"metadata hash mismatch: {relative}")
        metadata.append(record)
        raw[relative] = content or b""
    main_names, ple_expected = _names(
        raw["model.safetensors.index.json"],
        raw["ple-offload/manifest.json"],
        main_count,
    )
    main = [_scan(root, relative)[0] for relative in main_names]
    ple = [_scan(root, f"ple-offload/{name}")[0] for name in sorted(ple_expected)]
    for record in ple:
        if (record["size"], record["sha256"]) != ple_expected[
            PurePosixPath(record["relative"]).name
        ]:
            raise RuntimeError("PLE sidecar differs from its manifest")
    first = metadata + main + ple
    second = [_scan(root, item["relative"])[0] for item in first]
    if first != second:
        raise RuntimeError("model identity changed between full hash passes")
    root_after = root.lstat()
    before_identity = (
        root_before.st_dev,
        root_before.st_ino,
        stat.S_IMODE(root_before.st_mode),
        root_before.st_mtime_ns,
    )
    after_identity = (
        root_after.st_dev,
        root_after.st_ino,
        stat.S_IMODE(root_after.st_mode),
        root_after.st_mtime_ns,
    )
    if before_identity != after_identity:
        raise RuntimeError("model root changed between full hash passes")
    content_records = [
        {key: item[key] for key in ("relative", "sha256", "size")} for item in second
    ]
    content_root = hashlib.sha256(canonical_bytes(content_records)).hexdigest()
    return {
        "schema": SCHEMA,
        "model_path": str(root),
        "content_root_sha256": content_root,
        "metadata": second[: len(metadata)],
        "main_shards": second[len(metadata) : len(metadata) + len(main)],
        "ple_sidecars": second[len(metadata) + len(main) :],
    }


def _cleanup_owned(output: Path, opened: os.stat_result) -> None:
    try:
        current = output.lstat()
        if (current.st_dev, current.st_ino) == (opened.st_dev, opened.st_ino):
            output.unlink()
    except (FileNotFoundError, OSError):
        pass
    try:
        parent_fd = os.open(output.parent, DIRECTORY_FLAGS)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    except OSError:
        pass


def produce(root: Path, output: Path, meta: dict[str, str], count: int) -> dict:
    if not output.is_absolute() or output.parent.resolve(strict=True) != output.parent:
        raise RuntimeError("manifest output parent is not canonical")
    parent_before = output.parent.lstat()
    document = build_document(root, meta, count)
    payload = canonical_bytes(document) + b"\n"
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = parent_fd = guard_fd = -1
    opened = None
    try:
        fd = os.open(output, flags, 0o600)
        opened = os.fstat(fd)
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise RuntimeError("short manifest write")
            view = view[written:]
        os.fsync(fd)
        os.fchmod(fd, 0o444)
        os.fsync(fd)
        sealed = os.fstat(fd)
        if os.pread(fd, len(payload) + 1, 0) != payload:
            raise RuntimeError("held manifest bytes differ after sealing")
        guard_fd = os.open(
            output, os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0)
        )
        guarded = os.fstat(guard_fd)
        if (guarded.st_dev, guarded.st_ino) != (opened.st_dev, opened.st_ino):
            raise RuntimeError("manifest guard opened a replacement")
        closing, fd = fd, -1
        os.close(closing)
        parent_fd = os.open(output.parent, DIRECTORY_FLAGS)
        os.fsync(parent_fd)
        closing, parent_fd = parent_fd, -1
        os.close(closing)
        final, final_raw = _scan(output.parent, output.name, capture=True)
        expected_identity = (
            sealed.st_dev,
            sealed.st_ino,
            sealed.st_size,
            stat.S_IMODE(sealed.st_mode),
            sealed.st_mtime_ns,
        )
        final_identity = tuple(
            final[key] for key in ("device", "inode", "size", "mode", "mtime_ns")
        )
        parent_after = output.parent.lstat()
        if expected_identity != final_identity or final_raw != payload:
            raise RuntimeError("model manifest final identity/hash mismatch")
        if (parent_before.st_dev, parent_before.st_ino) != (
            parent_after.st_dev,
            parent_after.st_ino,
        ):
            raise RuntimeError("manifest output parent changed")
        closing, guard_fd = guard_fd, -1
        os.close(closing)
    except BaseException:
        if opened is None and fd >= 0:
            try:
                opened = os.stat(f"/proc/self/fd/{fd}")
            except OSError:
                pass
        if opened is not None:
            _cleanup_owned(output, opened)
        raise
    finally:
        for descriptor in (fd, parent_fd, guard_fd):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
    return {
        "schema": "atlas-b63-model-manifest-producer-receipt-v1",
        "manifest": str(output),
        "manifest_sha256": hashlib.sha256(payload).hexdigest(),
        "content_root_sha256": document["content_root_sha256"],
        "main_shards": len(document["main_shards"]),
        "ple_sidecars": len(document["ple_sidecars"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", required=True)
    args = parser.parse_args()
    if args.execute != EXECUTE:
        raise SystemExit("explicit full-model hashing authorization is required")
    receipt = produce(MODEL, OUTPUT, META_HASHES, 197)
    print(canonical_bytes(receipt).decode())


if __name__ == "__main__":
    main()
