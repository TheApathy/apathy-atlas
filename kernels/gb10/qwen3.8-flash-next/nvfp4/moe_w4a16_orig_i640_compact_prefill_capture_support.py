# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import subprocess
from pathlib import Path

HEX = re.compile(r"[0-9a-f]{64}\Z")
KEY = re.compile(r"[a-z0-9_.-]{1,96}\Z")
EVENT = struct.Struct("<QIIIIQqq64s")
EVENT_MAGIC = 0x4F49363430455631


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_id(path: Path) -> str:
    result = subprocess.run(
        ["/usr/bin/readelf", "--notes", "--wide", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    matches = re.findall(r"Build ID: ([0-9a-f]+)", result.stdout)
    if len(matches) != 1 or not 16 <= len(matches[0]) <= 128:
        raise ValueError("exactly one ELF build-id is required")
    return matches[0]


def identity(path: Path, require_mode: int | None = None) -> dict[str, str]:
    resolved = path.resolve(strict=True)
    stat = resolved.stat()
    mode = stat.st_mode & 0o777
    if not resolved.is_file() or (require_mode is not None and mode != require_mode):
        raise ValueError(f"invalid file/mode: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha(resolved),
        "mode": f"{mode:04o}",
        "dev": str(stat.st_dev),
        "ino": str(stat.st_ino),
        "size": str(stat.st_size),
        "build_id": build_id(resolved),
    }


def encode_kv(fields: dict[str, str]) -> bytes:
    if not fields or len(fields) > 2048:
        raise ValueError("manifest field count")
    lines = []
    for key in sorted(fields):
        value = str(fields[key])
        if (
            not KEY.fullmatch(key)
            or not value
            or len(value) > 4096
            or "\n" in value
            or "=" in key
        ):
            raise ValueError(f"invalid manifest field {key!r}")
        lines.append(f"{key}={value}\n")
    result = "".join(lines).encode()
    if len(result) > 1 << 20:
        raise ValueError("manifest too large")
    return result


def decode_kv(data: bytes, limit: int = 1 << 20) -> dict[str, str]:
    if not data or len(data) > limit or not data.endswith(b"\n") or b"\0" in data:
        raise ValueError("manifest framing")
    text = data.decode("utf-8")
    fields: dict[str, str] = {}
    previous = ""
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        if separator != "=" or not KEY.fullmatch(key) or not value or key <= previous:
            raise ValueError("manifest key order/shape")
        fields[key] = value
        previous = key
    if encode_kv(fields) != data:
        raise ValueError("noncanonical manifest")
    return fields


def write_immutable(path: Path, data: bytes) -> None:
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
    finally:
        os.close(descriptor)


def read_immutable(path: Path, limit: int = 1 << 20) -> bytes:
    stat = path.stat()
    if stat.st_mode & 0o222 or stat.st_size <= 0 or stat.st_size > limit:
        raise ValueError("artifact must be bounded and immutable")
    data = path.read_bytes()
    if len(data) != stat.st_size:
        raise ValueError("artifact size drift")
    return data


def parse_offsets(path: Path, rows: int) -> tuple[list[int], str]:
    data = read_immutable(path, 2052)
    if len(data) != 2052:
        raise ValueError("offset byte count")
    values = list(struct.unpack("<513I", data))
    if (
        values[0] != 0
        or values[-1] != rows * 10
        or any(b < a for a, b in zip(values, values[1:]))
    ):
        raise ValueError("offset contract")
    return values, hashlib.sha256(data).hexdigest()


def parse_event(path: Path, rows: int, nonce: str, pid: int) -> dict[str, str]:
    data = read_immutable(path, EVENT.size)
    if len(data) != EVENT.size:
        raise ValueError("event byte count")
    magic, version, event_pid, endpoint, size, source, seconds, nanos, raw_nonce = (
        EVENT.unpack(data)
    )
    if (
        (magic, version, event_pid, endpoint, size)
        != (
            EVENT_MAGIC,
            1,
            pid,
            rows * 10,
            2052,
        )
        or raw_nonce.decode() != nonce
        or source == 0
        or seconds <= 0
        or not 0 <= nanos < 1_000_000_000
    ):
        raise ValueError("event contract")
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "source": str(source),
        "seconds": str(seconds),
        "nanos": str(nanos),
    }


def model_manifest(model: Path) -> dict[str, str]:
    config = model / "config.json"
    index = model / "model.safetensors.index.json"
    mapping = json.loads(bounded_bytes(index, 16 << 20))["weight_map"]
    shards = sorted(set(mapping.values()))
    fields = {
        "schema": "oi640-model-v1",
        "config.sha256": sha(config),
        "index.sha256": sha(index),
        "shard.count": str(len(shards)),
    }
    for number, name in enumerate(shards):
        path = model / name
        fields[f"shard.{number:03}.path"] = name
        fields[f"shard.{number:03}.sha256"] = sha(path)
        fields[f"shard.{number:03}.size"] = str(path.stat().st_size)
    return fields


def validate_model(fields: dict[str, str], model: Path) -> None:
    if fields.get("schema") != "oi640-model-v1":
        raise ValueError("model schema")
    count = int(fields["shard.count"])
    if (
        not 1 <= count <= 256
        or sha(model / "config.json") != fields["config.sha256"]
        or sha(model / "model.safetensors.index.json") != fields["index.sha256"]
    ):
        raise ValueError("model root identity")
    index = json.loads(bounded_bytes(model / "model.safetensors.index.json", 16 << 20))
    expected = sorted(set(index["weight_map"].values()))
    observed = [fields[f"shard.{number:03}.path"] for number in range(count)]
    if observed != expected:
        raise ValueError("manifest does not enumerate the exact index shard set")
    for number in range(count):
        prefix = f"shard.{number:03}"
        name = fields[f"{prefix}.path"]
        if Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError("unsafe shard path")
        path = model / name
        if sha(path) != fields[f"{prefix}.sha256"] or path.stat().st_size != int(
            fields[f"{prefix}.size"]
        ):
            raise ValueError("shard drift")


def bounded_bytes(path: Path, limit: int) -> bytes:
    before = path.stat()
    if before.st_size <= 0 or before.st_size > limit:
        raise ValueError("bounded file size")
    data = path.read_bytes()
    after = path.stat()
    if len(data) != before.st_size or before != after:
        raise ValueError("bounded file drift")
    return data


def source_bundle(paths: list[Path], root: Path) -> str:
    lines = [f"{path.relative_to(root)}={sha(path)}\n" for path in sorted(paths)]
    return hashlib.sha256("".join(lines).encode()).hexdigest()
