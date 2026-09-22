# SPDX-License-Identifier: AGPL-3.0-only
"""Fail-closed b61 model reference capture and admission."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import nsys_b61_capture_support as support

SCHEMA = "atlas-qwen38-flash-next-model-reference-v1"
REFERENCE = Path(__file__).with_name("qwen38_flash_next_b61_model_reference.json")
# Deliberately non-authorizing until a separately reviewed 197-shard receipt is pinned.
REFERENCE_SHA256 = "UNRELEASED_AUTHORITATIVE_197_SHARD_MANIFEST"
CAPTURE_AUTHORIZATION = "CAPTURE_UNTRUSTED_REFERENCE_FOR_REVIEW"
MODEL_META = {
    "config.json": "e765305daba0951974308f4d32c075b52a6a45974730d273f2216718a994d624",
    "model.safetensors.index.json": "c654034a19be39baf2348dc02c818b555d3e0f2dc036346f58fe3623c1bc311d",
    "tokenizer.json": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
    "tokenizer_config.json": "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27",
    "ple-offload/manifest.json": "01911bc91039510642c8ebec4aa4f777bd533b317a0f448063707072aa3b5520",
}


def _main_shard_names() -> list[str]:
    index_path = support.MODEL / "model.safetensors.index.json"
    index = json.loads(index_path.read_bytes())
    names = sorted(set(index["weight_map"].values()))
    if len(names) != 197 or any(Path(name).name != name for name in names):
        raise RuntimeError("expected exactly 197 flat model shards")
    return names


def _records(names: list[str]) -> list[dict[str, Any]]:
    result = []
    for name in names:
        path = support.MODEL / name
        info = support.require_regular(path)
        result.append(
            {"name": name, "sha256": support.sha_file(path), "size": info.st_size}
        )
    return result


def _validate_records(records: object) -> list[dict[str, Any]]:
    if not isinstance(records, list) or len(records) != 197:
        raise RuntimeError("reference must contain exactly 197 main shards")
    names = []
    for item in records:
        if not isinstance(item, dict) or set(item) != {"name", "sha256", "size"}:
            raise RuntimeError("invalid model reference record")
        name, digest, size = item["name"], item["sha256"], item["size"]
        if not isinstance(name, str) or Path(name).name != name:
            raise RuntimeError("model reference shard is not flat")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise RuntimeError("invalid model reference hash")
        if type(size) is not int or size <= 0:
            raise RuntimeError("invalid model reference size")
        names.append(name)
    if names != sorted(set(names)):
        raise RuntimeError("model reference records are not sorted and unique")
    return records


def load_reference() -> dict[str, Any]:
    if not re.fullmatch(r"[0-9a-f]{64}", REFERENCE_SHA256):
        raise RuntimeError("authoritative 197-shard model reference is not pinned")
    support.require_regular(REFERENCE)
    if support.sha_file(REFERENCE) != REFERENCE_SHA256:
        raise RuntimeError("authoritative model reference hash drift")
    document = json.loads(REFERENCE.read_bytes())
    if set(document) != {"schema", "model_path", "index_sha256", "main_shards"}:
        raise RuntimeError("invalid authoritative model reference schema")
    if document["schema"] != SCHEMA or document["model_path"] != str(support.MODEL):
        raise RuntimeError("authoritative model reference identity drift")
    if document["index_sha256"] != MODEL_META["model.safetensors.index.json"]:
        raise RuntimeError("authoritative model index identity drift")
    _validate_records(document["main_shards"])
    return document


def _ple_records() -> list[dict[str, Any]]:
    manifest = json.loads((support.MODEL / "ple-offload/manifest.json").read_bytes())
    result = []
    for expected, entry in enumerate(
        sorted(manifest["entries"], key=lambda item: item["shard"])
    ):
        name = entry["file"]
        if entry["shard"] != expected or Path(name).name != name:
            raise RuntimeError("PLE sidecar inventory is not flat and contiguous")
        path = support.MODEL / "ple-offload" / name
        info = support.require_regular(path)
        digest = support.sha_file(path)
        if info.st_size != entry["bytes"] or digest != entry["sha256"]:
            raise RuntimeError(f"PLE sidecar drift: {name}")
        result.append({"name": name, "sha256": digest, "size": info.st_size})
    return result


def attest_model() -> dict[str, Any]:
    reference = load_reference()  # Rejects the placeholder before any model read.
    for relative, expected in MODEL_META.items():
        path = support.MODEL / relative
        support.require_regular(path)
        if support.sha_file(path) != expected:
            raise RuntimeError(f"model metadata drift: {relative}")
    names = _main_shard_names()
    if names != [item["name"] for item in reference["main_shards"]]:
        raise RuntimeError("model index/reference shard set drift")
    actual = _records(names)
    if actual != reference["main_shards"]:
        raise RuntimeError("authoritative main-shard inventory drift")
    ple = _ple_records()
    if any(
        support.sha_file(support.MODEL / name) != digest
        for name, digest in MODEL_META.items()
    ):
        raise RuntimeError("model metadata drift during inventory")
    return {
        "reference_sha256": REFERENCE_SHA256,
        "main_aggregate_sha256": hashlib.sha256(
            support.canonical_bytes(actual)
        ).hexdigest(),
        "ple_aggregate_sha256": hashlib.sha256(
            support.canonical_bytes(ple)
        ).hexdigest(),
        "main_shards": actual,
        "ple_sidecars": ple,
    }


def capture_candidate(output: Path, authorization: str) -> None:
    if authorization != CAPTURE_AUTHORIZATION:
        raise RuntimeError("explicit candidate-capture authorization missing")
    for relative, expected in MODEL_META.items():
        path = support.MODEL / relative
        support.require_regular(path)
        if support.sha_file(path) != expected:
            raise RuntimeError(f"model metadata drift: {relative}")
    names = _main_shard_names()
    document = {
        "schema": SCHEMA,
        "model_path": str(support.MODEL),
        "index_sha256": MODEL_META["model.safetensors.index.json"],
        "main_shards": _records(names),
    }
    if any(
        support.sha_file(support.MODEL / name) != digest
        for name, digest in MODEL_META.items()
    ):
        raise RuntimeError("model metadata drift during candidate capture")
    with output.open("xb") as stream:
        stream.write(support.canonical_bytes(document) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    output.chmod(0o444)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-reference", type=Path, required=True)
    parser.add_argument("--authorization", required=True)
    args = parser.parse_args()
    capture_candidate(args.capture_reference.resolve(), args.authorization)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
