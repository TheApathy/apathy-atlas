# SPDX-License-Identifier: AGPL-3.0-only
"""Inert orchestrator/consumer for raw post-PLE serial parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import secrets
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Callable

import ple_prefill_parity_capture as capture
import ple_prefill_parity_contract as contract
import ple_prefill_parity_validate as validate

CAMPAIGN_SCHEMA = "atlas-qwen38-flash-next-post-ple-parity-campaign-v1"
MODEL_SCHEMA = "atlas-b62-qwen38-flash-next-model-manifest-v1"
RECORD_KEYS = {"relative", "sha256", "size", "mode", "device", "inode", "mtime_ns"}


def _aarch64_pie(raw: bytes) -> bool:
    return (
        len(raw) >= 64
        and raw[:7] == b"\x7fELF\x02\x01\x01"
        and int.from_bytes(raw[16:18], "little") == 3
        and int.from_bytes(raw[18:20], "little") == 183
        and int.from_bytes(raw[20:24], "little") == 1
        and int.from_bytes(raw[52:54], "little") == 64
    )


def _manifest(
    raw: bytes,
    count: int,
    label: str,
    *,
    dotted_paths: bool,
    sorted_paths: bool,
) -> dict[str, str]:
    lines = raw.decode("utf-8", errors="strict").splitlines()
    if len(lines) != count or not raw.endswith(b"\n"):
        raise RuntimeError(f"{label} manifest census/newline drift")
    result: dict[str, str] = {}
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  ([^\s\\]+)", line)
        if match is None:
            raise RuntimeError(f"{label} manifest line drift")
        digest, name = match.groups()
        if name.startswith("./") is not dotted_paths:
            raise RuntimeError(f"{label} manifest path style drift")
        relative = PurePosixPath(name.removeprefix("./"))
        if (
            relative.is_absolute()
            or not relative.parts
            or ".." in relative.parts
            or str(relative) != name.removeprefix("./")
            or name in result
        ):
            raise RuntimeError(f"{label} manifest path drift")
        result[name] = digest
    if sorted_paths and list(result) != sorted(result):
        raise RuntimeError(f"{label} manifest ordering drift")
    return result


def _sealed(path: Path, digest: str | None, limit: int = 128 << 20) -> bytes:
    return capture.stable_bytes(
        path,
        expected_sha256=digest,
        expected_size=None,
        expected_mode=0o444,
        limit=limit,
    )[0]


def _build_authority() -> dict[str, Any]:
    if (
        not contract.BUILD.is_absolute()
        or contract.BUILD.resolve(strict=True) != contract.BUILD
    ):
        raise RuntimeError("capture build root is not canonical")
    elf, elf_identity = capture.stable_bytes(
        contract.ELF,
        expected_sha256=contract.PINS["elf"],
        expected_size=contract.ELF_SIZE,
        expected_mode=0o555,
        limit=contract.ELF_SIZE,
    )
    if not _aarch64_pie(elf):
        raise RuntimeError("capture server is not an AArch64 PIE ELF")
    raw_files = {
        name: _sealed(contract.BUILD / name, contract.PINS[pin])
        for name, (pin, _) in contract.BUILD_FILES.items()
    }
    source = _manifest(
        raw_files["source-manifest.sha256"],
        contract.SOURCE_FILES,
        "source",
        dotted_paths=True,
        sorted_paths=True,
    )
    selected = _manifest(
        raw_files["selected-source-manifest.sha256"],
        25,
        "selected",
        dotted_paths=False,
        sorted_paths=False,
    )
    ptx = _manifest(
        raw_files["ptx-manifest.sha256"],
        154,
        "PTX",
        dotted_paths=True,
        sorted_paths=True,
    )
    artifacts = _manifest(
        raw_files["artifact-manifest.sha256"],
        18,
        "artifact",
        dotted_paths=False,
        sorted_paths=False,
    )
    qwen = "./crates/spark-model/src/layers/qwen4_ple.rs"
    callsite = "crates/spark-model/src/model/trait_impl/prefill_b/forward_layers.rs"
    if source.get(qwen) != contract.PINS["capture_source"]:
        raise RuntimeError("capture source is absent from the full source manifest")
    if selected.get(callsite) != contract.PINS["capture_callsite"]:
        raise RuntimeError(
            "capture callsite is absent from the selected-source manifest"
        )
    if artifacts.get("release/spark") != contract.PINS["elf"]:
        raise RuntimeError("artifact manifest does not bind launched server ELF")
    for name, (pin, _) in contract.BUILD_FILES.items():
        if (
            name != "artifact-manifest.sha256"
            and artifacts.get(name) != contract.PINS[pin]
        ):
            raise RuntimeError("artifact manifest does not bind build authority files")
    if any(not name.endswith(".ptx") for name in ptx):
        raise RuntimeError("PTX manifest contains a non-PTX target")
    receipt = raw_files["BUILD_RECEIPT.md"].decode("utf-8", errors="strict")
    required = (
        f"- Target signature: `{contract.TARGET}`",
        "- Kernel census: 154 PTX modules, 9 model-specific overrides",
        f"- Source manifest SHA256: `{contract.PINS['source_manifest']}` ({contract.SOURCE_FILES:,} files)",
        f"- Selected-source manifest SHA256: `{contract.PINS['selected_manifest']}` (25 files)",
        f"- PTX manifest SHA256: `{contract.PINS['ptx_manifest']}` (154 files)",
    )
    if any(receipt.count(line) != 1 for line in required):
        raise RuntimeError("capture build receipt cross-link drift")
    elf_line = re.findall(
        r"(?m)^- ELF identity: path=release/spark sha256=([0-9a-f]{64}) "
        r"size=([0-9]+) device=([0-9]+) inode=([0-9]+) mtime_ns=([0-9]+) "
        r"nlink=([0-9]+) mode=0555 build_id=([0-9a-f]{40})$",
        receipt,
    )
    wanted_elf = (
        contract.PINS["elf"],
        contract.ELF_SIZE,
        elf_identity["dev"],
        elf_identity["ino"],
        elf_identity["mtime_ns"],
        1,
        contract.BUILD_ID,
    )
    if (
        len(elf_line) != 1
        or tuple(
            value if index in (0, 6) else int(value)
            for index, value in enumerate(elf_line[0])
        )
        != wanted_elf
    ):
        raise RuntimeError("capture receipt/launched ELF identity drift")
    return {
        "elf": elf_identity,
        "manifests": {
            key: contract.PINS[value[0]] for key, value in contract.BUILD_FILES.items()
        },
    }


def _records(value: object, count: int, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) != count:
        raise RuntimeError(f"model {label} census drift")
    names = []
    for record in value:
        if not isinstance(record, dict) or set(record) != RECORD_KEYS:
            raise RuntimeError(f"model {label} record schema drift")
        name = record["relative"]
        path = PurePosixPath(name) if type(name) is str else PurePosixPath("/")
        if (
            type(name) is not str
            or path.is_absolute()
            or ".." in path.parts
            or str(path) != name
        ):
            raise RuntimeError(f"model {label} relative path drift")
        if type(record["sha256"]) is not str or not re.fullmatch(
            r"[0-9a-f]{64}", record["sha256"]
        ):
            raise RuntimeError(f"model {label} hash drift")
        if any(
            type(record[key]) is not int or record[key] < 0
            for key in RECORD_KEYS - {"relative", "sha256"}
        ):
            raise RuntimeError(f"model {label} stat type drift")
        if record["size"] <= 0 or record["mode"] > 0o777:
            raise RuntimeError(f"model {label} stat value drift")
        names.append(name)
    if names != sorted(set(names)):
        raise RuntimeError(f"model {label} ordering/uniqueness drift")
    return value


def _directory_identity(path: Path) -> dict[str, int]:
    info = path.lstat()
    if path.resolve(strict=True) != path or not stat.S_ISDIR(info.st_mode):
        raise RuntimeError("authoritative model directory identity drift")
    return {
        "device": info.st_dev,
        "inode": info.st_ino,
        "mode": stat.S_IMODE(info.st_mode),
        "mtime_ns": info.st_mtime_ns,
    }


def _attest_model_records(groups: tuple[list[dict[str, Any]], ...]) -> None:
    for group in groups:
        for record in group:
            path = contract.MODEL / record["relative"]
            info = path.lstat()
            actual = (
                info.st_size,
                stat.S_IMODE(info.st_mode),
                info.st_dev,
                info.st_ino,
                info.st_mtime_ns,
            )
            expected = tuple(
                record[key] for key in ("size", "mode", "device", "inode", "mtime_ns")
            )
            if (
                path.resolve(strict=True) != path
                or not stat.S_ISREG(info.st_mode)
                or actual != expected
            ):
                raise RuntimeError("authoritative model record identity drift")


def _model_authority() -> dict[str, Any]:
    model_before = _directory_identity(contract.MODEL)
    ple_before = _directory_identity(contract.MODEL / "ple-offload")
    model_stat = contract.MODEL.lstat()
    if (
        not contract.MODEL.is_absolute()
        or contract.MODEL.resolve(strict=True) != contract.MODEL
        or not stat.S_ISDIR(model_stat.st_mode)
    ):
        raise RuntimeError("authoritative model root identity drift")
    raw, evidence = capture.stable_bytes(
        contract.MODEL_MANIFEST,
        expected_sha256=contract.PINS["model_manifest"],
        expected_size=None,
        expected_mode=0o444,
        limit=1 << 20,
    )
    document = capture.strict_json(raw)
    if not isinstance(document, dict) or set(document) != {
        "schema",
        "model_path",
        "content_root_sha256",
        "metadata",
        "main_shards",
        "ple_sidecars",
    }:
        raise RuntimeError("model manifest schema drift")
    if document["schema"] != MODEL_SCHEMA or document["model_path"] != str(
        contract.MODEL
    ):
        raise RuntimeError("model manifest target drift")
    groups = (
        _records(document["metadata"], 5, "metadata"),
        _records(document["main_shards"], 197, "main shards"),
        _records(document["ple_sidecars"], 128, "PLE sidecars"),
    )
    metadata_names = {
        "config.json",
        "model.safetensors.index.json",
        "ple-offload/manifest.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    if {record["relative"] for record in groups[0]} != metadata_names:
        raise RuntimeError("model metadata name set drift")
    names = [record["relative"] for group in groups for record in group]
    if (
        len(names) != len(set(names))
        or any(
            PurePosixPath(name).name != name or not name.endswith(".safetensors")
            for name in names[5:202]
        )
        or any(not name.startswith("ple-offload/") for name in names[-128:])
    ):
        raise RuntimeError("model manifest cross-group/path drift")
    _attest_model_records(groups)
    content = [
        {key: record[key] for key in ("relative", "sha256", "size")}
        for group in groups
        for record in group
    ]
    root = hashlib.sha256(contract.canonical_bytes(content)).hexdigest()
    if (
        document["content_root_sha256"] != root
        or root != contract.PINS["model_content_root"]
    ):
        raise RuntimeError("model content-root drift")
    model_after = _directory_identity(contract.MODEL)
    ple_after = _directory_identity(contract.MODEL / "ple-offload")
    if (model_after, ple_after) != (model_before, ple_before):
        raise RuntimeError("authoritative model directory changed during admission")
    return {
        "manifest": evidence,
        "content_root_sha256": root,
        "model_root": model_before,
        "ple_root": ple_before,
    }


def preflight_authority() -> dict[str, Any]:
    contract.require_released_pins()
    return {"build": _build_authority(), "model": _model_authority()}


def orchestrate(output: Path, run_one: Callable[..., dict[str, Any]]) -> dict[str, Any]:
    authority = preflight_authority()
    if not output.is_absolute() or output.exists():
        raise RuntimeError("campaign output must be an absent absolute path")
    output.mkdir(mode=0o700)
    runs = []
    specs = contract.request_specs()
    for index, (name, arm) in enumerate(contract.SCHEDULE):
        nonce = secrets.token_hex(32)
        root = output / f"capture-{index}-{name}-{arm}"
        root.mkdir(mode=0o700)
        result = run_one(name, arm, specs[name], nonce, root)
        if not isinstance(result, dict) or set(result) != {"process", "response"}:
            raise RuntimeError("runtime adapter result schema drift")
        run = {"name": name, "arm": arm, "nonce": nonce, "capture_root": root, **result}
        run["frames"] = capture.load_arm(
            root, arm, specs[name].prompt_tokens, nonce, result["process"]["pid"]
        )
        runs.append(run)
    return {"authority": authority, "result": validate.validate_campaign(runs)}


def load_campaign(path: Path) -> list[dict[str, Any]]:
    raw = _sealed(path, None)
    document = capture.strict_json(raw)
    if (
        not isinstance(document, dict)
        or set(document) != {"schema", "runs"}
        or document["schema"] != CAMPAIGN_SCHEMA
    ):
        raise RuntimeError("campaign manifest schema drift")
    specs, runs = contract.request_specs(), []
    for record in document["runs"]:
        if not isinstance(record, dict) or set(record) != {
            "name",
            "arm",
            "nonce",
            "capture_root",
            "request_sha256",
            "process",
            "response",
        }:
            raise RuntimeError("campaign run record schema drift")
        if record["request_sha256"] != contract.sha256(specs[record["name"]].wire):
            raise RuntimeError("campaign request wire drift")
        root = Path(record["capture_root"])
        run = {key: value for key, value in record.items() if key != "request_sha256"}
        run["capture_root"] = root
        run["frames"] = capture.load_arm(
            root,
            record["arm"],
            specs[record["name"]].prompt_tokens,
            record["nonce"],
            record["process"]["pid"],
        )
        runs.append(run)
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", required=True, choices=[contract.EXECUTE])
    parser.add_argument("--campaign-manifest", required=True, type=Path)
    args = parser.parse_args()
    preflight_authority()
    try:
        result = validate.validate_campaign(load_campaign(args.campaign_manifest))
    except validate.ParityMismatch as error:
        result = {
            "schema": contract.RESULT_SCHEMA,
            "qualified": False,
            "stop": "STOP-NO-TIMING",
            "performance_claim_allowed": False,
            "timing_allowed": False,
            "diagnostics": error.diagnostics,
        }
        print(
            json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
        )
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
