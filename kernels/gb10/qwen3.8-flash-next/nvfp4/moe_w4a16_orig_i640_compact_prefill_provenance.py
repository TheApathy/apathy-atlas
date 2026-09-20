# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from moe_w4a16_orig_i640_compact_prefill_capture_support import (
    HEX,
    decode_kv,
    encode_kv,
    identity,
    read_immutable,
    sha,
    source_bundle,
)

PREFIX = Path("kernels/gb10/qwen3.8-flash-next/nvfp4")
BASE = "moe_w4a16_orig_i640_compact_prefill"
SOURCE_NAMES = tuple(
    f"{BASE}{suffix}"
    for suffix in ".cu .cuh _capture.py _capture_preload.c _capture_run.py _capture_support.py _gemm.cuh _manifest.py _microgate.cu _microgate_buffers.cuh _microgate_cases.cuh _microgate_io.cuh _microgate_manifest.cuh _microgate_provenance.cuh _microgate_timing.cuh _plan.cuh _provenance.py _static_source_cases.py _static_support.py".split()
) + tuple(
    f"test_{BASE}{suffix}" for suffix in "_capture.py _provenance.py _static.py".split()
)
PRODUCER_NAMES = tuple(
    f"{BASE}_{part}.py"
    for part in "capture capture_run capture_support provenance".split()
)
TARGET = "gb10|qwen3.8-flash-next|nvfp4|sm_121f"
IDENTITY_KEYS = {"path", "sha256", "mode", "dev", "ino", "size", "build_id"}
B62 = {
    "binary.path": "/var/tmp/atlas-flash-b62-build-D6fncRTX/release/spark",
    "binary.sha256": "77fe8bc1ad3ed90c3b3fb4db5323e55d36c5fedc5b76852f36a445d25193d5bd",
    "binary.mode": "0555",
    "binary.dev": "66306",
    "binary.ino": "27698744",
    "binary.size": "38569952",
    "binary.build_id": "5d9d0cc7cf59c20cd2f12af01e6221a332698645",
    "source.manifest.path": "/var/tmp/atlas-flash-b62-build-D6fncRTX/source-manifest.sha256",
    "source.manifest.sha256": "703d2d4c651c1ae5c3a868f172f530000fd4c14050b569c74572b893c600b504",
    "selected.manifest.path": "/var/tmp/atlas-flash-b62-build-D6fncRTX/selected-source-manifest.sha256",
    "selected.manifest.sha256": "997218b8bce625a57bc2bfd52d384d54abd070d2ace0e8d85d079fe3b1e54ef7",
    "ptx.manifest.path": "/var/tmp/atlas-flash-b62-build-D6fncRTX/ptx-manifest.sha256",
    "ptx.manifest.sha256": "cbe5fd1ad856ae41a41a46f03870aa1e66563f38d9cf377152b5e40957281253",
    "upstream.receipt.path": "/var/tmp/atlas-flash-b62-build-D6fncRTX/BUILD_RECEIPT.md",
    "upstream.receipt.sha256": "c40ad52e50c6af734e6ff0c573d2ca34db80da9e8525a8dedf17b41bc793547a",
}


def source_receipt(entry: Path) -> dict[str, str]:
    directory = entry.resolve(strict=True).parent
    if len(directory.parents) <= 3:
        raise ValueError("producer must execute from the canonical source tree")
    root = directory.parents[3]
    if directory != root / PREFIX or entry.resolve() != directory / PRODUCER_NAMES[0]:
        raise ValueError("producer must execute from the canonical source tree")
    paths = [directory / name for name in SOURCE_NAMES]
    modules = {
        PRODUCER_NAMES[1]: "moe_w4a16_orig_i640_compact_prefill_capture_run",
        PRODUCER_NAMES[2]: "moe_w4a16_orig_i640_compact_prefill_capture_support",
        PRODUCER_NAMES[3]: __name__,
    }
    for name, module_name in modules.items():
        module = sys.modules.get(module_name)
        if (
            module is None
            or Path(module.__file__).resolve(strict=True) != directory / name
        ):
            raise ValueError("loaded producer source identity")
    producer = [directory / name for name in PRODUCER_NAMES]
    python = identity(Path(sys.executable))
    fields = {
        "schema": "oi640-producer-sources-v1",
        "source.count": str(len(paths)),
        "source.bundle_sha256": source_bundle(paths, root),
        "producer.count": str(len(producer)),
        "producer.bundle_sha256": source_bundle(producer, root),
        "producer.entry.path": str((PREFIX / PRODUCER_NAMES[0]).as_posix()),
        "producer.entry.sha256": sha(entry.resolve()),
    }
    fields.update({f"python.{key}": value for key, value in python.items()})
    for number, path in enumerate(paths):
        fields[f"source.{number:02}.path"] = str(path.relative_to(root))
        fields[f"source.{number:02}.sha256"] = sha(path)
    return fields


def _identity(
    fields: dict[str, str], prefix: str, path: Path, mode: int | None
) -> dict[str, str]:
    observed = identity(path, mode)
    if any(fields.get(f"{prefix}.{key}") != value for key, value in observed.items()):
        raise ValueError(f"{prefix} identity mismatch")
    return observed


def hook_receipt(
    path: Path, hook: Path, sources: dict[str, str]
) -> tuple[bytes, dict[str, str]]:
    body = read_immutable(path, 65536)
    fields = decode_kv(body, 65536)
    source_path = str(PREFIX / f"{BASE}_capture_preload.c")
    source_number = SOURCE_NAMES.index(f"{BASE}_capture_preload.c")
    required = {
        "schema": "oi640-hook-compile-v1",
        "profile": "release",
        "target": "aarch64-linux-gnu",
        "source.bundle_sha256": sources["source.bundle_sha256"],
        "source.path": source_path,
        "source.sha256": sources[f"source.{source_number:02}.sha256"],
        "compile.argc": "12",
        "compile.argv.01": "-std=c11",
        "compile.argv.02": "-O2",
        "compile.argv.03": "-DNDEBUG",
        "compile.argv.04": "-fPIC",
        "compile.argv.05": "-shared",
        "compile.argv.06": "-Wl,-z,relro,-z,now",
        "compile.argv.08": "-o",
        "compile.argv.11": "-ldl",
        "binary.mode": "0555",
    }
    exact_keys = set(required) | {
        "compiler.version_sha256",
        "compile.env_sha256",
        "build.nonce",
    }
    exact_keys |= {f"compiler.{key}" for key in IDENTITY_KEYS}
    exact_keys |= {f"binary.{key}" for key in IDENTITY_KEYS}
    exact_keys |= {f"compile.argv.{number:02}" for number in range(12)}
    if set(fields) != exact_keys or any(
        fields.get(key) != value for key, value in required.items()
    ):
        raise ValueError("hook compile receipt contract")
    compiler = Path(fields.get("compiler.path", ""))
    source = Path(fields.get("compile.argv.07", ""))
    expected_source = Path(__file__).resolve().parent / f"{BASE}_capture_preload.c"
    if source.resolve(strict=True) != expected_source.resolve(strict=True):
        raise ValueError("hook source path")
    if (
        fields.get("compile.argv.00") != str(compiler.resolve(strict=True))
        or fields.get("compile.argv.09") != str(hook.resolve(strict=True))
        or fields.get("compile.argv.10") != "-Wl,--build-id=sha1"
    ):
        raise ValueError("hook compile invocation")
    _identity(fields, "compiler", compiler, None)
    observed = _identity(fields, "binary", hook, 0o555)
    header = subprocess.run(
        ["/usr/bin/readelf", "--file-header", str(hook)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    symbols = subprocess.run(
        ["/usr/bin/readelf", "--dyn-syms", "--wide", str(hook)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    exported = [
        line.split()
        for line in symbols.splitlines()
        if line.split()[-1:] == ["cuMemcpyDtoH_v2"]
    ]
    if (
        re.search(r"Type:\s+DYN \(Shared object file\)", header) is None
        or len(exported) != 1
        or len(exported[0]) < 8
        or exported[0][3:6] != ["FUNC", "GLOBAL", "DEFAULT"]
        or exported[0][6] == "UND"
    ):
        raise ValueError("hook ELF export contract")
    for key in ("compiler.version_sha256", "compile.env_sha256", "build.nonce"):
        if not HEX.fullmatch(fields.get(key, "")):
            raise ValueError(f"hook receipt {key}")
    if len(set(fields["build.nonce"])) < 8 or encode_kv(fields) != body:
        raise ValueError("hook receipt nonce/canonical")
    return body, observed


def _sha_manifest(path: Path, count: int, ptx: bool = False) -> str:
    body = read_immutable(path, 1 << 20)
    lines = body.decode().splitlines()
    pattern = re.compile(r"([0-9a-f]{64})  (\S+)")
    parsed = [pattern.fullmatch(line) for line in lines]
    names = [match.group(2) for match in parsed if match]
    if (
        len(lines) != count
        or len(parsed) != count
        or any(match is None for match in parsed)
        or len(set(names)) != count
        or (
            ptx
            and any(
                not name.startswith("./t0__") or not name.endswith(".ptx")
                for name in names
            )
        )
    ):
        raise ValueError("sealed SHA256 manifest contract")
    return sha(path)


def server_receipt(path: Path, server: Path) -> tuple[bytes, dict[str, str]]:
    body = read_immutable(path, 65536)
    fields = decode_kv(body, 65536)
    fixed = {
        "schema": "oi640-server-build-v1",
        "target.signature": TARGET,
        "kernel.ptx_count": "154",
        "kernel.override_count": "9",
        "kernel.target_ptx_set_count": "1",
        "kernel.v2_count": "0",
        "source.count": "1839",
        "selected.count": "25",
        "ptx.count": "154",
        "binary.mode": "0555",
    }
    fixed.update(B62)
    exact_keys = set(fixed) | {f"binary.{key}" for key in IDENTITY_KEYS}
    exact_keys |= {
        f"{prefix}.manifest.{key}"
        for prefix in ("source", "selected", "ptx")
        for key in ("path", "sha256")
    }
    exact_keys |= {"upstream.receipt.path", "upstream.receipt.sha256"}
    if set(fields) != exact_keys or any(
        fields.get(key) != value for key, value in fixed.items()
    ):
        raise ValueError("server build receipt target/census")
    observed = _identity(fields, "binary", server, 0o555)
    for prefix, count, ptx in (
        ("source", 1839, False),
        ("selected", 25, False),
        ("ptx", 154, True),
    ):
        artifact = Path(fields.get(f"{prefix}.manifest.path", ""))
        digest = _sha_manifest(artifact, count, ptx)
        if fields.get(f"{prefix}.manifest.sha256") != digest:
            raise ValueError(f"{prefix} manifest cross-link")
    upstream = Path(fields.get("upstream.receipt.path", ""))
    upstream_body = read_immutable(upstream, 65536)
    if (
        sha(upstream) != fields.get("upstream.receipt.sha256")
        or not upstream_body.startswith(b"# Flash-Next b62 sealed")
        or encode_kv(fields) != body
    ):
        raise ValueError("server upstream/canonical receipt")
    return body, observed
