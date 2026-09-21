# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from moe_w4a16_orig_i640_compact_prefill_capture_support import (
    HEX,
    decode_kv,
    encode_kv,
    identity,
    model_manifest,
    read_immutable,
    sha,
    source_bundle,
    validate_model,
    write_immutable,
)
from moe_w4a16_orig_i640_compact_prefill_provenance import (
    BASE,
    B62,
    PREFIX,
    SOURCE_NAMES,
    TARGET,
    _sha_manifest,
    server_receipt,
)


def create_model(args: argparse.Namespace) -> None:
    model = args.model.resolve(strict=True)
    fields = model_manifest(model)
    validate_model(fields, model)
    write_immutable(args.output, encode_kv(fields))


def create_build(args: argparse.Namespace) -> None:
    root = args.root.resolve(strict=True)
    sources = [path.resolve(strict=True) for path in args.source]
    if len(sources) != len(SOURCE_NAMES) or len(set(sources)) != len(sources):
        raise ValueError("exact complete unique source set required")
    for source in sources:
        source.relative_to(root)
    model_bytes = read_immutable(args.model_manifest)
    model_fields = decode_kv(model_bytes)
    validate_model(model_fields, args.model.resolve(strict=True))
    compile_fields = decode_kv(read_immutable(args.compile_receipt, 65536), 65536)
    bundle = source_bundle(sources, root)
    before = identity(args.binary, 0o555)
    required = {
        "schema": "oi640-compile-v1",
        "profile": "release",
        "target": "sm_121a",
        "binary.mode": "0555",
        "source.count": str(len(SOURCE_NAMES)),
        "source.bundle_sha256": bundle,
        "model.manifest.sha256": sha(args.model_manifest),
    }
    required.update({f"binary.{key}": value for key, value in before.items()})
    if any(compile_fields.get(key) != value for key, value in required.items()):
        raise ValueError("compile receipt contract")
    for key in (
        "compile.argv_sha256",
        "compile.env_sha256",
        "toolchain.sha256",
        "build.nonce",
    ):
        if key not in compile_fields or not HEX.fullmatch(compile_fields[key]):
            raise ValueError(f"missing {key}")
    if len(set(compile_fields["build.nonce"])) < 8:
        raise ValueError("weak build nonce")
    fields = {
        "schema": "oi640-build-v1",
        "profile": "release",
        "target": "sm_121a",
        "source.count": str(len(sources)),
        "source.bundle_sha256": bundle,
        "model.manifest.sha256": sha(args.model_manifest),
        "model.config.sha256": model_fields["config.sha256"],
        "model.index.sha256": model_fields["index.sha256"],
        "model.shard.count": model_fields["shard.count"],
    }
    fields.update({f"binary.{key}": value for key, value in before.items()})
    fields.update(
        {f"compile_receipt.{key}": value for key, value in compile_fields.items()}
    )
    for number, source in enumerate(sorted(sources)):
        fields[f"source.{number:02}.path"] = str(source.relative_to(root))
        fields[f"source.{number:02}.sha256"] = sha(source)
    if identity(args.binary, 0o555) != before:
        raise ValueError("binary drift while sealing")
    write_immutable(args.output, encode_kv(fields))


def create_hook(args: argparse.Namespace) -> None:
    directory = Path(__file__).resolve().parent
    root = directory.parents[3]
    sources = [directory / name for name in SOURCE_NAMES]
    compiler = args.compiler.resolve(strict=True)
    output = args.binary.resolve()
    receipt = args.output.resolve()
    if not output.is_absolute() or output.exists() or receipt.exists():
        raise ValueError("hook binary/receipt must be absolute and new")
    nonce = args.nonce
    if not HEX.fullmatch(nonce) or len(set(nonce)) < 8:
        raise ValueError("hook build nonce")
    source = directory / f"{BASE}_capture_preload.c"
    bundle, source_digest = source_bundle(sources, root), sha(source)
    argv = [str(compiler), "-std=c11", "-O2", "-DNDEBUG", "-fPIC", "-shared", "-Wl,-z,relro,-z,now", str(source), "-o", str(output), "-Wl,--build-id=sha1", "-ldl"]  # fmt: skip
    environment = {"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"}
    compiler_id = identity(compiler)
    version = subprocess.run(
        [str(compiler), "--version"], env=environment, check=True, capture_output=True
    )
    subprocess.run(argv, env=environment, check=True)
    if [source_bundle(sources, root), sha(source), identity(compiler)] != [bundle, source_digest, compiler_id]:  # fmt: skip
        raise ValueError("hook source drift during compile")
    os.chmod(output, 0o555)
    with output.open("rb") as handle:
        os.fsync(handle.fileno())
    binary_id = identity(output, 0o555)
    fields = {
        "schema": "oi640-hook-compile-v1",
        "profile": "release",
        "target": "aarch64-linux-gnu",
        "source.bundle_sha256": bundle,
        "source.path": str(PREFIX / source.name),
        "source.sha256": source_digest,
        "compiler.version_sha256": hashlib.sha256(
            version.stdout + version.stderr
        ).hexdigest(),
        "compile.argc": str(len(argv)),
        "compile.env_sha256": hashlib.sha256(
            json.dumps(environment, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "build.nonce": nonce,
    }
    fields.update({f"compiler.{key}": value for key, value in compiler_id.items()})
    fields.update({f"binary.{key}": value for key, value in binary_id.items()})
    fields.update(
        {f"compile.argv.{number:02}": value for number, value in enumerate(argv)}
    )
    write_immutable(receipt, encode_kv(fields))
    if identity(output, 0o555) != binary_id:
        raise ValueError("hook binary drift while sealing")


def _receipt_value(body: str, prefix: str) -> str:
    found = [
        line[len(prefix) :] for line in body.splitlines() if line.startswith(prefix)
    ]
    if len(found) != 1:
        raise ValueError(f"missing/duplicate sealed receipt line: {prefix}")
    return found[0]


def create_server(args: argparse.Namespace) -> None:
    upstream = args.build_receipt.resolve(strict=True)
    body = read_immutable(upstream, 65536).decode()
    base = upstream.parent
    binary = args.binary.resolve(strict=True)
    manifests = {
        "source": (args.source_manifest.resolve(strict=True), 1839, False),
        "selected": (args.selected_manifest.resolve(strict=True), 25, False),
        "ptx": (args.ptx_manifest.resolve(strict=True), 154, True),
    }
    if (
        binary != base / "release/spark"
        or _receipt_value(body, "- Target signature: ") != f"`{TARGET}`"
    ):
        raise ValueError("sealed server target/path")
    census = "154 PTX modules, 9 model-specific overrides, one `TargetPtxSet`, no v2 PTX entry"
    if _receipt_value(body, "- Kernel census: ") != census:
        raise ValueError("sealed server census")
    observed = identity(binary, 0o555)
    elf = f"`release/spark`, SHA256 `{observed['sha256']}`, size {int(observed['size']):,}, final mode 0555"
    if (
        _receipt_value(body, "- ELF: ") != elf
        or _receipt_value(body, "- ELF build ID: ") != f"`{observed['build_id']}`"
    ):
        raise ValueError("sealed server ELF receipt")
    expected_names = {"source": "source-manifest.sha256", "selected": "selected-source-manifest.sha256", "ptx": "ptx-manifest.sha256"}  # fmt: skip
    labels = {"source": "Source", "selected": "Selected-source", "ptx": "PTX"}
    for name, (path, count, is_ptx) in manifests.items():
        if path != base / expected_names[name]:
            raise ValueError("sealed manifest path")
        digest = _sha_manifest(path, count, is_ptx)
        expected = f"`{digest}` ({count:,} files)"
        if _receipt_value(body, f"- {labels[name]} manifest SHA256: ") != expected:
            raise ValueError("sealed manifest receipt cross-link")
    fields = {
        "schema": "oi640-server-build-v1",
        "target.signature": TARGET,
        "kernel.ptx_count": "154",
        "kernel.override_count": "9",
        "kernel.target_ptx_set_count": "1",
        "kernel.v2_count": "0",
        "source.count": "1839",
        "selected.count": "25",
        "ptx.count": "154",
        "upstream.receipt.path": str(upstream),
        "upstream.receipt.sha256": sha(upstream),
    }
    fields.update({f"binary.{key}": value for key, value in observed.items()})
    for name, (path, _, _) in manifests.items():
        fields[f"{name}.manifest.path"] = str(path)
        fields[f"{name}.manifest.sha256"] = sha(path)
    if any(fields.get(key) != value for key, value in B62.items()):
        raise ValueError("not the exact sealed b62 build")
    write_immutable(args.output.resolve(), encode_kv(fields))
    server_receipt(args.output.resolve(), binary)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Seal I640 model/build provenance manifests")  # fmt: skip
    commands = result.add_subparsers(required=True)
    model = commands.add_parser("model")
    model.add_argument("--model", type=Path, required=True)
    model.add_argument("--output", type=Path, required=True)
    model.set_defaults(run=create_model)
    build = commands.add_parser("build")
    for name in "root model model-manifest compile-receipt binary output".split():
        build.add_argument(f"--{name}", type=Path, required=True)
    build.add_argument("--source", action="append", type=Path, required=True)
    build.set_defaults(run=create_build)
    hook = commands.add_parser("hook")
    for name in ("compiler", "binary", "output"):
        hook.add_argument(f"--{name}", type=Path, required=True)
    hook.add_argument("--nonce", required=True)
    hook.set_defaults(run=create_hook)
    server = commands.add_parser("server")
    for name in "build-receipt source-manifest selected-manifest ptx-manifest binary output".split():
        server.add_argument(f"--{name}", type=Path, required=True)
    server.set_defaults(run=create_server)
    return result


def main() -> int:
    args = parser().parse_args()
    args.run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
