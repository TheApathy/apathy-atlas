#!/usr/bin/env python3
"""Build or verify a content-addressed DeepSeek DFlash2 Vast manifest."""

import argparse
import hashlib
import json
import os
import pathlib
import tempfile


EXCLUDED_PARTS = {".git", "__pycache__", ".pytest_cache", ".ruff_cache"}


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(path: pathlib.Path):
    if path.is_file():
        yield pathlib.Path(), path
        return
    for item in sorted(path.rglob("*")):
        if any(part in EXCLUDED_PARTS for part in item.relative_to(path).parts):
            continue
        if item.is_symlink():
            raise RuntimeError(f"bundle inputs may not contain symlinks: {item}")
        if item.is_file():
            yield item.relative_to(path), item


def parse_entry(value: str) -> tuple[str, pathlib.Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("entry must be NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name or "/" in name or name in {".", ".."}:
        raise argparse.ArgumentTypeError(f"invalid bundle entry name: {name!r}")
    unresolved = pathlib.Path(raw_path).expanduser()
    if unresolved.is_symlink():
        raise argparse.ArgumentTypeError(
            f"bundle entry itself may not be a symlink: {unresolved}"
        )
    path = unresolved.resolve()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"bundle entry does not exist: {path}")
    return name, path


def write_json_atomic(output: pathlib.Path, payload: dict) -> None:
    """Replace a manifest only after its complete JSON is durable."""
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = pathlib.Path(handle.name)
        try:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def build(entries: list[tuple[str, pathlib.Path]], output: pathlib.Path) -> None:
    names = [name for name, _ in entries]
    if len(names) != len(set(names)):
        raise RuntimeError("bundle entry names must be unique")
    files = []
    total = 0
    for name, source in sorted(entries):
        for relative, path in iter_files(source):
            size = path.stat().st_size
            files.append(
                {
                    "path": (pathlib.Path(name) / relative).as_posix(),
                    "bytes": size,
                    "sha256": sha256(path),
                }
            )
            total += size
    payload = {
        "format": "atlas-deepseek-dflash2-bundle-v1",
        "files": files,
        "file_count": len(files),
        "total_bytes": total,
    }
    write_json_atomic(output, payload)
    print(f"bundle manifest OK: files={len(files)} bytes={total} output={output}")


def verify(manifest: pathlib.Path, root: pathlib.Path) -> None:
    if root.is_symlink():
        raise RuntimeError(f"bundle root may not be a symlink: {root}")
    root_resolved = root.resolve()
    payload = json.loads(manifest.read_text())
    if payload.get("format") != "atlas-deepseek-dflash2-bundle-v1":
        raise RuntimeError("unsupported bundle manifest format")
    files = payload.get("files")
    if not isinstance(files, list):
        raise RuntimeError("bundle manifest files must be a list")
    if payload.get("file_count") != len(files):
        raise RuntimeError(
            f"manifest file_count={payload.get('file_count')!r}, "
            f"actual entries={len(files)}"
        )

    declared_total = 0
    declared_paths = set()
    checked = 0
    for entry in files:
        if not isinstance(entry, dict):
            raise RuntimeError("bundle manifest entries must be objects")
        if set(entry) != {"path", "bytes", "sha256"}:
            raise RuntimeError(f"invalid bundle manifest entry keys: {entry!r}")
        if not isinstance(entry["path"], str) or not entry["path"]:
            raise RuntimeError(f"invalid manifest path: {entry['path']!r}")
        if not isinstance(entry["bytes"], int) or entry["bytes"] < 0:
            raise RuntimeError(f"invalid byte count for {entry['path']!r}")
        if (
            not isinstance(entry["sha256"], str)
            or len(entry["sha256"]) != 64
            or any(char not in "0123456789abcdef" for char in entry["sha256"])
        ):
            raise RuntimeError(f"invalid sha256 for {entry['path']!r}")
        relative = pathlib.PurePosixPath(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or "." in relative.parts:
            raise RuntimeError(f"unsafe manifest path: {relative}")
        normalized = relative.as_posix()
        if normalized in declared_paths:
            raise RuntimeError(f"duplicate bundle path: {normalized}")
        declared_paths.add(normalized)
        declared_total += entry["bytes"]
        path = root
        for part in relative.parts:
            path = path / part
            if path.is_symlink():
                raise RuntimeError(f"bundle path may not contain a symlink: {path}")
        try:
            path.resolve().relative_to(root_resolved)
        except ValueError as exc:
            raise RuntimeError(f"bundle path escapes root: {path}") from exc
        if not path.is_file():
            raise RuntimeError(f"missing bundle file: {path}")
        if path.stat().st_size != entry["bytes"]:
            raise RuntimeError(f"size mismatch: {path}")
        if sha256(path) != entry["sha256"]:
            raise RuntimeError(f"sha256 mismatch: {path}")
        checked += 1
    if payload.get("total_bytes") != declared_total:
        raise RuntimeError(
            f"manifest total_bytes={payload.get('total_bytes')!r}, "
            f"declared entries={declared_total}"
        )
    print(f"bundle verify OK: files={checked} bytes={declared_total} root={root}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument(
        "--entry", action="append", type=parse_entry, required=True
    )
    build_parser.add_argument("--output", type=pathlib.Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--manifest", type=pathlib.Path, required=True)
    verify_parser.add_argument("--root", type=pathlib.Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        build(args.entry, args.output)
    else:
        verify(args.manifest, args.root)


if __name__ == "__main__":
    main()
