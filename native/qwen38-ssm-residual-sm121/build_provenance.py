#!/usr/bin/python3
# SPDX-License-Identifier: AGPL-3.0-only
"""One-shot, provenance-locked CUDA build for the private residual ABI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NVCC = Path("/usr/local/cuda-13.0/bin/nvcc")
CXX = Path("/usr/bin/c++").resolve()
CUDART = Path("/usr/local/cuda-13.0/targets/sbsa-linux/lib/libcudart.so.13.0.96")
CUDA_DRIVER = Path("/usr/lib/aarch64-linux-gnu/libcuda.so.580.126.09")
PINNED_NVCC_SHA256 = "fbb111f057786ddd10ba723d993cc7dd43abf978b6baa32fedd3c9d806dc79e1"
PINNED_CXX_SHA256 = "fc02363794280f404c6ca6f5da1c8fe469be902e9de140d35d8573bb3393f53b"
PINNED_CUDART_SHA256 = "7bdba2b5b08cbdc85203c41cc94598adedb1bcfea7cb574ca693ac73599e4e63"
PINNED_CUDA_DRIVER_SHA256 = "477f19f28a791edb73f8e4c6ee27b7e2dd13300a7dabacd6f40e3eee3916d14f"
BUILD_ENV = {"PATH": "/usr/bin:/bin", "LC_ALL": "C"}

TOOL_PINS = {
    str(NVCC): PINNED_NVCC_SHA256,
    str(CXX): PINNED_CXX_SHA256,
    str(CUDART): PINNED_CUDART_SHA256,
    str(CUDA_DRIVER): PINNED_CUDA_DRIVER_SHA256,
}
REQUIRED_INPUTS = [
    "src/atlas_qwen38_ssm_residual.cu",
    "src/atlas_qwen38_ssm_residual_admission.cuh",
    "include/atlas_qwen38_ssm_residual.h",
    "exports.map",
    "build.sh",
    "build_provenance.py",
]


def valid_sha256(value: str) -> bool:
    return len(value) == 64 and value == value.lower() and all(c in "0123456789abcdef" for c in value)


def digest_fd(fd: int) -> str:
    result = hashlib.sha256()
    offset = 0
    while chunk := os.pread(fd, 1 << 20, offset):
        result.update(chunk)
        offset += len(chunk)
    return result.hexdigest()


@dataclass(frozen=True)
class Stamp:
    device: int
    inode: int
    mode: int
    size: int
    sha256: str


def stamp_fd(fd: int) -> Stamp:
    value = os.fstat(fd)
    if not stat.S_ISREG(value.st_mode):
        raise RuntimeError("held input is not a regular file")
    return Stamp(value.st_dev, value.st_ino, value.st_mode, value.st_size, digest_fd(fd))


class Held:
    def __init__(self, path: Path, expected: str | None = None):
        self.path = path.absolute()
        if self.path.resolve(strict=True) != self.path:
            raise RuntimeError(f"noncanonical or symlinked identity path: {path}")
        self.fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW)
        self.initial = stamp_fd(self.fd)
        self._verify_path()
        if expected is not None:
            if not valid_sha256(expected) or self.initial.sha256 != expected:
                raise RuntimeError(f"identity mismatch: {path}")

    def _verify_path(self) -> None:
        current = self.path.stat()
        if (current.st_dev, current.st_ino, current.st_mode, current.st_size) != (
            self.initial.device,
            self.initial.inode,
            self.initial.mode,
            self.initial.size,
        ):
            raise RuntimeError(f"path identity changed: {self.path}")

    def verify_unchanged(self) -> None:
        if stamp_fd(self.fd) != self.initial:
            raise RuntimeError(f"held bytes changed: {self.path}")
        self._verify_path()

    def copy_to(self, destination: Path) -> str:
        fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            offset = 0
            while chunk := os.pread(self.fd, 1 << 20, offset):
                view = memoryview(chunk)
                while view:
                    written = os.write(fd, view)
                    view = view[written:]
                offset += len(chunk)
            os.fsync(fd)
        finally:
            os.close(fd)
        os.chmod(destination, 0o400)
        return self.initial.sha256

    def close(self) -> None:
        os.close(self.fd)


def ensure_absent(*paths: Path) -> None:
    if any(path.exists() or path.is_symlink() for path in paths):
        raise RuntimeError("one-shot build output already exists")


def file_receipt(path: Path) -> dict[str, int | str]:
    held = Held(path)
    try:
        result = {"sha256": held.initial.sha256, "bytes": held.initial.size, "mode": held.initial.mode}
    finally:
        held.close()
    return result


def validate_authority(record: object) -> dict:
    if not isinstance(record, dict) or set(record) != {"schema", "inputs", "tools", "environment"}:
        raise RuntimeError("scheduler manifest fields mismatch")
    inputs = record["inputs"]
    if record["schema"] != "qwen38-ssm-residual-scheduler-v1" or not isinstance(inputs, dict):
        raise RuntimeError("scheduler manifest schema mismatch")
    if set(inputs) != set(REQUIRED_INPUTS) or not all(valid_sha256(value) for value in inputs.values()):
        raise RuntimeError("scheduler input authority mismatch")
    if record["tools"] != TOOL_PINS or record["environment"] != BUILD_ENV:
        raise RuntimeError("scheduler tool or environment authority mismatch")
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("scheduler_manifest")
    parser.add_argument("scheduler_manifest_sha256")
    parser.add_argument("output_dir")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    authority = Held(Path(args.scheduler_manifest), args.scheduler_manifest_sha256)
    authority_stat = os.fstat(authority.fd)
    if authority_stat.st_nlink != 1 or authority_stat.st_mode & 0o222:
        raise RuntimeError("scheduler manifest must be single-link and immutable")
    if authority.initial.size > 65536:
        raise RuntimeError("scheduler manifest is too large")
    raw_authority = os.pread(authority.fd, authority.initial.size, 0)
    if len(raw_authority) != authority.initial.size:
        raise RuntimeError("short scheduler manifest read")
    authority_record = validate_authority(json.loads(raw_authority))
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute() or output_dir.parent.resolve(strict=True) != output_dir.parent:
        raise RuntimeError("output_dir must be absolute under an existing canonical parent")
    os.mkdir(output_dir, 0o700)
    inputs_dir = output_dir / "inputs"
    os.mkdir(inputs_dir, 0o700)
    held_inputs = [Held(ROOT / relative, authority_record["inputs"][relative]) for relative in REQUIRED_INPUTS]
    held_copies: list[Held] = []
    tools = [
        Held(NVCC, TOOL_PINS[str(NVCC)]),
        Held(CXX, TOOL_PINS[str(CXX)]),
        Held(CUDART, TOOL_PINS[str(CUDART)]),
        Held(CUDA_DRIVER, TOOL_PINS[str(CUDA_DRIVER)]),
    ]
    try:
        copied = {}
        for relative, held in zip(REQUIRED_INPUTS, held_inputs, strict=True):
            destination = inputs_dir / Path(relative).name
            digest = held.copy_to(destination)
            held_copy = Held(destination, digest)
            held_copies.append(held_copy)
            copied[relative] = {"sha256": digest, "bytes": held_copy.initial.size, "path": str(destination)}
        authority_copy = inputs_dir / "SCHEDULER_MANIFEST.json"
        authority.copy_to(authority_copy)
        held_copies.append(Held(authority_copy, authority.initial.sha256))
        source = inputs_dir / "atlas_qwen38_ssm_residual.cu"
        object_file = output_dir / "atlas_qwen38_ssm_residual.o"
        library = output_dir / "libatlas_qwen38_ssm_residual.so"
        receipt = output_dir / "BUILD_RECEIPT.json"
        ensure_absent(object_file, library, receipt)
        commands = [
            [str(NVCC), "-std=c++17", "-O3", "--fmad=false", "-gencode=arch=compute_121a,code=sm_121a", f"-ccbin={CXX}", "-Xcompiler=-fPIC,-fvisibility=hidden", f"-I{inputs_dir}", "-c", str(source), "-o", str(object_file)],
            [str(CXX), "-shared", str(object_file), str(CUDART), str(CUDA_DRIVER), "-Wl,-soname,libatlas_qwen38_ssm_residual.so.0", f"-Wl,--version-script={inputs_dir / 'exports.map'}", "-o", str(library)],
        ]
        for command in commands:
            subprocess.run(command, cwd=output_dir, env=BUILD_ENV, check=True)
        for held in [authority, *held_inputs, *held_copies, *tools]:
            held.verify_unchanged()
        for path in [object_file, library]:
            os.chmod(path, 0o400)
        outputs = {str(path): file_receipt(path) for path in [object_file, library]}
        tool_receipts = {str(held.path): held.initial.sha256 for held in tools}
        record = {"schema": "qwen38-ssm-residual-build-v3", "scheduler_manifest": file_receipt(authority_copy), "inputs": copied, "tools": tool_receipts, "commands": commands, "environment": BUILD_ENV, "outputs": outputs}
        with receipt.open("x", encoding="utf-8") as handle:
            json.dump(record, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(receipt, 0o400)
    finally:
        for held in [authority, *held_inputs, *held_copies, *tools]:
            held.close()


if __name__ == "__main__":
    main()
