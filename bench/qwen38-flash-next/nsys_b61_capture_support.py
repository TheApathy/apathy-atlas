# SPDX-License-Identifier: AGPL-3.0-only
"""Locked inputs and fail-closed checks for the b61 Nsight capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
from pathlib import Path
from typing import Any

BINARY = Path("/var/tmp/atlas-flash-k16-qual-tVMYwsUZ/release/spark")
MODEL = Path("/home/flocka/models/Qwen3.8-Flash-Next-NVFP4-Offload")
ENDPOINT = "http://127.0.0.1:8998/v1/chat/completions"
PROMPT = (
    "Write a complete MinHeap class in Python with insert, extract_min, and heapify, "
    "then explain the complexity of each method."
)
EMPTY_SHA = hashlib.sha256(b"").hexdigest()
LOCKED_FILES = {
    BINARY: "8e24857c5ceee7043ca9b43e690f845935e0f0346665a2ec6e55afafaca144e2",
    BINARY.parent.parent
    / "BUILD_RECEIPT.md": "19c10e4b4f33836622a80cc63b6874feb8f2116f1e3a87c015725a236af0883e",
    BINARY.parent.parent
    / "source-manifest.sha256": "72a70d803cfe4ac5b38e8d6ea8d5c0cf9daffb790d5d39f8ee1506bdc324515b",
    BINARY.parent.parent
    / "ptx-manifest.sha256": "225f0114492f9327966e032171ce2db86cd6bd9827fc6b4f79d45d7107b21533",
    BINARY.parent.parent
    / "selected-source-manifest.sha256": "13de36f71b0c09bef2d0168d682f1fc962622b795cd16858855b13620582d018",
}
LOCKED_LINES = {
    "source-manifest.sha256": 1836,
    "ptx-manifest.sha256": 154,
    "selected-source-manifest.sha256": 18,
}
TARGET_ENV = {
    "ATLAS_DFLASH_ASYNC": "0",
    "ATLAS_PLE_CACHE_MB": "0",
    "LANG": "C.UTF-8",
    "LD_LIBRARY_PATH": "/usr/local/cuda-13.0/targets/sbsa-linux/lib",
    "PATH": "/usr/local/cuda-13.0/bin:/usr/local/bin:/usr/bin:/bin",
}
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NSYS = Path("/opt/nvidia/nsight-systems/2025.3.2/target-linux-sbsa-armv8/nsys")
PORT = 8998


def server_argv() -> list[str]:
    options = {
        "--model-from-path": str(MODEL),
        "--model-name": "qwen3.8-flash-next",
        "--kernel-target": "qwen3.8-flash-next",
        "--port": str(PORT),
        "--max-seq-len": "2048",
        "--max-prefill-tokens": "512",
        "--max-num-seqs": "1",
        "--max-batch-size": "1",
        "--ssm-cache-slots": "0",
        "--kv-cache-dtype": "bf16",
        "--gpu-memory-utilization": "0.90",
        "--oom-guard-mb": "4096",
        "--request-timeout": "300",
    }
    return (
        [str(BINARY), "serve"]
        + [value for item in options.items() for value in item]
        + ["--no-tui"]
    )


def launch_command(session: str) -> list[str]:
    command = [
        str(NSYS),
        "launch",
        f"--session-new={session}",
        "--cuda-graph-trace=node",
        "--inherit-environment=false",
        "--show-output=true",
        "--trace=cuda,nvtx",
        "--sample=none",
        "--cpuctxsw=none",
    ]
    command += [f"--env-var={key}={value}" for key, value in sorted(TARGET_ENV.items())]
    return command + server_argv()


def start_command(session: str, trace: Path) -> list[str]:
    return [
        str(NSYS),
        "start",
        f"--session={session}",
        "--export=sqlite",
        "--force-overwrite=false",
        f"--output={trace}",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture warmed b61 Nsight evidence")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-reservation-nonce", required=True)
    parser.add_argument(
        "--execute", required=True, choices=["SOURCE_FREE_B61_NSYS_CAPTURE"]
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{64}", args.gpu_reservation_nonce):
        parser.error("--gpu-reservation-nonce must be fresh lowercase 64-hex")
    return args


def nsys_capabilities(nsys: Path) -> dict[str, str]:
    require_regular(nsys)
    probes = {
        "version": ([str(nsys), "--version"], 0),
        "launch": ([str(nsys), "launch", "--help"], 0),
        "start": ([str(nsys), "start", "--help"], 0),
        "stop": ([str(nsys), "stop", "--help"], 0),
        "cancel": ([str(nsys), "cancel", "--help"], 0),
        "shutdown": ([str(nsys), "shutdown", "--help"], 0),
        "reports": ([str(nsys), "stats", "--help=reports"], 0),
        "report_catalog": ([str(nsys), "stats", "--help-reports"], 1),
    }
    text = {}
    for key, (command, expected_rc) in probes.items():
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if result.returncode != expected_rc:
            raise RuntimeError(
                f"Nsight capability probe failed: {key} rc={result.returncode}"
            )
        text[key] = result.stdout
    required = {
        "version": ("2025.3",),
        "launch": (
            "--session-new=",
            "--cuda-graph-trace=",
            "node",
            "--inherit-environment=",
            "--trace=",
            "--sample=",
            "--cpuctxsw=",
        ),
        "start": ("--session=", "--export=", "--force-overwrite=", "--output="),
        "stop": ("--session=",),
        "cancel": ("--session=",),
        "shutdown": ("--session=", "--kill="),
        "reports": ("cuda_gpu_trace",),
        "report_catalog": ("cuda_gpu_trace", "cuda_gpu_kern_gb_sum"),
    }
    for key, needles in required.items():
        if any(needle not in text[key] for needle in needles):
            raise RuntimeError(f"insufficient Nsight capability: {key}")
    return {
        key: hashlib.sha256(value.encode()).hexdigest() for key, value in text.items()
    }


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_regular(path: Path) -> os.stat_result:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise RuntimeError(f"not a retained regular file: {path}")
    return info


def attest_locked_files() -> dict[str, dict[str, Any]]:
    result = {}
    for path, expected in LOCKED_FILES.items():
        info = require_regular(path)
        actual = sha_file(path)
        if actual != expected:
            raise RuntimeError(f"locked hash drift: {path}")
        lines = None if path == BINARY else sum(1 for _ in path.open("rb"))
        result[str(path)] = {
            "sha256": actual,
            "size": info.st_size,
            "mode": stat.S_IMODE(info.st_mode),
            "lines": lines,
        }
    binary = result[str(BINARY)]
    if binary["size"] != 38_858_456 or binary["mode"] != 0o555:
        raise RuntimeError("sealed binary size/mode drift")
    root = BINARY.parent.parent
    if any(
        result[str(root / name)]["lines"] != count
        for name, count in LOCKED_LINES.items()
    ):
        raise RuntimeError("sealed manifest line-count drift")
    if any(
        item["mode"] != 0o444 for name, item in result.items() if name != str(BINARY)
    ):
        raise RuntimeError("sealed receipt/manifest mode drift")
    return result


def request_body(max_tokens: int) -> dict[str, Any]:
    return {
        "model": "qwen3.8-flash-next",
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": PROMPT}],
        "reasoning_effort": "none",
    }
