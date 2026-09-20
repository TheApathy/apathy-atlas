# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
MODEL = Path("/home/flocka/models/Qwen3.8-Flash-Next-NVFP4-Offload")
STEM = "moe_w4a16_orig_i640_compact_prefill"
FILES = [
    f"{STEM}.cuh",
    f"{STEM}.cu",
    f"{STEM}_plan.cuh",
    f"{STEM}_gemm.cuh",
    f"{STEM}_microgate.cu",
    f"{STEM}_microgate_io.cuh",
    f"{STEM}_microgate_buffers.cuh",
    f"{STEM}_microgate_cases.cuh",
    f"{STEM}_microgate_timing.cuh",
    f"{STEM}_microgate_manifest.cuh",
    f"{STEM}_microgate_provenance.cuh",
    f"{STEM}_capture_preload.c",
    f"{STEM}_capture_support.py",
    f"{STEM}_capture.py",
    f"{STEM}_capture_run.py",
    f"{STEM}_manifest.py",
    f"{STEM}_provenance.py",
    f"{STEM}_static_support.py",
    f"{STEM}_static_source_cases.py",
    f"test_{STEM}_static.py",
    f"test_{STEM}_capture.py",
    f"test_{STEM}_provenance.py",
]


def text(name: str) -> str:
    return (HERE / name).read_text()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def offsets_from_counts(counts: list[int]) -> list[int]:
    out = [0]
    for count in counts:
        out.append(out[-1] + count)
    return out


def planned_items(counts: list[int]) -> list[tuple[int, int]]:
    return [
        (expert, tile)
        for expert, count in enumerate(counts)
        for tile in range((count + 63) // 64)
    ]


def safetensor_header(path: Path) -> dict:
    with path.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        return json.loads(handle.read(size))
