#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""CPU/source and, when authorized, direct-SM121 compile checks for raw v2."""
from __future__ import annotations

import hashlib
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import unittest


HERE = pathlib.Path(__file__).resolve().parent
GB10 = HERE.parents[1]
HEADER = HERE / "moe_w4a16_routed_k16_v2.cuh"
CUDA = HERE / "moe_w4a16_routed_k16_v2.cu"
GATE = HERE / "moe_w4a16_routed_k16_v2_microgate.cu"
PLAN = HERE / "moe_w4a16_routed_k16_v2_plan.cuh"
COMPUTE = HERE / "moe_w4a16_routed_k16_v2_compute.cuh"
GATE_FIXTURE = HERE / "moe_w4a16_routed_k16_v2_microgate_fixture.cuh"
GATE_VALIDATION = HERE / "moe_w4a16_routed_k16_v2_microgate_validation.cuh"
GATE_TIMING = HERE / "moe_w4a16_routed_k16_v2_microgate_timing.cuh"
STATIC_MAIN = HERE / "test_moe_w4a16_routed_k16_v2_static.py"
STATIC_SUPPORT = HERE / "moe_w4a16_routed_k16_v2_static_support.py"
STATIC_CASES = HERE / "moe_w4a16_routed_k16_v2_static_source_cases.py"
CUDA_SOURCES = (CUDA, PLAN, COMPUTE)
GATE_SOURCES = (GATE, GATE_FIXTURE, GATE_VALIDATION, GATE_TIMING)
STATIC_SOURCES = (STATIC_MAIN, STATIC_SUPPORT, STATIC_CASES)
PARENT = HERE / "moe_w4a16_exact_k16.cu"
PARENT_GATE = HERE / "moe_w4a16_exact_k16_microgate.cu"
PARENTS = (
    GB10 / "common" / "moe_shared_expert_fused_batch3.cu",
    GB10 / "common" / "moe_shared_expert_fused.cu",
    GB10 / "common" / "moe_expert_gemv.cu",
)
SOURCE_ONLY = os.environ.get("ATLAS_V2_SOURCE_ONLY") == "1"

ROWS, HIDDEN, INTER, EXPERTS, TOP_K = 16, 2560, 640, 512, 10
ROUTES, WIDTH, DESCRIPTORS = ROWS * TOP_K, 4, 40


def nvcc() -> str:
    for item in (
        shutil.which("nvcc"),
        "/usr/local/cuda-13.0/bin/nvcc",
        "/usr/local/cuda/bin/nvcc",
    ):
        if item and pathlib.Path(item).is_file():
            return item
    raise unittest.SkipTest("CUDA 13 nvcc unavailable")


def digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance_defines() -> dict[str, str]:
    values = {
        "FLASH_NEXT_V2_HEADER_SHA256": digest(HEADER),
        "FLASH_NEXT_V2_CUDA_SHA256": digest(CUDA),
        "FLASH_NEXT_V2_MICROGATE_SHA256": digest(GATE),
        "FLASH_NEXT_V2_PLAN_SHA256": digest(PLAN),
        "FLASH_NEXT_V2_COMPUTE_SHA256": digest(COMPUTE),
        "FLASH_NEXT_V2_MICROGATE_FIXTURE_SHA256": digest(GATE_FIXTURE),
        "FLASH_NEXT_V2_MICROGATE_VALIDATION_SHA256": digest(GATE_VALIDATION),
        "FLASH_NEXT_V2_MICROGATE_TIMING_SHA256": digest(GATE_TIMING),
        "FLASH_NEXT_V2_EXACT_PARENT_SHA256": digest(PARENT),
        "FLASH_NEXT_V2_EXACT_PARENT_GATE_SHA256": digest(PARENT_GATE),
        "FLASH_NEXT_V2_BATCH_PARENT_SHA256": digest(PARENTS[0]),
        "FLASH_NEXT_V2_SERIAL_PARENT_SHA256": digest(PARENTS[1]),
        "FLASH_NEXT_V2_BLEND_PARENT_SHA256": digest(PARENTS[2]),
    }
    values.update({
        "FLASH_NEXT_EXACT_SOURCE_SHA256": values["FLASH_NEXT_V2_EXACT_PARENT_SHA256"],
        "FLASH_NEXT_MICROGATE_SOURCE_SHA256": values["FLASH_NEXT_V2_EXACT_PARENT_GATE_SHA256"],
        "FLASH_NEXT_PARENT_BATCH_SHA256": values["FLASH_NEXT_V2_BATCH_PARENT_SHA256"],
        "FLASH_NEXT_PARENT_SERIAL_SHA256": values["FLASH_NEXT_V2_SERIAL_PARENT_SHA256"],
        "FLASH_NEXT_PARENT_BLEND_SHA256": values["FLASH_NEXT_V2_BLEND_PARENT_SHA256"],
    })
    return values


def command(
    *sources: pathlib.Path,
    output: pathlib.Path,
    cubin: bool,
    definitions: dict[str, str] | None = None,
) -> list[str]:
    defines = [f'-D{name}="{value}"' for name, value in sorted((definitions or {}).items())]
    args = [
        nvcc(), "-std=c++17", "-O3", "--use_fast_math", "--fmad=false", "-lineinfo",
        "-gencode", "arch=compute_121a,code=sm_121a", *defines, *map(str, sources),
        "-o", str(output), "--resource-usage",
    ]
    if cubin:
        args.insert(1, "--cubin")
    return args


def fixed40_plan(ids: list[int]) -> list[tuple[str, int, tuple[int, ...]]]:
    if len(ids) != ROUTES or any(expert < 0 or expert >= EXPERTS for expert in ids):
        raise ValueError("hostile route")
    leaders: list[int] = []
    tails: list[int] = []
    for slot, expert in enumerate(ids):
        same = [i for i, value in enumerate(ids) if value == expert]
        rank = same.index(slot)
        if rank % WIDTH == 0 and rank + WIDTH <= len(same):
            leaders.append(slot)
        if rank >= len(same) - len(same) % WIDTH:
            tails.append(slot)
    descriptors = [
        ("homogeneous", ids[leader], tuple(
            slot for slot in range(leader, ROUTES) if ids[slot] == ids[leader]
        )[:WIDTH])
        for leader in leaders
    ]
    descriptors.extend(
        ("heterogeneous", 0xFFFF, tuple(tails[i : i + WIDTH]))
        for i in range(0, len(tails), WIDTH)
    )
    if len(descriptors) != DESCRIPTORS:
        raise AssertionError("fixed40 partition drift")
    return descriptors


def mixed_tails() -> list[int]:
    ids: list[int] = []
    expert = 0
    while len(ids) < ROUTES:
        ids.extend([expert] * min(expert % WIDTH + 1, ROUTES - len(ids)))
        expert += 1
    return ids


def route_case(divisor: int) -> list[int]:
    return [slot % divisor if divisor else 7 for slot in range(ROUTES)]




def read_sources(paths: tuple[pathlib.Path, ...]) -> str:
    return "".join(path.read_text() for path in paths)


ALL_SOURCE_FILES = (HEADER, *CUDA_SOURCES, *GATE_SOURCES, *STATIC_SOURCES)
