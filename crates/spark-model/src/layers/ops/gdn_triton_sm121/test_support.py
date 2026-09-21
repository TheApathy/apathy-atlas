# SPDX-License-Identifier: AGPL-3.0-only
"""Pure CPU helpers for the unregistered frozen-Triton loader tests."""

import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
NATIVE = REPO / "native/gdn-sglang-c143"
MANIFEST = NATIVE / "frozen_triton_c143_manifest.json"
RUST = (
    "mod.rs",
    "authority.rs",
    "digest.rs",
    "manifest.rs",
    "types.rs",
    "ffi.rs",
    "loader.rs",
    "launch.rs",
)
GATE_SOURCES = (
    "frozen_triton_c143_artifact.py",
    "frozen_triton_c143_attest.py",
    "frozen_triton_c143_authorization.py",
    "frozen_triton_c143_constants.py",
    "frozen_triton_c143_gate.py",
    "frozen_triton_c143_io.py",
    "frozen_triton_c143_layout.py",
    "frozen_triton_c143_manifest_policy.py",
    "frozen_triton_c143_provenance.py",
    "frozen_triton_c143_receipt.py",
    "frozen_triton_c143_receipt_metrics.py",
)
EXECUTOR_SOURCES = (
    "run_frozen_triton_c143.py",
    "frozen_triton_executor/__init__.py",
    "frozen_triton_executor/contract.py",
    "frozen_triton_executor/verified_loader.py",
    "frozen_triton_executor/cuda_driver.py",
    "frozen_triton_executor/buffers.py",
    "frozen_triton_executor/launch.py",
    "frozen_triton_executor/references.py",
    "frozen_triton_executor/case.py",
    "frozen_triton_executor/timing.py",
    "frozen_triton_executor/evidence.py",
    "frozen_triton_executor/receipt.py",
    "frozen_triton_executor/publication.py",
    "frozen_triton_executor/main.py",
)
U64_MAX = (1 << 64) - 1


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def rust_constant(source: str, name: str) -> str:
    match = re.search(rf"pub const {name}: &str =\s*\n?\s*\"([0-9a-z.\-]+)\"", source)
    if not match:
        raise AssertionError(f"missing Rust constant {name}")
    return match.group(1)


def source_bundle(names: tuple[str, ...]) -> str:
    records = {}
    for name in names:
        raw = (NATIVE / name).read_bytes()
        records[name] = [sha(raw), len(raw)]
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return sha(canonical)


def checked_mul(*values: int) -> int:
    value = 1
    for factor in values:
        if factor < 0 or (factor != 0 and value > U64_MAX // factor):
            raise OverflowError
        value *= factor
    return value


def layout(m: int) -> dict:
    if m not in (2_079, 8_192):
        raise ValueError("unsupported M")
    nt = (m + 63) // 64
    specs = (
        ("q_bf16", checked_mul(m, 16, 128, 2)),
        ("k_bf16", checked_mul(m, 16, 128, 2)),
        ("v_bf16", checked_mul(m, 48, 128, 2)),
        ("log_gate_f32", checked_mul(m, 48, 4)),
        ("beta_f32", checked_mul(m, 48, 4)),
        ("state_hvk_f32", checked_mul(48, 128, 128, 4)),
        ("g_cumsum_f32", checked_mul(m, 48, 4)),
        ("A_bf16", checked_mul(m, 48, 64, 2)),
        ("w_bf16", checked_mul(m, 48, 128, 2)),
        ("u_bf16", checked_mul(m, 48, 128, 2)),
        ("h_bf16", checked_mul(nt, 48, 128, 128, 2)),
        ("v_new_bf16", checked_mul(m, 48, 128, 2)),
        ("cu_seqlens_i32", 8),
        ("state_index_i32", 4),
        ("chunk_indices_i32", checked_mul(nt, 2, 4)),
        ("chunk_offsets_i64", 16),
    )
    segments, cursor = [], 0
    for name, size in specs:
        cursor = (cursor + 255) & ~255
        segments.append([name, cursor, size])
        cursor += size
    total = (cursor + 255) & ~255
    return {
        "nt": nt,
        "total_bytes": total,
        "external_bytes": {
            "atlas_qkv_bf16": checked_mul(m, 10_240, 2),
            "atlas_gate_beta_f32": checked_mul(m, 96, 4),
            "atlas_state_hkv_f32": checked_mul(48, 128, 128, 4),
            "output_bf16": checked_mul(m, 48, 128, 2),
        },
        "segments": segments,
    }


def valid_regions(m: int) -> dict[str, list[int]]:
    contract = layout(m)
    required = {**contract["external_bytes"], "workspace": contract["total_bytes"]}
    cursor, records = 0x1000, {}
    for name, size in required.items():
        alignment = 256 if name == "workspace" else 16
        cursor = (cursor + alignment - 1) & -alignment
        records[name] = [cursor, size]
        cursor += size + 4096
    return records


def validate_regions(m: int, records: dict[str, list[int]], stream: int) -> None:
    if stream == 0:
        raise ValueError("default stream")
    contract = layout(m)
    required = {**contract["external_bytes"], "workspace": contract["total_bytes"]}
    if records.keys() != required.keys():
        raise ValueError("region census")
    intervals = []
    for name, size in required.items():
        address, actual = records[name]
        alignment = 256 if name == "workspace" else 16
        if address == 0 or address % alignment or actual != size:
            raise ValueError("region admission")
        if address > U64_MAX - actual:
            raise OverflowError
        intervals.append((address, address + actual))
    intervals.sort()
    if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
        raise ValueError("alias")


def validate_kernel_census(kernels: list[dict]) -> None:
    roles = [kernel.get("role") for kernel in kernels]
    functions = [kernel.get("function") for kernel in kernels]
    if len(kernels) != 5 or len(set(roles)) != 5 or len(set(functions)) != 5:
        raise ValueError("kernel census")
