"""Checked workspace, buffer, and launch geometry contracts."""

from __future__ import annotations

from typing import Any

from frozen_triton_c143_constants import (
    KERNEL_CONTRACTS,
    POINTER_ALIGNMENT,
    SUPPORTED_M,
    WORKSPACE_ALIGNMENT,
)
from frozen_triton_c143_io import (
    GateError,
    align_up,
    checked_add,
    checked_mul,
    require_exact_keys,
    require_int,
)


def workspace_layout(m: int) -> dict[str, Any]:
    require_int(m, "M", minimum=1)
    if m not in SUPPORTED_M:
        raise GateError(f"unsupported M={m}; only M2079/M8192 are qualified")
    nt = (m + 63) // 64
    sizes = {
        "q_bf16": checked_mul(m, 16, 128, 2, label="q bytes"),
        "k_bf16": checked_mul(m, 16, 128, 2, label="k bytes"),
        "v_bf16": checked_mul(m, 48, 128, 2, label="v bytes"),
        "log_gate_f32": checked_mul(m, 48, 4, label="g_log bytes"),
        "beta_f32": checked_mul(m, 48, 4, label="beta bytes"),
        "state_hvk_f32": checked_mul(48, 128, 128, 4, label="state bytes"),
        "g_cumsum_f32": checked_mul(m, 48, 4, label="cumsum bytes"),
        "A_bf16": checked_mul(m, 48, 64, 2, label="A bytes"),
        "w_bf16": checked_mul(m, 48, 128, 2, label="w bytes"),
        "u_bf16": checked_mul(m, 48, 128, 2, label="u bytes"),
        "h_bf16": checked_mul(nt, 48, 128, 128, 2, label="h bytes"),
        "v_new_bf16": checked_mul(m, 48, 128, 2, label="v_new bytes"),
        "cu_seqlens_i32": 2 * 4,
        "state_index_i32": 4,
        "chunk_indices_i32": checked_mul(nt, 2, 4, label="chunk indices bytes"),
        "chunk_offsets_i64": 2 * 8,
    }
    external = {
        "atlas_qkv_bf16": checked_mul(m, 10_240, 2, label="atlas qkv bytes"),
        "atlas_gate_beta_f32": checked_mul(m, 96, 4, label="atlas gate bytes"),
        "atlas_state_hkv_f32": checked_mul(48, 128, 128, 4, label="atlas state"),
        "output_bf16": checked_mul(m, 48, 128, 2, label="output bytes"),
    }
    segments = []
    cursor = 0
    for name, size in sizes.items():
        cursor = align_up(cursor, WORKSPACE_ALIGNMENT, f"{name}.alignment")
        segments.append([name, cursor, size])
        cursor = checked_add(cursor, size, f"{name}.end")
    total = align_up(cursor, WORKSPACE_ALIGNMENT, "workspace.total")
    return {
        "nt": nt,
        "total_bytes": total,
        "external_bytes": external,
        "segments": segments,
    }


def validate_workspace_manifest(value: Any) -> None:
    expected = {str(m): workspace_layout(m) for m in SUPPORTED_M}
    if value != expected:
        raise GateError("workspace_layouts: checked layout drift")
    for layout in expected.values():
        intervals = sorted(
            (offset, offset + size, name) for name, offset, size in layout["segments"]
        )
        if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
            raise GateError("workspace_layouts: overlap")


def validate_runtime_extents(m: int, buffers: Any, *, stream: int) -> dict[str, Any]:
    layout = workspace_layout(m)
    required = dict(layout["external_bytes"])
    required["workspace"] = layout["total_bytes"]
    records = require_exact_keys(buffers, set(required), "runtime buffers")
    intervals = []
    for name, required_bytes in required.items():
        record = require_exact_keys(
            records[name], {"address", "bytes"}, f"buffer:{name}"
        )
        address = require_int(record["address"], f"buffer:{name}.address", minimum=1)
        length = require_int(record["bytes"], f"buffer:{name}.bytes", minimum=1)
        alignment = WORKSPACE_ALIGNMENT if name == "workspace" else POINTER_ALIGNMENT
        if address % alignment:
            raise GateError(f"buffer:{name}: unaligned")
        if length != required_bytes:
            raise GateError(f"buffer:{name}: exact extent required")
        end = checked_add(address, length, f"buffer:{name}.end")
        intervals.append((address, end, name))
    intervals.sort()
    if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
        raise GateError("runtime alias detected")
    require_int(stream, "stream", minimum=1)
    return layout


def resolved_launch_plan(attestation: dict[str, Any], m: int) -> list[dict[str, Any]]:
    nt = workspace_layout(m)["nt"]
    kernels = attestation["manifest"]["kernels"]
    plan = []
    for kernel in kernels:
        role = kernel["role"]
        contract = KERNEL_CONTRACTS[role]
        grid = [nt if value == "NT" else value for value in contract["grid"]]
        driver_values = ["runtime"] * (len(contract["driver"]) - 2) + [0, 0]
        plan.append(
            {
                "role": role,
                "function": contract["function"],
                "grid": grid,
                "block": contract["block"],
                "dynamic_shared_bytes": contract["shared"],
                "driver_params": kernel["driver_params"],
                "trailing_scratch_values_u64": driver_values[-2:],
            }
        )
    return plan
