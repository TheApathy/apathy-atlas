# SPDX-License-Identifier: AGPL-3.0-only
"""Exact frozen five-kernel route with the sole output-clear delta."""

from __future__ import annotations

import torch

from frozen_triton_executor.buffers import FrozenBuffers
from frozen_triton_executor.cuda_driver import F32, U32, U64, device_pointer
from frozen_triton_executor.launch import FrozenOracle


def run_nozero(
    oracle: FrozenOracle,
    buffers: FrozenBuffers,
    stream: torch.cuda.Stream,
    *,
    expected_seed: int | None = None,
) -> dict[str, bool]:
    v = buffers.views
    output = buffers.guards["output_bf16"].payload
    state = buffers.guards["atlas_state_hkv_f32"].payload
    buffers.reset_mutable()
    buffers.adapter_qkv()
    buffers.adapter_gate()
    buffers.adapter_state_in()
    v["A_bf16"].zero_()
    checks = {}
    if expected_seed is not None:
        checks["A_zero_before_kkt"] = bool((v["A_bf16"] == 0).all().item())
        checks["output_seed_before_kernel"] = bool(
            (output.view(torch.uint8) == expected_seed).all().item()
        )
    native_stream = int(stream.cuda_stream)
    oracle._launch(
        "cumsum",
        native_stream,
        [
            device_pointer(v["log_gate_f32"]),
            device_pointer(v["g_cumsum_f32"]),
            device_pointer(v["cu_seqlens_i32"]),
            device_pointer(v["chunk_indices_i32"]),
            (U32, oracle.m),
            (U64, 0),
            (U64, 0),
        ],
    )
    oracle._launch(
        "kkt_bc16_solve",
        native_stream,
        [
            device_pointer(v["k_bf16"]),
            device_pointer(v["g_cumsum_f32"]),
            device_pointer(v["beta_f32"]),
            device_pointer(v["A_bf16"]),
            device_pointer(v["cu_seqlens_i32"]),
            device_pointer(v["chunk_indices_i32"]),
            (U32, oracle.m),
            (U64, 0),
            (U64, 0),
        ],
    )
    oracle._launch(
        "recompute_w_u",
        native_stream,
        [
            device_pointer(v["k_bf16"]),
            device_pointer(v["v_bf16"]),
            device_pointer(v["beta_f32"]),
            device_pointer(v["w_bf16"]),
            device_pointer(v["u_bf16"]),
            device_pointer(v["A_bf16"]),
            device_pointer(v["g_cumsum_f32"]),
            device_pointer(v["cu_seqlens_i32"]),
            device_pointer(v["chunk_indices_i32"]),
            (U32, oracle.m),
            (U64, 0),
            (U64, 0),
        ],
    )
    oracle._launch(
        "chunk_recurrence_state",
        native_stream,
        [
            device_pointer(v["k_bf16"]),
            device_pointer(v["u_bf16"]),
            device_pointer(v["w_bf16"]),
            device_pointer(v["v_new_bf16"]),
            device_pointer(v["g_cumsum_f32"]),
            device_pointer(v["h_bf16"]),
            device_pointer(v["state_hvk_f32"]),
            device_pointer(v["state_index_i32"]),
            (U32, 786_432),
            device_pointer(v["cu_seqlens_i32"]),
            device_pointer(v["chunk_offsets_i64"]),
            (U32, oracle.m),
            (U64, 0),
            (U64, 0),
        ],
    )
    oracle._launch(
        "output",
        native_stream,
        [
            device_pointer(v["q_bf16"]),
            device_pointer(v["k_bf16"]),
            device_pointer(v["v_new_bf16"]),
            device_pointer(v["h_bf16"]),
            device_pointer(v["g_cumsum_f32"]),
            device_pointer(output),
            device_pointer(v["cu_seqlens_i32"]),
            device_pointer(v["chunk_indices_i32"]),
            (F32, 0.08838834764831845),
            (U32, oracle.m),
            (U64, 0),
            (U64, 0),
        ],
    )
    buffers.adapter_state_out()
    if expected_seed is not None:
        checks["state_out_transpose_exact"] = torch.equal(
            state, v["state_hvk_f32"].view(48, 128, 128).transpose(-1, -2)
        )
    return checks
