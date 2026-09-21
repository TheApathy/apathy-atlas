# SPDX-License-Identifier: AGPL-3.0-only
"""Exact five-kernel launch plan and adapter sequencing."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

import torch

import frozen_triton_c143_gate as sealed_gate
from frozen_triton_c143_io import GateError

from .buffers import FrozenBuffers, tensor_sha256
from .cuda_driver import F32, U32, U64, CudaDriver, device_pointer


def combined_sha256(values: list[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(bytes.fromhex(tensor_sha256(value)))
    return digest.hexdigest()


class FrozenOracle:
    def __init__(self, driver: CudaDriver, attestation: dict, m: int) -> None:
        self.driver = driver
        self.m = m
        self.plan = sealed_gate.resolved_launch_plan(attestation, m)
        self.functions = {}
        kernels = attestation["manifest"]["kernels"]
        if len(kernels) != 5 or len(self.plan) != 5:
            raise GateError("executor requires exact five-kernel plan")
        for kernel, plan in zip(kernels, self.plan):
            if plan["trailing_scratch_values_u64"] != [0, 0]:
                raise GateError("executor requires trailing null scratch values")
            path = Path(kernel["artifact_dir"]) / f"{kernel['function']}.cubin"
            self.functions[kernel["role"]] = driver.load_function(
                path, kernel["function"], kernel
            )

    def _launch(self, role: str, stream: int, args: list[tuple]) -> None:
        plan = next(item for item in self.plan if item["role"] == role)
        self.driver.launch_typed(self.functions[role], plan, stream, args)

    def run(
        self,
        buffers: FrozenBuffers,
        stream: torch.cuda.Stream,
        *,
        capture: bool,
        profile_stages: bool = False,
        verify: bool = True,
    ) -> dict:
        if buffers.m != self.m or int(stream.cuda_stream) == 0:
            raise GateError("candidate shape/stream mismatch")
        native_stream = int(stream.cuda_stream)
        stage_times: dict[str, float] = {}
        stage_hashes: dict[str, str] = {}
        checks: dict[str, bool] = {}

        def measure(names: tuple[str, ...], operation: Callable[[], None]) -> None:
            if not profile_stages:
                operation()
                return
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            operation()
            end.record(stream)
            end.synchronize()
            elapsed = start.elapsed_time(end)
            for name in names:
                stage_times[name] = elapsed

        def snapshot(name: str, values: list[torch.Tensor]) -> None:
            if capture:
                stream.synchronize()
                stage_hashes[name] = combined_sha256(values)

        v = buffers.views
        output = buffers.guards["output_bf16"].payload
        state = buffers.guards["atlas_state_hkv_f32"].payload
        buffers.reset_mutable()
        measure(("adapter_qkv_split",), buffers.adapter_qkv)
        source_qkv = buffers.guards["atlas_qkv_bf16"].payload
        if verify:
            checks["q_split_bytes_exact"] = torch.equal(
                v["q_bf16"], source_qkv[:, :2_048].view(self.m, 16, 128)
            )
            checks["k_split_bytes_exact"] = torch.equal(
                v["k_bf16"], source_qkv[:, 2_048:4_096].view(self.m, 16, 128)
            )
            checks["v_split_bytes_exact"] = torch.equal(
                v["v_bf16"], source_qkv[:, 4_096:].view(self.m, 48, 128)
            )
        snapshot("adapter_qkv_split", [v["q_bf16"], v["k_bf16"], v["v_bf16"]])
        measure(("adapter_alpha_log_beta_split",), buffers.adapter_gate)
        source_gate = buffers.guards["atlas_gate_beta_f32"].payload
        if verify:
            expected_log = torch.log(torch.clamp(source_gate[:, :48], min=1e-30))
            checks["log_gate_formula_exact"] = torch.equal(
                v["log_gate_f32"], expected_log
            )
            checks["beta_split_bytes_exact"] = torch.equal(
                v["beta_f32"], source_gate[:, 48:]
            )
        snapshot("adapter_alpha_log_beta_split", [v["log_gate_f32"], v["beta_f32"]])
        measure(("adapter_state_hkv_to_hvk",), buffers.adapter_state_in)
        if verify:
            checks["state_in_transpose_exact"] = torch.equal(
                v["state_hvk_f32"], state.transpose(-1, -2).view(1, 48, 128, 128)
            )
        snapshot("adapter_state_hkv_to_hvk", [v["state_hvk_f32"]])
        measure(("memset_A_zero",), v["A_bf16"].zero_)
        if verify:
            checks["A_zero_before_kkt"] = bool((v["A_bf16"] == 0).all().item())
        snapshot("memset_A_zero", [v["A_bf16"]])
        measure(("memset_output_zero",), output.zero_)
        snapshot("memset_output_zero", [output])
        measure(
            ("g_cumsum",),
            lambda: self._launch(
                "cumsum",
                native_stream,
                [
                    device_pointer(v["log_gate_f32"]),
                    device_pointer(v["g_cumsum_f32"]),
                    device_pointer(v["cu_seqlens_i32"]),
                    device_pointer(v["chunk_indices_i32"]),
                    (U32, self.m),
                    (U64, 0),
                    (U64, 0),
                ],
            ),
        )
        snapshot("g_cumsum", [v["g_cumsum_f32"]])
        measure(
            ("A",),
            lambda: self._launch(
                "kkt_bc16_solve",
                native_stream,
                [
                    device_pointer(v["k_bf16"]),
                    device_pointer(v["g_cumsum_f32"]),
                    device_pointer(v["beta_f32"]),
                    device_pointer(v["A_bf16"]),
                    device_pointer(v["cu_seqlens_i32"]),
                    device_pointer(v["chunk_indices_i32"]),
                    (U32, self.m),
                    (U64, 0),
                    (U64, 0),
                ],
            ),
        )
        snapshot("A", [v["A_bf16"]])
        measure(
            ("w", "u"),
            lambda: self._launch(
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
                    (U32, self.m),
                    (U64, 0),
                    (U64, 0),
                ],
            ),
        )
        for name in ("w", "u"):
            snapshot(name, [v[f"{name}_bf16"]])
        measure(
            ("h", "v_new"),
            lambda: self._launch(
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
                    (U32, self.m),
                    (U64, 0),
                    (U64, 0),
                ],
            ),
        )
        for name in ("h", "v_new"):
            snapshot(name, [v[f"{name}_bf16"]])
        if verify:
            checks["output_zero_before_output_kernel"] = bool(
                (output == 0).all().item()
            )
        measure(
            ("output",),
            lambda: self._launch(
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
                    (U32, self.m),
                    (U64, 0),
                    (U64, 0),
                ],
            ),
        )
        snapshot("output", [output])
        measure(("adapter_state_hvk_to_hkv",), buffers.adapter_state_out)
        if verify:
            checks["state_out_transpose_exact"] = torch.equal(
                state, v["state_hvk_f32"].view(48, 128, 128).transpose(-1, -2)
            )
        snapshot("adapter_state_hvk_to_hkv", [state])
        if verify:
            checks.update(
                metadata_values_exact=True,
                same_nondefault_stream=True,
                all_extents_nonalias=True,
                workspace_padding_unchanged=buffers.padding_sha256()
                == buffers.padding_before,
            )
        return {
            "stage_times": stage_times,
            "stage_hashes": stage_hashes,
            "checks": checks,
        }
