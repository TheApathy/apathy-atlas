# SPDX-License-Identifier: AGPL-3.0-only
"""Sealed fused-input library and exact FrozenBuffers adapter replacement."""

from __future__ import annotations

import ctypes
from pathlib import Path

import torch

from frozen_triton_c143_io import GateError
from frozen_triton_executor.buffers import FrozenBuffers
from frozen_triton_executor.verified_loader import SealedMemfd


def _bytes(value: torch.Tensor) -> int:
    return value.numel() * value.element_size()


def _raw_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    a = left.contiguous().view(torch.uint8).reshape(-1)
    b = right.contiguous().view(torch.uint8).reshape(-1)
    return a.numel() == b.numel() and bool(torch.equal(a, b))


class FusedInputLibrary:
    def __init__(self, path: Path, sha256: str, size: int) -> None:
        self.image = SealedMemfd.from_file(path, sha256, size, "gdn-fused-input")
        self._library = ctypes.CDLL(str(self.image.path))
        self.image.verify_executable_mapping()
        try:
            self._function = self._library.atlas_gdn_c143_pack_inputs_launch
        except AttributeError as error:
            raise GateError("fused input adapter symbol missing") from error
        pair = [ctypes.c_void_p, ctypes.c_uint64]
        self._function.argtypes = pair * 9 + [ctypes.c_uint32, ctypes.c_void_p]
        self._function.restype = ctypes.c_int

    def launch(self, buffers: "FusedBuffers", stream: int) -> None:
        v = buffers.views
        g = buffers.guards
        tensors = (
            g["atlas_qkv_bf16"].payload,
            g["atlas_gate_beta_f32"].payload,
            g["atlas_state_hkv_f32"].payload,
            v["q_bf16"],
            v["k_bf16"],
            v["v_bf16"],
            v["log_gate_f32"],
            v["beta_f32"],
            v["state_hvk_f32"],
        )
        args = []
        for value in tensors:
            args.extend(
                (ctypes.c_void_p(value.data_ptr()), ctypes.c_uint64(_bytes(value)))
            )
        status = self._function(
            *args, ctypes.c_uint32(buffers.m), ctypes.c_void_p(stream)
        )
        if status != 0:
            raise GateError(f"fused input adapter launch status {status}")


class FusedBuffers(FrozenBuffers):
    def __init__(
        self, fixture: dict[str, torch.Tensor], m: int, library: FusedInputLibrary
    ):
        super().__init__(fixture, m)
        self._fused_library = library
        self._stream = 0
        self._capture = False
        self._adapter_phase = 0
        self._adapter_checks: dict[str, bool] = {}

    def bind_stream(self, stream: torch.cuda.Stream, *, capture: bool) -> None:
        raw = int(stream.cuda_stream)
        if raw == 0:
            raise GateError("fused input adapter requires a nondefault stream")
        self._stream = raw
        self._capture = capture

    def reset_mutable(self) -> None:
        super().reset_mutable()
        self._adapter_phase = 0
        self._adapter_checks = {}

    def adapter_qkv(self) -> None:
        if self._adapter_phase != 0 or self._stream == 0:
            raise GateError("fused input adapter sequence/stream drift")
        self._fused_library.launch(self, self._stream)
        self._adapter_phase = 1

    def adapter_gate(self) -> None:
        if self._adapter_phase != 1:
            raise GateError("fused gate no-op is out of order")
        self._adapter_phase = 2

    def adapter_state_in(self) -> None:
        if self._adapter_phase != 2:
            raise GateError("fused state no-op is out of order")
        self._adapter_phase = 3
        if self._capture:
            self._capture_exact_intermediates()

    def _capture_exact_intermediates(self) -> None:
        v = self.views
        qkv = self.guards["atlas_qkv_bf16"].payload
        gate = self.guards["atlas_gate_beta_f32"].payload
        state = self.guards["atlas_state_hkv_f32"].payload
        expected_log = torch.log(torch.clamp(gate[:, :48], min=1e-30))
        self._adapter_checks = {
            "q_split_bytes_exact": _raw_equal(v["q_bf16"], qkv[:, :2_048]),
            "k_split_bytes_exact": _raw_equal(v["k_bf16"], qkv[:, 2_048:4_096]),
            "v_split_bytes_exact": _raw_equal(v["v_bf16"], qkv[:, 4_096:]),
            "beta_split_bytes_exact": _raw_equal(v["beta_f32"], gate[:, 48:]),
            "state_in_transpose_exact": _raw_equal(
                v["state_hvk_f32"], state.transpose(-1, -2).view(1, 48, 128, 128)
            ),
            "log_gate_bytes_exact": _raw_equal(v["log_gate_f32"], expected_log),
        }

    def adapter_checks(self) -> dict[str, bool]:
        if self._adapter_phase != 3 or not self._capture:
            raise GateError("fused adapter checks requested before capture")
        return dict(self._adapter_checks)
