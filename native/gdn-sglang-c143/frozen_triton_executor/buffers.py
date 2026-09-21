# SPDX-License-Identifier: AGPL-3.0-only
"""Guarded exact-layout tensors and Atlas-to-frozen adapters."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

import frozen_triton_c143_gate as sealed_gate
from frozen_triton_c143_io import GateError

CANARY_BYTES = 4096
HK, HV, K, V = 16, 48, 128, 128


def tensor_sha256(value: torch.Tensor) -> str:
    raw = value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


@dataclass
class Guarded:
    payload: torch.Tensor
    raw: torch.Tensor
    prefix: int
    suffix: int

    @classmethod
    def from_tensor(cls, source: torch.Tensor, prefix: int, suffix: int) -> "Guarded":
        size = source.numel() * source.element_size()
        raw = torch.empty(
            CANARY_BYTES + size + CANARY_BYTES, dtype=torch.uint8, device="cuda"
        )
        raw[:CANARY_BYTES].fill_(prefix)
        raw[CANARY_BYTES + size :].fill_(suffix)
        payload = (
            raw[CANARY_BYTES : CANARY_BYTES + size]
            .view(source.dtype)
            .view(source.shape)
        )
        payload.copy_(source)
        return cls(payload, raw, prefix, suffix)

    @classmethod
    def empty(
        cls,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        fill: int,
        prefix: int,
        suffix: int,
    ) -> "Guarded":
        source = torch.empty(shape, dtype=dtype, device="cuda")
        source.view(torch.uint8).fill_(fill)
        return cls.from_tensor(source, prefix, suffix)

    def clean(self) -> bool:
        size = self.payload.numel() * self.payload.element_size()
        return bool((self.raw[:CANARY_BYTES] == self.prefix).all().item()) and bool(
            (self.raw[CANARY_BYTES + size :] == self.suffix).all().item()
        )


class FrozenBuffers:
    def __init__(self, fixture: dict[str, torch.Tensor], m: int) -> None:
        self.fixture = fixture
        self.m = m
        self.layout = sealed_gate.workspace_layout(m)
        qkv = torch.empty((m, 10_240), dtype=torch.bfloat16, device="cuda")
        qkv[:, :2_048].copy_(fixture["q"].reshape(m, 2_048))
        qkv[:, 2_048:4_096].copy_(fixture["k"].reshape(m, 2_048))
        qkv[:, 4_096:].copy_(fixture["v"].reshape(m, 6_144))
        gate_beta = torch.empty((m, 96), dtype=torch.float32, device="cuda")
        gate_beta[:, :48].copy_(fixture["alpha"].reshape(m, 48))
        gate_beta[:, 48:].copy_(fixture["beta"].reshape(m, 48))
        state_hkv = (
            fixture["state"].reshape(48, 128, 128).transpose(-1, -2).contiguous()
        )
        self.initial_state_hkv = state_hkv
        self.guards = {
            "atlas_qkv_bf16": Guarded.from_tensor(qkv, 0x21, 0xD1),
            "atlas_gate_beta_f32": Guarded.from_tensor(gate_beta, 0x22, 0xD2),
            "atlas_state_hkv_f32": Guarded.from_tensor(state_hkv, 0x23, 0xD3),
            "output_bf16": Guarded.empty(
                (m, 48, 128), torch.bfloat16, 0x6A, 0x24, 0xD4
            ),
            "workspace": Guarded.empty(
                (self.layout["total_bytes"],), torch.uint8, 0xA5, 0x25, 0xD5
            ),
        }
        self.workspace = self.guards["workspace"].payload
        self.views = self._make_views()
        self._initialize_metadata()
        self.padding_before = self.padding_sha256()
        self._validate_extents()

    def _make_views(self) -> dict[str, torch.Tensor]:
        shapes: dict[str, tuple[torch.dtype, tuple[int, ...]]] = {
            "q_bf16": (torch.bfloat16, (self.m, 16, 128)),
            "k_bf16": (torch.bfloat16, (self.m, 16, 128)),
            "v_bf16": (torch.bfloat16, (self.m, 48, 128)),
            "log_gate_f32": (torch.float32, (self.m, 48)),
            "beta_f32": (torch.float32, (self.m, 48)),
            "state_hvk_f32": (torch.float32, (1, 48, 128, 128)),
            "g_cumsum_f32": (torch.float32, (self.m, 48)),
            "A_bf16": (torch.bfloat16, (self.m, 48, 64)),
            "w_bf16": (torch.bfloat16, (self.m, 48, 128)),
            "u_bf16": (torch.bfloat16, (self.m, 48, 128)),
            "h_bf16": (torch.bfloat16, (self.layout["nt"], 48, 128, 128)),
            "v_new_bf16": (torch.bfloat16, (self.m, 48, 128)),
            "cu_seqlens_i32": (torch.int32, (2,)),
            "state_index_i32": (torch.int32, (1,)),
            "chunk_indices_i32": (torch.int32, (self.layout["nt"], 2)),
            "chunk_offsets_i64": (torch.int64, (2,)),
        }
        result = {}
        for name, offset, size in self.layout["segments"]:
            dtype, shape = shapes[name]
            expected = torch.empty((), dtype=dtype).element_size()
            elements = 1
            for extent in shape:
                elements *= extent
            if elements * expected != size:
                raise GateError(f"workspace shape drift: {name}")
            result[name] = (
                self.workspace.narrow(0, offset, size).view(dtype).view(shape)
            )
        return result

    def _initialize_metadata(self) -> None:
        nt = self.layout["nt"]
        self.views["cu_seqlens_i32"].copy_(
            torch.tensor([0, self.m], dtype=torch.int32, device="cuda")
        )
        self.views["state_index_i32"].zero_()
        indices = torch.stack(
            (
                torch.zeros(nt, dtype=torch.int32, device="cuda"),
                torch.arange(nt, dtype=torch.int32, device="cuda"),
            ),
            dim=1,
        )
        self.views["chunk_indices_i32"].copy_(indices)
        self.views["chunk_offsets_i64"].copy_(
            torch.tensor([0, nt], dtype=torch.int64, device="cuda")
        )

    def _validate_extents(self) -> None:
        records = {}
        for name, guard in self.guards.items():
            records[name] = {
                "address": guard.payload.data_ptr(),
                "bytes": guard.payload.numel() * guard.payload.element_size(),
            }
        sealed_gate.validate_runtime_extents(self.m, records, stream=1)

    def padding_sha256(self) -> str:
        cursor = 0
        digest = hashlib.sha256()
        for _, offset, size in self.layout["segments"]:
            if offset > cursor:
                digest.update(
                    self.workspace[cursor:offset].detach().cpu().numpy().tobytes()
                )
            cursor = offset + size
        if cursor < self.workspace.numel():
            digest.update(self.workspace[cursor:].detach().cpu().numpy().tobytes())
        return digest.hexdigest()

    def adapter_qkv(self) -> None:
        source = self.guards["atlas_qkv_bf16"].payload
        self.views["q_bf16"].copy_(source[:, :2_048].view(self.m, 16, 128))
        self.views["k_bf16"].copy_(source[:, 2_048:4_096].view(self.m, 16, 128))
        self.views["v_bf16"].copy_(source[:, 4_096:].view(self.m, 48, 128))

    def adapter_gate(self) -> None:
        source = self.guards["atlas_gate_beta_f32"].payload
        torch.log(
            torch.clamp(source[:, :48], min=1e-30), out=self.views["log_gate_f32"]
        )
        self.views["beta_f32"].copy_(source[:, 48:])

    def adapter_state_in(self) -> None:
        source = self.guards["atlas_state_hkv_f32"].payload
        self.views["state_hvk_f32"].copy_(
            source.transpose(-1, -2).view(1, 48, 128, 128)
        )

    def adapter_state_out(self) -> None:
        target = self.guards["atlas_state_hkv_f32"].payload
        target.copy_(self.views["state_hvk_f32"].view(48, 128, 128).transpose(-1, -2))

    def reset_mutable(self) -> None:
        self.guards["atlas_state_hkv_f32"].payload.copy_(self.initial_state_hkv)

    def input_hashes(self) -> dict[str, str]:
        names = {
            "atlas_qkv_bf16": self.guards["atlas_qkv_bf16"].payload,
            "atlas_gate_beta_f32": self.guards["atlas_gate_beta_f32"].payload,
            "cu_seqlens_i32": self.views["cu_seqlens_i32"],
            "state_index_i32": self.views["state_index_i32"],
            "chunk_indices_i32": self.views["chunk_indices_i32"],
            "chunk_offsets_i64": self.views["chunk_offsets_i64"],
        }
        return {name: tensor_sha256(value) for name, value in names.items()}

    def all_canaries_clean(self) -> bool:
        return all(guard.clean() for guard in self.guards.values())
