# SPDX-License-Identifier: AGPL-3.0-only
"""Pinned Atlas/SGLang references and deterministic fixture families."""

from __future__ import annotations

import math
from typing import Callable

import torch

import gate as legacy_gate

from .buffers import Guarded, tensor_sha256
from .contract import ATLAS_LIBRARY, ATLAS_LIBRARY_BYTES, PINNED
from .verified_loader import SealedMemfd

HK, HV, K, V = 16, 48, 128, 128


def make_fixture(m: int, kind: str) -> dict[str, torch.Tensor]:
    if kind == "real":
        q, k, v, _, alpha, beta, state = legacy_gate.make_inputs(m)
    elif kind == "adversarial":
        q_index = torch.arange(m * HK * K, dtype=torch.int32, device="cuda").view(
            1, m, HK, K
        )
        q = (((q_index % 29).float() - 14.0) / 16.0).to(torch.bfloat16)
        k = ((((q_index * 7 + 3) % 31).float() - 15.0) / 16.0).to(torch.bfloat16)
        q = (
            torch.nn.functional.normalize(q.float(), dim=-1)
            .to(torch.bfloat16)
            .contiguous()
        )
        k = (
            torch.nn.functional.normalize(k.float(), dim=-1)
            .to(torch.bfloat16)
            .contiguous()
        )
        v_index = torch.arange(m * HV * V, dtype=torch.int32, device="cuda").view(
            1, m, HV, V
        )
        v = (
            ((((v_index * 5 + 1) % 41).float() - 20.0) / 64.0)
            .to(torch.bfloat16)
            .contiguous()
        )
        row = torch.arange(m * HV, dtype=torch.int32, device="cuda").view(1, m, HV)
        decay = -torch.exp(torch.linspace(-5.0, -1.5, HV, device="cuda")).view(1, 1, HV)
        log_gate = decay * torch.nn.functional.softplus(
            ((row % 19).float() - 9.0) / 4.0
        )
        alpha = torch.exp(log_gate).contiguous()
        beta = torch.sigmoid(((row * 11 % 37).float() - 18.0) / 5.0).contiguous()
        state_index = torch.arange(HV * V * K, dtype=torch.int32, device="cuda").view(
            1, HV, V, K
        )
        state = ((((state_index * 13 + 5) % 53).float() - 26.0) / 4096.0).contiguous()
    else:
        raise ValueError(f"unknown fixture kind {kind!r}")
    log_gate = torch.log(torch.clamp(alpha, min=1e-30)).contiguous()
    return {
        "q": q.contiguous(),
        "k": k.contiguous(),
        "v": v.contiguous(),
        "log_gate": log_gate,
        "alpha": alpha.contiguous(),
        "beta": beta.contiguous(),
        "state": state.contiguous(),
    }


def metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    left = candidate.float().reshape(-1).double()
    right = reference.float().reshape(-1).double()
    difference = left - right
    rms = torch.sqrt(torch.mean(difference * difference)).item()
    reference_rms = torch.sqrt(torch.mean(right * right)).item()
    denominator = max(
        torch.linalg.vector_norm(left).item() * torch.linalg.vector_norm(right).item(),
        1e-30,
    )
    return {
        "max_abs": torch.max(torch.abs(difference)).item(),
        "rms": rms,
        "relative_rms": rms / max(reference_rms, 1e-30),
        "cosine": torch.dot(left, right).item() / denominator,
    }


def worst_metrics(values: list[dict[str, float]]) -> dict[str, float]:
    return {
        "max_abs": max(value["max_abs"] for value in values),
        "rms": max(value["rms"] for value in values),
        "relative_rms": max(value["relative_rms"] for value in values),
        "cosine": min(value["cosine"] for value in values),
    }


class References:
    def __init__(self) -> None:
        self.atlas_image = SealedMemfd.from_file(
            ATLAS_LIBRARY,
            PINNED["atlas_bridge_sha256"],
            ATLAS_LIBRARY_BYTES,
            "atlas-gdn-wy32",
        )
        self.atlas = legacy_gate.AtlasBridge(self.atlas_image.path)
        self.atlas_image.verify_executable_mapping()
        _, self.sglang_sources, self.sglang = legacy_gate.load_pinned_sglang()

    def atlas_once(self, fixture: dict[str, torch.Tensor]) -> dict:
        m = fixture["q"].shape[1]
        state_source = (
            fixture["state"].reshape(48, 128, 128).transpose(-1, -2).contiguous()
        )
        state = Guarded.from_tensor(state_source, 0x31, 0xE1)
        output = Guarded.empty((1, m, 48, 128), torch.bfloat16, 0x77, 0x32, 0xE2)
        self.atlas.launch(
            state.payload,
            fixture["q"],
            fixture["k"],
            fixture["v"],
            fixture["alpha"],
            fixture["beta"],
            output.payload,
        )
        canonical_state = (
            state.payload.reshape(1, 48, 128, 128).transpose(-1, -2).contiguous()
        )
        return {
            "output": output.payload,
            "state": canonical_state,
            "guards": (state, output),
        }

    def sglang_once(self, fixture: dict[str, torch.Tensor]) -> dict:
        m = fixture["q"].shape[1]
        state = Guarded.from_tensor(fixture["state"], 0x33, 0xE3)
        cu = torch.tensor([0, m], dtype=torch.int32, device="cuda")
        state_index = torch.tensor([0], dtype=torch.int32, device="cuda")
        output, _, _ = self.sglang(
            q=fixture["q"],
            k=fixture["k"],
            v=fixture["v"],
            g=fixture["log_gate"],
            beta=fixture["beta"],
            initial_state=state.payload,
            initial_state_indices=state_index,
            cu_seqlens=cu,
            head_first=False,
            use_qk_l2norm_in_kernel=False,
        )
        return {"output": output, "state": state.payload, "guards": (state,)}

    def timed_atlas(self, fixture: dict[str, torch.Tensor]) -> Callable[[], dict]:
        m = fixture["q"].shape[1]
        initial = fixture["state"].reshape(48, 128, 128).transpose(-1, -2).contiguous()
        state = Guarded.from_tensor(initial, 0x41, 0xC1)
        output = Guarded.empty((1, m, 48, 128), torch.bfloat16, 0x55, 0x42, 0xC2)

        def invoke() -> dict:
            state.payload.copy_(initial)
            output.payload.zero_()
            self.atlas.launch(
                state.payload,
                fixture["q"],
                fixture["k"],
                fixture["v"],
                fixture["alpha"],
                fixture["beta"],
                output.payload,
            )
            canonical = state.payload.reshape(1, 48, 128, 128).transpose(-1, -2)
            return {
                "output": output.payload,
                "state": canonical,
                "guards": (state, output),
            }

        return invoke

    def timed_sglang(self, fixture: dict[str, torch.Tensor]) -> Callable[[], dict]:
        m = fixture["q"].shape[1]
        state = Guarded.from_tensor(fixture["state"], 0x43, 0xC3)
        cu = torch.tensor([0, m], dtype=torch.int32, device="cuda")
        state_index = torch.tensor([0], dtype=torch.int32, device="cuda")

        def invoke() -> dict:
            state.payload.copy_(fixture["state"])
            output, _, _ = self.sglang(
                q=fixture["q"],
                k=fixture["k"],
                v=fixture["v"],
                g=fixture["log_gate"],
                beta=fixture["beta"],
                initial_state=state.payload,
                initial_state_indices=state_index,
                cu_seqlens=cu,
                head_first=False,
                use_qk_l2norm_in_kernel=False,
            )
            return {"output": output, "state": state.payload, "guards": (state,)}

        return invoke


def timed_call(
    invoke: Callable[[], dict], stream: torch.cuda.Stream
) -> tuple[dict, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record(stream)
    result = invoke()
    end.record(stream)
    end.synchronize()
    return result, start.elapsed_time(end)


def result_identity(result: dict) -> dict[str, str | bool]:
    guards_clean = all(item.clean() for item in result["guards"])
    return {
        "output_sha256": tensor_sha256(result["output"]),
        "state_sha256": tensor_sha256(result["state"]),
        "guards_clean": guards_clean,
        "finite": bool(torch.isfinite(result["output"]).all().item())
        and bool(torch.isfinite(result["state"]).all().item()),
    }


def p90(values: list[float]) -> float:
    return sorted(values)[math.ceil(0.9 * len(values)) - 1]
