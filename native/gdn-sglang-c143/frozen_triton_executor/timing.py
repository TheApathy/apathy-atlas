# SPDX-License-Identifier: AGPL-3.0-only
"""Excluded stage profiling and balanced full-route timing."""

from __future__ import annotations

from typing import Callable

import torch

import frozen_triton_c143_gate as sealed_gate
from frozen_triton_c143_io import GateError

from .buffers import FrozenBuffers
from .launch import FrozenOracle
from .references import References


def candidate_result(buffers: FrozenBuffers) -> dict:
    output = buffers.guards["output_bf16"].payload.view(1, buffers.m, 48, 128)
    state = buffers.guards["atlas_state_hkv_f32"].payload
    canonical = state.view(1, 48, 128, 128).transpose(-1, -2)
    return {
        "output": output,
        "state": canonical,
        "guards": tuple(buffers.guards.values()),
    }


def result_clean(result: dict) -> bool:
    return (
        all(item.clean() for item in result["guards"])
        and bool(torch.isfinite(result["output"]).all().item())
        and bool(torch.isfinite(result["state"]).all().item())
    )


def candidate_callable(
    oracle: FrozenOracle, buffers: FrozenBuffers, stream: torch.cuda.Stream
) -> Callable[[], dict]:
    def invoke() -> dict:
        oracle.run(buffers, stream, capture=False, verify=False)
        return candidate_result(buffers)

    return invoke


def measure_full_timing(
    oracle: FrozenOracle,
    references: References,
    fixture: dict[str, torch.Tensor],
    stream: torch.cuda.Stream,
) -> tuple[dict[str, list[float]], dict[str, list[int]]]:
    candidate_buffers = FrozenBuffers(fixture, oracle.m)
    arms = {
        "candidate": candidate_callable(oracle, candidate_buffers, stream),
        "atlas": references.timed_atlas(fixture),
        "sglang": references.timed_sglang(fixture),
    }
    for _ in range(2):
        for invoke in arms.values():
            result = invoke()
            stream.synchronize()
            if not result_clean(result):
                raise GateError("timing warmup guard corruption")
    times = {name: [] for name in arms}
    positions = {name: [0, 0, 0] for name in arms}
    order = ("candidate", "atlas", "sglang")
    for repetition in range(22):
        shift = repetition % 3
        rotated = order[shift:] + order[:shift]
        for position, name in enumerate(rotated):
            positions[name][position] += 1
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            result = arms[name]()
            end.record(stream)
            end.synchronize()
            times[name].append(start.elapsed_time(end))
            if not result_clean(result):
                raise GateError(f"{name}: timed result gate failed")
    if any(max(counts) - min(counts) > 1 for counts in positions.values()):
        raise GateError("timing positions are not balanced")
    return times, positions


def measure_stages(
    oracle: FrozenOracle, fixture: dict[str, torch.Tensor], stream: torch.cuda.Stream
) -> dict[str, list[float]]:
    buffers = FrozenBuffers(fixture, oracle.m)
    values = {name: [] for name in sealed_gate.STAGE_NAMES}
    for _ in range(21):
        result = oracle.run(
            buffers,
            stream,
            capture=False,
            profile_stages=True,
            verify=False,
        )
        for name in sealed_gate.STAGE_NAMES:
            values[name].append(result["stage_times"][name])
    return values
