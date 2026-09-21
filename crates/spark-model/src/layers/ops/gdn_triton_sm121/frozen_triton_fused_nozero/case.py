# SPDX-License-Identifier: AGPL-3.0-only
"""Exact parity, boundary, determinism, and balanced timing gates."""

from __future__ import annotations

import statistics
from typing import Callable

import torch

from frozen_triton_c143_io import GateError
from frozen_triton_c143_receipt_metrics import metric_pair, p90
from frozen_triton_executor.buffers import FrozenBuffers, tensor_sha256
from frozen_triton_executor.case import candidate_result
from frozen_triton_executor.launch import FrozenOracle
from frozen_triton_executor.references import (
    References,
    make_fixture,
    metrics,
    result_identity,
)
from frozen_triton_executor.timing import result_clean
from frozen_triton_nozero.launch import run_nozero

from .adapter import FusedBuffers, FusedInputLibrary


def _identity(result: dict) -> tuple[str, str]:
    value = result_identity(result)
    if value["guards_clean"] is not True or value["finite"] is not True:
        raise GateError("fused-nozero result guard/finite failure")
    return value["output_sha256"], value["state_sha256"]


def _comparison(candidate: dict, reference: dict) -> dict:
    return {
        name: metrics(candidate[name], reference[name]) for name in ("output", "state")
    }


def _immutable_hashes(buffers: FrozenBuffers) -> dict[str, str]:
    return {
        **buffers.input_hashes(),
        "atlas_gate_beta_f32": tensor_sha256(
            buffers.guards["atlas_gate_beta_f32"].payload
        ),
    }


def _validate_nozero_checks(checks: dict[str, bool]) -> None:
    if set(checks) != {
        "A_zero_before_kkt",
        "output_seed_before_kernel",
        "state_out_transpose_exact",
    } or not all(checks.values()):
        raise GateError("approved nozero launch checks failed")


def _run_fused(oracle, buffers, stream, *, capture=False, seed=None) -> dict:
    buffers.bind_stream(stream, capture=capture)
    checks = run_nozero(oracle, buffers, stream, expected_seed=seed)
    if seed is not None:
        _validate_nozero_checks(checks)
    return candidate_result(buffers)


def adapter_boundary(
    library: FusedInputLibrary, m: int, stream: torch.cuda.Stream
) -> dict[str, bool]:
    buffers = FusedBuffers(make_fixture(m, "real"), m, library)
    special = torch.tensor(
        [float("nan"), float("inf"), -float("inf"), -0.0, 0.0, 1e-30, 1.0, 2.0],
        dtype=torch.float32,
        device="cuda",
    )
    buffers.guards["atlas_gate_beta_f32"].payload[0, : special.numel()].copy_(special)
    buffers.bind_stream(stream, capture=True)
    buffers.reset_mutable()
    buffers.adapter_qkv()
    buffers.adapter_gate()
    buffers.adapter_state_in()
    stream.synchronize()
    checks = buffers.adapter_checks()
    if not all(checks.values()) or not buffers.all_canaries_clean():
        raise GateError("fused input NaN/inf/log boundary parity failed")
    return checks


def correctness(
    oracle: FrozenOracle,
    references: References,
    fixture: dict[str, torch.Tensor],
    stream: torch.cuda.Stream,
    library: FusedInputLibrary,
) -> dict:
    candidate = FusedBuffers(fixture, oracle.m, library)
    candidate_before = _immutable_hashes(candidate)
    candidate_ids, adapter_checks = [], []
    for seed in (0x6A, 0xC3):
        candidate.guards["output_bf16"].payload.view(torch.uint8).fill_(seed)
        result = _run_fused(oracle, candidate, stream, capture=True, seed=seed)
        stream.synchronize()
        candidate_ids.append(_identity(result))
        adapter_checks.append(candidate.adapter_checks())
    candidate_after = _immutable_hashes(candidate)
    parent = FrozenBuffers(fixture, oracle.m)
    parent.guards["output_bf16"].payload.view(torch.uint8).fill_(0x6A)
    parent_before = _immutable_hashes(parent)
    parent_checks = run_nozero(oracle, parent, stream, expected_seed=0x6A)
    _validate_nozero_checks(parent_checks)
    stream.synchronize()
    parent_result = candidate_result(parent)
    parent_id = _identity(parent_result)
    parent_after = _immutable_hashes(parent)
    exact = (
        candidate_ids[0] == candidate_ids[1] == parent_id
        and candidate_before == candidate_after
        and parent_before == parent_after
        and candidate.all_canaries_clean()
        and parent.all_canaries_clean()
        and all(all(value.values()) for value in adapter_checks)
    )
    if not exact:
        raise GateError("fused-nozero exact parent/determinism/canary gate failed")
    fused_result = candidate_result(candidate)
    atlas = references.atlas_once(fixture)
    sglang = references.sglang_once(fixture)
    comparisons = {
        "candidate_vs_atlas": _comparison(fused_result, atlas),
        "candidate_vs_sglang": _comparison(fused_result, sglang),
    }
    for name, value in comparisons.items():
        metric_pair(value, name)
    return {
        "candidate_identity": candidate_ids[0],
        "nozero_parent_identity": parent_id,
        "adapter_checks": adapter_checks,
        "nozero_launch_checks": True,
        "comparisons": comparisons,
        "immutable_inputs": True,
        "canaries_clean": True,
        "deterministic": True,
    }


def _callables(
    oracle, references, fixture, stream, library
) -> dict[str, Callable[[], dict]]:
    fused = FusedBuffers(fixture, oracle.m, library)
    nozero = FrozenBuffers(fixture, oracle.m)
    zero = FrozenBuffers(fixture, oracle.m)

    def fused_call() -> dict:
        return _run_fused(oracle, fused, stream)

    def nozero_call() -> dict:
        run_nozero(oracle, nozero, stream)
        return candidate_result(nozero)

    def zero_call() -> dict:
        oracle.run(zero, stream, capture=False, verify=False)
        return candidate_result(zero)

    return {
        "fused": fused_call,
        "nozero": nozero_call,
        "zero": zero_call,
        "atlas": references.timed_atlas(fixture),
        "sglang": references.timed_sglang(fixture),
    }


def timing(oracle, references, fixture, stream, library) -> dict:
    arms = _callables(oracle, references, fixture, stream, library)
    for _ in range(2):
        for invoke in arms.values():
            if not result_clean(invoke()):
                raise GateError("fused-nozero timing warmup guard failure")
            stream.synchronize()
    values = {name: [] for name in arms}
    positions = {name: [0] * 5 for name in arms}
    names = tuple(arms)
    for repetition in range(25):
        order = names[repetition % 5 :] + names[: repetition % 5]
        for position, name in enumerate(order):
            positions[name][position] += 1
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record(stream)
            result = arms[name]()
            end.record(stream)
            end.synchronize()
            values[name].append(start.elapsed_time(end))
            if not result_clean(result):
                raise GateError(f"{name}: timed result gate failed")
    if any(count != 5 for counts in positions.values() for count in counts):
        raise GateError(f"fused-nozero timing imbalance: {positions!r}")
    medians = {name: statistics.median(value) for name, value in values.items()}
    p90s = {name: p90(value) for name, value in values.items()}
    paired = {
        name: statistics.median(a - f for a, f in zip(values[name], values["fused"]))
        for name in ("nozero", "zero", "atlas")
    }
    ratio = medians["fused"] / medians["sglang"]
    predicates = {
        "absolute": medians["fused"] < (2.5 if oracle.m == 2_079 else 10.0),
        "nozero_median": medians["fused"] < medians["nozero"],
        "nozero_p90": p90s["fused"] < p90s["nozero"],
        "paired_nozero": paired["nozero"] > 0.0,
        "zero_median": medians["fused"] < medians["zero"],
        "zero_p90": p90s["fused"] < p90s["zero"],
        "paired_zero": paired["zero"] > 0.0,
        "atlas_median": medians["fused"] < medians["atlas"],
        "atlas_p90": p90s["fused"] < p90s["atlas"],
        "paired_atlas": paired["atlas"] > 0.0,
        "sglang_ratio": ratio <= 1.20,
    }
    if not all(predicates.values()):
        raise GateError(f"fused-nozero strict timing failed: {predicates!r}")
    return {
        "samples_ms": values,
        "position_counts": positions,
        "medians_ms": medians,
        "p90_ms": p90s,
        "paired_gain_ms": paired,
        "candidate_to_sglang_ratio": ratio,
        "predicates": predicates,
    }


def run_shape(driver, attestation, references, m, stream, library) -> dict:
    oracle = FrozenOracle(driver, attestation, m)
    with torch.cuda.stream(stream):
        boundary = adapter_boundary(library, m, stream)
        evidence = [
            correctness(oracle, references, make_fixture(m, kind), stream, library)
            for kind in ("real", "adversarial")
        ]
        performance = timing(
            oracle, references, make_fixture(m, "real"), stream, library
        )
    stream.synchronize()
    return {
        "m": m,
        "boundary": boundary,
        "correctness": evidence,
        "timing": performance,
    }
