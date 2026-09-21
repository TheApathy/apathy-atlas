# SPDX-License-Identifier: AGPL-3.0-only
"""Correctness and balanced timing for the no-output-clear raw candidate."""

from __future__ import annotations

import statistics
from typing import Callable

import torch

from frozen_triton_c143_io import GateError
from frozen_triton_c143_receipt_metrics import metric_pair, p90
from frozen_triton_executor.buffers import FrozenBuffers
from frozen_triton_executor.case import candidate_result
from frozen_triton_executor.launch import FrozenOracle
from frozen_triton_executor.references import (
    References,
    make_fixture,
    metrics,
    result_identity,
)
from frozen_triton_executor.timing import result_clean

from .launch import run_nozero


def _identity(result: dict) -> tuple[str, str]:
    value = result_identity(result)
    if value["guards_clean"] is not True or value["finite"] is not True:
        raise GateError("result guard/finite failure")
    return value["output_sha256"], value["state_sha256"]


def _comparison(candidate: dict, reference: dict) -> dict:
    return {
        name: metrics(candidate[name], reference[name]) for name in ("output", "state")
    }


def correctness(
    oracle: FrozenOracle,
    references: References,
    fixture: dict[str, torch.Tensor],
    stream: torch.cuda.Stream,
) -> dict:
    buffers = FrozenBuffers(fixture, oracle.m)
    before = buffers.input_hashes()
    candidates = []
    checks = []
    for seed in (0x6A, 0xC3):
        buffers.guards["output_bf16"].payload.view(torch.uint8).fill_(seed)
        checks.append(run_nozero(oracle, buffers, stream, expected_seed=seed))
        stream.synchronize()
        result = candidate_result(buffers)
        candidates.append(
            {
                "output": result["output"].clone(),
                "state": result["state"].clone(),
                "guards": result["guards"],
            }
        )
    after = buffers.input_hashes()
    atlas = [references.atlas_once(fixture) for _ in range(2)]
    sglang = [references.sglang_once(fixture) for _ in range(2)]
    stream.synchronize()
    candidate_ids = [_identity(value) for value in candidates]
    atlas_ids = [_identity(value) for value in atlas]
    sglang_ids = [_identity(value) for value in sglang]
    if (
        before != after
        or candidate_ids[0] != candidate_ids[1]
        or atlas_ids[0] != atlas_ids[1]
        or sglang_ids[0] != sglang_ids[1]
        or not buffers.all_canaries_clean()
        or not all(all(value.values()) for value in checks)
    ):
        raise GateError("nozero correctness/seed/replay gate failed")
    comparisons = {
        "candidate_vs_atlas": _comparison(candidates[0], atlas[0]),
        "candidate_vs_sglang": _comparison(candidates[0], sglang[0]),
    }
    for name, value in comparisons.items():
        metric_pair(value, name)
    return {
        "seed_patterns": [0x6A, 0xC3],
        "candidate_identity": candidate_ids[0],
        "atlas_identity": atlas_ids[0],
        "sglang_identity": sglang_ids[0],
        "comparisons": comparisons,
        "checks": checks,
        "immutable_inputs": True,
        "canaries_clean": True,
    }


def _candidate_nozero(
    oracle: FrozenOracle, buffers: FrozenBuffers, stream: torch.cuda.Stream
) -> Callable[[], dict]:
    def invoke() -> dict:
        run_nozero(oracle, buffers, stream)
        return candidate_result(buffers)

    return invoke


def _candidate_zero(
    oracle: FrozenOracle, buffers: FrozenBuffers, stream: torch.cuda.Stream
) -> Callable[[], dict]:
    def invoke() -> dict:
        oracle.run(buffers, stream, capture=False, verify=False)
        return candidate_result(buffers)

    return invoke


def timing(
    oracle: FrozenOracle,
    references: References,
    fixture: dict[str, torch.Tensor],
    stream: torch.cuda.Stream,
) -> dict:
    arms = {
        "nozero": _candidate_nozero(oracle, FrozenBuffers(fixture, oracle.m), stream),
        "zero": _candidate_zero(oracle, FrozenBuffers(fixture, oracle.m), stream),
        "atlas": references.timed_atlas(fixture),
        "sglang": references.timed_sglang(fixture),
    }
    for _ in range(2):
        for invoke in arms.values():
            if not result_clean(invoke()):
                raise GateError("nozero timing warmup guard failure")
            stream.synchronize()
    values = {name: [] for name in arms}
    positions = {name: [0, 0, 0, 0] for name in arms}
    names = tuple(arms)
    for repetition in range(24):
        order = names[repetition % 4 :] + names[: repetition % 4]
        for position, name in enumerate(order):
            positions[name][position] += 1
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream)
            result = arms[name]()
            end.record(stream)
            end.synchronize()
            values[name].append(start.elapsed_time(end))
            if not result_clean(result):
                raise GateError(f"{name}: timing guard failure")
    if any(count != 6 for counts in positions.values() for count in counts):
        raise GateError(f"nozero timing position imbalance: {positions!r}")
    medians = {name: statistics.median(value) for name, value in values.items()}
    p90s = {name: p90(value) for name, value in values.items()}
    paired_atlas = statistics.median(
        a - c for a, c in zip(values["atlas"], values["nozero"])
    )
    paired_zero = statistics.median(
        z - c for z, c in zip(values["zero"], values["nozero"])
    )
    ratio = medians["nozero"] / medians["sglang"]
    predicates = {
        "absolute": medians["nozero"] < (2.5 if oracle.m == 2_079 else 10.0),
        "atlas_median": medians["nozero"] < medians["atlas"],
        "atlas_p90": p90s["nozero"] < p90s["atlas"],
        "paired_atlas": paired_atlas > 0.0,
        "sglang_ratio": ratio <= 1.20,
        "zero_median": medians["nozero"] < medians["zero"],
        "zero_p90": p90s["nozero"] < p90s["zero"],
        "paired_zero": paired_zero > 0.0,
    }
    if not all(predicates.values()):
        raise GateError(
            f"nozero strict timing failed: medians={medians!r} p90s={p90s!r} "
            f"ratio={ratio!r} paired_atlas={paired_atlas!r} "
            f"paired_zero={paired_zero!r} predicates={predicates!r}"
        )
    return {
        "samples_ms": values,
        "position_counts": positions,
        "medians_ms": medians,
        "p90_ms": p90s,
        "candidate_to_sglang_ratio": ratio,
        "paired_atlas_gain_ms": paired_atlas,
        "paired_zero_gain_ms": paired_zero,
        "predicates": predicates,
    }


def run_shape(driver, attestation, references, m: int, stream) -> dict:
    oracle = FrozenOracle(driver, attestation, m)
    with torch.cuda.stream(stream):
        evidence = [
            correctness(oracle, references, make_fixture(m, kind), stream)
            for kind in ("real", "adversarial")
        ]
        performance = timing(oracle, references, make_fixture(m, "real"), stream)
    stream.synchronize()
    return {"m": m, "correctness": evidence, "timing": performance}
