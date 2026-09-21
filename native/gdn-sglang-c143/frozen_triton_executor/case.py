# SPDX-License-Identifier: AGPL-3.0-only
"""M2079/M8192 correctness, replay, guard, and balanced timing cases."""

from __future__ import annotations

import hashlib

import torch

import frozen_triton_c143_gate as sealed_gate
from frozen_triton_c143_io import GateError

from .buffers import FrozenBuffers
from .cuda_driver import CudaDriver
from .launch import FrozenOracle
from .references import (
    References,
    make_fixture,
    metrics,
    result_identity,
    worst_metrics,
)
from .timing import measure_full_timing, measure_stages

FIXTURES = ("real", "adversarial")


def combine_hashes(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def candidate_result(buffers: FrozenBuffers) -> dict:
    output = buffers.guards["output_bf16"].payload.view(1, buffers.m, 48, 128)
    state_hkv = buffers.guards["atlas_state_hkv_f32"].payload
    state = state_hkv.view(1, 48, 128, 128).transpose(-1, -2).contiguous()
    return {
        "output": output,
        "state": state,
        "guards": tuple(buffers.guards.values()),
    }


def stage_finite(buffers: FrozenBuffers) -> dict[str, bool]:
    values = {
        "g_cumsum": buffers.views["g_cumsum_f32"],
        "A": buffers.views["A_bf16"],
        "w": buffers.views["w_bf16"],
        "u": buffers.views["u_bf16"],
        "h": buffers.views["h_bf16"],
        "v_new": buffers.views["v_new_bf16"],
        "output": buffers.guards["output_bf16"].payload,
        "state_hkv": buffers.guards["atlas_state_hkv_f32"].payload,
    }
    return {
        name: bool(torch.isfinite(value).all().item()) for name, value in values.items()
    }


def correctness_fixture(
    oracle: FrozenOracle,
    references: References,
    fixture: dict[str, torch.Tensor],
    kind: str,
    stream: torch.cuda.Stream,
) -> dict:
    buffers = FrozenBuffers(fixture, oracle.m)
    input_before = buffers.input_hashes()
    first_launch = oracle.run(buffers, stream, capture=True)
    stream.synchronize()
    first_candidate = candidate_result(buffers)
    first_candidate_identity = result_identity(first_candidate)
    first_output = first_candidate["output"].clone()
    first_state = first_candidate["state"].clone()
    finite = stage_finite(buffers)
    second_launch = oracle.run(buffers, stream, capture=True)
    stream.synchronize()
    second_candidate = candidate_result(buffers)
    second_candidate_identity = result_identity(second_candidate)
    input_after = buffers.input_hashes()
    atlas_first = references.atlas_once(fixture)
    sglang_first = references.sglang_once(fixture)
    atlas_second = references.atlas_once(fixture)
    sglang_second = references.sglang_once(fixture)
    stream.synchronize()
    identities = {
        "candidate_run1": first_candidate_identity,
        "candidate_run2": second_candidate_identity,
        "atlas_run1": result_identity(atlas_first),
        "atlas_run2": result_identity(atlas_second),
        "sglang_run1": result_identity(sglang_first),
        "sglang_run2": result_identity(sglang_second),
    }
    comparisons = {
        "candidate_vs_atlas": {
            field: metrics(value, atlas_first[field])
            for field, value in (("output", first_output), ("state", first_state))
        },
        "candidate_vs_sglang": {
            field: metrics(value, sglang_first[field])
            for field, value in (("output", first_output), ("state", first_state))
        },
    }
    deterministic = all(
        identities[f"{arm}_run1"][field] == identities[f"{arm}_run2"][field]
        for arm in ("candidate", "atlas", "sglang")
        for field in ("output_sha256", "state_sha256")
    )
    all_guards = buffers.all_canaries_clean() and all(
        item["guards_clean"] for item in identities.values()
    )
    if (
        input_before != input_after
        or not deterministic
        or not all_guards
        or not all(finite.values())
        or first_launch["stage_hashes"].keys() != second_launch["stage_hashes"].keys()
    ):
        raise GateError(f"{kind}: correctness/guard/replay gate failed")
    return {
        "kind": kind,
        "input_before": input_before,
        "input_after": input_after,
        "stage_run1": first_launch["stage_hashes"],
        "stage_run2": second_launch["stage_hashes"],
        "adapter_checks": {
            name: first_launch["checks"].get(name, False)
            and second_launch["checks"].get(name, False)
            for name in sealed_gate.ADAPTER_CHECKS
        },
        "finite": finite,
        "identities": identities,
        "comparisons": comparisons,
        "deterministic": deterministic,
        "all_guards_clean": all_guards,
    }


def run_shape(
    driver: CudaDriver,
    attestation: dict,
    references: References,
    m: int,
    stream: torch.cuda.Stream,
) -> dict:
    oracle = FrozenOracle(driver, attestation, m)
    fixtures = []
    with torch.cuda.stream(stream):
        for kind in FIXTURES:
            fixtures.append(
                correctness_fixture(
                    oracle, references, make_fixture(m, kind), kind, stream
                )
            )
        real = make_fixture(m, "real")
        stage_timing = measure_stages(oracle, real, stream)
        full_timing, positions = measure_full_timing(oracle, references, real, stream)
    stream.synchronize()
    comparisons = {}
    for reference in ("candidate_vs_sglang", "candidate_vs_atlas"):
        comparisons[reference] = {
            field: worst_metrics(
                [case["comparisons"][reference][field] for case in fixtures]
            )
            for field in ("output", "state")
        }
    stages = {
        name: {
            "run1": combine_hashes([case["stage_run1"][name] for case in fixtures]),
            "run2": combine_hashes([case["stage_run2"][name] for case in fixtures]),
        }
        for name in sealed_gate.STAGE_NAMES
    }
    inputs_before = {
        name: combine_hashes([case["input_before"][name] for case in fixtures])
        for name in sealed_gate.IMMUTABLE_NAMES
    }
    inputs_after = {
        name: combine_hashes([case["input_after"][name] for case in fixtures])
        for name in sealed_gate.IMMUTABLE_NAMES
    }
    return {
        "m": m,
        "fixtures": fixtures,
        "comparisons": comparisons,
        "stage_hashes": stages,
        "input_hashes_before": inputs_before,
        "input_hashes_after": inputs_after,
        "adapter_checks": {
            name: all(case["adapter_checks"][name] for case in fixtures)
            for name in sealed_gate.ADAPTER_CHECKS
        },
        "finite": {
            name: all(case["finite"][name] for case in fixtures)
            for name in sealed_gate.FINITE_NAMES
        },
        "canaries_clean": all(case["all_guards_clean"] for case in fixtures),
        "stage_timing_ms": stage_timing,
        "full_timing": {
            "samples_ms": full_timing,
            "position_counts": positions,
            "scope_includes": [
                "all_adapters",
                "A_and_output_initialization",
                "five_triton_kernels",
                "both_state_transposes",
            ],
        },
    }
