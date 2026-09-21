"""Correctness and timing screens for raw-oracle receipts."""

import math
import statistics
from typing import Any

from frozen_triton_c143_constants import STAGE_NAMES
from frozen_triton_c143_io import (
    GateError,
    require_exact_keys,
    require_int,
    require_number,
)


def timing_values(value: Any, label: str, minimum: int) -> list[float]:
    if type(value) is not list or len(value) < minimum:
        raise GateError(f"{label}: requires at least {minimum} samples")
    return [
        require_number(item, f"{label}[{index}]", positive=True)
        for index, item in enumerate(value)
    ]


def p90(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(0.9 * len(ordered)) - 1]


def balanced_positions(value: Any, arms: list[str], repetitions: int) -> bool:
    positions = require_exact_keys(value, set(arms), "position counts")
    columns = [0, 0, 0]
    for arm in arms:
        counts = positions[arm]
        if type(counts) is not list or len(counts) != 3:
            return False
        counts = [require_int(count, f"position_counts.{arm}") for count in counts]
        if sum(counts) != repetitions or max(counts) - min(counts) > 1:
            return False
        columns = [total + count for total, count in zip(columns, counts)]
    return all(total == repetitions for total in columns)


def metric_pair(value: Any, label: str) -> None:
    pair = require_exact_keys(value, {"output", "state"}, label)
    for name in ("output", "state"):
        metric = require_exact_keys(
            pair[name], {"max_abs", "rms", "relative_rms", "cosine"}, f"{label}.{name}"
        )
        for key in ("max_abs", "rms", "relative_rms"):
            if require_number(metric[key], f"{label}.{name}.{key}") < 0.0:
                raise GateError(f"{label}.{name}.{key}: must be nonnegative")
        cosine = require_number(metric["cosine"], f"{label}.{name}.cosine")
        if not -1.0 <= cosine <= 1.0:
            raise GateError(f"{label}.{name}: cosine outside [-1,1]")
        if cosine < 0.999:
            raise GateError(f"{label}.{name}: cosine below 0.999")
        if metric["relative_rms"] > 0.01:
            raise GateError(f"{label}.{name}: relative RMS exceeds 0.01")


def validate_metrics_and_timing(value: dict[str, Any], m: int) -> dict[str, float]:
    comparisons = require_exact_keys(
        value["comparisons"],
        {"candidate_vs_sglang", "candidate_vs_atlas"},
        "comparisons",
    )
    metric_pair(comparisons["candidate_vs_sglang"], "candidate_vs_sglang")
    metric_pair(comparisons["candidate_vs_atlas"], "candidate_vs_atlas")
    stage_timing = require_exact_keys(
        value["stage_timing_ms"], set(STAGE_NAMES), "stage timing"
    )
    stage_values = {
        stage: timing_values(stage_timing[stage], f"stage timing:{stage}", 21)
        for stage in STAGE_NAMES
    }
    full = require_exact_keys(
        value["full_timing"],
        {"samples_ms", "position_counts", "scope_includes"},
        "full timing",
    )
    arms = ["candidate", "atlas", "sglang"]
    samples = require_exact_keys(full["samples_ms"], set(arms), "full timing samples")
    values = {
        arm: timing_values(samples[arm], f"full timing:{arm}", 22) for arm in arms
    }
    repetitions = len(values["candidate"])
    if any(len(values[arm]) != repetitions for arm in arms):
        raise GateError("full timing sample count mismatch")
    if not balanced_positions(full["position_counts"], arms, repetitions):
        raise GateError("full timing positions are not balanced alternating")
    if full["scope_includes"] != [
        "all_adapters",
        "A_and_output_initialization",
        "five_triton_kernels",
        "both_state_transposes",
    ]:
        raise GateError("full timing excludes required work")
    candidate_median = statistics.median(values["candidate"])
    atlas_median = statistics.median(values["atlas"])
    sglang_median = statistics.median(values["sglang"])
    candidate_p90 = p90(values["candidate"])
    atlas_p90 = p90(values["atlas"])
    paired_gain = statistics.median(
        atlas - candidate
        for atlas, candidate in zip(values["atlas"], values["candidate"])
    )
    ratio = candidate_median / sglang_median
    metrics = {
        "candidate_median_ms": candidate_median,
        "candidate_p90_ms": candidate_p90,
        "atlas_median_ms": atlas_median,
        "atlas_p90_ms": atlas_p90,
        "sglang_median_ms": sglang_median,
        "candidate_to_sglang_ratio": ratio,
        "paired_atlas_gain_median_ms": paired_gain,
    }
    predicates = {
        "absolute_median": candidate_median < (2.5 if m == 2_079 else 10.0),
        "atlas_median": candidate_median < atlas_median,
        "atlas_p90": candidate_p90 < atlas_p90,
        "paired_gain": paired_gain > 0.0,
        "sglang_ratio": ratio <= 1.20,
    }
    if not all(predicates.values()):
        stage_medians = {
            stage: statistics.median(stage_values[stage]) for stage in STAGE_NAMES
        }
        raise GateError(
            f"strict full-route performance screen failed: "
            f"metrics={metrics!r} predicates={predicates!r} "
            f"stage_medians_ms={stage_medians!r}"
        )
    return metrics
