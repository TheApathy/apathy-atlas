"""Autotune, source, compiler, and license provenance attestation."""

import statistics
from pathlib import Path
from typing import Any

from frozen_triton_c143_artifact import artifact_record
from frozen_triton_c143_io import (
    GateError,
    decode_json,
    require_exact_keys,
    require_number,
    require_sha256,
    stable_read,
)


def attest_autotune(value: Any, attested: dict[str, dict[str, Any]]) -> None:
    autotune = require_exact_keys(value, {"path", "sha256", "selected"}, "autotune")
    path = Path(autotune["path"])
    expected = Path(
        "/home/flocka/.cache/sglang/triton/25SCUXQIWZDCXKGJ6MU3YSPICESKXIZ5ZPWZ4LVGZKFKGUKLOBIA/chunk_gated_delta_rule_fwd_kkt_solve_kernel.autotune.json"
    )
    if path != expected:
        raise GateError("autotune path drift")
    raw, actual_sha, file_stat = stable_read(path, "autotune", maximum_size=1 << 20)
    if actual_sha != require_sha256(autotune["sha256"], "autotune.sha256"):
        raise GateError("autotune SHA256 mismatch")
    tune = decode_json(raw, "autotune")
    expected_key = [
        48,
        16,
        128,
        16,
        "torch.bfloat16",
        "torch.float32",
        "torch.float32",
        "torch.bfloat16",
        "torch.int32",
        "torch.int32",
    ]
    if tune.get("key") != expected_key:
        raise GateError("autotune key drift")
    selected = autotune["selected"]
    if selected != {
        "BK": 64,
        "num_warps": 1,
        "num_stages": 3,
        "median_ms": 0.02252800017595291,
    }:
        raise GateError("autotune selection drift")
    candidates = []
    for config, timings in tune.get("configs_timings", []):
        values = [
            require_number(item, "autotune timing", positive=True) for item in timings
        ]
        candidates.append((statistics.median(values), config))
    if not candidates:
        raise GateError("autotune candidates missing")
    best_median, best = min(candidates, key=lambda pair: pair[0])
    if (
        best.get("kwargs") != {"BK": 64}
        or best.get("num_warps") != 1
        or best.get("num_stages") != 3
        or best_median != selected["median_ms"]
    ):
        raise GateError(
            "autotune selected configuration is not the pinned fastest median"
        )
    attested[str(path)] = {"sha256": actual_sha, "bytes": file_stat.st_size}


def _attest_records(
    root: Path,
    records: dict[str, Any],
    label: str,
    attested: dict[str, dict[str, Any]],
) -> None:
    for relative, record in records.items():
        expected_sha, expected_size = artifact_record(record, f"{label}:{relative}")
        path = root / relative
        _, actual_sha, file_stat = stable_read(
            path, f"{label}:{relative}", expected_size=expected_size
        )
        if actual_sha != expected_sha:
            raise GateError(f"{label}:{relative}: SHA256 mismatch")
        attested[str(path)] = {"sha256": actual_sha, "bytes": file_stat.st_size}


def attest_provenance(value: Any, attested: dict[str, dict[str, Any]]) -> None:
    provenance = require_exact_keys(
        value,
        {
            "sglang_root",
            "sglang_commit",
            "sglang_license",
            "fla_adaptation_notice",
            "audited_sources",
            "compiler_inputs",
            "triton_version",
            "triton_license",
            "target",
            "ptx_version",
            "redistribution_legal_review",
        },
        "provenance",
    )
    root = Path(provenance["sglang_root"])
    commit = provenance["sglang_commit"]
    if commit != "c14312a66420b75ca9a11bf1817c4db1fa26b097" or root != Path(
        f"/tmp/sglang-{commit}"
    ):
        raise GateError("SGLang root/commit drift")
    head_raw, _, _ = stable_read(root / ".git/HEAD", "SGLang HEAD", maximum_size=128)
    if head_raw.decode("ascii").strip() != commit:
        raise GateError("SGLang HEAD drift")
    expected_fields = {
        "sglang_license": "Apache-2.0",
        "triton_license": "MIT",
        "triton_version": "3.6.0",
        "target": "sm_121a",
        "ptx_version": "8.8",
        "redistribution_legal_review": "REQUIRED_BEFORE_DISTRIBUTION",
        "fla_adaptation_notice": "Copyright (c) 2023-2025, Songlin Yang, Yu Zhang",
    }
    if any(provenance[name] != expected for name, expected in expected_fields.items()):
        raise GateError("provenance/license policy drift")
    audited = provenance["audited_sources"]
    audited_paths = {
        "LICENSE",
        "python/sglang/kernels/ops/attention/fla/chunk.py",
        "python/sglang/kernels/ops/attention/fla/chunk_fwd.py",
        "python/sglang/kernels/ops/attention/fla/chunk_delta_h.py",
        "python/sglang/kernels/ops/attention/fla/chunk_o.py",
        "python/sglang/kernels/ops/attention/fla/cumsum.py",
        "python/sglang/kernels/ops/attention/fla/wy_fast.py",
        "python/sglang/kernels/ops/attention/fla/index.py",
        "python/sglang/kernels/ops/attention/fla/op.py",
        "python/sglang/kernels/ops/attention/fla/utils.py",
    }
    if type(audited) is not dict or set(audited) != audited_paths:
        raise GateError("audited SGLang source set drift")
    _attest_records(root, audited, "source", attested)
    compiler = provenance["compiler_inputs"]
    compiler_paths = {
        "/home/flocka/.local/lib/python3.12/site-packages/triton-3.6.0.dist-info/METADATA",
        "/home/flocka/.local/lib/python3.12/site-packages/triton-3.6.0.dist-info/licenses/LICENSE",
        "/home/flocka/.local/lib/python3.12/site-packages/triton/backends/nvidia/lib/libdevice.10.bc",
    }
    if type(compiler) is not dict or set(compiler) != compiler_paths:
        raise GateError("compiler input set drift")
    _attest_records(
        Path("/"),
        {path.lstrip("/"): record for path, record in compiler.items()},
        "compiler",
        attested,
    )
