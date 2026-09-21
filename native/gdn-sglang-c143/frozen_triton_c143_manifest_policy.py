"""Static manifest policy validation for the frozen oracle."""

from typing import Any

from frozen_triton_c143_constants import (
    EXPECTED_GEOMETRY,
    EXPECTED_SEQUENCE,
    IMMUTABLE_NAMES,
    STAGE_NAMES,
)
from frozen_triton_c143_io import GateError, require_exact_keys

MANIFEST_KEYS = {
    "schema",
    "candidate",
    "artifact_policy",
    "execution_policy",
    "geometry",
    "same_stream_sequence",
    "adapter_contracts",
    "runtime_scalar_contracts",
    "kernels",
    "autotune",
    "provenance",
    "workspace_layouts",
    "receipt_contract",
}


def validate_manifest_policy(manifest: Any) -> dict[str, Any]:
    value = require_exact_keys(manifest, MANIFEST_KEYS, "manifest")
    if value["schema"] != "atlas.gdn_c143.frozen_triton_manifest.v1":
        raise GateError("manifest schema drift")
    if value["candidate"] != "qwen38-gdn-c143-frozen-triton-full-family-raw-oracle":
        raise GateError("manifest candidate drift")
    policy = require_exact_keys(
        value["artifact_policy"],
        {
            "external_reference_only",
            "mutable_cache_is_not_runtime_authority",
            "copy_or_redistribute_artifact_bytes",
            "ptx_runtime_fallback",
            "production_authorized",
        },
        "artifact_policy",
    )
    if policy != {
        "external_reference_only": True,
        "mutable_cache_is_not_runtime_authority": True,
        "copy_or_redistribute_artifact_bytes": False,
        "ptx_runtime_fallback": False,
        "production_authorized": False,
    }:
        raise GateError("artifact policy is not fail closed")
    execution = require_exact_keys(
        value["execution_policy"],
        {
            "default_mode",
            "gpu_executor_implemented",
            "gpu_requires_explicit_flag",
            "gpu_requires_environment",
            "gpu_requires_nonce",
            "gpu_requires_reservation_receipt",
            "supported_m",
            "unqualified_m",
        },
        "execution_policy",
    )
    if execution != {
        "default_mode": "attest_only",
        "gpu_executor_implemented": False,
        "gpu_requires_explicit_flag": "I_UNDERSTAND_THIS_RUNS_CUDA",
        "gpu_requires_environment": "ATLAS_GDN_C143_TRITON_RAW_GPU=1",
        "gpu_requires_nonce": True,
        "gpu_requires_reservation_receipt": True,
        "supported_m": [2_079, 8_192],
        "unqualified_m": [32_768],
    }:
        raise GateError("execution policy drift")
    if value["geometry"] != EXPECTED_GEOMETRY:
        raise GateError("geometry drift")
    if value["same_stream_sequence"] != EXPECTED_SEQUENCE:
        raise GateError("same-stream sequence drift")
    adapters = require_exact_keys(
        value["adapter_contracts"],
        {
            "adapter_qkv_split",
            "adapter_alpha_log_beta_split",
            "adapter_state_hkv_to_hvk",
            "adapter_state_hvk_to_hkv",
            "initialization",
            "ordering",
        },
        "adapter_contracts",
    )
    if "identical nondefault caller stream" not in adapters["ordering"]:
        raise GateError("adapter ordering lost same nondefault stream")
    if "A is partially written and must be zeroed" not in adapters["initialization"]:
        raise GateError("A initialization contract drift")
    scalar = require_exact_keys(
        value["runtime_scalar_contracts"],
        {
            "T_i32",
            "NT",
            "cu_seqlens_i32",
            "state_index_i32",
            "chunk_indices_i32",
            "chunk_offsets_i64",
            "stride_init_state_i32",
            "scale_f32",
            "global_scratch_u64",
            "profile_scratch_u64",
        },
        "runtime_scalar_contracts",
    )
    if scalar != {
        "T_i32": "M",
        "NT": "ceil(M/64)",
        "cu_seqlens_i32": "[0,M]",
        "state_index_i32": "[0]",
        "chunk_indices_i32": "[[0,i] for i in range(NT)]",
        "chunk_offsets_i64": "[0,NT]",
        "stride_init_state_i32": 786_432,
        "scale_f32": 0.08838834764831845,
        "global_scratch_u64": 0,
        "profile_scratch_u64": 0,
    }:
        raise GateError("runtime scalar/metadata contract drift")
    return value


def validate_receipt_contract(value: Any) -> None:
    contract = require_exact_keys(
        value,
        {
            "schema",
            "qualification",
            "production_authorized",
            "stage_names",
            "immutable_inputs",
            "mutable_outputs",
            "min_cosine",
            "max_relative_rms",
            "minimum_stage_pairs",
            "minimum_full_repetitions",
            "median_limit_ms",
            "candidate_to_sglang_median_ratio_max",
            "require_candidate_median_lt_atlas",
            "require_candidate_p90_lt_atlas",
            "require_positive_paired_atlas_median_gain",
            "require_balanced_alternating_positions",
            "full_timing_includes",
        },
        "receipt_contract",
    )
    expected = {
        "schema": "atlas.gdn_c143.frozen_triton_raw_receipt.v1",
        "qualification": "PASS",
        "production_authorized": False,
        "stage_names": STAGE_NAMES,
        "immutable_inputs": IMMUTABLE_NAMES,
        "mutable_outputs": ["atlas_state_hkv_f32", "output_bf16"],
        "min_cosine": 0.999,
        "max_relative_rms": 0.01,
        "minimum_stage_pairs": 21,
        "minimum_full_repetitions": 22,
        "median_limit_ms": {"2079": 2.5, "8192": 10.0},
        "candidate_to_sglang_median_ratio_max": 1.2,
        "require_candidate_median_lt_atlas": True,
        "require_candidate_p90_lt_atlas": True,
        "require_positive_paired_atlas_median_gain": True,
        "require_balanced_alternating_positions": True,
        "full_timing_includes": [
            "all_adapters",
            "A_and_output_initialization",
            "five_triton_kernels",
            "both_state_transposes",
        ],
    }
    if contract != expected:
        raise GateError("receipt_contract: frozen gate drift")
