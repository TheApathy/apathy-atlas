"""Exact compiled-kernel ABI and artifact attestation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from frozen_triton_c143_constants import KERNEL_CONTRACTS
from frozen_triton_c143_io import (
    GateError,
    decode_json,
    require_exact_keys,
    require_int,
    require_sha256,
    stable_read,
)


def artifact_record(record: Any, label: str) -> tuple[str, int]:
    if type(record) is not list or len(record) != 2:
        raise GateError(f"{label}: expected [sha256,size]")
    return require_sha256(record[0], f"{label}.sha256"), require_int(
        record[1], f"{label}.size", minimum=1
    )


def ptx_driver_types(raw: bytes, function: str, expected_block_x: int) -> list[str]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GateError(f"{function}: PTX is not UTF-8") from exc
    if ".version 8.8" not in text or ".target sm_121a" not in text:
        raise GateError(f"{function}: PTX version/target drift")
    marker = f".visible .entry {function}("
    start = text.find(marker)
    if start < 0:
        raise GateError(f"{function}: PTX entry missing")
    end = text.find("\n)", start)
    if end < 0:
        raise GateError(f"{function}: PTX parameter list unterminated")
    body = text[start:end]
    types = re.findall(r"\.param\s+\.(u64|u32|f32)\b[^,\n]*\b\w+_param_\d+\b", body)
    match = re.search(r"\.reqntid\s+(\d+)(?:,\s*(\d+),\s*(\d+))?", text[end:])
    block = None if not match else tuple(int(item or 1) for item in match.groups())
    if block != (expected_block_x, 1, 1):
        raise GateError(f"{function}: PTX block geometry drift")
    return types


def validate_metadata(
    metadata: Any,
    manifest_metadata: Any,
    contract: dict[str, Any],
    role: str,
) -> None:
    value = require_exact_keys(
        metadata,
        {
            "hash",
            "target",
            "num_warps",
            "num_ctas",
            "num_stages",
            "warp_size",
            "maxnreg",
            "ptx_version",
            "ptx_options",
            "ir_override",
            "enable_fp_fusion",
            "enable_reflect_ftz",
            "launch_cooperative_grid",
            "launch_pdl",
            "supported_fp8_dtypes",
            "deprecated_fp8_dot_operand_dtypes",
            "default_dot_input_precision",
            "allowed_dot_input_precisions",
            "max_num_imprecise_acc_default",
            "extern_libs",
            "debug",
            "backend_name",
            "sanitize_overflow",
            "arch",
            "instrumentation_mode",
            "triton_version",
            "tensordesc_meta",
            "shared",
            "tmem_size",
            "global_scratch_size",
            "global_scratch_align",
            "profile_scratch_size",
            "profile_scratch_align",
            "name",
        },
        f"{role}.json",
    )
    expected_manifest = {
        "name": contract["function"],
        "target": {"backend": "cuda", "arch": 121, "warp_size": 32},
        "arch": "sm121",
        "triton_version": "3.6.0",
        "num_warps": contract["warps"],
        "num_stages": contract["stages"],
        "num_ctas": 1,
        "shared": contract["shared"],
        "tmem_size": 0,
        "global_scratch_size": 0,
        "profile_scratch_size": 0,
    }
    if manifest_metadata != expected_manifest:
        raise GateError(f"{role}: manifest metadata drift")
    if any(value[key] != expected for key, expected in expected_manifest.items()):
        raise GateError(f"{role}: compiled metadata resource drift")
    if value["global_scratch_align"] != 1 or value["profile_scratch_align"] != 1:
        raise GateError(f"{role}: compiled metadata drift")


def attest_kernels(kernels: Any, attested: dict[str, dict[str, Any]]) -> None:
    if type(kernels) is not list or len(kernels) != len(KERNEL_CONTRACTS):
        raise GateError("kernels: exact five-kernel family required")
    if [kernel.get("role") for kernel in kernels] != list(KERNEL_CONTRACTS):
        raise GateError("kernels: role/order drift")
    root = Path("/home/flocka/.cache/sglang/triton")
    exact_keys = {
        "role",
        "artifact_dir",
        "function",
        "files",
        "semantic_params",
        "driver_params",
        "grid",
        "block",
        "dynamic_shared_bytes",
        "static_shared_bytes",
        "registers_per_thread",
        "stack_bytes",
        "local_bytes",
        "metadata",
    }
    for kernel in kernels:
        item = require_exact_keys(
            kernel, exact_keys, f"kernel[{kernel.get('role', '?')}]"
        )
        role = item["role"]
        contract = KERNEL_CONTRACTS[role]
        artifact_dir = Path(item["artifact_dir"])
        if artifact_dir != root / contract["dir"]:
            raise GateError(f"{role}: artifact directory drift")
        if item["function"] != contract["function"]:
            raise GateError(f"{role}: function drift")
        if item["semantic_params"] != contract["semantic"]:
            raise GateError(f"{role}: semantic ABI drift")
        driver = item["driver_params"]
        if (
            type(driver) is not list
            or [entry[1] for entry in driver] != contract["driver"]
        ):
            raise GateError(f"{role}: driver ABI drift")
        if [entry[0] for entry in driver[-2:]] != [
            "global_scratch_null",
            "profile_scratch_null",
        ]:
            raise GateError(f"{role}: missing trailing null scratch ABI slots")
        if item["grid"] != contract["grid"] or item["block"] != contract["block"]:
            raise GateError(f"{role}: launch geometry drift")
        if item["dynamic_shared_bytes"] != contract["shared"]:
            raise GateError(f"{role}: dynamic shared drift")
        if item["static_shared_bytes"] != 1_024:
            raise GateError(f"{role}: static shared drift")
        if item["registers_per_thread"] != contract["registers"]:
            raise GateError(f"{role}: register record drift")
        if item["stack_bytes"] != 0 or item["local_bytes"] != 0:
            raise GateError(f"{role}: stack/local record drift")
        files = require_exact_keys(
            item["files"],
            {"cubin", "ptx", "json", "source", "ttir", "ttgir", "llir"},
            f"{role}.files",
        )
        raw_files = {}
        for extension, record in files.items():
            expected_sha, expected_size = artifact_record(record, f"{role}.{extension}")
            path = artifact_dir / f"{contract['function']}.{extension}"
            raw, actual_sha, file_stat = stable_read(
                path, f"{role}.{extension}", expected_size=expected_size
            )
            if actual_sha != expected_sha:
                raise GateError(
                    f"{role}.{extension}: SHA256 mismatch; expected={expected_sha} actual={actual_sha}"
                )
            raw_files[extension] = raw
            attested[str(path)] = {"sha256": actual_sha, "bytes": file_stat.st_size}
        if (
            ptx_driver_types(
                raw_files["ptx"], contract["function"], contract["block"][0]
            )
            != contract["driver"]
        ):
            raise GateError(f"{role}: compiled PTX driver ABI drift")
        validate_metadata(
            decode_json(raw_files["json"], f"{role}.json"),
            item["metadata"],
            contract,
            role,
        )
