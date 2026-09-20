# SPDX-License-Identifier: AGPL-3.0-only
"""Authorization, proof, execution, and exclusive nozero publication."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import frozen_triton_c143_gate as sealed_gate
from frozen_triton_c143_io import GateError
from frozen_triton_c143_output_overwrite import EXPECTED_SHA256, prove
from frozen_triton_executor.contract import (
    local_identity,
    sha256_file,
    validate_contract,
)
from frozen_triton_executor.main import parse_args


UPSTREAM = Path(
    "/tmp/sglang-c14312a66420b75ca9a11bf1817c4db1fa26b097/python/sglang/kernels/ops/attention/fla/chunk_o.py"
)
ARTIFACT_DIR = Path(
    "/home/flocka/.cache/sglang/triton/KTFBEMQNV7CPTR5VWT4OH2W5U4AWJVLJ2QV3CEDL433KBK2JBWNA"
)


def _attestation(attestation: dict, identity: dict) -> dict:
    return {
        "schema": "atlas.gdn_c143.frozen_triton_nozero_attestation.v1",
        "mode": "ATTEST_ONLY",
        "qualification": "ATTESTED_CPU_ONLY",
        "gpu_execution": False,
        "production_authorized": False,
        "default_off": True,
        "variant": "sole-delta-skip-output-clear",
        "manifest_sha256": attestation["manifest_sha256"],
        "artifact_attestation_sha256": attestation["artifact_attestation_sha256"],
        "gate_source_sha256": sealed_gate._gate_source_sha256(),
        "executor_source_sha256": identity["executor_source_sha256"],
        "m_sequence": [2_079, 8_192],
    }


def _execute(args, attestation: dict) -> dict:
    if os.environ.get("ATLAS_GDN_C143_NOZERO_RAW_GPU") != "1":
        raise GateError("nozero GPU environment gate is not exactly 1")
    if None in (
        args.gpu_execution_flag,
        args.reservation_nonce,
        args.reservation,
        args.execution_contract,
    ):
        raise GateError("GPU execution requires flag, nonce, reservation, and contract")
    gate_sha = sealed_gate._gate_source_sha256()
    authorization = sealed_gate.authorize_gpu_execution(
        explicit_flag=args.gpu_execution_flag,
        nonce=args.reservation_nonce,
        reservation_path=args.reservation,
        manifest_sha256=attestation["manifest_sha256"],
    )
    contract, contract_sha = validate_contract(
        args.execution_contract,
        attestation,
        gate_sha,
        authorization,
        args.reservation_nonce,
    )
    proofs = [prove(UPSTREAM, ARTIFACT_DIR, m) for m in (2_079, 8_192)]
    import torch

    from frozen_triton_executor.cuda_driver import CudaDriver
    from frozen_triton_executor.publication import publish_exclusive
    from frozen_triton_executor.references import References

    from .case import run_shape

    driver = CudaDriver()
    device = driver.device_identity()
    properties = torch.cuda.get_device_properties(0)
    if (
        properties.major,
        properties.minor,
        properties.multi_processor_count,
    ) != (12, 1, 48) or "GB10" not in properties.name:
        raise GateError("Torch and CUDA-driver device identity mismatch")
    stream = torch.cuda.Stream(device=0)
    if int(stream.cuda_stream) == 0:
        raise GateError("nozero executor requires a nondefault stream")
    references = References()
    cases = [
        run_shape(driver, attestation, references, m, stream) for m in (2_079, 8_192)
    ]
    stream.synchronize()
    if [case["m"] for case in cases] != [2_079, 8_192]:
        raise GateError("nozero M sequence drift")
    post_attestation = sealed_gate.attest_manifest()
    if post_attestation != attestation:
        raise GateError("post-run manifest/artifact attestation drift")
    post_gate_sha = sealed_gate._gate_source_sha256()
    current = local_identity(attestation, post_gate_sha)
    if post_gate_sha != gate_sha or any(
        key in current and current[key] != value for key, value in contract.items()
    ):
        raise GateError("post-run source identity drift")
    if (
        sha256_file(args.reservation.resolve(strict=True), "post-run reservation")
        != authorization["reservation_receipt_sha256"]
    ):
        raise GateError("post-run reservation drift")
    if (
        sha256_file(args.execution_contract.resolve(strict=True), "post-run contract")
        != contract_sha
    ):
        raise GateError("post-run contract drift")
    receipt = {
        "schema": "atlas.gdn_c143.frozen_triton_nozero_raw_receipt.v1",
        "qualification": "PASS",
        "production_authorized": False,
        "default_off": True,
        "variant": "sole-delta-skip-output-clear",
        "execution_contract_sha256": contract_sha,
        "manifest_sha256": attestation["manifest_sha256"],
        "artifact_attestation_sha256": attestation["artifact_attestation_sha256"],
        "gate_source_sha256": gate_sha,
        "executor_source_sha256": contract["executor_source_sha256"],
        "reservation_receipt_sha256": authorization["reservation_receipt_sha256"],
        "execution_contract": contract,
        "device": device,
        "m_sequence": [2_079, 8_192],
        "full_write_artifact_sha256": EXPECTED_SHA256,
        "full_write_proofs": proofs,
        "cases": cases,
    }
    driver.close_modules()
    result_path = Path(contract["result_path"])
    receipt_sha = publish_exclusive(result_path, receipt)
    return {
        "schema": "atlas.gdn_c143.frozen_triton_nozero_publication.v1",
        "qualification": "PASS",
        "production_authorized": False,
        "result_path": str(result_path),
        "receipt_sha256": receipt_sha,
    }


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        attestation = sealed_gate.attest_manifest()
        gate_sha = sealed_gate._gate_source_sha256()
        if not args.execute_gpu:
            if any(
                value is not None
                for value in (
                    args.gpu_execution_flag,
                    args.reservation_nonce,
                    args.reservation,
                    args.execution_contract,
                )
            ):
                raise GateError("GPU arguments require --execute-gpu")
            print(
                json.dumps(
                    _attestation(attestation, local_identity(attestation, gate_sha)),
                    sort_keys=True,
                )
            )
            return 0
        print(json.dumps(_execute(args, attestation), sort_keys=True))
        return 0
    except (GateError, OSError, RuntimeError, ValueError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 2
