# SPDX-License-Identifier: AGPL-3.0-only
"""Fail-closed command line for the isolated frozen-cubin executor."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import frozen_triton_c143_gate as sealed_gate
from frozen_triton_c143_io import GateError

from .contract import local_identity, sha256_file, validate_contract


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--attest-only", action="store_true")
    modes.add_argument("--execute-gpu", action="store_true")
    parser.add_argument("--gpu-execution-flag")
    parser.add_argument("--reservation-nonce")
    parser.add_argument("--reservation", type=Path)
    parser.add_argument("--execution-contract", type=Path)
    return parser.parse_args(argv)


def attest_only(attestation: dict, identity: dict) -> dict:
    return {
        "schema": "atlas.gdn_c143.frozen_triton_executor_attestation.v1",
        "mode": "ATTEST_ONLY",
        "qualification": "ATTESTED_CPU_ONLY",
        "gpu_execution": False,
        "executor_implemented": True,
        "production_authorized": False,
        "default_off": True,
        "manifest_sha256": attestation["manifest_sha256"],
        "artifact_attestation_sha256": attestation["artifact_attestation_sha256"],
        "gate_source_sha256": sealed_gate._gate_source_sha256(),
        "executor_source_sha256": identity["executor_source_sha256"],
        "m_sequence": [2_079, 8_192],
        "parent_comparison": "required_in_process_before_PASS",
    }


def execute(args: argparse.Namespace, attestation: dict) -> dict:
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
    # GPU-bearing imports are deliberately unreachable before authorization.
    import torch

    from .case import run_shape
    from .cuda_driver import CudaDriver
    from .publication import publish_exclusive
    from .receipt import build_envelope
    from .references import References

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
    stream_handle = int(stream.cuda_stream)
    if stream_handle == 0:
        raise GateError("executor requires a nondefault CUDA stream")
    references = References()
    cases = [
        run_shape(driver, attestation, references, m, stream) for m in (2_079, 8_192)
    ]
    stream.synchronize()
    post_gate_sha = sealed_gate._gate_source_sha256()
    if post_gate_sha != gate_sha:
        raise GateError("post-run sealed gate source drift")
    after = local_identity(attestation, post_gate_sha)
    for key, value in contract.items():
        if key in after and after[key] != value:
            raise GateError(f"post-run local identity drift: {key}")
    if (
        sha256_file(args.reservation.resolve(strict=True), "post-run reservation")
        != authorization["reservation_receipt_sha256"]
    ):
        raise GateError("post-run reservation receipt drift")
    if (
        sha256_file(
            args.execution_contract.resolve(strict=True), "post-run execution contract"
        )
        != contract_sha
    ):
        raise GateError("post-run execution contract drift")
    envelope = build_envelope(
        cases,
        attestation,
        contract,
        contract_sha,
        authorization,
        device,
        stream_handle,
    )
    driver.close_modules()
    receipt_sha = publish_exclusive(Path(contract["result_path"]), envelope)
    return {
        "schema": "atlas.gdn_c143.frozen_triton_executor_publication.v1",
        "qualification": "PASS",
        "production_authorized": False,
        "result_path": contract["result_path"],
        "receipt_sha256": receipt_sha,
        "m_sequence": [2_079, 8_192],
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
                    attest_only(attestation, local_identity(attestation, gate_sha)),
                    sort_keys=True,
                )
            )
            return 0
        print(json.dumps(execute(args, attestation), sort_keys=True))
        return 0
    except (GateError, OSError, RuntimeError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2
