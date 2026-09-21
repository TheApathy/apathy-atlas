# SPDX-License-Identifier: AGPL-3.0-only
"""Authorization, execution, rechecks, and exclusive raw receipt publication."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from .contract import NATIVE

if str(NATIVE) not in sys.path:
    sys.path.insert(0, str(NATIVE))

import frozen_triton_c143_gate as sealed_gate  # noqa: E402
from frozen_triton_c143_io import GateError  # noqa: E402
from frozen_triton_executor.main import parse_args  # noqa: E402

from .contract import (  # noqa: E402
    ADAPTER_LIBRARY,
    ADAPTER_LIBRARY_BYTES,
    ADAPTER_LIBRARY_SHA256,
    cpu_attestation,
    recheck_file,
    runtime_identity,
    validate_contract,
)
from .scheduler_authority import authorize_gpu_execution  # noqa: E402


def _execute(args, attestation: dict) -> dict:
    if (
        os.environ.get("ATLAS_GDN_C143_TRITON_RAW_GPU") != "1"
        or os.environ.get("ATLAS_GDN_C143_FUSED_NOZERO_RAW_GPU") != "1"
    ):
        raise GateError("fused-nozero GPU environment gate is not exactly 1")
    if None in (
        args.gpu_execution_flag,
        args.reservation_nonce,
        args.reservation,
        args.execution_contract,
    ):
        raise GateError("GPU execution requires flag, nonce, reservation, and contract")
    gate_sha = sealed_gate._gate_source_sha256()
    identity = runtime_identity(attestation, gate_sha)
    contract, contract_sha = validate_contract(
        args.execution_contract, identity, args.reservation_nonce
    )
    authorization = authorize_gpu_execution(
        explicit_flag=args.gpu_execution_flag,
        nonce=args.reservation_nonce,
        reservation_path=args.reservation,
        identity={**identity, "adapter_library_bytes": ADAPTER_LIBRARY_BYTES},
        contract_sha256=contract_sha,
    )
    try:
        return _execute_authorized(
            args, attestation, identity, contract, contract_sha, authorization
        )
    finally:
        authorization.close()


def _execute_authorized(
    args, attestation, identity, contract, contract_sha, authorization
):
    import torch

    from frozen_triton_executor.cuda_driver import CudaDriver
    from frozen_triton_executor.publication import publish_exclusive
    from frozen_triton_executor.references import References

    from .adapter import FusedInputLibrary
    from .case import run_shape

    library = FusedInputLibrary(
        ADAPTER_LIBRARY, ADAPTER_LIBRARY_SHA256, ADAPTER_LIBRARY_BYTES
    )
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
        raise GateError("fused-nozero executor requires a nondefault stream")
    references = References()
    cases = [
        run_shape(driver, attestation, references, m, stream, library)
        for m in (2_079, 8_192)
    ]
    stream.synchronize()
    if [case["m"] for case in cases] != [2_079, 8_192]:
        raise GateError("fused-nozero M sequence drift")
    post_attestation = sealed_gate.attest_manifest()
    if post_attestation != attestation:
        raise GateError("post-run manifest/artifact attestation drift")
    post_gate_sha = sealed_gate._gate_source_sha256()
    current = runtime_identity(post_attestation, post_gate_sha)
    if current != identity:
        raise GateError("post-run fused-nozero identity drift")
    library.image.verify_executable_mapping()
    recheck_file(args.execution_contract, contract_sha, "post-run execution contract")
    receipt = {
        "schema": "atlas.gdn_c143.fused_nozero_raw_receipt.v2",
        "qualification": "PASS",
        "variant": "fused-input-plus-no-output-clear",
        "production_authorized": False,
        "default_off": True,
        "execution_contract_sha256": contract_sha,
        "scheduler_manifest_sha256": authorization.manifest.sha256,
        "reservation_receipt_sha256": authorization.reservation.sha256,
        "build_receipt_sha256": authorization.build.sha256,
        "reservation_owner": authorization.owner,
        **{
            key: contract[key]
            for key in contract
            if key not in {"schema", "result_path"}
        },
        "device": device,
        "cases": cases,
    }
    driver.close_modules()
    authorization.recheck()
    result_path = Path(contract["result_path"])
    receipt_sha = publish_exclusive(result_path, receipt)
    return {
        "schema": "atlas.gdn_c143.fused_nozero_publication.v2",
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
            print(json.dumps(cpu_attestation(attestation, gate_sha), sort_keys=True))
            return 0
        print(json.dumps(_execute(args, attestation), sort_keys=True, allow_nan=False))
        return 0
    except (GateError, OSError, RuntimeError, ValueError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 2
