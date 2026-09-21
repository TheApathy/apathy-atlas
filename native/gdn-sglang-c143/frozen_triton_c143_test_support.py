"""Shared CPU fixtures for frozen Triton oracle hostile tests."""

import copy
import datetime as dt
import hashlib
import json
from pathlib import Path

import frozen_triton_c143_gate as gate

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "frozen_triton_c143_manifest.json"


def digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def valid_buffers(m: int) -> dict[str, dict[str, int]]:
    layout = gate.workspace_layout(m)
    sizes = dict(layout["external_bytes"])
    sizes["workspace"] = layout["total_bytes"]
    result = {}
    cursor = 0x1_0000_0000
    for name, size in sizes.items():
        alignment = (
            gate.WORKSPACE_ALIGNMENT if name == "workspace" else gate.POINTER_ALIGNMENT
        )
        cursor = (cursor + alignment - 1) & -alignment
        result[name] = {"address": cursor, "bytes": size}
        cursor += size + 0x10_000
    return result


def balanced_positions(repetitions: int) -> dict[str, list[int]]:
    base, extra = divmod(repetitions, 3)
    rows = []
    for shift in range(3):
        row = [base] * 3
        for index in range(extra):
            row[(shift + index) % 3] += 1
        rows.append(row)
    return dict(zip(("candidate", "atlas", "sglang"), rows))


def valid_receipt(attestation: dict, m: int = 2_079) -> dict:
    candidate, atlas, sglang = (2.0, 3.0, 2.0) if m == 2_079 else (8.0, 12.0, 7.0)
    immutable = {name: digest(f"input:{name}") for name in gate.IMMUTABLE_NAMES}
    return {
        "schema": "atlas.gdn_c143.frozen_triton_raw_receipt.v1",
        "qualification": "PASS",
        "production_authorized": False,
        "default_off": True,
        "m": m,
        "nt": gate.workspace_layout(m)["nt"],
        "manifest_sha256": attestation["manifest_sha256"],
        "gate_sha256": gate._gate_source_sha256(),
        "artifact_attestation_sha256": attestation["artifact_attestation_sha256"],
        "reservation_receipt_sha256": digest("reservation"),
        "m2079_pass_receipt_sha256": None if m == 2_079 else digest("m2079-pass"),
        "geometry": copy.deepcopy(gate.EXPECTED_GEOMETRY),
        "workspace": gate.workspace_layout(m),
        "stream": {
            "nondefault": True,
            "same_stream_all_stages": True,
            "stream_identity_sha256": digest("nondefault-stream-handle"),
        },
        "artifacts_matched_before": True,
        "artifacts_matched_after": True,
        "adapter_checks": {name: True for name in gate.ADAPTER_CHECKS},
        "canaries": {"all_clean": True, "checked_regions": gate.CANARY_REGIONS},
        "input_hashes_before": immutable,
        "input_hashes_after": dict(immutable),
        "stage_hashes": {
            name: {"run1": digest(f"stage:{name}"), "run2": digest(f"stage:{name}")}
            for name in gate.STAGE_NAMES
        },
        "finite": {name: True for name in gate.FINITE_NAMES},
        "comparisons": {
            reference: {
                field: {"max_abs": 0.0, "rms": 0.0, "relative_rms": 0.0, "cosine": 1.0}
                for field in ("output", "state")
            }
            for reference in ("candidate_vs_sglang", "candidate_vs_atlas")
        },
        "stage_timing_ms": {name: [0.1] * 21 for name in gate.STAGE_NAMES},
        "full_timing": {
            "samples_ms": {
                "candidate": [candidate] * 22,
                "atlas": [atlas] * 22,
                "sglang": [sglang] * 22,
            },
            "position_counts": balanced_positions(22),
            "scope_includes": [
                "all_adapters",
                "A_and_output_initialization",
                "five_triton_kernels",
                "both_state_transposes",
            ],
        },
        "publication": {
            "exclusive_create": True,
            "written_after_all_gates": True,
            "preexisting": False,
            "result_path": "/tmp/frozen-triton-c143-raw-result.json",
        },
    }


def write_manifest(value: dict, directory: str) -> Path:
    path = Path(directory) / "manifest.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path.resolve(strict=True)


def reservation(attestation: dict, directory: str, nonce: str) -> Path:
    now = dt.datetime.now(dt.timezone.utc)
    value = {
        "schema": "atlas.gpu.reservation.v1",
        "claim": "qwen38-gdn-c143-frozen-triton-raw-gpu-oracle",
        "owner": "/root/future-independent-owner",
        "scope": "local-gb10-cuda-execution",
        "authorization": "GPU_EXECUTION_APPROVED",
        "released": False,
        "nonce_sha256": hashlib.sha256(nonce.encode("ascii")).hexdigest(),
        "manifest_sha256": attestation["manifest_sha256"],
        "gate_sha256": gate._gate_source_sha256(),
        "expires_utc": (now + dt.timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
    }
    path = Path(directory) / "reservation.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    path.chmod(0o600)
    return path.resolve(strict=True)
