# SPDX-License-Identifier: AGPL-3.0-only
"""Exact future GB10 identity and compute-client inventory admission."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

import b62_prefill_ab_identity as identity

NVIDIA_SMI = Path("/usr/bin/nvidia-smi")
ENV = {"LANG": "C.UTF-8", "PATH": "/usr/bin:/bin"}


def _tool(expected_sha256: str) -> dict[str, Any]:
    raw, evidence = identity.stable_bytes(
        NVIDIA_SMI, expected_sha256=expected_sha256, max_bytes=32 << 20
    )
    if evidence["mode"] & 0o222 or not evidence["mode"] & 0o111 or not raw:
        raise RuntimeError("nvidia-smi executable mode/content drift")
    return evidence


def _query(
    arguments: list[str], expected_sha256: str
) -> tuple[list[str], dict[str, Any]]:
    before = _tool(expected_sha256)
    completed = subprocess.run(
        [str(NVIDIA_SMI), *arguments],
        check=False,
        capture_output=True,
        env=ENV,
        text=True,
        timeout=15,
    )
    after = _tool(expected_sha256)
    if before != after or completed.returncode != 0 or completed.stderr.strip():
        raise RuntimeError("nvidia-smi inventory command failed or drifted")
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    return lines, {
        "tool": before,
        "argv_sha256": hashlib.sha256(
            b"\0".join(arg.encode() for arg in arguments)
        ).hexdigest(),
        "stdout_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest(),
    }


def attest_inventory(
    reservation: dict[str, Any], expected_pids: set[int]
) -> dict[str, Any]:
    document = reservation["document"]
    expected_tool = document["nvidia_smi_sha256"]
    gpu_lines, gpu_evidence = _query(
        ["--query-gpu=uuid,name", "--format=csv,noheader"], expected_tool
    )
    if len(gpu_lines) != 1 or "," not in gpu_lines[0]:
        raise RuntimeError("single-GPU identity census mismatch")
    gpu_uuid, gpu_name = (part.strip() for part in gpu_lines[0].split(",", 1))
    if gpu_uuid != document["gpu_uuid"] or "GB10" not in gpu_name:
        raise RuntimeError("authorized GB10 identity mismatch")
    app_lines, app_evidence = _query(
        ["--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"],
        expected_tool,
    )
    rows = []
    for line in app_lines:
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2 or not parts[0].isdigit() or parts[1] != gpu_uuid:
            raise RuntimeError("malformed or foreign compute-client inventory")
        rows.append({"pid": int(parts[0]), "gpu_uuid": parts[1]})
    if (
        len(rows) != len({row["pid"] for row in rows})
        or {row["pid"] for row in rows} != expected_pids
    ):
        raise RuntimeError("GPU compute-client exclusivity drift")
    return {
        "gpu": {"uuid": gpu_uuid, "name": gpu_name},
        "compute_clients": sorted(rows, key=lambda row: row["pid"]),
        "gpu_query": gpu_evidence,
        "apps_query": app_evidence,
    }
