# SPDX-License-Identifier: AGPL-3.0-only
"""Bounded child-process execution for Nsight session controls."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, BinaryIO


def run_bounded(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    output: BinaryIO,
    timeout_seconds: float,
) -> dict[str, Any]:
    if type(timeout_seconds) not in (int, float) or timeout_seconds <= 0:
        raise RuntimeError("invalid Nsight control timeout")
    child = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=output,
        stderr=subprocess.STDOUT,
    )
    actions: list[str] = []
    timed_out = False
    try:
        returncode = child.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        timed_out = True
        actions.append("terminate")
        child.terminate()
        try:
            returncode = child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            actions.append("kill")
            child.kill()
            try:
                returncode = child.wait(timeout=5)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError("Nsight control child survived SIGKILL") from error
    return {
        "actions": actions,
        "returncode": returncode,
        "timed_out": timed_out,
        "timeout_seconds": timeout_seconds,
    }
