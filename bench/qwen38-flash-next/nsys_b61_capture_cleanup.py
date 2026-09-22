# SPDX-License-Identifier: AGPL-3.0-only
"""Exception-safe finalization for the b61 split-session capture."""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path
from typing import Any

import nsys_b61_capture_logs as log_support
import nsys_b61_capture_process as process_support
import nsys_b61_capture_support as support


def _snapshot_owned(launcher: Any, prior: list[dict[str, int]]) -> list[dict[str, int]]:
    current = process_support.process_snapshot(
        process_support.descendants(launcher.pid)
    )
    identities = {
        (item["pid"], item["starttime_ticks"]): item for item in prior + current
    }
    return list(identities.values())


def _control(
    label: str,
    command: list[str],
    log: Any,
    events: list[dict[str, Any]],
    errors: list[str],
) -> None:
    try:
        log_support.run_control(label, command, log, events, check=False)
        if events[-1]["returncode"] != 0:
            errors.append(f"{label}: nonzero exit")
    except Exception as error:
        errors.append(f"{label}: {error}")


def _terminate_owned(identity: dict[str, int]) -> list[str]:
    pid, start = identity["pid"], identity["starttime_ticks"]
    actions = []
    for name, sig in (("sigterm", signal.SIGTERM), ("sigkill", signal.SIGKILL)):
        if not process_support.original_pid_alive(pid, start):
            return actions
        if process_support.starttime_ticks(pid) != start:
            raise RuntimeError("refusing to signal reused descendant PID")
        os.kill(pid, sig)
        actions.append(name)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if not process_support.original_pid_alive(pid, start):
                return actions
            time.sleep(0.1)
    if process_support.original_pid_alive(pid, start):
        raise RuntimeError("owned descendant survived SIGKILL")
    return actions


def finalize(
    *,
    output: Path,
    receipt: dict[str, Any],
    candidate: bool,
    launcher: Any,
    target_pid: int | None,
    before: dict[str, Any] | None,
    collection: bool,
    session: str,
    server_log: Any,
    control_log: Any,
    events: list[dict[str, Any]],
    owned_descendants: list[dict[str, int]],
) -> bool:
    errors = []
    prior = list(owned_descendants)
    if launcher is not None:
        try:
            prior = _snapshot_owned(launcher, prior)
        except Exception as error:
            errors.append(f"descendant snapshot: {error}")
    if collection:
        _control(
            "cancel",
            [str(support.NSYS), "cancel", f"--session={session}"],
            control_log,
            events,
            errors,
        )
    if launcher is not None:
        _control(
            "shutdown",
            [
                str(support.NSYS),
                "shutdown",
                f"--session={session}",
                "--kill=sigterm",
            ],
            control_log,
            events,
            errors,
        )
    if target_pid is not None and before is not None:
        try:
            receipt["target_signal_actions"] = process_support.terminate_exact_target(
                target_pid, before, process_support.attest_target
            )
        except Exception as error:
            errors.append(f"target: {error}")
    if launcher is not None:
        try:
            receipt["launcher_cleanup"] = process_support.reap_launcher(launcher)
            if receipt["launcher_cleanup"]["exit_code"] != 0:
                errors.append("launcher: nonzero exit")
        except Exception as error:
            errors.append(f"launcher: {error}")
    descendant_actions = []
    for identity in prior:
        try:
            actions = _terminate_owned(identity)
            descendant_actions.append({**identity, "actions": actions})
        except Exception as error:
            errors.append(f"descendant {identity.get('pid')}: {error}")
    receipt["descendant_cleanup"] = descendant_actions
    for label, stream in (("server log", server_log), ("control log", control_log)):
        try:
            stream.close()
        except Exception as error:
            errors.append(f"{label} close: {error}")
    try:
        process_support.assert_cleanup(support.PORT, before, prior, launcher)
    except Exception as error:
        errors.append(f"postcheck: {error}")
    if candidate and not errors:
        try:
            receipt["server_log_census"] = log_support.validate_server_log(
                output / "server.log"
            )
            receipt["control_log_census"] = log_support.validate_control_log(
                output / "nsys-control.log", events
            )
        except Exception as error:
            errors.append(f"log validation: {error}")
    receipt["control_events"] = events
    if not errors:
        try:
            receipt["logs"] = {
                path.name: log_support.freeze(path)
                for path in (output / "server.log", output / "nsys-control.log")
            }
        except Exception as error:
            errors.append(f"log freeze: {error}")
    if errors:
        receipt["cleanup_errors"] = errors
    receipt["qualified"] = candidate and not errors
    log_support.write_json(output / "capture-receipt.json", receipt)
    return receipt["qualified"]
