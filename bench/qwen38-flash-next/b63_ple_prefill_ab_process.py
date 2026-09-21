# SPDX-License-Identifier: AGPL-3.0-only
"""Exact direct-server process/listener ownership and cleanup."""

from __future__ import annotations

import hashlib
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import b63_ple_prefill_ab_contract as contract

REPO = Path(__file__).resolve().parents[2]


def server_argv(model_path: Path) -> list[str]:
    options = {
        "--model-from-path": str(model_path),
        "--model-name": contract.MODEL_NAME,
        "--kernel-target": contract.MODEL_NAME,
        "--port": str(contract.PORT),
        "--max-seq-len": "2048",
        "--max-prefill-tokens": "2048",
        "--max-num-seqs": "1",
        "--max-batch-size": "1",
        "--ssm-cache-slots": "0",
        "--kv-cache-dtype": "bf16",
        "--gpu-memory-utilization": "0.90",
        "--oom-guard-mb": "4096",
        "--request-timeout": "600",
    }
    return (
        [str(contract.BINARY), "serve"]
        + [value for item in options.items() for value in item]
        + ["--no-tui"]
    )


def arm_environment(arm: str) -> dict[str, str]:
    return {
        **contract.ROUTE_ENV,
        "ATLAS_QWEN4_PLE_PREFILL_BATCH": contract.ARM_SELECTORS[arm],
        "LANG": "C.UTF-8",
        "LD_LIBRARY_PATH": "/usr/local/cuda-13.0/targets/sbsa-linux/lib",
        "PATH": "/usr/local/cuda-13.0/bin:/usr/local/bin:/usr/bin:/bin",
        "RUST_LOG": "info",
    }


def _tcp_listeners(port: int) -> list[dict[str, str]]:
    found = []
    for table in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        for line in table.read_text().splitlines()[1:]:
            fields = line.split()
            address, encoded_port = fields[1].rsplit(":", 1)
            if int(encoded_port, 16) == port and fields[3] == "0A":
                found.append(
                    {"table": table.name, "address": address, "inode": fields[9]}
                )
    return sorted(
        found, key=lambda item: (item["table"], item["address"], item["inode"])
    )


def assert_port_free() -> None:
    if _tcp_listeners(contract.PORT):
        raise RuntimeError(f"port {contract.PORT} already has a listener")


def validate_listener_records(listeners: list[dict[str, str]]) -> dict[str, str]:
    if len(listeners) != 1:
        raise RuntimeError("listener cardinality drift")
    listener = listeners[0]
    if listener["table"] != "tcp" or listener["address"] != "0100007F":
        raise RuntimeError("listener is not bound to exact IPv4 loopback")
    return listener


def starttime_ticks(pid: int) -> int:
    text = Path(f"/proc/{pid}/stat").read_text()
    close = text.rfind(")")
    if close < 0:
        raise RuntimeError("malformed process stat")
    return int(text[close + 2 :].split()[19])


def original_alive(pid: int, starttime: int) -> bool:
    try:
        return starttime_ticks(pid) == starttime
    except (FileNotFoundError, ProcessLookupError):
        return False


def session_members(session_id: int) -> set[int]:
    result = set()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            text = (entry / "stat").read_text()
            close = text.rfind(")")
            fields = text[close + 2 :].split()
            if close >= 0 and int(fields[3]) == session_id:
                result.add(int(entry.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
    return result


def drain_session(session_id: int, actions: list[str]) -> None:
    if session_members(session_id):
        actions.append("session-sigkill")
        os.killpg(session_id, signal.SIGKILL)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and session_members(session_id):
            time.sleep(0.1)
    if session_members(session_id):
        raise RuntimeError("server session descendants remain after cleanup")


def require_owned_signal_identity(pid: int, starttime: int) -> None:
    proc = Path(f"/proc/{pid}")
    if (
        not original_alive(pid, starttime)
        or os.getsid(pid) != pid
        or os.getpgid(pid) != pid
        or contract.sha256((proc / "exe").read_bytes()) != contract.BINARY_SHA256
    ):
        raise RuntimeError("refusing to signal drifted/reused server session")


def attest_target(pid: int, arm: str) -> dict[str, Any]:
    proc = Path(f"/proc/{pid}")
    executable = Path(os.readlink(proc / "exe"))
    argv = (proc / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
    environ = (proc / "environ").read_bytes().rstrip(b"\0").split(b"\0")
    decoded_env = dict(item.decode().split("=", 1) for item in environ if item)
    if executable != contract.BINARY or [item.decode() for item in argv] != server_argv(
        contract.MODEL
    ):
        raise RuntimeError("server executable/argv identity drift")
    if decoded_env != arm_environment(arm) or Path(os.readlink(proc / "cwd")) != REPO:
        raise RuntimeError("server environment/cwd identity drift")
    if os.getsid(pid) != pid or os.getpgid(pid) != pid or session_members(pid) != {pid}:
        raise RuntimeError("server session/process-group/descendant identity drift")
    listeners = _tcp_listeners(contract.PORT)
    listener = validate_listener_records(listeners)
    sockets = {
        os.readlink(path)
        for path in (proc / "fd").iterdir()
        if path.is_symlink() and os.readlink(path).startswith("socket:[")
    }
    if f"socket:[{listener['inode']}]" not in sockets:
        raise RuntimeError("listener is not owned by exact server PID")
    exe_sha256 = contract.sha256((proc / "exe").read_bytes())
    if exe_sha256 != contract.BINARY_SHA256:
        raise RuntimeError("running server ELF hash drift")
    return {
        "pid": pid,
        "starttime_ticks": starttime_ticks(pid),
        "exe_sha256": exe_sha256,
        "argv_sha256": hashlib.sha256(b"\0".join(argv) + b"\0").hexdigest(),
        "environment_sha256": hashlib.sha256(
            b"\0".join(sorted(environ)) + b"\0"
        ).hexdigest(),
        "environment_names": sorted(decoded_env),
        "listener": listener,
        "session_id": pid,
        "descendants": [],
    }


def wait_target(process: subprocess.Popen[Any], arm: str) -> dict[str, Any]:
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited during startup rc={process.returncode}")
        try:
            return attest_target(process.pid, arm)
        except (
            FileNotFoundError,
            PermissionError,
            ProcessLookupError,
            RuntimeError,
            ValueError,
        ):
            time.sleep(0.25)
    raise RuntimeError("exact server PID/listener did not become ready")


def terminate_unattested(process: subprocess.Popen[Any]) -> dict[str, Any]:
    actions = []
    if process.poll() is None:
        actions.append("popen-terminate")
        if os.getsid(process.pid) == process.pid:
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
    try:
        returncode = process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        actions.append("popen-kill")
        if os.getsid(process.pid) == process.pid:
            os.killpg(process.pid, signal.SIGKILL)
        else:
            process.kill()
        returncode = process.wait(timeout=5)
    drain_session(process.pid, actions)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and _tcp_listeners(contract.PORT):
        time.sleep(0.1)
    assert_port_free()
    receipt = {"actions": actions, "returncode": returncode}
    return {**receipt, "port_and_session_free": True}


def terminate_owned(process: subprocess.Popen[Any], starttime: int) -> dict[str, Any]:
    actions = []
    if original_alive(process.pid, starttime):
        require_owned_signal_identity(process.pid, starttime)
        actions.append("sigterm")
        os.killpg(process.pid, signal.SIGTERM)
    try:
        returncode = process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        if not original_alive(process.pid, starttime):
            returncode = process.wait(timeout=2)
        else:
            require_owned_signal_identity(process.pid, starttime)
            actions.append("sigkill")
            os.killpg(process.pid, signal.SIGKILL)
            returncode = process.wait(timeout=5)
    drain_session(process.pid, actions)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and _tcp_listeners(contract.PORT):
        time.sleep(0.1)
    assert_port_free()
    if original_alive(process.pid, starttime):
        raise RuntimeError("owned target remains alive after cleanup")
    receipt = {"actions": actions, "returncode": returncode}
    return {**receipt, "port_and_session_free": True}
