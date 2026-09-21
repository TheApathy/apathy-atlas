# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import hashlib
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import nsys_b61_capture_support as support


def listener_inodes(port: int) -> set[str]:
    found = set()
    for name in ("/proc/net/tcp", "/proc/net/tcp6"):
        for line in Path(name).read_text().splitlines()[1:]:
            fields = line.split()
            if int(fields[1].rsplit(":", 1)[1], 16) == port and fields[3] == "0A":
                found.add(fields[9])
    return found


def descendants(root: int) -> set[int]:
    result, pending = set(), [root]
    while pending:
        parent = pending.pop()
        path = Path(f"/proc/{parent}/task/{parent}/children")
        try:
            children = [int(value) for value in path.read_text().split()]
        except (FileNotFoundError, ProcessLookupError):
            continue
        for child in children:
            if child not in result:
                result.add(child)
                pending.append(child)
    return result


def starttime_ticks(pid: int) -> int:
    text = Path(f"/proc/{pid}/stat").read_text()
    close = text.rfind(")")
    if close < 0:
        raise RuntimeError("malformed proc stat")
    fields_after_comm = text[close + 2 :].split()
    return int(fields_after_comm[19])


def original_pid_alive(pid: int, starttime: int) -> bool:
    try:
        return starttime_ticks(pid) == starttime
    except (FileNotFoundError, ProcessLookupError):
        return False


def process_snapshot(pids: set[int]) -> list[dict[str, int]]:
    result = []
    for pid in sorted(pids):
        try:
            result.append({"pid": pid, "starttime_ticks": starttime_ticks(pid)})
        except (FileNotFoundError, ProcessLookupError):
            pass
    return result


def attest_target(pid: int) -> dict[str, Any]:
    proc = Path(f"/proc/{pid}")
    executable = Path(os.readlink(proc / "exe"))
    argv = (proc / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
    environ = (proc / "environ").read_bytes().rstrip(b"\0").split(b"\0")
    decoded_env = dict(item.decode().split("=", 1) for item in environ if item)
    atlas_env = {
        key: value for key, value in decoded_env.items() if key.startswith("ATLAS_")
    }
    expected_atlas = {
        key: value
        for key, value in support.TARGET_ENV.items()
        if key.startswith("ATLAS_")
    }
    if (
        executable != support.BINARY
        or [item.decode() for item in argv] != support.server_argv()
    ):
        raise RuntimeError("target executable/argv drift")
    if atlas_env != expected_atlas or Path(os.readlink(proc / "cwd")) != support.REPO:
        raise RuntimeError("target environment/cwd drift")
    sockets = {os.readlink(fd) for fd in (proc / "fd").iterdir() if fd.is_symlink()}
    owned = sorted(
        inode
        for inode in listener_inodes(support.PORT)
        if f"socket:[{inode}]" in sockets
    )
    return {
        "pid": pid,
        "starttime_ticks": starttime_ticks(pid),
        "exe_sha256": support.sha_file(proc / "exe"),
        "argv_sha256": hashlib.sha256(b"\0".join(argv) + b"\0").hexdigest(),
        "environment_sha256": hashlib.sha256(
            b"\0".join(sorted(environ)) + b"\0"
        ).hexdigest(),
        "environment_names": sorted(decoded_env),
        "listener_inodes": owned,
    }


def wait_http() -> None:
    import nsys_b61_capture_logs as log_support

    url = support.ENDPOINT.rsplit("/chat", 1)[0] + "/models"
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        try:
            payload, _ = log_support.http_json(url)
            if payload["data"][0]["id"] == "qwen3.8-flash-next":
                return
        except Exception:
            time.sleep(0.25)
    raise RuntimeError("model endpoint did not become ready")


class CaptureGuard:
    def __init__(self) -> None:
        self.phase, self.traced_requests = "ready", 0

    def warmup_done(self, body: dict[str, Any], semantic: dict[str, Any]) -> None:
        if (
            self.phase != "ready"
            or body != support.request_body(64)
            or semantic["completion_tokens"] != 64
        ):
            raise RuntimeError("warmup ordering/body violation")
        self.phase = "warm"

    def collection_started(self) -> None:
        if self.phase != "warm":
            raise RuntimeError("collection started before accepted warmup")
        self.phase = "collecting"

    def traced_request(self, body: dict[str, Any]) -> None:
        if (
            self.phase != "collecting"
            or body != support.request_body(400)
            or self.traced_requests
        ):
            raise RuntimeError("traced request count/body violation")
        self.traced_requests += 1

    def collection_stopped(self) -> None:
        if self.phase != "collecting" or self.traced_requests != 1:
            raise RuntimeError("collection must contain exactly one request")
        self.phase = "stopped"


def wait_target(
    root: int,
    attest: Callable[[int], dict[str, Any]],
    timeout: float = 240,
) -> tuple[int, dict[str, Any]]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for pid in descendants(root):
            try:
                evidence = attest(pid)
                if len(evidence["listener_inodes"]) == 1:
                    return pid, evidence
            except (
                FileNotFoundError,
                PermissionError,
                ProcessLookupError,
                RuntimeError,
                ValueError,
            ):
                pass
        time.sleep(0.25)
    raise RuntimeError("exact server process/listener did not become ready")


def _wait_original_gone(pid: int, starttime: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not original_pid_alive(pid, starttime):
            return True
        time.sleep(0.1)
    return not original_pid_alive(pid, starttime)


def terminate_exact_target(
    pid: int,
    expected: dict[str, Any],
    attest: Callable[[int], dict[str, Any]],
) -> list[str]:
    starttime = expected["starttime_ticks"]
    if not original_pid_alive(pid, starttime):
        return []
    if attest(pid) != expected:
        raise RuntimeError("refusing to signal reused/drifted target PID")
    actions = ["sigterm"]
    os.kill(pid, signal.SIGTERM)
    if _wait_original_gone(pid, starttime, 5):
        return actions
    if attest(pid) != expected:
        raise RuntimeError("refusing SIGKILL after target identity drift")
    actions.append("sigkill")
    os.kill(pid, signal.SIGKILL)
    if not _wait_original_gone(pid, starttime, 5):
        raise RuntimeError("exact target survived SIGKILL")
    return actions


def reap_launcher(launcher: subprocess.Popen[Any]) -> dict[str, Any]:
    actions = []
    try:
        exit_code = launcher.wait(timeout=10)
    except subprocess.TimeoutExpired:
        actions.append("terminate")
        launcher.terminate()
        try:
            exit_code = launcher.wait(timeout=5)
        except subprocess.TimeoutExpired:
            actions.append("kill")
            launcher.kill()
            try:
                exit_code = launcher.wait(timeout=5)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(
                    "Nsight launcher survived kill escalation"
                ) from error
    return {"exit_code": exit_code, "actions": actions}


def assert_cleanup(
    port: int,
    target: dict[str, Any] | None,
    prior_descendants: list[dict[str, int]],
    launcher: subprocess.Popen[Any] | None,
) -> None:
    if listener_inodes(port):
        raise RuntimeError("target port still has a listener after cleanup")
    identities = list(prior_descendants)
    if target is not None:
        identities.append(
            {"pid": target["pid"], "starttime_ticks": target["starttime_ticks"]}
        )
    if any(
        original_pid_alive(item["pid"], item["starttime_ticks"]) for item in identities
    ):
        raise RuntimeError("owned descendant remains alive after cleanup")
    if launcher is not None and launcher.poll() is None:
        raise RuntimeError("Nsight launcher remains alive after cleanup")
