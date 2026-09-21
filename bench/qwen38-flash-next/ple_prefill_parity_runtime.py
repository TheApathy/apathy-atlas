# SPDX-License-Identifier: AGPL-3.0-only
"""Reviewed-caller-only runtime adapter for post-PLE raw parity."""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import signal
import stat
import subprocess
import time
from pathlib import Path
from typing import Any

import ple_prefill_parity as parity
import ple_prefill_parity_authority as authority
import ple_prefill_parity_contract as contract
from ple_prefill_parity_log import BoundedServerLog as _BoundedServerLog
import ple_prefill_parity_validate as validate

REPO = Path(__file__).resolve().parents[2]
MAX_RESPONSE = 64 << 20


def _starttime_ticks(pid: int) -> int:
    raw = Path(f"/proc/{pid}/stat").read_text()
    close = raw.rfind(")")
    if close < 0:
        raise RuntimeError("malformed process stat")
    return int(raw[close + 2 :].split()[19])


def _alive(pid: int, starttime: int) -> bool:
    try:
        return _starttime_ticks(pid) == starttime
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return False


def _require_signal_identity(pid: int, starttime: int) -> None:
    proc = Path(f"/proc/{pid}")
    if (
        not _alive(pid, starttime)
        or os.getsid(pid) != pid
        or os.getpgid(pid) != pid
        or Path(os.readlink(proc / "exe")) != contract.ELF
        or hashlib.sha256((proc / "exe").read_bytes()).hexdigest()
        != contract.PINS["elf"]
    ):
        raise RuntimeError("refusing to signal a drifted parity process")


def _session_members(session_id: int) -> set[int]:
    result = set()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "stat").read_text()
            close = raw.rfind(")")
            fields = raw[close + 2 :].split()
            if close >= 0 and fields[0] != "Z" and int(fields[3]) == session_id:
                result.add(int(entry.name))
        except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError):
            continue
    return result


def _drain_session(session_id: int) -> int:
    signal_count = 0
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        members = _session_members(session_id)
        if not members:
            return signal_count
        identities = []
        for pid in members:
            try:
                identities.append((pid, _starttime_ticks(pid)))
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue
        for pid, starttime in identities:
            try:
                if _alive(pid, starttime) and pid in _session_members(session_id):
                    os.kill(pid, signal.SIGKILL)
                    signal_count += 1
            except (ProcessLookupError, PermissionError):
                continue
        time.sleep(0.1)
    if _session_members(session_id):
        raise RuntimeError("parity server session descendants survived cleanup")


def _listeners() -> list[dict[str, str]]:
    result = []
    for table in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        for line in table.read_text().splitlines()[1:]:
            fields = line.split()
            address, encoded_port = fields[1].rsplit(":", 1)
            if int(encoded_port, 16) == contract.PORT and fields[3] == "0A":
                result.append(
                    {"table": table.name, "address": address, "inode": fields[9]}
                )
    return sorted(result, key=lambda item: tuple(item.values()))


def _assert_port_free() -> None:
    if _listeners():
        raise RuntimeError("parity port already has a listener")


def _environment(arm: str, nonce: str, root: Path) -> dict[str, str]:
    return contract.arm_environment(arm, nonce, root)


def _attest(pid: int, arm: str, nonce: str, root: Path) -> dict[str, Any]:
    proc = Path(f"/proc/{pid}")
    argv_raw = (proc / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
    env_raw = (proc / "environ").read_bytes().rstrip(b"\0").split(b"\0")
    argv = [item.decode("utf-8", errors="strict") for item in argv_raw]
    environment = dict(
        item.decode("utf-8", errors="strict").split("=", 1) for item in env_raw if item
    )
    listeners = _listeners()
    if len(listeners) != 1 or listeners[0]["table"] != "tcp":
        raise RuntimeError("exact parity listener cardinality drift")
    if listeners[0]["address"] != "0100007F":
        raise RuntimeError("parity listener is not exact IPv4 loopback")
    sockets = {
        os.readlink(item)
        for item in (proc / "fd").iterdir()
        if item.is_symlink() and os.readlink(item).startswith("socket:[")
    }
    if f"socket:[{listeners[0]['inode']}]" not in sockets:
        raise RuntimeError("parity listener is not owned by the target PID")
    if (
        Path(os.readlink(proc / "exe")) != contract.ELF
        or argv != contract.server_argv()
        or environment != _environment(arm, nonce, root)
        or Path(os.readlink(proc / "cwd")) != REPO
        or os.getsid(pid) != pid
        or os.getpgid(pid) != pid
        or _session_members(pid) != {pid}
    ):
        raise RuntimeError("parity process identity or sole-delta drift")
    elf_sha256 = hashlib.sha256((proc / "exe").read_bytes()).hexdigest()
    if elf_sha256 != contract.PINS["elf"]:
        raise RuntimeError("running parity ELF hash drift")
    return {
        "pid": pid,
        "starttime_ticks": _starttime_ticks(pid),
        "elf_sha256": elf_sha256,
        "argv": argv,
        "environment": environment,
    }


def _http_json(method: str, path: str, body: bytes | None = None) -> Any:
    connection = http.client.HTTPConnection("127.0.0.1", contract.PORT, timeout=1200)
    headers = {} if body is None else {"Content-Type": "application/json"}
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        raw = response.read(MAX_RESPONSE + 1)
        status, content_type = response.status, response.getheader("Content-Type")
    finally:
        connection.close()
    if status != 200 or content_type != "application/json" or len(raw) > MAX_RESPONSE:
        raise RuntimeError("bounded parity HTTP response identity drift")
    return json.loads(
        raw,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"invalid JSON constant: {value}")
        ),
    )


def _wait_ready(
    process: subprocess.Popen[Any],
    arm: str,
    nonce: str,
    root: Path,
    root_authority: authority.Authority,
) -> dict[str, Any]:
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"parity server exited during startup rc={process.returncode}"
            )
        root_authority.inventory_subset({process.pid})
        try:
            identity = _attest(process.pid, arm, nonce, root)
            response = _http_json("GET", "/v1/models")
            if (
                isinstance(response, dict)
                and response.get("object") == "list"
                and isinstance(response.get("data"), list)
                and len(response["data"]) == 1
                and isinstance(response["data"][0], dict)
                and response["data"][0].get("id") == contract.MODEL_NAME
            ):
                return identity
        except (ConnectionError, OSError, RuntimeError, ValueError):
            pass
        time.sleep(0.25)
    raise RuntimeError("exact parity server did not become ready")


def _terminate(process: subprocess.Popen[Any], starttime: int) -> dict[str, Any]:
    actions: list[str] = []
    timeout_after_sigterm: int | None = None
    sigkill_identity_rechecked = False
    returncode: int | None = None
    termination_error: BaseException | None = None
    try:
        returncode = process.poll()
        if not _alive(process.pid, starttime):
            raise RuntimeError(
                f"owned parity process exited before SIGTERM rc={returncode}"
            )
        _require_signal_identity(process.pid, starttime)
        actions.append("sigterm")
        os.killpg(process.pid, signal.SIGTERM)
        try:
            returncode = process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            timeout_after_sigterm = 15
            if not _alive(process.pid, starttime):
                returncode = process.wait(timeout=2)
            else:
                _require_signal_identity(process.pid, starttime)
                sigkill_identity_rechecked = True
                actions.append("sigkill")
                os.killpg(process.pid, signal.SIGKILL)
                returncode = process.wait(timeout=5)
    except BaseException as error:
        termination_error = error
    cleanup_error: BaseException | None = None
    session_drain_signal_count = 0
    final_session_empty = False
    final_listener_empty = False
    try:
        session_drain_signal_count = _drain_session(process.pid)
    except BaseException as error:
        cleanup_error = error
    try:
        if process.poll() is None:
            process.wait(timeout=2)
    except BaseException as error:
        cleanup_error = cleanup_error or error
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and (
            _session_members(process.pid) or _listeners()
        ):
            time.sleep(0.1)
        final_session_empty = not _session_members(process.pid)
        final_listener_empty = not _listeners()
        if not final_session_empty or not final_listener_empty:
            raise RuntimeError(
                f"parity server teardown resources remain rc={returncode}"
            )
    except BaseException as error:
        cleanup_error = cleanup_error or error
    if termination_error is not None:
        if cleanup_error is not None:
            termination_error.add_note(
                f"owned cleanup also failed: {type(cleanup_error).__name__}: "
                f"{cleanup_error}"
            )
        raise termination_error
    if cleanup_error is not None:
        raise cleanup_error
    return {
        "actions": actions,
        "timeout_after_sigterm_seconds": timeout_after_sigterm,
        "returncode": returncode,
        "clean_exit": returncode == 0 and timeout_after_sigterm is None,
        "sigterm_identity_rechecked": True,
        "sigkill_identity_rechecked": sigkill_identity_rechecked,
        "session_drain_signal_count": session_drain_signal_count,
        "final_session_empty": final_session_empty,
        "final_listener_empty": final_listener_empty,
        "final_gpu_inventory_empty": False,
        "build_model_authority_stable": False,
    }


def _terminate_unattested(process: subprocess.Popen[Any]) -> None:
    failures: list[tuple[str, BaseException]] = []
    running = True
    try:
        running = process.poll() is None
    except BaseException as error:
        failures.append(("poll", error))
    if running:
        try:
            process.terminate()
        except BaseException as error:
            failures.append(("terminate", error))
    timed_out = False
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        timed_out = True
    except BaseException as error:
        failures.append(("initial wait", error))
    if timed_out:
        try:
            process.kill()
        except BaseException as error:
            failures.append(("kill", error))
    try:
        _drain_session(process.pid)
    except BaseException as error:
        failures.append(("session drain", error))
    try:
        process.wait(timeout=2)
    except BaseException as error:
        failures.append(("final reap", error))
    try:
        _assert_port_free()
    except BaseException as error:
        failures.append(("listener census", error))
    if failures:
        primary = failures[0][1]
        for label, error in failures[1:]:
            primary.add_note(f"{label} also failed: {type(error).__name__}: {error}")
        raise primary


class RuntimeAdapter:
    """Callable adapter with internal, time-bounded root authority."""

    def __init__(self) -> None:
        self._authority = authority.Authority()
        self._inputs = parity.preflight_authority()

    def __call__(
        self,
        name: str,
        arm: str,
        spec: contract.RequestSpec,
        nonce: str,
        root: Path,
    ) -> dict[str, Any]:
        root_stat = root.lstat()
        if (
            name != spec.name
            or spec != contract.request_specs().get(name)
            or type(nonce) is not str
            or re.fullmatch(r"[0-9a-f]{64}", nonce) is None
            or not root.is_absolute()
            or root.resolve(strict=True) != root
            or not stat.S_ISDIR(root_stat.st_mode)
            or stat.S_IMODE(root_stat.st_mode) != 0o700
        ):
            raise RuntimeError("parity runtime request/root identity drift")
        self._authority.recheck()
        if parity.preflight_authority() != self._inputs:
            raise RuntimeError("parity build/model authority changed before runtime")
        self._authority.inventory(set())
        _assert_port_free()
        server_log = _BoundedServerLog(root)
        server_log.start_before_spawn()
        process: subprocess.Popen[Any] | None = None
        starttime: int | None = None
        error: BaseException | None = None
        identity: dict[str, Any] | None = None
        lifecycle: dict[str, Any] | None = None
        response: Any = None
        cleanup_errors: list[tuple[str, BaseException]] = []
        server_log_receipt: dict[str, Any] | None = None
        try:
            previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
            try:
                process = subprocess.Popen(
                    contract.server_argv(),
                    cwd=REPO,
                    env=_environment(arm, nonce, root),
                    stdin=subprocess.DEVNULL,
                    stdout=server_log.stdout_fd,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
            server_log.after_spawn()
            starttime = _starttime_ticks(process.pid)
            identity = _wait_ready(process, arm, nonce, root, self._authority)
            if identity["starttime_ticks"] != starttime:
                raise RuntimeError("parity server PID reuse during startup")
            self._authority.inventory({process.pid})
            response = _http_json("POST", "/v1/chat/completions", spec.wire)
            if _attest(process.pid, arm, nonce, root) != identity:
                raise RuntimeError("parity process drift across request")
            self._authority.inventory({process.pid})
        except BaseException as caught:
            error = caught
        finally:
            if process is None:
                try:
                    server_log.abort_before_runtime()
                except BaseException as caught:
                    cleanup_errors.append(("pre-spawn server log abort", caught))
            else:
                try:
                    server_log.after_spawn()
                except BaseException as caught:
                    cleanup_errors.append(("server log writer close", caught))
                try:
                    if starttime is None:
                        _terminate_unattested(process)
                    else:
                        lifecycle = _terminate(process, starttime)
                except BaseException as caught:
                    cleanup_errors.append(("process/session/listener cleanup", caught))
                try:
                    self._authority.inventory(set())
                except BaseException as caught:
                    cleanup_errors.append(("final GPU inventory", caught))
                try:
                    if parity.preflight_authority() != self._inputs:
                        raise RuntimeError(
                            "parity build/model authority changed across runtime"
                        )
                except BaseException as caught:
                    cleanup_errors.append(("final build/model authority", caught))
                try:
                    server_log_receipt = server_log.finish()
                except BaseException as caught:
                    cleanup_errors.append(("bounded server log finish", caught))
        if error is not None:
            for label, caught in cleanup_errors:
                error.add_note(
                    f"{label} also failed: {type(caught).__name__}: {caught}"
                )
        elif cleanup_errors:
            error = cleanup_errors[0][1]
            for label, caught in cleanup_errors[1:]:
                error.add_note(
                    f"{label} also failed: {type(caught).__name__}: {caught}"
                )
        if error is not None:
            if process is None or not isinstance(error, Exception):
                raise error
            raise RuntimeError(
                f"{name}/{arm} runtime failed: {error}; "
                f"server_log={json.dumps(server_log_receipt, sort_keys=True, allow_nan=False)}"
            ) from error
        if identity is None or lifecycle is None:
            raise RuntimeError("parity runtime produced no complete process receipt")
        lifecycle = {
            **lifecycle,
            "final_gpu_inventory_empty": True,
            "build_model_authority_stable": True,
        }
        try:
            validate.admit_lifecycle(lifecycle)
        except Exception as error:
            raise RuntimeError(
                f"{name}/{arm} lifecycle failed: {error}; "
                f"server_log={json.dumps(server_log_receipt, sort_keys=True, allow_nan=False)}"
            ) from error
        return {"process": {**identity, "lifecycle": lifecycle}, "response": response}
