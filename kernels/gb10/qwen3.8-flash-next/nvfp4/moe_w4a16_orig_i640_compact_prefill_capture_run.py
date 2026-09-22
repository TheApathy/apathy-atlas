# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from moe_w4a16_orig_i640_compact_prefill_capture_support import (
    identity,
    parse_event,
    parse_offsets,
    read_immutable,
    sha,
    write_immutable,
)


def hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def wait_http(url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(0.25)
    raise TimeoutError("server readiness")


def wait_files(paths: list[Path], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if all(path.is_file() for path in paths):
            return
        time.sleep(0.05)
    raise TimeoutError("route capture files")


def process_bytes(pid: int, name: str) -> bytes:
    return Path(f"/proc/{pid}/{name}").read_bytes()


def listener_inode(pid: int, url: str) -> str:
    port = urlparse(url).port
    sockets: set[str] = set()
    for descriptor in Path(f"/proc/{pid}/fd").iterdir():
        try:
            target = os.readlink(descriptor)
        except OSError:
            continue
        if target.startswith("socket:[") and target.endswith("]"):
            sockets.add(target[8:-1])
    listeners: set[str] = set()
    for table in ("tcp", "tcp6"):
        for line in Path(f"/proc/{pid}/net/{table}").read_text().splitlines()[1:]:
            fields = line.split()
            if (
                len(fields) > 9
                and int(fields[1].rsplit(":", 1)[1], 16) == port
                and fields[3] == "0A"
                and fields[9] in sockets
            ):
                listeners.add(fields[9])
    if len(listeners) != 1:
        raise ValueError("exact owned listener required")
    return listeners.pop()


def run_shape(
    plan: dict,
    rows: int,
    output: Path,
    server_id: dict[str, str],
    hook_id: dict[str, str],
) -> dict[str, str]:
    label = f"m{rows}"
    request_path = Path(plan["shapes"][str(rows)]["request"])
    request = read_immutable(request_path, 16 << 20)
    if hash_bytes(request) != plan["shapes"][str(rows)]["request_sha256"]:
        raise ValueError("request hash")
    offsets_path, event_path = (
        output / f"{label}.offsets.u32",
        output / f"{label}.event",
    )
    response_path, log_path = (
        output / f"{label}.response.json",
        output / f"{label}.server.log",
    )
    environment = dict(plan["base_env"])
    environment.update(
        {
            "LD_PRELOAD": hook_id["path"],
            "ATLAS_MOE_EXACT_PREFILL_GRID": "1",
            "ATLAS_OI640_CAPTURE_MODE": "real-target-route-v1",
            "ATLAS_OI640_CAPTURE_NONCE": plan["nonce"],
            "ATLAS_OI640_CAPTURE_M2013": str(
                offsets_path if rows == 2013 else output / "unused2013"
            ),
            "ATLAS_OI640_CAPTURE_M8192": str(
                offsets_path if rows == 8192 else output / "unused8192"
            ),
            "ATLAS_OI640_CAPTURE_M2013_EVENT": str(
                event_path if rows == 2013 else output / "unused2013.event"
            ),
            "ATLAS_OI640_CAPTURE_M8192_EVENT": str(
                event_path if rows == 8192 else output / "unused8192.event"
            ),
        }
    )
    descriptor = os.open(
        log_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
    )
    process = subprocess.Popen(
        plan["base_argv"],
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=descriptor,
        stderr=subprocess.STDOUT,
        close_fds=True,
    )
    try:
        wait_http(plan["health_url"], 180)
        live_id = identity(Path(f"/proc/{process.pid}/exe"), 0o555)
        if (
            live_id != server_id
            or hook_id["path"] not in process_bytes(process.pid, "maps").decode()
        ):
            raise ValueError("server/hook process identity")
        cmdline_sha = hash_bytes(process_bytes(process.pid, "cmdline"))
        environ_sha = hash_bytes(process_bytes(process.pid, "environ"))
        listener = listener_inode(process.pid, plan["request_url"])
        query = urllib.request.Request(
            plan["request_url"],
            data=request,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(query, timeout=900) as response:
            response_bytes = response.read(16 << 20)
            if response.read(1):
                raise ValueError("response exceeds bound")
        write_immutable(response_path, response_bytes)
        wait_files([offsets_path, event_path], 30)
        usage = json.loads(response_bytes)["usage"]
        if usage["prompt_tokens"] != rows:
            raise ValueError("prompt token census")
        _, offset_sha = parse_offsets(offsets_path, rows)
        event = parse_event(event_path, rows, plan["nonce"], process.pid)
        if listener_inode(process.pid, plan["request_url"]) != listener:
            raise ValueError("listener identity drift")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(10)
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.close(descriptor)
    if (
        process.returncode not in (0, -15)
        or identity(Path(plan["server"]), 0o555) != server_id
    ):
        raise ValueError("server exit/identity drift")
    parsed = urlparse(plan["health_url"])
    with socket.socket() as probe:
        probe.settimeout(1)
        if probe.connect_ex((parsed.hostname or "127.0.0.1", parsed.port or 80)) == 0:
            raise ValueError("listener survived teardown")
    log = read_immutable(log_path, 64 << 20).decode(errors="strict")
    selector = [
        line for line in log.splitlines() if "QWEN4_PREFILL_SELECTOR_RECEIPT" in line
    ]
    required = (
        f"M={rows}",
        "family=attention",
        "serialized_fallback=false",
        "H=2560 L=48 E=512 TOPK=10 I=640 SI=640",
    )
    if (
        len(selector) != 1
        or any(marker not in selector[0] for marker in required)
        or log.count(f"Chunked prefill start: {rows} prompt tokens, chunk_size={rows}")
        != 1
        or log.count(f"ATLAS_EXPERT_LOAD: n_tokens={rows}") != 1
        or any(
            marker in log
            for marker in (" ERROR ", "panicked", "serialized_fallback=true")
        )
    ):
        raise ValueError("route log gate")
    return {
        "offset.path": str(offsets_path.resolve()),
        "offset.sha256": offset_sha,
        "event.path": str(event_path.resolve()),
        "event.sha256": event["sha256"],
        "event.source": event["source"],
        "request.sha256": hash_bytes(request),
        "request.path": str(request_path.resolve()),
        "response.path": str(response_path.resolve()),
        "response.sha256": sha(response_path),
        "log.path": str(log_path.resolve()),
        "log.sha256": sha(log_path),
        "process.pid": str(process.pid),
        "process.cmdline_sha256": cmdline_sha,
        "process.environ_sha256": environ_sha,
        "process.listener_inode": listener,
        "process.exit": str(process.returncode),
        "prompt_tokens": str(rows),
    }
