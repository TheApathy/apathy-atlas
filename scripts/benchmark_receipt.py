#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only

"""Create and verify immutable, no-secret Atlas benchmark receipts."""

import argparse
import csv
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import time
from urllib.parse import parse_qsl, urlsplit
import urllib.request
import uuid

SCHEMA = "atlas-benchmark-receipt-v2"
BOOT_ID_PATH = Path("/proc/sys/kernel/random/boot_id")
IDENTITY_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "generation_config.json",
)
SECRET_KEY_PARTS = ("API_KEY", "SECRET", "PASSWORD", "AUTH_TOKEN", "ACCESS_TOKEN")
SECRET_ARG_FLAGS = {
    "--access-token",
    "--api-key",
    "--auth-token",
    "--hf-token",
    "--password",
    "--secret",
    "--token",
}
SECRET_QUERY_KEYS = {
    "access_token",
    "api_key",
    "auth_token",
    "password",
    "secret",
    "token",
}
GPU_QUERY_FIELDS = (
    "uuid",
    "name",
    "driver_version",
    "pstate",
    "temperature.gpu",
    "power.draw",
    "clocks.sm",
    "clocks.mem",
    "memory.total",
    "memory.used",
    "memory.free",
)


def validate_boot_id(value: str) -> str:
    if type(value) is not str:
        raise ValueError("host boot ID is missing or malformed")
    try:
        parsed = str(uuid.UUID(value))
    except (ValueError, AttributeError) as error:
        raise ValueError("host boot ID is missing or malformed") from error
    if parsed != value:
        raise ValueError("host boot ID is not canonical")
    return value


def host_boot_id() -> str:
    try:
        value = BOOT_ID_PATH.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as error:
        raise ValueError("host boot ID is unavailable") from error
    return validate_boot_id(value)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def environment_sha256(environment: dict[str, str]) -> str:
    if type(environment) is not dict or any(
        type(key) is not str or type(value) is not str
        for key, value in environment.items()
    ):
        raise ValueError("full environment must be a string map")
    return hashlib.sha256(canonical_json(environment)).hexdigest()


class RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def urlopen_no_redirect(request, timeout: float):
    return urllib.request.build_opener(RejectRedirects).open(request, timeout=timeout)


def gpu_snapshot() -> dict:
    query = ",".join(GPU_QUERY_FIELDS)
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise ValueError("GPU identity query failed") from error
    rows = list(csv.reader(result.stdout.splitlines()))
    if len(rows) != 1 or len(rows[0]) != len(GPU_QUERY_FIELDS):
        raise ValueError("benchmark requires exactly one parseable GPU")
    values = dict(zip(GPU_QUERY_FIELDS, (value.strip() for value in rows[0])))
    if not all(values[key] and values[key] != "[N/A]" for key in GPU_QUERY_FIELDS[:3]):
        raise ValueError("GPU identity fields are missing")
    return {
        "identity": {key: values[key] for key in GPU_QUERY_FIELDS[:3]},
        "state": {key: values[key] for key in GPU_QUERY_FIELDS[3:]},
    }


def git_output(repo: Path, *args: str) -> bytes:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=repo,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"git identity failed: {' '.join(args)}") from error


def git_identity(repo: Path) -> dict:
    repo = repo.resolve()
    commit = git_output(repo, "rev-parse", "HEAD").decode().strip()
    status = git_output(repo, "status", "--porcelain=v1", "-z")
    tracked_diff = git_output(repo, "diff", "--binary", "HEAD")
    untracked = git_output(repo, "ls-files", "--others", "--exclude-standard", "-z")
    tree = hashlib.sha256()
    tree.update(b"status\0" + status)
    tree.update(b"tracked-diff\0" + tracked_diff)
    for raw_path in sorted(part for part in untracked.split(b"\0") if part):
        relative = os.fsdecode(raw_path)
        path = repo / relative
        tree.update(b"untracked\0" + raw_path + b"\0")
        if path.is_symlink():
            tree.update(b"symlink\0" + os.fsencode(os.readlink(path)))
        elif path.is_file():
            tree.update(bytes.fromhex(sha256_file(path)))
        else:
            raise ValueError(f"unsupported untracked identity path: {relative}")
    return {
        "repo": str(repo),
        "commit": commit,
        "dirty": bool(status),
        "tree_sha256": tree.hexdigest(),
    }


def checkpoint_identity(path: Path, label: str) -> dict:
    path = path.resolve()
    if not path.is_dir():
        raise ValueError(f"{label} directory is missing: {path}")
    config_path = path / "config.json"
    index_path = path / "model.safetensors.index.json"
    if not config_path.is_file():
        raise ValueError(f"{label} config.json is missing: {config_path}")
    if not index_path.is_file():
        raise ValueError(f"{label} model index is missing: {index_path}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} config.json is malformed") from error
    if type(config) is not dict or type(config.get("model_type")) is not str:
        raise ValueError(f"{label} model_type is missing or malformed")
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} model index is malformed") from error
    if type(index) is not dict or type(index.get("weight_map")) is not dict:
        raise ValueError(f"{label} model index weight_map is missing or malformed")
    weight_map = index["weight_map"]
    if not weight_map:
        raise ValueError(f"{label} model index weight_map is empty")
    shard_names = set()
    for tensor_name, shard_name in weight_map.items():
        if type(tensor_name) is not str or not tensor_name:
            raise ValueError(f"{label} model index tensor name is malformed")
        if type(shard_name) is not str or not shard_name:
            raise ValueError(f"{label} model index shard path is malformed")
        relative = Path(shard_name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(
                f"{label} model index shard path escapes checkpoint: {shard_name}"
            )
        shard_names.add(shard_name)
    shards = {}
    for shard_name in sorted(shard_names):
        shard_path = (path / shard_name).resolve()
        try:
            shard_path.relative_to(path)
        except ValueError as error:
            raise ValueError(
                f"{label} model index shard path escapes checkpoint: {shard_name}"
            ) from error
        if not shard_path.is_file():
            raise ValueError(f"{label} model shard is missing: {shard_name}")
        shards[shard_name] = {
            "size": shard_path.stat().st_size,
            "sha256": sha256_file(shard_path),
        }
    tokenizer_files = [
        name
        for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")
        if (path / name).is_file()
    ]
    if not tokenizer_files:
        raise ValueError(f"{label} tokenizer identity file is missing")
    files = {
        name: sha256_file(path / name)
        for name in IDENTITY_FILES
        if (path / name).is_file()
    }
    return {
        "path": str(path),
        "model_type": config["model_type"],
        "files": files,
        "weight_shards": shards,
    }


def binary_contains(path: Path, needle: bytes) -> bool:
    overlap = max(len(needle) - 1, 0)
    tail = b""
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            data = tail + chunk
            if needle in data:
                return True
            tail = data[-overlap:] if overlap else b""
    return False


def binary_identity(path: Path, required_strings: list[str]) -> dict:
    path = path.resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"benchmark binary is missing or not executable: {path}")
    for marker in required_strings:
        if (
            type(marker) is not str
            or not marker
            or not binary_contains(path, marker.encode())
        ):
            raise ValueError(f"benchmark binary is missing required marker: {marker!r}")
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
        "required_markers": list(required_strings),
    }


def validate_launch_inputs(argv: list[str], environment: dict[str, str]) -> None:
    if (
        type(argv) is not list
        or not argv
        or any(type(value) is not str for value in argv)
    ):
        raise ValueError("argv must be a nonempty list of strings")
    if type(environment) is not dict or any(
        type(key) is not str or type(value) is not str
        for key, value in environment.items()
    ):
        raise ValueError("environment must be a string map")
    for key in environment:
        upper = key.upper()
        if (
            upper == "TOKEN"
            or upper.endswith("_TOKEN")
            or any(part in upper for part in SECRET_KEY_PARTS)
        ):
            raise ValueError(f"secret-shaped environment key is forbidden: {key}")
    for value in argv:
        lowered = value.lower()
        if lowered.split("=", 1)[0] in SECRET_ARG_FLAGS:
            raise ValueError(
                f"secret-bearing argv flag is forbidden: {value.split('=', 1)[0]}"
            )
        parsed = urlsplit(value)
        if parsed.scheme and (
            parsed.username is not None or parsed.password is not None
        ):
            raise ValueError("credential-bearing URL is forbidden in argv")
        if parsed.scheme and any(
            key.lower() in SECRET_QUERY_KEYS for key, _ in parse_qsl(parsed.query)
        ):
            raise ValueError("credential-bearing URL query is forbidden in argv")


def build_receipt(
    *,
    repo: Path,
    binary: Path,
    model: Path,
    drafter: Path | None,
    argv: list[str],
    environment: dict[str, str],
    required_binary_strings: list[str],
) -> dict:
    validate_launch_inputs(argv, environment)
    inherited_environment = read_process_environment(os.getpid())
    inherited_environment.update(environment)
    manifest = {
        "schema": SCHEMA,
        "receipt_state": "PLANNED",
        "git": git_identity(Path(repo)),
        "binary": binary_identity(Path(binary), required_binary_strings),
        "model": checkpoint_identity(Path(model), "model"),
        "drafter": checkpoint_identity(Path(drafter), "drafter") if drafter else None,
        "argv": list(argv),
        "environment": dict(sorted(environment.items())),
        "full_environment_sha256": environment_sha256(inherited_environment),
    }
    digest = hashlib.sha256(canonical_json(manifest)).hexdigest()
    return {"manifest": manifest, "manifest_sha256": digest}


def verify_envelope(envelope: dict) -> bool:
    if type(envelope) is not dict:
        return False
    keys = set(envelope)
    if keys not in (
        {"manifest", "manifest_sha256"},
        {"manifest", "manifest_sha256", "activation", "activation_sha256"},
    ):
        return False
    if (
        type(envelope["manifest"]) is not dict
        or type(envelope["manifest_sha256"]) is not str
    ):
        return False
    expected = hashlib.sha256(canonical_json(envelope["manifest"])).hexdigest()
    if expected != envelope["manifest_sha256"]:
        return False
    if "activation" not in envelope:
        return True
    if (
        type(envelope["activation"]) is not dict
        or type(envelope["activation_sha256"]) is not str
    ):
        return False
    active_expected = hashlib.sha256(canonical_json(envelope["activation"])).hexdigest()
    return active_expected == envelope["activation_sha256"]


def validate_manifest(manifest: dict) -> None:
    required = {
        "schema": str,
        "receipt_state": str,
        "git": dict,
        "binary": dict,
        "model": dict,
        "argv": list,
        "environment": dict,
        "full_environment_sha256": str,
    }
    if any(
        type(manifest.get(key)) is not expected for key, expected in required.items()
    ):
        raise ValueError("benchmark receipt manifest is incomplete or malformed")
    if manifest["schema"] != SCHEMA:
        raise ValueError("benchmark receipt schema is unsupported")
    if manifest["receipt_state"] != "PLANNED":
        raise ValueError("benchmark receipt state is unsupported")
    if len(manifest["full_environment_sha256"]) != 64 or any(
        character not in "0123456789abcdef"
        for character in manifest["full_environment_sha256"]
    ):
        raise ValueError("benchmark receipt full environment digest is malformed")


def validate_activation(envelope: dict) -> None:
    if "activation" not in envelope:
        return
    activation = envelope["activation"]
    required = {
        "state": str,
        "manifest_sha256": str,
        "pid": int,
        "process_start_ticks": int,
        "process_exe_path": str,
        "process_exe_sha256": str,
        "process_argv": list,
        "environment": dict,
        "listen_port": int,
        "base_url": str,
        "model_id": str,
        "host_boot_id": str,
        "gpu_identity": dict,
        "gpu_state": dict,
    }
    if any(
        type(activation.get(key)) is not expected for key, expected in required.items()
    ):
        raise ValueError("benchmark receipt activation is incomplete or malformed")
    if activation["state"] != "ACTIVE_VERIFIED":
        raise ValueError("benchmark receipt activation state is unsupported")
    if activation["manifest_sha256"] != envelope["manifest_sha256"]:
        raise ValueError("benchmark receipt activation binds the wrong manifest")
    validate_boot_id(activation["host_boot_id"])
    identity_keys = set(GPU_QUERY_FIELDS[:3])
    state_keys = set(GPU_QUERY_FIELDS[3:])
    if (
        set(activation["gpu_identity"]) != identity_keys
        or set(activation["gpu_state"]) != state_keys
    ):
        raise ValueError("benchmark receipt GPU snapshot is malformed")
    if any(
        type(value) is not str
        for value in (
            *activation["gpu_identity"].values(),
            *activation["gpu_state"].values(),
        )
    ):
        raise ValueError("benchmark receipt GPU snapshot is malformed")


def read_envelope(path: Path) -> dict:
    path = Path(path).resolve()
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"benchmark receipt is unreadable: {path}") from error
    if not verify_envelope(envelope):
        raise ValueError("benchmark receipt envelope or digest is invalid")
    validate_manifest(envelope["manifest"])
    validate_activation(envelope)
    return envelope


def process_start_ticks(pid: int) -> int:
    try:
        stat = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
        fields = stat[stat.rindex(")") + 2 :].split()
        return int(fields[19])
    except (OSError, UnicodeError, ValueError, IndexError) as error:
        raise ValueError(f"benchmark process is unavailable: pid={pid}") from error


def process_argv(pid: int) -> list[str]:
    try:
        raw = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    except OSError as error:
        raise ValueError(
            f"benchmark process cmdline is unavailable: pid={pid}"
        ) from error
    argv = [os.fsdecode(value) for value in raw.split(b"\0") if value]
    if not argv:
        raise ValueError(f"benchmark process cmdline is empty: pid={pid}")
    return argv


def read_process_environment(pid: int) -> dict[str, str]:
    try:
        raw = (Path("/proc") / str(pid) / "environ").read_bytes()
    except OSError as error:
        raise ValueError(
            f"benchmark process environment is unavailable: pid={pid}"
        ) from error
    entries = {}
    for entry in raw.split(b"\0"):
        if b"=" in entry:
            key, value = entry.split(b"=", 1)
            entries[os.fsdecode(key)] = os.fsdecode(value)
    return entries


def process_environment(
    pid: int, expected: dict[str, str], expected_sha256: str
) -> dict[str, str]:
    entries = read_process_environment(pid)
    for key, value in expected.items():
        if entries.get(key) != value:
            raise ValueError(f"benchmark process environment mismatch: {key}")
    if environment_sha256(entries) != expected_sha256:
        raise ValueError("benchmark process full environment digest changed")
    return dict(sorted(expected.items()))


def process_executable(pid: int) -> tuple[str, str]:
    probe = Path("/proc") / str(pid) / "exe"
    try:
        path = probe.resolve(strict=True)
        digest = sha256_file(probe)
    except OSError as error:
        raise ValueError(
            f"benchmark process executable is unavailable: pid={pid}"
        ) from error
    return str(path), digest


def proc_listen_address(host: str) -> tuple[str, str]:
    try:
        address = ipaddress.ip_address(host)
    except ValueError as error:
        raise ValueError("benchmark host must be a literal loopback address") from error
    if not address.is_loopback or str(address) != host:
        raise ValueError("benchmark host must be a canonical literal loopback address")
    if address.version == 4:
        encoded = socket.inet_pton(socket.AF_INET, host)[::-1].hex().upper()
        return "tcp", encoded
    packed = socket.inet_pton(socket.AF_INET6, host)
    encoded = "".join(f"{word:08X}" for word in struct.unpack("=4I", packed))
    return "tcp6", encoded


def process_owns_listen_address(pid: int, host: str, port: int) -> bool:
    if type(port) is not int or not 0 < port <= 65535:
        raise ValueError(f"benchmark listen port is invalid: {port!r}")
    inodes = set()
    try:
        for fd in (Path("/proc") / str(pid) / "fd").iterdir():
            try:
                target = os.readlink(fd)
            except OSError:
                continue
            if target.startswith("socket:[") and target.endswith("]"):
                inodes.add(target[8:-1])
    except OSError as error:
        raise ValueError(
            f"benchmark process descriptors are unavailable: pid={pid}"
        ) from error
    table_name, expected_address = proc_listen_address(host)
    table = Path("/proc/net") / table_name
    for table in (table,):
        try:
            lines = table.read_text(encoding="ascii").splitlines()[1:]
        except OSError:
            continue
        for line in lines:
            fields = line.split()
            if len(fields) > 9 and fields[3] == "0A":
                local_address, raw_port = fields[1].rsplit(":", 1)
                local_port = int(raw_port, 16)
                if (
                    local_address == expected_address
                    and local_port == port
                    and fields[9] in inodes
                ):
                    return True
    return False


def server_model_id(base_url: str) -> str:
    parsed = urlsplit(base_url)
    if (
        parsed.scheme != "http"
        or parsed.hostname is None
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("benchmark base URL must be loopback HTTP")
    proc_listen_address(parsed.hostname)
    try:
        with urlopen_no_redirect(
            f"{base_url.rstrip('/')}/v1/models", timeout=5
        ) as response:
            payload = json.load(response)
        model_id = payload["data"][0]["id"]
    except (OSError, KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("benchmark server model identity is unavailable") from error
    if type(model_id) is not str or not model_id:
        raise ValueError("benchmark server model identity is malformed")
    return model_id


def current_activation(
    envelope: dict,
    pid: int,
    base_url: str,
    port: int,
    *,
    gpu_sample: dict | None = None,
    recorded_gpu_state: dict | None = None,
) -> dict:
    manifest = envelope["manifest"]
    planned_git = manifest["git"]
    planned_repo = planned_git.get("repo")
    if type(planned_repo) is not str or not planned_repo:
        raise ValueError("benchmark repository identity is malformed")
    try:
        base_port = urlsplit(base_url).port
    except ValueError as error:
        raise ValueError("benchmark base URL port is malformed") from error
    if base_port != port:
        raise ValueError("benchmark base URL and listening port disagree")
    start_ticks = process_start_ticks(pid)
    exe_path, exe_sha256 = process_executable(pid)
    if Path(exe_path).resolve() != Path(manifest["binary"]["path"]).resolve():
        raise ValueError(
            "benchmark process executable path does not match planned binary"
        )
    if exe_sha256 != manifest["binary"]["sha256"]:
        raise ValueError(
            "benchmark process executable digest does not match planned binary"
        )
    argv = process_argv(pid)
    if argv != manifest["argv"]:
        raise ValueError("benchmark process argv does not match planned argv")
    environment = process_environment(
        pid, manifest["environment"], manifest["full_environment_sha256"]
    )
    host = urlsplit(base_url).hostname
    if host is None or not process_owns_listen_address(pid, host, port):
        raise ValueError(f"benchmark process does not own listening port {port}")
    # Readiness retries must stay cheap. Hash the dirty source tree and model
    # only after the exact process owns the requested listener.
    if git_identity(Path(planned_repo)) != planned_git:
        raise ValueError("benchmark repository changed after planning")
    if (
        checkpoint_identity(Path(manifest["model"]["path"]), "model")
        != manifest["model"]
    ):
        raise ValueError("benchmark model checkpoint changed after planning")
    planned_drafter = manifest.get("drafter")
    if (
        planned_drafter is not None
        and checkpoint_identity(Path(planned_drafter["path"]), "drafter")
        != planned_drafter
    ):
        raise ValueError("benchmark drafter checkpoint changed after planning")
    gpu = gpu_sample if gpu_sample is not None else gpu_snapshot()
    return {
        "state": "ACTIVE_VERIFIED",
        "manifest_sha256": envelope["manifest_sha256"],
        "pid": pid,
        "process_start_ticks": start_ticks,
        "process_exe_path": exe_path,
        "process_exe_sha256": exe_sha256,
        "process_argv": argv,
        "environment": environment,
        "listen_port": port,
        "base_url": base_url.rstrip("/"),
        "model_id": server_model_id(base_url),
        "host_boot_id": host_boot_id(),
        "gpu_identity": gpu["identity"],
        "gpu_state": (
            recorded_gpu_state if recorded_gpu_state is not None else gpu["state"]
        ),
    }


def activate_envelope(envelope: dict, pid: int, base_url: str, port: int) -> dict:
    if not verify_envelope(envelope) or "activation" in envelope:
        raise ValueError("only a valid planned receipt can be activated")
    validate_manifest(envelope["manifest"])
    activation = current_activation(envelope, pid, base_url, port)
    return {
        **envelope,
        "activation": activation,
        "activation_sha256": hashlib.sha256(canonical_json(activation)).hexdigest(),
    }


def verify_active_envelope(envelope: dict, base_url: str) -> None:
    if not verify_envelope(envelope) or "activation" not in envelope:
        raise ValueError("active benchmark receipt is missing or invalid")
    validate_manifest(envelope["manifest"])
    validate_activation(envelope)
    activation = envelope["activation"]
    if activation["base_url"] != base_url.rstrip("/"):
        raise ValueError("active benchmark receipt base URL mismatch")
    live_gpu = gpu_snapshot()
    if live_gpu["identity"] != activation["gpu_identity"]:
        raise ValueError("active benchmark GPU identity changed")
    current = current_activation(
        {
            "manifest": envelope["manifest"],
            "manifest_sha256": envelope["manifest_sha256"],
        },
        activation["pid"],
        activation["base_url"],
        activation["listen_port"],
        gpu_sample=live_gpu,
        recorded_gpu_state=activation["gpu_state"],
    )
    if current != activation:
        raise ValueError("active benchmark runtime identity changed")


def receipt_state(envelope: dict, base_url: str | None = None) -> str:
    if "activation" not in envelope:
        return envelope["manifest"]["receipt_state"]
    if base_url is None:
        raise ValueError("active benchmark receipt requires a base URL")
    verify_active_envelope(envelope, base_url)
    return envelope["activation"]["state"]


def receipt_digest(envelope: dict) -> str:
    return envelope.get("activation_sha256", envelope["manifest_sha256"])


def wait_for_activation(
    envelope: dict, pid: int, base_url: str, port: int, timeout_seconds: float
) -> dict:
    if timeout_seconds <= 0.0:
        raise ValueError("benchmark activation timeout must be positive")
    deadline = time.monotonic() + timeout_seconds
    last_error = None
    while time.monotonic() < deadline:
        try:
            return activate_envelope(envelope, pid, base_url, port)
        except ValueError as error:
            last_error = error
            try:
                os.kill(pid, 0)
            except OSError as process_error:
                raise ValueError(
                    f"benchmark process exited before readiness: pid={pid}"
                ) from process_error
            time.sleep(1)
    raise ValueError(f"benchmark activation timed out: {last_error}")


def parse_json_argument(raw: str, expected_type: type, label: str):
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if type(value) is not expected_type:
        raise ValueError(f"{label} has the wrong JSON type")
    return value


def write_receipt(path: Path, encoded: str) -> None:
    try:
        with Path(path).open("x", encoding="utf-8") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
    except FileExistsError as error:
        raise ValueError(f"benchmark receipt already exists: {path}") from error


def main() -> None:
    if "--activate-receipt" in sys.argv:
        parser = argparse.ArgumentParser(
            description="Activate a planned benchmark receipt"
        )
        parser.add_argument("--activate-receipt", type=Path, required=True)
        parser.add_argument("--pid", type=int, required=True)
        parser.add_argument("--base-url", required=True)
        parser.add_argument("--port", type=int, required=True)
        parser.add_argument("--ready-timeout", type=float, default=600.0)
        parser.add_argument("--output", type=Path, required=True)
        args = parser.parse_args()
        try:
            envelope = wait_for_activation(
                read_envelope(args.activate_receipt),
                args.pid,
                args.base_url,
                args.port,
                args.ready_timeout,
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error
        encoded = json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n"
        try:
            write_receipt(args.output, encoded)
        except ValueError as error:
            raise SystemExit(str(error)) from error
        sys.stdout.write(encoded)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--drafter", type=Path)
    parser.add_argument("--argv-json", required=True)
    parser.add_argument("--environment-json", default="{}")
    parser.add_argument("--require-string", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        envelope = build_receipt(
            repo=args.repo,
            binary=args.binary,
            model=args.model,
            drafter=args.drafter,
            argv=parse_json_argument(args.argv_json, list, "argv"),
            environment=parse_json_argument(args.environment_json, dict, "environment"),
            required_binary_strings=args.require_string,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    encoded = json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n"
    if args.output:
        try:
            write_receipt(args.output, encoded)
        except ValueError as error:
            raise SystemExit(str(error)) from error
    sys.stdout.write(encoded)


if __name__ == "__main__":
    main()
