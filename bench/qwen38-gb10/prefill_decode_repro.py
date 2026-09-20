#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Provenance-locked Qwen3.8 C=1 prefill and decode qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import select
import stat
import statistics
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

PREFILL_TARGET_TPS = 2_000.0
DECODE_TARGET_TPS = 85.0
DECODE_COMPLETE_FINISH_REASON = "length"
DEFAULT_INPUTS = (2_048, 8_192, 32_768)
DEFAULT_PREFILL_CONTINUATION_TOKENS = 32
# Pinned by optimized-qwen/generation_config.json plus the Qwen ChatML
# role-boundary hard stop registered by the server at startup. A prefill probe
# measures TTFT, but still needs a complete continuation to prove that the
# request did not terminate on a model stop token after the timed first token.
PREFILL_SUPPRESSED_STOP_TOKEN_IDS = (248_044, 248_045, 248_046)
TARGET_VS_DFLASH = "target-vs-dflash"
SAME_MODE = "same-mode"
RUNTIME_MODES = ("no-spec", "dflash-v3")
REFERENCE_SCHEMA = "qwen38-prefill-reference-v6"
REPORT_SCHEMA = "qwen38-prefill-decode-v6"
SEMANTIC_ORACLE_SCHEMA = "qwen38-semantic-oracle-v1"
# v1 deliberately binds only evidence exposed by the current completion APIs:
# exact request/output hashes, prompt/completion counts, and finish semantics.
# A future schema may add completion-token IDs and tolerance-bounded top-k
# logits once the server exposes them; they must not be invented by retokenizing
# response text or silently made optional inside this exact-key v1 contract.
CORRECTED_ATTN_PTX_SHA256 = (
    "71d95d7815d36cc0df61070d125b1598464a67aaaac44425f5f4a0ba6d6e61c2"
)
SEMANTIC_ORACLE_MAX_BYTES = 4 * 1024 * 1024
ROUTE_MARKER_TOKEN = "engaged"
MAX_ROUTE_MARKERS = 128
MAX_ENVIRONMENT_ENTRIES = 256
QUALIFICATION_NONCE_ENV = "ATLAS_QUALIFICATION_RUN_NONCE"
QUALIFICATION_LOG_PREFIX = "ATLAS_QUALIFICATION_RUN"
QUALIFIABLE_SAME_MODE_FLAGS = frozenset(
    {
        "ATLAS_E2M1_GEMM",
        "ATLAS_E2M1_GEMM_DOWN_ONLY",
        "ATLAS_E2M1_KMAJOR",
        "ATLAS_E2M1_KMAJOR_M256",
        "ATLAS_E2M1_SILU_QUANT",
        "ATLAS_E2M1_STATIC_SCALE",
        "ATLAS_ATTN_GATE_BATCHED",
        "ATLAS_ATTN_QKV_EXACT_M17_ASTAGE",
        "ATLAS_DFLASH_PREFILL_PIPE",
        "ATLAS_GDN_C143_PREFILL",
        "ATLAS_GDN_PREFILL_GATECACHE",
        "ATLAS_PREFILL_ATTN_GATE_FUSED",
        "ATLAS_PREFILL_ATTN_BR128",
        "ATLAS_PREFILL_FFN_DUAL_FUSED",
        "ATLAS_PREFILL_FFN_FLASHINFER",
        "ATLAS_PREFILL_PROJ_FLASHINFER",
        "ATLAS_PREFILL_SSM_FLASHINFER",
        "ATLAS_PREFILL_FFN_FUSED_EPILOGUE",
        "ATLAS_PREFILL_FFN_PIPE",
        "ATLAS_PREFILL_KV_DUAL",
        "ATLAS_PREFILL_PROJ_FAST",
        "ATLAS_PREFILL_PROJ_PIPE",
        "ATLAS_PREFILL_QKNORM_ROPE",
        "ATLAS_SSM_PREFILL_CONV_L2",
        "ATLAS_SSM_PREFILL_PACK",
    }
)
COMMON_PROVENANCE_KEYS = (
    "source_revision",
    "binary_sha256",
    "kernel_bundle_sha256",
    "model_index_sha256",
    "tokenizer_sha256",
    "effective_environment_sha256",
)
DECODE_PROMPT = (
    "Write a complete MinHeap class in Python with insert, extract_min, and heapify, "
    "then explain the complexity of each method."
)

SEMANTIC_ORACLE_KEYS = frozenset(
    {
        "schema",
        "model",
        "model_index_sha256",
        "tokenizer_sha256",
        "corrected_attention_abi",
        "prefill_rows",
        "decode_rows",
    }
)
CORRECTED_ATTENTION_ABI_KEYS = frozenset(
    {"host_argument_count", "ptx_parameter_count", "ptx_sha256"}
)
PREFILL_ORACLE_ROW_KEYS = frozenset(
    {
        "target_prompt_tokens",
        "index",
        "request_body_sha256",
        "prompt_tokens",
        "stable_output_sha256",
        "requested_continuation_tokens",
        "completion_tokens",
        "finish_reason",
    }
)
DECODE_ORACLE_ROW_KEYS = frozenset(
    {
        "index",
        "request_body_sha256",
        "prompt_tokens",
        "stable_output_sha256",
        "requested_completion_tokens",
        "completion_tokens",
        "finish_reason",
    }
)


def is_route_engagement_line(line: bytes) -> bool:
    """Distinguish one-shot route markers from ordinary telemetry fields.

    Runtime timing records contain names such as ``async_engaged=0`` and some
    policy traces end in lower-case prose such as ``fast path engaged``. Those
    are not route attestations. Production route markers either use the
    explicit upper-case ``ENGAGED`` sentinel or name an upper-case ATLAS/DFLASH
    route immediately before the lower-case word ``engaged``.
    """

    return b"ENGAGED " in line or (
        (b"ATLAS_" in line or b"DFLASH_" in line) and b" engaged" in line
    )


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_hex(value: str, length: int) -> bool:
    return len(value) == length and all(
        character in "0123456789abcdef" for character in value
    )


def _require_exact_keys(value: object, keys: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        present = set(value) if isinstance(value, dict) else set()
        raise ValueError(
            f"{label} keys differ: missing={sorted(keys - present)!r} "
            f"unknown={sorted(present - keys)!r}"
        )
    return value


def validate_route_marker_spec(value: object) -> dict[str, int]:
    if not isinstance(value, dict) or not value:
        raise ValueError("provenance required_route_markers must be a non-empty object")
    if len(value) > MAX_ROUTE_MARKERS:
        raise ValueError(
            f"provenance required_route_markers exceeds {MAX_ROUTE_MARKERS} entries"
        )
    markers: dict[str, int] = {}
    for marker, count in value.items():
        if (
            not isinstance(marker, str)
            or not is_route_engagement_line(marker.encode("utf-8"))
            or len(marker) > 256
            or "\n" in marker
            or "\r" in marker
        ):
            raise ValueError(
                "route marker names must be single-line engagement messages of at most 256 characters"
            )
        if type(count) is not int or not 1 <= count <= MAX_ROUTE_MARKERS:
            raise ValueError(
                f"route marker {marker!r} count must be an integer from 1 to {MAX_ROUTE_MARKERS}"
            )
        markers[marker] = count
    ordered = sorted(markers)
    for index, marker in enumerate(ordered):
        if any(
            marker in other or other in marker
            for other_index, other in enumerate(ordered)
            if other_index != index
        ):
            raise ValueError("required route markers must not overlap")
    return {marker: markers[marker] for marker in ordered}


def validate_effective_environment(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ValueError("provenance effective_environment must be a non-empty object")
    if len(value) > MAX_ENVIRONMENT_ENTRIES:
        raise ValueError(
            f"provenance effective_environment exceeds {MAX_ENVIRONMENT_ENTRIES} entries"
        )
    environment: dict[str, str] = {}
    for key, item in value.items():
        if (
            not isinstance(key, str)
            or not key.startswith("ATLAS_")
            or key == QUALIFICATION_NONCE_ENV
            or not key
            or len(key) > 128
            or "=" in key
            or "\x00" in key
            or "\n" in key
            or "\r" in key
        ):
            raise ValueError(
                "effective environment keys must be bounded ATLAS_* names and must exclude the qualification nonce"
            )
        if (
            not isinstance(item, str)
            or len(item) > 4_096
            or "\x00" in item
            or "\n" in item
            or "\r" in item
        ):
            raise ValueError(
                f"effective environment value for {key!r} must be a bounded single-line string"
            )
        environment[key] = item
    return {key: environment[key] for key in sorted(environment)}


def validate_same_mode_environment_delta(
    value: object,
) -> dict[str, dict[str, str]]:
    if not isinstance(value, dict):
        raise ValueError("provenance same_mode_environment_delta must be an object")
    delta: dict[str, dict[str, str]] = {}
    for key, item in value.items():
        if (
            not isinstance(key, str)
            or not key.startswith("ATLAS_")
            or key == QUALIFICATION_NONCE_ENV
        ):
            raise ValueError("same-mode delta keys must be ATLAS_* names")
        if not isinstance(item, dict) or set(item) != {"control", "candidate"}:
            raise ValueError(
                f"same-mode delta for {key!r} must contain only control and candidate"
            )
        control = item.get("control")
        candidate = item.get("candidate")
        if (
            not isinstance(control, str)
            or not isinstance(candidate, str)
            or control == candidate
            or any(
                "\x00" in part or "\n" in part or "\r" in part
                for part in (control, candidate)
            )
        ):
            raise ValueError(
                f"same-mode delta for {key!r} must contain distinct single-line strings"
            )
        delta[key] = {"control": control, "candidate": candidate}
    ordered = {key: delta[key] for key in sorted(delta)}
    if ordered:
        if len(ordered) != 1:
            raise ValueError("same-mode qualification changes exactly one ATLAS flag")
        key, item = next(iter(ordered.items()))
        if key not in QUALIFIABLE_SAME_MODE_FLAGS or item != {
            "control": "0",
            "candidate": "1",
        }:
            raise ValueError(
                "same-mode qualification permits one known kernel flag changing exactly 0 to 1"
            )
    return ordered


def validate_provenance(value: object, expected_mode: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("provenance manifest must be a JSON object")
    provenance: dict[str, Any] = {}
    mode = value.get("runtime_mode")
    if mode != expected_mode:
        raise ValueError(
            f"provenance runtime_mode must be {expected_mode!r}, got {mode!r}"
        )
    provenance["runtime_mode"] = mode
    for key in COMMON_PROVENANCE_KEYS:
        item = value.get(key)
        if not isinstance(item, str):
            raise ValueError(f"provenance {key} must be a string")
        item = item.lower()
        expected_length = 40 if key == "source_revision" else 64
        if not _is_hex(item, expected_length):
            raise ValueError(
                f"provenance {key} must be {expected_length} lowercase hexadecimal characters"
            )
        provenance[key] = item
    environment = validate_effective_environment(value.get("effective_environment"))
    environment_sha256 = sha256(canonical_bytes(environment))
    if environment_sha256 != provenance["effective_environment_sha256"]:
        raise ValueError(
            "provenance effective_environment_sha256 does not match the canonical effective_environment object"
        )
    provenance["effective_environment"] = environment
    nonce = value.get("server_run_nonce")
    if not isinstance(nonce, str) or not _is_hex(nonce, 64):
        raise ValueError(
            "provenance server_run_nonce must be 64 lowercase hexadecimal characters"
        )
    provenance["server_run_nonce"] = nonce
    provenance["same_mode_environment_delta"] = validate_same_mode_environment_delta(
        value.get("same_mode_environment_delta")
    )
    full_environment_sha256 = value.get("full_environment_sha256")
    if not isinstance(full_environment_sha256, str) or not _is_hex(
        full_environment_sha256, 64
    ):
        raise ValueError(
            "provenance full_environment_sha256 must be 64 lowercase hexadecimal characters"
        )
    provenance["full_environment_sha256"] = full_environment_sha256
    command_line_sha256 = value.get("command_line_sha256")
    if not isinstance(command_line_sha256, str) or not _is_hex(command_line_sha256, 64):
        raise ValueError(
            "provenance command_line_sha256 must be 64 lowercase hexadecimal characters"
        )
    provenance["command_line_sha256"] = command_line_sha256
    if expected_mode == "dflash-v3":
        draft_hash = value.get("draft_index_sha256")
        if not isinstance(draft_hash, str) or not _is_hex(draft_hash.lower(), 64):
            raise ValueError(
                "dflash-v3 provenance draft_index_sha256 must be 64 lowercase hexadecimal characters"
            )
        provenance["draft_index_sha256"] = draft_hash.lower()
    provenance["required_route_markers"] = validate_route_marker_spec(
        value.get("required_route_markers")
    )
    return provenance


def resolve_runtime_mode(
    comparison_kind: str, writing_reference: bool, requested_mode: str | None
) -> str:
    if comparison_kind == TARGET_VS_DFLASH:
        expected = "no-spec" if writing_reference else "dflash-v3"
        if requested_mode is not None and requested_mode != expected:
            raise ValueError(
                f"{TARGET_VS_DFLASH} requires runtime mode {expected!r} for this stage"
            )
        return expected
    if comparison_kind == SAME_MODE:
        if requested_mode not in RUNTIME_MODES:
            raise ValueError(
                f"{SAME_MODE} requires --runtime-mode no-spec or dflash-v3"
            )
        return requested_mode
    raise ValueError(f"unsupported comparison kind {comparison_kind!r}")


def load_provenance(path: Path, expected_mode: str) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    return validate_provenance(json.loads(raw), expected_mode), sha256(raw)


def endpoint_port(endpoint: str) -> int:
    parsed = urllib.parse.urlparse(endpoint)
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path != "/v1/completions"
        or parsed.params
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError(
            "qualification endpoint must be exactly http://127.0.0.1:<port>/v1/completions"
        )
    try:
        port = parsed.port
    except ValueError as error:
        raise ValueError("qualification endpoint has an invalid port") from error
    if port is None:
        raise ValueError("qualification endpoint must include an explicit port")
    return port


def _listening_socket_inodes(proc_root: Path, port: int) -> set[str]:
    inodes: set[str] = set()
    path = proc_root / "net/tcp"
    try:
        lines = path.read_text().splitlines()
    except OSError as error:
        raise ValueError(f"cannot inspect IPv4 listener table: {error}") from error
    if not lines:
        raise ValueError("IPv4 listener table is empty")
    for line in lines[1:]:
        fields = line.split()
        if len(fields) < 10:
            raise ValueError("IPv4 listener table contains a malformed row")
        if fields[3] != "0A":
            continue
        address, separator, encoded_port = fields[1].rpartition(":")
        if not separator:
            raise ValueError("IPv4 listener table contains a malformed address")
        try:
            local_port = int(encoded_port, 16)
        except ValueError as error:
            raise ValueError("IPv4 listener table contains a malformed port") from error
        if local_port == port and address == "0100007F":
            inodes.add(fields[9])
    if len(inodes) != 1:
        raise ValueError(
            f"endpoint must have exactly one 127.0.0.1 listener socket, found {sorted(inodes)}"
        )
    return inodes


def _listener_pid(proc_root: Path, socket_inodes: set[str]) -> int:
    owners: set[int] = set()
    try:
        processes = list(proc_root.iterdir())
    except OSError as error:
        raise ValueError(f"cannot inspect process table: {error}") from error
    expected_links = {f"socket:[{inode}]" for inode in socket_inodes}
    for process in processes:
        if not process.name.isdecimal():
            continue
        try:
            if process.stat().st_uid != os.getuid():
                continue
        except FileNotFoundError:
            continue
        except OSError as error:
            raise ValueError(
                f"cannot identify process owner for {process.name}: {error}"
            ) from error
        try:
            descriptors = list((process / "fd").iterdir())
        except FileNotFoundError:
            continue
        except PermissionError:
            # Same-UID supervisors may be non-dumpable even though the exact
            # listener process is inspectable. Ignore an unrelated opaque
            # process; if the real listener is opaque, no owner is found and
            # the exact-one-owner check below still fails closed.
            continue
        except OSError as error:
            raise ValueError(
                f"procfs does not expose descriptors for same-UID process {process.name}: {error}"
            ) from error
        for descriptor in descriptors:
            try:
                target = os.readlink(descriptor)
            except FileNotFoundError:
                continue
            except PermissionError:
                continue
            except OSError as error:
                raise ValueError(
                    f"cannot inspect descriptor {descriptor}: {error}"
                ) from error
            if target in expected_links:
                owners.add(int(process.name))
                break
    if len(owners) != 1:
        raise ValueError(
            f"endpoint listener must resolve to exactly one process, found {sorted(owners)}"
        )
    return next(iter(owners))


def _writable_route_log_descriptors(
    proc_root: Path, route_identity: tuple[int, int]
) -> set[tuple[int, int]]:
    writers: set[tuple[int, int]] = set()
    try:
        processes = list(proc_root.iterdir())
    except OSError as error:
        raise ValueError(f"cannot inspect process table: {error}") from error
    for process in processes:
        if not process.name.isdecimal():
            continue
        try:
            if process.stat().st_uid != os.getuid():
                continue
        except FileNotFoundError:
            continue
        except OSError as error:
            raise ValueError(
                f"cannot identify process owner for {process.name}: {error}"
            ) from error
        try:
            descriptors = list((process / "fd").iterdir())
        except FileNotFoundError:
            continue
        except PermissionError:
            # An unrelated same-UID supervisor can be non-dumpable. Its
            # descriptors are opaque, but an opaque listener still fails the
            # exact stdout/stderr writer-set check in bind_live_server.
            continue
        except OSError as error:
            raise ValueError(
                f"procfs does not expose descriptors for same-UID process {process.name}: {error}"
            ) from error
        for descriptor in descriptors:
            try:
                descriptor_stat = descriptor.stat()
            except FileNotFoundError:
                continue
            except PermissionError:
                # See the directory-level case above. Never suppress fdinfo
                # errors after an FD is proven to name the route-log inode.
                continue
            except OSError as error:
                raise ValueError(
                    f"cannot inspect descriptor {descriptor}: {error}"
                ) from error
            if (descriptor_stat.st_dev, descriptor_stat.st_ino) != route_identity:
                continue
            try:
                fdinfo = (process / "fdinfo" / descriptor.name).read_text().splitlines()
            except OSError as error:
                raise ValueError(
                    f"cannot inspect descriptor flags for {descriptor}: {error}"
                ) from error
            flag_lines = [line for line in fdinfo if line.startswith("flags:")]
            if len(flag_lines) != 1:
                raise ValueError(f"descriptor flags for {descriptor} are malformed")
            try:
                flags = int(flag_lines[0].split()[1], 8)
                descriptor_number = int(descriptor.name)
            except (IndexError, ValueError) as error:
                raise ValueError(
                    f"descriptor flags for {descriptor} are malformed"
                ) from error
            if flags & os.O_ACCMODE != os.O_RDONLY:
                writers.add((int(process.name), descriptor_number))
    return writers


def _process_start_time_ticks(raw_stat: str) -> int:
    close = raw_stat.rfind(")")
    if close < 0:
        raise ValueError("listener /proc stat is malformed")
    fields = raw_stat[close + 1 :].split()
    if len(fields) <= 19:
        raise ValueError("listener /proc stat is truncated")
    try:
        start_time = int(fields[19])
    except ValueError as error:
        raise ValueError("listener /proc start time is malformed") from error
    if start_time <= 0:
        raise ValueError("listener /proc start time must be positive")
    return start_time


def _atlas_environment(raw: bytes) -> dict[str, str]:
    environment: dict[str, str] = {}
    for entry in raw.split(b"\x00"):
        if not entry.startswith(b"ATLAS_"):
            continue
        key_bytes, separator, value_bytes = entry.partition(b"=")
        if not separator:
            raise ValueError("listener contains a malformed ATLAS environment entry")
        try:
            key = key_bytes.decode("utf-8")
            value = value_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("listener ATLAS environment is not UTF-8") from error
        if key in environment:
            raise ValueError(f"listener repeats environment key {key!r}")
        if key != QUALIFICATION_NONCE_ENV:
            environment[key] = value
    return validate_effective_environment(environment)


def _sha256_fd(descriptor: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    while chunk := os.pread(descriptor, 1024 * 1024, offset):
        digest.update(chunk)
        offset += len(chunk)
    return digest.hexdigest()


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"semantic oracle repeats JSON key {key!r}")
        value[key] = item
    return value


def _semantic_oracle_snapshot(
    path: Path, descriptor: int
) -> tuple[bytes, os.stat_result]:
    for _ in range(3):
        before = os.fstat(descriptor)
        try:
            named_before = path.lstat()
        except OSError as error:
            raise ValueError(f"semantic oracle path disappeared: {error}") from error
        identity = (before.st_dev, before.st_ino)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.getuid()
            or before.st_mode & 0o222
            or before.st_size <= 0
            or before.st_size > SEMANTIC_ORACLE_MAX_BYTES
            or stat.S_ISLNK(named_before.st_mode)
            or identity != (named_before.st_dev, named_before.st_ino)
        ):
            raise ValueError(
                "semantic oracle must be a caller-owned, single-link, read-only "
                f"regular file of at most {SEMANTIC_ORACLE_MAX_BYTES} bytes"
            )
        chunks: list[bytes] = []
        offset = 0
        while offset < before.st_size:
            chunk = os.pread(
                descriptor, min(1024 * 1024, before.st_size - offset), offset
            )
            if not chunk:
                break
            chunks.append(chunk)
            offset += len(chunk)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
        try:
            named_after = path.lstat()
        except OSError as error:
            raise ValueError(f"semantic oracle path disappeared: {error}") from error
        if (
            len(raw) == before.st_size
            and (
                before.st_dev,
                before.st_ino,
                before.st_mode,
                before.st_nlink,
                before.st_uid,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            == (
                after.st_dev,
                after.st_ino,
                after.st_mode,
                after.st_nlink,
                after.st_uid,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            and (after.st_dev, after.st_ino)
            == (named_after.st_dev, named_after.st_ino)
        ):
            return raw, after
    raise ValueError("semantic oracle changed while it was being read")


def _validate_oracle_hash(value: object, label: str) -> str:
    if not isinstance(value, str) or not _is_hex(value, 64):
        raise ValueError(f"semantic oracle {label} must be a lowercase SHA-256")
    return value


def validate_semantic_oracle(value: object) -> dict[str, Any]:
    oracle = _require_exact_keys(value, SEMANTIC_ORACLE_KEYS, "semantic oracle")
    if oracle["schema"] != SEMANTIC_ORACLE_SCHEMA:
        raise ValueError(f"semantic oracle requires {SEMANTIC_ORACLE_SCHEMA}")
    model = oracle["model"]
    if (
        not isinstance(model, str)
        or not model
        or len(model) > 256
        or any(character in model for character in "\x00\r\n")
    ):
        raise ValueError("semantic oracle model is malformed")
    model_index_sha256 = _validate_oracle_hash(
        oracle["model_index_sha256"], "model_index_sha256"
    )
    tokenizer_sha256 = _validate_oracle_hash(
        oracle["tokenizer_sha256"], "tokenizer_sha256"
    )
    abi = _require_exact_keys(
        oracle["corrected_attention_abi"],
        CORRECTED_ATTENTION_ABI_KEYS,
        "semantic oracle corrected_attention_abi",
    )
    if (
        type(abi["host_argument_count"]) is not int
        or abi["host_argument_count"] != 13
        or type(abi["ptx_parameter_count"]) is not int
        or abi["ptx_parameter_count"] != 13
        or abi["ptx_sha256"] != CORRECTED_ATTN_PTX_SHA256
    ):
        raise ValueError(
            "semantic oracle requires corrected attention ABI 13/13 and the pinned PTX SHA-256"
        )

    prefill_rows = oracle["prefill_rows"]
    if not isinstance(prefill_rows, list) or len(prefill_rows) != 15:
        raise ValueError("semantic oracle requires exactly 15 prefill rows")
    expected_prefill = {
        (target, index) for target in DEFAULT_INPUTS for index in range(5)
    }
    seen_prefill: set[tuple[int, int]] = set()
    seen_prefill_requests: set[str] = set()
    validated_prefill: list[dict[str, Any]] = []
    for raw in prefill_rows:
        row = _require_exact_keys(
            raw, PREFILL_ORACLE_ROW_KEYS, "semantic oracle prefill row"
        )
        target = row["target_prompt_tokens"]
        index = row["index"]
        key = (target, index)
        if type(target) is not int or type(index) is not int or key not in expected_prefill:
            raise ValueError("semantic oracle prefill target/index is not canonical")
        if key in seen_prefill:
            raise ValueError(f"semantic oracle repeats prefill row {key}")
        request_hash = _validate_oracle_hash(
            row["request_body_sha256"], "prefill request_body_sha256"
        )
        output_hash = _validate_oracle_hash(
            row["stable_output_sha256"], "prefill stable_output_sha256"
        )
        if request_hash in seen_prefill_requests:
            raise ValueError("semantic oracle repeats a prefill request hash")
        prompt_tokens = row["prompt_tokens"]
        requested = row["requested_continuation_tokens"]
        completed = row["completion_tokens"]
        if (
            type(prompt_tokens) is not int
            or prompt_tokens <= 0
            or type(requested) is not int
            or requested < DEFAULT_PREFILL_CONTINUATION_TOKENS
            or type(completed) is not int
            or completed != requested
            or row["finish_reason"] != DECODE_COMPLETE_FINISH_REASON
        ):
            raise ValueError(
                "semantic oracle prefill counts/finish do not describe a complete response"
            )
        seen_prefill.add(key)
        seen_prefill_requests.add(request_hash)
        validated_prefill.append(
            {
                **row,
                "request_body_sha256": request_hash,
                "stable_output_sha256": output_hash,
            }
        )
    if seen_prefill != expected_prefill:
        raise ValueError("semantic oracle prefill rows do not cover the canonical workload")

    decode_rows = oracle["decode_rows"]
    if not isinstance(decode_rows, list) or len(decode_rows) != 5:
        raise ValueError("semantic oracle requires exactly five decode rows")
    seen_decode: set[int] = set()
    validated_decode: list[dict[str, Any]] = []
    for raw in decode_rows:
        row = _require_exact_keys(
            raw, DECODE_ORACLE_ROW_KEYS, "semantic oracle decode row"
        )
        index = row["index"]
        if type(index) is not int or index not in range(5) or index in seen_decode:
            raise ValueError("semantic oracle decode index is missing, repeated, or noncanonical")
        request_hash = _validate_oracle_hash(
            row["request_body_sha256"], "decode request_body_sha256"
        )
        output_hash = _validate_oracle_hash(
            row["stable_output_sha256"], "decode stable_output_sha256"
        )
        prompt_tokens = row["prompt_tokens"]
        requested = row["requested_completion_tokens"]
        completed = row["completion_tokens"]
        if (
            type(prompt_tokens) is not int
            or prompt_tokens <= 0
            or type(requested) is not int
            or requested != 400
            or type(completed) is not int
            or completed != requested
            or row["finish_reason"] != DECODE_COMPLETE_FINISH_REASON
        ):
            raise ValueError(
                "semantic oracle decode counts/finish do not describe a complete response"
            )
        seen_decode.add(index)
        validated_decode.append(
            {
                **row,
                "request_body_sha256": request_hash,
                "stable_output_sha256": output_hash,
            }
        )
    if seen_decode != set(range(5)):
        raise ValueError("semantic oracle decode rows do not cover the canonical workload")
    return {
        "schema": SEMANTIC_ORACLE_SCHEMA,
        "model": model,
        "model_index_sha256": model_index_sha256,
        "tokenizer_sha256": tokenizer_sha256,
        "corrected_attention_abi": dict(abi),
        "prefill_rows": validated_prefill,
        "decode_rows": validated_decode,
    }


def open_semantic_oracle(
    path: Path, expected_sha256: str
) -> tuple[dict[str, Any], str, int, tuple[int, ...]]:
    if not isinstance(expected_sha256, str) or not _is_hex(expected_sha256, 64):
        raise ValueError(
            "--expected-semantic-oracle-sha256 must be 64 lowercase hexadecimal characters"
        )
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as error:
        raise ValueError(
            f"cannot open semantic oracle without following symlinks: {error}"
        ) from error
    try:
        raw, opened = _semantic_oracle_snapshot(path, descriptor)
        actual_sha256 = sha256(raw)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "semantic oracle SHA-256 mismatch: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        parsed = json.loads(raw, object_pairs_hook=_reject_duplicate_json_keys)
        oracle = validate_semantic_oracle(parsed)
        frozen_stat = (
            opened.st_dev,
            opened.st_ino,
            opened.st_mode,
            opened.st_nlink,
            opened.st_uid,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        return oracle, actual_sha256, descriptor, frozen_stat
    except Exception:
        os.close(descriptor)
        raise


def assert_semantic_oracle_unchanged(
    path: Path,
    descriptor: int,
    expected_sha256: str,
    frozen_stat: tuple[int, ...],
) -> None:
    raw, current = _semantic_oracle_snapshot(path, descriptor)
    current_stat = (
        current.st_dev,
        current.st_ino,
        current.st_mode,
        current.st_nlink,
        current.st_uid,
        current.st_size,
        current.st_mtime_ns,
        current.st_ctime_ns,
    )
    if current_stat != frozen_stat or sha256(raw) != expected_sha256:
        raise ValueError("semantic oracle changed after admission")


def _normalized_environment_sha256(
    raw: bytes, excluded_keys: set[str] | frozenset[str] = frozenset()
) -> str:
    excluded = {key.encode() for key in excluded_keys} | {
        QUALIFICATION_NONCE_ENV.encode()
    }
    entries: list[bytes] = []
    names: set[bytes] = set()
    for entry in raw.split(b"\x00"):
        if not entry:
            continue
        name, separator, _ = entry.partition(b"=")
        if not separator or not name or name in names:
            raise ValueError(
                "listener contains a malformed or duplicate environment entry"
            )
        names.add(name)
        if name not in excluded:
            entries.append(entry)
    return sha256(b"\x00".join(sorted(entries)) + b"\x00")


def _normalized_target_command_line_sha256(raw: bytes, runtime_mode: str) -> str:
    if not raw or not raw.endswith(b"\x00"):
        raise ValueError("listener command line is missing or malformed")
    arguments = raw[:-1].split(b"\x00")

    def is_speculation_argument(argument: bytes) -> bool:
        return (
            argument == b"--dflash"
            or argument.startswith(b"--dflash=")
            or argument.startswith(b"--dflash-")
            or argument == b"--draft-model"
            or argument.startswith(b"--draft-model=")
            or argument == b"--speculative"
            or argument.startswith(b"--speculative=")
            or argument == b"--self-speculative"
            or argument.startswith(b"--self-speculative=")
            or argument == b"--ngram-speculative"
            or argument.startswith(b"--ngram-speculative=")
            or argument == b"--mtp-gate"
            or argument.startswith(b"--mtp-gate=")
        )

    if runtime_mode == "no-spec":
        if any(is_speculation_argument(argument) for argument in arguments):
            raise ValueError("no-spec command line contains a speculation argument")
        normalized = arguments
    elif runtime_mode == "dflash-v3":
        starts = [index for index, item in enumerate(arguments) if item == b"--dflash"]
        if len(starts) != 1:
            raise ValueError(
                "DFlash command line must contain one canonical runtime tuple"
            )
        start = starts[0]
        runtime = arguments[start : start + 7]
        if (
            len(runtime) != 7
            or runtime[0] != b"--dflash"
            or runtime[1] != b"--draft-model"
            or not runtime[2]
            or runtime[3] != b"--dflash-gamma"
            or not runtime[4].isdigit()
            or int(runtime[4]) <= 0
            or runtime[5:] != [b"--dflash-quantization", b"nvfp4"]
        ):
            raise ValueError(
                "DFlash command line does not use the canonical runtime tuple"
            )
        if any(
            is_speculation_argument(argument)
            for index, argument in enumerate(arguments)
            if not start <= index < start + 7
        ):
            raise ValueError(
                "DFlash command line contains an extra speculation argument"
            )
        normalized = arguments[:start] + arguments[start + 7 :]
    else:
        raise ValueError(f"unsupported runtime mode {runtime_mode!r}")
    return sha256(b"\x00".join(normalized) + b"\x00")


def process_attestation(pid: int, proc_root: Path = Path("/proc")) -> dict[str, Any]:
    if pid <= 0:
        raise ValueError("process attestation PID must be positive")
    process = proc_root / str(pid)
    try:
        raw_cmdline = (process / "cmdline").read_bytes()
        raw_environment = (process / "environ").read_bytes()
        executable_fd = os.open(process / "exe", os.O_RDONLY | os.O_CLOEXEC)
    except OSError as error:
        raise ValueError(f"cannot inspect process {pid}: {error}") from error
    try:
        executable_sha256 = _sha256_fd(executable_fd)
    finally:
        os.close(executable_fd)
    if not raw_cmdline or not raw_cmdline.endswith(b"\x00"):
        raise ValueError("process command line is missing or malformed")
    nonce_prefix = f"{QUALIFICATION_NONCE_ENV}=".encode()
    nonce_entries = [
        entry
        for entry in raw_environment.split(b"\x00")
        if entry.startswith(nonce_prefix)
    ]
    if len(nonce_entries) != 1:
        raise ValueError("process must contain exactly one qualification nonce")
    try:
        nonce = nonce_entries[0][len(nonce_prefix) :].decode("ascii")
    except UnicodeDecodeError as error:
        raise ValueError("process qualification nonce is not ASCII") from error
    if not _is_hex(nonce, 64):
        raise ValueError(
            "process qualification nonce is not 64 lowercase hexadecimal characters"
        )
    environment = _atlas_environment(raw_environment)
    return {
        "binary_sha256": executable_sha256,
        "command_line_sha256": sha256(raw_cmdline),
        "effective_environment": environment,
        "effective_environment_sha256": sha256(canonical_bytes(environment)),
        "full_environment_sha256": _normalized_environment_sha256(raw_environment),
        "server_run_nonce": nonce,
    }


def wait_process_attestation(
    pid: int,
    expected_binary: Path,
    endpoint: str,
    timeout_seconds: float,
    proc_root: Path = Path("/proc"),
) -> dict[str, Any]:
    if not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 300:
        raise ValueError("process attestation timeout must be in (0, 300] seconds")
    try:
        binary_fd = os.open(expected_binary, os.O_RDONLY | os.O_CLOEXEC)
    except OSError as error:
        raise ValueError(f"cannot open expected server binary: {error}") from error
    try:
        expected_binary_sha256 = _sha256_fd(binary_fd)
    finally:
        os.close(binary_fd)
    port = endpoint_port(endpoint)
    deadline = time.monotonic() + timeout_seconds
    last_error = "server process has not exec'd and bound the endpoint"
    while time.monotonic() < deadline:
        try:
            attestation = process_attestation(pid, proc_root)
            owner = _listener_pid(proc_root, _listening_socket_inodes(proc_root, port))
            if owner == pid and attestation["binary_sha256"] == expected_binary_sha256:
                return attestation
            last_error = "PID, executable, or listener owner does not match"
        except ValueError as error:
            last_error = str(error)
        time.sleep(0.1)
    raise ValueError(f"server attestation readiness timed out: {last_error}")


def open_route_log(path: Path) -> int:
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(
            f"cannot open route log without following symlinks: {error}"
        ) from error
    try:
        opened = os.fstat(descriptor)
        named = path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_nlink != 1
            or opened.st_uid != os.getuid()
            or stat.S_IMODE(opened.st_mode) != 0o600
            or stat.S_ISLNK(named.st_mode)
            or (opened.st_dev, opened.st_ino) != (named.st_dev, named.st_ino)
        ):
            raise ValueError(
                "route log must be a caller-owned mode-0600 single-link regular file"
            )
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def _route_log_snapshot(path: Path, descriptor: int) -> tuple[bytes, os.stat_result]:
    for _ in range(3):
        before = os.fstat(descriptor)
        try:
            named_before = path.lstat()
        except OSError as error:
            raise ValueError(f"route log path disappeared: {error}") from error
        identity = (before.st_dev, before.st_ino)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_uid != os.getuid()
            or stat.S_IMODE(before.st_mode) != 0o600
            or stat.S_ISLNK(named_before.st_mode)
            or identity != (named_before.st_dev, named_before.st_ino)
        ):
            raise ValueError("route log path no longer names the retained regular file")
        chunks: list[bytes] = []
        offset = 0
        while offset < before.st_size:
            chunk = os.pread(
                descriptor, min(1024 * 1024, before.st_size - offset), offset
            )
            if not chunk:
                break
            chunks.append(chunk)
            offset += len(chunk)
        after = os.fstat(descriptor)
        try:
            named_after = path.lstat()
        except OSError as error:
            raise ValueError(f"route log path disappeared: {error}") from error
        if identity != (after.st_dev, after.st_ino) or identity != (
            named_after.st_dev,
            named_after.st_ino,
        ):
            raise ValueError("route log identity changed during qualification")
        if (
            offset == before.st_size
            and before.st_size == after.st_size
            and before.st_mtime_ns == after.st_mtime_ns
        ):
            return b"".join(chunks), after
    raise ValueError("route log changed while it was being captured")


def bind_live_server(
    endpoint: str,
    route_log: Path,
    route_log_fd: int,
    provenance: dict[str, Any],
    proc_root: Path = Path("/proc"),
    expected_binding: dict[str, Any] | None = None,
    retained_executable_fd: int | None = None,
) -> dict[str, Any]:
    port = endpoint_port(endpoint)
    socket_inodes = _listening_socket_inodes(proc_root, port)
    socket_inode = int(next(iter(socket_inodes)))
    pid = _listener_pid(proc_root, socket_inodes)
    if expected_binding is not None and (
        pid != expected_binding["listener_pid"]
        or socket_inode != expected_binding["listener_socket_inode"]
    ):
        raise ValueError("endpoint listener PID or socket changed during qualification")
    process = proc_root / str(pid)
    try:
        start_before = _process_start_time_ticks((process / "stat").read_text())
        route_stat = os.fstat(route_log_fd)
        stdout_stat = (process / "fd/1").stat()
        stderr_stat = (process / "fd/2").stat()
    except OSError as error:
        raise ValueError(
            f"cannot bind listener output to route log: {error}"
        ) from error
    route_identity = (route_stat.st_dev, route_stat.st_ino)
    if (stdout_stat.st_dev, stdout_stat.st_ino) != route_identity or (
        stderr_stat.st_dev,
        stderr_stat.st_ino,
    ) != route_identity:
        raise ValueError(
            "listener stdout and stderr must both be directly redirected to the route log"
        )
    writers = _writable_route_log_descriptors(proc_root, route_identity)
    if writers != {(pid, 1), (pid, 2)}:
        raise ValueError(
            "listener stdout and stderr must be the route log's only writable descriptors; "
            f"found {sorted(writers)}"
        )
    try:
        raw_cmdline = (process / "cmdline").read_bytes()
        raw_environment = (process / "environ").read_bytes()
        executable_path_stat = (process / "exe").stat()
        executable_fd = (
            retained_executable_fd
            if retained_executable_fd is not None
            else os.open(process / "exe", os.O_RDONLY | os.O_CLOEXEC)
        )
    except OSError as error:
        raise ValueError(
            f"cannot inspect endpoint listener process: {error}"
        ) from error
    try:
        executable_stat = os.fstat(executable_fd)
        if (executable_stat.st_dev, executable_stat.st_ino) != (
            executable_path_stat.st_dev,
            executable_path_stat.st_ino,
        ):
            raise ValueError(
                "listener executable path no longer matches retained executable"
            )
        if expected_binding is None:
            executable_sha256 = _sha256_fd(executable_fd)
        else:
            executable_sha256 = expected_binding["executable_sha256"]
            if (executable_stat.st_dev, executable_stat.st_ino) != (
                expected_binding["executable_device"],
                expected_binding["executable_inode"],
            ):
                raise ValueError("listener executable changed during qualification")
    finally:
        if retained_executable_fd is None:
            os.close(executable_fd)
    route_bytes, stable_route_stat = _route_log_snapshot(route_log, route_log_fd)
    try:
        start_after = _process_start_time_ticks((process / "stat").read_text())
    except OSError as error:
        raise ValueError(
            f"listener exited during process inspection: {error}"
        ) from error
    if start_before != start_after:
        raise ValueError("listener process identity changed during inspection")
    if not raw_cmdline or not raw_cmdline.endswith(b"\x00"):
        raise ValueError("listener command line is missing or malformed")
    command_line_sha256 = sha256(raw_cmdline)
    if command_line_sha256 != provenance["command_line_sha256"]:
        raise ValueError("listener command line does not match provenance")
    if executable_sha256 != provenance["binary_sha256"]:
        raise ValueError("listener executable does not match provenance binary_sha256")
    environment = _atlas_environment(raw_environment)
    if environment != provenance["effective_environment"]:
        raise ValueError(
            "listener ATLAS environment does not exactly match provenance effective_environment"
        )
    nonce = provenance["server_run_nonce"]
    full_environment_sha256 = _normalized_environment_sha256(raw_environment)
    if full_environment_sha256 != provenance["full_environment_sha256"]:
        raise ValueError(
            "listener full environment does not match provenance full_environment_sha256"
        )
    base_environment_sha256 = _normalized_environment_sha256(
        raw_environment, frozenset(provenance["same_mode_environment_delta"])
    )
    target_environment_sha256 = _normalized_environment_sha256(
        raw_environment, frozenset({"DRAFT", "RUNTIME_MODE"})
    )
    target_command_line_sha256 = _normalized_target_command_line_sha256(
        raw_cmdline, provenance["runtime_mode"]
    )
    raw_entries = raw_environment.split(b"\x00")
    nonce_entries = [
        entry
        for entry in raw_entries
        if entry.startswith(f"{QUALIFICATION_NONCE_ENV}=".encode())
    ]
    if nonce_entries != [f"{QUALIFICATION_NONCE_ENV}={nonce}".encode()]:
        raise ValueError("listener qualification nonce does not match provenance")
    binding_line = f"{QUALIFICATION_LOG_PREFIX} pid={pid} nonce={nonce}".encode()
    route_lines = route_bytes.splitlines()
    if route_lines.count(binding_line) != 1:
        raise ValueError(
            "route log must contain exactly one matching qualification PID/nonce line"
        )
    binding_index = route_lines.index(binding_line)
    if any(is_route_engagement_line(line) for line in route_lines[:binding_index]):
        raise ValueError(
            "qualification PID/nonce line must precede every engagement marker"
        )
    return {
        "endpoint_port": port,
        "listener_socket_inode": socket_inode,
        "listener_pid": pid,
        "process_start_time_ticks": start_before,
        "executable_sha256": executable_sha256,
        "executable_device": executable_stat.st_dev,
        "executable_inode": executable_stat.st_ino,
        "cmdline_sha256": command_line_sha256,
        "target_command_line_sha256": target_command_line_sha256,
        "cmdline_bytes": len(raw_cmdline),
        "effective_environment_sha256": sha256(canonical_bytes(environment)),
        "full_environment_sha256": full_environment_sha256,
        "base_environment_sha256": base_environment_sha256,
        "target_environment_sha256": target_environment_sha256,
        "route_log_device": stable_route_stat.st_dev,
        "route_log_inode": stable_route_stat.st_ino,
        "server_run_nonce": nonce,
    }


def validate_process_binding_record(
    value: object, provenance: dict[str, Any]
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("reference process_binding must be an object")
    positive_integer_keys = (
        "endpoint_port",
        "listener_socket_inode",
        "listener_pid",
        "process_start_time_ticks",
        "executable_inode",
        "cmdline_bytes",
        "route_log_inode",
    )
    for key in positive_integer_keys:
        item = value.get(key)
        if type(item) is not int or item <= 0:
            raise ValueError(f"reference process_binding {key} must be positive")
    nonnegative_integer_keys = ("executable_device", "route_log_device")
    for key in nonnegative_integer_keys:
        item = value.get(key)
        if type(item) is not int or item < 0:
            raise ValueError(f"reference process_binding {key} must be non-negative")
    hash_keys = (
        "executable_sha256",
        "cmdline_sha256",
        "effective_environment_sha256",
        "full_environment_sha256",
        "base_environment_sha256",
        "target_environment_sha256",
        "target_command_line_sha256",
    )
    for key in hash_keys:
        item = value.get(key)
        if not isinstance(item, str) or not _is_hex(item, 64):
            raise ValueError(f"reference process_binding {key} is malformed")
    if value["executable_sha256"] != provenance["binary_sha256"]:
        raise ValueError("reference process binding uses a different executable")
    if (
        value["effective_environment_sha256"]
        != provenance["effective_environment_sha256"]
    ):
        raise ValueError("reference process binding uses a different ATLAS environment")
    if value["full_environment_sha256"] != provenance["full_environment_sha256"]:
        raise ValueError("reference process binding uses a different full environment")
    if value["cmdline_sha256"] != provenance["command_line_sha256"]:
        raise ValueError("reference process binding uses a different command line")
    if value.get("server_run_nonce") != provenance["server_run_nonce"]:
        raise ValueError("reference process binding uses a different run nonce")
    return {
        key: value[key]
        for key in (
            *positive_integer_keys,
            *nonnegative_integer_keys,
            *hash_keys,
            "server_run_nonce",
        )
    }


def capture_route_evidence(
    path: Path,
    required_markers: dict[str, int],
    process_binding: dict[str, Any],
    route_log_fd: int | None = None,
) -> dict[str, Any]:
    owned_descriptor = route_log_fd is None
    if route_log_fd is None:
        route_log_fd = open_route_log(path)
    try:
        raw, _ = _route_log_snapshot(path, route_log_fd)
    finally:
        if owned_descriptor:
            os.close(route_log_fd)
    observed = {marker: 0 for marker in required_markers}
    engaged_lines: list[bytes] = []
    encoded_markers = {marker: marker.encode("utf-8") for marker in required_markers}
    for line in raw.splitlines():
        if not is_route_engagement_line(line):
            continue
        matches = [
            marker for marker, encoded in encoded_markers.items() if encoded in line
        ]
        if len(matches) != 1:
            raise ValueError(
                "route log contains an unexpected or ambiguous engagement line; use a fresh log and declare every route marker"
            )
        marker = matches[0]
        if line.count(encoded_markers[marker]) != 1:
            raise ValueError(f"route log line repeats marker {marker!r}")
        observed[marker] += 1
        engaged_lines.append(line)
    for marker, expected in required_markers.items():
        actual = observed[marker]
        if actual != expected:
            raise ValueError(
                f"route marker {marker!r} expected {expected} occurrence(s), found {actual}"
            )
    engaged_bytes = b"\n".join(engaged_lines) + (b"\n" if engaged_lines else b"")
    return {
        "route_log_sha256": sha256(raw),
        "route_log_bytes": len(raw),
        "engaged_lines_sha256": sha256(engaged_bytes),
        "engaged_line_count": len(engaged_lines),
        "observed_route_markers": observed,
        "process_binding": process_binding,
    }


def validate_route_evidence_record(
    value: object,
    required_markers: dict[str, int],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("reference route_evidence must be an object")
    for key in ("route_log_sha256", "engaged_lines_sha256"):
        item = value.get(key)
        if not isinstance(item, str) or not _is_hex(item, 64):
            raise ValueError(f"reference route_evidence {key} is malformed")
    route_log_bytes = value.get("route_log_bytes")
    line_count = value.get("engaged_line_count")
    if type(route_log_bytes) is not int or route_log_bytes <= 0:
        raise ValueError("reference route_evidence route_log_bytes must be positive")
    if type(line_count) is not int or line_count != sum(required_markers.values()):
        raise ValueError("reference route_evidence engaged_line_count is inconsistent")
    observed = validate_route_marker_spec(value.get("observed_route_markers"))
    if observed != required_markers:
        raise ValueError(
            "reference route evidence does not match its provenance markers"
        )
    process_binding = validate_process_binding_record(
        value.get("process_binding"), provenance
    )
    return {
        "route_log_sha256": value["route_log_sha256"],
        "route_log_bytes": route_log_bytes,
        "engaged_lines_sha256": value["engaged_lines_sha256"],
        "engaged_line_count": line_count,
        "observed_route_markers": observed,
        "process_binding": process_binding,
    }


def target_ttft_ms(prompt_tokens: int, target_tps: float = PREFILL_TARGET_TPS) -> float:
    if prompt_tokens <= 0 or not target_tps > 0:
        raise ValueError("prompt tokens and target throughput must be positive")
    return prompt_tokens * 1_000.0 / target_tps


def effective_prefill_tps(prompt_tokens: int, server_ttft_ms: float) -> float:
    if prompt_tokens <= 0 or not server_ttft_ms > 0:
        raise ValueError("prompt tokens and server TTFT must be positive")
    return prompt_tokens * 1_000.0 / server_ttft_ms


def validate_workload(
    inputs: list[int], repetitions: int, decode_max_tokens: int
) -> None:
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if not inputs or any(tokens <= 0 for tokens in inputs):
        raise ValueError("input token bins must be positive")
    if len(set(inputs)) != len(inputs):
        raise ValueError("input token bins must be unique")
    if decode_max_tokens <= 0:
        raise ValueError("decode max tokens must be positive")


def validate_qualification_workload(
    inputs: list[int],
    repetitions: int,
    decode_max_tokens: int,
    continuation_tokens: int,
    prefill_target_tps: float,
    decode_target_tps: float,
) -> None:
    validate_workload(inputs, repetitions, decode_max_tokens)
    if inputs != list(DEFAULT_INPUTS):
        raise ValueError(
            "live qualification requires ordered input bins 2048 8192 32768"
        )
    if repetitions != 5:
        raise ValueError("live qualification requires exactly five repetitions")
    if decode_max_tokens != 400:
        raise ValueError("live qualification requires the 400-token decode probe")
    if continuation_tokens < DEFAULT_PREFILL_CONTINUATION_TOKENS:
        raise ValueError("live qualification requires at least 32 continuation tokens")
    if not math.isfinite(prefill_target_tps) or prefill_target_tps < PREFILL_TARGET_TPS:
        raise ValueError(
            "live qualification target must be at least 2000 prefill tok/s"
        )
    if not math.isfinite(decode_target_tps) or decode_target_tps < DECODE_TARGET_TPS:
        raise ValueError("live qualification target must be at least 85 decode tok/s")


def post_json(url: str, body: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=1_200) as response:
        return json.load(response)


def get_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def build_prefill_prompt(target_tokens: int, run: int, continuation_tokens: int) -> str:
    # The unique prefix prevents a prior run from satisfying this request via
    # prefix cache. Repeated single-word text keeps tokenization predictable;
    # the server-reported actual token count remains authoritative.
    return (
        f"Unique qualification prefix {target_tokens}-{run}.\n"
        + "measurement " * target_tokens
        + f"\nOutput the word atlas separated by spaces for at least {continuation_tokens} tokens."
    )


def measure_prefill(
    endpoint: str,
    model: str,
    target_tokens: int,
    run: int,
    target_tps: float,
    continuation_tokens: int,
) -> dict[str, Any]:
    body = {
        "model": model,
        "prompt": build_prefill_prompt(target_tokens, run, continuation_tokens),
        "temperature": 0.0,
        "max_tokens": continuation_tokens,
        "logit_bias": {
            str(token_id): -100.0
            for token_id in PREFILL_SUPPRESSED_STOP_TOKEN_IDS
        },
    }
    response = post_json(endpoint, body)
    choice = response["choices"][0]
    continuation = {
        "text": choice.get("text") or "",
        "finish_reason": choice.get("finish_reason"),
    }
    usage = response.get("usage", {})
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    ttft_ms = float(usage.get("time_to_first_token_ms") or 0.0)
    prompt_details = usage.get("prompt_tokens_details")
    cache_accounting_present = (
        isinstance(prompt_details, dict) and "cached_tokens" in prompt_details
    )
    cached_tokens = (
        int(prompt_details.get("cached_tokens") or 0)
        if cache_accounting_present
        else -1
    )
    completion_tokens = int(usage.get("completion_tokens") or 0)
    continuation_complete = completion_tokens >= continuation_tokens
    tps = effective_prefill_tps(prompt_tokens, ttft_ms)
    return {
        "index": run,
        "target_prompt_tokens": target_tokens,
        "prompt_tokens": prompt_tokens,
        "server_ttft_ms": ttft_ms,
        "effective_prefill_tokens_per_second": tps,
        "target_ttft_ms": target_ttft_ms(prompt_tokens, target_tps),
        "cached_prompt_tokens": cached_tokens,
        "cache_accounting_present": cache_accounting_present,
        "requested_continuation_tokens": continuation_tokens,
        "completion_tokens": completion_tokens,
        "finish_reason": continuation["finish_reason"],
        "continuation_complete": continuation_complete,
        "stable_output_sha256": sha256(canonical_bytes(continuation)),
        "performance_passes_target": (
            tps >= target_tps
            and cache_accounting_present
            and cached_tokens == 0
            and continuation_complete
            and continuation["finish_reason"] == DECODE_COMPLETE_FINISH_REASON
        ),
        "request_body_sha256": sha256(canonical_bytes(body)),
    }


def bind_semantic_oracle(
    prefill_rows: list[dict[str, Any]],
    decode_rows: list[dict[str, Any]],
    oracle_value: object,
    model: str,
    provenance: dict[str, Any],
) -> None:
    oracle = validate_semantic_oracle(oracle_value)
    if oracle["model"] != model:
        raise ValueError("semantic oracle model differs from the live server model")
    for key in ("model_index_sha256", "tokenizer_sha256"):
        if oracle[key] != provenance[key]:
            raise ValueError(f"semantic oracle {key} differs from live provenance")
    if len(prefill_rows) != 15 or len(decode_rows) != 5:
        raise ValueError("live workload does not contain exactly 15 prefill and five decode rows")

    oracle_prefill = {
        (row["target_prompt_tokens"], row["index"]): row
        for row in oracle["prefill_rows"]
    }
    prefill_fields = (
        "request_body_sha256",
        "prompt_tokens",
        "stable_output_sha256",
        "requested_continuation_tokens",
        "completion_tokens",
        "finish_reason",
    )
    seen_prefill: set[tuple[int, int]] = set()
    for row in prefill_rows:
        key = (row.get("target_prompt_tokens"), row.get("index"))
        if key in seen_prefill or key not in oracle_prefill:
            raise ValueError(f"live prefill row {key!r} is repeated or absent from oracle")
        seen_prefill.add(key)
        golden = oracle_prefill[key]
        for field in prefill_fields:
            if row.get(field) != golden[field]:
                raise ValueError(
                    f"live prefill row {key!r} differs from semantic oracle at {field}"
                )
        row["semantic_oracle_output_sha256"] = golden["stable_output_sha256"]
        row["matches_semantic_oracle"] = True
    if seen_prefill != set(oracle_prefill):
        raise ValueError("live prefill rows do not cover the semantic oracle")

    oracle_decode = {row["index"]: row for row in oracle["decode_rows"]}
    decode_fields = (
        "request_body_sha256",
        "prompt_tokens",
        "stable_output_sha256",
        "requested_completion_tokens",
        "completion_tokens",
        "finish_reason",
    )
    seen_decode: set[int] = set()
    for row in decode_rows:
        index = row.get("index")
        if type(index) is not int or index in seen_decode or index not in oracle_decode:
            raise ValueError(
                f"live decode row {index!r} is repeated or absent from oracle"
            )
        seen_decode.add(index)
        golden = oracle_decode[index]
        for field in decode_fields:
            if row.get(field) != golden[field]:
                raise ValueError(
                    f"live decode row {index} differs from semantic oracle at {field}"
                )
        row["semantic_oracle_output_sha256"] = golden["stable_output_sha256"]
        row["matches_semantic_oracle"] = True
    if seen_decode != set(oracle_decode):
        raise ValueError("live decode rows do not cover the semantic oracle")


def write_prefill_reference(
    path: Path,
    model: str,
    comparison_kind: str,
    provenance: dict[str, Any],
    provenance_manifest_sha256: str,
    route_evidence: dict[str, Any],
    semantic_oracle_sha256: str,
    rows: list[dict[str, Any]],
    decode_rows: list[dict[str, Any]],
) -> None:
    reference = {
        "schema": REFERENCE_SCHEMA,
        "comparison_kind": comparison_kind,
        "model": model,
        "provenance": provenance,
        "provenance_manifest_sha256": provenance_manifest_sha256,
        "route_evidence": route_evidence,
        "semantic_oracle_sha256": semantic_oracle_sha256,
        "rows": [
            {
                "target_prompt_tokens": row["target_prompt_tokens"],
                "index": row["index"],
                "prompt_tokens": row["prompt_tokens"],
                "server_ttft_ms": row["server_ttft_ms"],
                "effective_prefill_tokens_per_second": row[
                    "effective_prefill_tokens_per_second"
                ],
                "request_body_sha256": row["request_body_sha256"],
                "stable_output_sha256": row["stable_output_sha256"],
                "cached_prompt_tokens": row["cached_prompt_tokens"],
                "cache_accounting_present": row["cache_accounting_present"],
                "requested_continuation_tokens": row["requested_continuation_tokens"],
                "completion_tokens": row["completion_tokens"],
                "finish_reason": row["finish_reason"],
            }
            for row in rows
        ],
        "decode_rows": [
            {
                "index": row["index"],
                "request_body_sha256": row["request_body_sha256"],
                "prompt_tokens": row["prompt_tokens"],
                "stable_output_sha256": row["stable_output_sha256"],
                "requested_completion_tokens": row["requested_completion_tokens"],
                "completion_tokens": row["completion_tokens"],
                "finish_reason": row["finish_reason"],
            }
            for row in decode_rows
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_bytes(reference) + b"\n")


def _effective_environment_delta(
    control: dict[str, str], candidate: dict[str, str]
) -> dict[str, dict[str, str]]:
    if set(control) != set(candidate):
        raise ValueError(
            "same-mode control and candidate effective environments must contain identical keys"
        )
    return {
        key: {"control": control[key], "candidate": candidate[key]}
        for key in sorted(control)
        if control[key] != candidate[key]
    }


def nearest_rank_p90(values: list[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("p90 inputs must be positive finite measurements")
    ordered = sorted(values)
    return ordered[math.ceil(0.9 * len(ordered)) - 1]


def summarize_relative_prefill(
    rows: list[dict[str, Any]], required_targets: tuple[int, ...] = DEFAULT_INPUTS
) -> dict[str, Any]:
    summaries: dict[str, dict[str, Any]] = {}
    for target in required_targets:
        group = [row for row in rows if row.get("target_prompt_tokens") == target]
        if not group:
            raise ValueError(f"same-mode relative prefill requires target bin {target}")
        control = [float(row["reference_server_ttft_ms"]) for row in group]
        candidate = [float(row["server_ttft_ms"]) for row in group]
        control_median = statistics.median(control)
        candidate_median = statistics.median(candidate)
        control_p90 = nearest_rank_p90(control)
        candidate_p90 = nearest_rank_p90(candidate)
        median_wins = candidate_median < control_median
        p90_wins = candidate_p90 < control_p90
        summaries[str(target)] = {
            "repetitions": len(group),
            "control_median_ttft_ms": control_median,
            "candidate_median_ttft_ms": candidate_median,
            "median_speedup": control_median / candidate_median,
            "control_p90_ttft_ms": control_p90,
            "candidate_p90_ttft_ms": candidate_p90,
            "p90_speedup": control_p90 / candidate_p90,
            "median_wins": median_wins,
            "p90_wins": p90_wins,
            "passes": median_wins and p90_wins,
        }
    return {
        "required_targets": list(required_targets),
        "targets": summaries,
        "passes": all(summary["passes"] for summary in summaries.values()),
    }


def bind_prefill_reference(
    rows: list[dict[str, Any]],
    reference: object,
    candidate_provenance: dict[str, Any],
    candidate_route_evidence: dict[str, Any],
    semantic_oracle_sha256: str,
    comparison_kind: str = TARGET_VS_DFLASH,
) -> None:
    if not isinstance(reference, dict):
        raise ValueError("prefill reference must be a JSON object")
    schema = reference.get("schema")
    if schema != REFERENCE_SCHEMA:
        raise ValueError(f"prefill reference requires {REFERENCE_SCHEMA}")
    if (
        not isinstance(semantic_oracle_sha256, str)
        or not _is_hex(semantic_oracle_sha256, 64)
        or reference.get("semantic_oracle_sha256") != semantic_oracle_sha256
    ):
        raise ValueError("prefill reference uses a different semantic oracle")
    reference_comparison = reference.get("comparison_kind")
    if reference_comparison != comparison_kind:
        raise ValueError(
            f"prefill reference comparison kind {reference_comparison!r} does not match {comparison_kind!r}"
        )
    if comparison_kind == TARGET_VS_DFLASH:
        control_mode = "no-spec"
        if candidate_provenance.get("runtime_mode") != "dflash-v3":
            raise ValueError(
                "target-vs-dflash candidate runtime mode must be 'dflash-v3'"
            )
    elif comparison_kind == SAME_MODE:
        control_mode = candidate_provenance.get("runtime_mode")
        if control_mode not in RUNTIME_MODES:
            raise ValueError("same-mode candidate has an unsupported runtime mode")
    else:
        raise ValueError(f"unsupported comparison kind {comparison_kind!r}")
    control_provenance = validate_provenance(reference.get("provenance"), control_mode)
    control_route_evidence = validate_route_evidence_record(
        reference.get("route_evidence"),
        control_provenance["required_route_markers"],
        control_provenance,
    )
    candidate_route_evidence = validate_route_evidence_record(
        candidate_route_evidence,
        candidate_provenance["required_route_markers"],
        candidate_provenance,
    )
    for key in COMMON_PROVENANCE_KEYS[:-1]:
        if control_provenance[key] != candidate_provenance[key]:
            raise ValueError(
                f"prefill control and candidate provenance differ at {key}"
            )
    if comparison_kind == SAME_MODE and control_mode == "dflash-v3":
        if (
            control_provenance["draft_index_sha256"]
            != candidate_provenance["draft_index_sha256"]
        ):
            raise ValueError(
                "same-mode DFlash control and candidate use different draft indexes"
            )
    control_delta = control_provenance["same_mode_environment_delta"]
    candidate_delta = candidate_provenance["same_mode_environment_delta"]
    if comparison_kind == SAME_MODE:
        if not control_delta or control_delta != candidate_delta:
            raise ValueError(
                "same-mode manifests must declare the same non-empty environment delta"
            )
        actual_delta = _effective_environment_delta(
            control_provenance["effective_environment"],
            candidate_provenance["effective_environment"],
        )
        if actual_delta != control_delta:
            raise ValueError(
                "same-mode effective environment changes do not exactly match the declared delta"
            )
        control_binding = control_route_evidence["process_binding"]
        candidate_binding = candidate_route_evidence["process_binding"]
        if control_binding["cmdline_sha256"] != candidate_binding["cmdline_sha256"]:
            raise ValueError("same-mode control and candidate command lines differ")
        if control_binding["endpoint_port"] != candidate_binding["endpoint_port"]:
            raise ValueError("same-mode control and candidate endpoint ports differ")
        if (
            control_binding["base_environment_sha256"]
            != candidate_binding["base_environment_sha256"]
        ):
            raise ValueError(
                "same-mode control and candidate environments differ outside the qualified flag"
            )
        if control_binding["server_run_nonce"] == candidate_binding["server_run_nonce"]:
            raise ValueError("control and candidate must use different run nonces")
    elif control_delta or candidate_delta:
        raise ValueError(
            "target-vs-dflash manifests must use an empty same_mode_environment_delta"
        )
    elif (
        control_provenance["effective_environment"]
        != candidate_provenance["effective_environment"]
    ):
        raise ValueError(
            "target-vs-dflash control and candidate must use an identical ATLAS environment"
        )
    elif (
        control_route_evidence["process_binding"]["target_command_line_sha256"]
        != candidate_route_evidence["process_binding"]["target_command_line_sha256"]
    ):
        raise ValueError(
            "target-vs-dflash command lines differ outside the canonical DFlash tuple"
        )
    elif (
        control_route_evidence["process_binding"]["target_environment_sha256"]
        != candidate_route_evidence["process_binding"]["target_environment_sha256"]
    ):
        raise ValueError(
            "target-vs-dflash environments differ outside runtime mode, drafter, and nonce"
        )
    elif (
        control_route_evidence["process_binding"]["endpoint_port"]
        != candidate_route_evidence["process_binding"]["endpoint_port"]
    ):
        raise ValueError("target-vs-dflash endpoint ports differ")
    elif (
        control_route_evidence["process_binding"]["server_run_nonce"]
        == candidate_route_evidence["process_binding"]["server_run_nonce"]
    ):
        raise ValueError("control and candidate must use different run nonces")
    reference_rows = reference.get("rows")
    if not isinstance(reference_rows, list):
        raise ValueError("prefill reference rows must be a list")
    if len(reference_rows) != len(rows):
        raise ValueError("prefill candidate and reference repetition counts differ")
    by_request: dict[str, dict[str, Any]] = {}
    for row in reference_rows:
        if not isinstance(row, dict) or not isinstance(
            row.get("request_body_sha256"), str
        ):
            raise ValueError("prefill reference contains a malformed row")
        request_hash = row["request_body_sha256"]
        output_hash = row.get("stable_output_sha256")
        ttft = row.get("server_ttft_ms")
        if (
            not _is_hex(request_hash, 64)
            or not isinstance(output_hash, str)
            or not _is_hex(output_hash, 64)
        ):
            raise ValueError("prefill reference contains a malformed hash")
        if not isinstance(ttft, (int, float)) or not math.isfinite(ttft) or ttft <= 0:
            raise ValueError("prefill reference contains an invalid TTFT")
        if (
            row.get("cache_accounting_present") is not True
            or row.get("cached_prompt_tokens") != 0
        ):
            raise ValueError(
                "prefill reference is not an explicit zero-cache measurement"
            )
        requested = row.get("requested_continuation_tokens")
        completed = row.get("completion_tokens")
        if (
            type(requested) is not int
            or type(completed) is not int
            or requested < 32
            or completed < requested
            or row.get("finish_reason") != DECODE_COMPLETE_FINISH_REASON
        ):
            raise ValueError("prefill reference has an incomplete continuation")
        if request_hash in by_request:
            raise ValueError(f"prefill reference repeats request {request_hash}")
        by_request[request_hash] = row
    seen_candidate_requests: set[str] = set()
    for row in rows:
        request_hash = row["request_body_sha256"]
        if request_hash in seen_candidate_requests:
            raise ValueError(f"prefill candidate repeats request {request_hash}")
        seen_candidate_requests.add(request_hash)
        control = by_request.get(request_hash)
        if control is None:
            raise ValueError(
                "prefill reference is missing target/run request "
                f"{row['target_prompt_tokens']}/{row['index']}"
            )
        if control.get("target_prompt_tokens") != row.get(
            "target_prompt_tokens"
        ) or control.get("index") != row.get("index"):
            raise ValueError("prefill reference target/run metadata differs")
        if control.get("prompt_tokens") != row.get("prompt_tokens"):
            raise ValueError(
                "prefill control and candidate actual prompt counts differ"
            )
        matches = control.get("stable_output_sha256") == row["stable_output_sha256"]
        completion_matches = control.get("completion_tokens") == row.get(
            "completion_tokens"
        )
        requested_matches = control.get("requested_continuation_tokens") == row.get(
            "requested_continuation_tokens"
        )
        finish_matches = control.get("finish_reason") == row.get("finish_reason")
        row["reference_output_sha256"] = control.get("stable_output_sha256")
        row["reference_server_ttft_ms"] = float(control["server_ttft_ms"])
        row["ttft_delta_ms"] = row["server_ttft_ms"] - float(control["server_ttft_ms"])
        row["ttft_speedup"] = float(control["server_ttft_ms"]) / row["server_ttft_ms"]
        row["matches_reference"] = matches
        row["completion_matches_reference"] = completion_matches
        row["requested_continuation_matches_reference"] = requested_matches
        row["finish_matches_reference"] = finish_matches
        if comparison_kind == TARGET_VS_DFLASH:
            row["no_spec_reference_sha256"] = control.get("stable_output_sha256")
            row["matches_no_spec_reference"] = matches
        row["passes_target"] = (
            row["performance_passes_target"]
            and matches
            and completion_matches
            and requested_matches
            and finish_matches
        )


def measure_decode(
    endpoint: str,
    model: str,
    run: int,
    max_tokens: int,
    target_tps: float,
) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": DECODE_PROMPT}],
        "temperature": 0.0,
        "reasoning_effort": "none",
        "max_tokens": max_tokens,
    }
    started = time.monotonic()
    response = post_json(endpoint, body)
    wall_seconds = time.monotonic() - started
    choice = response["choices"][0]
    message = choice["message"]
    usage = response.get("usage", {})
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    content = message.get("content") or ""
    reasoning = message.get("reasoning_content") or ""
    completion_tokens = int(usage.get("completion_tokens") or 0)
    finish_reason = choice.get("finish_reason")
    completion_complete = completion_tokens == max_tokens
    finish_semantics_complete = finish_reason == DECODE_COMPLETE_FINISH_REASON
    server_tps = usage.get("response_token/s")
    rate = (
        float(server_tps)
        if server_tps is not None
        else completion_tokens / wall_seconds
    )
    stable_output = {
        "content": content,
        "reasoning_content": reasoning,
        "finish_reason": finish_reason,
    }
    return {
        "index": run,
        "prompt_tokens": prompt_tokens,
        "requested_completion_tokens": max_tokens,
        "completion_tokens": completion_tokens,
        "completion_complete": completion_complete,
        "finish_reason": finish_reason,
        "finish_semantics_complete": finish_semantics_complete,
        "server_decode_tokens_per_second": rate,
        "rate_source": "server" if server_tps is not None else "client_wall_clock",
        "wall_seconds": wall_seconds,
        "performance_passes_target": (
            rate >= target_tps and completion_complete and finish_semantics_complete
        ),
        "stable_output_sha256": sha256(canonical_bytes(stable_output)),
        "request_body_sha256": sha256(canonical_bytes(body)),
    }


def decode_reference_is_valid(rows: list[dict[str, Any]]) -> bool:
    if not rows:
        return False
    return (
        all(
            row.get("completion_complete") is True
            and row.get("finish_semantics_complete") is True
            for row in rows
        )
        and len({row.get("stable_output_sha256") for row in rows}) == 1
        and len({row.get("finish_reason") for row in rows}) == 1
    )


def bind_decode_reference(rows: list[dict[str, Any]], reference: object) -> None:
    if not isinstance(reference, dict):
        raise ValueError("decode reference must be a JSON object")
    if reference.get("schema") != REFERENCE_SCHEMA:
        raise ValueError(f"decode reference requires {REFERENCE_SCHEMA}")
    reference_rows = reference.get("decode_rows")
    if not isinstance(reference_rows, list) or not reference_rows:
        raise ValueError("decode reference rows must be a non-empty list")

    by_index: dict[int, dict[str, Any]] = {}
    reference_hashes: set[str] = set()
    reference_finishes: set[str] = set()
    for control in reference_rows:
        if not isinstance(control, dict) or type(control.get("index")) is not int:
            raise ValueError("decode reference contains a malformed row")
        index = control["index"]
        if index in by_index:
            raise ValueError(f"decode reference repeats index {index}")
        request_hash = control.get("request_body_sha256")
        output_hash = control.get("stable_output_sha256")
        prompt_tokens = control.get("prompt_tokens")
        requested = control.get("requested_completion_tokens")
        completed = control.get("completion_tokens")
        finish_reason = control.get("finish_reason")
        if not isinstance(request_hash, str) or not _is_hex(request_hash, 64):
            raise ValueError("decode reference request hash is malformed")
        if not isinstance(output_hash, str) or not _is_hex(output_hash, 64):
            raise ValueError("decode reference output hash is malformed")
        if (
            type(prompt_tokens) is not int
            or prompt_tokens <= 0
            or type(requested) is not int
            or type(completed) is not int
            or requested <= 0
            or completed != requested
        ):
            raise ValueError(
                "decode reference does not contain a complete requested response"
            )
        if finish_reason != DECODE_COMPLETE_FINISH_REASON:
            raise ValueError(
                "decode reference does not have max-token finish semantics"
            )
        by_index[index] = control
        reference_hashes.add(output_hash)
        reference_finishes.add(finish_reason)
    if len(reference_hashes) != 1 or len(reference_finishes) != 1:
        raise ValueError("decode reference is not deterministic")
    if len(rows) != len(reference_rows):
        raise ValueError("decode candidate and reference repetition counts differ")

    for row in rows:
        control = by_index.get(row.get("index"))
        if control is None:
            raise ValueError(f"decode reference is missing run {row.get('index')}")
        request_matches = control["request_body_sha256"] == row.get(
            "request_body_sha256"
        )
        requested_matches = control["requested_completion_tokens"] == row.get(
            "requested_completion_tokens"
        )
        prompt_matches = control["prompt_tokens"] == row.get("prompt_tokens")
        output_matches = control["stable_output_sha256"] == row.get(
            "stable_output_sha256"
        )
        finish_matches = control["finish_reason"] == row.get("finish_reason")
        completion_matches = control["completion_tokens"] == row.get(
            "completion_tokens"
        )
        row["reference_output_sha256"] = control["stable_output_sha256"]
        row["request_matches_reference"] = request_matches
        row["prompt_tokens_match_reference"] = prompt_matches
        row["requested_completion_matches_reference"] = requested_matches
        row["matches_reference"] = output_matches
        row["finish_matches_reference"] = finish_matches
        row["completion_matches_reference"] = completion_matches
        row["passes_target"] = bool(
            row.get("performance_passes_target")
            and request_matches
            and prompt_matches
            and requested_matches
            and output_matches
            and finish_matches
            and completion_matches
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8896/v1/completions")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inputs", type=int, nargs="+", default=list(DEFAULT_INPUTS))
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--decode-max-tokens", type=int, default=400)
    parser.add_argument(
        "--prefill-continuation-tokens",
        type=int,
        default=DEFAULT_PREFILL_CONTINUATION_TOKENS,
    )
    parser.add_argument("--prefill-target-tps", type=float, default=PREFILL_TARGET_TPS)
    parser.add_argument("--decode-target-tps", type=float, default=DECODE_TARGET_TPS)
    parser.add_argument("--provenance-manifest", type=Path)
    parser.add_argument(
        "--route-log",
        type=Path,
        help="fresh server stdout/stderr log containing the declared ENGAGED markers",
    )
    parser.add_argument(
        "--semantic-oracle",
        type=Path,
        help="read-only independent semantic oracle for every qualification request",
    )
    parser.add_argument(
        "--expected-semantic-oracle-sha256",
        help="exact lowercase SHA-256 of the independently approved semantic oracle",
    )
    parser.add_argument(
        "--comparison-kind",
        choices=(TARGET_VS_DFLASH, SAME_MODE),
        default=TARGET_VS_DFLASH,
    )
    parser.add_argument("--runtime-mode", choices=RUNTIME_MODES)
    reference_group = parser.add_mutually_exclusive_group()
    reference_group.add_argument("--write-prefill-reference", type=Path)
    reference_group.add_argument("--prefill-reference", type=Path)
    parser.add_argument("--budget-only", action="store_true")
    parser.add_argument(
        "--print-process-attestation",
        type=int,
        metavar="PID",
        help="print non-secret manifest fields observed from a running server process",
    )
    parser.add_argument(
        "--expected-binary",
        type=Path,
        help="expected server binary used to close the launcher-exec attestation race",
    )
    parser.add_argument(
        "--attestation-timeout-seconds",
        type=float,
        default=120.0,
    )
    args = parser.parse_args()

    if args.print_process_attestation is not None:
        if args.budget_only:
            parser.error(
                "--print-process-attestation and --budget-only are mutually exclusive"
            )
        if args.expected_binary is None:
            parser.error("--print-process-attestation requires --expected-binary")
        try:
            attestation = wait_process_attestation(
                args.print_process_attestation,
                args.expected_binary,
                args.endpoint,
                args.attestation_timeout_seconds,
            )
        except ValueError as error:
            parser.error(str(error))
        print(json.dumps(attestation, sort_keys=True))
        return 0

    try:
        validate_workload(args.inputs, args.repetitions, args.decode_max_tokens)
    except ValueError as error:
        parser.error(str(error))

    if args.budget_only:
        rows = [
            {
                "prompt_tokens": tokens,
                "target_ttft_ms": target_ttft_ms(tokens, args.prefill_target_tps),
            }
            for tokens in args.inputs
        ]
        print(
            json.dumps({"prefill_target_tps": args.prefill_target_tps, "budgets": rows})
        )
        return 0
    try:
        validate_qualification_workload(
            args.inputs,
            args.repetitions,
            args.decode_max_tokens,
            args.prefill_continuation_tokens,
            args.prefill_target_tps,
            args.decode_target_tps,
        )
    except ValueError as error:
        parser.error(str(error))
    if args.output is None:
        if args.write_prefill_reference is None:
            parser.error(
                "--output is required unless --budget-only or --write-prefill-reference is used"
            )
    if args.provenance_manifest is None:
        parser.error("--provenance-manifest is required for every live qualification")
    if args.route_log is None:
        parser.error("--route-log is required for every live qualification")
    try:
        expected_mode = resolve_runtime_mode(
            args.comparison_kind,
            args.write_prefill_reference is not None,
            args.runtime_mode,
        )
    except ValueError as error:
        parser.error(str(error))
    try:
        provenance, provenance_manifest_sha256 = load_provenance(
            args.provenance_manifest, expected_mode
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        parser.error(f"invalid provenance manifest: {error}")
    if args.write_prefill_reference is None and args.prefill_reference is None:
        parser.error(
            "--prefill-reference is required unless --write-prefill-reference is used"
        )
    if args.semantic_oracle is None or args.expected_semantic_oracle_sha256 is None:
        parser.error(
            "--semantic-oracle and --expected-semantic-oracle-sha256 are required "
            "for every live qualification"
        )
    try:
        (
            semantic_oracle,
            semantic_oracle_sha256,
            semantic_oracle_fd,
            semantic_oracle_stat,
        ) = open_semantic_oracle(
            args.semantic_oracle, args.expected_semantic_oracle_sha256
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        parser.error(f"invalid semantic oracle: {error}")

    try:
        route_log_fd = open_route_log(args.route_log)
        process_binding_before = bind_live_server(
            args.endpoint, args.route_log, route_log_fd, provenance
        )
        route_log_prefix, _ = _route_log_snapshot(args.route_log, route_log_fd)
        executable_fd = os.open(
            Path("/proc") / str(process_binding_before["listener_pid"]) / "exe",
            os.O_RDONLY | os.O_CLOEXEC,
        )
        executable_stat = os.fstat(executable_fd)
        if (
            executable_stat.st_dev,
            executable_stat.st_ino,
            _sha256_fd(executable_fd),
        ) != (
            process_binding_before["executable_device"],
            process_binding_before["executable_inode"],
            process_binding_before["executable_sha256"],
        ):
            raise ValueError("cannot retain the attested server executable")
        pid_fd = os.pidfd_open(process_binding_before["listener_pid"])
    except (OSError, ValueError) as error:
        parser.error(f"cannot bind qualification endpoint to server process: {error}")

    def assert_server_binding(stage: str) -> dict[str, Any]:
        nonlocal route_log_prefix
        poller = select.poll()
        poller.register(pid_fd, select.POLLIN)
        if poller.poll(0):
            parser.error(f"qualification listener exited during {stage}")
        try:
            current = bind_live_server(
                args.endpoint,
                args.route_log,
                route_log_fd,
                provenance,
                expected_binding=process_binding_before,
                retained_executable_fd=executable_fd,
            )
        except (OSError, ValueError) as error:
            parser.error(f"qualification listener changed during {stage}: {error}")
        if current != process_binding_before:
            parser.error(f"qualification listener identity changed during {stage}")
        try:
            current_log, _ = _route_log_snapshot(args.route_log, route_log_fd)
        except (OSError, ValueError) as error:
            parser.error(f"qualification route log changed during {stage}: {error}")
        if not current_log.startswith(route_log_prefix):
            parser.error(
                f"qualification route log was truncated or rewritten during {stage}"
            )
        route_log_prefix = current_log
        return current

    models_url = args.endpoint.rsplit("/completions", 1)[0] + "/models"
    try:
        assert_semantic_oracle_unchanged(
            args.semantic_oracle,
            semantic_oracle_fd,
            semantic_oracle_sha256,
            semantic_oracle_stat,
        )
    except (OSError, ValueError) as error:
        parser.error(f"semantic oracle changed before measurement: {error}")
    assert_server_binding("model discovery start")
    model = get_json(models_url)["data"][0]["id"]
    assert_server_binding("model discovery end")
    prefill = []
    for target_tokens in args.inputs:
        for run in range(args.repetitions):
            assert_server_binding(f"prefill {target_tokens}/{run} start")
            row = measure_prefill(
                args.endpoint,
                model,
                target_tokens,
                run,
                args.prefill_target_tps,
                args.prefill_continuation_tokens,
            )
            assert_server_binding(f"prefill {target_tokens}/{run} end")
            prefill.append(row)
            print(json.dumps({"prefill": row}, sort_keys=True), flush=True)
    decode = []
    for run in range(args.repetitions):
        assert_server_binding(f"decode {run} start")
        row = measure_decode(
            args.endpoint.replace("/completions", "/chat/completions"),
            model,
            run,
            args.decode_max_tokens,
            args.decode_target_tps,
        )
        assert_server_binding(f"decode {run} end")
        decode.append(row)
        print(json.dumps({"decode": row}, sort_keys=True), flush=True)
    try:
        process_binding_after = assert_server_binding("route evidence capture")
        route_evidence = capture_route_evidence(
            args.route_log,
            provenance["required_route_markers"],
            process_binding_after,
            route_log_fd,
        )
        assert_server_binding("final evidence seal")
        final_log, _ = _route_log_snapshot(args.route_log, route_log_fd)
        if sha256(final_log) != route_evidence["route_log_sha256"]:
            raise ValueError("route log changed after evidence capture")
        if _sha256_fd(executable_fd) != process_binding_before["executable_sha256"]:
            raise ValueError("server executable changed during qualification")
    except (OSError, ValueError) as error:
        parser.error(f"invalid route evidence: {error}")
    finally:
        os.close(executable_fd)
        os.close(pid_fd)
        os.close(route_log_fd)
    try:
        assert_semantic_oracle_unchanged(
            args.semantic_oracle,
            semantic_oracle_fd,
            semantic_oracle_sha256,
            semantic_oracle_stat,
        )
        bind_semantic_oracle(prefill, decode, semantic_oracle, model, provenance)
    except (OSError, ValueError) as error:
        parser.error(f"semantic oracle qualification failed: {error}")
    if args.write_prefill_reference is not None:
        try:
            assert_semantic_oracle_unchanged(
                args.semantic_oracle,
                semantic_oracle_fd,
                semantic_oracle_sha256,
                semantic_oracle_stat,
            )
        except (OSError, ValueError) as error:
            parser.error(f"semantic oracle changed before reference seal: {error}")
        write_prefill_reference(
            args.write_prefill_reference,
            model,
            args.comparison_kind,
            provenance,
            provenance_manifest_sha256,
            route_evidence,
            semantic_oracle_sha256,
            prefill,
            decode,
        )
        valid_reference = all(
            row["cache_accounting_present"]
            and row["cached_prompt_tokens"] == 0
            and row["continuation_complete"]
            for row in prefill
        ) and decode_reference_is_valid(decode)
        print(
            json.dumps(
                {
                    "prefill_reference": str(args.write_prefill_reference),
                    "valid": valid_reference,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        os.close(semantic_oracle_fd)
        return 0 if valid_reference else 2
    try:
        reference_raw = args.prefill_reference.read_bytes()
        reference = json.loads(reference_raw)
        bind_prefill_reference(
            prefill,
            reference,
            provenance,
            route_evidence,
            semantic_oracle_sha256,
            args.comparison_kind,
        )
        bind_decode_reference(decode, reference)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        parser.error(f"invalid prefill reference: {error}")
    prefill_rates = [row["effective_prefill_tokens_per_second"] for row in prefill]
    decode_rates = [row["server_decode_tokens_per_second"] for row in decode]
    output_hashes = {row["stable_output_sha256"] for row in decode}
    try:
        relative_prefill = (
            summarize_relative_prefill(prefill)
            if args.comparison_kind == SAME_MODE
            else None
        )
    except ValueError as error:
        parser.error(f"invalid relative prefill comparison: {error}")
    prefill_relative_passes = (
        relative_prefill["passes"] if relative_prefill is not None else None
    )
    report = {
        "schema": REPORT_SCHEMA,
        "comparison_kind": args.comparison_kind,
        "runtime_mode": expected_mode,
        "concurrency": 1,
        "model": model,
        "endpoint": args.endpoint,
        "provenance": provenance,
        "provenance_manifest_sha256": provenance_manifest_sha256,
        "route_evidence": route_evidence,
        "prefill_reference_sha256": sha256(reference_raw),
        "semantic_oracle_sha256": semantic_oracle_sha256,
        "prefill_continuation_tokens": args.prefill_continuation_tokens,
        "prefill_target_tokens_per_second": args.prefill_target_tps,
        "decode_target_tokens_per_second": args.decode_target_tps,
        "prefill": prefill,
        "decode": decode,
        "median_effective_prefill_tokens_per_second": statistics.median(prefill_rates),
        "median_decode_tokens_per_second": statistics.median(decode_rates),
        "prefill_passes": all(row["passes_target"] for row in prefill),
        "relative_prefill": relative_prefill,
        "prefill_relative_passes": prefill_relative_passes,
        "decode_passes": all(row["passes_target"] for row in decode),
        "decode_deterministic": len(output_hashes) == 1,
        "decode_complete": all(row["completion_complete"] for row in decode),
        "decode_matches_reference": all(row["matches_reference"] for row in decode),
        "decode_finish_matches_reference": all(
            row["finish_matches_reference"] for row in decode
        ),
        "prefill_matches_reference": all(row["matches_reference"] for row in prefill),
        "prefill_matches_semantic_oracle": all(
            row["matches_semantic_oracle"] for row in prefill
        ),
        "decode_matches_semantic_oracle": all(
            row["matches_semantic_oracle"] for row in decode
        ),
        "prefill_matches_no_spec_reference": (
            all(row["matches_no_spec_reference"] for row in prefill)
            if args.comparison_kind == TARGET_VS_DFLASH
            else None
        ),
    }
    try:
        assert_semantic_oracle_unchanged(
            args.semantic_oracle,
            semantic_oracle_fd,
            semantic_oracle_sha256,
            semantic_oracle_stat,
        )
    except (OSError, ValueError) as error:
        parser.error(f"semantic oracle changed before report seal: {error}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_bytes(report) + b"\n")
    os.close(semantic_oracle_fd)
    print(
        json.dumps({"output": str(args.output), **report}, sort_keys=True), flush=True
    )
    return (
        0
        if report["prefill_passes"]
        and report["prefill_relative_passes"] is not False
        and report["decode_passes"]
        and report["decode_deterministic"]
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
