"""Fail-closed authorization boundary for a future GPU executor."""

import datetime as dt
import hashlib
import os
import re
from pathlib import Path
from typing import Any

from frozen_triton_c143_io import (
    GateError,
    decode_json,
    require_bool,
    require_exact_keys,
    stable_read,
)

NONCE = re.compile(r"[0-9a-f]{64}\Z")


def authorize_gpu_execution(
    *,
    explicit_flag: str | None,
    nonce: str | None,
    reservation_path: Path | None,
    manifest_sha256: str,
    gate_source_sha256: str,
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    if os.environ.get("ATLAS_GDN_C143_TRITON_RAW_GPU") != "1":
        raise GateError("GPU authorization: environment gate is not exactly 1")
    if explicit_flag != "I_UNDERSTAND_THIS_RUNS_CUDA":
        raise GateError("GPU authorization: explicit execution flag missing")
    if nonce is None or NONCE.fullmatch(nonce) is None:
        raise GateError("GPU authorization: 64-hex nonce required")
    if reservation_path is None:
        raise GateError("GPU authorization: reservation receipt required")
    raw, receipt_sha, receipt_stat = stable_read(
        reservation_path, "GPU reservation", maximum_size=64 << 10
    )
    if receipt_stat.st_mode & 0o022:
        raise GateError(
            "GPU authorization: reservation receipt is group/world writable"
        )
    receipt = require_exact_keys(
        decode_json(raw, "GPU reservation"),
        {
            "schema",
            "claim",
            "owner",
            "scope",
            "authorization",
            "released",
            "nonce_sha256",
            "manifest_sha256",
            "gate_sha256",
            "expires_utc",
        },
        "GPU reservation",
    )
    if receipt["schema"] != "atlas.gpu.reservation.v1":
        raise GateError("GPU authorization: reservation schema drift")
    if receipt["claim"] != "qwen38-gdn-c143-frozen-triton-raw-gpu-oracle":
        raise GateError("GPU authorization: wrong reservation claim")
    if type(receipt["owner"]) is not str or not receipt["owner"].startswith("/root"):
        raise GateError("GPU authorization: invalid reservation owner")
    if (
        receipt["scope"] != "local-gb10-cuda-execution"
        or receipt["authorization"] != "GPU_EXECUTION_APPROVED"
        or require_bool(receipt["released"], "GPU reservation.released")
    ):
        raise GateError("GPU authorization: inactive reservation")
    if receipt["nonce_sha256"] != hashlib.sha256(nonce.encode("ascii")).hexdigest():
        raise GateError("GPU authorization: nonce mismatch")
    if receipt["manifest_sha256"] != manifest_sha256:
        raise GateError("GPU authorization: manifest binding mismatch")
    if receipt["gate_sha256"] != gate_source_sha256:
        raise GateError("GPU authorization: gate source binding mismatch")
    try:
        expires = dt.datetime.fromisoformat(
            receipt["expires_utc"].replace("Z", "+00:00")
        )
    except (AttributeError, ValueError) as exc:
        raise GateError("GPU authorization: invalid UTC expiry") from exc
    if expires.tzinfo is None or expires.utcoffset() != dt.timedelta(0):
        raise GateError("GPU authorization: expiry must be UTC")
    current = now or dt.datetime.now(dt.timezone.utc)
    if expires <= current or expires > current + dt.timedelta(hours=24):
        raise GateError("GPU authorization: reservation expired or exceeds 24h")
    return {"reservation_receipt_sha256": receipt_sha, "owner": receipt["owner"]}
