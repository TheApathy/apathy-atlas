# SPDX-License-Identifier: AGPL-3.0-only
"""Fail-closed full-write proof for the frozen Triton GDN output kernel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


class ProofError(RuntimeError):
    """The frozen artifact does not satisfy the full-write proof."""


EXPECTED_SHA256 = {
    "upstream": "c5e5b0f7ccdaa744c5e0eede8ec73a5767b322132a72ce46a56f04bfe4c07564",
    "source": "e3c3c68bd07b88f72c99d06ca67a23d9659f32d47690c2f62f1f73ae049aae62",
    "ttir": "9166dd7b58ceeaeba1fdb9bac9dd609219b006915a1652595260b9bec050c409",
    "ptx": "560efad0b4ee343b776e8fe8ff54d0f6be54c518974071621518cf6a2035056b",
    "cubin": "ba130d76ac7ae5fc1892cc0a0d85b0ca92bc65d1c348719fad4baa4bc0f8cb9d",
}


@dataclass(frozen=True)
class Geometry:
    batch: int = 1
    heads: int = 48
    grouped_heads: int = 16
    key_dim: int = 128
    value_dim: int = 128
    block_tokens: int = 64
    block_key: int = 128
    block_value: int = 64
    varlen: bool = True


EXACT_GEOMETRY = Geometry()
MAX_ARTIFACT_BYTES = 1 << 20


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProofError(message)


def _stable_read(
    path: Path, *, _test_after_read: Callable[[], None] | None = None
) -> bytes:
    try:
        path_before = os.lstat(path)
    except OSError as error:
        raise ProofError(f"missing artifact: {path}") from error
    _require(stat.S_ISREG(path_before.st_mode), "artifact is not a regular file")
    _require(path_before.st_nlink == 1, "artifact link-count drift")
    _require(path_before.st_size <= MAX_ARTIFACT_BYTES, "artifact exceeds size cap")
    flags = os.O_RDONLY | os.O_CLOEXEC
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ProofError(f"artifact stable open failed: {path}") from error
    try:
        opened = os.fstat(descriptor)
        _require(
            _identity(opened) == _identity(path_before), "artifact changed at open"
        )
        chunks = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(65_536, MAX_ARTIFACT_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            _require(total <= MAX_ARTIFACT_BYTES, "artifact exceeds size cap")
        if _test_after_read is not None:
            _test_after_read()
        descriptor_after = os.fstat(descriptor)
        path_after = os.lstat(path)
        expected = _identity(path_before)
        _require(_identity(descriptor_after) == expected, "artifact descriptor drift")
        _require(_identity(path_after) == expected, "artifact path drift")
        data = b"".join(chunks)
        _require(len(data) == path_before.st_size, "artifact short read")
        return data
    finally:
        os.close(descriptor)


def _validate_upstream(text: str) -> None:
    required = (
        "i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)",
        "p_o = tl.make_block_ptr(",
        "o, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0)",
        "tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))",
    )
    _require(all(item in text for item in required), "upstream store topology drift")
    _require("tl.load(p_o" not in text, "upstream output read detected")
    _require("atomic" not in text.lower(), "upstream atomic detected")


def _validate_ttir(text: str) -> None:
    required = (
        "%o_22 = tt.addptr %o, %v_20",
        "%14 = tt.splat %o_22",
        "%15 = tt.addptr %14, %b_v_121",
        "tt.store %15, %13, %b_v_128",
    )
    _require(all(item in text for item in required), "TTIR output dataflow drift")
    loads = [line for line in text.splitlines() if "tt.load" in line]
    _require(
        not any(re.search(r"%(?:o_22|14|15)\b", line) for line in loads),
        "TTIR output read detected",
    )
    _require(text.count("tt.store ") == 1, "TTIR store count drift")
    _require("atomic" not in text.lower(), "TTIR atomic detected")


def _validate_ptx(text: str) -> None:
    required = (
        "ld.param.b64 \t%rd60, [chunk_fwd_kernel_o_param_5];",
        "add.s64 \t%rd68, %rd60, %rd66;",
        "add.s64 \t%rd51, %rd68, %rd145;",
        "add.s64 \t%rd52, %rd68, %rd146;",
        "add.s64 \t%rd53, %rd68, %rd147;",
        "add.s64 \t%rd54, %rd68, %rd148;",
    )
    _require(all(item in text for item in required), "PTX output dataflow drift")
    global_stores = [line for line in text.splitlines() if "st.global" in line]
    _require(len(global_stores) == 4, "PTX global-store count drift")
    _require(
        all("st.global.v4.b32" in line for line in global_stores),
        "PTX output store width drift",
    )
    output_regs = ("%rd60", "%rd68", "%rd51", "%rd52", "%rd53", "%rd54")
    global_loads = [line for line in text.splitlines() if "ld.global" in line]
    _require(
        not any(reg in line for line in global_loads for reg in output_regs),
        "PTX output-address load detected",
    )
    lowered = text.lower()
    _require(
        "atom.global" not in lowered and "red.global" not in lowered,
        "PTX global atomic detected",
    )


def _validate_geometry(geometry: Geometry, tokens: int) -> dict[str, int]:
    _require(geometry == EXACT_GEOMETRY, "kernel geometry drift")
    _require(tokens > 0, "token count must be positive")
    _require(
        geometry.value_dim % geometry.block_value == 0,
        "value tiles do not partition output",
    )
    value_tiles = geometry.value_dim // geometry.block_value
    token_tiles = (tokens + geometry.block_tokens - 1) // geometry.block_tokens
    _require(value_tiles == 2, "output grid-x drift")
    _require(geometry.batch * geometry.heads == 48, "output grid-z drift")
    for token in {0, tokens - 1}:
        for head in {0, geometry.heads - 1}:
            for value in {0, geometry.block_value - 1, geometry.value_dim - 1}:
                owner = (
                    value // geometry.block_value,
                    token // geometry.block_tokens,
                    head,
                )
                _require(0 <= owner[0] < value_tiles, "invalid value owner")
                _require(0 <= owner[1] < token_tiles, "invalid token owner")
                _require(0 <= owner[2] < 48, "invalid head owner")
                local_t = token - owner[1] * geometry.block_tokens
                local_v = value - owner[0] * geometry.block_value
                _require(
                    0 <= local_t < 64 and 0 <= local_v < 64,
                    "valid output excluded by boundary predicate",
                )
    return {
        "tokens": tokens,
        "grid_x": value_tiles,
        "grid_y": token_tiles,
        "grid_z": geometry.batch * geometry.heads,
        "elements": tokens * geometry.heads * geometry.value_dim,
    }


def prove(
    upstream: Path, artifact_dir: Path, tokens: int, geometry: Geometry = EXACT_GEOMETRY
) -> dict[str, int | str]:
    paths = {
        "upstream": upstream,
        "source": artifact_dir / "chunk_fwd_kernel_o.source",
        "ttir": artifact_dir / "chunk_fwd_kernel_o.ttir",
        "ptx": artifact_dir / "chunk_fwd_kernel_o.ptx",
        "cubin": artifact_dir / "chunk_fwd_kernel_o.cubin",
    }
    held = {name: _stable_read(path) for name, path in paths.items()}
    for name, data in held.items():
        _require(
            hashlib.sha256(data).hexdigest() == EXPECTED_SHA256[name],
            f"{name} hash drift",
        )
    try:
        upstream_text = held["upstream"].decode("utf-8", errors="strict")
        ttir_text = held["ttir"].decode("utf-8", errors="strict")
        ptx_text = held["ptx"].decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise ProofError("compiler text artifact is not UTF-8") from error
    _validate_upstream(upstream_text)
    _validate_ttir(ttir_text)
    _validate_ptx(ptx_text)
    result: dict[str, int | str] = _validate_geometry(geometry, tokens)
    result["proof"] = "disjoint-exhaustive-output-overwrite"
    result["cubin_sha256"] = EXPECTED_SHA256["cubin"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--tokens", type=int, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            prove(args.upstream, args.artifact_dir, args.tokens),
            sort_keys=True,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()
