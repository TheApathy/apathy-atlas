# SPDX-License-Identifier: AGPL-3.0-only
"""Inert identity, workload, and sole-delta contract for raw PLE parity."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# fmt: off

SCHEMA = "atlas-qwen38-flash-next-post-ple-parity-frame-v1"
RESULT_SCHEMA = "atlas-qwen38-flash-next-post-ple-parity-v1"
EXECUTE = "EXECUTE_REVIEWED_CAPTURE_ELF_RAW_PLE_PARITY_WITH_ROOT_AUTHORITY"
MODEL_NAME = "qwen3.8-flash-next"
MODEL = Path("/home/flocka/models/Qwen3.8-Flash-Next-NVFP4-Offload")
MODEL_MANIFEST = Path("/var/tmp/atlas-b62-qwen38-flash-next-model-manifest.json")
BUILD = Path("/var/tmp/atlas-flash-b64-capture-build-HJfKPLKh")
ELF = BUILD / "release/spark"
PORT = 8996
BUILD_ID = "0be06300ea805695af3d4e3aa038e95fda2e6f84"
ELF_SIZE = 38_947_552
SOURCE_FILES = 1_840
TARGET = "gb10|qwen3.8-flash-next|nvfp4|sm_121f"
PINS = {
    "elf": "8a9b4bb8ea87388d1c932e15713172bf82977cc2f08033ecf4be737bdac9c814",
    "build_receipt": "d2b56cd278fe5ecc41e7db89b7b1b1845013834951d3376bcef231ec31351340",
    "source_manifest": "c051689e926d743d784062620b40e0835ee134726c4220e96f3f8e70d78e103a",
    "selected_manifest": "7aabde0e81076d52b300a7682145f6dcb303ad31bdf47fd382dac743c09a5c7b",
    "ptx_manifest": "a8d55eefb4caa9d3ca64dd0cf2943dc13255f9131ec246ad5c45d229279ca3f7",
    "artifact_manifest": "a18c6461e9080246b3a9a20b9770cbf640fa2f113ce767204caffe774ae8d57d",
    "capture_source": "0b161916165515752f85abf425aaff7c546ade987e5b4701076e62c22881c8c5",
    "capture_callsite": "0737ab71b0c466955b94d24fe38aa3b2cd550d1837998ff7d4af382358508093",
    "model_manifest": "b80d32b07ae96bef448cbd9868a60aa29f5fc953a85d4c9936704041c5478a94",
    "model_content_root": "b3b3cf49e8549d0b39d88afda9db4733aff7a4c591369dad45fcc04789dc9b2c",
}
BUILD_FILES = {
    "BUILD_RECEIPT.md": ("build_receipt", None),
    "source-manifest.sha256": ("source_manifest", None),
    "selected-source-manifest.sha256": ("selected_manifest", 25),
    "ptx-manifest.sha256": ("ptx_manifest", 154),
    "artifact-manifest.sha256": ("artifact_manifest", None),
}
TOKEN_PINS = {
    ("m38", 0): ("42132bb1ea9e52d7cf63d8d34428f15f60ff868a11d76f590ce0af77f71c79c2", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "42132bb1ea9e52d7cf63d8d34428f15f60ff868a11d76f590ce0af77f71c79c2", "42132bb1ea9e52d7cf63d8d34428f15f60ff868a11d76f590ce0af77f71c79c2"),
    ("m2013", 0): ("acb7ceb6fea8b24cf8ca79bb131e8461285fa64809d0219cdac92c22a4d03079", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855", "084b260ab016df661887fd8b5824073c5dde4818934b84417abd64d076d5734e", "084b260ab016df661887fd8b5824073c5dde4818934b84417abd64d076d5734e"),
    ("m2013", 2_000): ("acb7ceb6fea8b24cf8ca79bb131e8461285fa64809d0219cdac92c22a4d03079", "084b260ab016df661887fd8b5824073c5dde4818934b84417abd64d076d5734e", "acb7ceb6fea8b24cf8ca79bb131e8461285fa64809d0219cdac92c22a4d03079", "03863432552fda5d9c2d57df902854f96255e90940353a653bf0f69811d761f8"),
}
ARM_SELECTORS = {"serial0": "0", "whole_prompt1": "1"}
SCHEDULE = (("m38", "serial0"), ("m38", "whole_prompt1"), ("m2013", "serial0"), ("m2013", "whole_prompt1"))
FRAME_LAYOUT = {
    38: ((0, 38, True),),
    2_013: ((0, 2_000, True), (2_000, 13, False)),
}
ROUTE_ENV = {
    "ATLAS_DFLASH_ASYNC": "0",
    "ATLAS_DFLASH_MARKOV": "0",
    "ATLAS_DFLASH_PLD": "0",
    "ATLAS_DFLASH_RETRIEVAL": "0",
    "ATLAS_DFLASH_SAM": "0",
    "ATLAS_DFLASH_SPEC_CYCLE_V2": "0",
    "ATLAS_EXPERIMENTAL_NATIVE_QWEN4_DFLASH": "0",
    "ATLAS_GDN_PREFILL_TUNED": "0",
    "ATLAS_MOE_EXACT_PREFILL_GRID": "0",
    "ATLAS_MULTISEQ_GRAPHS": "0",
    "ATLAS_NVFP4_MOE_WORKLIST": "0",
    "ATLAS_PLE_CACHE_MB": "0",
    "ATLAS_PLE_PAGE_CACHE": "0",
    "ATLAS_PREFILL_FFN_FAST": "0",
    "ATLAS_PREFILL_FFN_PIPE": "0",
    "ATLAS_PREFILL_PHASE_PROFILE": "0",
    "ATLAS_PREFILL_PROJ_FAST": "0",
    "ATLAS_PREFILL_PROJ_PIPE": "0",
    "ATLAS_QWEN4_ATTN_PREFILL_BATCH": "0",
    "ATLAS_QWEN4_HYPER_PREFILL_GEMM": "0",
    "ATLAS_QWEN4_PLE_SEGMENTED_GRAPHS": "0",
    "ATLAS_QWEN4_QSA_PREFILL_GEMM": "0",
    "ATLAS_QWEN4_SSM_PREFILL_BATCH": "0",
    "ATLAS_QWEN4_SSM_PREFILL_FP32": "0",
    "ATLAS_SELF_CONTEXT_DRAFT": "0",
    "ATLAS_SELF_SPEC_SPARSE": "0",
    "ATLAS_SSM_MULTI_SEQ_GRAPH": "0",
    "ATLAS_WEIGHT_CACHE": "0",
}


@dataclass(frozen=True)
class RequestSpec:
    name: str
    wire: bytes
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    content_bytes: int
    content_sha256: str
    stable_sha256: str


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _wire(body: dict[str, Any]) -> bytes:
    return json.dumps(body, ensure_ascii=True, allow_nan=False).encode()


def _m2013_prompt() -> str:
    records = []
    for index in range(1, 124):
        values = {10: "ORCHID-7391", 61: "COBALT-2846", 118: "SAFFRON-9153"}
        if index in values:
            records.append(
                f"CHECKPOINT_{['ALPHA', 'BETA', 'GAMMA'][[10, 61, 118].index(index)]} is {values[index]}."
            )
        else:
            records.append(
                f"Record {index}: the archive entry is ordinary and contains no checkpoint value."
            )
    return " ".join(records) + (
        " Return only CHECKPOINT_ALPHA, CHECKPOINT_BETA, and CHECKPOINT_GAMMA "
        "in that order, separated by commas."
    )


def request_specs() -> dict[str, RequestSpec]:
    m38 = _wire(
        {
            "model": MODEL_NAME,
            "temperature": 0.0,
            "max_tokens": 400,
            "messages": [
                {
                    "role": "user",
                    "content": "Write a complete MinHeap class in Python with insert, extract_min, and heapify, then explain the complexity of each method.",
                }
            ],
            "reasoning_effort": "none",
        }
    )
    m2013 = _wire(
        {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": _m2013_prompt()}],
            "reasoning_effort": "none",
            "temperature": 0.0,
            "max_tokens": 32,
            "stream": False,
        }
    )
    if (len(m38), sha256(m38), len(m2013), sha256(m2013)) != (
        268,
        "cd22f3df9cef8042f6b30dfa82a76863dd1f42732f1b29736693c719e621fa93",
        9_377,
        "077c25c42b9c911affc71a6660159d4e13c2708c8d636938cd84f14b7d439095",
    ):
        raise RuntimeError("hard-pinned parity request wire drift")
    return {
        "m38": RequestSpec("m38", m38, 38, 400, "length", 1_490, "fce20d91ead45078417ef0aecef08e7d2b749e3d3f88cc77efebf824e11b90b6", "f811cdc565dff063074f8bb1d0bb3fd55c8b10f3438c517cf4d59128f23cf790"),
        "m2013": RequestSpec("m2013", m2013, 2_013, 27, "stop", 38, "e331cbae40fd262c53717fe8dafd3f98ef409a59a7b18fb7f736428d7e97bfa1", "1dac947db1eb23cab6c428f8c53c10eb01333a5546557c869e37811497e462c9"),
    }


def require_released_pins() -> None:
    values = (
        *PINS.values(),
        *(value for pins in TOKEN_PINS.values() for value in pins),
    )
    if any(
        len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value)
        for value in values
    ):
        raise RuntimeError("capture/model authority placeholders are unreleased")
    if len(BUILD_ID) != 40 or any(ch not in "0123456789abcdef" for ch in BUILD_ID):
        raise RuntimeError("capture ELF build-id placeholder is unreleased")
    if type(ELF_SIZE) is not int or ELF_SIZE <= 0:
        raise RuntimeError("capture ELF size placeholder is unreleased")
    if type(SOURCE_FILES) is not int or SOURCE_FILES <= 0:
        raise RuntimeError("capture source census placeholder is unreleased")


def server_argv() -> list[str]:
    options = {
        "--model-from-path": str(MODEL),
        "--model-name": MODEL_NAME,
        "--kernel-target": MODEL_NAME,
        "--port": str(PORT),
        "--max-seq-len": "2048",
        "--max-prefill-tokens": "2000",
        "--max-num-seqs": "1",
        "--max-batch-size": "1",
        "--ssm-cache-slots": "0",
        "--kv-cache-dtype": "bf16",
        "--gpu-memory-utilization": "0.90",
        "--oom-guard-mb": "4096",
        "--request-timeout": "1200",
    }
    return (
        [str(ELF), "serve"]
        + [value for item in options.items() for value in item]
        + ["--no-tui"]
    )


def arm_environment(arm: str, nonce: str, root: Path) -> dict[str, str]:
    if arm not in ARM_SELECTORS:
        raise RuntimeError("unknown PLE parity arm")
    return {
        **ROUTE_ENV,
        "ATLAS_QWEN4_PLE_PREFILL_BATCH": ARM_SELECTORS[arm],
        "ATLAS_QWEN4_PLE_PARITY_CAPTURE": "1",
        "ATLAS_QWEN4_PLE_PARITY_NONCE": nonce,
        "ATLAS_QWEN4_PLE_PARITY_ROOT": str(root),
        "LANG": "C.UTF-8",
        "LD_LIBRARY_PATH": "/usr/local/cuda-13.0/targets/sbsa-linux/lib",
        "PATH": "/usr/local/cuda-13.0/bin:/usr/local/bin:/usr/bin:/bin",
        "RUST_LOG": "info",
    }
# fmt: on
