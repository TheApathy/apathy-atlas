#!/usr/bin/env -S -i /usr/bin/python3 -I
# SPDX-License-Identifier: AGPL-3.0-only
"""Fail-closed, descriptor-bound launch profile for official Qwen3.8 DFlash2."""

import hashlib
import json
import math
import os
import re
import stat
import struct
import sys

EXPECTED_BINARY_SHA256 = "UNRELEASED_POST_OFFICIAL_DFLASH2_FIX_BINARY"
EXPECTED_ABI_ATTESTATION_SHA256 = "UNRELEASED_POST_OFFICIAL_DFLASH2_FIX_ATTESTATION"
EXPECTED_ATTN_PTX_SHA256 = "71d95d7815d36cc0df61070d125b1598464a67aaaac44425f5f4a0ba6d6e61c2"
BINARY_PATH = "/home/flocka/atlas/qwen38/dflash2-official/spark"
MODEL_DIR = "/home/flocka/atlas/qwen38/optimized-qwen"
DRAFT_DIR = "/home/flocka/atlas/qwen38/drafter-dflash2-incoai"
ABI_ATTESTATION_PATH = "/home/flocka/atlas/qwen38/dflash2-official/abi-v2.json"
SESSION_ROOT_TEMPLATE = "/run/user/{uid}/atlas-qwen38-dflash2-official-v2"

TARGET_FILES = (
    (
        "target_config",
        "config.json",
        "267be2125ee2ec272555748c87cc636b25a96107946f05491bd6043151c7fe4e",
        None,
    ),
    (
        "target_index",
        "model.safetensors.index.json",
        "f9ba0436d933e2362fb1bfa0508931c28b68f1fddbd5b94e6d50712e36a0636c",
        None,
    ),
    (
        "target_tokenizer",
        "tokenizer.json",
        "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3",
        None,
    ),
    (
        "target_tokenizer_config",
        "tokenizer_config.json",
        "5a205aa76328f59df93a9091d0496aeda4615c6b0c495cbd37e14c79c0cd0f94",
        None,
    ),
    (
        "target_chat_template",
        "chat_template.jinja",
        "c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041",
        None,
    ),
    (
        "target_generation_config",
        "generation_config.json",
        "d0dbf670c6a372817b2ff92d5d47e3130d35de9c3a7164ba455fd7a88255b362",
        None,
    ),
    (
        "target_quant_config",
        "hf_quant_config.json",
        "0c1004d622f835eef17ba193d8e966ecfd2d5218cbc8d25b7effff8cb4011893",
        None,
    ),
    (
        "target_weight:model-00001-of-00006.safetensors",
        "model-00001-of-00006.safetensors",
        "4b71c5d9d5027c88d1df9f53b93ad77b8d5e428d5d6f75ae42063704d94930b9",
        4013662656,
    ),
    (
        "target_weight:model-00002-of-00006.safetensors",
        "model-00002-of-00006.safetensors",
        "d0ceb042ef003ec63e00203823eb70d4815c9fc0430ba04a5442b6b8bde19093",
        4104675536,
    ),
    (
        "target_weight:model-00003-of-00006.safetensors",
        "model-00003-of-00006.safetensors",
        "b633a26c1d3b97bf061585363b99d4c01650b4e3f6d0cbb18e7c7e03ca2de1e4",
        4028901296,
    ),
    (
        "target_weight:model-00004-of-00006.safetensors",
        "model-00004-of-00006.safetensors",
        "485ea312626e2d1f3a820d9f7545809cef1e2e77414e3977a61d32dacd370bd0",
        4003421984,
    ),
    (
        "target_weight:model-00005-of-00006.safetensors",
        "model-00005-of-00006.safetensors",
        "2aa61bab1ed252fe3cdb479286b5b7add7e8ad567501add5f6eeef3987b03d31",
        4000515080,
    ),
    (
        "target_weight:model-00006-of-00006.safetensors",
        "model-00006-of-00006.safetensors",
        "8214cd21a46832380c06be552c42d9514c6cc9f27d98e09e2a09e36a8bd6e6ce",
        5380672368,
    ),
    (
        "target_weight:nvfp4_experts_mtp.safetensors",
        "nvfp4_experts_mtp.safetensors",
        "d97834985f2442cd8b7e48e8426636e2ef8836ad1c7c5fba4ad879a0fcc66685",
        849400408,
    ),
)
DRAFT_CONFIG_SHA256 = "873e3556509b0da06e29654ba00d4944888d4b5e8a33afde25f7eb27d321e980"
DRAFT_MODEL_SHA256 = "67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c"
DRAFT_MODEL_BYTES = 3848817896
DRAFT_HEADER_LENGTH = 8928
DRAFT_HEADER_SHA256 = "0c2c70601b30f8d1ca7d5794b817779ba2dcf1956cfc7d4f83e87091e1ab7c8c"
DRAFT_TENSOR_COUNT = 81
DRAFT_PARAMETER_COUNT = 1924404480

EXEC_ENV = {
    "ATLAS_DFLASH_ASYNC": "0",
    "ATLAS_DFLASH_CTX_WINDOW": "2047",
    "ATLAS_DFLASH_DRAFT_CAP": "7",
    "ATLAS_DFLASH_NOISE_ONLY": "1",
    "ATLAS_DFLASH_QUANT": "nvfp4",
    "ATLAS_DFLASH_SPEC_CYCLE_V2": "1",
    "ATLAS_DFLASH_SWA": "1",
    "CUDA_VISIBLE_DEVICES": "0",
    "HOME": "/nonexistent",
    "LANG": "C",
    "LC_ALL": "C",
    "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/usr/lib/aarch64-linux-gnu",
    "PATH": "/usr/bin:/bin",
    "RUST_LOG": "info",
}

DIRECTORY_OPEN_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
FILE_OPEN_FLAGS = os.O_RDONLY | os.O_NOFOLLOW
STABLE_FIELDS = (
    "st_dev",
    "st_ino",
    "st_mode",
    "st_nlink",
    "st_uid",
    "st_gid",
    "st_size",
    "st_mtime_ns",
    "st_ctime_ns",
)
ABI_SENTINEL_LINE = (
    b'EXPECTED_ABI_ATTESTATION_SHA256 = '
    b'"UNRELEASED_POST_OFFICIAL_DFLASH2_FIX_ATTESTATION"'
)


def die(message):
    raise SystemExit(f"serve-dflash2-official: {message}")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def path_components(path, label):
    if not path.startswith("/") or path == "/" or "//" in path:
        die(f"{label} must be a canonical absolute path")
    components = path.split("/")[1:]
    if any(component in ("", ".", "..") for component in components):
        die(f"{label} contains a forbidden path component")
    return components


def open_directory(path, label):
    components = path_components(path, label)
    descriptor = os.open("/", DIRECTORY_OPEN_FLAGS)
    try:
        for component in components:
            try:
                child = os.open(component, DIRECTORY_OPEN_FLAGS, dir_fd=descriptor)
            except OSError as error:
                die(f"{label} component is not a real directory: {error}")
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def open_file(path, label):
    components = path_components(path, label)
    parent = "/" + "/".join(components[:-1])
    parent_fd = open_directory(parent, label)
    try:
        try:
            return os.open(components[-1], FILE_OPEN_FLAGS, dir_fd=parent_fd)
        except OSError as error:
            die(f"{label} is unavailable or is a symlink: {error}")
    finally:
        os.close(parent_fd)


def stable_tuple(metadata):
    return tuple(getattr(metadata, field) for field in STABLE_FIELDS)


def hash_descriptor(descriptor):
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while chunk := os.read(descriptor, 8 * 1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def read_descriptor(descriptor, limit, label):
    metadata = os.fstat(descriptor)
    if metadata.st_size > limit:
        die(f"{label} exceeds its bounded read limit")
    os.lseek(descriptor, 0, os.SEEK_SET)
    data = bytearray()
    while chunk := os.read(descriptor, min(1024 * 1024, metadata.st_size - len(data))):
        data.extend(chunk)
    if len(data) != metadata.st_size:
        die(f"{label} was short-read")
    return bytes(data)


def open_artifact(path, label, expected_sha=None, expected_size=None, executable=False):
    descriptor = open_file(path, label)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            die(f"{label} must be a regular file")
        if before.st_nlink != 1 or stat.S_IMODE(before.st_mode) & 0o222:
            die(f"{label} must be immutable (one link and no write bits)")
        if executable and not stat.S_IMODE(before.st_mode) & 0o111:
            die(f"{label} must be executable")
        if expected_size is not None and before.st_size != expected_size:
            die(f"{label} size mismatch: expected {expected_size}, got {before.st_size}")
        actual_sha = hash_descriptor(descriptor)
        after = os.fstat(descriptor)
        if stable_tuple(before) != stable_tuple(after):
            die(f"{label} changed while hashing")
        if expected_sha is not None and actual_sha != expected_sha:
            die(f"{label} sha256 mismatch: expected {expected_sha}, got {actual_sha}")
        os.set_inheritable(descriptor, True)
        identity = {
            "path": path,
            "sha256": actual_sha,
            "size": after.st_size,
            "device": after.st_dev,
            "inode": after.st_ino,
            "mode": stat.S_IMODE(after.st_mode),
            "links": after.st_nlink,
            "mtime_ns": after.st_mtime_ns,
            "ctime_ns": after.st_ctime_ns,
            "retained_fd": descriptor,
        }
        return descriptor, identity, stable_tuple(after)
    except BaseException:
        os.close(descriptor)
        raise


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            die(f"duplicate JSON key {key!r}")
        value[key] = item
    return value


def reject_constant(value):
    die(f"non-finite JSON constant {value}")


def strict_json(raw, label):
    try:
        return json.loads(
            raw,
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        die(f"{label} is not strict JSON: {error}")


def required_field(value, path, expected):
    cursor = value
    for component in path:
        if type(cursor) is not dict or component not in cursor:
            die(f"draft config field {'.'.join(path)} is missing")
        cursor = cursor[component]
    if type(cursor) is not type(expected) or cursor != expected:
        die(f"draft config field {'.'.join(path)} must equal {expected!r}; got {cursor!r}")


def normalize_launcher(raw):
    pattern = rb'(?m)^EXPECTED_ABI_ATTESTATION_SHA256 = "[^"]+"$'
    normalized, count = re.subn(pattern, ABI_SENTINEL_LINE, raw)
    if count != 1:
        die("launcher has no unique normalized ABI authority field")
    return hashlib.sha256(normalized).hexdigest()


def assert_still_stable(opened):
    for label, descriptor, expected in opened:
        if stable_tuple(os.fstat(descriptor)) != expected:
            die(f"{label} changed after admission")


def acquire_session_root(path):
    components = path_components(path, "session path")
    parent_path = "/" + "/".join(components[:-1])
    parent_fd = open_directory(parent_path, "session path")
    try:
        try:
            os.mkdir(components[-1], 0o700, dir_fd=parent_fd)
        except FileExistsError:
            die("fixed launch session already exists")
        except OSError as error:
            die(f"cannot create fixed launch session: {error}")
        session_fd = os.open(components[-1], DIRECTORY_OPEN_FLAGS, dir_fd=parent_fd)
        session_stat = os.fstat(session_fd)
        path_stat = os.stat(components[-1], dir_fd=parent_fd, follow_symlinks=False)
        if stable_tuple(session_stat) != stable_tuple(path_stat):
            die("fixed launch session identity changed after creation")
        return parent_fd, session_fd, components[-1], (session_stat.st_dev, session_stat.st_ino)
    except BaseException:
        os.close(parent_fd)
        raise


def create_view(session_fd, name, artifacts):
    os.mkdir(name, 0o700, dir_fd=session_fd)
    view_fd = os.open(name, DIRECTORY_OPEN_FLAGS, dir_fd=session_fd)
    try:
        for filename, descriptor in artifacts:
            if "/" in filename or filename in ("", ".", ".."):
                die("invalid descriptor-view filename")
            os.symlink(f"/proc/self/fd/{descriptor}", filename, dir_fd=view_fd)
        os.fchmod(view_fd, 0o500)
        os.set_inheritable(view_fd, True)
        return view_fd
    except BaseException:
        os.close(view_fd)
        raise


def write_receipt(session_fd, payload):
    try:
        descriptor = os.open(
            "launch-receipt.json",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=session_fd,
        )
    except FileExistsError:
        die("fixed launch receipt already exists")
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                die("short write while creating fixed launch receipt")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.fsync(session_fd)


if len(sys.argv) != 1:
    die("CLI arguments are forbidden; this profile has no override surface")
if not re.fullmatch(r"[0-9a-f]{64}", EXPECTED_BINARY_SHA256):
    die("official binary identity is unreleased")
if not re.fullmatch(r"[0-9a-f]{64}", EXPECTED_ABI_ATTESTATION_SHA256):
    die("official ABI attestation identity is unreleased")
if os.execve not in os.supports_fd:
    die("this Python runtime cannot exec a retained descriptor")

opened = []
manifest = {}


def admit(path, label, digest=None, size=None, executable=False):
    descriptor, identity, stable = open_artifact(path, label, digest, size, executable)
    opened.append((label, descriptor, stable))
    manifest[label] = identity
    return descriptor


launcher_fd = admit(os.path.abspath(__file__), "launcher")
launcher_raw = read_descriptor(launcher_fd, 1024 * 1024, "launcher")
launcher_normalized_sha256 = normalize_launcher(launcher_raw)
binary_fd = admit(BINARY_PATH, "binary", EXPECTED_BINARY_SHA256, executable=True)
abi_fd = admit(ABI_ATTESTATION_PATH, "ABI attestation", EXPECTED_ABI_ATTESTATION_SHA256)

target_descriptors = {}
for label, filename, digest, size in TARGET_FILES:
    target_descriptors[filename] = admit(
        os.path.join(MODEL_DIR, filename), label, digest, size
    )
draft_config_fd = admit(
    os.path.join(DRAFT_DIR, "config.json"), "draft_config", DRAFT_CONFIG_SHA256
)
draft_model_fd = admit(
    os.path.join(DRAFT_DIR, "model.safetensors"),
    "draft_model",
    DRAFT_MODEL_SHA256,
    DRAFT_MODEL_BYTES,
)

abi = strict_json(read_descriptor(abi_fd, 64 * 1024, "ABI attestation"), "ABI attestation")
expected_abi = {
    "schema": "atlas-qwen38-attention-abi-v2",
    "binary_sha256": EXPECTED_BINARY_SHA256,
    "launcher_normalized_sha256": launcher_normalized_sha256,
    "host_argument_count": 13,
    "ptx_parameter_count": 13,
    "ptx_sha256": EXPECTED_ATTN_PTX_SHA256,
}
if type(abi) is not dict or set(abi) != set(expected_abi):
    die("ABI attestation keys do not match the exact schema")
for key, expected in expected_abi.items():
    if type(abi.get(key)) is not type(expected) or abi.get(key) != expected:
        die(f"ABI field {key} must equal {expected!r}; got {abi.get(key)!r}")

index_fd = target_descriptors["model.safetensors.index.json"]
index_json = strict_json(read_descriptor(index_fd, 1024 * 1024, "target model index"), "target model index")
if type(index_json) is not dict or type(index_json.get("weight_map")) is not dict:
    die("target model index must contain an object weight_map")
weight_values = list(index_json["weight_map"].values())
if any(type(value) is not str for value in weight_values):
    die("target model index weight_map values must be strings")
expected_weight_names = {
    filename for label, filename, _, _ in TARGET_FILES if label.startswith("target_weight:")
}
if set(weight_values) != expected_weight_names:
    die("target model index weight files do not match the exact target manifest")

draft = strict_json(
    read_descriptor(draft_config_fd, 1024 * 1024, "draft config"), "draft config"
)
for field_path, expected in (
    (("architectures",), ["DFlash2DraftModel"]),
    (("is_causal",), False),
    (("dtype",), "bfloat16"),
    (("hidden_act",), "silu"),
    (("hidden_size",), 5120),
    (("intermediate_size",), 17408),
    (("num_hidden_layers",), 5),
    (("num_attention_heads",), 32),
    (("num_key_value_heads",), 8),
    (("head_dim",), 128),
    (("vocab_size",), 248320),
    (("rms_norm_eps",), 1e-6),
    (("max_position_embeddings",), 262144),
    (("sliding_window",), 2048),
    (("use_sliding_window",), True),
    (("layer_types",), ["sliding_attention"] * 5),
    (("dflash_config", "block_size"), 8),
    (("dflash_config", "conv_kernel_size"), 2),
    (("dflash_config", "conv_group_size"), 16),
    (("dflash_config", "mask_token_id"), 248070),
    (("dflash_config", "selector_rank"), 256),
    (("dflash_config", "selector_top_k"), 16),
    (("dflash_config", "target_layer_ids"), [5, 19, 33, 47, 61]),
    (("rope_parameters", "rope_type"), "default"),
    (("rope_parameters", "rope_theta"), 10000000),
):
    required_field(draft, field_path, expected)
if "block_size" in draft or "rope_scaling" in draft:
    die("draft config contains a forbidden root block_size or rope_scaling override")

raw_length = os.pread(draft_model_fd, 8, 0)
if len(raw_length) != 8:
    die("draft safetensors header length is truncated")
header_length = struct.unpack("<Q", raw_length)[0]
if header_length != DRAFT_HEADER_LENGTH:
    die(f"draft safetensors header length mismatch: expected {DRAFT_HEADER_LENGTH}, got {header_length}")
raw_header = os.pread(draft_model_fd, header_length, 8)
if len(raw_header) != header_length:
    die("draft safetensors header is truncated")
header_sha = hashlib.sha256(raw_header).hexdigest()
if header_sha != DRAFT_HEADER_SHA256:
    die(f"draft safetensors header sha256 mismatch: expected {DRAFT_HEADER_SHA256}, got {header_sha}")
header = strict_json(raw_header, "draft safetensors header")
tensors = {key: value for key, value in header.items() if key != "__metadata__"}
if len(tensors) != DRAFT_TENSOR_COUNT:
    die(f"draft safetensors tensor count must equal {DRAFT_TENSOR_COUNT}")
parameter_count = 0
intervals = []
for name, tensor in tensors.items():
    if type(tensor) is not dict or tensor.get("dtype") != "BF16":
        die(f"draft tensor {name!r} is not BF16")
    shape = tensor.get("shape")
    offsets = tensor.get("data_offsets")
    if type(shape) is not list or any(type(dimension) is not int or dimension < 0 for dimension in shape):
        die(f"draft tensor {name!r} has an invalid shape")
    if (
        type(offsets) is not list
        or len(offsets) != 2
        or any(type(offset) is not int for offset in offsets)
        or offsets[0] < 0
        or offsets[1] < offsets[0]
    ):
        die(f"draft tensor {name!r} has invalid data offsets")
    count = math.prod(shape)
    if offsets[1] - offsets[0] != count * 2:
        die(f"draft tensor {name!r} BF16 extent does not match its shape")
    parameter_count += count
    intervals.append(tuple(offsets))
if parameter_count != DRAFT_PARAMETER_COUNT:
    die(f"draft safetensors parameter count must equal {DRAFT_PARAMETER_COUNT}")
cursor = 0
for start, end in sorted(intervals):
    if start != cursor:
        die("draft safetensors data offsets are not exactly contiguous")
    cursor = end
if cursor != DRAFT_MODEL_BYTES - 8 - DRAFT_HEADER_LENGTH:
    die("draft safetensors data extent does not equal the file extent")

assert_still_stable(opened)
session_root = SESSION_ROOT_TEMPLATE.format(uid=os.getuid())
session_parent_fd, session_fd, session_name, session_identity = acquire_session_root(session_root)
target_view_fd = create_view(
    session_fd, "target", [(filename, descriptor) for filename, descriptor in target_descriptors.items()]
)
draft_view_fd = create_view(
    session_fd,
    "draft",
    [("config.json", draft_config_fd), ("model.safetensors", draft_model_fd)],
)

argv = [
    BINARY_PATH,
    "serve",
    "--model-from-path",
    f"/proc/self/fd/{target_view_fd}",
    "--model-name",
    "qwen38-dflash2-official",
    "--port",
    "8977",
    "--bind",
    "127.0.0.1",
    "--kernel-target",
    "qwen3.8",
    "--gpu-memory-utilization",
    "0.55",
    "--kv-cache-dtype",
    "nvfp4",
    "--max-seq-len",
    "262144",
    "--max-prefill-tokens",
    "2047",
    "--max-batch-size",
    "1",
    "--max-num-seqs",
    "1",
    "--dflash",
    "--draft-model",
    f"/proc/self/fd/{draft_view_fd}",
    "--dflash-gamma",
    "7",
    "--dflash-quantization",
    "nvfp4",
    "--mtp-vocab",
    "248320",
    "--max-thinking-budget",
    "2048",
    "--request-timeout",
    "300",
    "--disable-confidence-early-stop",
    "--disable-simhash-watchdog",
    "--disable-loop-watchdog",
]
receipt = {
    "abi": abi,
    "argv": argv,
    "argv_sha256": hashlib.sha256(canonical(argv).encode()).hexdigest(),
    "drafter_semantics": {
        "is_causal": False,
        "layer_types": ["sliding_attention"] * 5,
        "rope_scaling": None,
        "rope_theta": 10000000,
    },
    "environment": EXEC_ENV,
    "environment_sha256": hashlib.sha256(canonical(EXEC_ENV).encode()).hexdigest(),
    "geometry": {
        "block_size": 8,
        "context_window": 2047,
        "gamma": 7,
        "k": 8,
        "sliding_window": 2048,
        "target_max_seq_len": 262144,
        "vocab_size": 248320,
    },
    "identity": manifest,
    "launcher_normalized_sha256": launcher_normalized_sha256,
    "schema": "atlas-qwen38-dflash2-official-launch-v2",
    "session_root": session_root,
    "status": "preflight-passed-launch-pending",
    "transport": "retained-fd-exec-and-model-view",
}
payload = (canonical(receipt) + "\n").encode()
write_receipt(session_fd, payload)
os.fchmod(session_fd, 0o500)

assert_still_stable(opened)
path_stat = os.stat(session_name, dir_fd=session_parent_fd, follow_symlinks=False)
if (path_stat.st_dev, path_stat.st_ino) != session_identity:
    die("fixed launch session identity changed before exec")
os.close(session_parent_fd)
print(
    "DFLASH2_OFFICIAL_LAUNCH_RECEIPT "
    f"path={session_root}/launch-receipt.json "
    f"sha256={hashlib.sha256(payload).hexdigest()} "
    "status=preflight-passed-launch-pending",
    file=sys.stderr,
    flush=True,
)
try:
    os.execve(binary_fd, argv, EXEC_ENV)
except OSError as error:
    die(f"retained-descriptor exec failed: {error}")
