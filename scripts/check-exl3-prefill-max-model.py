# SPDX-License-Identifier: AGPL-3.0-only

"""Fail-closed CPU preflight for the exact DeepSeek V4 EXL3 K2 profile."""

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import tomllib


SCHEMA = "atlas-exl3-prefill-max-model-v1"
EXPECTED_MODEL = Path(
    "/home/flocka/models/DeepSeek-V4-Flash-0731-EXL3-K2-calibrated-v1"
)
EXPECTED_TENSOR_COUNT = 142_973
EXPECTED_INDEX_TOTAL_SIZE = 83_963_579_128
EXPECTED_SHARDS = tuple(
    f"model-{number:05d}-of-00010.safetensors" for number in range(1, 11)
)
EXPERT_PARTS = ("mcg", "suh", "svh", "trellis")
EXPERT_PROJECTIONS = ("down_proj", "gate_proj", "up_proj")
MAX_CONFIG_BYTES = 1 << 20
MAX_INDEX_BYTES = 32 << 20
MAX_TOML_BYTES = 1 << 20
MAX_LAUNCHER_BYTES = 1 << 20

CONFIG_IDENTITY = {
    "architectures": ["DeepseekV4ForCausalLM"],
    "expert_dtype": "fp4",
    "head_dim": 512,
    "hidden_size": 4096,
    "model_type": "deepseek_v4",
    "moe_intermediate_size": 2048,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "num_attention_heads": 64,
    "num_experts_per_tok": 6,
    "num_hidden_layers": 43,
    "num_key_value_heads": 1,
    "o_lora_rank": 1024,
    "q_lora_rank": 1024,
    "qk_rope_head_dim": 64,
    "torch_dtype": "bfloat16",
    "vocab_size": 129280,
}

QUANT_IDENTITY = {
    "bits": 2.0,
    "checkpoint_format": "exl3",
    "codebook": "mcg",
    "format": "exl3",
    "group_size": -1,
    "method": "exl3",
    "quant_method": "exl3",
}

MODEL_TOML_IDENTITY = {
    "name": "deepseek-v4-flash",
    "layers_total": 43,
    "hidden_dim": 4096,
    "head_dim": 512,
    "q_heads": 64,
    "kv_heads": 1,
    "intermediate_size": 2048,
    "vocab_size": 129280,
    "kv_lora_rank": 512,
    "q_lora_rank": 1024,
    "o_lora_rank": 1024,
    "v_head_dim": 512,
    "qk_rope_head_dim": 64,
    "qk_nope_head_dim": 448,
    "num_experts": 256,
    "num_shared_experts": 1,
    "moe_intermediate_size": 2048,
    "top_k": 6,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_regular_file(path: Path, maximum: int, label: str) -> bytes:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise ValueError(f"{label} is missing") from error
    if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file")
    if metadata.st_size > maximum:
        raise ValueError(f"{label} exceeds the bounded preflight size")
    return path.read_bytes()


def parse_json(data: bytes, label: str) -> dict:
    try:
        value = json.loads(data)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is malformed") from error
    if type(value) is not dict:
        raise ValueError(f"{label} root must be an object")
    return value


def parse_model_toml(data: bytes) -> dict:
    try:
        value = tomllib.loads(data.decode("utf-8"))
    except (UnicodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError("MODEL.toml is malformed") from error
    if type(value) is not dict:
        raise ValueError("MODEL.toml root must be a table")
    return value


def require_identity(actual: dict, expected: dict, label: str) -> None:
    if type(actual) is not dict:
        raise ValueError(f"{label} must be an object")
    for key, expected_value in expected.items():
        actual_value = actual.get(key)
        if type(actual_value) is not type(expected_value) or actual_value != expected_value:
            raise ValueError(f"{label}.{key} does not match the exact profile")


def validate_config(config: dict) -> dict:
    require_identity(config, CONFIG_IDENTITY, "config")
    quantization = config.get("quantization_config")
    require_identity(quantization, QUANT_IDENTITY, "config.quantization_config")
    return {
        "layers": config["num_hidden_layers"],
        "num_experts": config["n_routed_experts"],
        "top_k": config["num_experts_per_tok"],
        "exl3_bits": quantization["bits"],
    }


def validate_model_toml(registry: dict) -> int:
    model = registry.get("model")
    require_identity(model, MODEL_TOML_IDENTITY, "MODEL.toml.model")
    model_types = registry.get("model_types")
    if type(model_types) is not list or model_types != [{"model_type": "deepseek_v4"}]:
        raise ValueError("MODEL.toml.model_types does not match the exact profile")
    behavior = registry.get("behavior")
    if type(behavior) is not dict or behavior.get("default_kv_dtype") != "fp8":
        raise ValueError("MODEL.toml behavior.default_kv_dtype must be fp8")
    calibration_tokens = behavior.get("fp8_kv_calibration_tokens")
    if type(calibration_tokens) is not int or calibration_tokens <= 0:
        raise ValueError("MODEL.toml FP8 KV calibration tokens must be a positive integer")
    return calibration_tokens


def require_canonical_expected_model(model: Path) -> Path:
    if not model.is_absolute():
        raise ValueError("model must use the canonical expected path")
    try:
        resolved = model.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError("canonical expected model directory is missing") from error
    if resolved != model or not model.is_dir() or model.is_symlink():
        raise ValueError("model must use the canonical expected path without symlinks")
    return resolved


def launcher_model_path(launcher_data: bytes) -> Path:
    try:
        source = launcher_data.decode("utf-8")
    except UnicodeError as error:
        raise ValueError("max-prefill launcher is not UTF-8") from error
    matches = re.findall(r"^export MODEL=([^\s]+)$", source, flags=re.MULTILINE)
    if len(matches) != 1 or not re.fullmatch(r"/[A-Za-z0-9._/-]+", matches[0]):
        raise ValueError("max-prefill launcher MODEL must be one literal absolute path")
    return Path(matches[0])


def validate_index(index: dict, model: Path) -> dict:
    metadata = index.get("metadata")
    require_identity(
        metadata,
        {"total_size": EXPECTED_INDEX_TOTAL_SIZE},
        "model index metadata",
    )
    weight_map = index.get("weight_map")
    if type(weight_map) is not dict or len(weight_map) != EXPECTED_TENSOR_COUNT:
        raise ValueError("model index tensor count does not match the exact checkpoint")

    shard_names = set()
    k_scale_tensors = 0
    v_scale_tensors = 0
    for tensor_name, shard_name in weight_map.items():
        if type(tensor_name) is not str or not tensor_name:
            raise ValueError("model index contains a malformed tensor name")
        if type(shard_name) is not str or not re.fullmatch(
            r"model-[0-9]{5}-of-[0-9]{5}\.safetensors", shard_name
        ):
            raise ValueError("model index contains a non-safe shard basename")
        shard_names.add(shard_name)
        k_scale_tensors += tensor_name.endswith(".k_scale")
        v_scale_tensors += tensor_name.endswith(".v_scale")

    if shard_names != set(EXPECTED_SHARDS):
        raise ValueError("model index shard set does not match the exact checkpoint")
    if k_scale_tensors != 0 or v_scale_tensors != 0:
        raise ValueError("exact K2 checkpoint must contain zero K/V scale tensors")

    for layer in range(43):
        for expert in range(256):
            prefix = f"model.layers.{layer}.mlp.experts.{expert}"
            for projection in EXPERT_PROJECTIONS:
                for component in EXPERT_PARTS:
                    if f"{prefix}.{projection}.{component}" not in weight_map:
                        raise ValueError("model index EXL3 expert matrix is incomplete")

    shard_stats = []
    total_shard_bytes = 0
    for shard_name in EXPECTED_SHARDS:
        shard_path = model / shard_name
        try:
            shard_stat = os.lstat(shard_path)
        except FileNotFoundError as error:
            raise ValueError("model index referenced shard is missing") from error
        if not stat.S_ISREG(shard_stat.st_mode) or shard_path.is_symlink():
            raise ValueError("model shard must be a regular non-symlink file")
        if shard_stat.st_size <= 0:
            raise ValueError("model shard must be non-empty")
        total_shard_bytes += shard_stat.st_size
        shard_stats.append(f"{shard_name}:{shard_stat.st_size}")

    stat_manifest = hashlib.sha256("\n".join(shard_stats).encode()).hexdigest()
    return {
        "tensor_count": len(weight_map),
        "shard_count": len(shard_names),
        "shard_bytes": total_shard_bytes,
        "shard_stat_manifest_sha256": stat_manifest,
        "k_scale_tensors": k_scale_tensors,
        "v_scale_tensors": v_scale_tensors,
    }


def build_report(repo_root: Path, expected_model: Path = EXPECTED_MODEL) -> dict:
    repo_root = repo_root.resolve(strict=True)
    model = require_canonical_expected_model(expected_model)
    launcher_path = repo_root / "scripts/exl3-prefill-max.sh"
    config_path = model / "config.json"
    index_path = model / "model.safetensors.index.json"
    parent_toml_path = repo_root / "kernels/gb10/deepseek-v4-flash/MODEL.toml"
    nvfp4_toml_path = parent_toml_path.parent / "nvfp4/MODEL.toml"

    launcher_data = read_regular_file(
        launcher_path, MAX_LAUNCHER_BYTES, "max-prefill launcher"
    )
    if launcher_model_path(launcher_data) != model:
        raise ValueError("max-prefill launcher does not select the expected checkpoint")

    config_data = read_regular_file(config_path, MAX_CONFIG_BYTES, "model config.json")
    index_data = read_regular_file(index_path, MAX_INDEX_BYTES, "model index")
    parent_toml_data = read_regular_file(
        parent_toml_path, MAX_TOML_BYTES, "parent MODEL.toml"
    )
    nvfp4_toml_data = read_regular_file(
        nvfp4_toml_path, MAX_TOML_BYTES, "NVFP4 MODEL.toml"
    )

    config_profile = validate_config(parse_json(config_data, "config.json"))
    index_profile = validate_index(parse_json(index_data, "model index"), model)
    parent_registry = parse_model_toml(parent_toml_data)
    nvfp4_registry = parse_model_toml(nvfp4_toml_data)
    if parent_registry != nvfp4_registry:
        raise ValueError("parent and NVFP4 MODEL.toml behavior registries diverge")
    calibration_tokens = validate_model_toml(nvfp4_registry)
    if (
        index_profile["k_scale_tensors"] == 0
        and index_profile["v_scale_tensors"] == 0
        and calibration_tokens <= 0
    ):
        raise ValueError("zero checkpoint K/V scales require positive FP8 calibration")

    return {
        "schema": SCHEMA,
        "status": "ok",
        "checkpoint": {
            "path": str(model),
            "config_sha256": sha256_file(config_path),
            "index_sha256": sha256_file(index_path),
            "tensor_count": index_profile["tensor_count"],
            "shard_count": index_profile["shard_count"],
            "shard_bytes": index_profile["shard_bytes"],
            "shard_stat_manifest_sha256": index_profile[
                "shard_stat_manifest_sha256"
            ],
        },
        "kernel_target": {
            "model_toml_sha256": sha256_file(nvfp4_toml_path),
            "parent_model_toml_sha256": sha256_file(parent_toml_path),
        },
        "launcher_sha256": sha256_file(launcher_path),
        "profile": config_profile,
        "kv": {
            "k_scale_tensors": index_profile["k_scale_tensors"],
            "v_scale_tensors": index_profile["v_scale_tensors"],
            "calibration_tokens": calibration_tokens,
        },
    }


def main() -> int:
    if len(sys.argv) != 1:
        error = "this exact-profile preflight accepts no path overrides"
        print(json.dumps({"schema": SCHEMA, "status": "error", "error": error}))
        return 2
    try:
        repo_root = Path(__file__).resolve(strict=True).parent.parent
        report = build_report(repo_root)
    except (OSError, ValueError) as error:
        message = " ".join(str(error).split())[:512]
        print(
            json.dumps(
                {"schema": SCHEMA, "status": "error", "error": message},
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(report, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
