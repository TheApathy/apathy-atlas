#!/usr/bin/env python3
"""CPU-only ABI gate for a trained Atlas-native DeepSeek DFlash2 checkpoint."""

import argparse
import hashlib
import json
import os
import pathlib
import tempfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=pathlib.Path)
    parser.add_argument("--report", type=pathlib.Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def tensor_shapes(checkpoint: pathlib.Path) -> dict[str, tuple[list[int], str]]:
    from safetensors import safe_open

    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise RuntimeError("safetensors index has no non-empty weight_map")
        for key, shard_name in weight_map.items():
            if not isinstance(key, str) or not key:
                raise RuntimeError(f"invalid weight-map tensor key: {key!r}")
            if not isinstance(shard_name, str) or not shard_name:
                raise RuntimeError(f"invalid shard name for {key}: {shard_name!r}")
            shard_path = pathlib.PurePosixPath(shard_name)
            if shard_path.is_absolute() or ".." in shard_path.parts:
                raise RuntimeError(f"unsafe shard path for {key}: {shard_name}")
        shards = sorted(set(weight_map.values()))
    else:
        weight_map = None
        shards = sorted(path.name for path in checkpoint.glob("*.safetensors"))
    if not shards:
        raise RuntimeError(f"no safetensors weights in {checkpoint}")
    result = {}
    actual_shards = {}
    for shard_name in shards:
        shard = checkpoint / shard_name
        if not shard.is_file():
            raise RuntimeError(f"missing safetensors shard: {shard}")
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in result:
                    raise RuntimeError(f"duplicate tensor key across shards: {key}")
                result[key] = (
                    list(handle.get_slice(key).get_shape()),
                    str(handle.get_slice(key).get_dtype()),
                )
                actual_shards[key] = shard_name
    if weight_map is not None:
        if set(weight_map) != set(result):
            missing = sorted(set(weight_map) - set(result))
            undeclared = sorted(set(result) - set(weight_map))
            raise RuntimeError(
                "safetensors index/tensor keys differ: "
                f"missing={missing[:5]} undeclared={undeclared[:5]}"
            )
        wrong_shards = [
            key for key, shard_name in weight_map.items()
            if actual_shards[key] != shard_name
        ]
        if wrong_shards:
            raise RuntimeError(
                f"safetensors index maps tensors to wrong shards: {wrong_shards[:5]}"
            )
    return result


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: pathlib.Path, payload: dict) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = pathlib.Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_abi(
    config: dict, shapes: dict[str, tuple[list[int], str]]
) -> dict:
    expected_config = {
        "hidden_size": 4096,
        "vocab_size": 129280,
        "num_target_layers": 43,
        "num_hidden_layers": 3,
        "block_size": 16,
        "intermediate_size": 11008,
        "head_dim": 128,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
    }
    for key, expected in expected_config.items():
        actual = config.get(key)
        if actual != expected:
            raise RuntimeError(f"config {key}={actual!r}, expected {expected!r}")
    dflash = config.get("dflash_config") or {}
    if dflash.get("target_layer_ids") != [1, 11, 22, 32, 43]:
        raise RuntimeError(
            f"invalid target_layer_ids: {dflash.get('target_layer_ids')}"
        )
    if dflash.get("mask_token_id") != 129000:
        raise RuntimeError(f"invalid mask_token_id: {dflash.get('mask_token_id')}")
    if dflash.get("causal") is not False:
        raise RuntimeError(f"DFlash attention must be non-causal: {dflash!r}")

    has_prefixed = "model.fc.weight" in shapes
    has_plain = "fc.weight" in shapes
    if has_prefixed == has_plain:
        raise RuntimeError(
            "checkpoint must contain exactly one of fc.weight or model.fc.weight"
        )
    prefix = "model." if has_prefixed else ""
    hidden = config["hidden_size"]
    intermediate = config["intermediate_size"]
    head_dim = config["head_dim"]
    q_width = config["num_attention_heads"] * head_dim
    kv_width = config["num_key_value_heads"] * head_dim
    required = {
        f"{prefix}fc.weight": [hidden, len(dflash["target_layer_ids"]) * hidden],
        f"{prefix}hidden_norm.weight": [hidden],
        f"{prefix}norm.weight": [hidden],
    }
    for layer in range(config["num_hidden_layers"]):
        base = f"{prefix}layers.{layer}"
        required.update(
            {
                f"{base}.input_layernorm.weight": [hidden],
                f"{base}.post_attention_layernorm.weight": [hidden],
                f"{base}.self_attn.q_proj.weight": [q_width, hidden],
                f"{base}.self_attn.k_proj.weight": [kv_width, hidden],
                f"{base}.self_attn.v_proj.weight": [kv_width, hidden],
                f"{base}.self_attn.o_proj.weight": [hidden, q_width],
                f"{base}.self_attn.q_norm.weight": [head_dim],
                f"{base}.self_attn.k_norm.weight": [head_dim],
                f"{base}.mlp.gate_proj.weight": [intermediate, hidden],
                f"{base}.mlp.up_proj.weight": [intermediate, hidden],
                f"{base}.mlp.down_proj.weight": [hidden, intermediate],
            }
        )
    for key, expected_shape in required.items():
        if key not in shapes:
            raise RuntimeError(f"missing required drafter tensor: {key}")
        actual_shape, dtype = shapes[key]
        if actual_shape != expected_shape:
            raise RuntimeError(f"{key} shape={actual_shape}, expected={expected_shape}")
        if dtype not in {"BF16", "F16", "F32"}:
            raise RuntimeError(f"{key} has unsupported training dtype {dtype}")

    return {
        "tensor_count": len(shapes),
        "required_tensor_count": len(required),
        "prefix": prefix,
        "target_hidden_size": hidden,
        "capture_width": len(dflash["target_layer_ids"]) * hidden,
        "target_layers": dflash["target_layer_ids"],
        "block_size": config["block_size"],
        "status": "ok",
    }


def main() -> None:
    args = parse_args()
    if args.report and args.report.exists() and not args.overwrite:
        raise RuntimeError(f"refusing to overwrite existing report: {args.report}")
    config_path = args.checkpoint / "config.json"
    if not config_path.is_file():
        raise RuntimeError(f"missing {config_path}")
    config = json.loads(config_path.read_text())
    report = validate_abi(config, tensor_shapes(args.checkpoint))
    report["checkpoint"] = str(args.checkpoint.resolve())
    artifacts = [config_path]
    index_path = args.checkpoint / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        artifacts.append(index_path)
        artifacts.extend(
            args.checkpoint / name for name in sorted(set(index["weight_map"].values()))
        )
    else:
        artifacts.extend(sorted(args.checkpoint.glob("*.safetensors")))
    report["artifact_sha256"] = {
        path.relative_to(args.checkpoint).as_posix(): sha256_file(path)
        for path in artifacts
    }
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        atomic_json(args.report, report)
    print(output, end="")


if __name__ == "__main__":
    main()
