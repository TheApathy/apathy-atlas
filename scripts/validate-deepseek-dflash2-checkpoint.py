#!/usr/bin/env python3
"""CPU-only ABI gate for a trained Atlas-native DeepSeek DFlash2 checkpoint."""

import argparse
import json
import pathlib


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=pathlib.Path)
    parser.add_argument("--report", type=pathlib.Path)
    return parser.parse_args()


def tensor_shapes(checkpoint: pathlib.Path) -> dict[str, tuple[list[int], str]]:
    from safetensors import safe_open

    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        shards = sorted(set(index["weight_map"].values()))
    else:
        shards = sorted(path.name for path in checkpoint.glob("*.safetensors"))
    if not shards:
        raise RuntimeError(f"no safetensors weights in {checkpoint}")
    result = {}
    for shard_name in shards:
        shard = checkpoint / shard_name
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in result:
                    raise RuntimeError(f"duplicate tensor key across shards: {key}")
                result[key] = (
                    list(handle.get_slice(key).get_shape()),
                    str(handle.get_slice(key).get_dtype()),
                )
    return result


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
    config_path = args.checkpoint / "config.json"
    if not config_path.is_file():
        raise RuntimeError(f"missing {config_path}")
    config = json.loads(config_path.read_text())
    report = validate_abi(config, tensor_shapes(args.checkpoint))
    report["checkpoint"] = str(args.checkpoint.resolve())
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
