#!/usr/bin/env python3
"""Export only the DeepSeek embedding/head needed by offline DFlash2 training."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


IDENTITY_FILES = (
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
EXPECTED = {
    "embed.weight": [129280, 4096],
    "head.weight": [129280, 4096],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]

    tensors = {}
    for key, expected_shape in EXPECTED.items():
        shard = source / weight_map[key]
        with safe_open(shard, framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(key)
        if list(tensor.shape) != expected_shape or str(tensor.dtype) != "torch.bfloat16":
            raise RuntimeError(
                f"{key}: got shape={list(tensor.shape)} dtype={tensor.dtype}, "
                f"expected shape={expected_shape} dtype=torch.bfloat16"
            )
        tensors[key] = tensor

    output.mkdir(parents=True, exist_ok=True)
    weights_name = "target-components.safetensors"
    save_file(tensors, output / weights_name)
    compact_index = {
        "metadata": {"total_size": sum(t.numel() * t.element_size() for t in tensors.values())},
        "weight_map": {key: weights_name for key in EXPECTED},
    }
    (output / "model.safetensors.index.json").write_text(
        json.dumps(compact_index, indent=2, sort_keys=True) + "\n"
    )
    # SpecForge only needs dimensions here, but AutoConfig must recognize the
    # model type before it can allocate the shared embedding/head. Preserve the
    # serving config separately and expose a Qwen3-compatible component config;
    # this does not describe or load the DeepSeek target body.
    target_config = json.loads((source / "config.json").read_text())
    (output / "deepseek_target_config.json").write_text(
        json.dumps(target_config, indent=2, sort_keys=True) + "\n"
    )
    component_config = {
        "architectures": ["Qwen3ForCausalLM"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": target_config["bos_token_id"],
        "eos_token_id": target_config["eos_token_id"],
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "max_position_embeddings": target_config["max_position_embeddings"],
        "model_type": "qwen3",
        "num_attention_heads": 32,
        "num_hidden_layers": 3,
        "num_key_value_heads": 8,
        "pad_token_id": target_config["eos_token_id"],
        "rms_norm_eps": 1e-6,
        "rope_theta": target_config["rope_theta"],
        "tie_word_embeddings": False,
        "vocab_size": 129280,
    }
    (output / "config.json").write_text(
        json.dumps(component_config, indent=2, sort_keys=True) + "\n"
    )
    for name in IDENTITY_FILES:
        path = source / name
        if path.is_file():
            shutil.copy2(path, output / name)

    print(output / weights_name)


if __name__ == "__main__":
    main()
