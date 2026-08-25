#!/usr/bin/env python3
"""CPU-streamed value parity for compact DeepSeek DFlash2 target components."""

import argparse
import hashlib
import json
import pathlib
import tempfile


EXPECTED = {
    "embed.weight": [129280, 4096],
    "head.weight": [129280, 4096],
}
IDENTICAL_METADATA = ("tokenizer.json", "tokenizer_config.json", "generation_config.json")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_digest(path: pathlib.Path, key: str, chunk_rows: int = 256) -> tuple[list[int], str, str]:
    import torch
    from safetensors import safe_open

    digest = hashlib.sha256()
    with safe_open(path, framework="pt", device="cpu") as handle:
        tensor = handle.get_slice(key)
        shape = list(tensor.get_shape())
        dtype = str(tensor.get_dtype())
        for start in range(0, shape[0], chunk_rows):
            chunk = tensor[start : start + chunk_rows]
            digest.update(chunk.contiguous().view(torch.uint8).numpy().tobytes())
    return shape, dtype, digest.hexdigest()


def atomic_json(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        temporary = pathlib.Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def validate_metadata(source: pathlib.Path, components: pathlib.Path) -> dict:
    hashes = {}
    for name in IDENTICAL_METADATA:
        source_hash = sha256_file(source / name)
        component_hash = sha256_file(components / name)
        if source_hash != component_hash:
            raise RuntimeError(f"component {name} differs from source")
        hashes[name] = component_hash

    source_config_hash = sha256_file(source / "config.json")
    preserved_config = components / "deepseek_target_config.json"
    if source_config_hash != sha256_file(preserved_config):
        raise RuntimeError("deepseek_target_config.json differs from source config.json")
    hashes["deepseek_target_config.json"] = source_config_hash

    component_config_path = components / "config.json"
    component_config = json.loads(component_config_path.read_text())
    expected_fields = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": 4096,
        "vocab_size": 129280,
        "num_hidden_layers": 3,
        "intermediate_size": 11008,
        "head_dim": 128,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
    }
    for key, expected in expected_fields.items():
        if component_config.get(key) != expected:
            raise RuntimeError(
                f"component config {key}={component_config.get(key)!r}, "
                f"expected {expected!r}"
            )
    hashes["config.json"] = sha256_file(component_config_path)
    return hashes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=pathlib.Path)
    parser.add_argument("components", type=pathlib.Path)
    parser.add_argument("--report", type=pathlib.Path, required=True)
    parser.add_argument("--chunk-rows", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.chunk_rows <= 0:
        raise RuntimeError("chunk-rows must be positive")
    if args.report.exists() and not args.overwrite:
        raise RuntimeError(f"refusing to overwrite existing report: {args.report}")

    source_index_path = args.source / "model.safetensors.index.json"
    component_index_path = args.components / "model.safetensors.index.json"
    source_index = json.loads(source_index_path.read_text())
    component_index = json.loads(component_index_path.read_text())
    tensors = {}
    for key, expected_shape in EXPECTED.items():
        source_path = args.source / source_index["weight_map"][key]
        component_path = args.components / component_index["weight_map"][key]
        source_meta = tensor_digest(source_path, key, args.chunk_rows)
        component_meta = tensor_digest(component_path, key, args.chunk_rows)
        if source_meta[:2] != (expected_shape, "BF16"):
            raise RuntimeError(f"source {key} has unexpected ABI: {source_meta[:2]}")
        if component_meta[:2] != (expected_shape, "BF16"):
            raise RuntimeError(f"component {key} has unexpected ABI: {component_meta[:2]}")
        if source_meta[2] != component_meta[2]:
            raise RuntimeError(f"component value hash differs for {key}")
        tensors[key] = {
            "shape": expected_shape,
            "dtype": "BF16",
            "value_sha256": source_meta[2],
        }

    metadata_hashes = validate_metadata(args.source, args.components)
    report = {
        "format": "atlas-deepseek-dflash2-components-v1",
        "source_index_sha256": sha256_file(source_index_path),
        "component_index_sha256": sha256_file(component_index_path),
        "metadata_sha256": metadata_hashes,
        "tensors": tensors,
        "status": "ok",
    }
    atomic_json(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
