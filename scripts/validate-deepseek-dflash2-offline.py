#!/usr/bin/env python3
"""Validate an exact SpecForge offline-hidden corpus without using a GPU."""

import argparse
import hashlib
import json
import os
import pathlib
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specforge-dir", type=pathlib.Path, required=True)
    parser.add_argument("--draft-config", type=pathlib.Path, required=True)
    parser.add_argument("--target-components", type=pathlib.Path, required=True)
    parser.add_argument("--corpus", type=pathlib.Path, required=True)
    parser.add_argument("--hidden-dir", type=pathlib.Path, required=True)
    parser.add_argument("--cache-dir", type=pathlib.Path, required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--min-rows", type=int, default=128)
    parser.add_argument(
        "--limit",
        type=int,
        default=128,
        help="validate the same post-filter deterministic prefix captured for training",
    )
    parser.add_argument("--chat-template", default="deepseek-v3")
    parser.add_argument("--is-preformatted", action="store_true")
    parser.add_argument("--num-proc", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_length <= 0 or args.min_rows <= 0 or args.num_proc <= 0:
        raise RuntimeError("max-length, min-rows, and num-proc must be positive")
    if args.limit < 0:
        raise RuntimeError("limit must be non-negative")
    os.environ["SPECFORGE_PAD_TO"] = str(args.max_length)
    sys.path.insert(0, str(args.specforge_dir))

    import torch
    from datasets import load_dataset
    from safetensors import safe_open
    from transformers import AutoTokenizer

    from specforge.data import build_eagle3_dataset

    config = json.loads(args.draft_config.read_text())
    layer_ids = config["dflash_config"]["target_layer_ids"]
    hidden_size = int(config["hidden_size"])
    expected_width = len(layer_ids) * hidden_size
    if layer_ids != [1, 11, 22, 32, 43] or hidden_size != 4096:
        raise RuntimeError(
            f"unexpected DeepSeek capture ABI: layers={layer_ids}, hidden={hidden_size}"
        )

    index_path = args.target_components / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    for key in ("embed.weight", "head.weight"):
        shard = args.target_components / index["weight_map"][key]
        with safe_open(shard, framework="pt", device="cpu") as handle:
            tensor = handle.get_slice(key)
            shape = list(tensor.get_shape())
            dtype = str(tensor.get_dtype())
            if shape != [129280, 4096] or dtype != "BF16":
                raise RuntimeError(
                    f"invalid target component {key}: shape={shape} dtype={dtype}"
                )

    tokenizer = AutoTokenizer.from_pretrained(
        args.target_components, local_files_only=True
    )
    raw = load_dataset("json", data_files=str(args.corpus))["train"]
    cache_contract = (
        f"{args.corpus}-{args.max_length}-{args.chat_template}-"
        f"{args.target_components}"
    )
    cache_key = hashlib.md5(cache_contract.encode()).hexdigest()
    dataset = build_eagle3_dataset(
        dataset=raw,
        tokenizer=tokenizer,
        chat_template=args.chat_template,
        max_length=args.max_length,
        is_preformatted=args.is_preformatted,
        cache_dir=str(args.cache_dir / "processed_dataset"),
        cache_key=cache_key,
        num_proc=args.num_proc,
    )
    dataset = dataset.filter(lambda row: row["loss_mask"].sum() >= 32)
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    if len(dataset) < args.min_rows:
        raise RuntimeError(
            f"only {len(dataset)} usable corpus rows; need at least {args.min_rows}"
        )

    missing = []
    checked = set()
    for row in dataset:
        # SpecForge keys the collated row, not the variable-length dataset row.
        # Batch size and sequence-parallel degree are one in both precompute and
        # the paid contract, so its collator pads directly to SPECFORGE_PAD_TO.
        raw_ids = row["input_ids"]
        if raw_ids.ndim == 1:
            raw_ids = raw_ids.unsqueeze(0)
        if raw_ids.shape[1] > args.max_length:
            raise RuntimeError(
                f"processed row length {raw_ids.shape[1]} exceeds {args.max_length}"
            )
        input_ids = torch.cat(
            [
                raw_ids,
                torch.zeros(
                    (1, args.max_length - raw_ids.shape[1]), dtype=raw_ids.dtype
                ),
            ],
            dim=1,
        )[0]
        key = hashlib.md5(input_ids.numpy().tobytes()).hexdigest()
        path = args.hidden_dir / f"{key}.pt"
        if not path.is_file():
            missing.append(key)
            continue
        if key in checked:
            continue
        hidden = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        expected_shape = (input_ids.numel(), expected_width)
        if hidden.dtype != torch.bfloat16 or tuple(hidden.shape) != expected_shape:
            raise RuntimeError(
                f"{path.name}: got dtype={hidden.dtype} shape={tuple(hidden.shape)}, "
                f"expected bf16 {expected_shape}"
            )
        checked.add(key)

    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(
            f"{len(missing)}/{len(dataset)} corpus rows lack keyed hidden states; "
            f"first keys: {preview}"
        )
    if len(checked) < args.min_rows:
        raise RuntimeError(
            f"only {len(checked)} unique keyed hidden rows; need at least "
            f"{args.min_rows} (corpus rows={len(dataset)})"
        )
    orphan_count = sum(
        1 for path in args.hidden_dir.glob("*.pt") if path.stem not in checked
    )
    print(
        f"offline corpus OK: rows={len(dataset)} unique_hidden={len(checked)} "
        f"width={expected_width} orphan_files={orphan_count} cache_key={cache_key}"
    )


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
