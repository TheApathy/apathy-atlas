#!/usr/bin/env python3
"""Drive Atlas with exact token IDs and materialize SpecForge hidden rows.

Atlas's DeepSeek-V4 target keeps the mHC residual as four FP32 streams between
layers.  ``ATLAS_DSPARK_DUMP`` is therefore the authoritative way to collapse
those streams to the 4096-wide target hidden expected by a drafter.  This tool
turns a five-layer dump into SpecForge's keyed, token-major BF16 tensors while
the requests are run serially against a plain Atlas server.
"""

import argparse
import hashlib
import json
import os
import pathlib
import struct
import sys
import tempfile
import time
import urllib.error
import urllib.request


MAGIC = 0x4453504B
HEADER = struct.Struct("<7I")
EXPECTED_MODEL_LAYERS = [0, 10, 21, 31, 42]
CACHE_CONTRACT_VERSION = "deepseek-dflash2-preprocess-v2"


def file_sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def preprocessing_cache_key(
    corpus: pathlib.Path,
    target_components: pathlib.Path,
    preprocessing_source: pathlib.Path,
    max_length: int,
    chat_template: str,
    is_preformatted: bool,
) -> str:
    contract = {
        "version": CACHE_CONTRACT_VERSION,
        "corpus_sha256": file_sha256(corpus),
        "tokenizer_sha256": file_sha256(target_components / "tokenizer.json"),
        "tokenizer_config_sha256": file_sha256(
            target_components / "tokenizer_config.json"
        ),
        "preprocessing_sha256": file_sha256(preprocessing_source),
        "max_length": max_length,
        "chat_template": chat_template,
        "is_preformatted": is_preformatted,
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--specforge-dir", type=pathlib.Path, required=True)
    parser.add_argument("--target-components", type=pathlib.Path, required=True)
    parser.add_argument("--corpus", type=pathlib.Path, required=True)
    parser.add_argument("--dump", type=pathlib.Path, required=True)
    parser.add_argument("--hidden-dir", type=pathlib.Path, required=True)
    parser.add_argument("--cache-dir", type=pathlib.Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8977/v1/completions")
    parser.add_argument("--model", default="deepseek-v4")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--min-loss-tokens", type=int, default=32)
    parser.add_argument(
        "--limit",
        type=int,
        default=128,
        help="post-filter deterministic row prefix (0 means every usable row)",
    )
    parser.add_argument("--chat-template", default="deepseek-v3")
    parser.add_argument("--is-preformatted", action="store_true")
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--read-timeout", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse shape-checked keyed rows from an interrupted capture",
    )
    parser.add_argument(
        "--capture-layers",
        default=",".join(map(str, EXPECTED_MODEL_LAYERS)),
        help="Atlas model-layer indices used by ATLAS_DSPARK_CAPTURE_LAYERS",
    )
    return parser.parse_args()


def read_exact(handle, size: int, timeout: float) -> bytes:
    deadline = time.monotonic() + timeout
    chunks = bytearray()
    while len(chunks) < size:
        part = handle.read(size - len(chunks))
        if part:
            chunks.extend(part)
            continue
        if time.monotonic() >= deadline:
            raise RuntimeError(f"timed out after {len(chunks)}/{size} dump bytes")
        time.sleep(0.05)
    return bytes(chunks)


def post_tokens(url: str, model: str, token_ids: list[int]) -> None:
    payload = json.dumps(
        {
            "model": model,
            "prompt": token_ids,
            "max_tokens": 1,
            "temperature": 0,
            "stream": False,
        }
    ).encode()
    request = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            body = response.read()
            if response.status != 200:
                raise RuntimeError(
                    f"Atlas returned HTTP {response.status}: {body[:500]!r}"
                )
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            f"Atlas returned HTTP {exc.code}: {exc.read()[:500]!r}"
        ) from exc


def save_tensor_atomic(torch, tensor, output: pathlib.Path) -> None:
    """Publish a capture row only after torch.save has completed successfully."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=output.parent, prefix=f".{output.name}.", suffix=".tmp", delete=False
    ) as handle:
        temporary = pathlib.Path(handle.name)
    try:
        torch.save(tensor, temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def read_prefill_row(
    dump,
    torch,
    token_count: int,
    max_length: int,
    capture_layers: list[int],
    timeout: float,
    row_index: int,
):
    hidden_size = 4096
    hidden = torch.zeros(
        (max_length, len(capture_layers), hidden_size), dtype=torch.bfloat16
    )
    captured = 0
    while captured < token_count:
        header = HEADER.unpack(read_exact(dump, HEADER.size, timeout))
        magic, kind, start, count, record_hidden, layers, _token = header
        if magic != MAGIC:
            raise RuntimeError(f"row {row_index}: bad dump magic 0x{magic:08x}")
        if kind != 0 or start != captured:
            raise RuntimeError(
                f"row {row_index}: expected prefill start {captured}, "
                f"got kind={kind} start={start}; capture server was contaminated"
            )
        if record_hidden != hidden_size or layers != len(capture_layers):
            raise RuntimeError(
                f"row {row_index}: dump ABI is h={record_hidden} layers={layers}, "
                f"expected h={hidden_size} layers={len(capture_layers)}"
            )
        if count == 0 or captured + count > token_count:
            raise RuntimeError(
                f"row {row_index}: invalid chunk count {count} at {captured}/"
                f"{token_count}"
            )
        layer_bytes = count * hidden_size * 2
        for slot in range(layers):
            data = bytearray(read_exact(dump, layer_bytes, timeout))
            values = torch.frombuffer(data, dtype=torch.bfloat16).reshape(
                count, hidden_size
            )
            hidden[captured : captured + count, slot, :].copy_(values)
        captured += count
    return hidden


def main() -> None:
    args = parse_args()
    capture_layers = [int(value) for value in args.capture_layers.split(",")]
    if capture_layers != EXPECTED_MODEL_LAYERS:
        raise RuntimeError(
            f"capture layers must be {EXPECTED_MODEL_LAYERS}, got {capture_layers}; "
            "these are Atlas model indices for HF hidden IDs [1,11,22,32,43]"
        )
    if args.max_length <= 0:
        raise RuntimeError("max-length must be positive")
    if args.limit < 0:
        raise RuntimeError("limit must be non-negative")
    if args.min_loss_tokens <= 0:
        raise RuntimeError("min-loss-tokens must be positive")
    if args.num_proc <= 0:
        raise RuntimeError("num-proc must be positive")
    if args.read_timeout <= 0:
        raise RuntimeError("read-timeout must be positive")
    if not args.dry_run and (not args.dump.is_file() or args.dump.stat().st_size != 0):
        raise RuntimeError(
            f"dump must exist and be empty before capture: {args.dump}; "
            "start a fresh capture server and do not share it with other requests"
        )

    os.environ["SPECFORGE_PAD_TO"] = str(args.max_length)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    sys.path.insert(0, str(args.specforge_dir))

    import torch
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from specforge.data import build_eagle3_dataset

    tokenizer = AutoTokenizer.from_pretrained(
        args.target_components, local_files_only=True
    )
    raw = load_dataset("json", data_files=str(args.corpus))["train"]
    cache_key = preprocessing_cache_key(
        args.corpus,
        args.target_components,
        args.specforge_dir / "specforge/data/preprocessing.py",
        args.max_length,
        args.chat_template,
        args.is_preformatted,
    )
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
    dataset = dataset.filter(lambda row: row["loss_mask"].sum() >= args.min_loss_tokens)
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
    if not len(dataset):
        raise RuntimeError("processed corpus has no usable rows")

    if args.dry_run:
        keys = set()
        token_counts = []
        for row in dataset:
            raw_ids = row["input_ids"]
            if raw_ids.ndim == 2:
                raw_ids = raw_ids[0]
            raw_ids = raw_ids.to(dtype=torch.int64, device="cpu").contiguous()
            if raw_ids.numel() > args.max_length:
                raise RuntimeError(
                    f"processed row has {raw_ids.numel()} tokens, max is {args.max_length}"
                )
            padded_ids = torch.cat(
                [
                    raw_ids,
                    torch.zeros(args.max_length - raw_ids.numel(), dtype=raw_ids.dtype),
                ]
            )
            keys.add(hashlib.md5(padded_ids.numpy().tobytes()).hexdigest())
            token_counts.append(int(raw_ids.numel()))
        gib = len(keys) * args.max_length * len(capture_layers) * 4096 * 2 / 2**30
        print(
            f"capture dry-run OK: rows={len(dataset)} unique={len(keys)} "
            f"duplicates={len(dataset) - len(keys)} token_min={min(token_counts)} "
            f"token_max={max(token_counts)} padded_hidden_gib={gib:.2f} "
            f"cache_key={cache_key}"
        )
        return

    args.hidden_dir.mkdir(parents=True, exist_ok=True)
    width = len(capture_layers) * 4096
    manifest_path = args.hidden_dir / "capture-manifest.jsonl"
    manifest_mode = "a" if args.resume else "x"
    created_keys = set()
    with (
        args.dump.open("rb", buffering=0) as dump,
        manifest_path.open(manifest_mode) as manifest,
    ):
        for row_index, row in enumerate(dataset):
            raw_ids = row["input_ids"]
            if raw_ids.ndim == 2:
                raw_ids = raw_ids[0]
            raw_ids = raw_ids.to(dtype=torch.int64, device="cpu").contiguous()
            token_count = int(raw_ids.numel())
            if token_count > args.max_length:
                raise RuntimeError(
                    f"row {row_index} has {token_count} tokens, max is {args.max_length}"
                )
            padded_ids = torch.cat(
                [
                    raw_ids,
                    torch.zeros(args.max_length - token_count, dtype=raw_ids.dtype),
                ]
            )
            key = hashlib.md5(padded_ids.numpy().tobytes()).hexdigest()
            output = args.hidden_dir / f"{key}.pt"
            if output.exists():
                if key in created_keys:
                    print(f"reused duplicate {row_index + 1}/{len(dataset)} {key}")
                    continue
                if not args.resume:
                    raise RuntimeError(
                        f"refusing to overwrite existing hidden row: {output}"
                    )
                existing = torch.load(
                    output, map_location="cpu", weights_only=True, mmap=True
                )
                if existing.dtype != torch.bfloat16 or tuple(existing.shape) != (
                    args.max_length,
                    width,
                ):
                    raise RuntimeError(
                        f"resume row {output.name} has dtype={existing.dtype} "
                        f"shape={tuple(existing.shape)}, expected bf16 "
                        f"{(args.max_length, width)}"
                    )
                created_keys.add(key)
                print(f"resumed {row_index + 1}/{len(dataset)} {key}")
                continue

            post_tokens(args.url, args.model, raw_ids.tolist())
            hidden = read_prefill_row(
                dump,
                torch,
                token_count,
                args.max_length,
                capture_layers,
                args.read_timeout,
                row_index,
            )

            save_tensor_atomic(torch, hidden.view(args.max_length, width), output)
            entry = {
                "row": row_index,
                "key": key,
                "tokens": token_count,
                "shape": [args.max_length, width],
                "capture_layers": capture_layers,
            }
            manifest.write(json.dumps(entry, separators=(",", ":")) + "\n")
            manifest.flush()
            created_keys.add(key)
            print(f"captured {row_index + 1}/{len(dataset)} {key} tokens={token_count}")

    print(
        f"capture complete: rows={len(dataset)} width={width} "
        f"cache_key={cache_key} manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()
