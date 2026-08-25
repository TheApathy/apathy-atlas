#!/usr/bin/env python3
"""Locked DeepSeek 8K-to-1M retrieval and decode sweep."""

import argparse
import hashlib
import json
import pathlib
import statistics
import tempfile
import time
import urllib.request


DEFAULT_CONTEXTS = [8192, 131072, 250000, 524288, 1000000]
SECRET = "CERULEAN-ORBIT-7391"


def build_prompt(tokenizer, token_count: int) -> list[int]:
    intro = tokenizer.encode(
        "Read the record carefully. Retain the secret exactly.\n",
        add_special_tokens=False,
    )
    needle = tokenizer.encode(
        f"\nThe secret is {SECRET}. Remember it exactly.\n", add_special_tokens=False
    )
    filler = tokenizer.encode(
        "The archive contains routine maintenance notes and ordinary observations.\n",
        add_special_tokens=False,
    )
    suffix = tokenizer.encode(
        "\nState the secret first, then explain in detail how you retained it.\n",
        add_special_tokens=False,
    )
    fixed = len(intro) + len(needle) + len(suffix)
    if token_count < fixed:
        raise ValueError(f"context {token_count} is below fixed prompt size {fixed}")
    fill_count = token_count - fixed
    repeated = (filler * ((fill_count + len(filler) - 1) // len(filler)))[:fill_count]
    midpoint = len(repeated) // 2
    tokens = intro + repeated[:midpoint] + needle + repeated[midpoint:] + suffix
    assert len(tokens) == token_count
    return tokens


def validate_contexts(
    values: str, max_position_embeddings: int, max_tokens: int, reps: int
) -> list[int]:
    try:
        contexts = [int(value) for value in values.split(",")]
    except ValueError as exc:
        raise ValueError(f"contexts must be comma-separated integers: {values!r}") from exc
    if not contexts or any(context <= 0 for context in contexts):
        raise ValueError("contexts must contain positive token counts")
    if len(contexts) != len(set(contexts)):
        raise ValueError("contexts must not contain duplicates")
    if max_tokens <= 1:
        raise ValueError("max-tokens must be greater than one for decode timing")
    if reps <= 0:
        raise ValueError("reps must be positive")
    for context in contexts:
        if context + max_tokens > max_position_embeddings:
            raise ValueError(
                f"context {context} + max_tokens {max_tokens} exceeds checkpoint "
                f"ceiling {max_position_embeddings}"
            )
    return contexts


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: pathlib.Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = pathlib.Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def stream_once(url: str, model: str, tokens: list[int], max_tokens: int) -> dict:
    body = json.dumps(
        {
            "model": model,
            "prompt": tokens,
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        separators=(",", ":"),
    ).encode()
    request = urllib.request.Request(
        f"{url.rstrip('/')}/v1/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first = None
    last = None
    pieces = []
    usage = {}
    with urllib.request.urlopen(request, timeout=7200) as response:
        for raw in response:
            if not raw.startswith(b"data: "):
                continue
            payload = raw[6:].strip()
            if payload == b"[DONE]":
                break
            event = json.loads(payload)
            usage = event.get("usage") or usage
            choices = event.get("choices") or []
            text = choices[0].get("text") if choices else None
            if text:
                now = time.perf_counter()
                first = first or now
                last = now
                pieces.append(text)
    ended = time.perf_counter()
    output = "".join(pieces)
    completion_tokens = int(usage.get("completion_tokens", 0))
    decode_seconds = max(0.0, (last or ended) - (first or ended))
    return {
        "prompt_tokens": int(usage.get("prompt_tokens", len(tokens))),
        "completion_tokens": completion_tokens,
        "ttft_seconds": (first or ended) - started,
        "decode_seconds": decode_seconds,
        "decode_tok_s": (completion_tokens - 1) / decode_seconds
        if completion_tokens > 1 and decode_seconds
        else 0.0,
        "retrieval_pass": SECRET in output,
        "output_sha256": hashlib.sha256(output.encode()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=pathlib.Path, required=True)
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8977")
    parser.add_argument("--model", default="deepseek-v4-flash-k2")
    parser.add_argument("--contexts", default=",".join(map(str, DEFAULT_CONTEXTS)))
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("deepseek-context-sweep.json"),
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise RuntimeError(f"refusing to overwrite existing output: {args.output}")

    from transformers import PreTrainedTokenizerFast

    config = json.loads(args.config.read_text())
    rope = config.get("rope_scaling") or {}
    rope_type = rope.get("rope_type", rope.get("type"))
    if config.get("max_position_embeddings", 0) < 1_000_000:
        raise RuntimeError("checkpoint does not declare at least 1M positions")
    if rope_type != "yarn" or rope.get("factor", 0) < 16:
        raise RuntimeError(f"checkpoint lacks the pinned 1M YaRN contract: {rope}")
    tokenizer_file = args.tokenizer / "tokenizer.json"
    if not tokenizer_file.is_file():
        raise RuntimeError(f"missing fast tokenizer: {tokenizer_file}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        args.tokenizer, local_files_only=True
    )
    contexts = validate_contexts(
        args.contexts,
        int(config["max_position_embeddings"]),
        args.max_tokens,
        args.reps,
    )
    plans = []
    token_sets = {}
    for context in contexts:
        tokens = build_prompt(tokenizer, context)
        token_sets[context] = tokens
        plans.append(
            {
                "context_tokens": context,
                "prompt_sha256": hashlib.sha256(
                    b"".join(token.to_bytes(4, "little") for token in tokens)
                ).hexdigest(),
                "needle": SECRET,
            }
        )
    result = {
        "inputs": {
            "config_sha256": sha256_file(args.config),
            "tokenizer_sha256": sha256_file(tokenizer_file),
        },
        "contract": {
            "max_position_embeddings": config["max_position_embeddings"],
            "rope_scaling": rope,
            "temperature": 0,
            "max_tokens": args.max_tokens,
        },
        "plans": plans,
        "cases": {},
    }
    if not args.plan_only:
        for context in contexts:
            runs = [
                stream_once(args.url, args.model, token_sets[context], args.max_tokens)
                for _ in range(args.reps)
            ]
            result["cases"][str(context)] = {
                "median_decode_tok_s": statistics.median(
                    run["decode_tok_s"] for run in runs
                ),
                "all_retrieval_pass": all(run["retrieval_pass"] for run in runs),
                "runs": runs,
            }
            print(
                f"{context}: retrieval={result['cases'][str(context)]['all_retrieval_pass']} "
                f"decode={result['cases'][str(context)]['median_decode_tok_s']:.2f} tok/s"
            )
    write_json_atomic(args.output, result)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
