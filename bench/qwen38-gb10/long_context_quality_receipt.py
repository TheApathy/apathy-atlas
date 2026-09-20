#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Run and seal the exact-token Qwen3.8 long-context quality ladder."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable

import long_context_quality_contract as contract
import long_context_quality_io as quality_io
import long_context_quality_workload as workload

REPETITIONS = 2
ARTIFACT_ARGUMENTS = {
    "binary_sha256": "binary",
    "build_receipt_sha256": "build_receipt",
    "kernel_bundle_sha256": "kernel_bundle",
    "model_config_sha256": "model_config",
    "model_index_sha256": "model_index",
    "tokenizer_sha256": "tokenizer",
    "server_receipt_sha256": "server_receipt",
}


def verify_artifacts(
    provenance: dict[str, Any], paths: dict[str, Path]
) -> dict[str, str]:
    if set(paths) != set(ARTIFACT_ARGUMENTS):
        raise ValueError("artifact path map differs from provenance contract")
    observed = {
        key: quality_io.stable_file_sha256(path, executable=key == "binary_sha256")
        for key, path in paths.items()
    }
    for key, digest in observed.items():
        if digest != provenance[key]:
            raise ValueError(f"artifact digest differs for {key}")
    return observed


def _tokenize_components(
    base_url: str,
    model: str,
    target: int,
    post: Callable[[str, str, object, float], tuple[dict, bytes]],
    timeout: float,
) -> tuple[dict[str, list[int]], dict[str, Any], dict[str, str]]:
    texts, codes = workload.component_texts(target)
    tokens: dict[str, list[int]] = {}
    receipts: dict[str, Any] = {}
    for label in workload.COMPONENT_KEYS:
        request = {"model": model, "prompt": texts[label]}
        response, raw = post(base_url, "/tokenize", request, timeout)
        token_ids = contract.validate_tokenize(response)
        tokens[label] = token_ids
        receipts[label] = {
            "text_sha256": contract.sha256(texts[label].encode("ascii")),
            "request_sha256": contract.sha256(contract.canonical_bytes(request)),
            "response_sha256": contract.sha256(raw),
            "token_ids_sha256": contract.sha256(contract.canonical_bytes(token_ids)),
            "token_count": len(token_ids),
        }
    return tokens, receipts, codes


def run_target(
    base_url: str,
    provenance: dict[str, Any],
    target: int,
    timeout: float,
    post: Callable[
        [str, str, object, float], tuple[dict, bytes]
    ] = quality_io.post_json,
) -> dict[str, Any]:
    tokens, token_receipts, codes = _tokenize_components(
        base_url, provenance["model"], target, post, timeout
    )
    prompt, positions = workload.assemble_prompt(target, tokens)
    request = workload.completion_request(provenance["model"], prompt)
    request_sha = contract.sha256(contract.canonical_bytes(request))
    observations = []
    for repetition in range(REPETITIONS):
        response, raw = post(base_url, "/v1/completions", request, timeout)
        if response.get("model") != provenance["model"]:
            raise ValueError("completion model differs from provenance")
        observation = contract.validate_completion(response, target, codes)
        observations.append(
            {
                "repetition": repetition,
                "raw_response_sha256": contract.sha256(raw),
                **observation,
            }
        )
    stable = {
        (
            item["completion_tokens"],
            item["finish_reason"],
            item["stable_output_sha256"],
        )
        for item in observations
    }
    if len(stable) != 1:
        raise ValueError("repeated completion result is nondeterministic")
    ttfts = [item["time_to_first_token_ms"] for item in observations]
    prefill_rates = [
        item["effective_prefill_tokens_per_second"] for item in observations
    ]
    return {
        "target_prompt_tokens": target,
        "prompt_token_ids_sha256": contract.sha256(contract.canonical_bytes(prompt)),
        "request_body_sha256": request_sha,
        "needle_codes": codes,
        "needle_positions": positions,
        "tokenize_receipts": token_receipts,
        "observations": observations,
        "median_time_to_first_token_ms": statistics.median(ttfts),
        "median_effective_prefill_tokens_per_second": statistics.median(prefill_rates),
        "every_repetition_at_least_2000_prefill_tps": all(
            rate >= 2_000 for rate in prefill_rates
        ),
    }


def harness_identity() -> dict[str, str]:
    paths = {
        "contract": Path(contract.__file__),
        "io": Path(quality_io.__file__),
        "workload": Path(workload.__file__),
        "runner": Path(__file__),
    }
    return {
        name: quality_io.stable_file_sha256(path, immutable=False)
        for name, path in paths.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=7200.0)
    for argument in ARTIFACT_ARGUMENTS.values():
        parser.add_argument(f"--{argument.replace('_', '-')}", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise ValueError("timeout must be finite and positive")
    base_url = quality_io.local_base_url(args.base_url)
    provenance = quality_io.load_provenance(args.provenance)
    paths = {
        key: getattr(args, argument) for key, argument in ARTIFACT_ARGUMENTS.items()
    }
    artifacts = verify_artifacts(provenance, paths)
    rows = [
        run_target(base_url, provenance, target, args.timeout)
        for target in workload.TARGET_TOKENS
    ]
    identity = harness_identity()
    receipt = {
        "schema": contract.RECEIPT_SCHEMA,
        "provenance": provenance,
        "provenance_sha256": contract.sha256(contract.canonical_bytes(provenance)),
        "verified_artifacts": artifacts,
        "harness_identity": identity,
        "harness_identity_sha256": contract.sha256(contract.canonical_bytes(identity)),
        "target_prompt_tokens": list(workload.TARGET_TOKENS),
        "repetitions": REPETITIONS,
        "rows": rows,
        "all_quality_passed": True,
        "all_prefill_2000_passed": all(
            row["every_repetition_at_least_2000_prefill_tps"] for row in rows
        ),
    }
    digest = contract.seal_receipt(args.output, receipt)
    print(json.dumps({"receipt": str(args.output), "sha256": digest}, sort_keys=True))


if __name__ == "__main__":
    main()
