#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Exact-token early/middle/late needle workload construction."""

from __future__ import annotations

import hashlib
from typing import Any

TARGET_TOKENS = (262_143, 262_144, 262_145, 500_000, 750_000, 999_000)
COMPONENT_KEYS = ("prefix", "filler", "early", "middle", "late", "query")
OUTPUT_TOKENS = 64
SEED = 0x38_1_000_000
GAP_WEIGHTS = (1, 15, 15, 1)


def _code(target: int, label: str) -> str:
    material = f"qwen3.8-long-context-v1:{target}:{label}".encode("ascii")
    return hashlib.sha256(material).hexdigest()[:16].upper()


def component_texts(target: int) -> tuple[dict[str, str], dict[str, str]]:
    if target not in TARGET_TOKENS:
        raise ValueError("target is outside the frozen quality ladder")
    codes = {label: _code(target, label) for label in ("early", "middle", "late")}
    texts = {
        "prefix": (
            "Long-context memory audit. Read every record. Ignore neutral filler. "
            "At the final query, recall the EARLY, MIDDLE, and LATE access codes.\n"
        ),
        "filler": (
            "Neutral archival filler: basalt cedar delta harbor ivory juniper. "
            "This sentence contains no access code.\n"
        ),
        "early": f"\n[EARLY_RECORD] The EARLY access code is {codes['early']}. [/EARLY_RECORD]\n",
        "middle": (
            f"\n[MIDDLE_RECORD] The MIDDLE access code is {codes['middle']}. "
            "[/MIDDLE_RECORD]\n"
        ),
        "late": f"\n[LATE_RECORD] The LATE access code is {codes['late']}. [/LATE_RECORD]\n",
        "query": (
            "\nFinal query. Return one line and no other text, using the access codes "
            "from the three records in this exact format: "
            "EARLY=<early-code> MIDDLE=<middle-code> LATE=<late-code>\nAnswer:"
        ),
    }
    return texts, codes


def _validate_components(components: object) -> dict[str, list[int]]:
    if not isinstance(components, dict) or tuple(components) != COMPONENT_KEYS:
        raise ValueError("component token map has wrong keys or order")
    for label, tokens in components.items():
        if not isinstance(tokens, list) or not tokens:
            raise ValueError(f"{label} token vector is empty")
        if any(
            type(token) is not int or not 0 <= token <= 0xFFFF_FFFF for token in tokens
        ):
            raise ValueError(f"{label} contains a non-u32 token")
    return components


def _filler(tokens: list[int], length: int) -> list[int]:
    repetitions, tail = divmod(length, len(tokens))
    return tokens * repetitions + tokens[:tail]


def assemble_prompt(
    target: int, components: object
) -> tuple[list[int], dict[str, int]]:
    if target not in TARGET_TOKENS:
        raise ValueError("target is outside the frozen quality ladder")
    token_map = _validate_components(components)
    fixed = sum(len(token_map[label]) for label in COMPONENT_KEYS if label != "filler")
    if fixed >= target:
        raise ValueError("fixed prompt components leave no filler capacity")
    remaining = target - fixed
    gaps = [remaining * weight // sum(GAP_WEIGHTS) for weight in GAP_WEIGHTS[:-1]]
    gaps.append(remaining - sum(gaps))
    prompt = list(token_map["prefix"])
    positions: dict[str, int] = {}
    for index, label in enumerate(("early", "middle", "late")):
        prompt.extend(_filler(token_map["filler"], gaps[index]))
        positions[label] = len(prompt)
        prompt.extend(token_map[label])
    prompt.extend(_filler(token_map["filler"], gaps[-1]))
    prompt.extend(token_map["query"])
    if len(prompt) != target:
        raise AssertionError("exact prompt assembly drifted")
    return prompt, positions


def completion_request(model: str, prompt_tokens: list[int]) -> dict[str, Any]:
    if not isinstance(model, str) or not model:
        raise ValueError("model must be non-empty")
    _validate_components(
        {
            "prefix": [0],
            "filler": [0],
            "early": [0],
            "middle": [0],
            "late": [0],
            "query": prompt_tokens,
        }
    )
    return {
        "model": model,
        "prompt": "",
        "prompt_token_ids": prompt_tokens,
        "max_tokens": OUTPUT_TOKENS,
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "top_n_sigma": 0.0,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "stream": False,
        "stop": [],
        "seed": SEED,
    }
