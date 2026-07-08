"""Dataset loaders for HumanEval and MBPP (sanitized).

Offline-first: each loader tries, in order,
  1. a full local JSONL (data/humaneval.jsonl / data/mbpp.jsonl) if present,
  2. the bundled small sample (data/*_sample.jsonl) — always available,
  3. the `datasets` library (openai_humaneval / mbpp), if installed + online.

Every problem is normalized to a `Problem` with a uniform shape so the runner
doesn't care where it came from.

  Problem.task_id        stable id
  Problem.prompt         text handed to the server (see build_prompt)
  Problem.test_program   fn(candidate_code_str) -> full runnable program string
                         that defines the candidate and runs the unit tests,
                         exiting 0 on success.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable, Iterator

_HERE = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_HERE, "data")


@dataclass(frozen=True)
class Problem:
    task_id: str
    prompt: str
    build_test_program: Callable[[str], str]
    # Kept for chat-style prompting / debugging.
    meta: dict


def _iter_jsonl(path: str) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ----------------------------------------------------------------------------
# HumanEval
# ----------------------------------------------------------------------------

def _humaneval_program(entry_point: str, test_src: str):
    """Return a builder that stitches candidate code + the check() harness."""

    def build(candidate_code: str) -> str:
        # candidate_code should define `entry_point`. The HumanEval `test` field
        # defines a `check(candidate)` fn; we invoke it with the entry point.
        return (
            candidate_code
            + "\n\n"
            + test_src
            + f"\n\ncheck({entry_point})\n"
            + "print('OK')\n"
        )

    return build


def _humaneval_from_record(rec: dict) -> Problem:
    ep = rec["entry_point"]
    return Problem(
        task_id=str(rec["task_id"]),
        prompt=rec["prompt"],
        build_test_program=_humaneval_program(ep, rec["test"]),
        meta={"entry_point": ep, "prompt": rec["prompt"],
              "canonical_solution": rec.get("canonical_solution", "")},
    )


def load_humaneval(limit: int | None = None) -> list[Problem]:
    full = os.path.join(_DATA, "humaneval.jsonl")
    sample = os.path.join(_DATA, "humaneval_sample.jsonl")
    path = full if os.path.exists(full) else None
    if path:
        recs = list(_iter_jsonl(path))
    elif _has_datasets():
        recs = _hf_humaneval()
    else:
        recs = list(_iter_jsonl(sample))
    probs = [_humaneval_from_record(r) for r in recs]
    return probs[:limit] if limit else probs


# ----------------------------------------------------------------------------
# MBPP (sanitized)
# ----------------------------------------------------------------------------

MBPP_PROMPT_TMPL = (
    "You are an expert Python programmer. Write a self-contained Python "
    "solution to the following task. Return only the code inside a single "
    "```python fenced block.\n\nTask: {text}\n\nYour solution must define the "
    "function(s) exercised by these tests:\n{tests}\n"
)


def _mbpp_program(test_list: list[str], setup: str):
    def build(candidate_code: str) -> str:
        body = candidate_code + "\n\n"
        if setup:
            body += setup + "\n\n"
        for t in test_list:
            body += t + "\n"
        body += "print('OK')\n"
        return body

    return build


def _mbpp_from_record(rec: dict) -> Problem:
    tests = rec["test_list"]
    # Sanitized MBPP carries test_imports (e.g. "import math" for isclose
    # assertions); full MBPP uses test_setup_code. Honor both.
    setup_parts = list(rec.get("test_imports") or [])
    if rec.get("test_setup_code"):
        setup_parts.append(rec["test_setup_code"])
    setup = "\n".join(setup_parts)
    # Full MBPP uses "text"; the sanitized split uses "prompt".
    desc = rec.get("text") or rec.get("prompt") or ""
    prompt = MBPP_PROMPT_TMPL.format(text=desc, tests="\n".join(tests))
    return Problem(
        task_id=str(rec["task_id"]),
        prompt=prompt,
        build_test_program=_mbpp_program(tests, setup),
        meta={"text": rec.get("text") or rec.get("prompt") or "", "code": rec.get("code", ""),
              "test_list": tests},
    )


def load_mbpp(limit: int | None = None) -> list[Problem]:
    full = os.path.join(_DATA, "mbpp.jsonl")
    sample = os.path.join(_DATA, "mbpp_sample.jsonl")
    if os.path.exists(full):
        recs = list(_iter_jsonl(full))
    elif _has_datasets():
        recs = _hf_mbpp()
    else:
        recs = list(_iter_jsonl(sample))
    probs = [_mbpp_from_record(r) for r in recs]
    return probs[:limit] if limit else probs


# ----------------------------------------------------------------------------
# Optional HuggingFace fallback (online only)
# ----------------------------------------------------------------------------

def _load_hf():
    """Import the real HuggingFace `datasets` library (not this module).

    This file used to be named datasets.py and shadowed the library; it is now
    eval_datasets.py. We still import via importlib to be explicit.
    """
    import importlib
    return importlib.import_module("datasets")


def _has_datasets() -> bool:
    if os.environ.get("EVALS_NO_HF") == "1":
        return False
    try:
        hf = _load_hf()
        return hasattr(hf, "load_dataset")
    except Exception:
        return False


def _hf_humaneval() -> list[dict]:  # pragma: no cover - needs network
    load_dataset = _load_hf().load_dataset
    ds = load_dataset("openai_humaneval", split="test")
    return [dict(r) for r in ds]


def _hf_mbpp() -> list[dict]:  # pragma: no cover - needs network
    load_dataset = _load_hf().load_dataset
    ds = load_dataset("mbpp", "sanitized", split="test")
    out = []
    for r in ds:
        out.append({
            "task_id": r["task_id"],
            "text": r.get("prompt") or r.get("text", ""),
            "code": r.get("code", ""),
            "test_list": r["test_list"],
            "test_setup_code": r.get("test_setup_code", ""),
        })
    return out
