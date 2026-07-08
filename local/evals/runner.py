"""Coding pass@1 runner: dataset -> server completion -> sandbox -> results.json.

Produces a results file consumable by abba.py:

    {
      "config": "<label>",
      "dataset": "humaneval|mbpp",
      "params": {...request params...},
      "pass_at_1": 0.87,
      "records": [
        {"task_id": "...", "passed": true, "status": "pass",
         "samples": [{"passed": ..., "status": ..., "code": "..."}]},
        ...
      ]
    }

This is the piece that TALKS TO THE GPU SERVER. The unit tests do NOT call it;
they test the pure pieces (extract, sandbox on trivial local code, score, abba).
The think-spec gate recipe runs this against the live server in a GPU window.
"""

from __future__ import annotations

import argparse
import json
import sys

from client import AtlasClient
from eval_datasets import load_humaneval, load_mbpp, Problem
from extract import extract_code, complete_function
from sandbox import run_code
from score import pass_at_k


def _candidate_for(problem: Problem, dataset: str, completion_text: str) -> str:
    """Turn a raw completion into candidate code appropriate for the dataset."""
    if dataset == "humaneval":
        # Completion mode: stitch continuation onto the prompt prefix (which
        # holds the signature + docstring), or use a standalone redefinition.
        return complete_function(problem.prompt, completion_text)
    # MBPP: chat/instruct style, expect a fenced block.
    return extract_code(completion_text)


_CHAT_INSTR = (
    "Complete the following Python function. Reply with ONLY the complete, "
    "runnable function definition (including any needed imports) inside a "
    "single ```python code fence. No tests, no explanation.\n\n```python\n{prompt}\n```"
)


def _eval_one(client, problem, dataset, *, n, max_tokens, temperature, seed,
              sb_timeout, mode="completion", thinking=False):
    """Draw n samples for one problem, sandbox each, return the record dict."""
    samples = []
    n_correct = 0
    for i in range(n):
        s = None if seed is None else seed + i
        if mode == "chat":
            # Instruct models don't reliably continue raw prefixes (they
            # re-emit docstrings / full solutions in varying shapes, breaking
            # the stitcher — measured 27/29 harness-artifact failures on arm
            # A 2026-07-08). Chat mode asks for ONE fenced standalone
            # function; extraction is then trivial and robust.
            comp = client.chat(
                [{"role": "user",
                  "content": _CHAT_INSTR.format(prompt=problem.prompt.rstrip())}],
                max_tokens=max_tokens, temperature=temperature, seed=s,
                enable_thinking=thinking,
            )
            code = extract_code(comp.text)
        else:
            comp = client.complete(
                problem.prompt, max_tokens=max_tokens,
                temperature=temperature, seed=s,
                # HumanEval completion mode benefits from stopping at test markers.
                stop=["\nclass ", "\nprint(", "\nif __name__"] if dataset == "humaneval" else None,
            )
            code = _candidate_for(problem, dataset, comp.text)
        program = problem.build_test_program(code)
        res = run_code(program, timeout=sb_timeout)
        if res.passed:
            n_correct += 1
        samples.append({
            "passed": res.passed,
            "status": res.status,
            "code": code,
            "completion": comp.text[-2000:],
            "stderr_tail": res.stderr[-400:] if res.stderr else "",
        })
    passed_at_1 = pass_at_k(n, n_correct, 1)  # == n_correct/n
    return {
        "task_id": problem.task_id,
        "passed": samples[0]["passed"] if n == 1 else n_correct > 0,
        "n_samples": n,
        "n_correct": n_correct,
        "pass_at_1": passed_at_1,
        "samples": samples,
    }


def run(dataset: str, *, label: str, base_url: str, model: str, limit=None,
        n=1, max_tokens=1024, temperature=0.0, seed=0, sb_timeout=10.0,
        out_path: str | None = None, mode="completion", thinking=False) -> dict:
    problems = (load_humaneval(limit) if dataset == "humaneval"
                else load_mbpp(limit))
    client = AtlasClient(base_url=base_url, model=model)

    records = []
    for p in problems:
        rec = _eval_one(client, p, dataset, n=n, max_tokens=max_tokens,
                        temperature=temperature, seed=seed, sb_timeout=sb_timeout,
                        mode=mode, thinking=thinking)
        records.append(rec)
        mark = "PASS" if rec["passed"] else "FAIL"
        print(f"[{dataset}] {p.task_id:16s} {mark} "
              f"({rec['n_correct']}/{rec['n_samples']})", flush=True)

    pass1 = sum(r["pass_at_1"] for r in records) / len(records) if records else 0.0
    result = {
        "config": label,
        "dataset": dataset,
        "params": {"model": model, "n": n, "max_tokens": max_tokens,
                   "temperature": temperature, "seed": seed,
                   "mode": mode, "thinking": thinking},
        "pass_at_1": pass1,
        "n_problems": len(records),
        "records": records,
    }
    print(f"\n[{dataset}] {label}: pass@1 = {pass1:.4f} over {len(records)} problems")
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"[{dataset}] wrote {out_path}")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description="Atlas coding pass@1 runner")
    ap.add_argument("--dataset", choices=["humaneval", "mbpp", "both"],
                    default="both")
    ap.add_argument("--label", required=True, help="config label (e.g. 'A_baseline')")
    ap.add_argument("--out", required=True, help="output results json (or prefix if --dataset both)")
    ap.add_argument("--base-url", default="http://127.0.0.1:8890")
    ap.add_argument("--model", default="aeon-27b-dflash")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n", type=int, default=1, help="samples per problem (pass@k)")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sb-timeout", type=float, default=10.0)
    ap.add_argument("--mode", choices=["completion", "chat"], default="completion")
    ap.add_argument("--thinking", action="store_true",
                    help="chat mode: enable_thinking=true (reasoning before answer)")
    args = ap.parse_args(argv)

    datasets = ["humaneval", "mbpp"] if args.dataset == "both" else [args.dataset]
    merged_records = []
    for ds in datasets:
        out = args.out if args.dataset != "both" else args.out.replace(
            ".json", f"_{ds}.json")
        r = run(ds, label=args.label, base_url=args.base_url, model=args.model,
                limit=args.limit, n=args.n, max_tokens=args.max_tokens,
                temperature=args.temperature, seed=args.seed,
                sb_timeout=args.sb_timeout, out_path=out,
                mode=args.mode, thinking=args.thinking)
        # Namespace task_ids by dataset so a merged file has unique keys.
        for rec in r["records"]:
            merged_records.append({**rec, "task_id": f"{ds}:{rec['task_id']}"})

    if args.dataset == "both":
        pass1 = (sum(x["pass_at_1"] for x in merged_records) / len(merged_records)
                 if merged_records else 0.0)
        merged = {
            "config": args.label, "dataset": "both",
            "params": {"model": args.model, "n": args.n,
                       "max_tokens": args.max_tokens,
                       "temperature": args.temperature, "seed": args.seed},
            "pass_at_1": pass1, "n_problems": len(merged_records),
            "records": merged_records,
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2)
        print(f"\n[both] merged pass@1 = {pass1:.4f}; wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
