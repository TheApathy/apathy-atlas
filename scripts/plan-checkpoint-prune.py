#!/usr/bin/env python3
"""Produce a non-destructive checkpoint-retention plan."""

import argparse
import json
import pathlib
import re


CHECKPOINT_RE = re.compile(r"^epoch_(\d+)_step_(\d+)$")


def tree_bytes(path: pathlib.Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def plan_root(root: pathlib.Path, keep: int) -> dict:
    checkpoints = []
    if root.is_dir():
        for item in root.iterdir():
            match = CHECKPOINT_RE.fullmatch(item.name)
            if not match or not item.is_dir():
                continue
            checkpoints.append(
                {
                    "path": str(item.resolve()),
                    "epoch": int(match.group(1)),
                    "step": int(match.group(2)),
                    "bytes": tree_bytes(item),
                    "has_config": (item / "config.json").is_file(),
                    "has_model": any(item.glob("*.safetensors")),
                    "has_training_state": (item / "training_state.pt").is_file(),
                }
            )
    checkpoints.sort(key=lambda item: (item["step"], item["epoch"]))
    split = max(0, len(checkpoints) - keep)
    candidates = checkpoints[:split]
    retained = checkpoints[split:]
    return {
        "root": str(root.resolve()),
        "retained": retained,
        "candidates": candidates,
        "candidate_bytes": sum(item["bytes"] for item in candidates),
        "status": "plan-only",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=pathlib.Path)
    parser.add_argument("--keep", type=int, default=1)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()
    if args.keep < 1:
        raise RuntimeError("keep must be at least one checkpoint per root")
    report = {
        "format": "atlas-checkpoint-prune-plan-v1",
        "plans": [plan_root(root, args.keep) for root in args.roots],
    }
    report["candidate_bytes"] = sum(
        plan["candidate_bytes"] for plan in report["plans"]
    )
    output = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()
