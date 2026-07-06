"""ABBA: paired-bootstrap confidence interval on a pass@1 delta (B - A).

This is THE ship gate for quality-neutral levers. Given two runs (A = lever off,
B = lever on) over the SAME problem set, PAIRED per task_id, it computes:

  - A score (mean pass@1), B score (mean pass@1)
  - delta = B - A
  - a bootstrap 95% CI on delta by resampling PROBLEMS with replacement
    (paired: for each resampled problem we take both A and B outcomes, so the
    pairing / shared-problem variance is preserved).
  - a verdict: "B not worse than A" iff the CI lower bound > -epsilon.

Why paired bootstrap: A and B are evaluated on identical problems, so a per-
problem difference d_i = b_i - a_i cancels problem difficulty. Resampling the
d_i (equivalently, resampling problems and recomputing both means) gives a CI on
the mean difference that respects the pairing and makes no normality assumption.

Pure stdlib (random) — no numpy dependency.

CLI:
    python abba.py resultsA.json resultsB.json [--epsilon 0.01] [--iters 10000]

Each results file is JSON:
    {"config": "...", "records": [{"task_id": "...", "passed": true/false}, ...]}
(Extra fields are ignored. See runner.py for the writer.)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class ABBAReport:
    n_problems: int
    a_score: float
    b_score: float
    delta: float
    ci_low: float
    ci_high: float
    ci_level: float
    epsilon: float
    verdict: str
    iters: int

    def as_dict(self) -> dict:
        return asdict(self)


def _load_pairs(path_a: str, path_b: str) -> tuple[list[int], list[int]]:
    a = _read_records(path_a)
    b = _read_records(path_b)
    common = [tid for tid in a if tid in b]
    if not common:
        raise ValueError("no overlapping task_ids between A and B result files")
    # Deterministic order for reproducibility.
    common.sort()
    a_vec = [1 if a[t] else 0 for t in common]
    b_vec = [1 if b[t] else 0 for t in common]
    return a_vec, b_vec


def _read_records(path: str) -> dict[str, bool]:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    out: dict[str, bool] = {}
    for r in d["records"]:
        out[str(r["task_id"])] = bool(r["passed"])
    return out


def paired_bootstrap(
    a_vec: list[int],
    b_vec: list[int],
    *,
    iters: int = 10000,
    ci_level: float = 0.95,
    epsilon: float = 0.01,
    seed: int = 1234,
) -> ABBAReport:
    """Compute the paired-bootstrap CI on delta = mean(B) - mean(A).

    a_vec, b_vec: aligned per-problem 0/1 pass indicators (same order/length).
    """
    if len(a_vec) != len(b_vec):
        raise ValueError("A and B vectors must be the same length (paired)")
    n = len(a_vec)
    if n == 0:
        raise ValueError("empty problem set")

    a_score = sum(a_vec) / n
    b_score = sum(b_vec) / n
    delta = b_score - a_score

    rng = random.Random(seed)
    deltas: list[float] = []
    idx_range = range(n)
    for _ in range(iters):
        # Resample problem indices with replacement; take BOTH a_i and b_i at
        # each — this preserves pairing.
        sa = 0
        sb = 0
        for _j in idx_range:
            k = rng.randrange(n)
            sa += a_vec[k]
            sb += b_vec[k]
        deltas.append((sb - sa) / n)

    deltas.sort()
    lo_q = (1.0 - ci_level) / 2.0
    hi_q = 1.0 - lo_q
    ci_low = _percentile(deltas, lo_q)
    ci_high = _percentile(deltas, hi_q)

    if ci_low > -epsilon:
        verdict = "B not worse than A (ship)"
    elif ci_high < -epsilon:
        verdict = "B WORSE than A (block)"
    else:
        verdict = "inconclusive (widen n or iters)"

    return ABBAReport(
        n_problems=n,
        a_score=a_score,
        b_score=b_score,
        delta=delta,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_level=ci_level,
        epsilon=epsilon,
        verdict=verdict,
        iters=iters,
    )


def _percentile(sorted_vals: list[float], q: float) -> float:
    """Linear-interpolation percentile on an already-sorted list. q in [0,1]."""
    if not sorted_vals:
        raise ValueError("empty")
    if q <= 0:
        return sorted_vals[0]
    if q >= 1:
        return sorted_vals[-1]
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    frac = pos - lo
    if lo + 1 < len(sorted_vals):
        return sorted_vals[lo] * (1 - frac) + sorted_vals[lo + 1] * frac
    return sorted_vals[lo]


def run_from_files(path_a: str, path_b: str, *, iters: int = 10000,
                   epsilon: float = 0.01, ci_level: float = 0.95,
                   seed: int = 1234) -> ABBAReport:
    a_vec, b_vec = _load_pairs(path_a, path_b)
    return paired_bootstrap(a_vec, b_vec, iters=iters, ci_level=ci_level,
                            epsilon=epsilon, seed=seed)


def _format(rep: ABBAReport, path_a: str, path_b: str) -> str:
    pct = int(rep.ci_level * 100)
    return (
        f"=== ABBA paired-bootstrap ===\n"
        f"A: {path_a}\n"
        f"B: {path_b}\n"
        f"paired problems : {rep.n_problems}\n"
        f"A pass@1        : {rep.a_score:.4f}\n"
        f"B pass@1        : {rep.b_score:.4f}\n"
        f"delta (B-A)     : {rep.delta:+.4f}\n"
        f"{pct}% CI on delta : [{rep.ci_low:+.4f}, {rep.ci_high:+.4f}]"
        f"  (bootstrap iters={rep.iters}, epsilon={rep.epsilon})\n"
        f"VERDICT         : {rep.verdict}\n"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ABBA paired-bootstrap CI on pass@1 delta")
    ap.add_argument("results_a")
    ap.add_argument("results_b")
    ap.add_argument("--epsilon", type=float, default=0.01,
                    help="ship if CI lower bound > -epsilon (default 0.01 = 1%%)")
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--ci", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--json", action="store_true", help="emit JSON only")
    args = ap.parse_args(argv)

    rep = run_from_files(args.results_a, args.results_b, iters=args.iters,
                         epsilon=args.epsilon, ci_level=args.ci, seed=args.seed)
    if args.json:
        print(json.dumps(rep.as_dict(), indent=2))
    else:
        print(_format(rep, args.results_a, args.results_b))
    # Exit non-zero if the lever is a quality regression, so CI can gate on it.
    return 0 if rep.verdict.startswith("B not worse") else 1


if __name__ == "__main__":
    sys.exit(main())
