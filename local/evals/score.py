"""pass@k estimation.

pass@1 is just the mean over problems of "did the single sample pass".

For n>1 samples per problem at temperature>0 we use the unbiased Chen et al.
(2021, "Evaluating Large Language Models Trained on Code") estimator:

    pass@k = 1 - C(n - c, k) / C(n, k)

where n = samples drawn, c = number that passed. Computed in log space to avoid
overflow and without numpy.
"""

from __future__ import annotations

from math import comb


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k for one problem: n samples, c correct, choose k."""
    if k <= 0:
        raise ValueError("k must be >= 1")
    if n < k:
        raise ValueError(f"n ({n}) must be >= k ({k})")
    if c >= n:
        return 1.0
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def aggregate_pass_at_k(per_problem: list[tuple[int, int]], k: int) -> float:
    """Mean pass@k over problems. per_problem = list of (n_samples, n_correct)."""
    if not per_problem:
        return 0.0
    return sum(pass_at_k(n, c, k) for (n, c) in per_problem) / len(per_problem)
