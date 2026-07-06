"""ABBA paired-bootstrap tests: known-delta brackets, identical straddles 0,
and the verdict logic + file-based CLI path."""
import json
import os
import tempfile

from abba import paired_bootstrap, run_from_files, _percentile


def test_identical_a_equals_b_straddles_zero():
    # A == B: delta 0, CI must straddle 0 and verdict is "not worse".
    vec = [1, 1, 0, 1, 0, 1, 1, 0, 1, 1] * 5
    rep = paired_bootstrap(vec, vec, iters=3000, seed=7)
    assert rep.delta == 0.0
    assert rep.ci_low <= 0.0 <= rep.ci_high
    assert rep.verdict.startswith("B not worse")


def test_known_positive_delta_is_bracketed():
    # B strictly better than A on a chunk of problems -> positive delta, CI
    # should bracket the true delta and its lower bound stays > -epsilon.
    n = 100
    a = [0] * n
    b = [0] * n
    # Make B pass 30 that A fails -> true delta = +0.30
    for i in range(30):
        b[i] = 1
    rep = paired_bootstrap(a, b, iters=5000, seed=3)
    assert abs(rep.delta - 0.30) < 1e-9
    assert rep.ci_low <= rep.delta <= rep.ci_high
    assert rep.ci_low > -rep.epsilon  # clearly not worse
    assert rep.verdict.startswith("B not worse")


def test_known_negative_delta_blocks():
    # B worse than A -> negative delta, CI high below -epsilon -> block.
    n = 100
    a = [1] * n
    b = [1] * n
    for i in range(30):
        b[i] = 0  # B fails 30 that A passes -> delta = -0.30
    rep = paired_bootstrap(a, b, iters=5000, seed=5, epsilon=0.01)
    assert abs(rep.delta + 0.30) < 1e-9
    assert rep.ci_high < -rep.epsilon
    assert "WORSE" in rep.verdict


def test_tiny_regression_within_epsilon_is_shippable():
    # A single-problem regression on a large set is within epsilon -> ship.
    n = 200
    a = [1] * n
    b = [1] * n
    b[0] = 0  # delta = -1/200 = -0.005, inside epsilon=0.01
    rep = paired_bootstrap(a, b, iters=5000, seed=11, epsilon=0.01)
    assert rep.delta < 0
    # Lower bound can dip below -epsilon on a resample; verdict may be
    # inconclusive OR ship, but must NOT be a hard block for so tiny a delta.
    assert "WORSE" not in rep.verdict


def test_percentile_helper():
    vals = list(range(101))  # 0..100 sorted
    assert _percentile(vals, 0.0) == 0
    assert _percentile(vals, 1.0) == 100
    assert abs(_percentile(vals, 0.5) - 50) < 1e-9
    assert abs(_percentile(vals, 0.025) - 2.5) < 1e-9


def test_run_from_files_paired():
    # Write two results files with overlapping task_ids and check pairing.
    a = {"config": "A", "records": [
        {"task_id": "t1", "passed": True},
        {"task_id": "t2", "passed": False},
        {"task_id": "t3", "passed": True},
    ]}
    b = {"config": "B", "records": [
        {"task_id": "t1", "passed": True},
        {"task_id": "t2", "passed": True},   # B fixes t2
        {"task_id": "t3", "passed": True},
    ]}
    with tempfile.TemporaryDirectory() as d:
        pa = os.path.join(d, "a.json")
        pb = os.path.join(d, "b.json")
        json.dump(a, open(pa, "w"))
        json.dump(b, open(pb, "w"))
        rep = run_from_files(pa, pb, iters=2000, seed=1)
    assert rep.n_problems == 3
    assert abs(rep.a_score - 2 / 3) < 1e-9
    assert abs(rep.b_score - 1.0) < 1e-9
    assert abs(rep.delta - 1 / 3) < 1e-9
