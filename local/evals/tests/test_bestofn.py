"""best-of-N ARBITRATION tests — GPU-free, no server.

The logic under test (given N candidate codes + tests -> pick a passing one;
all-fail graceful degradation; partial-pass ranking) is exercised two ways:

  * with a FAKE sandbox runner (canned pass/fail per code) so the selection
    policy is tested in isolation, deterministically, with zero subprocess cost;
  * with the REAL sandbox on trivial local code, to prove the wiring is honest.

None of these talk to the GPU server. `generate_candidates` / `best_of_n`'s
network half is covered separately by the live HumanEval run (needs the server).
"""
import sys
import os

# local/evals is on path for the building blocks (client/sandbox/extract);
# local/ (one level up) is where bestofn.py lives.
_EVALS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _EVALS)
sys.path.insert(0, os.path.dirname(_EVALS))

from bestofn import (  # noqa: E402
    arbitrate,
    score_candidate,
    generate_candidates,
    best_of_n,
    CandidateResult,
    BestOfNResult,
)
from sandbox import SandboxResult  # noqa: E402


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

def _fake_runner(pass_predicate):
    """Build a sandbox runner that passes iff pass_predicate(source) is True."""

    def runner(source, timeout=10.0, **kw):
        ok = pass_predicate(source)
        return SandboxResult(
            passed=ok, status="pass" if ok else "fail",
            returncode=0 if ok else 1, stdout="OK" if ok else "", stderr="",
        )

    return runner


# The test builder just marks the program so the fake runner can key on it.
def _builder(code):
    return code + "\n# RUN_TESTS\n"


# ---------------------------------------------------------------------------
# 1. First full pass wins (lowest index)
# ---------------------------------------------------------------------------

def test_first_passing_candidate_wins():
    codes = ["BAD_a", "GOOD_b", "GOOD_c"]
    runner = _fake_runner(lambda s: s.startswith("GOOD"))
    res = arbitrate(codes, _builder, runner=runner)
    assert res.winner_index == 1        # first GOOD, not the later one
    assert res.code == "GOOD_b"
    assert res.passed is True
    assert res.any_passed is True
    assert res.n_passed == 2


def test_single_candidate_pass():
    res = arbitrate(["GOOD"], _builder, runner=_fake_runner(lambda s: True))
    assert res.winner_index == 0
    assert res.passed is True
    assert res.n == 1


# ---------------------------------------------------------------------------
# 2. All-fail graceful degradation -> pass@1 candidate (index 0)
# ---------------------------------------------------------------------------

def test_all_fail_degrades_to_first_candidate():
    codes = ["x0", "x1", "x2"]
    runner = _fake_runner(lambda s: False)  # nothing passes
    res = arbitrate(codes, _builder, runner=runner)
    assert res.passed is False
    assert res.any_passed is False
    assert res.n_passed == 0
    # Graceful: falls back to candidate 0 (the pass@1 answer), never worse.
    assert res.winner_index == 0
    assert res.code == "x0"


def test_empty_candidates_is_wellformed():
    res = arbitrate([], _builder, runner=_fake_runner(lambda s: True))
    assert res.n == 0
    assert res.winner_index == -1
    assert res.passed is False


# ---------------------------------------------------------------------------
# 3. Partial-pass ranking when none fully pass
# ---------------------------------------------------------------------------

def test_partial_ranking_picks_highest_score():
    # No candidate passes the FULL program, but candidates differ on how many
    # of the individual sub-tests they satisfy. Highest partial score wins.
    partial = ["assert f(1) == 1", "assert f(2) == 2", "assert f(3) == 3"]

    def runner(source, timeout=10.0, **kw):
        # Full program (has RUN_TESTS marker) always fails.
        if "# RUN_TESTS" in source:
            return SandboxResult(False, "fail", 1, "", "boom")
        # Sub-test programs: candidate "c_two" satisfies 2 of 3, "c_one" 1 of 3,
        # "c_zero" none. We key on the candidate prefix embedded in `source`.
        passes = 0
        if source.startswith("c_two"):
            passes = "f(1)" in source or "f(2)" in source
        elif source.startswith("c_one"):
            passes = "f(1)" in source
        return SandboxResult(bool(passes), "pass" if passes else "fail",
                             0 if passes else 1, "", "")

    codes = ["c_zero", "c_one", "c_two"]
    res = arbitrate(codes, _builder, runner=runner, partial_tests=partial)
    assert res.passed is False          # nobody fully passed
    assert res.winner_index == 2        # c_two: 2/3 sub-tests, the best
    # Scores are recorded per candidate.
    scores = {c.index: c.score for c in res.candidates}
    assert scores[2] > scores[1] > scores[0]


def test_partial_tie_keeps_lowest_index():
    partial = ["assert True"]
    # Both candidates satisfy the single sub-test but neither passes full.
    def runner(source, timeout=10.0, **kw):
        if "# RUN_TESTS" in source:
            return SandboxResult(False, "fail", 1, "", "")
        return SandboxResult(True, "pass", 0, "", "")

    res = arbitrate(["a", "b"], _builder, runner=runner, partial_tests=partial)
    assert res.winner_index == 0        # tie -> stable lowest index


# ---------------------------------------------------------------------------
# 4. score_candidate unit behavior
# ---------------------------------------------------------------------------

def test_score_candidate_full_pass_short_circuits():
    calls = {"n": 0}

    def runner(source, timeout=10.0, **kw):
        calls["n"] += 1
        return SandboxResult(True, "pass", 0, "", "")

    passed, status, score = score_candidate(
        "code", _builder, runner=runner, partial_tests=["a", "b", "c"])
    assert passed is True and score == 1.0
    # Full pass must NOT run the partial sub-tests (short-circuit).
    assert calls["n"] == 1


def test_score_candidate_partial_fraction():
    def runner(source, timeout=10.0, **kw):
        if "# RUN_TESTS" in source:
            return SandboxResult(False, "fail", 1, "", "")
        # exactly one sub-test (the one containing "keep") passes
        ok = "keep" in source
        return SandboxResult(ok, "pass" if ok else "fail", 0 if ok else 1, "", "")

    passed, status, score = score_candidate(
        "code", _builder, runner=runner,
        partial_tests=["keep", "drop1", "drop2", "drop3"])
    assert passed is False
    assert abs(score - 0.25) < 1e-9


# ---------------------------------------------------------------------------
# 5. REAL sandbox integration (trivial local code, still no server)
# ---------------------------------------------------------------------------

def test_arbitrate_with_real_sandbox():
    # Candidate A is wrong (returns x+1), B is correct (returns x*2).
    good = "def solve(x):\n    return x * 2\n"
    bad = "def solve(x):\n    return x + 1\n"

    def builder(code):
        return code + "\nassert solve(3) == 6\nassert solve(5) == 10\nprint('OK')\n"

    # Real run_code via default runner.
    res = arbitrate([bad, good], builder)
    assert res.passed is True
    assert res.winner_index == 1
    assert res.code == good


def test_arbitrate_real_sandbox_all_wrong_degrades():
    bad1 = "def solve(x):\n    return 0\n"
    bad2 = "def solve(x):\n    return 1\n"

    def builder(code):
        return code + "\nassert solve(3) == 6\nprint('OK')\n"

    res = arbitrate([bad1, bad2], builder)
    assert res.passed is False
    assert res.any_passed is False
    assert res.winner_index == 0        # graceful pass@1 fallback


# ---------------------------------------------------------------------------
# 6. Diversity guard: n>1 with no diversity source is refused
# ---------------------------------------------------------------------------

def test_generate_refuses_undiverse_config():
    # temperature=0 AND seed=None with n>1 => all identical => useless.
    class _DummyClient:
        def chat(self, *a, **k):
            raise AssertionError("should not be called")

    import pytest
    with pytest.raises(ValueError, match="diversity"):
        generate_candidates(_DummyClient(), [{"role": "user", "content": "x"}],
                            n=4, temperature=0.0, seed=None)


def test_generate_n1_temp0_is_allowed():
    # n=1 is a plain single call; temp-0 determinism is fine there.
    class _DummyClient:
        def chat(self, messages, **k):
            from client import Completion
            return Completion(text="```python\nx=1\n```", wall_s=0.01)

    comps, wall = generate_candidates(
        _DummyClient(), [{"role": "user", "content": "x"}],
        n=1, temperature=0.0, seed=None)
    assert len(comps) == 1
    assert wall >= 0.0


# ---------------------------------------------------------------------------
# 7. best_of_n end-to-end with a FAKE client (no server) — wiring + accounting
# ---------------------------------------------------------------------------

def test_best_of_n_end_to_end_fake_client():
    # Fake client returns different code per seed so arbitration has real choice.
    class _FakeClient:
        def chat(self, messages, *, max_tokens, temperature, seed,
                 enable_thinking=False):
            from client import Completion
            # seed 0/1 -> wrong; seed 2 -> correct.
            body = {
                0: "def solve(x):\n    return x + 1",
                1: "def solve(x):\n    return x - 1",
                2: "def solve(x):\n    return x * 2",
            }.get(seed, "def solve(x):\n    return 0")
            return Completion(text=f"```python\n{body}\n```", wall_s=0.5)

    def builder(code):
        return code + "\nassert solve(3) == 6\nprint('OK')\n"

    res = best_of_n(
        "double x", builder, n=3, temperature=0.7, seed=0,
        client=_FakeClient(),
    )
    assert isinstance(res, BestOfNResult)
    assert res.passed is True
    assert res.winner_index == 2
    assert res.n_passed == 1
    assert res.any_passed is True
    # Accounting is populated.
    assert res.gen_wall_s >= 0.0
    assert res.total_wall_s >= res.gen_wall_s
    # sum of per-candidate latencies (serial proxy) >= slowest (overlap proxy).
    assert res.sum_candidate_wall_s >= res.max_candidate_wall_s > 0.0
