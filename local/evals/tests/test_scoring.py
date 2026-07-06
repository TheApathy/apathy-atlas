"""Scoring tests: HumanEval program build + a known-correct / known-wrong
solution scored end-to-end THROUGH THE SANDBOX (no server), plus pass@k math."""
from eval_datasets import load_humaneval, load_mbpp
from sandbox import run_code
from score import pass_at_k, aggregate_pass_at_k


def _get_humaneval(task_id):
    for p in load_humaneval():
        if p.task_id == task_id:
            return p
    raise AssertionError(f"{task_id} not in sample set")


def test_humaneval_correct_solution_passes():
    p = _get_humaneval("HumanEval/23")  # strlen
    correct = p.prompt + "    return len(string)\n"
    prog = p.build_test_program(correct)
    r = run_code(prog, timeout=10)
    assert r.passed, r.stderr


def test_humaneval_wrong_solution_fails():
    p = _get_humaneval("HumanEval/23")  # strlen
    wrong = p.prompt + "    return len(string) + 1  # off by one\n"
    prog = p.build_test_program(wrong)
    r = run_code(prog, timeout=10)
    assert not r.passed


def test_humaneval_canonical_solution_passes():
    # Every bundled HumanEval problem's canonical solution must pass its tests.
    for p in load_humaneval():
        code = p.prompt + p.meta["canonical_solution"]
        r = run_code(p.build_test_program(code), timeout=10)
        assert r.passed, f"{p.task_id} canonical failed: {r.stderr[-300:]}"


def test_mbpp_correct_solution_passes():
    p = load_mbpp()[0]  # similar_elements
    r = run_code(p.build_test_program(p.meta["code"]), timeout=10)
    assert r.passed, r.stderr


def test_mbpp_wrong_solution_fails():
    p = load_mbpp()[0]
    wrong = "def similar_elements(a, b):\n    return ()  # always empty\n"
    r = run_code(p.build_test_program(wrong), timeout=10)
    assert not r.passed


def test_mbpp_canonical_solutions_pass():
    for p in load_mbpp():
        r = run_code(p.build_test_program(p.meta["code"]), timeout=10)
        assert r.passed, f"{p.task_id} canonical failed: {r.stderr[-300:]}"


def test_pass_at_k_math():
    assert pass_at_k(1, 1, 1) == 1.0
    assert pass_at_k(1, 0, 1) == 0.0
    # n=5, c=1: pass@1 = 1 - C(4,1)/C(5,1) = 1 - 4/5 = 0.2
    assert abs(pass_at_k(5, 1, 1) - 0.2) < 1e-9
    # n=5, c=1, k=5: 1 - C(4,5)/C(5,5) but n-c<k -> 1.0
    assert pass_at_k(5, 1, 5) == 1.0
    # n=10, c=5, k=1 -> 0.5
    assert abs(pass_at_k(10, 5, 1) - 0.5) < 1e-9


def test_aggregate_pass_at_k():
    per = [(5, 5), (5, 0), (5, 1)]  # 1.0, 0.0, 0.2
    assert abs(aggregate_pass_at_k(per, 1) - (1.2 / 3)) < 1e-9
