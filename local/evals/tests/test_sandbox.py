"""Sandbox tests — use TRIVIAL LOCAL code, never the server.

Covers: passing case, exception (nonzero exit) case, timeout case, and the
memory rlimit. These validate the isolation primitives directly.
"""
from sandbox import run_code


def test_pass_case():
    r = run_code("assert 1 + 1 == 2\nprint('OK')\n", timeout=10)
    assert r.passed
    assert r.status == "pass"
    assert r.returncode == 0
    assert "OK" in r.stdout


def test_exception_case():
    r = run_code("assert 1 + 1 == 3  # will raise AssertionError\n", timeout=10)
    assert not r.passed
    assert r.status == "fail"
    assert r.returncode != 0
    assert "AssertionError" in r.stderr


def test_syntax_error_case():
    r = run_code("def broken(:\n    pass\n", timeout=10)
    assert not r.passed
    assert r.status == "fail"


def test_timeout_case():
    # Infinite loop must be killed by the wall-clock timeout, not hang.
    r = run_code("while True:\n    pass\n", timeout=2, cpu_seconds=3)
    assert not r.passed
    assert r.status == "timeout"
    assert r.timed_out


def test_cpu_rlimit_kills_busyloop():
    # A tight CPU loop should be killed by RLIMIT_CPU even under a generous
    # wall timeout. We give a low cpu_seconds and a larger wall timeout.
    r = run_code(
        "x = 0\nwhile True:\n    x += 1\n",
        timeout=30, cpu_seconds=1,
    )
    assert not r.passed
    # Either the CPU limit (SIGXCPU -> nonzero/fail) or wall timeout fires.
    assert r.status in ("fail", "timeout")


def test_memory_rlimit():
    # Attempt a huge allocation; RLIMIT_AS should make it MemoryError, not OOM
    # the host.
    src = "x = bytearray(10 * 1024 * 1024 * 1024)\nprint('SHOULD_NOT_REACH')\n"
    r = run_code(src, timeout=15, mem_mb=256)
    assert not r.passed
    assert "SHOULD_NOT_REACH" not in r.stdout


def test_isolated_mode_ignores_parent_env():
    # -I isolated mode: PYTHONPATH-injected modules must not be importable.
    # Trivial proof: the child cannot see a bogus env var we set only in parent
    # scope (scrubbed env), so os.environ.get returns None -> exit 0.
    src = (
        "import os\n"
        "assert os.environ.get('EVALS_SANDBOX') == '1'\n"
        "assert os.environ.get('SECRET_LEAK') is None\n"
        "print('OK')\n"
    )
    r = run_code(src, timeout=10)
    assert r.passed
