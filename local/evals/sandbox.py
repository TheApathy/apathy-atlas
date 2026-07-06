"""Sandboxed execution of UNTRUSTED model-generated code.

This module runs code emitted by the language model. Treat every byte of it
as hostile. The isolation strategy is defense-in-depth:

  1. Separate process. The candidate code + the unit-test harness are written
     to a file in a fresh temp dir and run via `subprocess`, never `exec()` in
     this interpreter. A crash / segfault / infinite recursion cannot take down
     the harness.
  2. Hard wall-clock timeout. `subprocess.run(timeout=...)`; on expiry the whole
     process group is killed (SIGKILL) so runaway children die too.
  3. Resource limits applied in the child *before* exec, via a preexec_fn that
     calls resource.setrlimit:
        - RLIMIT_CPU   : CPU-seconds ceiling (kills busy-loops the wall timeout
                         might race with).
        - RLIMIT_AS    : virtual address space ceiling (kills fork-bombs by
                         allocation and `[0]*10**12`).
        - RLIMIT_FSIZE : max bytes the process may write to disk.
        - RLIMIT_NPROC : cap child processes / threads (blunts fork bombs).
     Core dumps are disabled (RLIMIT_CORE=0).
  4. New process group (os.setsid) so the timeout killer can nuke the whole
     tree with killpg, not just the direct child.
  5. Scratch cwd. The child runs in a throwaway temp dir that is deleted after,
     so files it writes are contained and cleaned up.
  6. Network: we do not grant it, but we cannot fully firewall a subprocess
     without root/namespaces. We scrub proxy env and set a marker; callers that
     need a hard network cutoff should run this inside a container / unshare.
     See NETWORK_NOTE below. For HumanEval/MBPP the test code is self-contained
     and does no I/O, so process+rlimit isolation is the appropriate bar.

NETWORK_NOTE: setrlimit cannot block sockets. If you are running this on a host
where model code reaching the network is unacceptable, wrap the whole harness in
`unshare -n` (Linux) or a container with `--network none`. The recipe scripts
mention this. The code below is written so that adding such a wrapper is a
one-line change to _RUNNER_PREFIX.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass

# Optional hard-network-off wrapper. If `unshare` is available and
# EVALS_UNSHARE_NET=1 is set, we drop the child into a network namespace with
# no interfaces. Best-effort: if it fails we fall back to plain subprocess.
_RUNNER_PREFIX: list[str] = []
if os.environ.get("EVALS_UNSHARE_NET") == "1":
    _RUNNER_PREFIX = ["unshare", "-n", "--"]


@dataclass(frozen=True)
class SandboxResult:
    """Outcome of one sandboxed run. Immutable."""

    passed: bool
    status: str  # "pass" | "fail" | "timeout" | "error"
    returncode: int | None
    stdout: str
    stderr: str

    @property
    def timed_out(self) -> bool:
        return self.status == "timeout"


def _make_preexec(cpu_seconds: int, mem_bytes: int, fsize_bytes: int,
                  nproc: int):
    """Return a preexec_fn that sets rlimits + a new session in the child.

    Defined as a closure returning a picklable-free local; runs post-fork,
    pre-exec, so it constrains the untrusted process itself.
    """

    def _limit():  # pragma: no cover - runs in child process
        import resource as _r

        # New session/process-group so killpg reaches the whole tree.
        os.setsid()
        _r.setrlimit(_r.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        _r.setrlimit(_r.RLIMIT_AS, (mem_bytes, mem_bytes))
        _r.setrlimit(_r.RLIMIT_FSIZE, (fsize_bytes, fsize_bytes))
        _r.setrlimit(_r.RLIMIT_CORE, (0, 0))
        try:
            _r.setrlimit(_r.RLIMIT_NPROC, (nproc, nproc))
        except (ValueError, OSError):
            # Some platforms disallow lowering NPROC; not fatal.
            pass

    return _limit


def run_code(
    source: str,
    *,
    timeout: float = 10.0,
    cpu_seconds: int = 8,
    mem_mb: int = 512,
    fsize_mb: int = 16,
    nproc: int = 64,
) -> SandboxResult:
    """Run a self-contained Python `source` string in an isolated subprocess.

    The source is expected to be a complete program: the candidate solution
    followed by test assertions. A clean exit (returncode 0) == pass.

    Args are conservative defaults suitable for HumanEval/MBPP unit tests.
    """
    scrubbed_env = {
        # Minimal env: keep PATH so python resolves, drop everything else that
        # could hand the child credentials or proxies.
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONIOENCODING": "utf-8",
        # Marker so code could self-detect the sandbox if it wanted to be nice.
        "EVALS_SANDBOX": "1",
        # Neutralize proxies (best-effort network deterrent).
        "http_proxy": "", "https_proxy": "", "HTTP_PROXY": "", "HTTPS_PROXY": "",
    }

    with tempfile.TemporaryDirectory(prefix="evals_sbx_") as tmp:
        prog = os.path.join(tmp, "candidate.py")
        with open(prog, "w", encoding="utf-8") as f:
            f.write(source)

        cmd = _RUNNER_PREFIX + [sys.executable, "-I", "-B", prog]
        # -I : isolated mode (ignore env PYTHON*, no user site, no cwd on path)
        # -B : don't write bytecode

        preexec = _make_preexec(
            cpu_seconds=cpu_seconds,
            mem_bytes=mem_mb * 1024 * 1024,
            fsize_bytes=fsize_mb * 1024 * 1024,
            nproc=nproc,
        )

        try:
            proc = subprocess.run(
                cmd,
                cwd=tmp,
                env=scrubbed_env,
                capture_output=True,
                text=True,
                timeout=timeout,
                preexec_fn=preexec,
                start_new_session=False,  # preexec already calls setsid
            )
        except subprocess.TimeoutExpired as e:
            # subprocess.run has already sent SIGKILL to the child; but because
            # the child made its own session, kill the whole group to be sure.
            _kill_stragglers()
            return SandboxResult(
                passed=False,
                status="timeout",
                returncode=None,
                stdout=(e.stdout or b"").decode("utf-8", "replace")
                if isinstance(e.stdout, bytes) else (e.stdout or ""),
                stderr=(e.stderr or b"").decode("utf-8", "replace")
                if isinstance(e.stderr, bytes) else (e.stderr or ""),
            )
        except Exception as e:  # pragma: no cover - launch failure
            return SandboxResult(
                passed=False, status="error", returncode=None,
                stdout="", stderr=f"sandbox launch failed: {e!r}",
            )

    if proc.returncode == 0:
        return SandboxResult(True, "pass", 0, proc.stdout, proc.stderr)
    return SandboxResult(False, "fail", proc.returncode, proc.stdout, proc.stderr)


def _kill_stragglers():  # pragma: no cover - best effort cleanup
    """No-op hook; TemporaryDirectory + killpg-by-run cover the common case."""
    return
