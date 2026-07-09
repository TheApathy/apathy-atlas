"""Concurrent best-of-N with sandbox arbitration (BREAKTHROUGH-IDEAS #12).

The "highest coding + coherence" lever. For a coding request we decode N
independent candidate solutions IN PARALLEL against the Atlas OpenAI-compatible
server, execute every candidate in the untrusted-code sandbox against the
provided unit tests, and return the candidate that PASSES. This buys pass@N
*correctness* at roughly pass@1 *wall-clock*, because the validated lossless
concurrency (ATLAS_MULTISEQ_GRAPHS) runs the N requests simultaneously
server-side (one 14GB weight read serves N sequences — see
memory/concurrency-win-2026-07-06).

This is an ORCHESTRATION layer, not an engine change. It composes with the
existing server via plain HTTP; nothing here touches the Rust engine.

----------------------------------------------------------------------------
WHY temperature>0 / seed-diversity IS MANDATORY (the core design constraint)
----------------------------------------------------------------------------
Best-of-N only helps if the N candidates DIFFER. At temperature=0 the model is
(near-)deterministic, so all N samples are identical — you pay N× the compute
(hidden by concurrency) for exactly ONE distinct solution: pass@N collapses to
pass@1. Diversity therefore requires one of:

  * temperature > 0 (e.g. 0.6-0.8): stochastic sampling makes each draw an
    independent shot at the solution. This is the primary knob.
  * distinct seed per candidate: even at a fixed temperature, a different seed
    decorrelates the RNG stream so the N draws explore different completions.

We do BOTH by default: temperature>0 AND a distinct seed per candidate (base
seed + i). Seeds also make each *individual* candidate reproducible for
debugging while the ensemble stays diverse.

TRADE-OFF (accepted, documented): this abandons the temp-0 md5-determinism that
the rest of the harness leans on. That determinism is not the oracle here — the
SANDBOX ARBITER is. Correctness is decided by "does the candidate pass its unit
tests", which is a stronger, semantic oracle than byte-identity. So giving up
temp-0 determinism is the right trade for this lever: we are optimizing the
correctness axis, not the reproducibility axis.

----------------------------------------------------------------------------
ARBITRATION
----------------------------------------------------------------------------
`arbitrate()` is a PURE function of (candidate codes, test-program builder,
sandbox runner). It:
  1. Runs each candidate in the sandbox against the unit tests.
  2. Returns the FIRST candidate that fully passes.
  3. If none fully pass, ranks by a partial score (tests-passed when the tests
     can be split; else the binary pass/fail) and returns the best-scoring one.
  4. Graceful degradation: if nothing runs at all, returns candidate 0 (the
     pass@1 answer) so best-of-N never does WORSE than a single call.

Because arbitration is pure and takes an injected sandbox runner, its full
decision logic — first-pass selection, all-fail degradation, partial-pass
ranking — is unit-tested GPU-free with CANNED candidate codes.
"""

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Sequence

# Local imports — this file lives in local/, its building blocks in local/evals/.
sys.path.insert(0, __file__.rsplit("/", 1)[0] + "/evals")

from client import AtlasClient, Completion  # noqa: E402
from extract import extract_code  # noqa: E402
from sandbox import run_code, SandboxResult  # noqa: E402


# ---------------------------------------------------------------------------
# Result types (immutable)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CandidateResult:
    """One candidate's arbitration outcome."""

    index: int
    code: str
    passed: bool
    status: str           # sandbox status: pass|fail|timeout|error
    score: float          # partial-credit score in [0, 1]; 1.0 == full pass
    completion: str = ""  # raw model text (tail), for debugging
    wall_s: float = 0.0   # this candidate's request latency (0 if unknown)


@dataclass(frozen=True)
class BestOfNResult:
    """Final selection plus full accounting."""

    winner_index: int
    code: str
    passed: bool                       # did the winner fully pass its tests
    n: int
    candidates: tuple[CandidateResult, ...] = field(default_factory=tuple)
    any_passed: bool = False           # did ANY candidate pass (pass-any@N)
    n_passed: int = 0                  # how many of N passed
    gen_wall_s: float = 0.0            # wall-clock of the CONCURRENT generation
    arb_wall_s: float = 0.0            # wall-clock of sandbox arbitration
    total_wall_s: float = 0.0          # gen + arb
    max_candidate_wall_s: float = 0.0  # slowest single request (overlap proxy)
    sum_candidate_wall_s: float = 0.0  # summed request latencies (serial proxy)


# ---------------------------------------------------------------------------
# Arbitration — PURE, GPU-free, unit-tested
# ---------------------------------------------------------------------------

def score_candidate(
    code: str,
    build_test_program: Callable[[str], str],
    *,
    sb_timeout: float = 10.0,
    runner: Callable[..., SandboxResult] = run_code,
    partial_tests: Sequence[str] | None = None,
    partial_prelude: str = "",
) -> tuple[bool, str, float]:
    """Sandbox one candidate. Return (passed, status, score in [0,1]).

    If `partial_tests` is given (a list of independent assertion/statement
    strings), we run them ONE AT A TIME on top of the candidate to compute a
    fractional score, so all-fail ensembles can still be ranked. Otherwise the
    score is binary (1.0 on pass, 0.0 otherwise) from a single full run.
    """
    full = runner(build_test_program(code), timeout=sb_timeout)
    if full.passed:
        return True, full.status, 1.0

    if partial_tests:
        n_pass = 0
        for t in partial_tests:
            prog = code + "\n\n" + partial_prelude + "\n" + t + "\nprint('OK')\n"
            r = runner(prog, timeout=sb_timeout)
            if r.passed:
                n_pass += 1
        score = n_pass / len(partial_tests) if partial_tests else 0.0
        return False, full.status, score

    return False, full.status, 0.0


def arbitrate(
    codes: Sequence[str],
    build_test_program: Callable[[str], str],
    *,
    sb_timeout: float = 10.0,
    runner: Callable[..., SandboxResult] = run_code,
    partial_tests: Sequence[str] | None = None,
    partial_prelude: str = "",
    completions: Sequence[str] | None = None,
    wall_times: Sequence[float] | None = None,
) -> BestOfNResult:
    """Pick the best candidate. PURE given `runner`; the GPU-free arbiter.

    Policy:
      1. First candidate that FULLY passes wins (stable, lowest index).
      2. Else the highest partial-score candidate wins (ties -> lowest index).
      3. Else (empty/degenerate) candidate 0 is the graceful pass@1 fallback.
    """
    if not codes:
        # Nothing to arbitrate: return a well-formed empty result.
        return BestOfNResult(winner_index=-1, code="", passed=False, n=0)

    results: list[CandidateResult] = []
    for i, code in enumerate(codes):
        passed, status, score = score_candidate(
            code, build_test_program, sb_timeout=sb_timeout, runner=runner,
            partial_tests=partial_tests, partial_prelude=partial_prelude,
        )
        results.append(CandidateResult(
            index=i, code=code, passed=passed, status=status, score=score,
            completion=(completions[i] if completions and i < len(completions) else ""),
            wall_s=(wall_times[i] if wall_times and i < len(wall_times) else 0.0),
        ))

    # 1. First full pass (stable order == lowest index).
    winner = next((r for r in results if r.passed), None)
    # 2. Else highest score; max() is stable so ties keep the lowest index.
    if winner is None:
        winner = max(results, key=lambda r: r.score)
    # 3. max() over a non-empty list always yields a candidate -> index 0 when
    #    all scores are equal (e.g. all 0.0) == graceful pass@1 fallback.

    n_passed = sum(1 for r in results if r.passed)
    sum_wall = sum(r.wall_s for r in results)
    max_wall = max((r.wall_s for r in results), default=0.0)
    return BestOfNResult(
        winner_index=winner.index,
        code=winner.code,
        passed=winner.passed,
        n=len(codes),
        candidates=tuple(results),
        any_passed=n_passed > 0,
        n_passed=n_passed,
        max_candidate_wall_s=max_wall,
        sum_candidate_wall_s=sum_wall,
    )


# ---------------------------------------------------------------------------
# Concurrent generation — talks to the GPU server
# ---------------------------------------------------------------------------

def generate_candidates(
    client: AtlasClient,
    messages: list[dict],
    *,
    n: int,
    temperature: float = 0.7,
    seed: int | None = 0,
    max_tokens: int = 1024,
    enable_thinking: bool = False,
) -> tuple[list[Completion], float]:
    """Fire N chat requests CONCURRENTLY. Returns (completions, gen_wall_s).

    Diversity: temperature>0 AND a distinct seed per candidate (base seed + i).
    The N requests are dispatched from N threads so they land on the server
    together and are batched by the multiseq CUDA graphs — the whole point of
    this lever. `gen_wall_s` is the wall-clock of the WHOLE concurrent batch
    (submit -> all-returned), which is the number that should be ~1× a single
    call when the server overlaps them.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if n > 1 and temperature <= 0.0 and seed is None:
        # Refuse the useless config loudly: N identical candidates.
        raise ValueError(
            "best-of-N with n>1 needs diversity: set temperature>0 or seed!=None"
        )

    def _one(i: int) -> Completion:
        s = None if seed is None else seed + i
        return client.chat(
            messages, max_tokens=max_tokens, temperature=temperature,
            seed=s, enable_thinking=enable_thinking,
        )

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n) as ex:
        comps = list(ex.map(_one, range(n)))
    gen_wall = time.time() - t0
    return comps, gen_wall


_CHAT_INSTR = (
    "Complete the following Python task. Reply with ONLY the complete, "
    "runnable solution (including any needed imports) inside a single "
    "```python code fence. No tests, no explanation.\n\n{prompt}"
)


def best_of_n(
    prompt: str,
    build_test_program: Callable[[str], str],
    *,
    n: int = 4,
    temperature: float = 0.7,
    seed: int | None = 0,
    max_tokens: int = 1024,
    sb_timeout: float = 10.0,
    client: AtlasClient | None = None,
    base_url: str = "http://127.0.0.1:8890",
    model: str = "aeon-27b-dflash",
    enable_thinking: bool = False,
    partial_tests: Sequence[str] | None = None,
    partial_prelude: str = "",
    wrap_instruction: bool = True,
) -> BestOfNResult:
    """End-to-end: concurrently decode N candidates, sandbox-arbitrate, return.

    `build_test_program(candidate_code) -> full runnable program` is the same
    contract the runner uses (Problem.build_test_program), so best_of_n drops
    straight onto HumanEval/MBPP problems.

    Wall-clock accounting is attached to the result: gen_wall_s (concurrent
    generation) vs max/sum candidate latency lets the caller verify the
    "N candidates ≈ 1 candidate wall-clock" claim.
    """
    client = client or AtlasClient(base_url=base_url, model=model)
    content = _CHAT_INSTR.format(prompt=prompt.rstrip()) if wrap_instruction else prompt
    messages = [{"role": "user", "content": content}]

    comps, gen_wall = generate_candidates(
        client, messages, n=n, temperature=temperature, seed=seed,
        max_tokens=max_tokens, enable_thinking=enable_thinking,
    )
    codes = [extract_code(c.text) for c in comps]
    completions = [c.text[-2000:] for c in comps]
    wall_times = [c.wall_s for c in comps]

    t1 = time.time()
    arb = arbitrate(
        codes, build_test_program, sb_timeout=sb_timeout,
        partial_tests=partial_tests, partial_prelude=partial_prelude,
        completions=completions, wall_times=wall_times,
    )
    arb_wall = time.time() - t1

    # Reassemble with full accounting (BestOfNResult is frozen -> rebuild).
    return BestOfNResult(
        winner_index=arb.winner_index,
        code=arb.code,
        passed=arb.passed,
        n=arb.n,
        candidates=arb.candidates,
        any_passed=arb.any_passed,
        n_passed=arb.n_passed,
        gen_wall_s=gen_wall,
        arb_wall_s=arb_wall,
        total_wall_s=gen_wall + arb_wall,
        max_candidate_wall_s=arb.max_candidate_wall_s,
        sum_candidate_wall_s=arb.sum_candidate_wall_s,
    )


# ---------------------------------------------------------------------------
# Live HumanEval driver: n=1 (pass@1) vs n=N (pass-any@N) + wall-clock ratio
# ---------------------------------------------------------------------------
# This half TALKS TO THE GPU SERVER. It is intentionally NOT imported by the
# unit tests (which cover the pure arbiter). Run it in a GPU window with the
# server up and ATLAS_MULTISEQ_GRAPHS=1 so the N requests actually overlap.

def _humaneval_bon(base_url, model, limit, n, temperature, seed, max_tokens,
                   sb_timeout, out_path):
    """Compare best-of-N (n candidates, sandbox-arbitrated) against n=1.

    For each problem we run best_of_n TWICE:
      * n=1  -> the pass@1 baseline (single greedy-ish call).
      * n=N  -> pass-any@N via concurrent generation + sandbox arbitration.
    Reports both correctness numbers AND the wall-clock ratio (bon / single),
    the "~free via concurrency" claim.
    """
    import json
    import statistics

    # eval_datasets lives beside the building blocks in local/evals.
    from eval_datasets import load_humaneval

    problems = load_humaneval(limit)
    client = AtlasClient(base_url=base_url, model=model)

    n1_correct = 0
    bon_correct = 0
    single_walls, bon_walls, gen_walls = [], [], []
    ratios, overlap_ratios = [], []
    records = []

    for p in problems:
        # n=1 baseline. seed fixed; temperature 0 is fine for the single shot
        # (determinism is acceptable at n=1 — no diversity needed).
        r1 = best_of_n(
            p.prompt, p.build_test_program, n=1, temperature=0.0, seed=seed,
            max_tokens=max_tokens, sb_timeout=sb_timeout, client=client,
        )
        # n=N best-of-N. temperature>0 + distinct seeds => diverse candidates.
        rN = best_of_n(
            p.prompt, p.build_test_program, n=n, temperature=temperature,
            seed=seed, max_tokens=max_tokens, sb_timeout=sb_timeout,
            client=client,
        )

        n1_correct += int(r1.passed)
        bon_correct += int(rN.passed)
        single_walls.append(r1.gen_wall_s)
        bon_walls.append(rN.gen_wall_s)
        gen_walls.append(rN.gen_wall_s)
        # Wall ratio: how much MORE wall-clock did N candidates cost vs 1.
        if r1.gen_wall_s > 0:
            ratios.append(rN.gen_wall_s / r1.gen_wall_s)
        # Overlap ratio inside the N batch: sum(latencies)/gen_wall. >1 means
        # the server overlapped them (ideal ~N); ~1 means they serialized.
        if rN.gen_wall_s > 0:
            overlap_ratios.append(rN.sum_candidate_wall_s / rN.gen_wall_s)

        records.append({
            "task_id": p.task_id,
            "n1_passed": r1.passed,
            "bon_passed": rN.passed,
            "bon_n_passed": rN.n_passed,
            "bon_winner": rN.winner_index,
            "single_gen_wall_s": r1.gen_wall_s,
            "bon_gen_wall_s": rN.gen_wall_s,
            "bon_sum_wall_s": rN.sum_candidate_wall_s,
            "bon_max_wall_s": rN.max_candidate_wall_s,
        })
        print(f"[bon] {p.task_id:16s} n1={'P' if r1.passed else '.'} "
              f"bon={'P' if rN.passed else '.'} ({rN.n_passed}/{n}) "
              f"wallN/1={rN.gen_wall_s / r1.gen_wall_s:.2f} "
              f"overlap={rN.sum_candidate_wall_s / max(rN.gen_wall_s,1e-9):.2f}x",
              flush=True)

    npb = len(problems)
    pass1 = n1_correct / npb if npb else 0.0
    passN = bon_correct / npb if npb else 0.0
    summary = {
        "n_problems": npb,
        "n": n,
        "temperature": temperature,
        "pass_at_1": pass1,
        "pass_any_at_n": passN,
        "lift": passN - pass1,
        "median_wall_ratio_N_over_1": statistics.median(ratios) if ratios else None,
        "mean_wall_ratio_N_over_1": statistics.fmean(ratios) if ratios else None,
        "median_overlap_ratio": statistics.median(overlap_ratios) if overlap_ratios else None,
        "total_single_wall_s": sum(single_walls),
        "total_bon_wall_s": sum(bon_walls),
        "aggregate_wall_ratio": (sum(bon_walls) / sum(single_walls)) if sum(single_walls) else None,
    }
    result = {"config": "bestofn_humaneval", "summary": summary, "records": records}
    print("\n=== best-of-N HumanEval ===")
    print(f"  problems           : {npb}")
    print(f"  pass@1  (n=1)      : {pass1:.4f}  ({n1_correct}/{npb})")
    print(f"  pass-any@{n} (n={n}) : {passN:.4f}  ({bon_correct}/{npb})")
    print(f"  correctness LIFT   : {passN - pass1:+.4f}")
    print(f"  wall ratio N/1     : median {summary['median_wall_ratio_N_over_1']}, "
          f"aggregate {summary['aggregate_wall_ratio']}")
    print(f"  intra-batch overlap: median {summary['median_overlap_ratio']}x "
          f"(1.0=serial, {n}.0=perfect overlap)")
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"  wrote {out_path}")
    return result


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(
        description="Concurrent best-of-N with sandbox arbitration (HumanEval)")
    ap.add_argument("--base-url", default="http://127.0.0.1:8890")
    ap.add_argument("--model", default="aeon-27b-dflash")
    ap.add_argument("--limit", type=int, default=60,
                    help="number of HumanEval problems (default 60)")
    ap.add_argument("--n", type=int, default=4, help="candidates per problem")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--sb-timeout", type=float, default=10.0)
    ap.add_argument("--out", default="/tmp/bestofn_humaneval.json")
    args = ap.parse_args(argv)
    _humaneval_bon(
        args.base_url, args.model, args.limit, args.n, args.temperature,
        args.seed, args.max_tokens, args.sb_timeout, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
