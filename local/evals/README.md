# Atlas quality-eval harness

Standalone Python harness that objectively gates whether a lever **changes
output quality**, so we can ship non-lossless speed wins (e.g.
`ATLAS_THINK_SPEC=1`'s ~1.7x thinking speedup) with confidence and validate new
drafters (v5). No dependency on the Rust crates. Server is the OpenAI-compatible
Atlas endpoint at `http://127.0.0.1:8890` (model `aeon-27b-dflash`).

## File tree

```
local/evals/
  client.py          OpenAI-compatible client (stdlib urllib): complete()/chat()
  eval_datasets.py   HumanEval + MBPP loaders (offline sample -> local jsonl -> HF)
  extract.py         code extraction (fenced ```python, bare code, stitching)
  sandbox.py         UNTRUSTED code exec: subprocess + rlimits + timeout + scrub
  score.py           pass@1 / unbiased pass@k (Chen et al.)
  runner.py          dataset -> server -> sandbox -> results.json  (PRIMARY)
  abba.py            paired-bootstrap CI on pass@1 delta (THE SHIP GATE)  (CLI)
  mtbench.py         MT-Bench-style quality, pluggable judge      (SECONDARY scaffold)
  bfcl.py            tool-calling accuracy                        (TERTIARY scaffold)
  gate_think_spec.sh ready-to-run think-spec quality gate         (FIRST CONSUMER)
  data/
    humaneval_sample.jsonl  5 bundled problems (offline)
    mbpp_sample.jsonl       5 bundled problems (offline)
  tests/             pytest, GPU-free, all pass CPU-side
```

## How each eval scores

- **Coding pass@1 (PRIMARY, objective, no judge):** For each problem, prompt the
  server, extract code, build a self-contained program (candidate + the
  dataset's unit tests), run it in the sandbox; exit 0 == pass. pass@1 = mean
  over problems. With `--n>1` at temperature>0, uses the unbiased pass@k
  estimator `1 - C(n-c,k)/C(n,k)`.
- **ABBA (statistical ship gate):** Given results A (lever off) and B (lever on)
  over the SAME problems, PAIRED per task_id, computes delta = pass@1(B) -
  pass@1(A) and a bootstrap 95% CI by resampling problems with replacement
  (paired: both A and B outcomes taken per resampled problem, preserving the
  shared-problem variance). Verdict "B not worse than A" iff CI lower bound >
  -epsilon (default 1%). Stats are decoupled from booting — abba.py takes two
  result files, so results survive and the CI is re-runnable on CPU.
- **MT-Bench (SECONDARY scaffold):** generate answers to a tiny open-ended set,
  score 1-10 with a pluggable judge (`server` self-judge or `none`). Directional
  only; NOT the ship gate.
- **BFCL (TERTIARY scaffold):** AST-match tool call (name + args) vs gold on a
  handful of single-function cases. `text` mode works today; `native`
  (grammar-enforced tool_calls) needs the client to forward `tools=[...]` once
  the live response shape is confirmed.

## Sandbox safety (runs UNTRUSTED model output)

Model-generated code is treated as hostile. Isolation is defense-in-depth
(see the module docstring in `sandbox.py`):

1. **Separate subprocess** — never `exec()` in-process; a crash/segfault/
   recursion cannot take down the harness. Runs with `python -I -B` (isolated
   mode: ignores `PYTHON*` env, no user site, cwd off path; no bytecode).
2. **Hard wall-clock timeout** — `subprocess.run(timeout=...)`; child makes its
   own session (setsid) so the whole tree can be killed.
3. **rlimits set in child pre-exec** — `RLIMIT_CPU` (busy-loop), `RLIMIT_AS`
   (allocation bombs), `RLIMIT_FSIZE` (disk writes), `RLIMIT_NPROC` (fork
   bombs), `RLIMIT_CORE=0`.
4. **Scrubbed env + throwaway cwd** — minimal PATH only, proxies neutralized,
   temp dir deleted after.
5. **Network** — rlimits cannot block sockets. For a HARD network cutoff set
   `EVALS_UNSHARE_NET=1` (wraps the child in `unshare -n`, no interfaces).
   HumanEval/MBPP tests are self-contained and do no I/O, so process+rlimit
   isolation is the appropriate bar; the unshare hook is there for paranoia.

## Running the think-spec gate (next GPU window)

`ATLAS_THINK_SPEC` is a BOOT-TIME env var, so the harness cannot flip it live —
it orchestrates two boots. In a GPU window (NOT while v5 training owns the GPU):

```bash
# Full pipeline: boots A (no lever) and B (ATLAS_THINK_SPEC=1), runs
# HumanEval+MBPP pass@1 at temp=0 for each, then the ABBA CI. Exits 0 to SHIP.
bash local/evals/gate_think_spec.sh
#   PORT=8890  LIMIT=  (blank=full)  EPSILON=0.01  ITERS=10000
```

Or do it by hand:

```bash
# ARM A: boot baseline, then
python3 local/evals/runner.py --dataset both --label A_baseline \
    --out /tmp/resultsA.json --temperature 0 --seed 0
# ARM B: reboot with ATLAS_THINK_SPEC=1, then
python3 local/evals/runner.py --dataset both --label B_thinkspec \
    --out /tmp/resultsB.json --temperature 0 --seed 0
# Stats (CPU, re-runnable):
python3 local/evals/abba.py /tmp/resultsA.json /tmp/resultsB.json --epsilon 0.01
```

Notes: **temp=0** for reproducibility. Coding pass@1 uses `/v1/completions` (no
thinking) and proves the lever doesn't regress non-thinking coding. To exercise
the thinking path itself, also compare chat `enable_thinking=true` outputs A-vs-B
(reasoning tokens may differ — spec reorders decode — but final content should be
semantically equivalent); see STEP 4 in the gate script.

## Validating a new drafter (v5)

Same ABBA flow: A = current drafter, B = v5 drafter (set the drafter env in the
serve boot). pass@1 must be non-worse (lossless verify already guarantees byte
equality for lossless drafters; ABBA catches quality drift for non-lossless
configs). Full HumanEval (164) + MBPP gives the tightest CI.

## Full datasets (optional)

The bundled samples make everything run offline. For the real numbers, either
drop `data/humaneval.jsonl` / `data/mbpp.jsonl` in place, or let the `datasets`
library fetch them online (auto-used when present and `EVALS_NO_HF` is unset).

## Tests (GPU-free)

```bash
cd local/evals && python3 -m pytest tests/ -q   # 29 passed
```
Covers: sandbox (pass/exception/syntax-error/timeout/CPU-rlimit/mem-rlimit/
isolated-env), code extraction (fenced/bare/prefer-python/stitching/stop), pass@k
math, HumanEval+MBPP known-correct vs known-wrong scored through the sandbox,
canonical solutions all pass, and ABBA (identical A==B straddles 0; known +/-
delta bracketed; tiny regression shippable; percentile helper; file-based pairing).
