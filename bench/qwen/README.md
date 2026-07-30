# Qwen3.6-27B champion — DFlash reproduction harness

This directory reproduces the **champion** speculative-decoding configuration for
a dense Qwen3.6-27B target with a DFlash drafter, measured single-stream on
GB10-class hardware (Grace Blackwell, `sm_121f`, unified LPDDR5x).

It is the sibling of `bench/laguna/` on the `laguna` branch and is deliberately
the same shape: one `env.sh` that owns every path and setting, one self-verifying
launcher, one deterministic decode benchmark. It is **not** a re-parameterization
of that harness. The two stacks disagree on γ, on KV dtype, on which log lines
carry the accept counter, and on which gates are safe to enable. Settings do not
transfer between them; see [Traps](#5-five-things-that-will-cost-you-a-day).

---

## 1. Prerequisites

| What | Notes |
|---|---|
| A CUTLASS source checkout | `CUTLASS_HOME` must point at it. If you have built vLLM from source you already have one under its `.deps/`. |
| Target checkpoint | A Qwen3.6-27B snapshot directory (must contain `config.json`). |
| DFlash drafter checkpoint | The matching DFlash drafter snapshot directory. |
| CUDA toolkit + a Blackwell-class GPU | ~50 GB of free memory at launch; γ=16 inflates the SSM-MTP pool and KV by roughly 24 GB over a non-speculative serve. |

Nothing here has a machine-specific default. Every path comes from the
environment, and the scripts refuse to guess:

```bash
export QWEN_MODEL=/path/to/Qwen3.6-27B-target
export QWEN_DRAFT=/path/to/Qwen3.6-27B-DFlash-drafter
export QWEN_BIN=/path/to/spark          # optional; defaults to ./target/release/spark
```

## 2. Build

```bash
CUTLASS_HOME=/path/to/cutlass bash bench/qwen/build_cutlass.sh
```

A bare `cargo build --release` is **not** equivalent. It produces a `spark` that
defaults to a different kernel target, cannot load this model, and fails only at
serve time. The build script sets the four `ATLAS_TARGET_*` variables that select
the `kernels/gb10/qwen3.6-27b/nvfp4` kernel set, and asserts that set exists
before starting a twenty-minute compile.

## 3. Serve

```bash
bash bench/qwen/serve_champion.sh
```

This launcher asserts its own invariants and refuses to report a healthy serve
unless all of them hold. Every assert exists because its absence once produced a
clean-looking but wrong measurement:

1. **Gate parity (pre-launch).** `env.sh`'s `qwen_champion_env()` is a second copy
   of the configuration in `local/serve-aeon-27b-dflash.sh`, the script the
   published numbers were measured on. `verify_gate_parity.sh` re-derives the
   gate set from that launcher on every run and refuses to serve on any
   difference. It runs *before* the preflight kill, so a config error does not
   cost you a running serve. This one is not hypothetical: eleven gates were
   missing from `env.sh` for a while, because the check that cleared it had been
   pointed at a stale copy of the launcher and so agreed with itself. Run
   `bash bench/qwen/verify_gate_parity.sh --self-test` to confirm the comparator
   fires on a planted divergence before you trust it passing.
2. **Gate guard (pre-launch).** Every `ATLAS_*` gate the stack exports must exist
   as a string in the binary. A gate the binary cannot read is a silent no-op,
   and the serve is then healthy while running a configuration nobody asked for.
   The gate list is derived from the function body at runtime, never from a
   hand-maintained copy.
3. **Kernel target.** Matched as a prefix — the serve prints `<model>, <quant>`,
   and pinning the quant suffix would produce a false failure the day it changes
   for an unrelated reason.
4. **Determinism.** The same temperature-0 prompt twice must be byte-identical.
   Without this, every hash comparison in the harness is measured against a
   moving target.
5. **No SSM corruption signature.** See trap 1 below — the failure mode is a run
   of `!` in otherwise valid output, not a crash.
6. **Speculation actually ran, at width 16.** A serve that silently decodes
   serially is fast enough to look plausible and produces numbers that are not
   about DFlash at all.

## 4. Measure

```bash
python3 bench/qwen/decode_bench.py --tag champion \
    --log bench/qwen/ab/serve-champion.log \
    --json-out bench/qwen/ab/champion.json \
    --dump-text bench/qwen/ab/text
```

Six prompts ordered easy → hard, temperature 0, thinking off. Each row is
classified DFlash-or-serial **before** it is aggregated, and a row that cannot be
graded is reported as `UNGRADED` rather than folded into either bucket. A count
of zero must never be readable as "nothing wrong".

### Files

| File | Purpose |
|---|---|
| `env.sh` | Every path, port, gate and geometry setting for the harness. |
| `verify_gate_parity.sh` | Re-derives the gate set from `local/serve-aeon-27b-dflash.sh` and fails on drift. That launcher, not `env.sh`, is the source of truth. |
| `build_cutlass.sh` | Release build with the correct kernel target. |
| `serve_champion.sh` | Self-verifying launcher (the six asserts above). |
| `benchenv.py` | Log scraping. Owns the accept anchor — read it before editing a regex. |
| `decode_bench.py` | The six-prompt deterministic decode benchmark. |

---

## 5. Measured numbers

All figures below are from `results.md` in this repository, which is the
measurement record these scripts were built around. They are **content-mix
dependent**: this stack's throughput varies by more than 4× across prompt types,
so a single headline number would be a fiction.

| Content | tok/s | Source |
|---|---|---|
| Counting (highly predictable) | ~73 baseline, up to ~80–83 | `results.md` (records-grade baseline; ~83 named as the ceiling) |
| Novel coding | 17.8 → ~18.6 | `results.md` — PPL-neutral accept-gate relaxation, +4.5% |
| Prose | ~12.9 | `results.md` records-grade baseline |

Acceptance on novel coding sits at roughly **3.35 of 16** drafted tokens. Raising
it was attempted repeatedly and is documented as a dead end:

> the ~3/16 coding floor is drafter-**INTRINSIC**; the unlock to ≥13/16 /
> 30–80 tok/s is a drafter **RETRAIN** (or a bit-exact tree-verify kernel), not
> an accept-gate relaxation.
> — `results.md`

Two levers were measured and are worth knowing about because they look promising
and are not:

- **Relaxed accept gate.** PPL-neutral settings buy +1.7–4.5% on coding and
  nothing more; pushing harder reaches ~19.5 tok/s but at measurable perplexity
  drift and counting corruption. On the high-entropy tokens where the drafter
  actually misses, the draft is far down the target's distribution — not a
  top-2/3 near-miss the gate could rescue.
- **Adaptive γ.** No coding gain, and counting collapses from ~80 to ~52 tok/s.

Quality gates for the two non-lossless gates in the stack, from the launcher's
own record: `ATLAS_THINK_SPEC` scored HumanEval thinking-mode pass@1 **97.5%
with** the flag versus **95.0% without** (CI `[0, +0.075]`, "not worse" → ship),
and the shipped drafter checkpoint scored **95.0% vs 95.0%**, delta CI `[0,0]` —
identical outputs on every problem.

---

## 6. Five things that will cost you a day

**1. γ=16 is a correctness constraint, not a tuning knob.**
γ=16 plus the prefix gives K=17 verify tokens, and K=17 is the only width routed
through the fused `gdn_wy17_k` SSM kernel that saves all 17 intermediates. Every
K in 5..16 falls through to a sequential per-token SSM path that produces NaN at
positions K-3..K-1 and corrupts the SSM rollback. It does not crash — the target
emits a correct first token followed by a run of `!`. `env.sh` refuses to launch
a `DRAFT_CAP != γ` pair, and `serve_champion.sh` separately checks the *output*
for the signature, because the first guard only checks what was set.

**2. The M_TILE=16 NVFP4 attention path corrupts at K=17.**
`ATLAS_TC_NVFP4_M16` and `ATLAS_TC_NVFP4_M16_MS_ATTN` are exported as `0` on
purpose and are not dead entries to tidy away. With them on, verify's deep-slot
argmax repeats earlier digits, greedy determinism breaks across requests, and
acceptance collapses from ~15.6/16 to ~1.5/16 — which reads exactly like a
drafter-quality regression and sends you after the wrong thing entirely. Every
other combination of the KGAMMA/TRANSPOSED gates is token-exact. It bought about
20 ms of verify time.

**3. The drafter must be NVFP4; bf16 is not a slower fallback, it is unusable.**
With `ATLAS_DFLASH_QUANT=bf16` every drafter projection runs a dense GEMM over
the full context and propose time scales to ~1.75 s per step at sequence length
800 — coding prompts simply time out. Drafter precision moves *acceptance*, never
committed output: the target's logits are the source of truth either way.

**4. A bare `accepted=` regex double-counts every step.**
Three different log families contain that substring, and two of them are emitted
once per step when `ATLAS_DFLASH_STEP_TIMING=1`. Counting both leaves the
histogram's shape and the mean accept unchanged and merely doubles `steps` — so
the token-accounting guard reads ~2.0 instead of ~1.0 and every row is
misreported as "not really DFlash". `benchenv.py` anchors on the `N/M (P%)` shape
that uniquely identifies the ungated verify line, and the anchor is tested
against a fixture containing all three families.

**5. Phase timings here are microseconds, and only exist when you ask for them.**
The champion config does not set `ATLAS_DFLASH_STEP_TIMING`, so `verify_ms` and
`propose_ms` are legitimately absent on a normal run and render as `-`, never as
`0.0`. An unmeasured phase must not be readable as a measured zero. Set
`QWEN_STEP_TIMING=1` for the breakdown, and note that doing so is a deviation
from the configuration these numbers describe.
