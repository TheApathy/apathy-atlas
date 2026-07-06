# Decoding-Efficiency Wave — Levers 1–3 (env-gated, default-OFF)

All three levers are OUTPUT-SHAPING (they change the thinking-span token
stream), so they are gated by **eval quality**, not md5. With every ATLAS_*
var unset the code is inert and counting-eval md5 is byte-identical.

## Levers & env vars

| Lever | Env var | Effect |
|-------|---------|--------|
| 1. Hesitation penalty | `ATLAS_HESITATION_PENALTY=<float>` | Subtract `<float>` from ~50 hesitation-token logits (`wait`/`but`/`alternatively`/`however`/`actually`/`hmm`/`let me reconsider`-class, incl. leading-space + capitalization variants) during the `<think>` span only. |
| 2. Soft `</think>` exit bias | `ATLAS_THINK_EXIT_BIAS=<max>` + `ATLAS_THINK_SOFT_START=<tokens>` | Add a positive bias to the `</think>` logit that grows linearly from 0 at `<tokens>` thinking-tokens to `<max>` at the hard budget. Soft landing vs blunt truncation. |
| 3. Adaptive thinking budget | `ATLAS_ADAPTIVE_THINK=1` | Probe difficulty over the first 48 thinking tokens (mean top-1 confidence); easy (high conf) → budget × 0.4, hard (low conf) → budget × 1.5, clamped to `[32, base]`. |

### Lever 3 difficulty-signal note (honest deviation from brief)

The brief specified the drafter accept-rate over the first ~48 thinking tokens
as the difficulty signal. Investigation showed speculative decoding (DFlash /
MTP) is **bypassed during the thinking span** — `verify_dflash_step::dflash_masked_accept`
returns early on `inside_thinking`, and `scheduler::run`'s MTP dispatch gates
on `!inside_thinking`. So no drafter accept-rate exists while `inside_thinking`.
The implemented signal is the **per-token top-1 softmax confidence** (already
computed in the F2 path), which is available every thinking token and is the
same monotone difficulty proxy (easy reasoning = high confidence). The
`DifficultyProbe` API is signal-agnostic (`observe(f32 in [0,1])`), so swapping
in a real accept-rate later is a one-line change at the call site.

## Where it lives

- `crates/spark-server/src/scheduler/thinking_efficiency.rs` — config parse,
  hesitation-id builder, exit-bias ramp, adaptive-budget math, `DifficultyProbe`,
  `top1_confidence`. All pure, unit-tested (`thinking_efficiency_tests.rs`, 34 tests).
- Startup wiring: `main_modules/serve_phases/tokenizer_runtime.rs` (builds the
  hesitation id set from the tokenizer, installs the config OnceLock).
- Apply site: `scheduler/decode_logits_seq.rs` (`process_seq_logits`), in the
  existing `inside_thinking` logit-shaping block next to F1 reflection-suppress.
- Per-seq state: `ActiveSeq.difficulty_probe` (`scheduler/types.rs`).

## Stats-log samples

Startup (flags on):
```
INFO Hesitation penalty ENABLED (ATLAS_HESITATION_PENALTY=2.5): 48 token IDs resolved
INFO Soft </think> exit bias ENABLED (ATLAS_THINK_EXIT_BIAS=6, soft_start=256 tokens): linear ramp to budget
INFO Adaptive thinking budget ENABLED (ATLAS_ADAPTIVE_THINK=1): difficulty-scaled over first 48 thinking tokens
```

Per-request (Lever 3 commit, INFO):
```
INFO ADAPTIVE_THINK: rescaled thinking budget from difficulty probe seq_thinking_tokens=48 mean_confidence=0.91 base_budget=2000 adaptive_budget=800
```
(Hard prompt example: `mean_confidence=0.42 base_budget=2000 adaptive_budget=2000` — 1.5× clamped to base ceiling.)

Per-token shaping (Lever 1/2, TRACE only — off the hot path by default):
```
TRACE THINK_EFFICIENCY: shaped 49 logit slots thinking_tokens=300 touched=49
```

## Validation plan (run in the next GPU window — DO NOT run now)

1. **md5 parity (all flags OFF):** counting eval md5 must be **byte-identical**
   to the pre-change baseline. This is the safety gate — the module is inert.
2. **Thinking-token reduction (flags ON):** on a fixed prompt set, expect
   **−15…30%** thinking tokens per response with `ATLAS_HESITATION_PENALTY=2.5`
   + `ATLAS_THINK_EXIT_BIAS=6 ATLAS_THINK_SOFT_START=256`, and additionally
   with `ATLAS_ADAPTIVE_THINK=1`.
3. **Answer-quality spot set:** confirm no regression on a small hand-checked
   set (math/code correctness) — output shaping must not degrade answers.
4. **#36 eval-gate note:** run the #36 eval gate with each lever independently
   and combined; record token-count delta + quality.

## Test checklist (already passing, CPU-side)

- [x] `variants_*` — hesitation variant expansion (space + capitalization).
- [x] `build_ids_*` — single-token filter + dedup from a fake tokenizer.
- [x] `parse_*` — env parse rules (unset inert, 0.0 penalty off, exit-bias > 0,
      junk falls back).
- [x] `exit_bias_*` — ramp math (before soft_start = 0, at budget = max, linear
      midpoint, budget=None = 0, degenerate window, monotonic).
- [x] `adaptive_*` — budget scaling (easy shrink, hard clamps to ceiling,
      midpoint, floor, signal clamp).
- [x] `shaping_*` — apply-site (noop when inactive, penalty applied,
      out-of-range id skipped, exit bias needs think_end id).
- [x] `top1_*` — confidence (uniform low, peaked high, empty/-inf = 0).
- [x] `probe_*` — DifficultyProbe (not-ready before window, commits once,
      averages, ignores past-window, hard grows).

34/34 pass: `ATLAS_TARGET_MODEL=qwen3.6-27b cargo test --release -p spark-server thinking_efficiency`.

## Lever 3 (predictive turn prefetch): see `predictive-turn-prefetch-design.md`
Delivered as a design doc (> 1 day to implement correctly — no retained per-turn
decode state, no next-turn signal, cancellation/block-accounting hazards).
Idle hook point identified: `scheduler/mod_helpers.rs:76` (`cv.wait`).
