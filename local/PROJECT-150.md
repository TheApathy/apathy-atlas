# PROJECT 150 — 150 tok/s single-stream, dense 27B, ZERO coherence loss (goal set 2026-07-08)

NON-NEGOTIABLE INVARIANT: the dense model is the ONLY oracle. Every emitted token is the dense model's
greedy argmax (md5 constitution for lossless paths; ABBA CI for anything touching numerics). Coherence
is never traded — only redundant dense weight reads are.

## The physics case (why 150 is real, not hope)
- Today: 14 GB dense read per step / ~8 accepted (coding) = 1.75 GB/token → 42 tok/s.
- Hierarchical execution: half-cost GHOST (W3-mixed + sparse-column, ~7 GB/read) executes blocks
  speculatively; DENSE audits 64 tokens per full 14 GB read (2x chunked K=32 passes), SSM-checkpoint +
  KV-truncate rollback on divergence (machinery EXISTS and is validated).
- Amortized bytes: ghost ~0.44 GB/tok + audit ~0.44 GB/tok ≈ 0.9 GB/token → ~310 tok/s bandwidth
  ceiling; ~150 real after acceptance efficiency (~0.5-0.6). Higher accept → closer to ceiling.

## Phase plan (each phase independently shippable + gated)
### Phase A — acceptance + serial-time stack → 80-90 tok/s coding (ALL IN FLIGHT)
- Echo-drafting (propose→0, verify-argmax salvage) + target-guided branching  [queued behind deeptail]
- mtp-vocab 96000 (kills 6-11% forced-miss)                                   [A/B queued, one flag]
- Free-slots K=32 deep-tail fix (+30-46% coding accept)                       [agent fixing NOW]
- Async propose‖verify overlap (#20) as echo's complement                     [queued]
- Gates: md5 constitution (all lossless).

### Phase B — build the GHOST → the half-cost executor (COMPONENTS IN FLIGHT)
- W3 mixed-precision FFN for tolerant layers (repack tool + w3a16 kernels)    [agent building NOW]
- Sparse-column down_proj GEMV (66-74% sparsity MEASURED, kernels compiled)   [microbench next GPU]
- Ghost config = dense arch + W3-FFN + sparse-down: target ~2x faster forward, same tokenizer/KV/SSM shapes.
- Gate: ghost needs NO quality gate of its own (it is never the oracle) — only speed + its accept-vs-dense.

### Phase C — HIERARCHICAL EXECUTOR (the centerpiece, design first)
- Scheduler: drafter→ghost spec-decode for 4 blocks (64 tok) optimistically; dense chunked audit
  (2x K=32 verify) over the ghost-committed tokens; on first divergence → rollback (SSM checkpoint ring
  + KV truncate + token truncate — all existing) and resume from dense-corrected token.
- Key design questions: checkpoint cadence/ring depth for 64-token rollback; audit batching (2x32 vs
  4x16); ghost/dense state isolation (ghost runs its OWN SSM/KV or re-derives? — cheapest: ghost owns a
  shadow slot, dense audit recomputes exact state as today's verify already does); accept accounting.
- Gate: md5 constitution EXACTLY (output ≡ dense greedy). This is spec-decode with a smarter middle tier —
  same losslessness argument as DFlash today.

### Phase D — keep the audit windows full
- Block-chaining γ→32 (adaptive, high-accept spans), thought memoization for thinking spans,
  SSM prefix-state cache for agent loops (TTFT), self-batching for app-level throughput (~150 effective today).

## Milestones
- M1: Phase A validated → coding ≥ 80 (from 42)                    [target: next 2 GPU windows]
- M2: ghost forward measured ≥ 1.7x dense forward                  [W3 + sparse microbench]
- M3: hierarchical executor md5-exact on counting + coding          [the moment]
- M4: coding ≥ 120, counting ≥ 150                                  [tuning: audit cadence, accept stack]
- M5: 150 sustained on mixed workload, ABBA quality parity, ship.
