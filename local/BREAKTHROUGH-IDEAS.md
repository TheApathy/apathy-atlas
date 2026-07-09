# Breakthrough Ideas Catalog — Atlas/GB10 (captured 2026-07-08)

Unifying principle: **speculation is a general principle, not a decoding trick.** Speculate on any
expensive computation as long as one exact oracle + rollback exists. Our oracle discipline (md5
constitution for lossless paths, ABBA pass@1 CI for quality-gated paths, SSM checkpoints + KV truncate
for rollback) is the asset that makes every idea below safe to attempt.

Physics ground truth: single-stream is weight-bandwidth-bound (14 GB/step @ 273 GB/s ≈ 120 ms floor,
proven at PTX level). Wins = fewer bytes, more tokens per read, or less serial time. Verify computes
K×248,320 logits per step and uses K argmaxes — 99.999% of paid-for intelligence is discarded.

## Tier 1 — mine the discarded verify logits (PURSUING FIRST)
1. **Echo-drafting (Jacobi salvage).** ~~REFUTED as standalone (2026-07-08, measured)~~: salvage accept
   0.85/16 mean, 83% of echo chains die at position 0 — a one-token substitution derails the target's
   continuation nearly always (joins recycle/denoise in the dead post-miss-salvage family, now with
   per-position data). LOSSLESS both ways; machinery committed default-OFF. LIVE VARIANT: as the FREE
   draft inside async propose-verify overlap (#20) — a poor echo costs nothing there, and the measured
   7.7% of >=5-accept salvages are pure profit.
2. **Target-guided branching.** Top-2/3 of verify logits at each position → the K=32 free slots. Branches
   sourced from the target's own second choices; no drafter, no training. Needs deep-tail fix landed.

## Tier 2 — captured 2026-07-08 (first wave)
3. **Optimistic hierarchical execution.** Sparse+W3 ghost of the same weights executes 4 blocks
   optimistically; dense audits 64 tokens in ONE weight read; rollback via existing SSM checkpoints.
   Dense bandwidth amortized 4×/token, lossless (dense = only oracle). The wall-breaker.
4. **State-trajectory extrapolation drafting.** 48/64 layers are linear dynamical systems; extrapolate the
   residual-stream trajectory with an online linear predictor, decode through lm_head — draft the model's
   internal physics without running the network. Training-free EAGLE.
5. **Negative-latency agent prefill.** Prefill predictable next-prompts (tool outputs) during tool-execution
   idle → TTFT ≈ 0 for agent loops. Design doc exists: local/predictive-turn-prefetch-design.md.
6. **Cross-request chain cache.** Persistent token-CDN of accepted chains keyed by context hash across
   sessions. Boilerplate drafts instantly from fleet history.

## Tier 3 — the crazy-but-necessary wave
7. **Serve-time self-distillation (the model shrinks itself in production).** Background loop distills
   adjacent SSM-layer pairs into fused layers from live traffic, hot-swaps one at a time, ABBA-gated,
   instant rollback. 64→~48 effective layers ≈ 25% fewer bytes at the wall, trained on the box's own
   workload. Self-improve loop (proven 2× on the drafter) pointed at the target.
8. **Speculative side-effects (agent co-design).** Run cargo check/tests in the sandbox on PARTIALLY
   generated code; restart on divergence. Execute tool calls at the earliest grammar-unambiguous prefix.
   Speculation escapes the token domain into the world; sandbox rollback = the oracle.
9. **Rejection-driven drafter calibration.** Every rejection = supervised pair (drafter said, target wanted).
   Online logit-bias table → the drafter learns from being wrong, live, gradient-free. Hats without training.
10. **Optimistic streaming with retractions.** Stream drafts to the client pre-verify with a retraction
    protocol. Perceived latency = propose latency (~30ms/16 tok). Safe for agent consumers today.
11. **Weight-tile forensics.** Hash 4-bit weight tiles across layers for near-duplicates → L2-pin dedupe.
    Probably 0%; costs 10 minutes to check.

## Tier 4 — the even-crazier wave (2026-07-08, second brainstorm)
12. **Concurrent best-of-N with sandbox arbitration — BUILT + MEASURED (2026-07-08, local/bestofn.py).**
    HumanEval x40, n=4, temp0.7: pass@1 92.5% -> pass-any@4 97.5% = **+5pt CORRECTNESS LIFT**. Cost NOT
    free: n=4 concurrency overlap = 2.5x (not 4x — thinking-mode chat outputs vary in length so the batch
    doesn't pack perfectly) → ~1.6x generation wall + N sandbox runs. VERDICT: a genuine QUALITY axis
    nothing else touches — +5pt coding correctness for ~1.6x latency, tunable (n=2 ≈ most of lift at ~1.3x).
    Strong for app-building (correct-that-runs > fast-but-wrong). SHIP as an opt-in coding mode.
13. **Span-copy speculation ("edit-script drafting").** For editing/refactoring: the drafter emits ONE
    action — "copy context span [a,b]" — instead of 500 tokens. Chunked K=32 verify windows ride through
    the span at full accept; compose with hierarchical execution (sparse ghost verifies the copy, dense
    audits once) for enormous refactor throughput. SAM retrieval already finds the spans.
14. **Depth-exit hierarchical execution.** Same as #3 but on the DEPTH axis: run layers 1..40 + early
    LM head as the optimistic executor when drafter+early-head agree with margin (24 unread layers ≈ 5 GB
    saved/step); full-depth audit + rollback. Early-exit made LOSSLESS by the audit.
15. **Attention-pointer drafting.** The model's own attention tells us where it's looking in context; draft
    the tokens FOLLOWING the attended position (the model is about to copy/paraphrase from there). Beats
    n-gram retrieval on paraphrase; needs a cheap attention-argmax probe exported from the kernel.
16. **Grammar-constrained drafting.** Apply the grammar bitmask to the DRAFTER during propose so it never
    wastes a slot on syntax the target's grammar would forbid. Free accept in tool-call/JSON mode; easy.
17. **Disaggregated speculation over LAN.** A second box on the tailnet runs the drafter and streams drafts
    (~5ms RTT); GB10 only verifies. Propose cost → 0 without scheduler surgery. Home-lab splitwise.

## Status pointers
- Tier 1 build queued behind the deep-tail agent (scheduler file ownership).
- Sparse-col GEMV microbench + mtp-vocab 96000 A/B + full-164 evals = next GPU batch.
- In flight: deep-tail fix (GPU), IMMA decomposition research, W3 mixed-precision build.

## THE 100-TOK/S CODING RECIPE (dense, honest arithmetic, 2026-07-08)
Today: ~8 accepted / ~190ms step (40ms propose + 150ms verify) = 42 tok/s.
1. Echo-drafting/async overlap (propose→0): step 150ms → 53 tok/s
2. Vocab-cap 96k (kill 6% forced-miss): accept ~9.5 → 63
3. Free-slots + target-guided branching (+30% accept): ~12 → 80
4. W3 mixed precision (~16 tolerant layers, verify 150→130ms): → 92
5. Block-chaining γ→32 / hierarchical sparse execution: accept 14+ → 100+
Five compounding levers, all in flight, each attacking a different term. No miracle required.

## Tier 5 — crazier still (2026-07-08)
18. **Self-batching app generation.** An app = multiple files; generate all N files as N concurrent
    LOSSLESS streams (concurrency proven byte-exact) at 208 aggregate tok/s → wall-clock app builds
    ~2.7x faster with zero single-stream change. Effective coding throughput for app-building ~150+
    TODAY. Requires only agent-side planning to emit independent file tasks.
19. **SSM prefix-state caching — INVESTIGATED 2026-07-08: mechanism EXISTS** (prefix_lookup.rs checks
    ssm_snapshot_tokens; "no SSM snapshot -> recomputing all KV" fallback warning in tree). The open
    question is COVERAGE: how often agent-loop prefix hits find a snapshot vs recompute. Next: telemetry
    run on agent-like traffic (repeat-prefix chat turns), count snapshot-hit vs no-snapshot warnings;
    if poor, widen snapshot save points (per-request-end saves). Downgraded from build to measurement.
20. **Thought memoization.** Cache reasoning traces by problem-shape hash; replay as the DRAFT for the
    thinking span on similar problems → near-full accept when the model has "thought this before."
21. **Persistent-L2 pinning.** cudaMemAdvise persistence window for small-hot tensors (conv1d, norms,
    hot embedding rows). Bounded, grubby, free.
