# Model Card — AEON-Q36-27B on Atlas/GB10 (the served stack)

*Last updated 2026-07-08. Every number below is measured on this box, with methodology + gate named.
Result provenance: /path/to/aeon-tps/VALIDATION-3{5,6}-RESULTS.md, AGENTIC-BATTERY-RESULTS.md.*

## Identity
- **Target model:** AEON-Q36-27B-Full — DENSE 27B (Qwen3.6-class, 0 experts, all weights active/token),
  64 layers (16 full-attention + 48 gated-delta SSM), hidden 5120, vocab 248,320, NVFP4 (~14 GB),
  text + vision. Abliterated checkpoint.
- **Drafter:** drafter-v5-goheavy (PRIVATE — never published). 6-layer DFlash block-diffusion, γ=16,
  NVFP4, trained on 8,712 self-generated NVFP4-aligned samples (the self-improving loop, 2 generations).
- **Engine:** Atlas (pure Rust + CUDA), branch qwen.
- **Hardware:** 1× DGX Spark GB10 — 273 GB/s LPDDR5X unified, 48 SMs, no FP4 tensor cores.
  Naive dense AR ceiling on this hardware: ~19 tok/s. Everything above it = lossless speculation.

## Throughput (single stream, temp 0, serve-script defaults)
| Modality | tok/s | Harness | vs naive dense ceiling |
|---|---|---|---|
| Coding (1200-tok gen) | **48.2** | bench_wave2, n=5 | 2.5× |
| Counting/structured | **82–85** | bench_wave2, n=5 | 4.4× |
| Prose | **15.3** | bench_spec, n=3 | 0.8× |
| Code editing (retrieval) | **~71** | add_method bench | 3.7× |
| Thinking-inclusive | **40.0** | chat + enable_thinking | 2.1× |

**Concurrent:** 208 tok/s aggregate @ c=8 (2.71×), every stream byte-identical to single-stream
(verified incl. mid-batch compaction). Multi-user config: ATLAS_MULTISEQ_GRAPHS=1 (+SSM flags).

## Quality
| Eval | Score | Method |
|---|---|---|
| HumanEval pass@1 | **95.0%** (n=60; full-164 PENDING) | chat mode, temp 0, sandboxed execution |
| HumanEval thinking | **97.5%** (n=40) | enable_thinking + THINK_SPEC |
| MBPP-sanitized | PENDING (battery running) | 257 problems |
| Tool-calling (Toolery, 4 tiers) | PENDING (battery running) | 143 deterministic scenarios |
| App-build smoke (multi-file + tests) | PENDING (battery running) | sandbox-executed |

## Integrity guarantees (what "lossless" means here)
- **Bit-losslessness:** greedy counting output md5 == `91a6ff90d50736f779c09db67a96db2d` through every
  shipped lever (spec decode, drafter swaps, WY17-lazy, concurrency, 96k vocab). The dense target's
  greedy argmax authors every emitted token; drafters/speculation change SPEED only.
- **Quality-gated (not bit-lossless):** ATLAS_THINK_SPEC=1 (batched-verify numerics differ during
  thinking; gated by ABBA paired-bootstrap: 97.5% vs 95.0%, not-worse CI). Pending: its TOOL-CALLING
  gate (battery running — default may roll back if it degrades calls).
- Known measurement caveats: coding output is run-to-run nondeterministic at temp 0 (near-tie flips);
  counting is the only bit-constitution; quality gates use paired statistics.

## Serving defaults (shipped, each with evidence)
DRAFT_MODEL=v5-goheavy (ABBA CI[0,0]) · ATLAS_WY17_LAZY=8+COMMIT (md5-lossless, +2-7%) ·
ATLAS_THINK_SPEC=1 (quality-gated) · MTP_VOCAB=96000 (coding +11-70% by harness; kills the 6%/11%
code/prose proposal-impossibility tax; counting −3%) · ATLAS_DFLASH_SAM=1 adaptive ·
splitK/fused-gateup kernel family (md5-exact).

## Known limitations
- Prose acceptance is the weak modality (drafter guesses rare tokens poorly even now that it CAN propose them).
- Long-context (>9k) acceptance unmeasured (probe queued).
- Streaming is not incremental (client-side tok/s reads falsely low — fix queued, task #41).
- Server stability: one known SSM-slot leak in exotic configs (upstream fix pending merge, #24/#32).
- Abliterated base: refusal behavior removed by upstream checkpoint; quality measured, safety not re-evaluated.

## Roadmap anchors (PROJECT-150)
Async propose‖verify (instrumenting) → W3+sparse ghost (components built; sidecars packed) →
hierarchical optimistic execution (designed) → 150 tok/s single-stream target with the dense model
remaining the only oracle. Aggregate: cross-seq batched verify (designed) → 400+.
