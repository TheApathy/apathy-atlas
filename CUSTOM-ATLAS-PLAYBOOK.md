# Custom-Atlas Playbook — the GB10 delta over public Atlas

Generated from the real diff: **`origin/main` (public Avarok/atlas) → `qwen`**.
Divergence `ddc7080f`; delta **564 files, +113,917 / −3,222**, **203 new `ATLAS_*` tuning levers**, and
a family of new GB10 CUDA kernels. This is the anatomy of "why our Atlas runs high" — a starting point
folks can reproduce, without shipping the raw source.

> **This file is LOCAL / internal.** It captures the *shape* of the modifications. Decide deliberately
> what (if any) subset is published — the exact champion flag values, the tuned drafter, and the
> measured ceiling are the competitive core.

---

## 1. Where the delta lives (by subsystem)

| Subsystem | Files | What changed |
|---|---|---|
| `kernels/gb10/common` + `.../qwen3.6-27b/nvfp4` | ~85 | new NVFP4/FP8 CUDA kernels (below) |
| `research/ddtree_port` | ~100 | tree-drafting (DDTree) port + prototypes + tests |
| `crates/spark-server/src/scheduler` | 29 | DFlash verify scheduler (batched verify, branch, adaptive γ) |
| `crates/spark-model/src/layers/{ops,qwen3_ssm,qwen3_attention,dflash_head,moe}` | ~70 | layer routing to the fast kernels + drafter head |
| `crates/spark-model/src/{model,weight_map}` | ~20 | weight loading, NVFP4/FP8 routing, KV-precision-per-layer |
| `local/evals`, `bench/`, `tools/dflash_layer_diff` | ~60 | the measurement harness (sweeps, retrain, layer-diff, quality gates) |

## 2. The lever families (203 flags → 6 groups)

The knobs cluster into six categories — this *is* the tuning surface:

1. **DFlash spec-decode (79 levers)** — the biggest lever. Batched verify, adaptive γ, branch/caterpillar
   drafting, thinking-time spec, context window, accept-fallback. Acceptance = free throughput.
2. **FFN / kernel routing (18) + split-K (8)** — route the slow dense-BF16 GEMMs (SSM out-proj, QKVZ,
   FFN gate/up/down, lm_head, paged-decode) onto fast NVFP4 **split-K** kernels. Mostly bit-exact.
3. **DDTree tree-drafting (13)** — budget, top-k, DFS-reorder, tree-aware verify. Large-budget tree
   drafting for higher acceptance on branchy (code) workloads.
4. **Attention (13)** — QKV split-K / fused / mega / M16, sliding-window, K-γ.
5. **SSM (11) + WY17 (6)** — gated-delta-rule multi-seq + WY17 lazy-commit (skip per-token state writes,
   reconstruct on partial accept). The SSM path is ~50% of the step at the bandwidth wall.
6. **Numerics (KV 11, NVFP4/E2M1/TC_NVFP4 ~8, MTP 5, THINK 5, SAM 2)** — per-layer KV precision,
   W4A4 E2M1 GEMM, MTP depth, thinking-decode, SAM retrieval.

## 3. New GB10 kernel families

- **`inferspark_prefill_paged_*` (11 "turbo" variants)** — bf16k/fp8k × turbo2v…8v paged prefill.
- **`gated_delta_rule_tree{,_wy}.cu`** — ancestor-correct SSM recurrence for tree drafting + WY17 lazy.
- **`causal_conv1d_*_multi_seq.cu`** — multi-sequence conv (concurrency).
- **`w4a16_gemm_v2`, `w4a16_gemv_sparse_cols`, `w3a16_{gemv,gemm}`** — NVFP4 + sub-4-bit GEMMs.
- **`tree_kv_scatter`, `mla_prefill`, `tq_plus_innerq`, `vision_encoder`** — supporting kernels.

## 4. How to reproduce this class of tuning (the method, not the answer key)

1. **Start from the public recipe** (`@atlas/qwen3.6-27b-nvfp4`) and build with `ATLAS_TARGET_MODEL=qwen3.6-27b`.
2. **Profile per-kernel** to find the slow GEMM (the SSM out-proj / FFN are usual suspects at the BW wall).
3. **Flip its split-K / NVFP4 fast-path flag**, A/B tok/s, and **md5-gate** (bit-exact) or **pass@1-gate**
   (quality-preserving) the output. Keep only wins that hold the gate.
4. **Raise acceptance** with a DFlash drafter matched to your *served precision* (NVFP4-captured target
   hiddens, not BF16 — mismatched precision degrades acceptance).
5. **Sweep** γ / K-γ / budget per modality (`bench/sweep_*.sh` pattern) — the optimum differs for
   code vs counting vs prose; don't leave a global setting on that costs another modality.
6. **Audit kernel-load health** every serve (embedded / used / missing-fallback) so nothing silently
   drops to generic dispatch.

The public challenge's [`harness/ENGINES.md`](.) + `kernel_health.py` package steps 1–3 and 6 for
anyone; steps 4–5 are where the retraining/sweeping work is.

---

## 5. Publish decision (for the maintainer)

Three tiers, increasingly sensitive:

- **Tier 1 — method (safe to democratize):** the 6 lever *families* + the profile→flip→gate loop above.
  This is genuinely a starting point and reveals no specific config. *(Already the shape of the public
  `MODIFIED-ATLAS.md` guide.)*
- **Tier 2 — the lever names:** the 203 `ATLAS_*` flag names + which subsystem each hits. Reveals the
  optimization surface; competitors learn *what* we tune.
- **Tier 3 — the champion answer key:** the exact winning flag values, the tuned drafter weights, the
  measured ceiling. This is the edge.

Recommendation: publish **Tier 1** (democratizes the "how" and gives folks a real on-ramp) and hold
Tier 2/3 unless you explicitly choose to open them. Say the word and I'll shape the public doc to
whichever tier you pick.
