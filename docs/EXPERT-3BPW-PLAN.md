# 3.0 bpw Trellis Experts — Feasibility & Plan (2026-08-10)

Goal: cut MoE expert bytes from 4.02 GB/token (MXFP4+E8M0, ~4.25 bpw
effective) to ~3.0 bpw so plain decode can pass the current 34 tok/s byte
ceiling toward 28+ tok/s measured (`docs/DECODE-WATERFALL-2026-08-10.md`,
lever #4: −6.8 ms → 29.9 tok/s arithmetic). This document answers: what the
reference stack's "3.0bpw Trellis" actually is, how we obtain the checkpoint,
what the kernels cost on GB10, the quality gate, and the phasing.

Sources examined (read-only, on this box):

- `/home/flocka/sparkinfer-ref` — 0xSero serving recipe + the **actual
  quantized checkpoint on disk** under `data/tp1/` (~106 GB, root-owned)
- `/home/flocka/sparkinfer-upstream` — SparkInfer/B12X kernels (CuTe DSL +
  vendored ExLlamaV3 CUDA)
- `/home/flocka/models/DeepSeek-V4-Flash-162B` — our 88 GB FP8 master
  (144 routed experts, top-6, 43 layers, hidden 4096, moe_inter 2048)

---

## 1. FORMAT — what "3.0bpw Trellis" actually is

It is **EXL3** (TurboDerp / ExLlamaV3), a QTIP-style **trellis-coded
quantization**. It is NOT a grouped-int + LUT scheme: there are **no group
scales, no zero points, no codebook table in memory**.

Hard evidence from the checkpoint
(`/home/flocka/sparkinfer-ref/data/tp1/quantization_config.json`):

```
"quant_method": "exl3", "bits": 3.0, "codebook": "mcg",
"format": "exl3-trellis",
"exllamav3_revision": "787d1582267117d6ee83c90014f03b525b14754f",
"source_format": "packed_e2m1_fp4_with_ue8m0_scales"
```

Per-matrix tensor layout (safetensors header of
`data/tp1/exl3-layer-005-tp1-rank0.safetensors`), for w1 (K=4096 → N=2048):

| field | dtype | shape | meaning |
|---|---|---|---|
| `trellis` | I16 | `[256, 128, 48]` = `[K/16, N/16, 48]` | 48 u16 = **96 B per 16×16 tile** = exactly 3.000 bits/weight, bit-packed |
| `suh` | F16 | `[4096]` = `[K]` | input-side random sign vector (with Hadamard rotation) |
| `svh` | F16 | `[2048]` = `[N]` | output-side sign vector |
| `mcg` | I32 | scalar | codebook selector/marker (multiplicative congruential) |

Mechanics (vendored ExLlamaV3 source, MIT-licensed, at
`/home/flocka/sparkinfer-upstream/b12x/gemm/trellis_linear/csrc/vendor/quant/`):

- **Decode**: each weight is a 16-bit window into the tile's bit-stream,
  windows overlap and advance by `bits` (=3) per weight
  (`exl3_dq.cuh: dq/dq2/dq4/dq8`). Window → FP16 value via the "3INST"
  hash decode (`codebook.cuh: decode_3inst`, cb=1 "mcg"):
  `x *= 0xCBAC1FED; lop3(x, 0x8fff8fff, 0x3b603b60); w = lo_half + hi_half`.
  ~3 ALU instructions per weight plus one funnel-shift for window extraction.
- **Scale-free**: weights are quantized in a rotated space —
  `W ≈ diag(suh)·H·Ŵ·H·diag(svh)` where H is a Hadamard transform
  (`hadamard_inner.cuh`). At run time the *activation* gets sign-flip +
  Hadamard before the GEMV and the output gets the inverse. This replaces
  all group scales; that is why the payload is exactly 3.0 bpw.
- **Overhead**: per expert triplet (w1/w2/w3) suh+svh = 36,864 B + 3×4 B mcg
  vs 9.437 MB trellis payload → **0.39 %**; effective **3.012 bpw**.
- **Allocation**: the format supports non-uniform per-expert bits (SparkInfer
  has a mixed K3/K4 one-launch MoE kernel:
  `b12x/moe/_shared/kernels/w4a16/mixed_trellis.py`, `tier0_bits/tier1_bits`),
  but the shipped artifact is **uniform K=3**: `tier-bitmap-3.0bpw.json` has
  216 experts × 43 layers = 9,288 triplets, all `k=3`
  (`EXL3_MANIFEST.json: achieved_routed_bpw: 3.0`).
- **What stays high precision** (tensor index of the reference checkpoint):
  only routed expert `w1/w2/w3` are EXL3. Attention (wq_a/wq_b/wkv/wo_a/wo_b),
  **shared experts**, gate, embed, head all remain FP8 e4m3 + ue8m0 128×128
  block scales (`base_quantization_config`).
- SparkInfer's serving kernels decode the same trellis to **E4M3 bytes** and
  feed `m16n8k32` MMA fragments (`b12x/moe/_shared/kernels/trellis_decode.py`,
  `b12x/_lib/intrinsics.py` §"TRELLIS-3.0 (QTIP/EXL3 3INST cb=1)") — i.e. the
  format supports both fp16-GEMV decode (ExLlamaV3 style) and fp8-MMA decode
  (SparkInfer style).

**Verdict**: real trellis coding, not "3-bit int in disguise". The decode is
cheap (see §3) and the format is the reason 3.0 bpw holds quality where
3-bit grouped-int does not.

## 2. CHECKPOINT — how we get 3 bpw experts

Key context: the reference artifact
(HF `0xSero/deepseek-v4-flash-0731-spark`, rev `22f28d32…`) is a
**216-expert REAP model**; our master
(`/home/flocka/models/DeepSeek-V4-Flash-162B`) is the **144-expert** variant.
They are different models — per-token decode bytes are identical (top-6
either way) but residency and quality baselines differ.

Their quantization provenance (this de-risks option b): the EXL3 payload was
produced by **stock exllamav3 at pinned revision `787d1582`**, from an
*already-MXFP4* source (`source_format: packed_e2m1_fp4_with_ue8m0_scales`),
with a documented calibration recipe (`CALIBRATION_COVERAGE.json`):
1.09 M tokens over four axes (general / legal / code_agentic /
reasoning_termination), **natural routing** (no forced expert activation),
≥1024 routes per expert per layer (greedy hash-coverage top-up),
`down_projection_input: exact_clamped_swiglu_activation`. Compute: RunPod.

Options, ranked:

| rank | option | what | effort | cost/risk |
|---|---|---|---|---|
| 1 (ship) | **(b) run exllamav3 `787d1582` on our 144-expert master** | dequant FP8 master → BF16, quantize routed experts only, keep everything else FP8/NVFP4 as today | ~1–2 wks: stand up exllamav3 (arch support proven at that revision; Transformers 5.13.1 loads the model), reproduce the calibration recipe (corpus + natural-routing capture), quant run (~GPU-days; RunPod precedent), loader for `trellis/suh/svh/mcg` schema | preserves model identity → all existing baselines/gates stay valid; our FP8 master is a better-conditioned source than their MXFP4 source |
| 2 (de-risk, do FIRST) | **(a) use their checkpoint as kernel oracle** | already coalesced TP1 **on this box** at `/home/flocka/sparkinfer-ref/data/tp1` (root-owned, ~106 GB) | ~2–4 days for a read-only loader; zero quant compute | 216-expert model → NOT a drop-in serve target (gate dims, REAP plan, +5.3 GB residency vs today: 216×43×25.17M×0.3765 B = 88.1 GB routed vs our current 82.8). License: recipe is Apache-2.0, EXL3 code MIT; the **model card license must be verified** before any redistribution. Use it to validate the loader, the dequant unit test, and the decode kernel against known-good weights *before* spending on a quant run |
| 3 (reject) | (c) write our own trellis quantizer | QTIP-style Hessian-weighted trellis search + Hadamard pipeline | 3–6 wks, high risk | no benefit over (b) — the quantizer is open source, MIT, pinned, and proven on this exact model family |

Note: `EXL3_MANIFEST.json` counts 11,008 quantized triplets (=256×43) while
the served bitmap has 9,288 (=216×43) — they quantized the full pre-REAP
expert set and serve a subset. Harmless, but confirm the loader keys off the
tier bitmap, not the manifest.

## 3. KERNEL — GB10 cost model and geometry

Constants (playbook §5): 48 SMs, sm_121, 2.4–2.6 GHz, 229 GB/s achieved
ceiling, w4a16 dequant family issue-bound at ~35 TFLOP/s equivalent.
Per expert triplet: 3 × 4096×2048 = 25.166 M weights.

### Decode M=1 byte budget

| per layer per token | MXFP4 today | EXL3 3.012 bpw |
|---|---:|---:|
| routed 6 experts | 80.2 MB | **56.9 MB** |
| shared expert (NVFP4, unchanged) | 13.4 MB | 13.4 MB |
| total | 93.6 MB (measured, 486 µs @192 GB/s) | **70.2 MB** |
| × 43 layers | 4.02 GB/token | **3.02 GB/token** |
| MoE time @192 GB/s achieved | 20.9 ms | 15.7 ms (−5.2) |
| @220 GB/s | 18.3 ms | 13.7 ms (−7.2) |
| @229 GB/s ceiling | 18.1 ms | 13.2 ms (−7.7) |

(The task brief's "~62 MB/layer, 273 µs" is the routed-only + overhead view;
same arithmetic.)

### Dequant ALU vs the stream — does decode become the bottleneck?

Required weight rate at ceiling: 229 GB/s ÷ 0.3765 B/w = **608 Gweights/s**.

Batched decode cost (`dq8` path: 8 weights per 2×u32 window loads):
per 8 weights ≈ 8 SHF (funnel) + 8 IMAD + 8 LOP3 + 4 half2 packs + 4 HFMA2
dot ≈ 34 lane-ops → **~4.3 ops/weight** (MXFP4 LUT path is ~1.5). At
608 Gw/s that is **~2.6 T lane-ops/s** against ~23 T lane-ops/s of mixed
INT/FP32 issue capacity (48 SM × 2.5 GHz × ~192 lanes) → **~11–15 %
utilization**. Cross-check: GEMV FLOPs at ceiling are 1.2 TFLOP/s, 30× under
the measured 35 TFLOP/s issue ceiling of the existing dequant family.

**Verdict: trellis dequant is NOT the bottleneck — if and only if the kernel
decodes in dq8 batches with the 96-B tile staged in registers/smem.** The
naive per-weight `dq()` does two 4-B loads per 3-bit weight (≈21× read
amplification) and ~3× the ops; that variant WOULD be the bottleneck. This is
a tiling constraint, not a throughput problem.

### Geometry: current kernel does NOT fit; new tiling needed

Our decode MoE GEMV (`kernels/gb10/common/moe_shared_expert_fused_t.cu`,
dispatched from `crates/spark-model/src/layers/moe/forward_phase.rs`):
BLOCK 32, thread-owns-one-output, warp = 32 adjacent N, streams transposed
`[K/2, N]` nibbles 32 B/warp/K-iter. (The dense GEMV family geometry —
grid N/4, 4 rows/block, 64 thr/row — is likewise per-(k,n) addressable.)
Trellis weights are **not per-(k,n) addressable**: they live in bit-serial
16×16-weight/96-B tiles. A thread cannot fetch "its" weight without decoding
a window.

New decode tiling (natural fit, ~1–2 wks incl. microtests):

- warp ↔ one 16×16 tile per iteration: 32 lanes × dq8 = 256 weights — exact;
  96 B/warp/iter, tiles contiguous in the I16 tensor → clean coalescing;
- lane owns 8 consecutive `t_offset`s, accumulates per-N partials,
  shuffle-reduce across the K-side lanes; block covers a 16-wide N strip;
- activation pre-pass per expert matrix: `x' = H·(suh ⊙ x)` (4096-vec),
  output post-pass `y = svh ⊙ H·ŷ` (2048-vec) — O(K log K), amortized,
  `hadamard_inner.cuh` is vendorable (MIT);
- keep the dual-format shared-expert structure (grid.y = expert slot,
  RIDER-A block-uniform branch) so the NVFP4 shared expert rides along;
- **m-row (γ-verify) variant required** from day one: the DSpark verify path
  needs the same MROW treatment as today's `_m` family, and the
  exact-verify law applies (see §4 — partial exactness is worse than none).

### Prefill — and the DOUBLE-RESIDENT question

Residency arithmetic (our 144-expert model, 8.6 GB currently free):

| copy | bytes |
|---|---:|
| routed experts, MXFP4 (today, resident) | 82.8 GB |
| routed experts, EXL3 3.012 bpw | 58.7 GB |
| **double-resident (both)** | **+58.7 GB over today → infeasible by ~50 GB. DEAD.** |
| replace MXFP4 with EXL3 | **frees 24.1 GB** (→ KV / longer context / draft headroom) |

So trellis must serve **both** decode and prefill. Two options:

- **P1 (bring-up, ~3–5 days)**: per-layer dequant-to-scratch for prefill:
  decode trellis → MXFP4 (or e4m3-GS128) into a 1.9 GB/layer scratch, then
  run the existing grouped GEMM unchanged. Cost ≈ +3.3 GB traffic/layer
  ≈ +14 ms/layer ≈ +0.6 s per prefill pass → −15–20 % prefill throughput on
  a 2.4 K prompt (amortizes on long prompts; multiplies under chunked
  prefill — measure). Numerics caveat: prefill sees MXFP4(trellis(W)), a
  second quantization — it must pass the same gate.
- **P2 (end state, ~2–3 wks)**: grouped trellis GEMM decoding straight to
  MMA fragments. Two proven references to port from: ExLlamaV3's
  `exl3_gemm_kernel.cuh` (vendored MIT; the `trellis_k6_small.cu` launcher
  shows the pattern — TileM16/K32/N128, smem-staged tiles, decode→fragments)
  and SparkInfer's CuTe `W4A16FusedMoeKernel` (single cooperative
  FC1/act/FC2 grid, decode to E4M3 `m16n8k32` B-fragments).

## 4. QUALITY PROTOCOL

This is a playbook **Tier 3** change (precision/model-quality decision —
explicit sign-off required). Full gate, in ladder order (playbook §6):

1. **Unit oracle**: our dequant bit-exact vs exllamav3 reference dequant on
   random tiles + real tensors from the reference checkpoint (option a).
2. **Microtest oracle** (`moe_unified_t_m_microtest` pattern): trellis GEMV
   vs BF16-dequant CPU oracle — cosine ≥ 0.999 overall, worst tile ≥ 0.99;
   standalone GB/s recorded; exit code reflects the gate.
3. **Serve A/B, same binary, env flag**: four-workload decode probe
   (prose/code/repeat/quote — never repeat-only); 5-run medians;
   temperature-0 transcripts archived.
4. **`tool-eval-bench --short` ≥ 90/100, 0 failures** — with the
   **TC-14 numerics-borderline canary watched explicitly**: it is the known
   borderline scenario; any flip there is a stop-ship, not noise.
5. **Prose-quality A/B** (the grouped-o-proj protocol): blind pairwise vs
   the MXFP4 baseline. Atlas is currently ahead of the reference stack on
   prose; that lead must not regress.
6. **Perplexity + agreement on a fixed set**: held-out shards matching the
   four calibration axes (e.g. 64 × 2048 tokens each of general / legal /
   code_agentic / reasoning_termination); gates: ΔPPL ≤ +1.5 % relative vs
   MXFP4, top-1 agreement ≥ 97 % on a fixed 10 K-token greedy replay.
7. **Speculation**: acceptance length (tok/step) within 2 % of MXFP4 under
   `ATLAS_MTP_GATE_FORCE=1`. The exact-verify GEMV chain must get a
   trellis-exact twin — measured law: *partial* exactness in the verify
   chain is worse than none.
8. **Long-context canary**: fixed begin/middle/end fact + checksum probe at
   4 context lengths (the reference `context_coherence.py` pattern).

**Fallback**: `ATLAS_EXPERT_TRELLIS=0/1` env gate; **MXFP4 stays the default
until sign-off**; both artifacts kept; default flips only after the full
ladder + soak, with before/after numbers in the commit message (the commit
log is the lab notebook). Note the fallback constraint: MXFP4 and EXL3 can't
be co-resident, so the flag selects at load time, not per-request.

## 5. PHASING — smallest safe increment first

Graphed plain-decode baseline: 45.5 ms/step = 21.9 tok/s. Independent
low-risk byte cuts first (each gated by steps 3–5 of §4 only), the expert
requant last.

| step | change | Δms/tok | expected tok/s | risk |
|---|---|---:|---:|---|
| S1 | **NVFP4 lm_head** (FP8 529.5 → 265 MB; 2.26 → ~1.1 ms) | −1.1 | **22.5** | low — NVFP4-attention precedent was quality-positive; Tier 3-lite gate |
| S2 | **FP8 for the two remaining BF16 MLA tensors** (bf16 n512 4.19 MB + n144 1.18 MB, ×43) | −0.5 | **22.8** | low — FP8-native wq_a precedent raised the gate 90→93 |
| S3 | (parallel, other agents) MoE GEMV 192 → ~220 GB/s | −2.6 | **24.2** | none if bit-identical |
| S4 | **Trellis decode kernel offline** vs the reference checkpoint (option a) — loader + unit oracle + microtest; no serve integration | 0 | — | pure de-risk; validates format/kernel/cosine before any quant spend |
| S5 | **Quantize our 144-expert master** (option b: exllamav3 @787d1582, calibration recipe of §2) | 0 | — | offline |
| S6 | **Integrate**: trellis decode GEMV + m-row verify variant + prefill P1 scratch path, env-gated, full §4 ladder | −5.2 (@192) … −7.7 (@229) | **25.9 – 27.6** (with S1–S3 banked) | Tier 3 — the sign-off step |
| S7 | Prefill P2 grouped trellis GEMM (recover prefill regression), then default flip | — | — | engineering |

Landing zone: **~26 tok/s from trellis alone (conservative achieved-BW), ~28
with S1–S3 banked, 29.9 at the waterfall's ceiling arithmetic**. 28 tok/s is
NOT reachable from the byte cut alone — it requires the trellis experts AND
the MoE bandwidth work AND the lm_head/MLA byte trims together. Side benefit
of S6: 24.1 GB freed (§3) for KV/context.

## 6. Open questions / risks

1. **exllamav3 calibration forwards on our master** — pinned revision is
   proven on this model family, but our 144-expert config must load in HF
   Transformers for the Hessian capture; verify early in S5.
2. **Prefill quality under P1** (double quantization MXFP4∘EXL3) — gate it;
   if it fails, P2 becomes a blocker for the default flip, not a follow-up.
3. **Chunked prefill × P1 scratch** — per-chunk re-dequant could multiply
   the +0.6 s cost; measure before accepting P1.
4. **Reference model-card license** — verify before touching their weights
   for anything beyond local kernel validation.
5. **Spec-decode interaction** — the verify chain's exact-GEMV twin (§4.7)
   is mandatory scope of S6, not optional.
