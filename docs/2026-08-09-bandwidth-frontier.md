# The bandwidth frontier — 2026-08-09 measurements and the revised route

Three measurement campaigns in one day changed the decode plan. All numbers
measured on this GB10 unless marked otherwise. Probe scripts:
`scripts/bw_ceiling.py`, `scripts/expert_structure.py` (results in
`docs/probes/expert-structure-20260809.json`), plus the reference-stack
mining below.

## 1. The real kernel bandwidth ceiling is ~229 GB/s, not 183

`scripts/bw_ceiling.py`, torch 2.10 cu130, 1–4 GiB working sets, idle GPU:

| pattern | GB/s | % of 273 theoretical |
|---|---:|---:|
| streaming read (weight-stream analog) | **228.9** | 83.8% |
| bf16 GEMV, 1 GiB contiguous | **225.5** | 82.6% |
| d2d copy (read+write) | 219.9 | 80.5% |
| bf16 GEMV at per-expert shape (100 MB) | 168.0 | 61.6% |
| 64-byte random gather | 129.3 | 47.4% |

The long-used "183 GB/s achievable" figure was our best kernel result, not a
hardware property. A contiguous GEMV reaches 225 on this silicon. The two
degraded rows reproduce our production kernels' numbers exactly:

- per-expert-sized GEMV 168 ≈ why plain decode sits at 154;
- 64B random gather 129 ≈ the m=6 dedup verify's 135.

**Consequence:** plain decode moves 7.227 GB/step. At 154 GB/s that is
46.9 ms = 20 tok/s. At 210+ GB/s it is ≤34.4 ms = **≥29 tok/s, bit-exact,
zero weight changes.** This REVISES the earlier "no kernel win left in MoE"
verdict — that verdict was measured against the false 183 ceiling. The
missing kernel is the expert-major contiguous streaming layout: large
contiguous reads, layer-fused, instead of per-expert working sets and
64B-granule gathers.

## 2. The expert weights are maximally incompressible — do not re-chase

`scripts/expert_structure.py`: layers 0/21/42 × 16 experts, w1 dequantized
from MXFP4. Every quality-exact byte-reduction door is measured shut:

| door | measurement | verdict |
|---|---|---|
| low-rank factorization | rank-1024/2048 keeps 82% energy; 1536 keeps 95% | dead — spectra nearly flat |
| shared basis across experts | rank-1024 shared captures 42–48% vs 82% per-expert | dead — experts do not share a subspace |
| value-level delta coding | cross-expert cosine 0.001–0.003 | dead — orthogonal |
| symbol-level delta coding | mutual information ≈ 2×10⁻⁵ bits | dead — independent codes |
| per-symbol lossless coding | (codex probe) 3.88/4.0 bits entropy → 1.028× max | dead — measured 2026-08-03 |

The only byte reduction available is lossy requantization (Trellis-style
vector quantization), which works by accepting controlled distortion, not by
finding structure.

## 3. The reference stack that beats us, mined

`github.com/0xSero/deepseek-v4-flash-0731-spark-sparkinfer` measures decode
34.3–48.9 (median 38.1) and prefill 1,055 tok/s at 252,047 uncached tokens
on the same GB10 class. Mining result: **their patches contain zero
performance kernels** — everything fast is stock SparkInfer (CUTLASS-DSL)
plus the model artifact. Four mechanisms:

1. **3.0 bpw EXL3/Trellis experts, REAP-pruned to 216** (vs our ~4.25 bpw
   MXFP4). The byte lever. Their median 38.1 over their measured 1.43×
   spec multiplier implies a plain floor of ~26.6.
2. **B12X wo_projection**: WO-A + WO-B as native MXFP8 GEMMs, activation
   quantized inline, inverse-RoPE fused into the input quant. Maps to our
   verify C_oproj (13.0 ms) and propose d_o_proj (7.4 ms at 27 GB/s).
3. **Fixed K5 + FULL CUDA graph at capture size [6]** — their whole stack is
   plan → bind (views only) → run (capture-safe). Our verify is eager with
   four host-side in-loop side effects.
4. **Sparse MLA** (topk-selected KV) — the 252K-prefill enabler; their
   true-NVFP4 432-byte KV record corrupted text and is DISABLED (they ship
   584-byte padded FP8) — independent confirmation of our NVFP4-KV findings.

Their compact K64 draft is a MEMORY fix (fits 262K KV), not a propose-speed
fix — our drafter MoE is only 1.47 ms of the 19.4 ms propose; the propose
cost is projections, addressed by mechanism 2.

The stack is being reproduced on this box (docker, port 8000) for a same-box
plain-floor measurement.

## The revised route

Decode, ranked by expected tok/s per unit work (was: requant-first):

| # | lever | expected | quality risk |
|---|---|---|---|
| 1 | expert-major streaming MoE decode GEMV, 154→210+ GB/s | plain 20 → 27–29 | none (bit-exact reads) |
| 2 | MXFP8 fused o_proj (port of B12X wo_projection) | −13 ms verify, −7 ms propose | gated on tool-eval-bench ≥90 |
| 3 | acceptance 2.68 → ~3.1 tok/step (bimodal zero-accept thread) | multiplier 1.0 → ~1.4× | none |
| 4 | 3.0 bpw Trellis requant | multiplies everything; sustained 35–38 | gated; separate campaign |

With levers 1–3: plain ~27–29, DFlash ~30–35. Lever 4 takes the median to
35–38 with peaks 40+. Prefill is unchanged: the tensor-core rewrite of
`prefill_attn_compressed.cu` (fully specified in
`docs/kernels/prefill-attn-tensorcore.md`) remains the whole job — ~950
without weight changes; the last decade to ~1055 is quant-dependent.

## Bench-harness bug fixed alongside

`scripts/dsflash-serve-bench.sh` and `dsflash-serve-long.sh` set
`ATLAS_DSPARK_ADAPTIVE=1`, which exists nowhere in the code — the real
switch is `ATLAS_DFLASH_ADAPTIVE=1`. Decode numbers measured through those
scripts ran with adaptive speculation silently off.
