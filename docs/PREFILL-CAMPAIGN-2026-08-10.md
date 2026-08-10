# Prefill campaign 2026-08-09/10 — 385 → 792 tok/s, and what 1000 costs

All numbers measured on this GB10, DeepSeek-V4-Flash-162B, N=2410-token
prompt, TTFT from the streaming API (prefill excludes decode). Every landed
change passed `tool-eval-bench --short` at ≥90/100 (12 pass / 3 partial /
0 fail); one raised it to 93.

## The ladder

| # | change | commit | tok/s | TTFT |
|---|---|---|---:|---:|
| — | session start | — | 385 | 5.16 s* |
| 1 | TC prefill attention (m16n8k16 rewrite) | `6d8216a3` | 441 | — |
| 2 | TC compressor projections + split waterfall | `1b4bd7f2` | 459 | — |
| 3 | TC kernel for the ratio-0 dense layers | `889abbfe` | 562 | 4.29 s |
| 4 | wq_a via the FP8 pipelined GEMM (gate 93/100) | `0fbef449` | 599 | 4.03 s |
| 5 | token-tiled hc_pre | `17481eaa` | 629 | 3.83 s |
| 6 | compressor + wq_a → `dense_gemm_bf16_pipelined` | `0e368e6f` | 718 | 3.35 s |
| 7 | kv_proj + BF16 projection arm → same kernel | `415734fa` | 784 | 3.08 s |
| 8 | strided pipelined GEMMs + in-place wo_a | `af92018a` | **792** | **3.04 s** |

\* at N=911 the session actually began at 40 tok/s; 385 is the same-day
figure after the first attention work, at the N=2410 basis used throughout.

## What actually produced the 2.06×

**Five of the eight steps were fast kernels ALREADY IN THE TREE wired to slow
ones.** Not new algorithms — dispatch archaeology:

- the two compressor projections and `wq_a`'s fallback ran `dense_gemm_tc`
  (16×64 tile, K-step 16, B loaded scalar at stride K). `dense_gemm_bf16_pipelined`
  (128×128, 2-stage cp.async) measures **18×** faster at those exact shapes
  (8.07 → 0.44 ms at M=2410 N=1024 K=4096), cosine 1.000000 both ways.
- `kv_proj` was pinned to the SCALAR GEMM by a NaN workaround whose comment
  names `dense_gemm_tc` — a different kernel, still disabled. **30×**
  (6.96 → 0.23 ms).
- `v4_project_prefill`'s BF16 arm (wq_b, the 8 wo_a groups, wo_b) also ran
  the scalar GEMM.
- `wq_a` skipped its own checkpoint-native FP8 mirror; using it was both
  faster AND quality-positive (gate 90 → 93).

The lesson for the next campaign: **profile which kernel a path actually
dispatches before designing a new one.** Two of the four planned kernel
campaigns evaporated on contact with that question.

## Instruments built (reusable)

- `scripts/prefill_waterfall.py` — parses `ATLAS_PROFILE=1` serve logs into a
  per-bucket waterfall; the FFN wrapper matched its interior buckets to 0.1 ms,
  which is what validates the method.
- `hprof!` layer-glue probes in `prefill_inner` (`hc0_pre_attn`, `xw_attn_block`,
  `hc1_mid`, `xw_ffn_block`, `hc2_post_ffn`) + `aprof!` sub-buckets
  (`1a_wq_a_norm`, `1b_wq_b`, `4a_csa_compressor`, `4b_attn_kernel`,
  `4c_ring_seed`, `4d_dense_fallback`, `6_o_proj`). The waterfall now closes to
  within ~55 ms of measured TTFT — no unattributed lump.
- `scripts/clock_ramp_probe.py` — cold-vs-hot TTFT with SM-clock sampling.
- Oracles: `prefill_attn_tc_microtest` (parameterized S/window/ratio),
  `w8a8_gemm_microtest`, `hc_pre_microtest`, `w4a16_gemv_grouped_microtest`.

## Measured NO-GOs — do not re-chase

| hypothesis | measurement | verdict |
|---|---|---|
| SM clock ramp explains the serve-vs-standalone gap | cold 5.16 s vs hot 5.21 s, clocks 2.4–2.6 GHz throughout both | refuted |
| per-stage `synchronize()` calls cost real time | `ATLAS_V4_STAGE_SYNCS=0` changed nothing end-to-end | refuted |
| `csa_compress` is worth −0.25 s | bucket subtraction puts the whole non-GEMM tail at ≤12 ms/pass, likely ~3 | 20-80× overestimate; the bucket was its GEMMs |
| hc_pre re-reads x 25× → register-resident x | bit-identical but 4.23 vs 3.97 ms (occupancy loss) | NO-GO |
| the decode hc_pre split helps at prefill width | 17.8 vs 3.97 ms (inverted locality) | NO-GO |
| wo_a gather/scatter costs ~316 MB/layer of real time | in-place is +1% — the 2D d2d copies are cheap; the win was launch count | overestimate, change kept |
| MoE prefill can reach 229 GB/s | the family's own FLOP/s ceiling is ~35 TF/s (measured via the shared expert at 4,600 FLOP/byte); gate_up is already at 89% of it | unreachable without fewer instructions per weight byte |

## The road from 792 to 1000

Target TTFT at N=2410 is **2.41 s**; we are at 3.04. Need **−0.63 s**.

Remaining budget (per pass, from the closing waterfall):

| bucket | cost | available | how |
|---|---:|---:|---|
| MoE gate_up + silu_down | ~1.25 s | **−0.20 s** | P0 micro-fixes (packed bf16x2 epilogue, wider dequant store, kill the 1.84M empty CTAs — all bit-identical) + `mxf8f6f4.block_scale` native MMA (numerics-flagged, A precision unchanged) |
| o_proj + wq_b | ~0.85 s | **−0.10 s** | w8a8 FP8-native MMA (kernel + oracle built: cos 0.9997, 1.22× on the K=8192 shape); wiring plan complete |
| TC attention | ~0.55 s | **−0.15 s** | round 2: KT=32 tiles, cp.async staging, HCA full-causal specialization |
| hc glue | ~0.30 s | **−0.05 s** | fuse hc_post + norms |
| shared_expert | ~0.15 s | **−0.05 s** | runs `w4a16_gemm_t` at K_STEP=32 with 4,600 FLOP/byte and still only 35 TF/s — move to a K64 body or `predequant_for_prefill` |
| **realistic total** | | **−0.55 s** | → TTFT ~2.49 s ≈ **970 tok/s** |

**1000 at this prompt length is reachable but tight**, and the last ~3% would
have to come from the MoE W4A4 path (activations to e2m1), which is a genuine
quality tradeoff and must be decided explicitly, not slipped in. Everything
above the W4A4 line is numerics-neutral or cosine-gated.

Note on length: the curve is nearly flat (590 @ 1525, 610 @ 3065 measured
before this batch of fixes) because the HCA layers attend full-causally and
grow quadratically — so "1000 at longer prompts" is not an escape hatch, and
the per-bucket work above is the only road.
