# Decode Waterfall — 2026-08-10

The full attribution of a plain (non-speculative) decode step for
DeepSeek-V4-Flash-162B on one GB10, produced by the same method that closed the
prefill waterfall: enumerate dispatches, microtest each kernel standalone,
bucket the serve with synced probes, reconcile until the numbers close.

**Headline: decode does NOT have the prefill disease.** Every M=1 GEMV in the
hot path already runs on the fastest in-tree kernel, and the MLA projection
chain executes at the DRAM ceiling. The remaining decode headroom is (1) MoE
expert-GEMV bandwidth, (2) the spec-armed suspended path running eager, and
(3) fewer weight bytes — a quality decision, not an engineering one.

## 1. Ground-truth numbers

| measurement | value | source |
|---|---|---|
| plain decode, graphed | **45.3–45.8 ms/step = 21.9 tok/s** | LOOP_TRACE + decode_ab_probe, uniform across code/repeat/quote/prose |
| plain decode, eager | 50.4–54 ms/step | LOOP_TRACE pre-calibration + ATLAS_PROFILE run |
| graph transition | at token ~256 | FP8-KV calibration freeze re-enables graphs (see FIELD-NOTES) |
| scheduler/host outside `model.decode` | **0.00 ms** | LOOP_TRACE drain/phases/tail all zero |
| byte floor | 6.7 GB/token → **34 tok/s ceiling** at 229 GB/s | census below |

## 2. Kernel census (standalone, L2-defeating cold-weight rotation)

`cargo run --release -p spark-model --example decode_gemv_audit
--features cuda,gpu-examples` (ATLAS_TARGET_MODEL=deepseek-v4-flash):

| site | N | K | MB/call | µs/call | GB/s |
|---|---:|---:|---:|---:|---:|
| wq_a.fp8 (×43) | 1024 | 4096 | 4.19 | 21.8 | 192 |
| kv_proj.fp8 (×43) | 512 | 4096 | 2.10 | 12.3 | 170 |
| wq_b.nvfp4 (×43) | 32768 | 1024 | 18.87 | 71.2 | 265 |
| wo_b.nvfp4 (×43) | 4096 | 8192 | 18.87 | 75.9 | 249 |
| bf16 n512 (×43) | 512 | 4096 | 4.19 | 23.0 | 182 |
| bf16 n144 (×43) | 144 | 4096 | 1.18 | 8.2 | 144 |
| drafter down (×1) | 4096 | 12288 | 100.7 | 438 | 230 |
| drafter lm (×1) | 129280 | 256 | 66.2 | 273 | 243 |
| lm_head.fp8 (×1) | 129280 | 4096 | 529.5 | 2257 | 235 |
| **MLA chain aggregate** | | | **2125/token** | **9.14 ms** | **232.6 — at ceiling** |

All four GEMV families share one geometry (grid = N/4, 4 rows/block, 64
threads/row). The only sub-ceiling entries are small-N shapes whose absolute
cost is trivial (≤0.3 ms/token combined). There is nothing to swap.

## 3. The serve waterfall (eager, ATLAS_PROFILE=1, 254 tokens, prose)

Buckets sum to 52.1 + head 2.1 = 54.2 vs measured 54.0 — **closed to <1%**.
Beware the two mu characters: HC/V4 probes print U+00B5 `µs`, the MoE probes
print U+03BC `μs` — a grep for one silently drops the other (this hid the
entire MoE bucket during the first aggregation pass).

| ms/token | stage | µs/call | calls/token |
|---:|---|---:|---:|
| 20.92 | MoE exp_unified_t | 486.5 | 43 |
| 4.29 | V4 wq_b | 99.9 | 43 |
| 3.78 | V4 wo_a_grouped | 87.9 | 43 |
| 3.75 | V4 wo_b | 87.2 | 43 |
| 3.38 | HC pre-attn | 78.5 | 43 |
| 3.35 | HC pre-ffn | 77.9 | 43 |
| 3.13 | V4 wq_a | 72.8 | 43 |
| 2.55 | MoE-km exp_splitk_m_t | 1255 | 2 |
| 1.13 | V4 paged_attn | 26.3 | 43 |
| 1.10 | HC post-attn | 25.6 | 43 |
| 0.83 | V4 wkv | 19.2 | 43 |
| 0.54 | MoE gate | 12.7 | 43 |
| 0.36 | MoE topk | 8.4 | 43 |
| ~2.3 | rope/norm/cache/writeback glue (8 probes) | 5–8 | 43 |
| 0.27 | MoE wsum_blend | 6.2 | 43 |
| 2.1 | lm_head + sampling (PROFILE `head`) | | 1 |

### Reconciliation notes (what the buckets really mean)

- **MoE exp_unified_t 486 µs/layer = 192 GB/s** on 93.6 MB of expert weights —
  exactly matching the standalone dedup microtest. The one true bandwidth
  laggard: ceiling would be 409 µs (−3.3 ms/token).
- **MLA buckets read high in-serve** (15.8 ms vs 9.1 standalone). The V4
  `prof!` syncs each bucket but unprofiled neighbors (input_norm, residual
  adds) bill into the next synced bucket — e.g. wq_a shows 72.8 µs for a
  21.8 µs kernel. The 9.1 ms standalone number is the truth; the ~6 ms delta
  is norms/glue plus per-probe sync, not GEMV slowness.
- **HC pre 78 µs/site is NOT the fused path.** One-shot dispatch log confirms
  `split=true` (both kernels loaded). Standalone amortized split cost is
  ~20 µs/site; the eager bucket is latency-bound (two dependent small kernels
  + launch + sync). Under graphs the launch overhead disappears; HC's true
  graphed cost is ~2–3.5 ms/token, not the 8.1 the eager table suggests.
- **Eager − graphed = ~8 ms/step** ≈ 860 launches × ~8 µs launch/sync cost.
  This is what CUDA graphs buy; it is already banked in steady state.

## 4. Ranked levers (what will actually move decode)

| # | lever | est. gain | risk/nature |
|---|---|---|---|
| 1 | Graph the spec-armed suspended serial path (currently eager: prose 19.75 vs plain 21.9) | +2 tok/s on prose/quote under spec | none — engineering; restores adaptive's "never slower than plain" contract |
| 2 | MoE expert GEMV 192 → ~220 GB/s (SASS/bank-conflict audit, same method as the prefill shared-expert fix) | +1.5–2 tok/s all workloads | none if bit-identical like prior MoE fixes |
| 3 | Small-N GEMV shapes (kv_proj 170, n144 at 144 GB/s) | +0.2 tok/s | none; low priority |
| 4 | **Fewer bytes**: 3.0 bpw Trellis experts (4.0→2.6 GB/token; ceiling 34→~44), NVFP4 lm_head (529→265 MB) | +6–10 tok/s | **quality decision — needs explicit sign-off + gate** |
| 5 | Drafter/acceptance quality on prose (spec side; C=2.64 decomposition) | workload-dependent | model work, not kernels |

Items 1+2 take plain ~21.9 → ~25 and spec-prose to parity or better; item 4 is
the only road past the 34 tok/s byte ceiling — where the reference stack's 38
lives.

## 5. Reproduction

```
# census            (kills any resident serve first — will OOM otherwise)
ATLAS_TARGET_MODEL=deepseek-v4-flash cargo run --release -p spark-model \
  --example decode_gemv_audit --features cuda,gpu-examples

# HC standalone at decode shape (T=1; default 2410 = prefill)
... --example hc_pre_microtest --features cuda,gpu-examples -- 1

# serve waterfall (eager: PROFILE suppresses graphs by design)
scripts/dsflash-serve-bench.sh <name> 5 ... ATLAS_PROFILE=1
# then aggregate "(V4|HC|MoE)...: Nµs|μs" lines — BOTH mu characters.

# graphed truth + host-share
scripts/dsflash-serve-bench.sh <name> - ... ATLAS_LOOP_TRACE=1
```
