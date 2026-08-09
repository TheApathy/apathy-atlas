# Atlas engine — what is still on the table (and what is measured OFF it)

The complete optimization ledger as of 2026-08-09. Every row is backed by a
measurement in this repo, `docs/2026-08-09-bandwidth-frontier.md`,
`docs/SESSION-STATE.md`, or the microtest oracles. **If an idea is not in the
OPEN tables and is in a CLOSED table, do not re-chase it without new
evidence.**

Targets this ledger serves: plain **27–29** · DFlash sustained **30–35**
(35–38 with requant) · prefill **~950** (last decade quant-dependent).

## Decode — OPEN, ranked by prize/effort

| # | item | evidence | prize | status |
|---|---|---|---|---|
| 1 | **MLA projection GEMV bandwidth** (m=1 composite ~109 GB/s derived; drafter `d_o_proj` measured **27 GB/s**) — port the reference's MXFP8 fused wo-projection (inline activation quant + fused inverse-RoPE) | plain step: ~3.2 GB non-MoE takes ~29 ms of 50; probe says contiguous GEMV does 225 | ~−11 ms plain step → plain ~26; −13 ms verify C_oproj; −7 ms propose | next up; confirm split with `ATLAS_PROFILE` plain run first |
| 2 | **MoE decode GEMV 193 → ~220 GB/s** (m=1) and **dedup verify 183 → ~210** | microtest oracle: per-row 193, partitioned+bucketed 183 effective vs 229 measured ceiling | ~−3 ms plain, ~−10 ms verify | open; ceiling now known to be real |
| 3 | **Acceptance bimodality** — 31% of steps accept NOTHING, 22% near-full | `ATLAS_DSPARK_ACCEPT_LOG` histogram | 2.68 → ~3.08 tok/step = 1.17× → 1.34× multiplier | open; the single biggest spec lever |
| 4 | **Verify CUDA-graph capture** (reference ships FULL graph, capture size 6) | our verify is eager; four host-side in-loop side effects documented in the ddtree capture-bug list | launch-gap + scheduling overhead across 43 layers | open; requires device-residency refactor |
| 5 | **Adaptive + low-gear actually ON in the bench config** | phantom `ATLAS_DSPARK_ADAPTIVE` flag fixed 2b55e285 — all bench decode numbers ran adaptive-OFF | measured 2026-08-04: repeat 26.4 / quote 23.5 vs plain 21.5 | fixed; rerun bench to re-baseline |
| 6 | **3.0 bpw Trellis requant** (or adopt-their-artifact) | reference: median 38.1 implies ~26.6 plain floor at 3.0 bpw; experts measured incompressible losslessly | multiplies everything: sustained 35–38, peaks 40+ | separate quality-gated campaign |

## Decode — CLOSED by measurement (do not re-chase)

| item | measurement | date |
|---|---|---|
| Lossless weight compression, ALL forms | entropy 3.88/4.0 bits (1.028×); rank-1024 keeps 82% energy; shared basis 42–48% vs 82% own; cross-expert cos ~0.002; code MI ~2e-5 bits | 08-03 / 08-09 |
| Persistent Stage-0 host-work MoE kernel | NO-GO gate: 115–163 GB/s, loses to control on every routing shape | in-tree oracle |
| `ATLAS_DFLASH_ASYNC` propose overlap | 19.96 vs 20.00 tok/s; does not overlap | 08-09 |
| γ=3 as operating point | routes to `verify_k3_step`, not the dflash path | 08-09 |
| `ATLAS_DFLASH_SPEC_THINK` | 19.74 vs 19.85 | 08-09 |
| Drafter wo_a gather/scatter 2D-copy batching | 7.79 → 7.39 ms; launches were not the cost, the GEMM shape is | 08-09 |
| Compact K64 draft as a propose-speed fix | drafter MoE is 1.47 ms of 19.4 ms propose; their draft is a MEMORY fix | 08-09 mining |
| MoE gate / w4a16 gate lever | 6.4× kernel win, ~0 end-to-end; dead code on V4 | earlier, memory |
| CUDA graphs for short benches | FP8-KV calibration suppresses them; real sustained +18% already banked | 08-06 |

## Prefill — OPEN

| # | item | evidence | prize |
|---|---|---|---|
| 1 | **Tensor-core rewrite of `prefill_attn_compressed.cu`** | scalar kernel at 2.2 TFLOPS = 45% of prefill; design + verified MMA fragment math complete in `docs/kernels/prefill-attn-tensorcore.md` | 385 → ~800 tok/s |
| 2 | `q_latent_expand` GEMMs (261 ms at ~11 TFLOPS) | `ATLAS_PROFILE` budget | ~800 → ~950 |
| 3 | Measure our prefill curve at 8K/32K/128K | reference's 1055 is at 252K — not our 911-token regime; our plateau is unmeasured | honest comparison + maybe free wins |
| 4 | Sparse-MLA / indexer path for long context | their 252K enabler; V4 has native compress_ratios | new serving envelope, separate campaign |

## Prefill — CLOSED

| item | measurement |
|---|---|
| MoE prefill kernels | 149 of... old-183: re-quote vs 229 ceiling before declaring closed — but it is 694 ms of 23,200 ms; NOT the lever either way |
| `max_m_tiles` grid overprovision | real but worth ~2% |
| nsys prefill profiling on this model | 6-min weight load overruns trace buffers; use `ATLAS_PROFILE` probes |

## Engine-level honesty — where Atlas is NOT behind

- **Plain decode per byte moved is at reference parity** (20.02 vs 20.0
  measured; their higher floor is 3.0 bpw weights, not better kernels).
- The **dedup multi-row verify MoE** (+60.9% over per-row, bit-exact) has no
  equivalent in the reference stack — they brute-force fixed K5 through
  graphs instead.
- NVFP4 attention transcode is quality-POSITIVE here (93 vs 90) and ships;
  their true-NVFP4 KV record corrupted text and is disabled.
- The bench/tooling loop (microtest oracles, sha256 text-exactness gates,
  accept histograms, phase profilers) is measurement infrastructure the
  reference recipe does not have.

## The composite picture

```
plain 50 ms/step today:
  ~21 ms  MoE GEMV      @ 193 GB/s   → ceiling 229 leaves ~3 ms
  ~29 ms  MLA+rest      @ ~109 GB/s  → ceiling ~220 leaves ~11 ms
ceiling-perfect plain ≈ 36 ms ≈ 28 tok/s  ← matches the reference floor arithmetic
```

Update this file when a row changes state. A row moves to CLOSED only with a
measurement, never with an argument.
