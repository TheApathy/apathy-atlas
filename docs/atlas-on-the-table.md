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
| 1 | **MLA projection GEMV bandwidth** — wo_a launch fragmentation FIXED by `w4a16_gemv_grouped` (bit-identical, 153→228 GB/s, commit de828e35); verify o-proj now `w4a16_gemv_grouped_batchm` (bit-exact + 3.07× the per-row exact cost). REMAINING: drafter `d_o_proj` (dense BF16 8-launch, measured 27 GB/s) and the wq/wkv chain | grouped microtest, three gates PASS | banked: wo_a plain + verify; open: drafter, A-proj | partial — landed 2026-08-09 |
| 2 | **MoE decode GEMV 193 → ~220 GB/s** (m=1) and **dedup verify 183 → ~210** | microtest oracle: per-row 193, partitioned+bucketed 183 effective vs 229 measured ceiling | ~−3 ms plain, ~−10 ms verify | open; ceiling now known to be real |
| 3 | **Acceptance bimodality** | `ATLAS_DSPARK_ACCEPT_LOG` | 2.68 → ~3.08 tok/step | **LARGELY BANKED 2026-08-09** (`757492de`+default flip): the FULL exact GEMV chain (`ATLAS_VERIFY_EXACT_GEMV`, now default) lifts acceptance 2.83 → **2.92–3.01** tok/step, zero-accept 20.3 → 17.7%, prose +1 tok/s, repeat parity, quality gate 90/100 (12/3/0). Key law learned: a PARTIALLY exact chain is worse than either extreme (o-proj-only measured 2.54). Remaining to 3.08+: head-gate `dense_gemv_batchm` exactness (unverified third family), batched attention/rope drift — the layer-diff harness can now convict non-GEMV stages cleanly |
| 4 | **Verify CUDA-graph capture** (reference ships FULL graph, capture size 6) | our verify is eager; four host-side in-loop side effects documented in the ddtree capture-bug list | launch-gap + scheduling overhead across 43 layers | open; requires device-residency refactor |
| 5 | **Adaptive + low-gear actually ON in the bench config** | phantom `ATLAS_DSPARK_ADAPTIVE` flag fixed 2b55e285 — all bench decode numbers ran adaptive-OFF | measured 2026-08-04: repeat 26.4 / quote 23.5 vs plain 21.5 | re-baselined 2026-08-09 with new kernels, γ=5: code 22.6 / **repeat 28.4 (record)** / quote 21.7 / prose 19.8. Same-box reference: code 37.4 / repeat 58.6 / **prose 19.5 (we are AHEAD on prose)** |
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
| 1 | **TC attention round 2**: KT=32 tiles (halve syncs), cp.async staging, occupancy (39KB smem = 2 blocks/SM) | round 1 SHIPPED 2026-08-09 (6d8216a3, default-on, gate 90/100 held): uniform 4.1–4.2× — CSA 28.8→7.0, HCA full-causal 76.9→18.3 ms/call @S=2176; prefill 385→420-441. **The doc's 11× assumed window-bounded work; the HCA (ratio=128) layers are FULL-CAUSAL and quadratic — they dominate at length** | 4.2× → ~8× on the attention slice |
| 2 | `q_latent_expand` — now co-#1: **14.3 ms/layer = 0.62 s/pass @N=2177** | `ATLAS_PROFILE` buckets 2026-08-09 | 3× → −0.4 s/pass |
| 3 | `kv_proj` 6.7 ms/layer (0.29 s/pass) + the non-attention remainder (~3 s of the 4.99 s TTFT@2177: MoE prefill block, norms, hc, fixed overhead — needs its own profile pass) | same buckets | the actual road to 1000 |
| 4 | Measure the curve at longer N + sparse-MLA path for long context | reference's 1055 is at 252K | length amortization |

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

## Full-stack decode decomposition — measured 2026-08-09 end-of-session

```
verify  = 106 ms   (exact GEMV chain; m6 MoE at 183 GB/s is the bulk)
propose = 12.6 ms  (true, PROPOSE_PROF; STEP_TIMING bucket reads ~20 inflated)
plain   = 50 ms    →  C = 2.37;  accept 2.9-3.0 four-workload, 3.54 bench mix
today:  code 22.5 / repeat 29.9 / quote 21.0 / prose 20.8-20.9
```

The 28+ arithmetic, all measured quantities: at accept 3.0, 28 tok/s needs
step ≤ 107 ms → verify ≤ ~94. Route: m6 MoE dedup 183 → 210 GB/s (−10 ms;
the ONE remaining verify item — persistent Stage-0 already NO-GO, T_BLOCK
saturated, needs a new idea against the 229 ceiling) + propose lm_head
2.7 → ~1 ms (batch the FP8 tile) → step ~98 ms → **30+ at accept 3.0**.
Prefill next: TC round 2 (KT=32/cp.async/occupancy), wq_b w8a16_gemm_pipelined
(~4% of peak), the 13 unconditional per-layer synchronize() calls.

## m6 MoE dedup 183→210 — attempt log (2026-08-10)

- **Cross-group double-buffer prefetch: NO-GO measured.** Partitioned leg
  regressed 1.777 → 1.983 ms (+12%): the 2× pbyte/scale buffers cost more in
  register pressure at MROW=6 than the latency hiding buys — consistent with
  the kernel's own occupancy notes. Reverted.
- bf16x2 activation loads: kept (bit-identical, gate 1 PASS, 1.777 → 1.761 ms).
- __launch_bounds__ sweep: ALREADY DONE in-tree — `(256,2)` swept 1-5 and
  documented; the m6 arm deliberately runs 128 regs so ptxas does deep load
  pipelining (long_scoreboard 9.94→2.28 cycles/issue). This EXPLAINS the
  double-buffer NO-GO: the compiler already pipelines; manual buffers only
  add register pressure.
- Remaining structural ideas (each ~half-day, oracle-iterable): VEC=4 width;
  gate_up activations staged in smem (down already does); cp.async.bulk/TMA.
  The kernel is near its practical optimum — 183 of 229 with multi-row
  gather access may be close to real. CONSEQUENCE for the 28+ route: lean
  on acceptance (3.0→3.3 via head-gate + attention exactness) and propose
  lm_head NVFP4 (2.7→~1.4 ms, drafter-only numerics = acceptance-gated)
  as the nearer levers.
