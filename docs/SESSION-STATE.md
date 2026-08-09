# DeepSeek-V4-Flash on one GB10 — state, targets, next action

Branch `combined-residency`. All numbers measured on this box unless marked
arithmetic. Serve config for every number below:

```
scripts/dsflash-serve-bench.sh <name> 5 \
  ATLAS_V4_ATTN_NVFP4=1 ATLAS_V4_ATTN_RELEASE_BF16=1
```

Both attention flags are required together — `RELEASE_BF16` alone dies with
"requires successful NVFP4 transcodes for wq_b, wo_a, wo_b". The pair is
quality-POSITIVE (tool-eval-bench 93 vs 90 on BF16 attention) and frees ~8 GiB.

## Where we are

| | session start | now | reference |
|---|---|---|---|
| prefill @912 tok | 40 tok/s | **385** | ~960–1010 |
| TTFT @3281 tok | 90.6 s | **8.5 s** | — |
| decode (median) | 17–20 | **20.3** | 28.6 |
| plain decode | — | **20.02** | **20.0 — PARITY** |
| tool-eval-bench | 23/100 | **90/100**, 0 fail | — |
| bench wall | 1804 s | **329 s** | — |

## Decode: the gap is entirely the speculative multiplier

Plain decode is at parity with the reference. Everything missing is DSpark.

```
spec step 131.8 ms at 2.68 tok/step | plain step 50.0 ms
C = 2.59   speedup = 2.68 / 2.59 = 1.02x   (matches measured ~1.0x)
reference: C = 2.16 at 3.08 tok/step = 1.43x
```

Decomposition (`ATLAS_DSPARK_PROPOSE_PROF=1`, `ATLAS_DSPARK_ACCEPT_LOG=1`,
always with `ATLAS_MTP_GATE_FORCE=1` so the gate cannot hide acceptance):

- verify 113 ms = **2.26** of C · propose 19.4 ms = **0.39** of C
- **Driving propose to zero still leaves C at 2.26 > 2.16. Verify must also
  come down.**
- acceptance **2.68 tok/step / 42%**, strongly bimodal: 31% of steps accept
  NOTHING, 22% accept a near-full block.
- inside propose: `d_o_proj` 7.39 ms (8 wo_a `dense_gemm` per stage, 201 MB at
  **27 GB/s** vs 183 achievable), `a_hc_pre` 3.68, `b_qkv_proj` 3.37,
  `lm_head` 2.71, `stage_moe` 1.47, **`c_attn_kernel` 0.62 — attention is free**.
  The drafter's PROJECTIONS are the cost.

Route to 28.5, all three required:
```
verify  113 -> ~95 ms   m=6 MoE dedup 135 -> 183 GB/s ([[moe-dedup-bandwidth-vs-rows]])
propose  19 -> ~10 ms   small-M GEMM kernels for the projections
=> C ~2.10, at today's 2.68 tok/step = 1.28x = 25.5 tok/s
then acceptance 42% -> 74% (3.08 tok/step) = 1.47x = 29.4 tok/s
```

## Prefill: one kernel away from ~950

Budget at N=911 (`ATLAS_PROFILE=1`): core attention **864.6 ms = 45%**, running
1.90 TFLOP in 864.6 ms = **2.2 TFLOPS** against 250–500 TFLOPS of BF16 tensor
core. Memory side is already fixed (139 → 20.7 ms/layer); what remains is
scalar FP32 math.

**NEXT ACTION: rewrite `prefill_attn_compressed.cu` on tensor cores.** Full
design, verified m16n8k16 fragment layouts, block mapping, the three traps and
the validation gates are in `docs/kernels/prefill-attn-tensorcore.md`. Expect
864.6 → ~78 ms/pass, prefill → ~800 tok/s; the `q_latent_expand` GEMMs then
take it to ~950. The last ~10% to 1055 is quant-dependent (reference D2R runs
on 2-bit experts; our MoE is at 149 of 183 GB/s, no kernel win left).

## Measured dead ends — do not re-chase

- `ATLAS_DFLASH_ASYNC=1` — 19.96 vs 20.00 tok/s; propose still times 35–53 ms,
  it does not actually overlap.
- `γ=3` — routes to `verify_k3_step`, not the dflash path (`dispatch_min=4`).
- `ATLAS_DFLASH_SPEC_THINK=1` — 19.74 vs 19.85 with thinking on.
- Batching the drafter wo_a gather/scatter to 2D copies (240 → 48 launches) —
  7.79 → 7.39 ms. The launches were not the cost; the small-M GEMM shape is.
- `ATLAS_MOE_PREFILL_EXACT_TILES=1` — the ~100× `max_m_tiles` grid
  overprovision is real in the code but worth ~2%.
- MoE prefill is FINE: 694 ms of a 23,200 ms prefill, `grouped_gate_up` at
  149 of 183 GB/s.
- **nsys cannot profile prefill here** — the ~6-min weight load overruns its
  buffers and the trace is dropped, even with `--delay`. Use the in-tree
  `ATLAS_PROFILE=1` probes (`aprof!` in `prefill/cache_skip_v4.rs`,
  `prof!` in the MoE prefill).

## Corrections made this session (do not regress to the old beliefs)

1. Online acceptance is **2.68 tok/step**, not the long-recorded ~1.0.
2. Propose is **19.4 ms**, not the 36 ms `ATLAS_DFLASH_STEP_TIMING` reports
   (that bucket also carries catch-up seeding + async collection). So verify
   is 2.26 of C, not 2.00 — propose alone cannot reach the target.
3. Plain decode measures **17.0** on the BF16-attention build with varied
   300-token generations; the older "21.9" came from sustained repeat-prompt
   runs. With NVFP4 attention it is 20.02.

## Instruments added this session

| env | what |
|---|---|
| `ATLAS_DSPARK_ACCEPT_LOG=1` | accept histogram + draft accept% + zero-accept rate |
| `ATLAS_DSPARK_PROPOSE_PROF=1` | propose phase split |
| `ATLAS_PROFILE=1` | now also attributes the V4 attention prefill (`aprof!`) |
| `ATLAS_DFLASH_STEP_TIMING=1` | verify vs propose (propose bucket is inflated — see above) |
| `ATLAS_MTP_GATE_FORCE=1` | bypass the throughput gate; REQUIRED when measuring acceptance |

Scratchpad harnesses: `decode_vet.py` (streaming, prefill excluded,
`ATLAS_VET_THINK=1` for thinking mode), `prefill_scan.py`, `toolchain_probe.py`,
`quality_probe.py`.
