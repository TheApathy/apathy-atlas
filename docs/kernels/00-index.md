# Atlas GB10 Kernel Documentation — Index

**Scope:** every CUDA kernel Atlas launches when serving **DeepSeek-V4-Flash-162B** on a
**single NVIDIA GB10 (DGX Spark)**, documented per kernel with launch geometry, dtypes,
bytes moved, and its roofline gap.

| Doc | Covers | Kernels |
|---|---|---|
| [01-mla-attention.md](01-mla-attention.md) | MLA / CSA attention, RoPE-scatter, cache assembly, DSpark drafter attention, mHC from the attention side | 34 |
| [02-moe.md](02-moe.md) | Routing, top-k, permute, the `_t` decode GEMV family, the `_m` MROW dedup verify family, grouped GEMM | 90+ |
| [03-gemm-gemv.md](03-gemm-gemv.md) | Dense GEMM/GEMV: `w4a16_*`, `w8a16_*`, `dense_gemm_*`, `dense_gemv_*`, NVFP4 variants | 80+ |
| [04-elementwise-norm-cache.md](04-elementwise-norm-cache.md) | mHC hyper-connections, RMS norm family, residual/elementwise, RoPE, quantisers, KV cache append, argmax | 100+ |

---

## Hardware ground truth

Queried directly (`cudaGetDeviceProperties`, 2026-08-03) — **use these numbers, not the
ones in kernel comments**:

```
name = NVIDIA GB10   SMs = 48   cc = 12.1
sharedMemPerMultiprocessor = 102400 B (100 KiB)
maxThreadsPerMultiProcessor = 1536
memoryBusWidth = 256 bits
```

Matches `crates/atlas-core/src/device.rs:16` (`NUM_SMS = 48`) and
`kernels/gb10/HARDWARE.toml` (273 GB/s LPDDR5X, 120 GB unified).

> **Correction propagated into these docs:** several kernel comments (e.g.
> `hyper_connection.cu:220`, `:253`) assert **25 SMs** and size their grids to it.
> That is wrong by 1.9×. `hc_pre_mix` launching `mix_hc+1 = 25` blocks "to match the
> SM count" therefore idles **23 of 48 SMs**. See
> [04 § hc_pre_mix](04-elementwise-norm-cache.md#hc_pre_mix).

**Roofline conversions used throughout:** 273 GB/s ⇒ **1 MiB = 3.84 µs**, 1 MB = 3.66 µs.
Eager kernel-launch floor ≈ **4–6 µs**, so any kernel moving < ~1.5 MB is launch-bound.

---

## The measured step budget — what these docs have to explain

γ=2 DSpark speculative decode, **CUDA graphs confirmed ON**, seq_len 733–860,
`ATLAS_MTP_TIMING=1` (task #23):

```
fwd  (verify, M=2)  = 117 – 145 ms      ← 63% of the step
propose (DSpark)    =  63 –  74 ms      ← 34% of the step
save_hidden         = 1.4 –  3.6 ms
commit/trim/sync/marconi/ep/propose_mask  < 1 ms each
------------------------------------------------------
TOTAL               = 186 – 213 ms   →  8.4 tok/s at 61% accept (~1.6 tok/step)
```

Against the 273 GB/s roofline:

| phase | unique bytes it *must* read | roofline | measured | gap |
|---|---|---|---|---|
| verify fwd (M=2) | ~11-expert union × 12.6 MB + attn + dense ≈ 160 MB/layer × 43 = **6.9 GB** | **~25 ms** | 117–145 ms | **≈5×** |
| propose (DSpark) | whole drafter store = **10.86 GB** read once end-to-end | **~40 ms** | 63–74 ms | **≈1.7×** (so it re-reads) |

**This is the conclusion of the whole investigation.** Three hypotheses were tested and
all three are dead:

1. **CUDA graphs** — measured at **12.5 tok/s graphed vs 12.9 ungraphed**, i.e. worth ~0%.
   (~900 launches × ~5 µs ≈ 4.5 ms against a ~124 ms step.) Every earlier benchmark had
   silently run eager because `fp8_kv_calibration_tokens = 256`
   (`kernels/gb10/deepseek-v4-flash/MODEL.toml:86`) forces `suppress_graphs = true` until
   `seq_len > 266` — see `crates/spark-model/src/model/trait_impl/verify_fused.rs:176-194`.
   Fixing that was correct but is **not a throughput lever**.
2. **Acceptance rate** — 61% at γ=2 ≈ 1.6 tok/step. Even a perfect drafter caps the win at
   the verify-forward cost, which is itself 5× off roofline.
3. **The compressed-KV pool asymmetry** — real, but a small share of a step whose two
   dominant phases are both memory-inefficient by multiples.

**The kernels are the bottleneck.** Everything below is the accounting for that 5×.

---

## Cross-document findings ledger

Sorted by estimated saving on the γ=2 step. Each row links to the full write-up.

### Verify forward — attention

> **Correction (measured after the ledger was written): the attention estimates below are
> too high, because the KV pools fit in L2.** `cudaGetDeviceProperties` on this box reports
> `l2CacheSize = 25165824` (**24 MB**), `persistingL2CacheMaxSize = 18874368`. At the served
> `--max-seq-len 1024`, one layer's raw KV is `1024 × 576 = 576 KB` per pool, `1.13 MB` for
> K+V — 0.05 % of L2. So the "64 CTAs re-stream the same rows" and "K and V are read twice"
> findings are **not** costing DRAM bandwidth; the second and subsequent reads are L2 hits.
> What they do cost is **L2 bandwidth and load-issue slots**, which matters only if the
> kernel is issue-bound rather than DRAM-bound.
>
> Concretely: the redundant V traffic is `43 layers × 576 KB = 24 MB` per verify step, i.e.
> **~90 µs** at 273 GB/s if it were DRAM — against a measured 117–145 ms step. The attention
> findings are therefore worth **tens to a few hundred µs**, not the 2–15 ms claimed. The
> honest conclusion is that **essentially the entire 5× gap lives in the MoE GEMVs**, whose
> weights (~6.9 GB/step) are far too large for L2 and must come from DRAM every step.
> Re-read the MoE section below as the real priority list; the rows here are cheap
> hygiene, not the fix.

| Est. | Finding | Doc |
|---|---|---|
| **8–15 ms** | `mla_paged_decode_fp8` grids `blockIdx.x = q_head` over **64 heads while `kv_heads = 1`**, so all 64 CTAs stream byte-identical latent rows. Fix: head-tiled CTA (one CTA serves 8–16 Q heads). `mla_paged_decode_fp8.cu:60-64`, launcher `ops/kv_cache.rs:888`. | [01 #1](01-mla-attention.md#findings--prioritised-optimisation-list) |
| **3–5 ms** (γ=2)<br>**15–25 ms** (γ=6) | The M-row verify runs attention in a **plain `for i in 0..n` loop**, one `mla_paged_decode_fp8` launch per row, each re-reading the whole KV window. Phases A and C are already batched; phase B is not. `multi_seq/mla.rs:497-522`. | [01 #3](01-mla-attention.md#findings--prioritised-optimisation-list) |
| **2–4 ms** | **K and V are loaded twice from the same addresses.** In MLA `K == V` by construction (`mla_cache_assemble_fp8.cu:87`, `mla_absorbed.cu:313-318`, `prefill/cache_skip_v4.rs:592-596`), yet the kernel loads the K row at `:136-152` then re-loads the identical V row at `:179-191` (compressed arm `:313`/`:337`). Bit-exact to delete. | [01 #2](01-mla-attention.md#findings--prioritised-optimisation-list) |
| **1.5–3 ms** | **Byte-granular FP8 loads.** `k_latent[i]` on `const unsigned char*` = 16 separate 1-byte loads/thread/position. The correct pattern already exists in-tree at `paged_decode_attn_fp8_mla.cu` (`unpack4_fp8` on `uint32`, now behind `load_lane_fp8`/`FP8_U32_OK`) — but copy the idea, not its offsets: that kernel's `lane_id * 18` is misaligned (see the bug table below). Here `lane_id * 16` is 16-B aligned, so `uint4` is legal. | [01 #4](01-mla-attention.md#findings--prioritised-optimisation-list) |
| **0.8–1.5 ms** | `smem_o[NUM_WARPS][512]` cross-warp reduction is a **512-iteration scalar loop in one warp**, 3 levels, 7/8 of the CTA idle. `mla_paged_decode_fp8.cu:366-387`. | [01 #5](01-mla-attention.md#findings--prioritised-optimisation-list) |
| **1–2 ms** @8k | Compressed-arm scan is O(ctx/4) with a scalar shuffle tree per 512-B block and no `BC`-style batching. `mla_paged_decode_fp8.cu:301-346`. | [01 #8](01-mla-attention.md#findings--prioritised-optimisation-list) |
| **0.5–1.5 ms** | No `__launch_bounds__` on `mla_paged_decode_fp8` with ~160 live FP32 registers ⇒ likely 1 CTA/SM. Also `--fmad=false` is global (`KERNEL.toml:4`). | [01 #9](01-mla-attention.md#findings--prioritised-optimisation-list) |

### Verify forward — MoE

| Est. | Finding | Doc |
|---|---|---|
| **~29 ms** @γ=6 | The `_m` MROW kernels run at **136 GB/s vs the single-row path's 194 GB/s** on identical access patterns. Pure overhead — items below are the mechanisms. `moe_shared_expert_fused_t.cu:983`, `:1162`. | [02 #1](02-moe.md#optimisation-opportunities-priority-order) |
| **~15–35 ms** | **Union size is the real floor.** Weights are already read once per distinct expert; total bytes now depend only on how many distinct experts the M rows pick (19.5 of 36 at γ=6). Lever is routing (align drafter gate with target), not the kernel. | [02 #2](02-moe.md#optimisation-opportunities-priority-order) |
| **6–10 ms** @γ=6 | `gate_up` **never stages its `A` activation rows in smem** — `silu_down` does. Every block re-reads the same 49 KB of activations. `moe_shared_expert_fused_t.cu:1073-1078`, `:1120-1121`. | [02 #3](02-moe.md#optimisation-opportunities-priority-order) |
| **4–8 ms** @γ=6 | `silu_down` dynamic smem = `mrow*k*4/split` = **12288 B** at MROW=6, capping a 32-thread block at 8 blocks/SM = **~17% occupancy**. Store staged activations BF16, or raise `T_BLOCK` for the `_m` path. `fp8_moe.rs:507`. | [02 #4](02-moe.md#optimisation-opportunities-priority-order) |
| **1–2 ms** | `mrow_gather_slots` leader election is a **serial O(72) scan on thread 0** with 31 lanes idle; trivially a `__ballot_sync`/`__ffs`/`__popc` prefix. `moe_shared_expert_fused_t.cu:936-981`. | [02 #5](02-moe.md#optimisation-opportunities-priority-order) |
| **~5.5%** of MoE | **MXFP4-E8M0 is cheaper than NVFP4** (block 32, no global scale2) and the `_e8m0` kernel variants are already compiled. Config flip. | [02](02-moe.md#findings) |
| launch-side | `moe_gate_topk_fused` (`moe_gate_topk.cu:46`) is **completely uncoalesced** and still on the shared-memory E2M1 LUT (~127 GB/s vs the branch-free 194). Task #22. | [02 Part A](02-moe.md#moe_gate_topk_fused) |
| — | **~2.2× of the MoE gap is still unexplained** after all of the above. `T_BLOCK = 32` (one warp per block, `fp8_moe.rs:295`) is the prime suspect. **Needs an Nsight Compute pass** — this is the single largest open question in the docs. | [02 #1](02-moe.md#optimisation-opportunities-priority-order) |

### Verify forward — dense projections

| Est. | Finding | Doc |
|---|---|---|
| **−7.4 ms** | Attention projections run on **FP8 (`w8a16`) instead of NVFP4 (`w4a16`)** — the `nv4_ok` branch at `multi_seq/mla.rs:531` isn't taken. Half the bytes are available for free. | [03](03-gemm-gemv.md#findings) |
| **−11.7 ms** @γ=6 | **`dense_gemv_fp8w_batchm<8>` does not exist.** `dense_gemv_fp8w_batch2.cu:72` caps at M=2, so the γ=6 lm_head falls back to a 7× serial loop over a 529 MB weight. | [03 F0/F1](03-gemm-gemv.md#findings) |
| **0.9–1.4 ms** | `w4a16_gemv_batch2/batch3` and the whole `qg`/`qkvz`/`dual_*` family are stuck on the **old narrow K-loop** (`u32 packed4`, `:352`/`:1092`) while the M=1 and batch4+ paths use the upgraded `u64 packed8` (`:95`/`:470`). | [03 F1](03-gemm-gemv.md#f1-w4a16_gemv_batch2batch3-and-the-qgqkvzdual_-family-are-stuck-on-the-old-narrow-k-loop--09-14-msstep) |
| **0.5–0.9 ms** | `wo_a` (block-diagonal, `o_groups = 8`) is launched **8× serially** — 344 launches/step. `mla.rs:534-580`. | [03 F2](03-gemm-gemv.md#f2-w8a16_gemv_batch4_ld-on-the-block-diagonal-wo_a-is-launched-8-serially--05-09-msstep) |
| **0.2–0.4 ms** | `_ld` GEMVs read `attn_batch` with **`lda = 32768`** — 4096-element strided rows destroy L2 reuse. | [03 F3](03-gemm-gemv.md#f3-the-_ld-gemvs-read-attn_batch-with-lda32768--4096-element-strided-rows-kill-l2-reuse--02-04-msstep) |
| **0.85 ms** | lm_head still FP8; NVFP4 halves it. | [03](03-gemm-gemv.md#findings) |
| **REFUTED** | *"Batched GEMVs re-load the weight per row"* — **false for every kernel in the tree.** Verified line-by-line: 107.15 MB at M=1 vs 107.99 MB at M=6. Full verdict table in 03. This was my prime suspect and it is wrong. | [03 F0](03-gemm-gemv.md#f0-the-prime-suspect-is-false--every-batched-gemv-does-amortise-the-weight-load) |

### Elementwise / norm / mHC (fires 86× per step, 2 sites × 43 layers)

| Est. | Finding | Doc |
|---|---|---|
| **~0.5 ms** | **`hc_post` sharding asymmetry — a live bug.** `decode_inner.rs:505-509` computes `post_shards = 16`; the post-FFN site at `:757-772` uses `hc_post_sharded(..., 16)` but the post-attention site at **`:664-677` calls the unsharded `ops::hc_post`**, which hard-codes `shards = 1` ⇒ grid `(1,1,1)`, **one SM of 48**. Same bug on the verify path at `multi_seq/mod.rs:355`, `:490`, `:543`. One-line fix per site, zero numerical consequence. | [04 #3](04-elementwise-norm-cache.md#findings) |
| **~0.5 ms** | `hc_pre_mix` grid is 25 blocks sized to a **false 25-SM assumption** — 23 of 48 SMs idle. Split-k to 50 blocks. | [04 #1](04-elementwise-norm-cache.md#hc_pre_mix) |
| **0.495 ms** | `hc_fn` is `[24, 16384]` **FP32 = 1.5 MiB**, streamed at all 86 sites = **129 MiB/step**. Largest single number in doc 04. BF16 storage halves it. | [04 #1](04-elementwise-norm-cache.md#findings) |
| **63 µs** | The mHC highway `hc_streams` is `[T,4,4096]` **FP32**, read+written 86×/step. BF16 was tried and **collapsed generation** — correctness-constrained, not a bug. Minimise round trips instead. | [04 #2](04-elementwise-norm-cache.md#findings) |
| latency | **Sinkhorn = 20 sequential iterations inside `if (tid == 0)`**, 255 of 256 threads parked, 86×/step. Dropping the final column projection was A/B-tested and regressed coherence onset 150→90 tokens, so it must stay. Only safe lever is `sinkhorn_iters` (a numerics question). | [04 #5](04-elementwise-norm-cache.md#findings) |
| systemic | **Grid underfill at T=1**: `rms_norm` grid `(1,1,1)`, `argmax_bf16` grid `(1,1,1)` over the **129280**-entry vocab (`argmax_bf16.cu:14`), `hc_head`, `hc_mean`, `hc_expand` all one-block. The `rms_norm` short-row fast path never engages (`hidden <= 256 && rows >= 1024` — H=4096, T=1 fails both). | [04 #4](04-elementwise-norm-cache.md#findings) |
| small | `residual_add.cu` is 11 kernels of **entirely scalar BF16** (1 elem/thread) where `uint4` gives 8× fewer transactions; `quant_rowwise_fp8.cu:38` and `per_token_group_quant_fp8.cu:39` **read the activation twice** (absmax pass then quantise pass); `fused_qkv_norm_rope_cache_write_bf16` (`fused_verify_elemwise.cu:78`) exists but the M=1 decode path doesn't use it (−2064 launches); several RoPE kernels do a per-thread **FP64 `pow()`** at 1/64 rate. | [04](04-elementwise-norm-cache.md#findings) |

### Propose (DSpark drafter) — 63–74 ms against a 40 ms whole-store roofline

Not yet a separate document; both hot spots are in
`crates/spark-model/src/layers/dspark_head.rs`.

| Est. | Finding |
|---|---|
| **~7.8 ms** | **lm_head is run once per block row** (`b = 5`) in a `for r in 0..b` loop (~`:800`): 529 MB FP8 × 5 = **2.65 GB ⇒ 9.7 ms**, where one batched M=5 GEMV would read the weight once (**1.94 ms**). Blocked on the same missing `dense_gemv_fp8w_batchm<8>` as the γ=6 lm_head above. |
| **?** | **`gpu.synchronize(stream)` inside the 5-iteration Markov chain loop** (~`:837`) drains the pipeline 5× per propose. The chain is genuinely sequential (`prev = tok` feeds the next `markov_w1` row lookup) so it cannot be removed outright, but a device-side chain or an event-based dependency would avoid the full host round-trip. |
| — | Ruled out: `dbg()` at `:240` early-returns unless `ATLAS_DSPARK_DEBUG=1`; `seed_position` `main_proj` GEMV at `:319` is [4096 × 12288] ≈ 100 MB ⇒ ~0.37 ms. |

### Correctness bugs found while documenting

| Severity | Bug |
|---|---|
| ~~Live landmine~~ **FIXED** | `paged_decode_attn_fp8_mla.cu`: `VEC_U32_FP8 = 576/(32*4) = 4` truncated, so the FP8 loops covered `4*4 = 16` of `VEC_BF16 = 18` elements — 64 of 576 dims never read for K or V while `o_reg[16..17]` *was* still written. **Investigating the fix turned up a second, worse defect in the same expression:** each lane's slice starts at byte `lane_id * 18`, which is only 2-byte aligned for odd lanes, so the `(const unsigned int*)` casts were issuing **misaligned 4-byte loads in 3 of every 4 lanes** — undefined behaviour, not merely dropped dims. Both are now gated on `FP8_U32_OK (HDIM % (WARP_SIZE*4) == 0)`, with a byte-granular `load_lane_fp8()` fallback that is correct at any HDIM. Verified at the instruction level: the emitted PTX contains 72 `ld.global.u8` (2 kernels × K+V × 18 dims) and the misaligned batched staging path is constant-folded away. |
| Bug | `hc_post` unsharded at 4 call sites (see ledger above) — performance only, no numerical effect. |
| Doc bug | `dense_gemv_bf16.cu:40-41` has a stale comment. |
| Doc bug | `hyper_connection.cu:220`, `:253` assert 25 SMs; the device has 48. |

---

## Non-findings — checked and cleared

Recording these matters as much as the findings; each one closed off a line of attack.

- **Batched GEMVs do NOT re-load weights per row.** Verified across the entire
  `w4a16_*`/`w8a16_*`/`dense_gemv_*` family with a per-kernel verdict table
  ([03](03-gemm-gemv.md#weight-reuse-verdict-table-the-critical-question)).
  Measured: 107.15 MB at M=1 vs 107.99 MB at M=6.
- **The `_m` MoE kernels do NOT re-load weights per row.** Leader election at
  `moe_shared_expert_fused_t.cu:936-981` hoists loads outside the `for m` loop
  (`:1092`, `:1270`); arithmetic intensity scales 3.5 → 21 FLOP/byte. M=6 costs 4.2× M=1
  because **the expert union is 2.93× larger**, not because of reloads.
- **Task #19 was based on an arithmetic error.** "115 µs/expert batched vs 81 µs
  per-token" divided 2247 µs by the worst-case *slot* count (36) instead of the
  distinct-expert *union* (19.5). Correct: **66 µs vs 81 µs — the batched kernel is
  faster.** Recorded in commit `2a957f1c`; task closed as invalid.
- **The `ROWS_` ladder's surplus-row FMAs are free** — the alternative runtime
  `if (m >= M) break` was measured at a fixed **~21% per-byte penalty** (commit `29c6ca6d`).
- **Split-K finalize kernels are bandwidth-trivial** — ~0.3 MB against ~94 MB/layer.
- **`grid.y = total_routed + 1`, not `+ num_tokens`** — the shared expert is read once per
  layer regardless of M. Already optimal.
- **CUDA graphs are worth ~0%** — see the step budget above.

---

## How to reproduce the measurements

```bash
export PATH=/usr/local/cuda/bin:$PATH
export ATLAS_TARGET_MODEL=deepseek-v4-flash          # required on every build
cargo build --release -p spark-server

scripts/dspark-stop.sh          # waits for exit + 20 s CUDA-context settle
ATLAS_UNIFIED_MOE_LAYOUT=1 ATLAS_DSPARK_CAPTURE=1 ATLAS_MTP_TIMING=1 \
  scripts/dspark-serve.sh
```

**Use a prompt of ~450+ tokens.** `fp8_kv_calibration_tokens = 256` keeps CUDA graphs
suppressed until `seq_len > 266`; a short prompt measures the eager path no matter what
`ATLAS_DEBUG_NO_GRAPH` says. Confirm graphs actually engaged:

```
grep -E "FP8 calibration frozen|CUDA graph captured" serve.log
```

Both lines must appear. `ATLAS_MTP_TIMING=1` then prints a per-phase summary every 25 steps.

Other knobs used in these docs: `ATLAS_HC_SPLIT=0` (restore the fused one-block `hc_pre`
pathology for A/B), `ATLAS_DFLASH_DEBUG_NO_GRAPH=1` (verify-path eager only),
`ATLAS_VERIFY_PROFILE`, `ATLAS_PROFILE`, `ATLAS_DSPARK_DEBUG=1`.

---

## Where to start

**Re-ordered after the 24 MB L2 measurement** (see the correction box above). The attention
items were originally ranked first on the assumption that redundant KV reads hit DRAM. They
do not — the whole KV working set is 1.13 MB per layer at `--max-seq-len 1024`. The MoE
weights are the only thing in the step that genuinely cannot be cached, so they are the only
place a multi-millisecond win can come from.

1. **[#31] Nsight Compute on the `_m` MoE GEMVs.** This is now #1, not #5. ~6.9 GB of NVFP4
   expert weights must cross DRAM every verify step; that is ~25 ms of the measured 117–145 ms
   and the remaining ~5× is unexplained. Nothing else in the step is big enough to matter.
   Measurement, not a fix — but the fix is not designable without it.
2. **[#22] Fused gate GEMV + top-k** — `moe_gate_topk_fused` is completely uncoalesced
   (`moe_gate_topk.cu:46-177`) and runs once per layer per step.
3. **[#25] Fuse the duplicate V load** — *done*; keep it because it is free and bit-exact by
   construction, but expect ~0.1 ms, not the 2–4 ms originally estimated.
4. **[#26] Batch the M-row verify attention** — still worth doing at γ=6, where it collapses
   6 kernel *launches* per layer into 1. The win is launch overhead and occupancy, not
   bandwidth.
5. **Shard `hc_post`** at the 4 unsharded call sites — one line each, no numerics. A 1-block
   grid on a 48-SM part is indefensible regardless of how small the kernel is.
6. **Head-tile `mla_paged_decode_fp8`** — demoted from #4. The 64× amplification is real but
   L2-absorbed; a real kernel rewrite for a sub-millisecond return.
