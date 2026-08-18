# MoE Kernels — Per-Kernel Bandwidth Accounting

Target: **DeepSeek-V4-Flash-162B** on one **NVIDIA GB10** (DGX Spark), sm_121, LPDDR5X,
**273 GB/s** peak memory bandwidth, 120 GB unified.

Model shape (from `kernels/gb10/deepseek-v4-flash/nvfp4/MODEL.toml`):

| field | value |
|---|---|
| `hidden_dim` | 4096 |
| `moe_intermediate_size` | 2048 |
| `num_experts` | 256 (slots) |
| `num_shared_experts` | 1 |
| `top_k` | 6 |
| `layers_total` | 43 |
| `mtp_layers` | 5 |
| `vocab_size` | 129280 |
| `default_kv_dtype` | fp8 |

Per routed expert, NVFP4 (E2M1 4-bit weights + FP8-E4M3 per-16 block scales + f32 global scale2):

```
gate:  [2048, 4096]  -> 4096*2048/2 =  4.19 MB packed + 4096*2048/16 = 0.52 MB scales
up:    [2048, 4096]  -> 4.19 MB + 0.52 MB
down:  [4096, 2048]  -> 4.19 MB + 0.52 MB
                        ------------------
                        12.58 MB + 1.57 MB = 14.16 MB per expert
```

At top_k=6 + 1 shared expert that is **7 x 14.16 = 99.1 MB per layer per token** if every
expert is distinct, or **~94 MB** counting only the packed+scale bytes actually touched by
the fused gate_up/silu_down pair (gate and up are fused into one `[2*2048, 4096]` tensor,
so the gate_up kernel reads 8.39 MB packed + 1.05 MB scales and the down kernel reads
4.19 MB + 0.52 MB, 14.15 MB total). Roofline at 273 GB/s: **~51.9 us per expert**,
**~363 us per layer**, **~15.6 ms per token across 43 layers** for the MoE weights alone.

Notation used throughout:

- `M` = number of activation rows in flight. `M=1` plain decode, `M=2` gamma=2 verify,
  `M=6` gamma=6 verify (`MOE_DECODE_MAX_ROWS`).
- `T_SPLIT` = split-K factor, `4` (`forward_phase.rs:19`, `sizes.rs:11`).
- `T_SPLIT_VEC` = per-lane byte vector width, `2` (`fp8_moe.rs:357`).
- `T_BLOCK` = threads per block, `32` (`fp8_moe.rs:295`) — one warp.
- `MROW` = compile-time max gathered rows in the dedup kernels, `6`.
- `GS_R/GS_S` = routed/shared NVFP4 block size (16); `E8M0_R/E8M0_S` = native-MXFP4
  E8M0 exponent-scale mode (block size 32, no global scale2).

---

## Findings

### Verdict on the per-row weight-reload question

**REFUTED. In the batched/verify `_m` (MROW dedup) MoE path the NVFP4 expert weights are
streamed from HBM exactly ONCE PER DISTINCT EXPERT ID, fully amortised across every row
routed to that expert. There is no per-row weight reload anywhere in that path.**

The proof is three links long and all of it is in
`kernels/gb10/common/moe_shared_expert_fused_t.cu`:

1. **Leader election** — `mrow_gather_slots<MROW>` at
   `moe_shared_expert_fused_t.cu:936-981`. Each `grid.y` block owns one flat routed slot
   `y`. Thread 0 scans `s_idx[0..y)` for an earlier slot holding the same expert id `e`;
   if one exists the block is a **duplicate** and sets `s_m = 0`. Otherwise it is the
   **leader** and gathers every slot `s >= y` with `s_idx[s] == e` into `s_slot[0..m)`.

   ```cuda
   const unsigned int e = s_idx[y];
   bool leader = true;
   for (unsigned int s = 0; s < y; ++s) {
       if (s_idx[s] == e) { leader = false; break; }
   }
   if (leader) {
       for (unsigned int s = y; s < total_routed && m < MROW; ++s) {
           if (s_idx[s] == e) s_slot[m++] = s;
       }
   }
   s_m = m;   // 0 => duplicate slot, nothing to do
   ```

2. **Duplicate blocks exit before touching weight memory** — the caller does
   `if (m == 0) return;` immediately after the gather, so a duplicate slot issues **zero**
   global loads against `B_packed` / `B_scale`.

3. **The weight load and E2M1 decode are hoisted OUTSIDE the row loop.** In
   `GATEUP_M_ACCUM` (`moe_shared_expert_fused_t.cu:1092`) and `SILUDOWN_M_ACCUM`
   (`:1270`):

   ```cuda
   load_vec_u8<VEC>(B_packed + (unsigned long long)k_half * N + n, byte);
   float w_lo[VEC], w_hi[VEC];
   for (int v = 0; v < VEC; ++v) {
       w_lo[v] = e2m1_decode(byte[v] & 0xFu) * sc[v];
       w_hi[v] = e2m1_decode((byte[v] >> 4) & 0xFu) * sc[v];
   }
   for (int m = 0; m < (ROWS_); ++m) {                    // <-- row loop is INSIDE
       float a_lo = __bfloat162float(A_row[m][k_half * 2]);
       float a_hi = __bfloat162float(A_row[m][k_half * 2 + 1]);
       for (int v = 0; v < VEC; ++v) acc[m][v] += a_lo * w_lo[v] + a_hi * w_hi[v];
   }
   ```

   One `load_vec_u8` per weight byte, `ROWS_` FMAs against it. `acc[ROWS_][VEC]` lives in
   registers.

The corollary is that the reported **"115 us/expert batched vs 81 us/expert per-token"**
measurement is **not reproducible from this source and is contradicted by the repo's own
recorded numbers**. Commit `2a957f1c` states: *"Per expert the batched kernel is faster
(66us vs 81us) — it is at the bandwidth wall, and the union is simply large."* The
115 us/expert figure almost certainly came from dividing the `exp_splitk_m_t` wall time by
`M * top_k` (the worst-case slot count, 36 at gamma=6) instead of by the **distinct**
expert union (measured mean 19.5). 2247 us / 19.5 = 115 us; 2247 us / 36 = 62 us. That is
the arithmetic error.

Also note `exp_splitk_m_t` is **not a CUDA kernel name**. It is a `prof!` label at
`crates/spark-model/src/layers/moe/forward_km.rs:173`
wrapping the whole of `dispatch_splitk_m_t` — two GEMVs plus two finalize kernels. It is
the only occurrence of that string in the repository.

**So why does an M=6 verify cost ~6x an M=1 decode instead of ~1.05x?** Because the
**expert union is large**, not because bytes are re-read per row. Measured
(`union_stats.rs`, commit `baa21190`): `mean_unique_experts = 19.5` of
`mean_routed_slots = 35.4` — a 45% overlap saving available, realised as a 1.30x
end-to-end win (`exp_splitk_m_t` 2247 us vs 6 x 487 us). Six rows on a hash-routed layer
select **near-disjoint** expert sets, so the union approaches the `M * top_k` worst case.
The earlier claim (commit `29c6ca6d`) that hash layers pick identical top-6 for every row
was an artefact of a routing bug where the per-token path read `token_ids[0]`
unconditionally; commit `2a957f1c` fixed that and explicitly corrects the claim.

### Optimisation opportunities, priority order

Baseline for the estimates: per-layer verify budget 4.55 ms at gamma=6, of which MoE is
3.26 ms (commit `2a957f1c`); 43 layers; plain decode MoE ~15.6 ms/token at roofline.

| # | Opportunity | Where | Est. saving |
|---|---|---|---|
| 1 | Close the 136 -> 194 GB/s gap in the `_m` kernels | `moe_shared_expert_fused_t.cu:983`, `:1162` | **~29 ms per gamma=6 verify step** (~0.68 ms/layer) |
| 2 | Shrink the expert union (routing-aware verify batching / drafter-target expert alignment) | `forward_km.rs:234`, `mod.rs:625` | up to **~60 ms/step** at the 45%-overlap ceiling; ~15 ms/step realistically |
| 3 | Stage gate_up `A` rows in shared memory like `silu_down` already does | `moe_shared_expert_fused_t.cu:1073-1078`, `:1120-1121` | **~6-10 ms/step** at gamma=6 |
| 4 | Cut `silu_down` dynamic smem so a 32-thread block is not occupancy-capped | `fp8_moe.rs:507`, `moe_shared_expert_fused_t.cu:1162` | **~4-8 ms/step** |
| 5 | Parallelise `mrow_gather_slots` leader election across the warp | `moe_shared_expert_fused_t.cu:936-981` | **~1-2 ms/step** |
| 6 | Port branch-free `e2m1_decode` into the 7 legacy LUT-based decode kernels | see list below | 0 on the hot path; ~1.5x on any fallback |
| 7 | Fix `moe_sort_by_expert` / `moe_build_tile_worklist` serial single-block phases | `moe_permute.cu:200`, `:273` | prefill only, ~0.2 ms/prefill call |

**1. The 136 vs 194 GB/s gap (largest single win).** Commit `f5aac0c4` records: *"The
kernel still only reaches ~136 GB/s on the union it reads against the single-row path's
194 GB/s."* The `_m` kernels are reading the same weight bytes with the same VEC=2 access
pattern as the M=1 path, so the 30% deficit is pure overhead — leader-election serial
scan, dynamic-smem occupancy loss, and the un-staged `A` row gathers. Recovering it takes
the per-layer verify MoE from 3.26 ms to ~2.29 ms: **0.97 ms x 43 = ~42 ms**, of which
items 3-5 below are the identified mechanisms; conservatively **~29 ms/step**.

**2. Union size is the floor, not per-expert cost.** With weights already read once per
distinct expert, the only remaining lever on total bytes is *how many distinct experts the
M rows select*. At gamma=6 the measured union is 19.5/36. Driving it toward the microtest
"overlap 3" regime (21/36 distinct, 1.92x ceiling, +31.8% measured in `f5aac0c4`) requires
changing routing, not the kernel. Concretely: bias the DSpark drafter's own gate toward
the target model's expert selection, or reorder the verify batch so tokens with
overlapping routing are grouped. The measured microtest ladder for MROW=6 (32 tokens,
pool 36, top_k 6):

| routing | distinct/36 | ceiling | measured |
|---|---|---|---|
| disjoint | 36 | 1.14x | -0.1% |
| random | 23 | 1.76x | +23.8% |
| overlap 3 | 21 | 1.92x | +31.8% |
| identical | 6 | 6.00x | +85.9% |

The measured column tracks ~55-65% of the ceiling, so a union reduction from 19.5 to ~14
distinct would be worth roughly 25% of the MoE verify time: **~35 ms/step**.

**3. gate_up re-reads the `A` activation rows from HBM per row.** `silu_down` stages its
input into dynamic shared memory (`s_act`, `moe_shared_expert_fused_t.cu:483` and the
`_m` variant at `:1162`), but `gate_up` does not — `A_row[m]` at `:1073-1078` are raw
global `__nv_bfloat16*` pointers dereferenced inside the row loop at `:1120-1121`. Every
block in `grid.x` (32 blocks at N=2048, VEC=2, T_BLOCK=32) re-reads the same
`M * 4096 * 2 = 49 KB` of activations. At gamma=6 with a 19.5-expert union across 8
grid.z slices: `19.5 * 32 * 8 * 49 KB` is nominally 244 MB/layer of activation traffic,
almost all of it L2 hits — but it is what pins the kernel below its weight-stream
roofline. Staging `A` into smem once per block (or into registers, since 4096 BF16 =
8 KB per row) should recover most of the gap. Estimated **~6-10 ms/step**.

**4. `silu_down` dynamic smem scales linearly with MROW.**
`crates/spark-model/src/layers/ops/fp8_moe.rs:507`:

```rust
let smem_bytes = mrow * (k as usize * size_of::<f32>()) as u32 / split;
```

At `mrow=6, k=2048, split=4` that is **12288 B** per 32-thread block, versus **2048 B**
on the M=1 path. sm_121 gives 100 KB of smem per SM, so 12 KB caps residency at 8 blocks
per SM — 8 warps, 256 threads out of a 1536-thread SM budget: **~17% occupancy**. With
only 48 SMs on GB10 and 37 x 4 = 148 blocks to schedule for `silu_down` at gamma=6, this
directly limits memory-level parallelism. Two fixes: store the staged activations as BF16
instead of F32 (halves to 6 KB), or raise `T_BLOCK` from 32 to 64/128 for the `_m` path
only so the smem/thread ratio drops. Estimated **~4-8 ms/step**.

**5. Leader election is serial on thread 0 with 31 lanes idle.**
`mrow_gather_slots` (`:936-981`) runs an `O(y)` scan plus an `O(total_routed - y)` gather
entirely inside `if (threadIdx.x == 0)`, then `__syncthreads()`. At gamma=6,
`total_routed = 36`, so worst case is 36 + 36 = 72 serial iterations of dependent
`s_idx[]` smem loads before any weight byte is fetched — roughly 1-2 us of pure latency
per block, times 37 grid.y x 8 grid.z blocks. It is trivially a warp ballot: each lane
tests one slot, `__ballot_sync` + `__ffs` gives the leader, `__popc` + prefix gives the
gather offsets. Estimated **~1-2 ms/step**.

**6. Seven kernels still use the shared-memory E2M1 LUT.** The branch-free
`e2m1_decode` at `moe_shared_expert_fused_t.cu:56` replaced a `__shared__ float
s_lut[16]` that bank-conflicted and held the GEMV at ~127 GB/s; the branch-free form
reaches ~194 GB/s. Still on the LUT: `moe_gate_topk.cu`, `moe_expert_gemv.cu`,
`moe_expert_gemv_fused.cu`, `moe_sorted_prefill.cu`, `moe_prefill.cu`,
`moe_decode_atomic_c4.cu`, `moe_expert_relu2_down_shared.cu`. None are on the DeepSeek-V4
hot path today, so the saving is 0 ms/step now, but any fallback dispatch (batch sizes
outside the `_m` ladder, or a model without the `_t` transposed weights) pays ~1.5x.

**7. Prefill-only serial phases.** `moe_sort_by_expert` (`moe_permute.cu:200`) does its
histogram with `atomicAdd` into smem and then a **single-block, single-thread** exclusive
prefix sum over `num_experts` (256 iterations). `moe_build_tile_worklist`
(`moe_permute.cu:273`) is a single-thread triple-nested loop. Both are microseconds but
they serialise the whole grouped-GEMM launch chain.

### Non-findings (checked and cleared)

- **No per-row weight reload in the `_m` path** — see verdict above.
- **The split-K finalize kernels are bandwidth-trivial.** Source comment at
  `moe_shared_expert_fused_t.cu:852`: *"this kernel is bandwidth-trivial — ~0.3 MB against
  the ~94 MB/layer the GEMVs stream"*. Confirmed by the byte counts in their sections
  below.
- **The `ROWS_` ladder's surplus-row FMAs are free.** Arms exist at 1, 2, 4, MROW
  (`:1144-1145`); a 3-row group runs the 4-row arm with row 3 pointed at `A` row 0 and
  dropped at emit. That is one wasted FMA per weight byte per surplus row — arithmetic,
  not bandwidth, on a kernel with AI ~0.5 FLOP/byte. The alternative (a runtime
  `if (m >= M) break` inside the k loop) was **measured at a fixed ~21% per-byte penalty**
  (commit `29c6ca6d`). The ladder is correct.
- **`grid.y = total_routed + 1`, not `+ num_tokens`.** `fp8_moe.rs:432`: *"one block-set
  computes the shared projection for every row"*. The shared expert is read once per
  layer regardless of M. Correct and already optimal.

---

## Launch geometry summary

Derived from `crates/spark-model/src/layers/ops/fp8_moe.rs`
at hidden=4096, moe_intermediate=2048, top_k=6, `T_BLOCK=32`, `T_SPLIT_VEC=2`,
`T_SPLIT=4`.

| path | kernel | grid | block | dyn smem |
|---|---|---|---|---|
| M=1 decode | `..._t_e8m0_v2s4` gate_up | `[32, 7, 8]` | `[32,1,1]` | 0 |
| M=1 decode | `..._t_e8m0_v2s4` down | `[64, 7, 4]` | `[32,1,1]` | 2048 B |
| M=1 decode | `moe_gate_up_partial_finalize` | `[64, 7, 2]` | `[32,1,1]` | 0 |
| M=1 decode | `moe_down_partial_finalize` | `[128, 7, 1]` | `[32,1,1]` | 0 |
| M=2 verify | `..._m2v2s4` gate_up | `[32, 13, 8]` | `[32,1,1]` | 0 |
| M=2 verify | `..._m2v2s4` down | `[64, 13, 4]` | `[32,1,1]` | 4096 B |
| M=6 verify | `..._m6v2s4` gate_up | `[32, 37, 8]` | `[32,1,1]` | 0 |
| M=6 verify | `..._m6v2s4` down | `[64, 37, 4]` | `[32,1,1]` | 12288 B |
| M=6 verify | `moe_gate_up_partial_finalize_m` | `[64, 42, 2]` | `[32,1,1]` | 0 |
| M=6 verify | `moe_down_partial_finalize_m` | `[128, 42, 1]` | `[32,1,1]` | 0 |

`grid.y = total_routed + 1` where `total_routed = M * top_k`; the `+1` is the shared
expert. Finalize `grid.y = moe_splitk_m_rows(top_k, M) = M*top_k + M`
(`fp8_moe.rs:374`).

---

# Part A — Routing and top-k kernels

These are all tiny: one block per token, a few KB of gate logits. At `num_experts=256`
the gate logit tensor is 256 f32 = 1 KB per token. Every kernel here is **latency-bound,
not bandwidth-bound** — they exist to keep the routing decision on-device so the MoE
dispatch can be CUDA-graph captured without a D2H sync.

## moe_hash_route

`kernels/gb10/common/moe_hash_route.cu:25`

DeepSeek-V4's hash routing. For each token it reads the static `tid2eid[token_id]` table
(a `[vocab_size, top_k]` i32 map baked at load time), copies the `top_k` expert ids into
`out_expert_ids`, and separately evaluates the learned gate to produce only the **weights**
— expert *selection* is static, the gate contributes gating scalars. This is what makes
gamma=6 verify unions near-disjoint: the six speculated tokens are six different vocab
ids, so they index six unrelated rows of `tid2eid`.

- **grid/block**: one block of 256 threads per token; grid `[num_tokens]`. Launched from
  `crates/spark-model/src/layers/moe/forward_km.rs:234`
  (`route_rows_flat`, one launch per row) and from the batched path via
  `moe_hash_route_batched`.
- **dtypes**: `tid2eid` i32, gate logits f32, output ids i32 + weights f32.
- **smem**: none beyond a small reduction scratch.
- **bytes/call**: `top_k * 4` (ids) + `num_experts * 4` (gate row) + `top_k * 4` (weights
  out) = 24 + 1024 + 24 = **1072 B per token**. M=1: 1.07 KB. M=2: 2.14 KB. M=6: 6.43 KB.
  Roofline at 273 GB/s: **~4 ns at M=1** — entirely launch-latency dominated (~3-5 us
  actual).
- **AI**: ~1 FLOP/byte. **Latency-bound.**
- **Inefficiency**: `route_rows_flat` at `forward_km.rs:234` issues **one launch per row**
  rather than using `moe_hash_route_batched`. At gamma=6 that is 6 launches x 43 layers =
  258 extra kernel launches per verify step, ~1.3 ms of pure launch overhead if not
  graph-captured. Under CUDA graph capture the cost collapses; outside it this is real.

## moe_hash_route_batched

`kernels/gb10/common/moe_hash_route.cu:70`

Same computation with `grid = [num_tokens]` so all M rows route in one launch. Reads
`token_ids[blockIdx.x]` rather than `token_ids[0]` — this is the fix from commit
`2a957f1c`; the pre-fix per-token path read index 0 unconditionally and routed rows 1..M
with row 0's experts, which is why online acceptance measured 2.30 against an offline
3.81.

- **bytes/call**: `M * 1072 B`. M=6: **6.43 KB**, ~24 ns at roofline.
- **Recommendation**: this is the kernel `forward_km.rs:234` should be calling.

## moe_topk_softmax

`kernels/gb10/common/moe_topk.cu:30`

Classic softmax-then-top-k. One block per token, 256 threads. Reads the `[num_experts]`
f32 logit row into registers, does a block-wide max reduction, `expf`, a sum reduction,
then `top_k` iterative argmax passes each masking the previous winner. Writes `top_k` i32
ids and `top_k` f32 weights, renormalised to sum 1.

- **grid/block**: `[num_tokens]` x `[256,1,1]`.
- **bytes/call**: `num_experts*4` in + `top_k*8` out = 1024 + 48 = **1072 B/token**.
- **AI**: ~6 FLOP/byte (softmax + 6 argmax passes over 256). **Latency-bound.**
- **Inefficiency**: `top_k` sequential full-array argmax passes = 6 x 256 = 1536
  comparisons serialised as 6 dependent block reductions. A single bitonic partial sort
  would be one pass. Irrelevant at 1 KB but it is ~6 dependent `__syncthreads()`.

## moe_topk_softmax_f32

`kernels/gb10/common/moe_topk.cu:189`

Identical to `moe_topk_softmax` but takes f32 logits directly rather than BF16, skipping
the up-convert. Same geometry, same 1072 B, same bound.

## moe_topk_softmax_batched

`kernels/gb10/common/moe_topk.cu:316`

`grid = [num_tokens]` batched form. `M * 1072 B`. Used by the batched prefill dispatch.

## moe_topk_sigmoid

`kernels/gb10/common/moe_topk_sigmoid.cu:22`

Sigmoid gating with the **noaux_tc correction bias**: scores are `sigmoid(logit) +
correction_bias[e]` for selection, but the emitted weight is the *uncorrected*
`sigmoid(logit)`, renormalised over the chosen top_k and scaled by `routed_scaling_factor`.
This is the DeepSeek-V3/V4 auxiliary-loss-free load-balancing scheme — the bias term
shifts selection without perturbing the gradient path.

- **grid/block**: `[num_tokens]` x `[256,1,1]`.
- **bytes/call**: `num_experts*4` logits + `num_experts*4` correction bias + `top_k*8` out
  = 1024 + 1024 + 48 = **2096 B/token**. M=6: 12.6 KB, ~46 ns roofline.
- **Bound**: latency. The correction bias is a persistent `[256]` f32 vector, L2-resident
  after the first layer.

## moe_topk_sigmoid_batched

`kernels/gb10/common/moe_topk_sigmoid.cu:130`

Batched form, `grid=[num_tokens]`. `M * 2096 B`.

## moe_topk_sqrtsoftplus

`kernels/gb10/common/moe_topk_sqrtsoftplus.cu:22`

The DeepSeek-V4-Flash gate: score = `sqrt(softplus(logit))` = `sqrt(log(1+exp(x)))`, again
with a selection-time correction bias. Numerically the `softplus` is done in the
overflow-safe form `max(x,0) + log1p(exp(-|x|))`.

- **grid/block**: `[num_tokens]` x `[256,1,1]`.
- **bytes/call**: **2096 B/token**, same as sigmoid.
- **AI**: highest of the routing kernels — `expf`, `log1pf`, `sqrtf` per expert = ~40
  FLOP-equivalents x 256 = ~10k FLOP against 2 KB, so **~5 FLOP/byte** but with
  transcendental-unit serialisation. Still **latency-bound** at 256 experts / 256 threads
  = 1 expert per thread.

## moe_topk_sqrtsoftplus_batched

`kernels/gb10/common/moe_topk_sqrtsoftplus.cu:129`

Batched form. `M * 2096 B`. This is the one that *should* be used at gamma=6 instead of
the per-row loop in `forward_km.rs:234`.

## moe_gate_topk_fused

`kernels/gb10/common/moe_gate_topk.cu:46`

Fuses the gate projection itself into the top-k: computes `logits = W_gate @ x` for the
NVFP4-quantised `[num_experts, hidden]` gate weight, then softmax-top-k, in one launch.
Avoids materialising the logit tensor.

- **grid/block**: `[num_tokens]` x `[256,1,1]`; each thread owns one expert's dot product
  over hidden=4096.
- **bytes/call**: gate weight `256 * 4096 / 2 = 512 KB` packed + `256*4096/16 = 64 KB`
  scales + `4096*2 = 8 KB` activation = **584 KB per call**, independent of M if batched.
  Roofline: **2.1 us**. Across 43 layers that is **92 us/token** — non-trivial, ~0.6% of
  the MoE budget.
- **AI**: `2 * 256 * 4096 = 2.1 MFLOP / 584 KB` = **3.6 FLOP/byte**. **Bandwidth-bound.**
- **Inefficiency**: uses the **shared-memory E2M1 LUT** decode, not the branch-free form.
  At the measured 127 vs 194 GB/s that costs ~0.9 us/layer, **~39 us/token**. Porting
  `e2m1_decode` from `moe_shared_expert_fused_t.cu:56` is a two-line change.
  Also: one thread per expert means each thread does a strided 4096-element walk — the
  256 threads of the block read 256 *different* rows of `W_gate` simultaneously, so with
  the `[num_experts, hidden/2]` output-major layout the accesses are 2048 B apart and
  **completely uncoalesced**. A transposed `[hidden/2, num_experts]` gate weight would make
  lane `n` read byte `n` of a contiguous line. This is likely a >2x win on this kernel.

---

# Part B — Permute, blend, and utility kernels

## moe_permute_tokens

`kernels/gb10/common/moe_permute.cu:20`

Gathers `hidden`-wide BF16 token rows into expert-sorted order using a precomputed
`sorted_to_orig` index. `out[i] = in[sorted_to_orig[i]]`.

- **grid/block**: `[num_sorted_rows]` x `[256,1,1]`, each block copies one 4096-element row
  with 16 elements per thread.
- **bytes/call**: `2 * num_rows * hidden * 2` (read + write). At prefill with 4096 tokens
  and top_k=6: `2 * 24576 * 8192 = 402 MB`, **1.47 ms**. At decode M=1 this path is not
  used at all — the `_t` GEMV family reads activations in place.
- **AI**: 0 FLOP/byte. **Pure bandwidth.**
- **Inefficiency**: it is a full materialised copy of the activation tensor at top_k
  expansion. The `_t` decode path avoids it entirely by indexing; the prefill path pays
  it. Fusing the gather into the grouped-GEMM's A-tile load would remove 402 MB per
  prefill.

## moe_unpermute_reduce

`kernels/gb10/common/moe_permute.cu:45`

Inverse of the above with the weighted sum folded in: for each original token, sums its
`top_k` expert outputs scaled by the gate weights.

- **grid/block**: `[num_tokens]` x `[256,1,1]`.
- **bytes/call**: `num_tokens * top_k * hidden * 2` read + `num_tokens * hidden * 2` write
  = at 4096 tokens `4096*6*4096*2 + 4096*4096*2 = 201 + 34 = 235 MB`, **0.86 ms**.
- **AI**: `top_k` FMAs per element = 6 FLOP / 14 B = **0.43 FLOP/byte**. **Bandwidth-bound.**

## moe_count_experts

`kernels/gb10/common/moe_permute.cu:77`

Histogram of expert ids into a `[num_experts]` i32 counter array via `atomicAdd`.

- **grid/block**: `[div_ceil(num_rows, 256)]` x `[256,1,1]`.
- **bytes/call**: `num_rows * 4` read + 256 atomics. At 24576 rows: **98 KB**, ~0.4 us.
- **Inefficiency**: global `atomicAdd` with no smem privatisation. With 256 experts and
  24576 rows the average contention is 96 increments per counter — tolerable, but a
  per-block smem histogram flushed once would cut global atomics 96x.

## moe_unpermute_reduce_indexed

`kernels/gb10/common/moe_permute.cu:95`

Same as `moe_unpermute_reduce` but takes an explicit `[num_tokens, top_k]` slot-index
table instead of assuming contiguous top_k blocks. Same byte count and bound; used when
the sort produced a non-trivial permutation.

## moe_batched_blend

`kernels/gb10/common/moe_permute.cu:126`

Blends the routed-expert output with the shared-expert output for a batch of rows:
`out[t] = shared[t] + sum_k w[t][k] * routed[t][k]`.

- **grid/block**: `[num_tokens, ?]` x `[256,1,1]`.
- **bytes/call**: at M=6, hidden=4096: `6 * (6+1) * 4096 * 2` read + `6*4096*2` write =
  **393 KB**, ~1.4 us. Per layer per verify step. 43 layers: **62 us**.
- **AI**: 7 FMA / 16 B = **0.44 FLOP/byte**. **Bandwidth-bound**, but small.

## moe_sort_by_expert

`kernels/gb10/common/moe_permute.cu:200`

Counting sort of `(token, slot)` pairs by expert id. Phase 1: smem histogram with
`atomicAdd`. Phase 2: **single-thread exclusive prefix sum over `num_experts`**. Phase 3:
scatter with `atomicAdd` on the running offsets.

- **grid/block**: launched as a **single block** (the prefix sum requires global ordering).
- **bytes/call**: `num_rows * 4 * 3` = at 24576 rows **295 KB**, ~1.1 us at roofline but
  the serial prefix sum adds 256 dependent smem iterations.
- **Inefficiency**: the single-block, single-thread prefix sum. `num_experts=256` so it is
  256 serial adds; a warp-level `__shfl_up_sync` scan would be 8 steps. Also the whole
  kernel is one block on a 25-SM machine — **4% occupancy**. Prefill-only.

## moe_build_tile_worklist

`kernels/gb10/common/moe_permute.cu:273`

Compacts the `(expert, m_tile, n_tile)` triple loop into a dense worklist so the grouped
GEMM launches exactly the tiles that have rows, rather than a rectangular
`[n_tiles, max_m_tiles, num_experts]` grid where most `(m_tile, expert)` pairs are empty.

- **grid/block**: **single thread**, single block.
- **bytes/call**: `num_experts * 4` read, `num_tiles * 12` write. Trivially small.
- **Inefficiency**: literally one thread running a triple-nested loop over
  `num_experts x max_m_tiles x n_tiles`. At 256 experts this is thousands of serial
  iterations, ~10-50 us. Should be a parallel scan + compaction. Prefill-only, but it is
  on the critical path of every grouped-GEMM launch.

## moe_silu_mul

`kernels/gb10/common/moe_silu_mul.cu:14`

`out[i] = silu(gate[i]) * up[i]` with the gate clamped to the NVFP4 representable range
before the sigmoid, matching the reference implementation's saturation.

- **grid/block**: `[div_ceil(n, 256)]` x `[256,1,1]`.
- **bytes/call**: `2 * n * 2` read + `n * 2` write = `6n` B. At `n = num_rows * 2048`:
  M=1 decode `6 * 7 * 2048 = 86 KB` (~0.3 us); M=6 verify `6 * 37 * 2048 = 455 KB`
  (~1.7 us). 43 layers at M=6: **72 us**.
- **AI**: 1 `expf` + 3 FLOP per 6 B = **~1 FLOP/byte** (plus a transcendental).
  **Bandwidth-bound.**
- **Note**: on the `_t` decode path this is **fused into `silu_down`** and never launched.
  It exists for the split gate_up/down prefill path.

## silu_mul_noclamp

`kernels/gb10/common/moe_silu_mul.cu:42`

Same without the clamp — used when the downstream weight format has no saturation
requirement (BF16/FP8 experts). Identical geometry and byte count.

## moe_transpose_u8_batched

`kernels/gb10/common/moe_transpose_batched.cu:21`

Transposes packed u8 (2x E2M1 nibbles) expert weight tensors from the original
`[N, K/2]` output-major layout to the `[K/2, N]` input-major `_t` layout that the decode
GEMVs need for coalescing. Runs **once at model load**, not per token.

- **grid/block**: `[div_ceil(cols,32), div_ceil(rows,32), num_experts]` x `[32,8,1]`
  — from `crates/spark-model/src/layers/ops/fp8_moe.rs:260`.
  32x32 smem tile, 8 rows per thread.
- **smem**: `32 * 33` u8 = 1056 B (the +1 pad kills the bank conflict).
- **bytes/call**: `2 * num_experts * N * K/2`. For the full model: 43 layers x 256 experts
  x 12.58 MB x 2 = **277 GB**, ~17 minutes at 273 GB/s if done naively. In practice it is
  done once and cached to disk.
- **Why it matters**: the `_t` layout is what makes `B_packed[k_half * N + n]` coalesced —
  lane `n` within a warp reads byte `n` of a contiguous 32-byte (VEC=1) or 64-byte (VEC=2)
  line. In the original `[N, K/2]` layout, lane `n` would read `n * K/2` bytes apart,
  giving 32 separate 32 B transactions per warp instead of 1-2. This transpose is
  responsible for roughly a **10x** improvement in achieved bandwidth on the GEMV.

---

# Part C — The `_t` decode GEMV family (the hot path)

Everything in this section lives in
`kernels/gb10/common/moe_shared_expert_fused_t.cu` (1467
lines). This is the file that determines decode speed.

Shared infrastructure:

- **`e2m1_decode`** (`:56`) — branch-free 4-bit E2M1 to f32. Replaced a
  `__shared__ float s_lut[16]` that bank-conflicted; worth ~127 -> ~194 GB/s.
- **`load_vec_u8<VEC>`** (`:76`) — VEC-wide packed load (`uchar2` at VEC=2, `uchar4` at
  VEC=4). VEC=2 is the production setting.
- **`store_vec_bf16<VEC>`** (`:93`).
- Template parameters `<GS_R, E8M0_R, GS_S, E8M0_S, VEC, SPLIT>` — routed/shared block
  size and E8M0-vs-NVFP4 mode are compile-time, so the scale decode is branch-free.

## gate_up_shared_t_impl (device template)

`moe_shared_expert_fused_t.cu:133`

The single-row fused gate+up projection. `blockIdx.y == 0` selects the shared expert;
`blockIdx.y >= 1` selects routed slot `y-1`. `blockIdx.z` selects `{gate, up} x {split
slice}`. Each of the 32 lanes owns `VEC` adjacent output columns `n`; the k loop walks
`k_len = K / SPLIT` halves, loading `B_packed[k_half * N + n]` (coalesced), decoding two
nibbles, and FMA-ing against `A[k_half*2]` and `A[k_half*2+1]`.

- **grid**: `[N/(T_BLOCK*VEC), top_k+1, 2*SPLIT]` = `[32, 7, 8]` at production settings
  (`fp8_moe.rs:404`).
- **block**: `[32,1,1]`.
- **smem**: none.
- **dtypes**: A BF16, B packed u8 (2x E2M1), scales FP8-E4M3 (or E8M0), accumulate f32,
  partials f32.
- **bytes per expert per call**: `2 * 2048 * 4096 / 2 = 8.39 MB` packed +
  `2*2048*4096/16 = 1.05 MB` scales + `4096 * 2 = 8 KB` A (re-read by all 32 grid.x
  blocks, L2) + `2*2048*4*SPLIT = 65.5 KB` f32 partials out.
  Total **~9.51 MB per expert**. At 7 experts (6 routed + shared): **66.6 MB/layer**.
  Roofline: **244 us/layer**, **10.5 ms/token across 43 layers**.
- **AI**: `2 * 2 * 2048 * 4096 = 33.6 MFLOP / 9.51 MB` = **3.5 FLOP/byte**. GB10 peak is
  well above 100 FLOP/byte for BF16 FMA, so **firmly bandwidth-bound**.
- **Measured**: ~194 GB/s achieved (commit `f5aac0c4`), i.e. **71% of peak**.
- **Weights read**: ONCE per expert. `M=1`, so the question is trivially yes here.

## silu_down_shared_t_impl (device template)

`moe_shared_expert_fused_t.cu:483`

Fuses `silu(gate) * up` with the down projection. Stages the `K = moe_intermediate` slice
of activations into **dynamic shared memory** `s_act` (`k_len = K / SPLIT` f32 = 2048 B at
SPLIT=4), computing the SiLU product once per block rather than per output column. Then
the same coalesced `[K/2, N]` walk over the down weight.

- **grid**: `[N/(T_BLOCK*VEC), top_k+1, SPLIT]` = `[64, 7, 4]` (`fp8_moe.rs:481`).
- **block**: `[32,1,1]`.
- **smem**: `K * 4 / SPLIT = 2048 B` at M=1 (`fp8_moe.rs:507` with `mrow=1`).
- **bytes per expert per call**: `4096 * 2048 / 2 = 4.19 MB` packed + `0.52 MB` scales +
  gate/up activations `2 * 2048 * 2 = 8 KB` + `4096 * 4 * SPLIT = 65.5 KB` partials out.
  Total **~4.79 MB per expert**, **33.5 MB/layer** at 7 experts.
  Roofline: **123 us/layer**, **5.3 ms/token**.
- **AI**: `2 * 4096 * 2048 = 16.8 MFLOP / 4.79 MB` = **3.5 FLOP/byte**. **Bandwidth-bound.**
- **Combined gate_up + down**: **100 MB/layer**, **367 us/layer**, **15.8 ms/token** —
  matching the 15.6 ms figure derived from the model shape at the top of this document,
  and consistent with the ~47.6 ms/token plain-decode target once attention, norms, and
  the LM head are added.

## moe_expert_gate_up_shared_t / _e8m0 / _v4 / _e8m0_v4 / _v2 / _e8m0_v2

`moe_shared_expert_fused_t.cu:290, 322, 355, 383, 415, 443`

Six `extern "C"` entry points instantiating `gate_up_shared_t_impl` at `SPLIT=1` with
`VEC in {1, 4, 2}` x `{NVFP4 GS=16, native MXFP4 E8M0 GS=32}`. Non-split-K, so each block
computes a complete output element and writes BF16 directly — no partials, no finalize.

- **grid**: `[div_ceil(n, T_BLOCK), top_k+1, 2]` = `[64, 7, 2]` at VEC=1
  (`fp8_moe.rs:299`).
- **bytes**: same weight traffic as the impl above minus the 65.5 KB of partials
  (**9.44 MB/expert**), plus a `2*2048*2 = 8 KB` BF16 output.
- **Why they lost to split-K**: at `grid = [64, 7, 2]` = **896 blocks of 32 threads** on
  48 SMs, but the 1-byte-per-lane load pattern at VEC=1 is latency-limited, not
  occupancy-limited. `fp8_moe.rs:348-356` records the measurement: *"the 1-byte-per-lane
  load ... is pinned near 130 GB/s no matter how many warps are resident — 4x the warps
  moved it only 128 -> 136 GB/s ... VEC=2 with the warps put back by split-K measured
  197 GB/s, or 32.5 -> 20.6 ms/token of MoE."* VEC=2 widens each lane's transaction to 2 B
  (64 B/warp), and SPLIT=4 restores the block count that VEC=2 would otherwise have
  quartered.

## moe_expert_silu_down_shared_t / _e8m0 / _v4 / _e8m0_v4 / _v2 / _e8m0_v2

`moe_shared_expert_fused_t.cu:630, 653, 676, 698, 721, 743`

The corresponding six non-split-K down entries. `smem = K * 4 = 8192 B` (SPLIT=1).
**4.72 MB/expert**. Same VEC/format matrix, same conclusion.

## GATEUP_SPLIT_ENTRY / DOWN_SPLIT_ENTRY (16 split-K entries)

`moe_shared_expert_fused_t.cu:774` and `:805`

Two macros generating the production single-row split-K entries:
`moe_expert_gate_up_shared_t{,_e8m0}_{v2s2,v2s4,v4s2,v4s4}` and
`moe_expert_silu_down_shared_t{,_e8m0}_{v2s2,v2s4,v4s2,v4s4}` — 8 each, 16 total.
Production uses **`_e8m0_v2s4`**.

Each block computes a `K/SPLIT` slice and writes an f32 partial at
`partials[(ks * rows + row) * N + n]`. The finalize sums in **ascending `ks` order**, which
makes the result bit-reproducible run to run (but *not* bit-equal to the SPLIT=1 path,
since f32 addition is not associative).

- **grid**: `[N/(32*VEC), top_k+1, 2*SPLIT]` gate_up, `[N/(32*VEC), top_k+1, SPLIT]` down.
- **partials traffic**: gate_up `2 * 2048 * 4 * 4 = 65.5 KB` per expert; down
  `4096 * 4 * 4 = 65.5 KB`. Combined **131 KB/expert**, **917 KB/layer** — **0.9%** of the
  100 MB/layer weight stream. The split-K reduction overhead is negligible; the 4x block
  count it buys is worth far more.
- **Split-K accuracy note**: the partial buffer is `MOE_DECODE_MAX_SPLIT = 4` deep
  (`crates/spark-runtime/src/buffers/sizes.rs:11`), so SPLIT
  cannot exceed 4 without a buffer resize.

## moe_gate_up_partial_finalize

`moe_shared_expert_fused_t.cu:857`

Sums the `SPLIT` f32 partials per output element and writes BF16.

- **grid**: `[div_ceil(n, T_BLOCK), moe_splitk_m_rows(top_k, num_tokens), 2]` = `[64, 7, 2]`
  at M=1 (`fp8_moe.rs:404`); block `[32,1,1]`.
- **bytes/call**: read `2 * 2048 * 4 * SPLIT * 7 = 459 KB`, write `2*2048*2*7 = 57 KB`.
  Total **516 KB/layer**, **1.9 us**. Source comment at `:852`: *"this kernel is
  bandwidth-trivial — ~0.3 MB against the ~94 MB/layer the GEMVs stream"*. Confirmed.
- **AI**: 3 adds per 20 B = **0.15 FLOP/byte**. Bandwidth-bound but irrelevant.

## moe_down_partial_finalize

`moe_shared_expert_fused_t.cu:883`

Same for the down projection, plus the **gate-weight scaling and accumulation into the
output**: multiplies each routed expert's contribution by its gate weight and adds it to
the shared-expert result. This is why there is no separate blend kernel on the `_t` decode
path.

- **grid**: `[div_ceil(n, T_BLOCK), moe_splitk_m_rows(top_k, num_tokens), 1]` = `[128, 7, 1]`.
- **bytes/call**: read `4096 * 4 * 4 * 7 = 459 KB`, write `4096 * 2 = 8 KB`.
  **467 KB/layer**, **1.7 us**.

---

# Part D — The `_m` MROW dedup verify GEMV family

Same file. This is where the gamma>1 verify time is spent, and where the per-row weight
reload question is answered.

## mrow_gather_slots (device function)

`moe_shared_expert_fused_t.cu:936-981`

Not a `__global__`, but the load-bearing piece. Documented in full in the Findings verdict
above. Contract (header comment at `:903-930`):

```
gate_up: grid = [N/(block*VEC), num_tokens*top_k + 1, 2*SPLIT]
down:    grid = [N/(block*VEC), num_tokens*top_k + 1, SPLIT]
         smem = MROW * K*4/SPLIT
```

- Thread 0 does an `O(y)` duplicate scan then an `O(total_routed - y)` gather; 31 lanes
  wait at `__syncthreads()`.
- `s_slot[MROW]` and `s_m` live in static smem.
- `is_shared` (`blockIdx.y == 0`) takes a different branch: it gathers the first `MROW`
  token rows unconditionally, since the shared expert applies to every row.
- **Cost**: up to `2 * total_routed = 72` dependent smem loads at gamma=6, ~1-2 us per
  block of pure serial latency before the first weight byte is requested. See Finding 5.

## gate_up_shared_t_m_impl (device template)

`moe_shared_expert_fused_t.cu:983`

The MROW gate_up. After the gather, `A_row[m]` is set to
`A + s_slot[m]/top_k * K` (routed) or `A + m * K` (shared) at `:1073-1078`. The k loop is
`GATEUP_M_ACCUM` at `:1092` — weight load and decode outside, `ROWS_` FMAs inside.
Accumulators are `float acc[ROWS_][VEC]` in registers: at `ROWS_=6, VEC=2` that is 12 f32
= 12 registers, plus `w_lo[2]`/`w_hi[2]`/`sc[2]` = 6, comfortably within budget.

- **grid**: `[2048/(32*2), 6*6+1, 2*4]` = **`[32, 37, 8]`** at gamma=6
  (`fp8_moe.rs:404`).
- **block**: `[32,1,1]`. **smem**: none dynamic; `s_slot[6] + s_m` static.
- **Weights read: ONCE per distinct expert.**

Bytes moved, at the measured union of **19.5 distinct experts** of 36 slots (plus 1
shared = 20.5 expert weight streams):

| M | slots | distinct (measured) | weight bytes | partials | total | roofline @273 GB/s |
|---|---|---|---|---|---|---|
| 1 | 6 | 7 | 66.1 MB | 0.46 MB | **66.6 MB** | **244 us** |
| 2 | 12 | ~9.6 | 90.7 MB | 0.79 MB | **91.5 MB** | **335 us** |
| 6 | 36 | ~20.5 | 193.6 MB | 1.68 MB | **195.3 MB** | **715 us** |

The M=6 / M=1 ratio at roofline is **2.93x**, not 6x — that is the dedup working. What
the caller observes as ~6x is the *combination* of (a) a union that is 2.93x larger and
(b) the kernel running at 136 GB/s instead of 194 GB/s, which is another **1.43x**.
2.93 x 1.43 = **4.2x**. The remaining gap to 6x is the silu_down smem occupancy loss and
the routing/blend launches.

- **AI**: `2 * 2 * 2048 * 4096 * ROWS_ / 9.51 MB`. At `ROWS_=6` that is
  **21 FLOP/byte** — 6x the single-row intensity, exactly as expected from perfect weight
  reuse. Still **bandwidth-bound** (GB10's BF16 FMA roofline is far above 21), but the
  margin has narrowed 6x, which is precisely why the batched path is worth having.
- **Inefficiencies**:
  - `A_row[m]` are **raw global pointers**, dereferenced inside the k loop at
    `:1120-1121`. No smem staging, unlike `silu_down`. See Finding 3.
  - The `ROWS_` ladder (`MROW_ARM` at `:1144`, `GATEUP_M_ROWS` at `:1145`) has arms at
    1, 2, 4, 6. A 3-row group runs the 4-arm with one wasted FMA per byte; a 5-row group
    runs the 6-arm with one wasted FMA. Cheap on a 3.5 FLOP/byte kernel. Comment at
    `:1135`: *"The arms are a LADDER, not one per M... surplus rows pointed at A row 0 and
    dropped by the `m < M` guard at emit."*
  - Serial leader election (Finding 5).

## silu_down_shared_t_m_impl (device template)

`moe_shared_expert_fused_t.cu:1162`

The MROW down projection. Stages `MROW` rows of SiLU-product activations into dynamic
smem (`mrow * K * 4 / SPLIT` bytes), then `SILUDOWN_M_ACCUM` at `:1270` with the same
weight-outside/rows-inside structure.

- **grid**: `[4096/(32*2), 37, 4]` = **`[64, 37, 4]`** at gamma=6 (`fp8_moe.rs:481`).
- **block**: `[32,1,1]`.
- **dyn smem**: `mrow * k * 4 / split` = `6 * 2048 * 4 / 4 = 12288 B` at gamma=6
  (`fp8_moe.rs:507`), versus 4096 B at M=2 and 2048 B at M=1.
- **Weights read: ONCE per distinct expert.**
- **bytes/call** at the 20.5-stream union: `20.5 * 4.71 MB` weights + `20.5 * 65.5 KB`
  partials = **97.9 MB**, **359 us**.
- **AI at ROWS_=6**: `2 * 4096 * 2048 * 6 / 4.79 MB` = **21 FLOP/byte**.
- **Inefficiency — the occupancy tax.** 12288 B/block on sm_121's 100 KB/SM caps
  **8 blocks/SM = 8 warps = 256 threads** of a 1536-thread budget: **~17% occupancy**.
  There are `64 * 37 * 4 = 9472` blocks to schedule across 48 SMs, so the tail is fine,
  but 8 concurrent warps per SM is thin for hiding LPDDR5X latency on a pure streaming
  kernel. Two independent fixes:
  1. Stage as **BF16** rather than F32 — the activations came from BF16 and are immediately
     multiplied by an f32 weight, so f32 staging buys nothing numerically. Halves to
     6144 B, doubling resident blocks to 16.
  2. Raise `T_BLOCK` to 64 or 128 **for the `_m` path only**. At `T_BLOCK=128`,
     `grid.x = 4096/(128*2) = 16` and smem per *thread* drops 4x.

  Combined, these should recover a large share of the 136 -> 194 GB/s gap. See Finding 4.

## The V2 wide-load tier: `_m{2,6,8}v4s4` (`ATLAS_MOE_SPLITK_V2=1`, default OFF)

The two inefficiencies called out above — narrow weight requests and the raw-global
`A_row[m]` reads — are closed by the V2 entries (12 of them, `GATEUP_M_ENTRY_ACT` /
`DOWN_M_ENTRY` at VEC=4), selected by `splitk_m_t_v2_handles` when
`ATLAS_MOE_SPLITK_V2=1`:

- **VEC=4 at the SAME SPLIT=4.** Weight requests go 64 → 128 B/warp (SASS: the packed
  prefetch is `LD.E` 32-bit instead of `LD.E.U16`, still 16-deep unrolled). Unlike the
  single-row v4s8 tier, no SPLIT bump is needed: the `_m` grid has `M*top_k + 1` y-slots
  (37 at γ=6 vs the single row's 7), so grid.x halving does not starve 48 SMs. Same
  split points + same per-output FMA order = **bit-identical to v2s4** (GATE V2 in
  `moe_unified_t_m_microtest` byte-compares them).
- **ACT_SMEM on gate_up.** All gathered rows' bf16 activation slices for the block's
  k-window are staged into dynamic smem once (`MROW * K/SPLIT * 2` B = 12 KiB at m6),
  then the K loop's `16*M` per-scale-group activation reads are `LDS` instead of M
  dependent 4-byte warp-uniform global loads per weight byte pair. At M=6 the incumbent
  issues 96 activation LDGs against 17 weight LDGs per scale-group — that inverted
  request ratio is the 206 → 135 GB/s falloff vs the m=1 sibling.
- The down kernel already staged activations; its V2 is the VEC widening alone.

Registers 84-101, zero spills, `__launch_bounds__(256,2)` held. Every pre-existing
kernel's SASS is instruction-identical after the change (85/85 verified) — the tier is
purely additive.

## GATEUP_M_ENTRY / DOWN_M_ENTRY (12 MROW entries)

`moe_shared_expert_fused_t.cu:1332` and `:1365`, instantiated at `:1399-1411`

Twelve `extern "C"` entries: `moe_expert_gate_up_shared_t{,_e8m0}_m{1,2,6}v2s4` and
`moe_expert_silu_down_shared_t{,_e8m0}_m{1,2,6}v2s4`. `MROW in {1,2,6}` is a **template
parameter**, so `s_slot[MROW]` and the `acc[ROWS_][VEC]` register file are sized exactly.

Handle selection is at
`crates/spark-model/src/layers/moe/mod.rs:625`
(`splitk_m_t_handles`): a narrowest-first candidate list
`[(2, m2 handles), (MOE_VERIFY_MAX_ROWS=6, m6 handles)]`. A 4-row verify picks the m6
kernel and pays 2 surplus rows of FMAs — bandwidth-neutral.

- **Kill switches**: `ATLAS_MOE_SPLITK=0` and `ATLAS_MOE_SPLITK_M=0`
  (`crates/spark-model/src/layers/moe/forward_phase.rs:29,77`).
  The `_m` path additionally requires `num_tokens >= 2`.

## moe_gate_up_partial_finalize_m

`moe_shared_expert_fused_t.cu:1419`

MROW finalize. `rows = total_routed + num_tokens` — the routed slots plus one shared-expert
row per token.

- **grid**: `[div_ceil(2048,32), 36+6, 2]` = **`[64, 42, 2]`** at gamma=6.
- **bytes/call**: read `2*2048*4*4*42 = 2.75 MB`, write `2*2048*2*42 = 344 KB`.
  **3.1 MB/layer**, **11.4 us**. 43 layers: **0.49 ms/step** — 1.6% of the verify MoE
  budget. Still trivial, but 6x less trivial than at M=1.

## moe_down_partial_finalize_m

`moe_shared_expert_fused_t.cu:1446`

- **grid**: `[div_ceil(4096,32), 42, 1]` = **`[128, 42, 1]`**.
- **bytes/call**: read `4096*4*4*42 = 2.75 MB`, write `4096*2*6 = 49 KB`.
  **2.8 MB/layer**, **10.3 us**; 43 layers **0.44 ms/step**.
- Applies the gate weights and accumulates routed + shared into the final output, same as
  the M=1 variant.

---

# Part E — Legacy decode GEMV kernels (not on the DeepSeek-V4 hot path)

These predate the `_t` transposed layout and the split-K/MROW work. They are kept for
fallback dispatch and for models without transposed weights. All of them use the
**shared-memory E2M1 LUT** rather than the branch-free decode, and all of them read the
`[N, K/2]` output-major layout, so they run at roughly **127 GB/s** rather than 194.

## moe_expert_gemv

`kernels/gb10/common/moe_expert_gemv.cu:57`

Generic per-expert NVFP4 GEMV, one output element per warp, `[N, K/2]` layout.

- **grid/block**: `[div_ceil(n, warps_per_block), num_routed]` x `[256,1,1]`.
- **bytes/expert**: `N*K/2 + N*K/16 + K*2` = for the down shape `4.19 + 0.52 MB` =
  **4.71 MB**.
- **AI**: 3.5 FLOP/byte. **Bandwidth-bound at ~127 GB/s effective** because the weight
  row for output `n` is contiguous in `k` but the warp's 32 lanes split *one* row, so each
  lane reads `K/64` bytes strided by 1 — actually coalesced *within* a row, but the LUT
  decode's smem bank conflicts throttle it.
- **Inefficiency**: LUT decode. Port `e2m1_decode`.

## moe_weighted_sum

`kernels/gb10/common/moe_expert_gemv.cu:164`

`out[i] = sum_k w[k] * expert_out[k][i]`.

- **bytes**: `top_k * hidden * 2` read + `hidden * 2` write = **57 KB** at top_k=6,
  hidden=4096. ~0.2 us. Bandwidth-bound, trivial.

## moe_weighted_sum_blend

`kernels/gb10/common/moe_expert_gemv.cu:195`

Same plus the shared-expert term. **65 KB**, ~0.24 us.

## moe_expert_gemv_gate_up

`kernels/gb10/common/moe_expert_gemv_fused.cu:53`

Fused gate+up in the untransposed layout. **9.44 MB/expert**, AI 3.5 FLOP/byte,
bandwidth-bound, LUT decode.

## moe_expert_gemv_gate_up_2x

`kernels/gb10/common/moe_expert_gemv_fused.cu:163`

Two output columns per warp — the ancestor of `VEC=2`. Halves the block count for the same
work; the measurement in `fp8_moe.rs:348-356` shows this alone is not enough without
split-K to restore occupancy.

## moe_expert_gemv_silu_down

`kernels/gb10/common/moe_expert_gemv_fused.cu:303`

**4.71 MB/expert**, AI 3.5, bandwidth-bound, LUT decode.

## moe_expert_gemv_silu_down_2x

`kernels/gb10/common/moe_expert_gemv_fused.cu:417`

Two-column variant.

## moe_expert_gemv_silu_down_wide

`kernels/gb10/common/moe_expert_gemv_fused.cu:552`

Wider tile still — 4 or 8 columns per warp. Same weight traffic; the tradeoff is register
pressure against block count, and on a 25-SM machine the block count wins until VEC=2.

## moe_expert_relu2_down_shared

`kernels/gb10/common/moe_expert_relu2_down_shared.cu:53`

Nemotron-style `relu(x)^2` activation fused with the down projection (DeepSeek uses SiLU;
this is for the Nemotron family). Stages the activation in smem like `silu_down_t`.

- **bytes/expert**: **4.71 MB**. AI 3.5. Bandwidth-bound. LUT decode.

## moe_decode_atomic_c4_silu_down_accum

`kernels/gb10/common/moe_decode_atomic_c4.cu:37`

An alternative decode strategy: instead of writing per-expert outputs and reducing later,
each expert block **`atomicAdd`s its gate-weighted contribution directly into a shared
`routed_accum` f32 buffer**. Saves the `top_k * hidden` intermediate tensor.

- **bytes/expert**: `4.19 + 0.52 MB` weights + `hidden * 4` atomics = **4.71 MB**.
- **Inefficiency — atomics contention.** With `top_k=6` experts all `atomicAdd`ing into
  the same `[4096]` f32 buffer, and `hidden/32 = 128` blocks per expert, that is
  `6 * 4096 = 24576` f32 atomics per token per layer landing on 16 KB of memory. GB10's L2
  handles f32 atomics at reduced throughput; the 6-way conflict on every address
  serialises. The `_t` path's approach — write `SPLIT` partials, sum in a separate
  finalize — moves 0.9% more bytes and has **zero** atomic contention. That is why the
  `_t` path won.

## moe_decode_atomic_c4_finalize

`kernels/gb10/common/moe_decode_atomic_c4.cu:183`

Converts the f32 `routed_accum` to BF16 and adds the shared expert.
`4096 * 4` read + `4096 * 2` write = **24 KB**, ~0.09 us.

---

# Part F — The `moe_shared_expert_fused*` variant matrix

Fourteen files, ~40 kernels. They form a 3-axis product: **weight format** x **batch
width** x **layout**. Rather than repeat the byte accounting, here is the matrix and the
per-family deltas. All of them compute the same thing: fused gate+up projection, then
fused SiLU-multiply + down projection, with the shared expert handled by
`blockIdx.y == 0`.

| file | format | batch | layout | kernels |
|---|---|---|---|---|
| `moe_shared_expert_fused.cu` | NVFP4 | 1 | `[N,K/2]` | 4 |
| `moe_shared_expert_fused_t.cu` | NVFP4 + E8M0 | 1, MROW | `[K/2,N]` | 22 |
| `moe_shared_expert_fused_batch2.cu` | NVFP4 | 2, N | `[N,K/2]` | 12 |
| `moe_shared_expert_fused_batch2_t.cu` | NVFP4 | 2 | `[K/2,N]` | 2 (macro) |
| `moe_shared_expert_fused_batch3.cu` | NVFP4 | 3 | `[N,K/2]` | 3 |
| `moe_shared_expert_fused_batch3_t.cu` | NVFP4 | 3 | `[K/2,N]` | 2 |
| `moe_shared_expert_fused_bf16.cu` | BF16 | 1 | `[N,K]` | 2 |
| `moe_shared_expert_fused_bf16_batch2.cu` | BF16 | 2 | `[N,K]` | 2 |
| `moe_shared_expert_fused_fp8.cu` | FP8-E4M3 | 1 | `[N,K]` | 2 |
| `moe_shared_expert_fused_fp8_t.cu` | FP8-E4M3 | 1 | `[K,N]` | 2 |
| `moe_shared_expert_fused_fp8_batch2.cu` | FP8 | 2 | `[N,K]` | 3 |
| `moe_shared_expert_fused_fp8_batch2_t.cu` | FP8 | 2 | `[K,N]` | 2 |
| `moe_shared_expert_fused_fp8_batch3.cu` | FP8 | 3 | `[N,K]` | 3 |
| `moe_shared_expert_fused_fp8_batch3_t.cu` | FP8 | 3 | `[K,N]` | 2 |
| `moe_shared_expert_fused_w3.cu` | W3A16 | 1, N | `[N,K]` | 6 |

**Format axis — bytes per expert (gate_up + down, hidden=4096, inter=2048):**

| format | packed | scales | total | roofline/expert | 7 experts/layer | 43 layers |
|---|---|---|---|---|---|---|
| NVFP4 (4b + FP8/16) | 12.58 MB | 1.57 MB | **14.15 MB** | 51.8 us | 99.1 MB / 363 us | **15.6 ms** |
| MXFP4-E8M0 (4b + E8M0/32) | 12.58 MB | 0.79 MB | **13.37 MB** | 49.0 us | 93.6 MB / 343 us | **14.7 ms** |
| W3A16 (3b) | 9.44 MB | 1.57 MB | **11.01 MB** | 40.3 us | 77.1 MB / 282 us | **12.1 ms** |
| FP8-E4M3 (8b) | 25.17 MB | 0 | **25.17 MB** | 92.2 us | 176 MB / 645 us | **27.7 ms** |
| BF16 (16b) | 50.33 MB | 0 | **50.33 MB** | 184 us | 352 MB / 1290 us | **55.5 ms** |

This table is the whole argument for NVFP4 on this machine: BF16 experts alone would cost
55.5 ms/token, exceeding the entire 47.6 ms plain-decode budget. **MXFP4-E8M0 is 5.5%
cheaper than NVFP4** purely because the E8M0 scales are 1 byte per 32 values instead of
1 byte per 16 — worth **~0.9 ms/token** if the accuracy holds. Both variants are compiled
(`_e8m0` suffix throughout `_t`), so this is a config flip, not a code change.

**Batch axis.** `batch2`/`batch3` predate the MROW dedup. Their `_impl` bodies load the
weight byte once and FMA against 2 or 3 rows — the same reuse structure as `_m` — but
crucially they have **no leader election**: every routed slot is its own block, so an
expert selected by 2 rows is streamed **twice**. That is the per-row reload the user was
worried about, and it is real *in the `batch2`/`batch3` fallback path*, just not in `_m`.
`forward_km.rs` prefers `_m`; `batch2_t` is the fallback when the `_m` handles are absent
(measured 16.2 wall / 17.0 server tok/s versus 19.9 / 21.0 for MROW=2, commit `29c6ca6d`).

**`batchN` variants** (`moe_shared_expert_fused_batch2.cu:377, 512, 712, 1044, 1215, 1473`
and the `_v5` pair at `:1473, 1666`) take `M` as a **runtime** argument. `_v5` and
`moe_expert_down_dedup_batchN{,_v5}` (`:903, 1666`) introduce a dedup, but with a runtime
row bound — and the runtime `if (m >= M) break` inside the k loop is exactly what commit
`29c6ca6d` measured at a **fixed ~21% per-byte penalty**. That measurement is why the `_m`
family uses a compile-time `ROWS_` ladder instead. These `batchN` kernels are superseded.

**`moe_silu_precompute_batchN`** (`moe_shared_expert_fused_batch2.cu:875`) materialises the
SiLU product to global memory so `moe_expert_down_dedup_batchN` can read it. That is
`M * top_k * 2048 * 2` = 442 KB round-trip at M=6 versus the `_t` path's smem staging —
another reason `_t` won.

**Layout axis.** Every `_t` file uses the input-major layout so lane `n` reads adjacent
bytes. The non-`_t` files use output-major, which forces either one-row-per-warp (limiting
parallelism to `N` warps) or strided lane access. See `moe_transpose_u8_batched` above for
the ~10x coalescing argument.

**`moe_weighted_sum_blend_batch2 / _batch3 / _fp8_batch2 / _fp8_batch3`**
(`moe_shared_expert_fused_batch2.cu:1835`, `_batch3.cu:335`, `_fp8_batch2.cu:410`,
`_fp8_batch3.cu:408`) — the blend step for the non-`_t` batch paths. `M * (top_k+1) *
hidden * 2` read + `M * hidden * 2` write; at M=3, **196 KB**, ~0.7 us. On the `_t` path
this is folded into `moe_down_partial_finalize{,_m}`.

**`_v5` single-row variants** (`moe_shared_expert_fused.cu:380, 563`) — an earlier
optimisation generation of the untransposed single-row path (wider tiles, more registers).
Superseded by `_t`.

---

# Part G — Grouped GEMM and prefill kernels

These are the prefill / large-batch path. They use tensor cores
(`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`) and `cp.async` double buffering,
so unlike everything above they are **compute-bound at large M and bandwidth-bound at
small M**. At decode M=1..6 they would be catastrophically inefficient (a 64x64 M-tile
holding 1 useful row), which is why the GEMV family exists at all.

The crossover: a grouped GEMM tile amortises the weight read across `M_TILE` rows, so its
effective per-row weight cost is `1/M_TILE` of the GEMV's. But it must *launch* a full
M_TILE regardless. At `M_TILE=64` and M=6, 90% of the FLOPs are wasted — yet the *bytes*
are the same as the GEMV's, so it is not actually slower on bandwidth. The reason the GEMV
wins at M<=6 is the **expert union**: the grouped GEMM's ptrtable grid is
`[n_tiles, max_m_tiles, num_experts]` = up to 256 experts, and even with the worklist
compaction it launches per-expert tiles for every expert with >=1 row, with no dedup of
the weight stream across the M dimension beyond what the tile already gives.

## moe_bf16_grouped_gemm

`kernels/gb10/common/moe_bf16_grouped_gemm.cu:86`

BF16 x BF16 grouped GEMM over expert-sorted rows with an `expert_offsets` prefix-sum
table.

- **grid**: `[n_tiles, max_m_tiles, num_experts]`, block 128.
- **bytes**: `num_experts * N * K * 2` weights. **50.3 MB/expert**.
- **AI at M_TILE=64**: `2 * 64 * N * K / (N*K*2)` = **64 FLOP/byte**. Compute-bound at
  full tiles.

## moe_fp8_grouped_gemm

`kernels/gb10/common/moe_fp8_grouped_gemm.cu:281`

FP8-E4M3 weights, BF16 activations, `__launch_bounds__`-annotated. **25.2 MB/expert**,
**128 FLOP/byte** at M_TILE=64.

## moe_w3a16_grouped_gemm / _ptrtable

`kernels/gb10/common/moe_w3a16_grouped_gemm.cu:55, 222`

3-bit weights. The `_ptrtable` variant takes an array of per-expert base pointers rather
than assuming a contiguous `[num_experts, ...]` tensor, which lets experts live in separate
allocations (needed for partial offload). **9.44 MB/expert**, the cheapest format here.

The 3-bit unpack is the awkward part: 3 bits does not divide a byte, so each 32-bit word
holds 10 values with 2 bits wasted, or values straddle word boundaries. Either way the
decode is more ALU-heavy than E2M1's nibble split.

## moe_w4a16_grouped_gemm / _ptrtable / _ptrtable_t

`kernels/gb10/common/moe_w4a16_grouped_gemm.cu:37, 231, 419`

The common-directory NVFP4 grouped GEMM. `_t` takes the transposed weight layout.
**14.15 MB/expert**.

## moe_w8a8_grouped_gemm

`kernels/gb10/common/moe_w8a8_grouped_gemm.cu:143`

INT8 x INT8 with `mma.sync...s32.s8.s8.s32`. **25.2 MB/expert**, integer tensor cores.

## moe_expert_gate_up_shared_prefill

`kernels/gb10/common/moe_prefill.cu:59`

Prefill gate_up: one block per `(token_tile, expert)` pair, tiled over N. Reads the weight
once per tile and amortises across the tile's tokens.

- **bytes**: `9.44 MB/expert` weights + `M_TILE * 4096 * 2` activations.
- **AI at M_TILE=32**: **32 FLOP/byte**. Compute-bound.
- LUT-based E2M1 decode.

## moe_expert_silu_down_shared_prefill

`kernels/gb10/common/moe_prefill.cu:211`

**4.71 MB/expert**. Same structure.

## moe_weighted_sum_blend_prefill

`kernels/gb10/common/moe_prefill.cu:360`

`num_tokens * (top_k+1) * hidden * 2` read. At 4096 tokens: **235 MB**, **0.86 ms**.
Bandwidth-bound, AI 0.44.

## moe_sorted_gate_up

`kernels/gb10/common/moe_sorted_prefill.cu:54`

Gate_up over expert-**sorted** rows, so all rows for an expert are contiguous and the
weight is read exactly once per expert per N-tile. This is the prefill analogue of the
MROW dedup, and it is the structurally correct answer at large M.

- **bytes**: `9.44 MB/expert x num_experts_touched`. At prefill with 4096 tokens all 256
  experts are touched: **2.42 GB/layer**, **8.9 ms/layer** — but amortised over 4096
  tokens that is **2.2 us/token**, versus the decode path's 244 us/token. That ratio, ~110x,
  is the entire reason decode is memory-bound and prefill is not.
- LUT-based E2M1 decode — worth porting, ~1.5x on prefill.

## moe_sorted_silu_down

`kernels/gb10/common/moe_sorted_prefill.cu:179`

**4.71 MB/expert x 256** = 1.21 GB/layer, 4.4 ms/layer.

## nemotron_moe_topk_sigmoid_batched

`kernels/gb10/common/nemotron_moe_prefill.cu:53`

Nemotron routing: sigmoid gate, batched. `M * 2096 B`. Latency-bound.

## nemotron_moe_up_prefill

`kernels/gb10/common/nemotron_moe_prefill.cu:163`

Nemotron has no gate projection (relu^2 replaces SiLU-mul), so this is a single
`[inter, hidden]` up projection rather than a fused `[2*inter, hidden]`.
**Half the gate_up traffic: 4.72 MB/expert.**

## nemotron_moe_relu2_down_prefill

`kernels/gb10/common/nemotron_moe_prefill.cu:299`

`relu(x)^2` fused with down. **4.71 MB/expert**.

## nemotron_moe_weighted_sum_prefill

`kernels/gb10/common/nemotron_moe_prefill.cu:446`

Same shape as `moe_weighted_sum_blend_prefill`. **235 MB** at 4096 tokens.

---

# Part H — `kernels/gb10/deepseek-v4-flash/nvfp4/moe_w4a16_grouped_gemm.cu`

The model-specific NVFP4 grouped GEMM, 66 KB of source, **11 `__global__` kernels**. This
is a specialised copy of the common `moe_w4a16_grouped_gemm.cu` with DeepSeek-V4-Flash's
exact shapes baked in and `cp.async` pipelining added.

Tile constants:

| constant | value |
|---|---|
| `M_TILE` | 64 |
| `N_TILE_SM` | 64 |
| `N_TILE_LG` | 128 |
| `K_STEP` | 16 |
| `K_STEP_T` | 32 |
| `K_STEP_T64` | 64 |
| `PAD_T64` | 8 |

## moe_w4a16_grouped_gemm_ptrtable / _e8m0

`kernels/gb10/deepseek-v4-flash/nvfp4/moe_w4a16_grouped_gemm.cu:188, 204`

Baseline: `M_TILE=64, N_TILE_SM=64, K_STEP=16`, no `cp.async`. Untransposed weights.

- **grid**: `[ptrtable_legacy_grid_x(n_out), max_m_tiles, num_experts]`, block 128 —
  `crates/spark-model/src/layers/ops/moe_grouped_a.rs:153`.
- **bytes/expert**: `N*K/2 + N*K/16`. For gate_up `[4096, 4096]`: **9.44 MB**.
- **AI at full tile**: `2 * 64 * 64 * K / (64*K/2 + 64*K/16)` = **~29 FLOP/byte**.
- **`_e8m0`**: identical but E8M0 scales, `9.44 - 0.26 = ` **8.92 MB/expert**, 5.5% less.

## moe_w4a16_grouped_gemm_t / _e8m0

`:469, 485`

Transposed weights, `N_TILE_LG=128`, `K_STEP_T=32`, **2-stage `cp.async` double
buffering**. The larger N tile halves the number of A-tile re-reads; `cp.async` overlaps
the global-to-smem copy of stage `k+1` with the MMA of stage `k`.

- **grid**: `[div_ceil(n_out, 128), max_m_tiles, num_experts]`, block 128 —
  `moe_grouped_a.rs:189`.
- **smem**: 2 stages x (A tile `64*32*2` + B tile `128*32/2` + scales) ≈ 2 x 6.1 KB =
  **12.2 KB**.
- **AI**: `2 * 64 * 128 * K / (128*K/2 + 128*K/16)` = **~29 FLOP/byte**, same ratio, but
  the A-tile traffic drops 2x and latency is hidden.

## moe_w4a16_grouped_gemm_t_k64 / _e8m0

`:734, 750`

`K_STEP_T64=64` with `PAD_T64=8` on the smem leading dimension to break bank conflicts on
the 4-bit unpack. Deeper K step means fewer `cp.async` commit/wait boundaries per output
tile.

- **grid**: `[div_ceil(n_out,128), max_m_tiles, num_experts]` — `moe_grouped_a.rs:277`.
- **smem**: 2 x (A `64*64*2` + B `(128+8)*64/2` + scales) ≈ 2 x 12.5 KB = **25 KB**.
  On 100 KB/SM that is 4 blocks/SM = 512 threads; adequate for a compute-bound kernel.
- The `PAD_T64=8` is doing real work: without it, a `[*, 64]` u8 smem tile has all 32 lanes
  of a warp hitting the same bank on a column read.

## moe_w4a16_fused_gate_up_t_k64 / _e8m0

`:992, 1013`

Fuses the gate and up projections into one kernel over the `[2*inter, hidden]` fused
weight, so the A tile is loaded **once** for both halves instead of twice.

- **Saving**: the A tile at `M_TILE=64, K=4096` is `64*4096*2 = 512 KB` per expert per
  k-sweep. Fusing halves the A traffic. Against a 9.44 MB weight stream that is a **5%**
  reduction — modest, but free.
- Emits the SiLU product directly, so `moe_silu_mul` is not launched.

## moe_w4a16_fused_gate_up_t / _e8m0

`:1257, 1278`

Same fusion at `K_STEP_T=32` rather than 64 — lower smem (12.2 KB), higher occupancy,
for shapes where the k-loop is short enough that the extra pipeline stages do not pay.

## moe_fp8_grouped_gemm_ptrtable_t

`:1307`

FP8 fallback in the same file, transposed layout. **25.2 MB/expert** — 2.7x the NVFP4
traffic. Present for the FP8 KV/weight configuration; not used with NVFP4 experts.

---

# Appendix — where the decode-step time actually goes

Combining the numbers above for a gamma=6 verify step (43 layers):

| component | bytes/layer | roofline | measured share |
|---|---|---|---|
| `gate_up_shared_t_m` (20.5-stream union) | 97.4 MB | 357 us | dominant |
| `silu_down_shared_t_m` (20.5-stream union) | 97.9 MB | 359 us | dominant |
| `moe_gate_up_partial_finalize_m` | 3.1 MB | 11.4 us | 1.6% |
| `moe_down_partial_finalize_m` | 2.8 MB | 10.3 us | 1.4% |
| routing (`moe_hash_route` x6) | 6.4 KB | ~0 | launch-bound |
| **MoE total** | **201 MB** | **738 us** | |
| measured (commit `2a957f1c`) | | **3260 us** | |

The gap between the 738 us roofline and the 3260 us measured is **4.4x**, and it decomposes
as roughly:

- **1.43x** from running at 136 GB/s instead of 194 GB/s (Findings 3, 4, 5).
- **1.41x** from 194 GB/s being 71% of the 273 GB/s peak (LPDDR5X efficiency floor —
  hard to move).
- **~2.2x** unaccounted: kernel launch overhead across `43 * 4 = 172` MoE kernels per
  step, the routing launches, the blend, and the tail effect of scheduling 9472 32-thread
  blocks across 48 SMs.

That last bucket is the biggest single unexplained term and is worth a Nsight Compute pass
before any further kernel work. The 32-thread block size (`fp8_moe.rs:295`) is the prime
suspect: at 9472 blocks x 37 grid.y the scheduler is dispatching an enormous number of
tiny blocks, and each one pays the leader-election serial prologue.
