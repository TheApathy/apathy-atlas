# Decode Occupancy Census — M=1 Plain Decode

Every CUDA kernel launch on the DeepSeek-V4-Flash M=1 (single-token, non-speculative)
decode path, with its grid at the real decode shapes, its resident-CTA capacity from
measured ptxas output, and a verdict on whether the under-fill is fixable.

This census was opened because the megakernel feasibility study
([`MEGAKERNEL-FEASIBILITY-2026-08-12.md`](MEGAKERNEL-FEASIBILITY-2026-08-12.md))
found `hc_post` launching **1 CTA on a 48-SM machine** at the main decode site
while the never-taken sibling branch 37 lines above launched 16 — with the
correct shard count already computed in scope and simply not passed. That fix
landed as `4f9dd9bf`. The question this document answers is: **how many more are
there?**

Answer: **the class is real but now exhausted at the launcher.** Of 34 launches
per layer, 26 run under 192 CTAs and 21 run under 48 — but almost all of that is
the machine running out of *work*, not the launcher choosing a bad grid. Four
more launcher-only bit-identical fixes were found and applied (§2.2), plus five
off-M=1-path call sites still carrying the original `hc_post` bug (§2.1).

One clean instance of the original bug survives and is **not** fixed here
because it needs a kernel edit rather than a launch change: `hc_mean` (§3.4) is
a pure per-element map that, like `hc_expand`, has no `blockIdx.y` term at all —
the same asymmetry inside the same file that `hc_post` had. Everything else
needs a kernel edit too, and the two that are actually worth money —
`hc_pre_mix` (§3.1) and the `rms_norm` family (§3.5) — reassociate reductions,
so they are oracle-gated.

**Device**: NVIDIA GB10, sm_121, **48 SMs**, 1536 threads/SM, 102400 B smem/SM,
65536 32-bit registers/SM. (`cudaGetDeviceProperties`; mirrored in
`crates/atlas-core/src/device.rs:16` as `NUM_SMS = 48`.)

**Shapes**: `num_tokens = 1`, `hidden = 4096`, 43 layers, `hc_mult = 4`,
`mix_hc = (2+hc)*hc = 24`, `top_k = 6`, **144** routed experts
(`moe_intermediate = 2048`; note `MODEL.toml`'s `num_experts = 256` is stale —
the REAP checkpoint's `config.json` says 144), `vocab = 129280`,
MLA `q_lora = 1024`, `kv_lora = 512`, `qk_nope = 448`, `qk_rope = 64`,
`nq = 64`, `nkv = 1`, `hd = 512`, `q_dim = 32768`, `mla_cache_dim = 576`,
`o_lora = 1024`, `o_groups = 8`.

**Config in force**: `ATLAS_HC_SPLIT` split path, `ATLAS_V4_DECODE_FUSED` **off**,
FP8 KV, `ATLAS_UNIFIED_MOE_LAYOUT=1`, `ATLAS_V4_ATTN_NVFP4=1` (every in-tree
serve script sets it), MoE `_t` split-K `(split=4, vec=2)`, `t_block() = 64`.

---

## 0. The distinction this census is built on

Three completely different things look identical in a profile — "grid smaller
than 48":

- **GEOMETRY-LIMITED** — independent work exists and the launch simply does not
  ask for it. The grid is a bad choice. Fixable, often in one line, often
  bit-identically. *This is the bug class.* `hc_post` was the archetype.
- **WORK-LIMITED** — the kernel's entire output at M=1 is smaller than one CTA's
  worth of lanes. A rope extract over 64 elements cannot fill 48 SMs however it
  is launched. **No launch change helps.** Only fusion removes the cost, and the
  cost is launch + DRAM latency, not bandwidth.
- **REDUCTION-BLOCKED** — plenty of work along the axis you would split, but the
  kernel closes that axis with a block-wide `__syncthreads` / warp-shuffle /
  shared-memory reduction. Splitting it needs partials + a finalize pass, which
  **reassociates the sum** and is therefore not bit-identical. Kernel work,
  oracle-gated.

A fourth shows up repeatedly in the mHC family and deserves its own name because
it *looks* like the fixable case and is not:

- **BLOCK-SIZE-LOCKED** — a pure per-element map with more elements than one
  CTA's lanes, so more CTAs are legal and bit-identical, but the grid-stride
  uses a *compile-time* block constant (`HC_BLOCK`) instead of `blockDim.x`. The
  launcher cannot shrink the block to buy CTAs without silently creating gaps in
  the index space. Fixable and bit-identical, but it is a kernel edit
  (`HC_BLOCK` → `blockDim.x`), so per this task's scope it is documented below
  rather than written.

`resident CTAs` is the cooperative-residency capacity, computed from measured
ptxas output by the formula in
[`bench/megakernel-feasibility/occupancy_census.sh`](../bench/megakernel-feasibility/occupancy_census.sh):

```
regs_per_cta  = (threads/32) * ceil(regs*32/256) * 256
blocks_per_SM = min( 1536/threads, 65536/regs_per_cta, 102400/smem_per_cta )
resident CTAs = 48 * blocks_per_SM
```

"SM reach" is `min(grid,48)/48` — can the grid even touch every SM at one block
per SM? "Sat" is `grid/resident` — does it saturate the SMs it does touch?
**A launch under 48 CTAs cannot fill the machine at all.**

---

## 1. Census — one decode layer

34 kernel launches + 1 D2D copy = **35 graph nodes per layer**, ×43 =
**1462 nodes/token**, plus ~6 at step level. (This independently reproduces the
count in `MEGAKERNEL-FEASIBILITY-2026-08-12.md` §3, arrived at by three separate
sweeps: 20 attention, 7 MoE, 6 mHC, 1 extra `rms_norm`.)

**14,832 CTAs per layer**, of which 11,640 (78%) are four GEMVs.

### 1a. mHC (`decode_inner.rs`, `ops/hyper_connection.rs`, `hyper_connection.cu`)

| kernel | grid @ M=1 | blk | CTAs | regs | smem | blk/SM | resident | SM reach | sat | ×/layer | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `hc_pre_mix` | (1,25,1) | 512 | **25** | 40 | 2048 | 3 | 144 | 52% | 17% | 2 | **REDUCTION-BLOCKED** — biggest remaining prize, §3.1 |
| `hc_pre_finish` | (1,16,1) | 256 | **16** | 40 | 16 | 6 | 288 | 33% | 6% | 2 | BLOCK-SIZE-LOCKED, §3.3 |
| `hc_post` | (1,16,1) | 256 | **16** | 40 | 0 | 6 | 288 | 33% | 6% | 2 | was **1 CTA** → fixed `4f9dd9bf`; residual §3.3 |

Wrappers: `ops/hyper_connection.rs:155` (`hc_pre_split`), `:241`
(`hc_post_sharded`). Call sites: `decode_inner.rs:528`, `:698`, `:803`.
Shard count `post_shards` computed at `decode_inner.rs:521`.

Step-level (once per token, not per layer):

| kernel | grid | blk | CTAs | regs | smem | resident | ×/token | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `hc_expand` | (1,1,1) | 256 | **1** | 40 | 0 | 288 | 1 (layer 0) | BLOCK-SIZE-LOCKED — kernel has **no `gridDim.y` at all** (`hyper_connection.cu:48`: `for (d = tid; d < H; d += HC_BLOCK)`), §3.4 |
| `hc_head` | (1,1,1) | 256 | **1** | 48 | 1044 | 240 | 1 (layer 42) | REDUCTION-BLOCKED — streams ~576 KiB on ONE SM, §3.2 |

### 1b. MLA attention (`decode/attention_forward_v4.rs`, `mla_absorbed.cu`, `mla_paged_decode_fp8.cu`)

| # | kernel | grid @ M=1 | blk | CTAs | verdict |
|---:|---|---|---:|---:|---|
| 1 | `rms_norm_vanilla` (input_norm) | (1,1,1) | 1024 | **1** | REDUCTION-BLOCKED, §3.5 |
| 2 | `w8a16_gemv` (wq_a) | (256,1,1) | 256 | 256 | ok |
| 3 | `rms_norm_vanilla` (q_a_norm) | (1,1,1) | 1024 | **1** | WORK-LIMITED (1024 elems) + reduction-blocked |
| 4 | `w4a16_gemv` (wq_b) | (8192,1,1) | 256 | 8192 | ok |
| 5 | `rms_norm` (q_b_norm) | (64,1,1) | 512 | **64** | WORK-LIMITED — 64 heads is the structural ceiling |
| 6 | `w8a16_gemv` (wkv) | (128,1,1) | 256 | **128** | `N_PER_BLOCK` compile-time 4, §3.6 |
| 7 | `rms_norm_vanilla` (kv_a_norm) | (1,1,1) | 512 | **1** | WORK-LIMITED (512 elems = 1 KiB) |
| — | K→V D2D 1024 B | — | — | — | not a kernel |
| 8 | `mla_q_rope_extract_batched` | (16,1,1)→**(64,1,1)** | 256→**64** | 16→**64** | **FIXED §2.2** |
| 9 | `mla_q_rope_extract_batched` (K) | (1,1,1) | 64 | **1** | WORK-LIMITED — total = 1·1·64 = 64 elements |
| 10 | `rope_forward_yarn_interleaved` | (65,1,1) | 128 | **65** | WORK-LIMITED at seq_len=1; **do not shrink the block**, §3.7 |
| 11 | `mla_q_rope_writeback_batched` | (16,1,1)→**(64,1,1)** | 256→**64** | 16→**64** | **FIXED §2.2** |
| 12 | `mla_q_rope_writeback_batched` (K) | (1,1,1) | 64 | **1** | WORK-LIMITED |
| 13 | `mla_cache_assemble_batched` | (1,1,1) | 576 | **1** | WORK-LIMITED **and** `blockIdx.y` never read |
| 14 | `reshape_and_cache_flash_fp8` | (1,1,1) | 256 | **1** | WORK-LIMITED (1152 elems) |
| 15 | `mla_paged_decode_fp8_kvalias` | (64,1,1) | 256 | **64** | **REDUCTION-BLOCKED — the biggest structural prize**, §3.8 |
| 16 | `mla_q_rope_extract_batched` (derot) | (16,1,1)→**(64,1,1)** | 256→**64** | 16→**64** | **FIXED §2.2** |
| 17 | `rope_forward_yarn_interleaved_inv` | (64,1,1) | 128 | **64** | WORK-LIMITED |
| 18 | `mla_q_rope_writeback_batched` (derot) | (16,1,1)→**(64,1,1)** | 256→**64** | 16→**64** | **FIXED §2.2** |
| 19 | `w4a16_gemv_grouped` (wo_a) | (2048,1,1) | 256 | 2048 | ok |
| 20 | `w4a16_gemv` (wo_b) | (1024,1,1) | 256 | 1024 | ok |

`mla_paged_decode_fp8` ptxas: 128 regs, 16448 B smem → **2 blocks/SM, resident
96**. Its 64-CTA grid is a 1.33-wave launch.

Plus one more `rms_norm` per layer at `decode_inner.rs:760` (`post_attn_norm`,
grid (1,1,1), block 1024) — same verdict as #1.

### 1c. MoE (`moe/forward.rs`, `moe/forward_phase.rs`, `moe_shared_expert_fused_t.cu`)

Live path: `MoeLayer::forward_residual` (`moe/forward.rs:49`) →
`use_t_layout_for_decode()` → `dispatch_unified_t_decode`
(`moe/forward_phase.rs:369`) → `unified_t_split_k` → `(split=4, vec=2)` →
the `_e8m0_v2s4` split-K pair. The exl3 / bf16 / fp8 / w3 branches are dead on
V4 (the loader never populates those pointer tables).

| # | kernel | grid @ M=1 | blk | CTAs | ×/token | verdict |
|---:|---|---|---:|---:|---:|---|
| 1 | `dense_gemv_bf16` (gate logits) | (36,1,1) | 256 | **36** | ×43 | REDUCTION-BLOCKED, §3.9 |
| 2a | `moe_hash_route` | (1,1,1) | 256 | **1** | ×3 | WORK-LIMITED — `if (threadIdx.x != 0) return;` |
| 2b | `moe_topk_sqrtsoftplus` | (1,1,1) | 256 | **1** | ×40 | WORK-LIMITED — iterative top-6, all state `__shared__` |
| 3 | `moe_expert_gate_up_shared_t_e8m0_v2s4` | (16,7,8) | 64 | 896 | ×43 | ok |
| 4 | `moe_gate_up_partial_finalize` | (32,7,2) | 64 | 448 | ×43 | ok |
| 5 | `moe_expert_silu_down_shared_t_e8m0_v2s4` | (32,7,4) | 64 | 896 | ×43 | ok |
| 6 | `moe_down_partial_finalize` | (64,7,1) | 64 | 448 | ×43 | ok |
| 7 | `moe_weighted_sum_blend` | (16,1,1) | 256 | **16** | ×43 | WORK-LIMITED + block-locked, §3.10 |

### 1d. Step level (outside the 43 layers)

**6 launches + 1 D2D per token, 32,326 CTAs — of which 32,320 are the LM head.**
The entire rest of the step level is 6 CTAs moving ~250 KB.

| kernel | grid @ M=1 | blk | CTAs | ×/token | verdict |
|---|---|---:|---:|---:|---|
| token embedding | — | — | **0** | 1 | **not a kernel** — `cuMemcpyDtoDAsync_v2` of 8192 B (`gpu_impl.rs:295` ← `impl_a3.rs:35`) |
| `hc_mean` | (1,1,1) | 256 | **1** | **3** | **BLOCK-SIZE-LOCKED + no `gridDim.y` at all** — the `hc_post` bug, unfixed. §3.4 |
| `hc_head` | (1,1,1) | 256 | **1** | 1 | REDUCTION-BLOCKED, §3.2 |
| `rms_norm_vanilla` (final norm) | (1,1,1) | 1024 | **1** | 1 | REDUCTION-BLOCKED, §3.5 |
| `dense_gemv_fp8w` (lm_head) | (32320,1,1) | 256 | 32,320 | 1 | ok — streams 529 MB, saturating |
| `argmax_bf16` | (1,1,1) | 1024 | **1** | 0 or 1 | REDUCTION-BLOCKED — **and not launched by default**, §3.13 |

`hc_mean` fires 3×/token (DSpark capture layers 40/41/42) and **is live on plain
decode**: `scripts/dspark-serve.sh:29` sets `ATLAS_DSPARK_CAPTURE=1`
unconditionally, including `plain` mode.

The serve head is FP8, not NVFP4 (`scripts/dspark-serve.sh:46`
`--lm-head-dtype fp8`).

Two launches on this path are **dead code** — the handle is `KernelHandle(0)`
and the launch never fires: `bf16_scale_inplace` (`impl_a3.rs:62`;
`config.embed_scale` defaults 0.0, only the Gemma-4 parser sets it) and the
logit softcap (`impl_a3.rs:431`; `final_logit_softcapping` defaults 0.0,
Gemma-only).

---

## 2. Fixes applied

All four are **bit-identical by construction**. The argument in every case is
the same shape and is recorded in a comment at the change site, as the
`hc_post` fix did.

### 2.1 `hc_post` — the original bug, five more live call sites

`4f9dd9bf` fixed the main M=1 site. Sweeping every caller in the tree found
**five more** call sites still on the unsharded `ops::hc_post` (grid.y = 1):

| site | path | grid before | grid after |
|---|---|---|---|
| `multi_seq/mod.rs:420` | γ-verify, attention site | (n,1,1) | (n,16,1) |
| `multi_seq/mod.rs:555` | γ-verify, per-row FFN (K2 arm) | (1,1,1) | (1,16,1) |
| `multi_seq/mod.rs:626` | γ-verify, per-row FFN (wide arm) | (1,1,1) | (1,16,1) |
| `dspark_head.rs:971`, `:1025` | DSpark drafter, both HC sites × 5 stages | (b,1,1) | (b,16,1) |

**These are not on the M=1 plain path** — they are the speculative verify and
the drafter. They are reported and fixed here because they are literally the
same bug with the same one-line fix, and `docs/kernels/04-elementwise-norm-cache.md:19`
had already flagged the `multi_seq` trio without them being fixed. At n=6 the
verify sites put 6 rows on 6 of 48 SMs; the per-row loop sites put **one row on
one SM**, n times per layer.

**Bit-identity argument** (identical for all): `hc_post` is a grid-stride loop
over the hidden dim —

```c
// hyper_connection.cu:637
for (unsigned int d = blockIdx.y * HC_BLOCK + tid; d < H; d += HC_BLOCK * gridDim.y) {
```

— with every `d` independent: no `__syncthreads`, no atomics, no shared-memory
reduction, and `out[j*H + d]` written only by the lane that owns `d`. Extra
blocks therefore only partition the same lanes; each output element is computed
from the same inputs in the same order. The grid is constant at capture time
(H is fixed), so it is CUDA-graph-safe.

### 2.2 Rope extract / writeback — 4 launches/layer, 16 → 64 CTAs

`ops/prefill_attn_a.rs`, new helper `rope_copy_launch_dims`, used by
`mla_q_rope_extract_batched` and `mla_q_rope_writeback_batched`. At decode these
fire four times per layer (Q extract, Q writeback, derot extract, derot
writeback) = **172 launches/token**.

`total = nq*rope = 4096`. At the hardcoded 256 threads that is
`div_ceil(4096,256) = 16` CTAs — 16 of 48 SMs. Raising the grid alone does
nothing (surplus CTAs fall straight out of the loop); the block has to shrink in
the same edit. The helper keeps 256 threads whenever that already reaches 48
CTAs — so **prefill is bit-for-bit and geometry-for-geometry unchanged** — and
drops to 64 threads otherwise, giving 64 CTAs at the same one-element-per-thread
density and the same coalescing (64 contiguous BF16 = 128 B).

**Bit-identity argument**: both kernels are pure element copies over one flat
index, fully parameterised on `blockDim.x` and `gridDim.x` —

```c
// mla_absorbed.cu:199 (extract) and :222 (writeback)
for (unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total;
     idx += gridDim.x * blockDim.x) {
    ...
    q_rope_out[t*nq*rope + head*rope + r] = q_full[t*q_dim + head*hd + nope + r];
}
```

The body is `dst[f(idx)] = src[g(idx)]`. There is **no arithmetic at all** — no
reduction, no shared memory, no `__syncthreads`, no atomics — so any
`(grid, block)` partition of `total` writes every destination element exactly
once with a byte-identical copied value. This is the strongest form of
bit-identity available: the output does not depend on the partition even in
principle.

**Honest expected value: small, possibly unmeasurable.** Each launch moves
16 KB (0.07 µs of bytes at 229 GB/s) and is launch/latency-bound, not
occupancy-bound. Going 16 → 64 CTAs does not reduce launch overhead. Estimated
**0.05–0.10 ms/token**, and it could be 0. It is applied because it is free and
provably safe, not because it is expected to move tok/s.

### 2.3 The "25 SMs" comments — corrected

`hyper_connection.cu:416,439,472` asserted the box has 25 SMs. It has 48
(`atlas-core/src/device.rs:16`). The coincidence that `mix_hc + 1 == 25` equals
the *wrongly remembered* SM count is **the documented reason `hc_pre_mix` was
never split** — the comment at `:472` literally justifies the grid as
SM-filling. Comments corrected and the real occupancy recorded inline. No code
change; ptxas output verified identical (40 regs, 2048 B smem, 1 barrier).

Grep confirms no *live* 25-SM arithmetic anywhere: the only hits are comments
(`moe_w4a16_grouped_gemm.cu:954`, `moe_shared_expert_fused_t.cu:127`, and the
three above). Every launcher that consults the SM count uses
`atlas_core::device::sm121::NUM_SMS = 48`.

---

## 3. Documented, not written — kernel changes with estimates

Ordered by expected value.

### 3.1 `hc_pre_mix` split-K — ~0.45–0.50 ms/token *(the known-open item)*

**Grid today**: `(1, mix_hc+1, 1)` = **25 CTAs**, 512 threads, resident capacity
144. 52% SM reach, 17% saturation. Fires **86×/token** and streams the 1.5 MiB
`hc_fn` matrix each time — 129 MiB/token, the single largest number in
`docs/kernels/04-elementwise-norm-cache.md`.

**What the grid *should* be**: `(T, mix_hc+1, S)` with `S = 2`, i.e. **50 CTAs**,
each `z` owning a contiguous half of `nvec`, writing `mix_partial[t][m][z]`,
plus a finalize pass summing the `S` partials in fixed `z` order. `S = 2` is the
right number — it is the smallest split that clears 48 SMs, and beyond that the
finalize traffic starts to matter.

**Why it is NOT a launcher change** (correcting
`MEGAKERNEL-FEASIBILITY-2026-08-12.md` §6, which called this "one launcher
line"): the kernel does not read `blockIdx.z`, so a `gridDim.z = 2` launch would
run 50 blocks that compute and write *the same 25 answers twice* — correct
output, zero speedup. Each block computes a complete 16384-term dot product and
closes it with a 512-lane shared-memory tree reduction
(`hyper_connection.cu:498-503`). Splitting `k` across blocks means partials plus
a finalize.

**Not bit-identical.** The current per-thread accumulation is strided
(`k = tid, tid+512, …`) and the tree combines leaf `t` with leaf `t+256` at the
first level. Any contiguous split-k regroups the summation. The kernel's own
comment already records that its `float4` conversion cost ~1e-7 drift and broke
decode↔prefill bit-identity — this would be a second such step. **Gate on the
quality oracle**, and note `DECODE-WATERFALL-2026-08-10.md:150` ("partial
exactness is worse than none").

### 3.2 `hc_head` mix/finish split — ~0.03 ms/token

Grid `(1,1,1)`, **1 CTA**, resident 240. Runs once per token (last layer) but
streams `head_fn [4, 16384]` FP32 = 256 KiB plus five passes over the 64 KiB
stream vector ≈ **576 KiB on a single SM**. Reduction-blocked: an RMS reduce
plus four sequential dot products, each with a `hc_block_reduce`
(`hyper_connection.cu:672-700`).

The fix is exactly the `hc_pre` → `hc_pre_mix`/`hc_pre_finish` transformation
that already exists one function above it: one block per `m` (plus one for the
RMS), then a sharded finish. Same non-bit-identity caveat as §3.1. Low value
because it fires once per token, not 86×.

### 3.3 `HC_BLOCK` → `blockDim.x` in the mHC map kernels — ~0.05–0.08 ms/token

`hc_post` (now 16 CTAs) and `hc_pre_finish` (16 CTAs) are **pure per-element
maps** over `d` — `hc_post` accumulates over `hc = 4` streams in fixed order,
`hc_pre_finish` likewise. There is no reduction over `d`, so **any** lane
repartition is bit-identical. But both hardcode the stride:

```c
d = blockIdx.y * HC_BLOCK + tid;  d < H;  d += HC_BLOCK * gridDim.y
```

with `HC_BLOCK` a compile-time `256` (`hyper_connection.cu:22`). At `H = 4096`
that caps the useful grid at `4096/256 = 16` CTAs — **33% SM reach** — and
launching with `block = 64` would leave holes at `d = 64..255`. Replacing
`HC_BLOCK` with `blockDim.x` in the two stride expressions makes
`block = 64, grid.y = 64` legal, covering all 48 SMs at the same 4096 total
lanes. One-token edit, bit-identical, but a kernel edit.

### 3.4 `hc_mean` and `hc_expand` have no `gridDim.y` at all — ~0.005 ms/token

**This is the `hc_post` bug, still live, in the same file.** Three kernels in
`hyper_connection.cu` are pure per-element maps over `d`; one of them reads
`gridDim.y` and two do not:

```c
// hc_post   :637  — FIXED, reads gridDim.y
for (unsigned int d = blockIdx.y * HC_BLOCK + tid; d < H; d += HC_BLOCK * gridDim.y)
// hc_mean   :729  — no blockIdx.y term at all
for (unsigned int d = tid; d < H; d += HC_BLOCK)
// hc_expand :48   — likewise
for (unsigned int d = tid; d < H; d += HC_BLOCK)
```

Because there is no `blockIdx.y` term, raising the grid does not shard — every
extra block redundantly recomputes and rewrites the **identical** values. So
these are *doubly* locked: the launcher can neither shrink the block (§3.3) nor
raise the grid. Both are structurally one CTA.

`hc_mean` is the one that matters: it fires **3×/token** (DSpark capture layers
40/41/42) and is live on plain decode. `hc_expand` fires once. Both are pure
maps (`hc_mean` sums 4 streams per `d` in fixed order; `hc_expand` broadcasts),
so sharding is trivially bit-identical once the stride is fixed — the same
3-token edit as §3.3, after which `post_shards` (already computed and sitting
unused in scope at `decode_inner.rs:521`) can be threaded through.

Small in bytes — 72 KiB per launch, ~0.005 ms/token total — but it is the
cleanest remaining instance of the original bug and should be fixed alongside
§3.3 in one kernel touch.

### 3.5 The `rms_norm` family — ~0.15–0.25 ms/token

Grid `(num_tokens,1,1)` = **1 CTA**, block `hidden.min(1024)`
(`ops/norm.rs:34-35`). Fires ~86×/token at `H = 4096` (input_norm and
post_attn_norm per layer), plus the smaller `q_a_norm` / `kv_a_norm` /
`q_b_norm`. `docs/kernels/04:21` prices it: 2.06 MiB/step = 7.9 µs at roofline
but **~250 µs at single-SM bandwidth**.

Reduction-blocked: `token = blockIdx.x`, warp-shuffle into
`__shared__ float warp_sums[32]`, `__syncthreads`, then rescale
(`rms_norm.cu:52-102`). Splitting the hidden axis needs a two-pass or
atomic-partials rewrite — reassociating, so oracle-gated.

**The better lever is fusion, not grid.** `docs/kernels/04:718` already proposes
folding the RMS reduction into `hc_pre_finish`, which writes the very vector
`rms_norm` reads back on the next launch and already holds the streams in
registers — "the highest-confidence structural fusion in this document". That
removes the launch entirely rather than making a bad grid less bad.

The warp-row fast path (`rms_norm_warp_row`) can never fire here:
`rms_norm_short_row_eligible` requires `hidden <= 256 && num_rows >= 1024`
(`ops/norm.rs:75-80`), and decode is `H = 4096, rows = 1`.

### 3.6 `w8a16_gemv` at N=512 (`wkv`) — 128 CTAs, low value

512 output rows exist but `N_PER_BLOCK` is a compile-time 4
(`w8a16_gemv.cu:124`), so 128 CTAs is all the launcher can ask for. Templating
`N_PER_BLOCK` would reach 512 CTAs. The K axis is closed by an in-block
reduction. `DECODE-WATERFALL-2026-08-10.md` already prices the whole small-N
GEMV category at ≤0.3 ms/token combined and marks it "nothing to swap" — this is
inside that budget.

### 3.7 `rope_forward_yarn_interleaved` — do NOT "fix" the idle threads

65 CTAs of 128 threads, but 3/4 of each block early-returns at `seq_len = 1`
(`pairs_per_pos = 32`, `pos_per_block = 4`, only pos 0 exists). The obvious
"fix" — shrink the block to 32 — is a **latent correctness bug**:
`pos_per_block = 128 / pairs_per_pos` (`rope.cu:374`) uses a **literal 128**,
not `blockDim.x`, so a 32-thread block silently drops positions at any
`seq_len > 1` while looking correct at `seq_len = 1`. Recorded so nobody
re-chases it.

### 3.8 MLA paged-decode split-K — the biggest structural prize

`mla_paged_decode_fp8_kvalias`, grid `(nq=64, 1, 1)` = **64 CTAs**, 128 regs +
16448 B smem → 2 blocks/SM → **resident 96**. A 1.33-wave launch: 48 SMs run one
block, 16 run a second, then a ragged tail.

`grep -n gridDim mla_paged_decode_fp8.cu` returns **zero hits** — both axes are
identities (`q_head = blockIdx.x` `:88`, `seq_idx = blockIdx.y` `:89`). The KV
axis, which has real work (128-token raw window plus compressed blocks per
head), is closed by a cross-warp shared-memory softmax merge
(`__shared__ float smem_o[NUM_WARPS][512]` `:449`, reduce `:464-489`). Splitting
it needs a genuine split-K MLA kernel plus a reduce pass — **a kernel-authoring
job, not a launch change.**

Note the vestigial split-K plumbing: `run_paged_decode.rs:313-318` computes
`num_splits = NUM_SMS / current_ctas` in the **Nvfp4** arm; the V4-Flash FP8 arm
never computes it and silently discards the `workspace: splitk_workspace()`
pointer passed at `attention_forward_v4.rs:570`. **It is inert** — at M=1 it
would evaluate to 1 anyway (`num_q_heads = 64 >= NUM_SMS = 48`) and no
`mla_paged_decode_splitk_fp8` kernel exists. Recorded so "split-K is on" is not
mistaken for a description of this path.

### 3.9 MoE gate `dense_gemv_bf16` — 36 CTAs, ×43/token

The only MoE launch with real unexploited work: 144×4096 BF16 = 1.18 MB moved by
**36 CTAs**, below even 1-block-per-SM. The output axis is fully consumed
(`n = blockIdx.x * N_PER_BLOCK + local_out`, `if (n >= N) return;`
`dense_gemv_bf16.cu:44-45`); the free axis is K, and K is reduced *inside* the
block (`__shfl_down_sync` `:89` then `__shared__ float smem[N_PER_BLOCK*2]`
`:93` + `__syncthreads` `:100`). The kernel never reads `gridDim`.

A grid.y k-split with f32 partials and a finalize is structurally the same thing
`moe_gate_up_partial_finalize` already does two rows below it in the same
dispatch. Reassociating ⇒ oracle-gated. **But note the prior measurement**:
[`moe-gate-lever-closed`] recorded a 6.4× kernel win here that bought **nothing**
end-to-end. Treat 3.9 as closed unless that measurement is overturned.

### 3.10 `moe_weighted_sum_blend` — 16 CTAs, work-limited

One output element per thread, no grid-stride (`j = blockIdx.x*blockDim.x + tid;
if (j >= hidden) return;` `moe_expert_gemv.cu:268-269`), so `grid.x` past
`ceil(4096/256) = 16` buys nothing and grid.y/z are unread. The block size is
also not tradeable: 256 threads / 8 warps is baked into the gate-scalar
reduction (`__shared__ float s_warp_sums[8]` `:215`, `for (w=0;w<8;w++)` `:258`).
The free axis is the `top_k` loop (`:272`), which needs atomics or a finalize.
Low value.

### 3.11 Untuned knob worth recording: MoE `T_SPLIT`

`T_SPLIT = 4` is a bare literal at `forward_phase.rs:19`, capped by
`MOE_DECODE_MAX_SPLIT = 4` at `buffers/sizes.rs:11` — and that constant's own
doc calls itself the "sizing SSOT for `moe_splitk_partials`", i.e. it is a
**scratch-buffer budget, not an occupancy calculation**. Grepping the whole MoE
dispatch for `NUM_SMS` finds nothing. The one hardware-derived constant on the
path is `t_block() = 64` (`fp8_moe.rs:292-331`), whose docstring does reason from
48 SMs and the 24-CTA/SM cap.

Latent hazard: `T_SPLIT_WIDE = 8` (`forward_phase.rs:25`, via
`ATLAS_MOE_GEMV_V2=1`) exceeds `MOE_DECODE_MAX_SPLIT` and only fits because
`MOE_DECODE_MAX_ROWS = 8` over-provisions the partial buffer. Lowering
`MAX_ROWS` would make it silently fall back. (It is not a CTA win either —
gate_up goes (8,7,16)=896 vs (16,7,8)=896.)

### 3.12 Off-census referral: FP8 calibration stalls the device

`fp8_calibration.rs:165` does `gpu.synchronize(stream)` plus a blocking
`copy_d2h` inside `observe()`, called per layer per token from
`write_kv_cache.rs:482`, and it re-fires every 128 tokens **even after freeze**
(`:145-151`). That is 43 full device stalls on every 128th token in eager decode.
Not a launch-geometry issue, so out of scope here — but it is the same family as
the "FP8 calibration suppresses CUDA graphs" finding and is worth a separate
look. Graph replay skips it.

### 3.13 `argmax_bf16` — 1 CTA over 129,280, but not on the default path

Grid is the literal `[1,1,1]`, block 1024: **one CTA scanning 258 KB from a
single SM**, 127 elements per thread. The kernel never references `blockIdx` at
all (`argmax_bf16.cu:29`: `for (i = tid; i < n; i += stride)` with
`stride = blockDim.x`), so as with §3.4 a bigger grid just rescans. The real
blocker is the shared tree reduction over `s_val`/`s_idx` (`:42-50`) spanning
the vocab axis. Needs a two-pass partial-max + reduce.

**Two things that make this lower priority than it looks:**

1. **The default serve path never launches it.** `decode_logits_step.rs:81`
   takes the GPU-argmax branch only when every active sequence has
   `temperature == 0.0`, and `MODEL.toml` gives DeepSeek-V4 `temperature = 1.0`
   in all three sampling profiles. The default path instead does a blocking
   `copy_logits_to_host` of `vocab*2 = 258,560 B/token`
   (`decode_logits_step.rs:116`) and sorts on the CPU. So the default cost here
   is a D2H plus a host sort, **not** a kernel — a different problem than this
   census is about, and probably a bigger one.
2. **Single-block is not a determinism requirement, but the geometry pins the
   tie-break.** The tree merge uses strict `>` (`:44`), so on an exact tie the
   lower `tid` wins — and thread `tid` owns indices `{tid, tid+1024, …}`, so the
   winner is the lowest `i mod 1024`, **not** the lowest index. Any change to
   `blockDim` or shard count silently changes which token wins a tie. Given this
   repo's recorded argmax-flip sensitivity, a sharded rewrite **must** make the
   tie-break explicitly lowest-index rather than inherit it from geometry.

---

## 4. Scoreboard

| | count |
|---|---:|
| launches per layer | 34 (+1 D2D = 35 graph nodes) |
| launches at step level | 6 (+1 D2D) |
| graph nodes per token | **1462 + 6 = 1468** |
| CTAs per layer / per step level | 14,832 / 32,326 |
| launches under 192 CTAs (per layer) | **26 of 34** |
| launches under 48 CTAs — cannot fill the machine at all | **21 of 34** (17 after §2.2) |
| step-level launches under 48 CTAs | **4 of 6** (all but the LM head; 6 CTAs total) |
| under-filled *and* geometry-fixable in the launcher alone | **4** (§2.2) — **all fixed** |
| under-filled, bit-identically fixable, but needs a kernel edit | 7 (§3.3 ×2, §3.4 ×2, §3.6, §3.10, §3.2) |
| under-filled and reduction-blocked (oracle-gated) | 5 (§3.1, §3.5, §3.8, §3.9, §3.13) |
| under-filled and genuinely work-limited (nothing to do) | 11 |
| additional off-M=1-path instances of the `hc_post` bug | **5 call sites** (§2.1) — **all fixed** |
| dead launches on the step path (handle 0, never fire) | 2 |

**Estimated recovery from the applied fixes: 0.05–0.10 ms/token on the M=1 path
(≈0.1–0.2%), plus an unmeasured but structurally larger win on the γ-verify and
drafter paths from §2.1** (those sites went from 1 CTA to 16, the same 16× SM
reach the original `4f9dd9bf` fix bought, at 6 sites/layer on verify and 2
sites/stage × 5 stages on the drafter).

**Estimated recovery still on the table: ~0.7–0.85 ms/token**, dominated by
`hc_pre_mix` split-K (~0.45–0.50) and the `rms_norm` fusion (~0.15–0.25) —
both oracle-gated. `mla_paged_decode` split-K is unquantified here and is a
kernel-authoring project.

**The honest headline: the launcher is now clean.** The `hc_post` find was real
and there were four more instances of it, but after this sweep there is no
remaining launch on the M=1 decode path whose grid can be improved without
touching a kernel. The decode under-fill that remains is a *kernel* problem —
which is what `DECODE-WATERFALL-2026-08-10.md` concluded from the other
direction.

---

## 5. Validation

No GPU was used to produce this document; every number is from source or from
`nvcc` on the host.

```bash
# 1. Resource usage (regs/smem) for every M=1 decode kernel — feeds the
#    blocks/SM and resident-CTA columns.
bench/megakernel-feasibility/occupancy_census.sh

# 2. The mHC comment edits are comment-only: ptxas output must be unchanged
#    (hc_pre_mix: 40 regs, 2048 B smem, 1 barrier).
nvcc -arch=sm_121a -O3 --fmad=false -DTQ_PLUS_SIGNS -cubin --resource-usage \
     kernels/gb10/deepseek-v4-flash/nvfp4/hyper_connection.cu -o /tmp/hc.cubin

# 3. No unsharded hc_post callers remain on decode or verify (prefill keeps it
#    deliberately — grid.x = T already fills the machine there).
grep -rn 'ops::hc_post(' crates/          # -> only prefill_inner.rs

# 4. No live 25-SM arithmetic anywhere.
grep -rn '25 SM' kernels/ crates/         # -> comments only, none in a grid
grep -rn 'NUM_SMS' crates/atlas-core/src/device.rs   # -> 48

# 5. Build.
cargo check -p spark-model --features cuda
```

**On-GPU validation still owed** (this task was run with no GPU):

```bash
# a. Bit-identity of the applied fixes. The rope copies cannot change bits
#    (pure copy) and hc_post sharding cannot either, but confirm end to end:
ATLAS_TARGET_MODEL=deepseek-v4-flash cargo run --release -p spark-model \
  --example decode_ab_probe --features cuda,gpu-examples
#    -> logits hash must match the pre-change plain oracle exactly.

# b. Throughput. Expect ~0.1% on plain decode (§2.2 is honestly near-zero) and
#    a larger effect on the verify path from §2.1. Compare same-binary:
scripts/dspark-serve.sh   # plain vs --dflash, LOOP_TRACE
```
