# MLA / CSA Attention Kernels — DeepSeek-V4-Flash-162B on one GB10

Target shape used throughout: **43 layers, all FullAttention**, `q_heads=64`, `kv_heads=1`,
`kv_lora_rank=512`, `qk_rope_head_dim=64`, `qk_nope_head_dim=448`, `hidden=4096`,
`q_lora_rank=1024`, `o_lora_rank=1024`, MLA cache token = 576 B FP8 (512 latent + 64 rope),
raw sliding window = **128**, CSA ratio = **4**.
Hardware: sm_121, **48 SMs** (`crates/atlas-core/src/device.rs:16`), **273 GB/s** LPDDR5X.

Roofline reference: **273 GB/s = 273 B/ns = 0.273 MB/µs**. 1 MB of traffic ≈ **3.66 µs**.
Kernel-launch floor on this box ≈ **4–6 µs** per launch (eager), so any kernel moving
< ~1.5 MB is launch/latency-bound, not bandwidth-bound.

---

## Findings — prioritised optimisation list

| # | Finding | Where | Est. saving / decode step |
|---|---|---|---|
| 1 | **`mla_paged_decode_fp8` re-reads the whole KV working set once per Q head.** grid.x = 64 heads, and every CTA loads the *same* 576-B/token K and V rows. With `kv_heads=1` there is exactly one KV row per position, so the 64 CTAs multiply DRAM demand by 64×. Two-arm working set is (128 window + `comp_block_count`) × 576 B × 2 (K and V pointers, same data) ≈ 147 KB raw + 512 B/comp-block; ×64 heads ×43 layers. At 4 k context (`comp≈1024`) that is 64×43×(147 KB + 1.05 MB) ≈ **3.3 GB/token → 12 ms** if it misses L2; measured L2 hit-rate makes it far less but it is the single largest structural amplification in decode. Fix: **head-tiled CTA** (one CTA serves 8–16 Q heads, loads each K/V row once into registers/smem, does 8–16 dots). | `kernels/gb10/deepseek-v4-flash/nvfp4/mla_paged_decode_fp8.cu:60-64`, launcher `crates/spark-model/src/layers/ops/kv_cache.rs:888` | **8–15 ms** |
| 2 | **K and V are read twice from the same bytes.** In MLA `K == V`; the kernel is even passed `k_pool_ptr` / `v_pool_ptr` which for V4 point at caches holding *identical* content (see `mla_cache_assemble_fp8.cu:87` — V's rope tail is written from `k_val`, and `mla_absorbed.cu:317` writes `rope_val` to both). The decode kernel loads the K row (`:136`) then re-loads the byte-identical V row (`:179`). Deleting the V load halves the attention DRAM traffic. | `mla_paged_decode_fp8.cu:136-152` vs `:179-191` | **2–4 ms** |
| 3 | **`M=2` / `M=6` verify re-runs the whole per-row attention loop.** `ms_mla_decode_v4_flash` batches phases A and C (projections, weights read once) but phase B is a plain `for i in 0..n` loop that calls `attention_forward_v4` per row (`multi_seq/mla.rs:498-522`), i.e. `n` separate `mla_paged_decode_fp8` launches, each re-reading the *entire* KV window + compressed pool. At γ=6 that is 6× the attention traffic (and 6×43 = 258 extra launches × ~5 µs = 1.3 ms of pure launch overhead). The kernel is already `num_seqs`-capable via `blockIdx.y` — but co-launching would still re-read KV per row unless Q is tiled in the M direction inside the CTA. Fix: add an **M-row inner loop over `q_reg[M][18]`** so the K/V row loaded once serves all M query rows. | `crates/spark-model/src/layers/qwen3_attention/trait_impl/multi_seq/mla.rs:498-522` | **γ=2: 3–5 ms; γ=6: 15–25 ms** |
| 4 | **Byte-granular FP8 loads — no vectorisation.** The K/V loads are `k_latent[i]` on `const unsigned char*`, 16 separate 1-byte loads per thread per position (`:138`, `:181`). The generic `paged_decode_attn_fp8_mla.cu` already does the right thing (`unpack4_fp8` on `uint32` loads, `:155-159`). Converting the MLA kernel to `uint4` (16 B) loads cuts the instruction count 16× and lets the memory pipe coalesce a full 512-B row per warp. | `mla_paged_decode_fp8.cu:138`, `:150`, `:181`, `:189`, `:315`, `:340` | **1.5–3 ms** |
| 5 | **`smem_o[NUM_WARPS][512]` cross-warp reduction is fully serial in one warp.** The tree reduction at `:366-387` has `warp_id < stride` do a **512-iteration scalar loop** over shared memory, with `__syncthreads()` between the 3 levels. At level 0 only 4 of 8 warps are active, level 1 only 2, level 2 only 1 — so the last level is 32 threads doing 512 sequential smem read-modify-writes ×3 levels, ~4 500 smem ops with 7/8 of the CTA idle. Fix: reduce along `lane`-owned dims (each lane already owns 16 contiguous dims — index `smem_o[w][lane*16+i]`, not `[0..512]`), making all 32 lanes participate; 16× fewer serial steps. | `mla_paged_decode_fp8.cu:366-387` (identical in `mla_paged_decode.cu:317-338`) | **0.8–1.5 ms** |
| 6 | **`hc_head` and `hc_mean` are still one-block-per-token at decode.** `hc_pre` was already split (`hc_pre_mix`/`hc_pre_finish`) precisely because a single block streaming 1.5 MiB of `hc_fn` at single-SM bandwidth costs tens of ms. `hc_head` has the same shape: `hc=4` rows × `hc*H=16384` fp32 = **256 KiB** streamed by ONE block. At the ~8 GB/s single-SM rate quoted in the source comment that is **~32 µs**; it runs once per token (final collapse) plus once per MTP/DSpark stage. | `kernels/.../hyper_connection.cu:433`, launcher `crates/spark-model/src/layers/ops/hyper_connection.rs:205` | **0.03–0.2 ms** (more if a DSpark stage runs it per stage) |
| 7 | ~~**Latent correctness bug: `paged_decode_attn_fp8` (MLA build) drops 2 of 18 dims.**~~ **FIXED.** With `HDIM 576`, `VEC_U32_FP8 = 576/(32*4) = 4` (integer division), so the FP8 loops covered `4*4 = 16` elements while `VEC_BF16 = 18` — dims 16 and 17 of every lane (64 of 576) never read for K or V, but `o_reg[16..17]` still written out. **A second defect surfaced while fixing it:** the lane slice begins at byte `lane_id * VEC_BF16 = lane_id * 18`, so the `(const unsigned int*)` casts were misaligned (18 % 4 = 2) for every odd lane — UB in 3 of 4 lanes, independent of the truncation. Fix: `FP8_U32_OK` guards the packed path on `HDIM % (WARP_SIZE*4) == 0`, and `load_lane_fp8()` falls back to byte-granular loads otherwise. Applied to both `paged_decode_attn_fp8` and `paged_decode_attn_splitk_fp8`; the BC-batched staging path is disabled (`aligned_count = 0`) when the geometry is invalid. Still off the V4 hot path — V4 routes to `mla_paged_decode_fp8` — so this is correctness insurance, not a speedup. | `paged_decode_attn_fp8_mla.cu` `FP8_U32_OK` / `load_lane_fp8` | correctness, 0 ms |
| 8 | **Compressed-arm scan is O(context/4) and unbounded.** The compressed loop (`:296-347`) walks `comp_block_count` blocks with a *scalar* per-block warp shuffle reduction — one 32-lane `__shfl_xor` tree per compressed block, and the block is only 512 B. There is no `BC`-style batching like the raw arm has. At 8 k context that is 2 048 blocks / 8 warps = 256 serialised shuffle trees per warp. Applying the same `BC=4` batching used for the raw arm amortises the reduction 4×. | `mla_paged_decode_fp8.cu:301-346` | **1–2 ms at 8 k ctx** |
| 9 | `mla_paged_decode_fp8` has **no `__launch_bounds__`**. With `float k_vals[4][16] + v_vals[4][16] + q_reg[16] + o_reg[16]` = 160 FP32 registers live in the batched path, occupancy is likely 1 block/SM (256 threads). Adding `__launch_bounds__(256, 2)` or splitting the K/V staging would let 2 CTAs/SM overlap DRAM latency. Also `--fmad=false` is set globally in `KERNEL.toml:4`, doubling FMA instruction count in every one of these kernels. | `mla_paged_decode_fp8.cu:38`; `KERNEL.toml:4` | **0.5–1.5 ms** |
| 10 | `smem_o[8][512] f32 = 16 KiB` + 2×32 B. Fine for smem capacity, but combined with the register pressure above it is the second occupancy limiter. Storing `o_reg` in BF16 in smem halves it. | `mla_paged_decode_fp8.cu:352` | occupancy only |

**Total realistically addressable at γ=2**: roughly **15–25 ms** of the ~124 ms speculative step,
concentrated in findings 1–4. Findings 1 and 3 are the same root cause seen from two angles —
*the KV row is re-read once per (Q head × verify row) instead of once*.

---

## mla_paged_decode_fp8

`kernels/gb10/deepseek-v4-flash/nvfp4/mla_paged_decode_fp8.cu:38`

**The hottest kernel in the decode path.** Fused two-arm MLA decode attention for
DeepSeek-V4-Flash. For one query token it computes, in a single online softmax:

1. the **raw arm** — the last `sliding_window = 128` positions of the paged FP8 KV cache,
   each token being a 576-byte row `[latent(512) | rope(64)]`;
2. the **compressed arm** — `comp_block_count` entries of a flat FP8 compressed-KV pool
   (`COMP_BLOCK_DIM = 512` bytes per block, latent-only, no rope);
3. a **per-head attention sink** `s_aux` folded into the softmax denominator only.

The Q vector for a head is 512 BF16 dims. The rope trick: `K_rope[0:63]` is loaded *over*
`k_vals[448:511]` (lanes 28–31) so a plain 512-dim dot yields
`dot(Q_nope, K_latent[0:447]) + dot(Q_rope, K_rope)` — the compressed arm deliberately skips
this overwrite (`:304-311`) to match the rope-free prefill oracle.

Because MLA passes the same latent as key and value, the "V" load mirrors the K rope overwrite
(`:184-191`) so the attention output carries the rope, which the dispatch de-rotates per eq. 26.

### Launch

| | |
|---|---|
| Grid | `(num_q_heads=64, num_seqs, 1)` |
| Block | `(256, 1, 1)` = 8 warps |
| Launcher | `crates/spark-model/src/layers/ops/kv_cache.rs:861-911` (`.grid([num_q_heads, num_seqs, 1]).block([256,1,1])`) |
| Call site | `crates/spark-model/src/layers/qwen3_attention/decode/run_paged_decode.rs:136-162` (`KvCacheDtype::Fp8 if is_v4_flash`) |
| `sliding_window` | hard-coded **128** at `run_paged_decode.rs:156` |
| `comp_pool` / `comp_blocks` | `mla.compressor.pool` and `self.v4_comp_pool_filled` (`run_paged_decode.rs:99-106`); `None` compressor → `NULL` + 0 → compressed arm is a no-op |
| `inv_sqrt_d` | `1/sqrt(512)` (see the comment block at `run_paged_decode.rs:89-94`) |

At `num_seqs = 1` that is **64 CTAs on 48 SMs** — 1.33 waves, so ~33 % of SMs sit idle in the
second wave. At the M=2 / M=6 verify widths the kernel is *not* co-launched with
`num_seqs = M`; the multi-seq path launches it M separate times (see finding 3).

### Dtypes / memory / registers

* **In**: `Q` BF16 `[64 × 512]` = 64 KiB; `K_cache`/`V_cache` `unsigned char` (FP8-E4M3);
  `comp_pool` `unsigned char`; `sinks` FP32 `[64]`; `block_tables`/`seq_lens` `int`.
* **Out**: `O` BF16 `[64 × 512]` = 64 KiB.
* **Shared**: `smem_m[8]` + `smem_l[8]` + `smem_o[8][512]` f32 = **16 448 B** (`:350-352`).
* **Registers**: no `__launch_bounds__`. Batched path holds `k_vals[4][16]` + `v_vals[4][16]`
  + `q_reg[16]` + `o_reg[16]` = **160 live f32** plus addressing → almost certainly
  ≥ 168 arch regs, i.e. 1 CTA/SM (occupancy 12.5 % of the 2 048-thread SM budget).
* **Unrolls**: every inner loop is `#pragma unroll`; `BC = 4` positions per iteration.
* `--fmad=false` (`KERNEL.toml:4`) prevents FMA contraction in *all* of these kernels.

### Bytes moved per call, and roofline

Per CTA (one Q head), the raw arm reads 128 positions × 576 B **twice** (K then V):

```
raw   = 128 × 576 × 2 = 147 456 B    (73 728 B if K==V dedup)
comp  = comp_block_count × 512 × 2   (K then V, same bytes)
Q/O   = 512 × 2 × 2   = 2 048 B
```

Per **layer** (64 CTAs — every CTA loads the same KV):

| context | comp blocks | ideal (KV once) | as written (×64 heads, K+V) | roofline @273 GB/s |
|---|---|---|---|---|
| 512 | 128 | 73.7 KB + 65.5 KB + 128 KB Q/O | 9.4 MB + 4.2 MB + 0.13 MB | ideal **0.98 µs** vs **50 µs** |
| 4 096 | 1 024 | 73.7 KB + 524 KB + 128 KB | 9.4 MB + 33.6 MB + 0.13 MB | ideal **2.65 µs** vs **158 µs** |
| 8 192 | 2 048 | 73.7 KB + 1.05 MB + 128 KB | 9.4 MB + 67.1 MB + 0.13 MB | ideal **4.6 µs** vs **281 µs** |

Across **43 layers** at 4 k context: ideal ≈ **0.11 ms/token**, as-written DRAM demand
≈ **1.86 GB → 6.8 ms**. In practice the 64 CTAs of a layer run concurrently and the raw-window
147 KB stays L2-resident (GB10 L2 is large enough), so the *measured* cost is between these —
but the compressed pool at 4 k+ (33 MB of L2-hostile streaming per layer) does not fit and is
paid at DRAM rate.

**Arithmetic intensity** (as written): 2 FLOP per FP8 byte for the dot, 2 more for the
accumulate → **~4 FLOP/B**. GB10 does ~O(100) FLOP/B at peak. Firmly **bandwidth-bound**, and
*additionally* occupancy-bound (1 CTA/SM) so it cannot even saturate the 273 GB/s.

### Concrete inefficiencies

1. **KV re-read per Q head (finding 1).** `blockIdx.x` = q_head; every CTA independently walks
   the identical KV rows. `kv_heads = 1`, so the amplification factor is the full 64. A
   head-tiled CTA (`grid.x = 64/HT`, each CTA holding `q_reg[HT][16]`) reduces DRAM demand by
   `HT×` with no algorithmic change — the online softmax state simply becomes `m[HT]`, `l[HT]`,
   `o_reg[HT][16]`. `HT = 4` fits registers today; `HT = 8` after finding 2 frees the `v_vals`
   array.
2. **K and V read separately from byte-identical data (finding 2).** `mla_cache_assemble_fp8.cu:87`
   writes `k_val` into *both* caches for the rope tail, and `:67-68` writes the same latent to
   both; `mla_absorbed.cu:313-318` does the same for the BF16 path. The decode kernel then reads
   `k_block` (`:136`) and `v_block` (`:179`) at the same offsets. Since `k_scale == v_scale == 1.0`
   for V4 (asserted in `cache_skip_v4.rs`), `v_vals[b][i] == k_vals[b][i]` exactly. Delete the V
   load and reuse `k_vals`. Halves attention DRAM.
3. **Scalar 1-byte loads (finding 4).** `k_latent[i]` in a 16-iteration loop compiles to 16
   `LDG.U8`. The sibling generic kernel does `const unsigned int* k32 = ...; unpack4_fp8(k32[i], ...)`
   (`paged_decode_attn_fp8_mla.cu`, now `load_lane_fp8`'s `FP8_U32_OK` branch). **Copy the idea,
   not that kernel's offsets** — there the lane offset is `lane_id * 18`, which is misaligned and
   was the bug in finding 7. *Here* the lane offset `lane_id * 16` is 16-byte aligned and the token
   stride is 576 (16-byte aligned), so `uint4` loads are legal. Alignment must be re-derived per
   kernel from `VEC_BF16`, never assumed.
4. **Serial 512-wide smem reduction (finding 5).** `:381-383` — `for (i = 0; i < 512; i++)` inside
   `if (warp_id < stride)`. Should be `for (i = 0; i < VEC_BF16; i++) smem_o[w][lane*16+i]`.
5. **Compressed arm has no `BC` batching (finding 8).** Each block costs a full 5-step warp
   shuffle tree (`:323-325`) for one 512-B row.
6. **Compressed arm loads K and V from the identical pointer** — `k_latent = c_block + kv_latent_offset`
   (`:313`) and `v_latent = c_block + kv_latent_offset` (`:337`). Literally the same address,
   read twice. Trivially removable.
7. **`__expf` on the critical path 3× per position** in the remainder loop (`:255-256`, `:281`).
   The `BC=4` path amortises this correctly; the remainder does not. With `128 % 4 == 0` the
   remainder loop only fires on ragged block boundaries — low priority.
8. **`chunk_size = ceil(win_len/8)` warp split (`:102`)**: with `win_len = 128`, each warp gets
   exactly 16 positions = 4 `BC` iterations. Balanced. But when the sequence is shorter than 8
   (early decode) most warps idle.

---

## mla_paged_decode_nvfp4

`kernels/gb10/deepseek-v4-flash/nvfp4/mla_paged_decode.cu:78`

NVFP4 sibling of the above: same CTA/warp decomposition, same online softmax, same
`smem_o[8][512]` reduction. Differences: KV is 4-bit E2M1 packed two-per-byte with an FP8-E4M3
scale per 16-element group, so a block has a `data_section` then a `scale_section`
(`block_stride_bytes`, `data_section_bytes` are both passed in). Dequant goes through a
16-entry shared LUT (`e2m1_lut`, `:107-115`) with a single 64-bit packed load per 16 elements
(`:46-62`) — properly vectorised, unlike the FP8 variant.

**Launch**: grid `(num_q_heads, num_seqs, 1)`, block `(256,1,1)` —
`crates/spark-model/src/layers/ops/kv_cache.rs:829-848`, dispatched from
`run_paged_decode.rs:58-79` under `KvCacheDtype::Nvfp4 if is_v4_flash`.

**Not the V4-Flash default** — `MODEL.toml` sets `default_kv_dtype = "fp8"`, so this path is
dormant for the target config.

**Bytes/call**: KV is 576/2 = 288 data B + 36 scale B = **324 B/token** (vs 576 FP8), so the raw
arm is 128 × 324 × 2 = 82 944 B/CTA. Same ×64-head amplification.
Roofline at 4 k ctx, 43 layers, full seq (this variant has **no sliding-window arg** — it walks
`[0, seq_len)` at `:152-156`): 64 × 43 × 4096 × 324 × 2 = **7.3 GB → 26.8 ms/token**. It is
strictly worse than the FP8 kernel *because it lacks the 128-window clamp*, not because of dtype.

**Inefficiencies**: (a) same ×64 head re-read; (b) same serial `for (i=0;i<512;i++)` smem
reduction at `:332`; (c) **no sliding window and no compressed arm** — it is the pre-CSA
version; (d) V's rope dims are *never* loaded (`:220-223` comment "only latent portion, no rope"),
which disagrees with the FP8 kernel's V-rope mirror and with `mla_cache_assemble_batched`
(`mla_absorbed.cu:313-318`) — a latent numerical divergence if this path is ever re-enabled;
(e) `if (lane_id < 16)` rope dequant (`:191`) leaves half the warp idle and writes into
`k_vals[b][0..3]`, clobbering latent dims 0–3 rather than the 448–511 tail the FP8 kernel uses —
these two kernels do **not** compute the same thing.

---

## paged_decode_attn (HDIM=576 MLA build)

`kernels/gb10/deepseek-v4-flash/nvfp4/paged_decode_attn_mla.cu:59`

Generic BF16 paged decode compiled with `#undef HDIM / #define HDIM 576` (`:5-6`) →
`VEC_BF16 = 18`, `VEC_U32 = 9`. One CTA per `(q_head, seq)`, 8 warps split the sequence, `BC = 4`
positions batched, online softmax, `smem_o[8][576]` f32 = **18 432 B**.

Used by the **absorbed-MLA decode chain** (`ms_mla_decode_one` step 6,
`crates/spark-model/src/layers/qwen3_attention/trait_impl/multi_seq/mla.rs:881-900`, via
`ops::paged_decode_attn_bf16` with `self.paged_decode_mla_k`), bound at
`crates/spark-model/src/layers/qwen3_attention/init.rs:366`. This is the **V3-style absorbed
path**, not the V4-Flash direct-KV path — V4-Flash takes the `o_lora_rank > 0` branch
(`multi_seq/mla.rs:160-162`) and never reaches here.

**Grid/block**: `(num_q_heads, num_seqs, 1)` × `(256,1,1)`.
**Bytes/call**: BF16 576-dim rows = 1 152 B/token ×2 (K,V). At 4 k ctx, per head:
9.4 MB; ×64 heads ×43 layers = 26 GB → **95 ms**. This is why V4-Flash does not use it.

**Split-K variant** `paged_decode_attn_splitk` (`:316`) — grid `(num_q_heads, num_splits, num_seqs)`,
writes `[head_dim f32, m, l]` partials to a workspace; `paged_decode_attn_reduce` (`:484`) merges
them with grid `(num_q_heads, num_seqs, 1)`, block `(32,1,1)`. Split count is derived from
`NUM_SMS / (num_q_heads * split_ref_seqs(num_seqs))` (`run_paged_decode.rs:594-599`); with
`num_q_heads = 64 > 48` this is always 1 for V4, so split-K is **dead code on this model**.

---

## paged_decode_attn_fp8 (HDIM=576 MLA build)

`kernels/gb10/deepseek-v4-flash/nvfp4/paged_decode_attn_fp8_mla.cu:66`

FP8-E4M3 generic paged decode, `#define HDIM 576` (`:5`). Structurally identical to the BF16
sibling but loads `uint32` and unpacks 4 FP8/word via `unpack4_fp8` (`:47-60`) — **this is the
vectorised loading pattern `mla_paged_decode_fp8.cu` should adopt**.

**Grid/block**: `(num_q_heads, num_seqs, 1)` × `(256,1,1)`, launcher `ops::paged_decode_attn_fp8`.
Reached only through the generic `_ =>` arm of `run_paged_decode.rs:659-689` (non-V4 models, or
V4 with `is_v4_flash == false`).

**Shared**: `smem_o[8][576]` f32 = 18 432 B. `VEC_BF16 = 18`, `VEC_U32 = 9`,
`VEC_U32_FP8 = 576/128 = **4**`.

> **CORRECTNESS BUG (finding 7).** `VEC_U32_FP8 = HDIM/(WARP_SIZE*4) = 576/128 = 4` truncates.
> All FP8 K/V loops run `i < VEC_U32_FP8 = 4` covering `q_reg[0..15]` / `o_reg[0..15]`
> (`:158`, `:168-172`, `:216-223`, `:233-237`, `:252-258`), but `VEC_BF16 = 18` — dims 16 and 17
> of each lane (= dims 512–575 of the head, i.e. exactly the **rope tail**) are never read from
> K or V. `o_reg[16]`/`o_reg[17]` are zero-initialised and then written to the output at
> `:276-278` / `:312-317`. The kernel silently drops the entire rope contribution. It is not on
> the V4-Flash hot path, but any fallback to it produces wrong attention.

**Split-K**: `paged_decode_attn_splitk_fp8` (`:327`), grid `(num_q_heads, num_splits, num_seqs)`;
`paged_decode_attn_reduce_fp8` (`:507`), grid `(num_q_heads, num_seqs, 1)` block `(32,1,1)`,
workspace row `[head_dim f32, m, l]`. Same `VEC_U32_FP8` truncation in the split-K body.

---

## paged_decode_attn (HDIM=512)

`kernels/gb10/deepseek-v4-flash/nvfp4/paged_decode_attn_512.cu:43`

BF16 paged decode at `HDIM 512` (`:6`) → `VEC_BF16 = 16`, `VEC_U32 = 8`, cleanly divisible, no
truncation bug. `smem_o[8][512]` f32 = 16 384 B. Same 8-warp / `BC=4` / online-softmax skeleton;
supports `sliding_window`.

Selected by `run_paged_decode.rs:561-565` (`KvCacheDtype::Bf16`, `head_dim > 256`) and by the
Turbo dtypes (`:241-245`, `:275-279`). Not used by V4-Flash FP8 decode.
Grid `(num_q_heads, num_seqs, 1)` × `(256,1,1)`.

Also hosts `paged_decode_attn_splitk` (`:305`) and `paged_decode_attn_reduce` (`:473`) at the
same shapes as the MLA build.

**Bytes/call**: 512 × 2 B × 2 (K,V) = 2 048 B/token. Bandwidth-bound; same ×`num_q_heads`
KV re-read structural issue as every kernel in this family.

---

## paged_decode_attn_nvfp4

`kernels/gb10/deepseek-v4-flash/nvfp4/paged_decode_attn_nvfp4.cu:87`

NVFP4 generic paged decode at `HDIM 512` (`:18`). E2M1 nibble pairs + FP8 group scales
(group = 16), 16-entry `e2m1_lut` in shared memory (`:116`), 64-bit packed loads.
`smem_o[8][512]` f32 = 16 384 B + 64 B LUT.

**Launch**: `ops::paged_decode_attn_nvfp4`, grid `(num_q_heads, num_seqs, 1)` × `(256,1,1)`,
from `run_paged_decode.rs:216-237` (non-V4 NVFP4) and reused verbatim for the Turbo4/3/2/8
dtypes (`:251`, `:280`) which share the `(block_stride, data_section)` interface.

**Bytes/call**: 512/2 + 512/16 = 288 B/token/side, ×2 sides = 576 B/token — 3.6× less than BF16.
Split-K pair: `paged_decode_attn_splitk_nvfp4` (`:329`), `paged_decode_attn_reduce_nvfp4` (`:506`).

Not on the V4-Flash path.

---

## mla_batched_gemv

`kernels/gb10/deepseek-v4-flash/nvfp4/mla_absorbed.cu:33`

Per-head batched GEMV: `output[head, n] = Σ_k weight[head, n, k] · input[head, k]`, all heads in
parallel. Used for the V3-style **Q absorption** (`Q_nope @ W_UK^T`) and **V extraction**
(`attn_latent @ W_UV^T`).

**Grid**: `(ceil(N_out / 8), num_heads, 1)`; **Block**: `(256,1,1)`. Each block owns
`N_PER_BLOCK*2 = 8` output elements for one head; 64 threads (2 warps) reduce K per output;
`threads_per_out = 64`. Input loaded as `unsigned long long` (4 BF16 per load, `:61,:67`);
**weights loaded scalar** (`B[n1*K + base_k + 0..3]`, `:80-83`) — 4 separate `LDG.U16` where a
64-bit load would do, and `B` is the *big* operand.

**Shared**: `s_partial[8][2]` f32 = 64 B. **Launcher**: `crates/spark-model/src/layers/ops/kv_cache.rs:690`.

**Dead for V4-Flash**: the absorption weights `W_UK`/`W_UV` are NULL stubs for V4-Flash
(`multi_seq/mla.rs:236-249` — "V4-Flash uses the DIRECT-KV attention algorithm, NOT the absorbed-MLA
chain"). Only V3/V4-Pro reach it.

**Inefficiency**: the weight operand is read scalar while the tiny input operand is vectorised —
exactly backwards. Also every `(head, n_tile)` block re-reads the full `input[head, 0..K]`
(`:65-75`), i.e. `ceil(N_out/8)` redundant reads of the activation; that is cheap
(K is small) but the weight scalar loads are not.

## mla_q_rope_scatter

`kernels/gb10/deepseek-v4-flash/nvfp4/mla_absorbed.cu:139`

Fuses "extract Q_rope from `q_full[nq, hd]` at offset `nope`" with "scatter into the strided
`q_absorbed_buf[nq, mla_cache_dim]`" and "write a contiguous `q_rope_contiguous[nq*rope]` for the
RoPE kernel" — one read, two writes, replacing 128 D2D copies per decode step.

**Grid** `(1,1,1)`, **Block** `(256,1,1)` (`multi_seq/mla.rs:709-723` via `ops::mla_q_rope_scatter`).
Total elements = `64 × 64 = 4 096` BF16 = 8 KiB in, 16 KiB out. **24 KB → 0.09 µs roofline**;
entirely launch-latency-bound (~5 µs). A single block on 48 SMs, but the work is trivial.

**Inefficiency**: grid `(1,1,1)` — 1/48 of the GPU. Irrelevant at 24 KB, but it *is* one of
~15 such single-block micro-kernels per layer; 43 layers × ~5 µs of launch each is real
(**~0.2 ms/step** for this kernel family collectively; CUDA-graph capture should absorb it).

## mla_q_rope_writeback

`kernels/gb10/deepseek-v4-flash/nvfp4/mla_absorbed.cu:165`

Scatters post-RoPE `q_rope_direct[nq*rope]` back into `q_absorbed_buf[head*mla_cache_dim + kv_lora ..]`.
**Grid** `(1,1,1)`, **Block** `(256,1,1)` (`multi_seq/mla.rs:809-820`). 4 096 elements, 16 KB
traffic. Launch-bound. Strided writes (stride 576 × 2 B between heads) are uncoalesced but the
volume is negligible.

## mla_q_rope_extract_batched

`kernels/gb10/deepseek-v4-flash/nvfp4/mla_absorbed.cu:188`

Prefill/CSA batched form of the extract: gathers `q_full[t, head, nope..nope+rope)` into a
contiguous `[N, nq*rope]`. **Grid** `(ceil(total/256),1,1)`, **Block** `(256,1,1)`, grid-stride loop.
Also used at **decode** by the CSA compressed-block rope path with `num_tokens=1, nq=1`
(`cache_skip_v4.rs:508-519` and `:840`-region catch-up), where it moves 128 B — pure launch overhead.

## mla_q_rope_writeback_batched

`kernels/gb10/deepseek-v4-flash/nvfp4/mla_absorbed.cu:212`

Inverse of the above. Same grid/block, same grid-stride loop, same decode-time 128-B usage
(`cache_skip_v4.rs:536-548`).

## mla_kv_assemble_batched

`kernels/gb10/deepseek-v4-flash/nvfp4/mla_absorbed.cu:240`

Prefill helper: `blockIdx.y == 0` assembles `K = [k_nope | k_rope broadcast]` for `nkv` heads;
`blockIdx.y == 1` extracts `V` from the `nope+v_dim` interleaved `kv_expanded`.
**Grid** `(num_tokens, 2, 1)`, **Block** `(256,1,1)`. Replaces `3 × N × nkv` D2D copies.
Prefill-only. Reads/writes are BF16 contiguous within a head; the k_rope broadcast (`:266`) is a
1-element-per-thread read of the same 64-value row by every head — L1-resident, harmless.

## mla_cache_assemble_batched

`kernels/gb10/deepseek-v4-flash/nvfp4/mla_absorbed.cu:288`

Builds `K_cache = [latent | k_rope]` and `V_cache = [latent | k_rope]` (BF16) for N tokens.
**Grid** `(num_tokens,1,1)`, **Block** `(mla_cache_dim or 256,1,1)`.

The comment at `:308-317` documents the load-bearing fix: **V's rope tail must equal K's**, not
zeros, because MLA passes the latent as both key and value. **This confirms `K == V` bit-for-bit
in the V4 cache** and is the direct justification for finding 2 (delete the V load in
`mla_paged_decode_fp8`).

Traffic: `N × 576 × 2 B` read + `2 × N × 576 × 2 B` written = `N × 3.4 KB`. Prefill-scale.

## mla_q_final_assemble_batched

`kernels/gb10/deepseek-v4-flash/nvfp4/mla_absorbed.cu:328`

Interleaves `q_absorbed[N, nq*kv_lora]` and `q_rope[N, nq*rope]` into
`q_final[N, nq*(kv_lora+rope)]`. **Grid** `(ceil(N*nq*576/256),1,1)`, **Block** `(256,1,1)`
(`ops/prefill_attn_a.rs:190-215`), grid-stride. Fully coalesced on the write side; the read side
does an integer `div`/`mod` by `nq*mla_cache_dim` per element (`:340-343`) — 3 integer divisions
per element on a memory-trivial kernel. Replacing with shifts/precomputed reciprocals is free but
the kernel is not hot.

## mla_cache_assemble

`kernels/gb10/deepseek-v4-flash/nvfp4/mla_absorbed.cu:358`

Single-token decode form: one block, `threadIdx.x` covers `mla_cache_dim`. Replaces 4 D2D copies
+ 1 memset per decode step. **Grid** `(1,1,1)`, **Block** `(max(mla_cache_dim,256),1,1)` —
`ops/kv_cache.rs:784-795`.

> **INCONSISTENCY**: this kernel writes `v_cache_entry[idx] = 0` for the rope tail (`:375`),
> which is exactly the bug the batched sibling's comment (`:308-317`) says was fixed. It is only
> reached on the absorbed-MLA (non-V4-Flash) chain (`multi_seq/mla.rs:834-846`), so V4-Flash is
> unaffected — but it is a divergence from `mla_cache_assemble_fp8_batched`, which correctly
> writes the rope to both.

Traffic: 576×2 read + 2×576×2 written ≈ 3.4 KB. Pure launch overhead (~5 µs).

---

## mla_cache_assemble_fp8_batched

`kernels/gb10/deepseek-v4-flash/nvfp4/mla_cache_assemble_fp8.cu:32`

Prefill path: reads BF16 `k_bf16[N, mla_cache_dim]` / `v_bf16[N, kv_lora]`, scales by
`k_scale`/`v_scale`, and writes FP8-E4M3 `k_cache_fp8` / `v_cache_fp8`, both at
`mla_cache_dim = 576`. The rope tail (`:70-89`) writes **K's rope value into both caches** — this
is the V4 `K == V` invariant, stated explicitly at `:71-76`.

**Grid** `(num_tokens, 1, 1)`, **Block** `(256,1,1)`, strided `for (d = tid; d < 576; d += 256)`
so each thread handles 2–3 dims.

**Dtypes**: BF16 in, FP8 out. **Shared**: none. **Registers**: trivial.
**Traffic**: `N × (576+512) × 2` read + `N × 2 × 576` written = `N × 3.3 KB`.
At a 4 k prompt that is 13.5 MB → **49 µs**, once per layer. Bandwidth-bound but tiny.

**Inefficiencies**: (a) 1-byte FP8 stores, no vectorisation — 576 scattered `STG.U8` per token
where 36 `STG.128` would do; (b) the `for (head = 0; head < nkv; head++)` loop (`:57`, `:79`) is a
1-trip loop for V4 (`nkv=1`) that the compiler must still prove; (c) the `unsigned long long`
index arithmetic is recomputed inside the head loop.

---

## csa_compress

`kernels/gb10/deepseek-v4-flash/nvfp4/csa_compress.cu:20`

The DeepSeek Sparse Attention **compressor**. Produces one compressed KV entry per window of
`ratio` source tokens via a **per-dimension online softmax** over the window slots, gated by
`gate[tok, d] + ape[r, d]`:

```
C[w, d] = Σ_s softmax_s(gate[w,s,d] + ape[r,d]) · kv[w,s,d]
```

CSA mode (`ratio = 4`, `proj_dim = 2·head_dim`): two interleaved series with a `2·ratio`-wide
overlap — slots `[0, ratio)` read the **previous** window's first `head_dim` half (Ca),
slots `[ratio, 2·ratio)` read the **current** window's second half (Cb). Window 0's Ca half is
masked. HCA mode (`ratio = 128`, `proj_dim = head_dim`): a single non-overlapping window.

`ape` is **FP32** — the comment at `:23-26` records that reading it as BF16 read half of the
wrong element and corrupted the gate on every compressed layer.

**Launch**

| | prefill | decode / catch-up |
|---|---|---|
| Grid | `(n_win, 1, 1)` | `(launch_win, 1, 1)` |
| Block | `(256, 1, 1)` | `(256, 1, 1)` |
| Site | `crates/spark-model/src/layers/qwen3_attention/prefill/cache_skip_v4.rs:477-485` | `crates/spark-model/src/layers/qwen3_attention/decode/attention_forward_v4.rs:807-819` |

**Dtypes**: `kv`/`gate` BF16, `ape` FP32, `out` BF16. **Shared**: none. **Registers**: 3 f32 of
softmax state per dim, tiny.

**Bytes per call at decode** (`launch_win` is typically 1–2 windows, `head_dim = 512`,
`proj_dim = 1024`, `ratio = 4`): each output dim reads `2·ratio = 8` `(gate, kv)` BF16 pairs
plus `2·ratio` FP32 `ape` entries = `8×2×2 + 8×4 = 64 B` per dim × 512 dims = **32 KB per
window**, plus the `t_rows × proj_dim` BF16 projection inputs the caller produced. Roofline
**0.12 µs** — launch-bound.

**Bytes at prefill** (4 k prompt, `n_win = 1024`): 1024 × 32 KB = **33 MB → 120 µs/layer**,
×43 = 5.2 ms of prefill. Acceptable.

**Inefficiencies**: (a) `for (d = threadIdx.x; d < head_dim; d += blockDim.x)` with
`head_dim = 512`, `blockDim = 256` → each thread does exactly 2 dims; the **inner slot loop is
serial and dependent** (online softmax), so ILP is 2. (b) The reads `gate[tok*proj_dim + d]`
stride by `proj_dim = 1024` elements between slots — each slot's 256-thread read is coalesced
within a window but the 8 slots touch 8 different 2 KiB rows. (c) `ape[r*proj_dim + d]` is
re-read per window from DRAM instead of being staged into shared memory once
(`ratio × proj_dim × 4 B = 16 KB` for CSA — fits smem, would eliminate half the traffic).
(d) `__expf` is called twice per slot per dim (`:52-53`) even though `eo` is 1.0 whenever the
running max does not move.

---

## prefill_attn_compressed

`kernels/gb10/deepseek-v4-flash/nvfp4/prefill_attn_compressed.cu:23`

The **prefill oracle** for CSA: one softmax over
`[ raw sliding-window causal KV | compressed windowed-causal KV | per-head sink ]`.
Query at position `t` attends raw keys in `[max(0, t+1-window), t]`, compressed entries `w` with
`(w+1)·ratio ≤ t+1`, plus the sink logit (denominator only).

This kernel **defines the semantics** `mla_paged_decode_fp8` must match; note that it dots the
compressed arm over dims `0..head_dim-1` with `head_dim = hd_mla = 512`, so the compressed arm is
**rope-free** — which is why the decode kernel deliberately skips the rope overwrite for
compressed blocks (`mla_paged_decode_fp8.cu:304-311`).

**Launch**: grid `(nq=64, ceil(S/16), 1)`, block `(128,1,1)` —
`crates/spark-model/src/layers/qwen3_attention/prefill/cache_skip_v4.rs:588-607`.
Layout = 16 query rows × 8 dim-lanes; each dim-lane owns 64 of the 512 dims; the cross-lane
reduction is 3 `__shfl_xor` steps over 8 lanes (`:70-72`).
`inv_sqrt_d = 1/sqrt(512)` (`:606`), `sliding_window = V4_WINDOW = 128` (`:605`).

**Dtypes**: all BF16 except `sinks` FP32. **Shared**: none. **Registers**: `o_acc[64]` f32 = 64
registers per thread → this alone caps occupancy at ~4 CTAs/SM for a 128-thread block.

**Bytes/call**: per query row, `(128 raw + t/4 compressed) × 512 × 2 B × 2 (K,V)`. At `t = 4096`:
`(128 + 1024) × 2 048 = 2.36 MB` per query row, and each of the 8 dim-lanes reads only its 64
dims — so 2.36 MB per row total across the lane group. `S = 4096` rows × 64 heads → **619 GB**
per layer if nothing caches. In practice K/V for a 16-row tile is reused across the tile and
across heads via L2. Still: this is the dominant prefill cost and it is **O(S²/ratio)**.

**Inefficiencies**: (a) `o_acc[64]` in registers is the occupancy killer; (b) K and V are read
from the *same pointer* — the launcher passes `k_out` for both K and V, and `comp_k` for both Kc
and Vc (`cache_skip_v4.rs:592-596`), so `ATTEND(Kr, Vr)` reads the same cache line twice (`:68`
and `:78`); (c) no `BR`-tile shared-memory staging of K — every one of the 16 rows in the tile
re-reads the same key row from L1/L2 independently; (d) the compressed loop starts at `w = 0`
every row, so the innermost work grows linearly with `q_row` with no blocking.

---

## inferspark_prefill_512

`kernels/gb10/deepseek-v4-flash/nvfp4/inferspark_prefill_512.cu:12`

Scalar reference prefill flash-attention at `HDIM 512` for the **non-CSA (full-attention)**
V4 layers, with the per-head attention sink applied (`:120-130` — the comment records that
omitting it made prefill diverge from decode and corrupt the prompt KV).

**Launch**: grid `(num_q_heads, ceil(seq_len/16), batch)`, block `(128,1,1)` —
`crates/spark-model/src/layers/ops/prefill_attn_main_a.rs:85-101` (`prefill_attention_512_sink`).
Layout identical to `prefill_attn_compressed`: 16 rows × 8 dim-lanes × 64 dims,
`o_acc[64]` f32 registers, 3-step `__shfl_xor` reduction.

**Dtypes**: BF16 Q/K/V/O, FP32 `sinks`. **Shared**: none.
**Bytes/call**: `kv_len × 512 × 2 B × 2` per query row; `O(S²)` causal. Same
register-pressure and no-smem-staging issues as `prefill_attn_compressed`. Prefill-only.

---

## mla_prefill_attn_320

`kernels/gb10/deepseek-v4-flash/nvfp4/mla_prefill_attn.cu:26`

Absorbed-MLA prefill attention at `MLA_HDIM 576` (the "320" in the name is retained only for Rust
symbol compatibility, `:24-25`). Scalar BF16 dot products with FP32 accumulation; no tensor cores
— written to dodge an SM121 PTX JIT issue with the tensor-core prefill at HDIM=320 (`:6-7`).

**Launch**: grid `(num_q_heads, ceil(seq_len/16), batch)`, block `(256,1,1)` —
`crates/spark-model/src/layers/ops/prefill_attn_a.rs:250+` (`mla_prefill_attention_320`).
256 threads = 16 query rows × 16 lanes; each lane owns 36 dims (`576/16`), `acc_o[36]` f32.

**Dtypes**: BF16 in/out. **Shared**: none. **Registers**: `acc_o[36]` + addressing.

**Inefficiencies**: (a) the 16-lane sub-warp reduction uses a full `0xFFFFFFFF` mask with offsets
1,2,4,8 and then a `__shfl_sync` broadcast from `(warp_lane/16)*16` (`:99-112`) — correct but
4 extra shuffles per key; (b) **the KV tile loop `for (kv_start; ...; kv_start += MLA_BC)` is
vestigial** — the inner loop immediately walks positions one at a time (`:85`), so `MLA_BC = 16`
buys nothing, there is no tiling and no shared-memory staging; (c) V4-Flash does not use the
absorbed chain, so this is dormant for the target model.

---

## mla_fused_prefill

`kernels/gb10/deepseek-v4-flash/nvfp4/mla_fused_prefill.cu:21`

Fuses **Q absorption + attention + V extraction + KV-cache write** into one kernel: one CTA per
`(head, query_token)` does `Q_absorbed = Q_nope[448] @ W_UK^T`, builds
`Q_final[576] = [Q_absorbed | Q_rope]` in shared memory, runs the causal online softmax over all
KV tokens, then `V_out[512] = attn_latent @ W_UV^T`. Eliminates 6 launches and all intermediate
buffer traffic per layer.

**Launch**: grid `(nq, seq_len, 1)`, block `(256,1,1)` —
`crates/spark-model/src/layers/ops/prefill_attn_a.rs:144-189`.

**Shared**: `smem_q[576]` f32 (2 304 B) + `smem_dot[8]` (32 B) + `smem_latent[512]` f32 (2 048 B)
= **4 384 B**. **Registers**: `acc_latent[2]` — deliberately light so the weights can stream.

**Bytes/call**: `W_UK` is `[kv_lora=512, nope=448]` BF16 = **459 KB per KV head**, `W_UV` is
`[v_dim=512, kv_lora=512]` BF16 = **524 KB**. With `num_kv_heads = 1` **every one of the
`nq × seq_len` CTAs reads the same 983 KB of weights** (`:65`, `:202`). At `seq_len = 4096`:
`64 × 4096 × 983 KB = 258 GB` of weight demand per layer — utterly L2-bound. The header comment
("L2 cached per head, 32KB + 64KB") assumes the V3 shape (`nope=64`, `kv_lora=256`), which is
**7–8× smaller than the V4-Flash shape**. This kernel does not scale to V4-Flash dimensions.

**Inefficiencies**: (a) the Q-absorption inner loop (`:73-75`) re-reads `q_nope_ptr[k]` from
global for every one of the 512 output dims — 448×512 scalar global loads per CTA where a single
shared-memory stage of Q_nope (896 B) would do; (b) the same for `W_UK` rows, read scalar; (c) the
attention loop does a **full 8-warp `__syncthreads()`-gated reduction per KV position** (`:138-158`)
— two `__syncthreads()` per key, with a single-thread scalar sum of 8 partials at `:151-155`; at
`seq_len = 4096` that is 8 192 barriers per CTA; (d) `W_UV` dot at `:206-210` is again scalar over
`kv_lora = 512`. Dormant for V4-Flash (absorbed chain unused) but should not be revived as-is.

---

## grouped_gemm_mla

`kernels/gb10/deepseek-v4-flash/nvfp4/grouped_gemm_mla.cu:35`

`G` independent small GEMMs in one launch: `C_g[M, N_g] = A_g[M, K_g] @ B_g[N_g, K_g]^T`.
Block-diagonal GEMM without zero padding. Used for MLA Q absorption (`G=32, K_g=nope, N_g=kv_lora`)
and V extraction (`G=32, K_g=kv_lora, N_g=v_dim`).

**Launch**: grid `(M*G, ceil(N_g/4), 1)`, block `(256,1,1)` —
`crates/spark-model/src/layers/ops/prefill_attn_a.rs:224-249`. `GG_TILE_N = 4` outputs per block,
64 threads (2 warps) reduce `K_g` per output.
**Shared**: `s_partial[4][2]` f32 = 32 B.

**Inefficiencies**: (a) both operands are read **scalar** (`A_row[k]`, `B_row[k]`, `:72-73`) —
one BF16 element per load, per thread; a `uint4` load would fetch 8; (b) every `n_tile` block
re-reads the full `A_row[0..K_g]` — with `K_g = 448` and `ceil(N_g/4) = 128` tiles, the activation
row is read **128×**; (c) `s_partial` is written then `__syncthreads()`, but only `k_lane == 0`
reads it — a `__shfl_down` across the 2 warps via one smem word would be cheaper;
(d) `if (n_idx >= N_g) return;` after `__syncthreads()` would be a hang, but the return at `:63`
precedes the barrier at `:90` — divergent early return before a `__syncthreads()` is UB if any
thread in the block exits. With `N_g` a multiple of 4 it never fires, but it is fragile.

Dormant for V4-Flash (absorbed chain).

---

## dspark_rope

`kernels/gb10/deepseek-v4-flash/nvfp4/dspark_drafter.cu:29`

Interleaved RoPE for the DSpark block drafter: rotates the last `rope_dim = 64` dims of each head
in place, viewing them as `[32, 2]` `(re, im)` pairs. Plain θ=10000, **no YaRN** (the drafter is
pure sliding-window attention, `:15-17`). `inverse = 1` multiplies by the conjugate — that is the
MLA output de-rotation.

**Launch**: grid `(rows, heads, 1)`, block `(ROPE_DIM/2 = 32, 1, 1)` —
`crates/spark-model/src/layers/dspark_head.rs:301-303`. One thread per rotation pair.

At the propose shape (`rows = 5` block rows, `heads = 64`): grid `(5, 64)` = **320 CTAs of
32 threads** = 10 240 threads, each doing 2 BF16 reads, 2 writes, and 2 FP32 `freqs` reads.
Traffic ≈ `5 × 64 × 64 × (2×2 + 2×4)` = 245 KB → **0.9 µs**. Launch-bound. Called twice per
drafter stage (K rope and output de-rotation).

**Inefficiency**: 32-thread blocks waste 7/8 of each SM's warp slots; `heads` should be folded
into `threadIdx.y` so a block is at least 256 threads. Negligible at this size.

## dspark_attn

`kernels/gb10/deepseek-v4-flash/nvfp4/dspark_drafter.cu:69`

The drafter's attention: `rows = 5` query rows (1 committed + 4 noise), each attending
**bidirectionally** (no causal mask inside the block) over all 5 block KV rows plus a
`ring_vis ≤ 128` sliding window of `main_kv` rows. MQA — one shared 512-dim KV row per position.
Per-head attention-sink logit joins the denominator only. `V == K` (the full 512-dim row).

Explicitly optimised for **correctness against `inference/model.py`, not bandwidth** (`:10-12`) —
the propose forward is dominated by MoE and lm_head reads.

**Launch**: grid `(b = rows, HEADS = 64, 1)`, block `(128,1,1)` —
`crates/spark-model/src/layers/dspark_head.rs:576-589`. `scale = 1/sqrt(512)`.
**Shared**: `s_scores[160]` + `s_red[128]` + 2 scalars f32 = **1 160 B**.

**Bytes/call**: per CTA, pass 1 reads `nk = ring_vis + rows ≤ 133` KV rows × 512 × 2 B = **136 KB**;
pass 2 (`:139-147`) **re-reads every one of those 133 rows again**, per output dim group. Total
per CTA ≈ 272 KB; × `5 × 64 = 320` CTAs = **87 MB per stage → 319 µs**. With 5 drafter stages that
is **1.6 ms**, entirely from re-reading the same 136 KB ring 320× (finding: the ring is shared by
all 64 heads and all 5 rows, exactly like the target's KV).

**Inefficiencies**: (a) **pass 1 is catastrophically strided** — each thread owns one key and
loops `for (d = 0; d < head_dim; ++d)` scalar (`:98-100`), so 128 threads each stream a *different*
512-B row scalar; zero coalescing. Should be one warp per key with lane-strided dims, or stage
`q` in smem and use a warp-reduce; (b) **pass 2 re-reads all `nk` KV rows** with a `d`-strided
access into each (`:141-146`) — column-major access across 133 rows, again fully uncoalesced;
(c) the `k < ring_vis ? ring : blk_kv` ternary is evaluated per element in the innermost loop;
(d) 3 full block barriers per reduction tree × 2 trees (`:112-115`, `:129-132`) with
`blockDim = 128` = 14 `__syncthreads()`. Staging the ring rows into shared memory once per CTA
(133 × 512 B = 68 KB — too big; but a 32-key tile is 16 KB) would fix (a) and (b) together.

---

## hc_expand

`kernels/gb10/deepseek-v4-flash/nvfp4/hyper_connection.cu:38`

Broadcasts one BF16 hidden state into `hc_mult = 4` identical FP32 streams:
`streams[t, i, d] = hidden[t, d]`. **Grid** `(T,1,1)`, **Block** `(256,1,1)` —
`crates/spark-model/src/layers/ops/hyper_connection.rs:15-32`.

Traffic at decode (`T=1, H=4096`): 8 KB read, `4 × 4096 × 4 = 64 KB` written = **72 KB → 0.26 µs**.
Launch-bound. Grid `(1,1,1)` at decode → 1/48 of the GPU, but irrelevant at this size.

## hc_pre

`kernels/gb10/deepseek-v4-flash/nvfp4/hyper_connection.cu:58`

Collapses the `hc_mult` residual streams to one (RMS-rescaled mix-logits → sigmoid `pre` weights)
and emits the `post` vector and the doubly-stochastic `comb` matrix (Sinkhorn) for the matching
`hc_post`. **One block per token**: grid `(T,1,1)`, block `(256,1,1)` —
`ops/hyper_connection.rs:38-72`.

Pass 2 (`:100-111`) streams the entire `hc_fn` matrix `[mix_hc=24, hc*H=16384]` FP32 =
**1.5 MiB**, one `mix_hc` row at a time with a full block reduction and **two `__syncthreads()`
per row** (`:107`, `:110`). Pass 3 collapses `y[d] = Σ_i pre[i]·x[i,d]`.

**Shared**: `red[256]` + `s_rsqrt` + `s_mix[24]` + `s_pre[4]` f32 = **1 140 B**.

**At decode this is a disaster** and the source says so (`:196-208`): T=1 means one block on
48 SMs streaming 1.5 MiB at single-SM bandwidth (~8 GB/s measured), and there are **two HC sites
per layer × 43 layers = ~129 MiB/token**, i.e. tens of ms for work that costs 0.5 ms at
254 GB/s. **Decode must use `hc_pre_split` instead.** `hc_pre` remains correct and appropriate for
prefill, where `grid.x = T` fills the GPU.

> Note: `crates/spark-model/src/layers/dspark_head.rs:455` and `:664` call `ops::hc_pre` (the
> fused, one-block form) — the DSpark drafter runs the **slow** variant at its `B = 5` token width.
> 5 blocks on 48 SMs streaming 1.5 MiB each. Worth switching to `hc_pre_split`.

## hc_pre_mix

`kernels/gb10/deepseek-v4-flash/nvfp4/hyper_connection.cu:228`

Decode half of the `hc_pre` split. One block per mix row (`m = 0..mix_hc-1`) plus one extra block
(`m == mix_hc`) for the `Σx²` RMS reduction, so the 24 independent dot products issue concurrently
across SMs. `HC_MIX_BLOCK = 512` threads, `float4` loads (`:248-267`).

**Launch**: grid `(T, mix_hc+1 = 25, 1)`, block `(512,1,1)` — `ops/hyper_connection.rs:102-114`.
**Shared**: `red[512]` f32 = 2 048 B.

The comment at `:217-226` is explicit about the reasoning: the loop is a dependent accumulate, so
throughput is (bytes in flight)/(DRAM latency); 256 scalar-loading threads measured **19 GB/s**
even with all SMs busy, and 512 threads × 16-byte loads puts 8× more in flight.

**Traffic**: `hc_fn` 1.5 MiB + 25 re-reads of `x` (64 KiB each, L2-resident) = 1.5 MiB DRAM →
**5.7 µs** at 273 GB/s (vs ~190 µs for the fused form at 8 GB/s single-SM). ×2 sites ×43 layers =
**0.49 ms/token** — this split is worth ~15 ms/token and is already in place.

**Note**: the `float4` lanes reassociate the sum relative to the scalar `hc_pre` (~1e-7 relative
drift), so **decode's collapse is not bit-identical to prefill's** (documented at `:223-226`).

**Remaining inefficiency**: 25 blocks on 48 SMs = 52 % occupancy at the grid level; the extra
RMS block is deliberately separate (`:252-254`) so that no block does double work. Could split
each `m` row across 2 blocks with an atomic/second-pass reduce to use all 48 SMs.

## hc_pre_finish

`kernels/gb10/deepseek-v4-flash/nvfp4/hyper_connection.cu:291`

Second half of the split: `blockIdx.y == 0`'s thread 0 does the scalar Sinkhorn
(`hc × hc = 4×4`, `sinkhorn_iters = 20`) and writes `post`/`comb`; every block recomputes the
~30-flop `pre` weights locally rather than round-tripping them; then the collapse
`y[d] = Σ_i pre[i]·x[i,d]` is sharded over `d`.

**Launch**: grid `(T, ceil(H/256) = 16, 1)`, block `(256,1,1)` — `ops/hyper_connection.rs:115-129`.
**Shared**: `s_pre[4]` f32 = 16 B.

**Traffic**: reads `x` (64 KiB FP32) once across the 16 blocks + `mix_in` (100 B), writes
`y_out` 8 KiB BF16 = **72 KB → 0.26 µs**. Launch-bound.

**Inefficiency**: the Sinkhorn is **entirely serial on one thread** — 20 iterations × 2 passes ×
16 divisions = ~640 dependent FP32 divides, plus a softmax, on a single lane while 255 threads
wait at the `__syncthreads()` (`:381`). At ~4 cycles/divide on one lane that is a few µs of
pure serialisation per HC site, ×2 sites ×43 layers ≈ **0.2–0.4 ms/token**. A 16-thread
cooperative Sinkhorn (one thread per `comb[i][j]`) would cut it ~10×. The final exact column
projection (`:370-375`) is documented as load-bearing for coherence onset (`:368-369`) — do not
remove it, but it can be parallelised.

## hc_post

`kernels/gb10/deepseek-v4-flash/nvfp4/hyper_connection.cu:398`

Expands the sublayer output back into the `hc_mult` streams, mixing the saved residual streams
through the doubly-stochastic `comb`:
`out[t,j,d] = post[t,j]·block_out[t,d] + Σ_i comb[t,i,j]·residual[t,i,d]`.
`out` may alias `residual` (all `hc` residual values are read into `rv[]` before any write, `:421`).

**Launch**: grid `(T, shards, 1)`, block `(256,1,1)` — `ops/hyper_connection.rs:175-201`.
`shards = 1` for prefill (grid.x=T fills the GPU); decode passes a larger `shards` so the
embarrassingly-parallel `d` loop uses more than one SM.
**Shared**: none. **Registers**: `rv[4]`.

**Traffic** at decode (`T=1, H=4096, hc=4`): reads `block_out` 8 KB BF16 + `residual`
`4×4096×4 = 64 KB` FP32, writes `out` 64 KB = **136 KB → 0.5 µs**. Two sites × 43 layers =
**43 µs/token**. Bandwidth-trivial; launch-bound at ~5 µs × 86 = 0.43 ms — CUDA-graph capture is
the lever here, not the kernel.

**Inefficiency**: FP32 residual streams cost 2× the BF16 alternative for 4 096×4 values.
`hc` streams in BF16 would halve this kernel's and `hc_pre_mix`'s `x` traffic — but the FP32
"highway" is presumably deliberate for numerical headroom.

## hc_head

`kernels/gb10/deepseek-v4-flash/nvfp4/hyper_connection.cu:433`

Final collapse before the LM head: a single learned sigmoid-weighted sum over the streams.
Structurally `hc_pre` with `mix_hc → hc = 4` rows and no Sinkhorn.

**Launch**: grid `(T,1,1)`, block `(256,1,1)` — `ops/hyper_connection.rs:205-232`.
Called at `crates/spark-model/src/layers/deepseek_v4_mtp.rs:406` and
`crates/spark-model/src/layers/dspark_head.rs:722`.
**Shared**: `red[256]` + `s_rsqrt` + `s_pre[4]` = 1 044 B.

**Traffic**: `head_fn` is `[hc=4, hc*H=16384]` FP32 = **256 KiB**, streamed by **one block**
(`:469-483`) with two `__syncthreads()` per row. At the ~8 GB/s single-SM rate the source
measured for `hc_pre`, that is **~32 µs**; at full bandwidth it would be 0.94 µs.
This is **finding 6** — `hc_head` never got the `hc_pre_mix` treatment. Once per token for the
main model, but the DSpark drafter calls it per stage.

**Fix**: same split as `hc_pre` — grid `(T, hc+1, 1)` for the 4 row dot products + RMS, then a
finish kernel. ~30 µs/token, more with MTP/DSpark stages.

## hc_mean

`kernels/gb10/deepseek-v4-flash/nvfp4/hyper_connection.cu:499`

DSpark target-hidden capture: the **plain mean** over the `hc` streams (`h.mean(dim=2)` in the
official `inference/model.py`), BF16 out. The comment at `:493-497` is load-bearing — the
drafter's `main_proj` is trained on the plain mean, **not** the learned `hc_head` collapse, so
substituting `hc_head` would feed the drafter an out-of-distribution input.

**Launch**: grid `(T,1,1)`, block `(256,1,1)` — `ops/hyper_connection.rs:239-259`,
called at `crates/spark-model/src/model/impl_b3.rs:607`.
**Traffic**: 64 KiB FP32 read + 8 KiB BF16 written = **72 KB → 0.26 µs**. Launch-bound.
No weights, so the single-block shape costs nothing here.

---

## Appendix — what to fix first, concretely

1. **Head-tile `mla_paged_decode_fp8`.** Change `blockIdx.x` from `q_head` to `q_head_group`,
   hold `q_reg[HT][16]`, `m[HT]`, `l[HT]`, `o_reg[HT][16]`. Drop the V load (K==V). With the V
   array gone, `HT = 8` fits in ~150 registers. DRAM demand for the attention arms drops
   **16×** (8× from head tiling, 2× from K==V).
2. **M-tile the same kernel** for the verify widths: add an `M` dimension (`grid.z` or an inner
   loop) so the γ=2/γ=6 verify loads each K/V row once for all M rows, replacing the M separate
   launches at `multi_seq/mla.rs:498-522`.
3. **Vectorise the FP8 loads** to `uint4` + `unpack4_fp8` (copy the pattern from
   `paged_decode_attn_fp8_mla.cu:47-60`).
4. **Fix the cross-warp smem reduction** to index `smem_o[w][lane*16+i]` instead of `[0..512]`.
5. **Split `hc_head`** the way `hc_pre` was split, and switch the DSpark drafter's
   `ops::hc_pre` calls (`dspark_head.rs:455`, `:664`) to `hc_pre_split`.
6. ~~**Fix `VEC_U32_FP8` truncation** in `paged_decode_attn_fp8_mla.cu`~~ — **done**, see finding 7.
   Guarded with `FP8_U32_OK (HDIM % (WARP_SIZE*4) == 0)` plus a byte-granular
   `load_lane_fp8()` fallback; this also fixed the misaligned uint32 loads that the
   original finding missed.
