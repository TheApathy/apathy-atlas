# Dense GEMM / GEMV kernels (`kernels/gb10/common/` + `deepseek-v4-flash/nvfp4/`)

Target: **DeepSeek-V4-Flash-162B**, one **NVIDIA GB10 (DGX Spark)**, sm_121, LPDDR5X,
**273 GB/s**, 120 GB, **48 SMs**.

Shapes confirmed from `/home/flocka/models/DeepSeek-V4-Flash-162B/config.json`:

| symbol | value | source |
|---|---|---|
| `hidden_size` (`h`) | 4096 | config |
| `num_attention_heads` (`nq`) | 64 | config |
| `head_dim` (`hd`) | 512 | config |
| `q_dim = nq*hd` | **32768** | `mla.rs:276` |
| `q_lora_rank` | 1024 | config |
| `kv_lora_rank` | 512 | (`head_dim` MLA latent); `mla_cache_dim = 512+64 = 576` |
| `qk_rope_head_dim` | 64 | config |
| `o_lora_rank` | 1024 | config |
| `o_groups` | 8 | config |
| `group_in = q_dim/o_groups` | **4096** | `mla.rs:279` |
| `latent_dim = o_groups*o_lora` | **8192** | `mla.rs:281` |
| `moe_intermediate_size` | 2048 | config |
| `n_routed_experts` / `top_k` / shared | 144 / 6 / 1 | config |
| `num_hidden_layers` | 43 | config |
| `vocab_size` | 129280 | config |

Roofline convention throughout: `t_roof = bytes / 273e9` seconds. Decode is batch=1;
the speculative verify runs **M=2** (γ=1) and **M=6/7** (γ=5/6) rows.

---

## Findings

### F0. The prime suspect is FALSE — every batched GEMV *does* amortise the weight load

The hypothesis in the brief ("M=2 costs ~2.6× M=1 because the `*_batch*` kernels
re-read the weight per row") is **disproven by the source**. Every single
`*_batch2` / `*_batch3` / `*_batch4` / `*_batch8` / `*_batch16` / `*_batchm`
kernel in this tree loads the weight chunk **once** per K-iteration into a
register (`uint4` / `unsigned long long` / `uint32`), dequantises it **once**
into a `float wf[16]` / `float bf[8]` register array, and only then MACs it into
`M` independent FP32 accumulators. Weight DRAM traffic is **O(1) in M**.
See the per-kernel verdict table below and the individual `##` sections.

**Consequence:** the M=2 → 2.6× regression is *not* a kernel-level weight
re-stream. It is (in descending order of measured size):

1. **MoE expert-union growth.** At M=1 the 6 routed experts of one token are read.
   At M=2 the union of two tokens' top-6 is ~11 experts, at M=6 ~26.
   NVFP4, 3 matrices of 2048×4096 per expert = 14.16 MB/expert:
   - M=1: 6 experts → 84.9 MB/layer → **0.311 ms** → ×43 = **13.4 ms**
   - M=2: ~11 experts → 155.7 MB/layer → 0.570 ms → ×43 = **24.5 ms** (+11.2 ms)
   - M=6: ~26 experts → 368.1 MB/layer → 1.348 ms → ×43 = **58.0 ms** (+44.6 ms)

   This single effect accounts for essentially all of `124 ms (γ=2) − 47.6 ms (M=1)`
   once the linear phase-B cost below is added. It is *inherent to MoE*, not a bug.

2. **Phase B of the MLA decode is a per-row Rust loop, linear in M by construction**
   (`crates/spark-model/src/layers/qwen3_attention/trait_impl/multi_seq/mla.rs:498-522`).
   Every token gets its own `attention_forward_v4(skip_qkv:true)` call: rope,
   `mla_cache_assemble_batched`, `write_kv_cache`, `paged_attn`. That is 43 layers ×
   M × ~5 kernel launches. At ~6-8 µs of launch+latency each that is
   **~1.5 ms per extra row across the model**, plus the paged-attention KV re-read.

3. **The `batch_ok` gate can silently drop to the per-row fallback**
   (`mla.rs:297-305`). It requires *all five* of `wq_a_fp8`, `wq_b_fp8`,
   `wkv_a_fp8`, `wo_a_fp8`, `wo_b_fp8` **plus** `expert_up_out_bytes() >= need`.
   `need = n*(2*q_dim + 2*kv_dim + q_lora + latent_dim)*2` bytes
   = n × 155 648 B; at n=6 that is **934 KB** of borrowed MoE scratch.
   If it fails, the fallback re-reads the **whole 107 MB/layer FP8 projection set
   once per row** — 0.392 ms × 43 × (M−1) = **+16.9 ms per extra row**.
   The `ROUTE_ONCE` log at `mla.rs:314` prints which route was taken; check it.

**Priority-ordered optimisation list follows.** ms figures are per decode step
(43 layers) at 273 GB/s unless stated.

---

### F1. `w4a16_gemv_batch2/batch3` and the `qg`/`qkvz`/`dual_*` family are stuck on the OLD narrow K-loop — **~0.9-1.4 ms/step**

`w4a16_gemv.cu:95` (M=1) and `w4a16_gemv_batchm_impl<MAX_M>` (`w4a16_gemv.cu:470`)
were both upgraded to a **64-bit `packed8` weight load + two `uint4` activation
loads + two-chunks-in-flight** loop, with an explicit comment citing
"ncu: 72% of warp stalls are long-scoreboard on GB10".

But `w4a16_gemv_batch2` (`w4a16_gemv.cu:352`), `w4a16_gemv_batch3`
(`:1092`), `w4a16_gemv_qg` (`:657`), `w4a16_gemv_qkvz` (`:757`),
`w4a16_gemv_qg_batch2` (`:865`), `w4a16_gemv_dual_batch2` (`:980`),
`w4a16_gemv_qg_batch3` (`:1208`), `w4a16_gemv_dual_batch3` (`:1336`) all still use
the **`K8` loop**: a single 32-bit `packed4` load (8 weights), one `uint4`
activation, no software pipelining. That is half the load width and no latency
hiding on a memory system whose whole problem is load latency.

**Fix:** delete `w4a16_gemv_batch2` / `batch3` outright and route n=2/3 through
`w4a16_gemv_batchm_impl<4>` (already exists, already correct, already the modern
loop). For `qg`/`qkvz`/`dual_*`, port the `packed8` + 2-chunk body verbatim.

**Estimated:** the M=1 upgrade was worth roughly the observed long-scoreboard
reduction; conservatively 15-25 % on the affected NVFP4 GEMV time. On the V4
verify path the NVFP4 attention projections are 60.2 MB/layer → 0.220 ms ×43 =
9.5 ms; a 15 % latency recovery on the n=2/3 arms ≈ **1.4 ms/step at γ=1**.
Zero risk (the batchm kernel is already bit-identical per row).

### F2. `w8a16_gemv_batch4_ld` on the block-diagonal `wo_a` is launched 8× serially — **~0.5-0.9 ms/step**

`mla.rs:534-580` loops `for g in 0..o_groups` (8 groups), one
`w8a16_gemv_batch4_ld` per group, each with `N=o_lora=1024`, `K=group_in=4096`,
`lda=q_dim=32768`, `ldc=latent_dim=8192`.

Per-group grid is `ceil(1024/4) = 256` CTAs = **5.3 waves on 48 SMs**. That is
fine *per launch*, but there are **8 sequential launches per layer × 43 layers =
344 launches per step** for what is one 8192×4096 block-diagonal matrix. Every
launch pays full kernel-launch + tail-wave latency, and the eight are strictly
ordered on one stream with no overlap.

**Fix:** add a `blockIdx.y = group` dimension to the `_ld` kernels, with
`w_off`/`s_off`/A-offset/C-offset derived from `blockIdx.y`. Grid becomes
`(256, 8, 1)` = 2048 CTAs = 42.7 waves, one launch instead of eight.
Removes 7×43 = **301 launches/step**. At a measured ~3-5 µs of unhidden launch
+ tail per small launch on this platform: **0.9-1.5 ms/step**; conservatively
**0.5-0.9 ms** after CUDA-graph capture absorbs some of it.

### F3. The `_ld` GEMVs read `attn_batch` with `lda=32768` — 4096-element strided rows kill L2 reuse — **~0.2-0.4 ms/step**

In `w4a16_gemv_batchm_impl` / `w8a16_gemv_batchm_impl`, the activation read is
`A + t*lda`, and for `wo_a` group `g` the base is
`attn_batch + g*group_in*2`. Each of the M rows therefore touches a
4096-element (8 KB) window at a 64 KB stride. With M=6 that is 6 disjoint 8 KB
windows per group, 48 KB live — fine for L2, but the 8 groups × 43 layers means
the *same* `attn_batch` buffer is re-walked 8 times per layer with a stride that
defeats any prefetcher.

**Fix (cheap):** stage the M×group_in activation slice into shared memory once
per CTA before the K-loop. `group_in=4096` BF16 × M=6 = 48 KB — too big for
48 KB smem with the LUT, but `group_in/4 = 1024`-element chunks × 6 = 12 KB fits
comfortably and would make every subsequent `k16` iteration a smem hit instead of
an L2 hit. Estimated **0.2-0.4 ms/step**; low confidence, needs ncu.

### F4. `w4a16_gemv` group-scale reads are byte-granular and uncoalesced — **~0.1-0.3 ms/step**

Every batched NVFP4 kernel re-reads
`B_scale[n*num_groups + scale_group]` as a **single byte** inside the k-loop
(`w4a16_gemv.cu:479` in `batchm_impl`; likewise `:120`, `:380`, `:690`, …).
At `threads_per_out=64`, the 64 lanes of one output row read 64 bytes that are
**16 K-elements apart** in the scale array — i.e. one byte per 32-byte sector for
the group they need. The scale array is `K/16` bytes per row, so the true
footprint is 1/16 of the weight footprint (6.25 %), but the *transaction* count
is one 32-B sector per lane per iteration, inflating it toward 2× the weight
traffic in the worst case.

**Fix:** each lane needs `scale_group = k16/1` — with `packed8` (16 weights) the
group index advances exactly 1 per k16 iteration, so lanes `l, l+64, l+128…`
need consecutive scale bytes. Load them as a `uint4` (16 scales) once per 16
k16-iterations and keep them in registers, or stage the row's whole `K/16` scale
vector into smem (`K=4096 → 256 B/row`, 4 rows/CTA = 1 KB). Estimated
**0.1-0.3 ms/step**; the NVFP4 scale traffic is 3.8 MB/layer nominal but the
sector inflation is the real cost.

### F5. Occupancy floor: `wkv_a` (N=512) launches only 128 CTAs = 2.7 waves — **~0.1 ms/step**

`grid = ceil(N/4)` (`gemm_quant.rs:110`, `fp8_gemv_batch.rs:32/62/95/122`).
For the V4 decode projections:

| projection | N | grid CTAs | waves on 48 SMs |
|---|---|---|---|
| `wkv_a` | 512 | 128 | **2.7** |
| `wq_a` | 1024 | 256 | 5.3 |
| `wo_a` (per group) | 1024 | 256 | 5.3 |
| `wo_b` | 4096 | 1024 | 21.3 |
| `wq_b` | 32768 | 8192 | 170.7 |
| lm_head | 129280 | 32320 | 673 |

At 2.7 waves the tail wave is 26 % of the launch and the kernel cannot saturate
273 GB/s. `wkv_a` is only 2.1 MB so it is 0.008 ms roofline but likely runs at
~0.03 ms wall. **Fix:** split-K for N<2048 (a `blockIdx.y` K-split with an
atomic/2-pass reduce, mirroring `dense_gemm_splitk_partial`/`_reduce`) or simply
`N_PER_BLOCK=2` for these shapes to double the CTA count. **~0.1 ms/step.**

### F6. `dense_gemv_bf16.cu:40-41` has a stale comment (documentation bug, 0 ms)

The comment says `threads_per_out = 32` and "which of 8 outputs"; the code
computes `BLOCK_SIZE/N_PER_BLOCK = 256/4 = 64` threads over **4** outputs.
Same wrong text is duplicated in the `gemm_quant.rs:108` doc-comment
("8 outputs/block, 32 threads (1 warp) per output"). Misleads anyone tuning it.

---

### Weight-reuse verdict table (the CRITICAL QUESTION)

Verdict is **REUSED** iff the weight bytes are pulled from DRAM once per
K-iteration and applied to all M rows from registers.

| kernel | file:line | M handled | weight loaded per k-iter | verdict |
|---|---|---|---|---|
| `w4a16_gemv` | `w4a16_gemv.cu:62` | 1 | `u64 packed8` ×2 chunks | n/a (M=1) |
| `w4a16_gemv_sw` | `w4a16_gemv.cu:226` | 1 | `u64 packed8` | n/a (M=1) |
| `w4a16_gemv_logits` | `w4a16_gemv.cu:270` | 1 | `u64 packed8` | n/a (M=1) |
| `w4a16_gemv_batch2` | `w4a16_gemv.cu:352` | 2 | `u32 packed4`, **once** | **REUSED** |
| `w4a16_gemv_batch3` | `w4a16_gemv.cu:1092` | 3 | `u32 packed4`, **once** | **REUSED** |
| `w4a16_gemv_batch4` | `w4a16_gemv.cu:569` | ≤4 | `u64 packed8`, **once** | **REUSED** |
| `w4a16_gemv_batch4_ld` | `w4a16_gemv.cu:585` | ≤4 | `u64 packed8`, **once** | **REUSED** |
| `w4a16_gemv_batch8` | `w4a16_gemv.cu:602` | ≤8 | `u64 packed8`, **once** | **REUSED** |
| `w4a16_gemv_batch8_ld` | `w4a16_gemv.cu:616` | ≤8 | `u64 packed8`, **once** | **REUSED** |
| `w4a16_gemv_batch16` | `w4a16_gemv.cu:632` | ≤16 | `u64 packed8`, **once** | **REUSED** |
| `w4a16_gemv_qg` | `w4a16_gemv.cu:657` | 1 | `u32 packed4` | n/a (M=1) |
| `w4a16_gemv_qkvz` | `w4a16_gemv.cu:757` | 1 | `u32 packed4` | n/a (M=1) |
| `w4a16_gemv_qg_batch2` | `w4a16_gemv.cu:865` | 2 | `u32 packed4`, **once** | **REUSED** |
| `w4a16_gemv_qg_batch3` | `w4a16_gemv.cu:1208` | 3 | `u32 packed4`, **once** | **REUSED** |
| `w4a16_gemv_dual_batch2` | `w4a16_gemv.cu:980` | 2 | `u32 packed4`, **once** | **REUSED** (per projection) |
| `w4a16_gemv_dual_batch3` | `w4a16_gemv.cu:1336` | 3 | `u32 packed4`, **once** | **REUSED** (per projection) |
| `w4a16_gemv_dual` | `w4a16_gemv_fused.cu:51` | 1 | `u32 packed4` | n/a (M=1) |
| `w4a16_gemv_silu_input` | `w4a16_gemv_fused.cu:156` | 1 | `u32 packed4` | n/a (M=1) |
| `w4a16_gemv_dual_sw` | `w4a16_gemv_fused.cu:303` | 1 | `u32 packed4` | n/a (M=1) |
| `w4a16_gemv_silu_input_sw` | `w4a16_gemv_fused.cu:393` | 1 | `u32 packed4` | n/a (M=1) |
| `w8a16_gemv` | `w8a16_gemv.cu:110` | 1 | `uint4` (16 FP8) | n/a (M=1) |
| `w8a16_gemv_batch4` | `w8a16_gemv_batch4.cu:198` | ≤4 | `uint4`, **once** | **REUSED** |
| `w8a16_gemv_batch4_ld` | `w8a16_gemv_batch4.cu:216` | ≤4 | `uint4`, **once** | **REUSED** |
| `w8a16_gemv_batch8` | `w8a16_gemv_batch4.cu:235` | ≤8 | `uint4`, **once** | **REUSED** |
| `w8a16_gemv_batch8_ld` | `w8a16_gemv_batch4.cu:249` | ≤8 | `uint4`, **once** | **REUSED** |
| `w8a16_gemv_batch16` | `w8a16_gemv_batch4.cu:266` | ≤16 | `uint4`, **once** | **REUSED** |
| `w8a16_gemv_dual` | `w8a16_gemv_fused.cu:123` | 1 | `uint4` | n/a (M=1) |
| `w8a16_gemv_silu_input` | `w8a16_gemv_fused.cu:249` | 1 | `uint4` | n/a (M=1) |
| `dense_gemv_bf16` | `dense_gemv_bf16.cu:33` | 1 | `uint4` (8 BF16) | n/a (M=1) |
| `dense_gemv_bf16_fp32out` | `dense_gemv_bf16.cu:120` | 1 | `uint4` | n/a (M=1) |
| `dense_gemv_bf16_batch2` | `dense_gemv_bf16_batch2.cu:32` | 2 | `uint4`, **once** | **REUSED** |
| `dense_gemv_bf16_batchm` | `dense_gemv_bf16_batchm.cu:40` | ≤16 | `uint4`, **once** → `float bf[8]` | **REUSED** |
| `dense_gemv_fp8w` | `dense_gemv_fp8w.cu:131` | 1 | `uint4` (16 FP8) | n/a (M=1) |
| `dense_gemv_fp8w_batch2` | `dense_gemv_fp8w_batch2.cu:72` | 2 | `uint4`, **once** | **REUSED** |

**Tiled GEMMs (M-tile ≥ 16):** weight reuse is by construction — the B tile is
staged in shared memory and consumed by all M rows of the CTA tile. Listed for
completeness: `w4a16_gemm`, `w4a16_gemm_t`, `w8a16_gemm`, `w8a16_gemm_pipelined`,
`w8a16_gemm_t`, `w8a16_gemm_t_pipelined`, `w8a16_gemm_t_m128`, `dense_gemm_bf16*`,
`dense_gemm_tc`, `dense_gemm_splitk_partial`, `fp8_gemm_t_blockscaled`,
`fp8_fp8_gemm_ldmab`, `fp8_gemm_t`, `fp8_fp8_gemm_t`, `w4a16_gemm_t_k64`,
`w4a16_gemm_t_m128`, `fp8_gemm_t_m128`, `fp8_fp8_gemm_t_m128` — all **REUSED
(via smem tile)**. These are *prefill* kernels; **none of them are on the V4-Flash
decode path**, because at M≤6 they pad to a 64- or 128-row MMA tile (10-21×
compute over-provision) and the GEMV family beats them.

---

# Kernel reference

## w4a16_gemv

`kernels/gb10/common/w4a16_gemv.cu:62`

`C[1, N] = A[1, K] · dequant(B_packed[N, K/2])`. NVFP4 W4A16, **not** transposed:
`B_packed[N, K/2]` uint8 (low nibble = W[n, 2j], high = W[n, 2j+1]),
`B_scale[N, K/16]` FP8-E4M3 per group of 16 K, plus scalar FP32 `scale2`.
Effective **0.5625 B/weight**.

- Grid `(ceil(N/4), 1, 1)`, Block `(256, 1, 1)`. `N_PER_BLOCK=4`,
  `threads_per_out = 256/4 = 64` (2 warps per output row).
  Launcher: `crates/spark-model/src/layers/ops/gemm_quant.rs` (`w4a16_gemv`);
  decode call site for the NVFP4 lm_head: `crates/spark-model/src/model/impl_a1.rs:77`.
- Dtypes: A BF16 (`uint4` = 8 BF16 per vector load), B uint8-packed E2M1,
  scale FP8-E4M3 byte, accumulate FP32. No tensor cores — scalar `fmaf`.
- Shared memory: `__shared__ float s_lut[16]` (E2M1 LUT) +
  `__shared__ float smem[N_PER_BLOCK*2]` for the cross-warp reduce. ~96 B.
- Inner loop (`:95`): **two chunks in flight**, `stride2 = threads_per_out*2`.
  Loads `unsigned long long packed8` (8 bytes = **16 weights**) and two `uint4`
  activations (16 BF16). The group scale is factored **out** of the 16-FMA block:
  the code accumulates `part` unscaled then does `acc = fmaf(scale, part, acc)`.
  The header comments this as a direct response to "ncu: 72 % of warp stalls are
  long-scoreboard on GB10".
- Reduction: `__shfl_down_sync` within each 32-lane half, then a 2-entry smem
  cross-warp add, then lane 0 writes `C[n]`.

**Bytes and roofline** — this kernel is M=1 only, so the numbers are the pure
weight stream. For the V4-Flash NVFP4 attention projections
(`wq_b` N=32768 K=1024; `wo_a` 8×(N=1024,K=4096); `wo_b` N=4096 K=8192):

| shape | weight B | scale B | total | t_roof |
|---|---|---|---|---|
| `wq_b` 32768×1024 | 16.78 MB | 2.10 MB | 18.87 MB | 0.069 ms |
| `wo_a` all 8 groups 8192×4096 | 16.78 MB | 2.10 MB | 18.87 MB | 0.069 ms |
| `wo_b` 4096×8192 | 16.78 MB | 2.10 MB | 18.87 MB | 0.069 ms |
| `wq_a` 1024×4096 | 2.10 MB | 0.26 MB | 2.36 MB | 0.009 ms |
| **per-layer NVFP4 attn total** | | | **60.2 MB** | **0.220 ms** |
| **×43 layers** | | | **2.59 GB** | **9.48 ms** |

Activations at M=1 are 0.168 MB/layer = **0.0006 ms** — 0.3 % of the weight cost.

**Arithmetic intensity:** 2 FLOP per 0.5625 weight bytes = **3.56 FLOP/byte**.
The GB10 BF16 FMA peak is ~O(30) TFLOP/s; 273 GB/s × 3.56 = 0.97 TFLOP/s.
**Hard bandwidth-bound** (30× below compute roof) — as long as the loop can
actually keep 273 GB/s of loads in flight, which at 5.3-170 waves it can for
N≥1024. At N=512 (`wkv_a`, 2.7 waves) it is **occupancy/latency-bound**.

**Inefficiencies:** (a) the per-group scale byte is a 1-byte uncoalesced read
per k16 iteration (Finding F4); (b) `E2M1_LUT` is a 16-entry smem table indexed
by data-dependent nibbles — smem handles divergent indices fine, so this is OK,
but it is 16 dependent smem loads per `packed8`; a `prmt`-based bit-math decode
would remove them. (c) no `__launch_bounds__`, so ptxas picks the register count.

## w4a16_gemv_sw

`kernels/gb10/common/w4a16_gemv.cu:226`

Single-warp-per-output variant of the above. `N_PER_BLOCK_SW = 8`,
`threads_per_out = 32`. Each lane keeps **two accumulators** reproducing the
warp-A / warp-B lane partition of the 64-thread kernel, so the reduction order —
and therefore the output — is **bit-identical** to `w4a16_gemv`.
Consequence: **no shared memory for the reduction and no `__syncthreads()`**.
Grid `(ceil(N/8), 1, 1)`, Block `(256,1,1)`.

Same bytes/roofline as `w4a16_gemv`. Doubles the outputs per CTA, halving the
CTA count — which *hurts* at the small-N shapes (`wkv_a` would drop to 64 CTAs =
1.3 waves) and helps at `wq_b`/lm_head where waves are abundant and the barrier
removal is free.

## w4a16_gemv_logits

`kernels/gb10/common/w4a16_gemv.cu:270`

`w4a16_gemv` specialised for the NVFP4 lm_head: output is **FP32 logits**
(`float* C`) rather than BF16, so the argmax/sampler reads full precision.
Launcher `crates/spark-model/src/model/impl_a1.rs:78`.

At V4-Flash `N=129280, K=4096`: weight 264.7 MB + scale 33.1 MB = **297.8 MB →
1.091 ms** per decode step. Grid = 32320 CTAs = **673 waves** — fully
bandwidth-bound, saturates the memory system, no occupancy concern.
Note the V4-Flash lm_head is FP8 in the default config (529.5 MB → **1.940 ms**,
via `dense_gemv_fp8w`), so this NVFP4 arm is only live under
`--lm-head-dtype nvfp4`. **Converting the lm_head to NVFP4 saves 0.85 ms/step.**

## w4a16_gemv_batch2

`kernels/gb10/common/w4a16_gemv.cu:352`

M=2 NVFP4 GEMV. **VERDICT: weight REUSED.** The loop is:

```
for (k8 = lane; k8 < K8; k8 += threads_per_out) {
    unsigned int packed4 = *(const unsigned int*)(B_packed + n*half_K + k8*4);
    ... one scale byte ...
    float wf[8];                       // dequantised ONCE
    uint4 a0 = ((const uint4*)A0)[k8]; // row 0
    uint4 a1 = ((const uint4*)A1)[k8]; // row 1
    acc0 += ... wf[j];                 // both rows consume wf[]
    acc1 += ... wf[j];
}
```

One `uint32` weight load (4 bytes = **8 weights**) per k8 iteration serves both
rows. DRAM weight traffic identical to M=1.

**Bytes at V4-Flash `wo_b` (N=4096, K=8192):** weight 16.78 MB (M-independent) +
activations 2×8192×2 = 32 KB + output 2×4096×2 = 16 KB. Roofline **0.069 ms**
vs 0.069 ms at M=1 — the batching is exactly free on bandwidth.

**Inefficiency (F1):** this kernel is on the **old narrow K8 loop** — 32-bit
weight load, single `uint4` activation, no two-chunks-in-flight. Its M=1 sibling
`w4a16_gemv` was upgraded to `packed8` + dual-chunk; this one was not.
`w4a16_gemv_batchm_impl<4>` (`:470`) already implements the modern loop and
handles M=2 correctly — **route n=2 there and delete this kernel.**

## w4a16_gemv_batch3

`kernels/gb10/common/w4a16_gemv.cu:1092`

Identical structure to `batch2` with a third accumulator/activation row.
**VERDICT: weight REUSED.** Same old-K8-loop inefficiency (F1).

## w4a16_gemv_batch4 / _batch4_ld / _batch8 / _batch8_ld / _batch16

`kernels/gb10/common/w4a16_gemv.cu:569 / 585 / 602 / 616 / 632`

Five thin `extern "C"` wrappers around the **single template**
`w4a16_gemv_batchm_impl<MAX_M>` at `w4a16_gemv.cu:470`. The `_ld` variants take
explicit `lda` / `ldc` row strides for GEMVs over a column slice of a wider
matrix — exactly the V4-Flash block-diagonal `wo_a`
(`lda = q_dim = 32768`, `ldc = latent_dim = 8192`).

**VERDICT: weight REUSED — unambiguously.** The template body:

```cuda
for (unsigned int k16 = lane; k16 < K16; k16 += threads_per_out) {
    unsigned long long packed8 = *(const unsigned long long*)(B_packed + n*half_K + k16*8);
    unsigned char scale_byte  = B_scale[n*num_groups + scale_group];
    float scale = (float)fp8 * scale2;
    float wf[16];
    #pragma unroll
    for (int b = 0; b < 8; b++) {                 // dequantised ONCE
        unsigned char byte_val = (unsigned char)(packed8 >> (b*8));
        wf[b*2]   = s_lut[byte_val & 0xF] * scale;
        wf[b*2+1] = s_lut[byte_val >> 4]  * scale;
    }
    // Reuse the scaled weights across each activation row.
    #pragma unroll
    for (int t = 0; t < MAX_M; t++) {
        if ((unsigned int)t >= M) continue;       // PREDICATED, not `break`
        const __nv_bfloat16* At = A + (unsigned long long)t * lda;
        uint4 a_lo = ((const uint4*)At)[k16*2];
        uint4 a_hi = ((const uint4*)At)[k16*2+1];
        ... acc[t] += ...
    }
}
```

One 64-bit DRAM load = 16 weights, one scale byte, one 16-entry dequant, then
`MAX_M` register-resident MACs. **Weight DRAM traffic is O(1) in M.**

The `if (t >= M) continue;` (rather than `break`) is deliberate and load-bearing:
a data-dependent exit would make `acc[t]` a dynamic index, forcing the register
array to **local memory**. The `#pragma unroll` + predicate keeps `acc[]` in
registers. Register cost: `MAX_M` FP32 accumulators + `wf[16]` + 4 `uint4`
staging = ~40 regs at MAX_M=4, ~52 at MAX_M=16.

**Grid/Block:** `(ceil(N/4), 1, 1)`, `(256,1,1)`.
Launcher: `crates/spark-model/src/layers/ops/gemm_quant.rs` (`w4a16_gemv_batchm`,
`w4a16_gemv_batch4_ld`). Decode call sites:
`crates/spark-model/src/layers/qwen3_attention/trait_impl/multi_seq/mla.rs:540`
(`wo_a`, per group) and `mla.rs:574` (`wo_b`).

**Bytes / roofline at the V4-Flash `wo_a` group shape (N=1024, K=4096):**

| M | weight+scale | activations (BF16) | output | total | t_roof |
|---|---|---|---|---|---|
| 1 | 2.359 MB | 8 KB | 2 KB | 2.369 MB | 0.00868 ms |
| 2 | 2.359 MB | 16 KB | 4 KB | 2.379 MB | 0.00871 ms |
| 6 | 2.359 MB | 48 KB | 12 KB | 2.419 MB | 0.00886 ms |

**M=6 costs 2.1 % more bytes than M=1.** Compute grows 6×
(2·N·K·M FLOP = 50.3 MFLOP at M=6 vs 8.4 at M=1) but 50.3 MFLOP / 2.42 MB =
**20.8 FLOP/byte**, still ~1.5× under the machine balance (273 GB/s × 20.8 =
5.7 TFLOP/s vs a ~30 TFLOP/s FMA roof). **Still bandwidth-bound at M=6.**
This is the single strongest evidence that the batched GEMVs are doing their job.

**Occupancy:** `N=1024 → 256 CTAs = 5.3 waves` — adequate but the tail wave is
6 % of the work. Combined with **8 sequential group launches** (F2) this is the
biggest structural waste on the O-projection.

## w4a16_gemv_qg / w4a16_gemv_qkvz

`kernels/gb10/common/w4a16_gemv.cu:657 / :757`

M=1 fused NVFP4 GEMVs for the GDN/SSM `q,g` and `qkvz` in-projections
(`crates/spark-model/src/layers/qwen3_ssm/init.rs:42,117`). Not on the
V4-Flash MLA decode path (V4-Flash is MLA + MoE, no GDN layers). Both use the
old `K8`/`packed4` loop (F1). Weight reuse: n/a (M=1).

## w4a16_gemv_qg_batch2 / _qg_batch3 / _dual_batch2 / _dual_batch3

`kernels/gb10/common/w4a16_gemv.cu:865 / :1208 / :980 / :1336`

Batched siblings of the above. **VERDICT: weight REUSED** — each loads the
`packed4` chunk once, dequantises to `float wf[8]`, and MACs into 2 (or 3)
accumulator sets. The `dual_*` variants additionally fuse **two projections**
sharing the same activation: the weight for projection 1 and projection 2 are
separate DRAM streams but each is read once for all M rows.
All four are on the old K8 loop (F1). Not on the V4-Flash path.

## w4a16_gemv_dual

`kernels/gb10/common/w4a16_gemv_fused.cu:51`

Fuses gate + up NVFP4 projections into one launch; `blockIdx.z` selects the
projection, both read the same `A[1,K]`. Grid `(ceil(N/4), 1, 2)`, Block `(256,1,1)`.
Halves launch count for the shared-FFN decode path (4 kernels → 2 per layer with
`w4a16_gemv_silu_input`). M=1 only. Old K8 loop.

For the V4-Flash **shared expert** (`moe_intermediate=2048`, `h=4096`, NVFP4):
gate+up = 2×2048×4096×0.5625 = 9.44 MB → **0.0346 ms/layer** → ×43 = **1.49 ms**.

## w4a16_gemv_silu_input

`kernels/gb10/common/w4a16_gemv_fused.cu:156`

Reads `gate_out[K]` and `up_out[K]` BF16, computes `silu(gate)*up` **inline** as
the activation, then GEMVs against the NVFP4 down weights. Eliminates the
separate `silu_mul` kernel and its 2 round-trips of the `K=2048` intermediate.
Grid `(ceil(N/4), 1, 1)`, Block `(256,1,1)`. M=1.

Down projection N=4096, K=2048 NVFP4 = 4.72 MB → **0.0173 ms/layer**, ×43 =
**0.74 ms**. The eliminated `silu_mul` would have been 3×2048×2 = 12 KB — trivial
in bytes but a whole extra launch × 43 layers ≈ 0.2 ms of launch overhead saved.

## w4a16_gemv_dual_sw / w4a16_gemv_silu_input_sw

`kernels/gb10/common/w4a16_gemv_fused.cu:303 / :393`

Single-warp-per-output (`N_PER_BLOCK_SW=8`, `threads_per_out=32`) mirrors of the
two fused kernels, using the same two-accumulator trick as `w4a16_gemv_sw` for
bit-identical output with no smem reduce and no `__syncthreads()`.

## w4a16_gemm

`kernels/gb10/common/w4a16_gemm.cu:87`

**NOT COMPILED for DeepSeek-V4-Flash.** `kernels/gb10/common/w4a16_fp8_ldmab.cu`
documents that "the model-specific `nvfp4/w4a16_gemm.cu` files fully **SHADOW**
`common/w4a16_gemm.cu` (collect_cu_files dedups by stem)". The V4-Flash build
uses `kernels/gb10/deepseek-v4-flash/nvfp4/w4a16_gemm.cu:32` instead.

For reference (this file is the generic/other-model path):
`C[M,N] = A[M,K] · dequant(B_packed[N, K/2])`. `M_TILE=64`, `N_TILE=64`,
`K_STEP=16`, `PAD=2`, `GROUP_SIZE=16`. 128 threads (4 warps × 16 M-rows).
`__shared__ __nv_bfloat16 smem_A[64][18]` + `smem_B[16][66]` ≈ 4.4 KB.
`float acc[8][4]` = 32 registers. Tensor core:
`mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`, 8 N-tiles/warp.
B dequant is **uncoalesced** in this variant: thread `i` reads
`B_packed[gn*half_K + k_pair]` where `gn = idx % N_TILE` varies fastest, so the
64 lanes stride `half_K = K/2` bytes apart — one 32-B sector each.
Weight reuse across M: **REUSED via smem tile** (all 64 M-rows consume smem_B).

## w4a16_gemm_t

`kernels/gb10/common/w4a16_gemm.cu:183`

Transposed-layout twin: `B_packed[K/2, N]`, `B_scale[K/16, N]`. Identical tiling
and MMA; the only change is the global index
`B_packed[k_pair*N + gn]` so **consecutive threads read consecutive N addresses**
→ coalesced 128-B transactions. Also shadowed by the nvfp4 file for V4-Flash.

## w4a16_dequant

`kernels/gb10/common/w4a16_gemm.cu:277`

Standalone NVFP4 → BF16 expansion, `B_packed[N, K/2]` → `B_bf16[N, K]`.
One thread per packed byte → 2 BF16. Grid `ceil(N*K/2 / 256)`, Block 256.
No Rust launcher found — dead code in the current tree (`grep` for
`"w4a16_dequant"` across `crates/` returns nothing).

## w8a16_gemv

`kernels/gb10/common/w8a16_gemv.cu:110`

`C[1,N] = A[1,K] · dequant(B[N,K])` for **block-scaled FP8-E4M3** weights:
`B[N,K]` uint8, `block_scale[N/128, K/128]` FP32. 1 B/weight + 4 B per 128×128
block (0.0002 B/weight) = **1.0002 B/weight**.

- Grid `(ceil(N/4), 1, 1)`, Block `(256,1,1)`, `threads_per_out = 64`.
- `__shared__ float s_lut[256]` — the E4M3 LUT staged from `__constant__` into
  **shared** memory, because constant-cache reads *serialise* across a warp on
  divergent (data-dependent) indices while smem services them in parallel.
  This is a real, documented GB10 win.
- Inner loop: one `uint4` = **16 FP8 weights** + two `uint4` = 16 BF16
  activations per iteration. Scalar `fmaf`, FP32 accumulate. Scale is a single
  FP32 read per 128-K block, hoisted out of the 16-element body.
- Two-stage reduce: `__shfl_down_sync` then 2-entry smem cross-warp.

**Bytes / roofline at the V4-Flash FP8 attention projections (M=1):**

| projection | N | K | weight | scale | t_roof |
|---|---|---|---|---|---|
| `wq_a` | 1024 | 4096 | 4.19 MB | 1.0 KB | 0.0154 ms |
| `wq_b` | 32768 | 1024 | 33.55 MB | 8 KB | 0.1229 ms |
| `wkv_a` | 512 | 4096 | 2.10 MB | 0.5 KB | 0.0077 ms |
| `wo_a` (8 groups) | 8192 | 4096 | 33.55 MB | 8 KB | 0.1229 ms |
| `wo_b` | 4096 | 8192 | 33.55 MB | 8 KB | 0.1229 ms |
| **per layer** | | | **106.98 MB** | | **0.392 ms** |
| **×43** | | | **4.60 GB** | | **16.85 ms** |

**That 16.85 ms is 35 % of the 47.6 ms plain-decode step.** Switching the
attention projections from FP8 to NVFP4 (60.2 MB/layer) drops it to **9.48 ms**,
a **7.4 ms/step saving** — this is the single biggest lever in the dense path
and the `nv4_ok` branch at `mla.rs:531` already implements it. Verify
`ATLAS_V4_ATTN_NVFP4` is on and `wo_a_nvfp4`/`wo_b_nvfp4` are populated.

**AI:** 2 FLOP / 1.0002 B = **2.0 FLOP/byte**. Hard bandwidth-bound.

## w8a16_gemv_batch4 / _batch4_ld / _batch8 / _batch8_ld / _batch16

`kernels/gb10/common/w8a16_gemv_batch4.cu:198 / 216 / 235 / 249 / 266`

Five wrappers over `w8a16_gemv_batchm_impl<MAX_M>` at
`w8a16_gemv_batch4.cu:102`. **VERDICT: weight REUSED.** Structure identical to
the NVFP4 template: one `uint4 b_data` (16 FP8 bytes) load per k16, one
`float wf[16]` dequant via `s_lut[]`, then a predicated `for (t < MAX_M)` loop
of register MACs into `float acc[MAX_M]`.

The file header states the design intent explicitly: this "replaces
`w8a16_gemm_pipelined` for n≤4, which pads M to a 128-row MMA tile
(**32× compute over-provision**)".

- Grid `(ceil(N/4), 1, 1)`, Block `(256,1,1)`.
  Launcher `crates/spark-model/src/layers/ops/fp8_gemv_batch.rs:50` (`_batch4`),
  `:83` (`_batch4_ld`).
- V4-Flash decode call sites (the hot path):
  `mla.rs:397` (`wq_a`: M=n, N=1024, K=4096),
  `mla.rs:~470` (`wq_b`: N=32768, K=1024, when NVFP4 absent),
  `mla.rs:~485` (`wkv_a`: N=512, K=4096),
  `mla.rs:558` (`wo_a` per group, `_ld`: N=1024, K=4096, lda=32768, ldc=8192),
  `mla.rs:588` (`wo_b`: N=4096, K=8192).
- `MAX_M` selection: `mla.rs:73` `batch_gemv_for(n)` picks the `batch4` pair for
  n≤4 and the `batch8` pair for n≤8. **γ=6 verify (n=7) uses `_batch8`.**

**Bytes / roofline, full V4-Flash attention projection set per layer:**

| M | weight+scale | activations+outputs | total | t_roof | ×43 |
|---|---|---|---|---|---|
| 1 | 106.98 MB | 0.168 MB | 107.15 MB | 0.3925 ms | 16.88 ms |
| 2 | 106.98 MB | 0.336 MB | 107.32 MB | 0.3931 ms | 16.90 ms |
| 6 | 106.98 MB | 1.008 MB | 107.99 MB | 0.3956 ms | 17.01 ms |

**M=6 costs 0.8 % more bytes than M=1.** The batched FP8 GEMVs are working as
designed; there is no M-scaling problem here.

**AI at M=6:** 2·106.98e6·6 FLOP / 107.99 MB = **11.9 FLOP/byte** — still
bandwidth-bound (273 × 11.9 = 3.2 TFLOP/s vs ~30 TFLOP/s roof).

**Inefficiencies:** `_batch4_ld` is launched 8× serially for `wo_a` (F2);
the activation stride `lda=32768` (F3).

## w8a16_gemv_dual

`kernels/gb10/common/w8a16_gemv_fused.cu:123`

FP8 mirror of `w4a16_gemv_dual`. `blockIdx.z` picks gate vs up; both read the
same BF16 `A[1,K]`. `__shared__ float s_lut[256]`, `uint4` weight loads,
FP32 block scale hoisted per 128-K. Grid `(ceil(N/4), 1, 2)`, Block `(256,1,1)`.
M=1. Launcher `crates/spark-model/src/layers/ops/moe_prefill.rs:287`, call site
`crates/spark-model/src/layers/dense_ffn.rs:687`.

## w8a16_gemv_silu_input

`kernels/gb10/common/w8a16_gemv_fused.cu:249`

FP8 mirror of `w4a16_gemv_silu_input`. Reads `gate_out` + `up_out` BF16 as
two `uint4` pairs each, computes `silu(g)*u` with `__expf`, MACs against the
`uint4`-loaded FP8 down weights. Grid `(ceil(N/4), 1, 1)`, Block `(256,1,1)`.
Launcher `moe_prefill.rs:324`, call site `dense_ffn.rs:701`.

Note the activation-side cost: it reads **2 BF16 vectors of length K** where the
non-fused version reads 1. At K=2048 that is +8 KB per CTA-row — negligible
against the 4096×2048 = 8.4 MB FP8 weight.

## w8a16_gemm

`kernels/gb10/common/w8a16_gemm.cu:86`

Production non-transposed block-scaled FP8 GEMM. `M_TILE=64`, `N_TILE=64`,
`K_STEP=16`, `PAD=2`, `FP8_BLOCK=128`. Block `(128,1,1)` = 4 warps × 16 M-rows.
Grid `(ceil(N/64), ceil(M/64), 1)`.
`__shared__ __nv_bfloat16 smem_A[64][18]` + `smem_B[16][66]` ≈ 4.4 KB.

**Two-level FP32 accumulation** (the DeepGEMM/vLLM numerics contract):
`float inner_acc[8][4]` accumulates the m16n8k16 BF16 MMA outputs over the 8
K-steps of a 128-K block, with `smem_B` holding **unscaled** BF16-cast E4M3
weights (lossless — E4M3's 3 mantissa bits ⊂ BF16's 7). At each 128-K boundary:
`outer_acc[i][j] += inner_acc[i][j] * scale; inner_acc = 0`. The scale is applied
**once per block on the FP32 accumulator**, never per-element, never folded into
BF16. `n_block = cta_n / 128` is constant per CTA because `N_TILE=64 ≤ 128`.

Weight reuse across M: **REUSED via smem tile**. Prefill only — at M≤6 it pads to
a 64-row tile (10.7× over-provision) and loses to `w8a16_gemv_batch*`.

Header comment measures it at ~5.6 TFLOP/s on large shapes.

## w8a16_dequant

`kernels/gb10/common/w8a16_gemm.cu:224`

FP8 → BF16 expansion, one thread per FP8 byte, block scale applied per element.
No Rust launcher (`grep "w8a16_dequant"` across `crates/` returns nothing) —
dead code.

## w8a16_gemm_pipelined

`kernels/gb10/common/w8a16_gemm_pipelined.cu:174`

The occupancy-tuned rewrite. `PM_M_TILE=128`, **`PM_N_TILE=32`**, `PM_K_STEP=32`
(2 × m16n8k16 sub-MMAs), `PM_A_STRIDE=40`, `PM_STAGES=2`.
Block `(256,1,1)` = 8 warps. Grid `(ceil(N/32), ceil(M/128), 1)`
(`gemm_quant.rs:327`, and `examples/w8a16_microtest.rs:182` confirms the grid).

The `PM_N_TILE` header block is a documented sweep and the most useful tuning
datum in the tree:

| N_TILE | acc regs | regs/thread | CTAs/SM | occupancy | result |
|---|---|---|---|---|---|
| 128 | 128 | 168 | 1 (8 warps) | 12.5 % | baseline, "SM fully stalls on barriers" |
| 64 | 64 | 95 | 2 (16 warps) | 25 % | +45 % |
| **32** | **32** | **56** | **4 (32 warps)** | **50 %** | **+68 % — chosen** |
| 16 | — | 40 | — | — | regresses (B re-stream dominates) |

Measured **~12 TFLOP/s** = 2.1× the 64×64 `w8a16_gemm` (5.6) and +72 % over the
128×128 1-stage draft (7.0).

Three named levers, all in the source:
- **Lever 1** (`:229`): stage the 256-entry E4M3 LUT in `__shared__ float
  smem_lut[256]` — `__constant__` reads *broadcast-serialise* on divergent
  data-dependent indices; smem services them one-transaction-per-bank.
- **Lever 2** (`PM_K_STEP` 16→32): two MMAs per resident K-step halves the
  per-K-step barrier triple (raw-B sync → dequant → smem_B sync → MMA → reuse sync).
- **Lever 3** (`:245`): `smem_B` stored `[n][k]` **K-contiguous**, so the MMA's
  B fragment (two consecutive-K BF16) is a **single aligned u32 smem load**
  instead of two strided 16-bit loads + shift/or.

Pipeline: `cp.async.cg.shared.global [..], 16` with `commit_group` /
`wait_group`. TMA / `cp.async.bulk` explicitly **avoided** — the header states
they "silently corrupt on sm_121". A `cp_async_wait_le()` switch dispatches the
runtime in-flight count to the compile-time-immediate `wait_group`.
smem = 15.5 KB/CTA at 2 stages. The 2/3/4-stage sweep is within noise
(12.1 / 11.85 / 11.93 TFLOP/s) — **the kernel is occupancy/issue-bound, not
global-load-latency-bound.**

Two-level FP32 accumulation preserved exactly from `w8a16_gemm`.

Weight reuse across M: **REUSED via smem tile.** **Not on the decode path** —
at M≤6 it pads to 128 rows = **21× compute over-provision**; this is precisely
what `w8a16_gemv_batch4` was written to replace.

## w8a16_gemm_t

`kernels/gb10/common/w8a16_gemm_t.cu:151`

Transposed block-scaled FP8 GEMM. `B_t[K, N]` (N contiguous) +
`block_scale_t[K/128, N/128]`. `M_TILE=64`, `N_TILE=128` declared but
`cta_n = blockIdx.x * 64` — each CTA actually handles **64 N columns**;
`K_STEP=32` declared but the loop is `k_base += 16`. Block `(128,1,1)`, grid
`(ceil(N/64), ceil(M/64), 1)` (confirmed `examples/w8a16t_microtest.rs:180`).
smem `smem_A[64][18]` + `smem_B[16][66]`.

Key win over `w8a16_gemm`: `B_t[gk*N + gn]` with `gn = idx % 64` varying fastest
→ **coalesced 128-B reads**. Same two-level FP32 accumulation; the scale index is
`block_scale_t[k_block * n_scale_blocks + n_block]`.
Uses its own private `E4M3_LUT_T[256]` in `__constant__` (not the smem staging of
the pipelined variants — a missed Lever-1 application).

Weight reuse: **REUSED via smem tile.** Prefill only.

## w8a16_gemm_t_pipelined

`kernels/gb10/common/w8a16_gemm_t.cu:415`

Transposed twin of `w8a16_gemm_pipelined`, with all three levers plus a
**transpose-on-dequant**: `smem_Braw[stage][PT_K_STEP][PT_N_TILE]` mirrors the
N-contiguous global `B_t` for coalesced `cp.async`, and the dequant step writes
into the K-contiguous `smem_B[stage][PT_N_TILE][PT_K_STEP+PAD]` for the u32 MMA
fragment load. `PT_M_TILE=128`, `PT_N_TILE=32`, `PT_K_STEP=32`, `PT_STAGES=2`,
smem ≈ 26.9 KB + 1 KB LUT. Block `(256,1,1)`,
grid `(ceil(N/32), ceil(M/128), 1)` (`gemm_quant.rs:715`,
`examples/w8a16t_microtest.rs:182`).
Call site: `crates/spark-model/src/layers/qwen3_attention/prefill/paged_oproj.rs:133`.

Weight reuse: **REUSED via smem tile.** Prefill only.

## transpose_fp8

`kernels/gb10/common/w8a16_gemm_t.cu:607`

`B[N,K] → B_t[K,N]`, one thread per byte, no tiling → the **write** is
uncoalesced (stride N). One-time at model load
(`crates/spark-model/src/weight_map/quantized.rs:448`,
`crates/spark-model/src/layers/qwen3_attention/prefill_weights.rs:187`), so the
cost is amortised over the process lifetime. A 32×32 smem tile transpose would
make it ~10× faster if load time ever matters.

## transpose_block_scale

`kernels/gb10/common/w8a16_gemm_t.cu:624`

`scale[N/128, K/128] → scale_t[K/128, N/128]` FP32, one thread per element.
Tiny (at V4-Flash `wo_b`: 32×64 = 2048 floats). Load-time only.

## w8a16_gemm_t_m128

`kernels/gb10/common/w8a16_gemm_t_m128.cu:62`

128 M-rows × 128 N-cols per CTA, split into **two 64-row chunks** (warps 0-3 →
chunk 0, warps 4-7 → chunk 1). `WM128_K_STEP=32`. Transposed `B_t[K,N]` is
loaded **coalesced** into `smem_Braw[2][32][128]` (N-contiguous) then
**transposed on dequant** into K-contiguous `smem_B[2][128][34]`.
smem ≈ 47 KB. `__launch_bounds__(256, 2)`. Block `(256,1,1)`.
Registered at `crates/spark-model/src/layers/dense_ffn.rs:329` and
`crates/spark-model/src/layers/qwen3_attention/init.rs:196`; launcher
`gemm_quant.rs:696`.

Purpose: halve the number of times B is streamed from DRAM during large-M
prefill (8 M-tile groups instead of 16 at M≈1015).

Weight reuse: **REUSED via smem tile.** Prefill only.

## fp8_gemm_t_blockscaled

`kernels/gb10/common/fp8_gemm_t_blockscaled.cu:113`

True **W8A8** with a full FP32 epilogue — vLLM-equivalent block-FP8 numerics:

```
C[M,N] = bf16( Σ_g ( Σ_{k∈g} A_fp8[m,k] · B_fp8[n,k] ) · a_scale[m,g] · b_scale[n/128,g] )
```

`A_fp8[M,K]` per-token-per-128 quantised (from `per_token_group_quant_fp8`),
`a_scale[M, K/128]` FP32, `B_fp8[N,K]`, `b_scale[N/128, K/128]` FP32.

- Tile `M_TILE=64` × `N_TILE_LG=128` × `K_STEP_T=32`, `K_BLOCK=128`
  (`K_STEPS_PER_BLOCK=4`). Block `(128,1,1)` = 4 warps × 16 M-rows.
  Grid `(ceil(N/128), ceil(M/64), 1)`. Launcher `gemm_quant.rs:395`;
  call sites `crates/spark-model/src/layers/moe/forward_prefill_fp8.rs:84,97,138`.
- smem: `smem_Af[2][64][32]` (4 KB) + `smem_Bf[2][128][32]` (8 KB) = 12 KB.
- `float inner_acc[16][4]` + `float outer_acc[16][4]` = **128 accumulator
  registers** — this is exactly the register pressure the `w8a16_gemm_pipelined`
  sweep identified as the occupancy killer (168 regs → 1 CTA/SM). At 128 acc regs
  this kernel is almost certainly at 1-2 CTAs/SM. **A `N_TILE_LG=64` variant
  would likely be a large win for prefill**, mirroring the pipelined sweep.
- MMA: `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` — **native FP8
  tensor cores on sm_121**. Double-buffered `cp.async.ca.shared.global [..],16,pred`
  with byte-count predication (`src_bytes = pred ? 16 : 0`).
- SCALE/HIP portability: `#if defined(__SCALE__)` replaces the e4m3 MMA with two
  m16n8k16 BF16 MMAs, decoding each E4M3 byte via `scl_fp8()` bit math and
  shuffling fragments across lanes (`ATLAS_GA` macro). The `#else` arm is the
  verbatim e4m3 PTX so NVIDIA codegen is byte-identical.

Weight reuse: **REUSED via smem tile.** MoE prefill only.

**AI:** at M=64 tile, 2·64·128·32 FLOP per (64·32 + 128·32) B loaded per K-step =
**87 FLOP/byte** — compute-bound territory; the limiter is occupancy, not DRAM.

## fp8_fp8_gemm_ldmab

`kernels/gb10/common/w4a16_fp8_ldmab.cu:53`

The fastest FP8 prefill GEMM in the tree. `__launch_bounds__(256, 2)`,
`__shared__ unsigned char smem_Ai[2][128][32]` + `smem_Bi[2][128][32]` (8 KB),
`float acc[16][4]`. Grid `(ceil(N/128), ceil(M/128), 1)`, Block `(256,1,1)`.
Double-buffered `cp.async.ca` with `LABF_LOADS` / `LABF_COMPUTE` macros.

The distinguishing feature: **`ldmatrix.sync.aligned.m8n8.x4.b16` for BOTH A and
B** fragments, feeding `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`.
Header: "**ncu-proven 2.1× over the scalar-load `fp8_gemm_t`**, cosine 1.000000
vs `fp8_fp8_gemm_t`".

Note the contradiction worth recording: `dense_gemm_tc.cu` claims `ldmatrix` is
**broken on sm_121**; this kernel uses it successfully. The `dense_gemm_tc`
comment is stale or describes a different (b16 vs b8 / swizzle) usage.

The header also carries the build-system fact that explains the whole kernel
layout: it lives in `common/` because "the model-specific `nvfp4/w4a16_gemm.cu`
files fully **SHADOW** `common/w4a16_gemm.cu` (collect_cu_files dedups by stem)".

Launcher `crates/spark-model/src/layers/ops/gemm_fp8_prefill.rs:47`.
Weight reuse: **REUSED via smem tile.** Prefill only.

## dense_gemm_bf16

`kernels/gb10/common/dense_gemm_bf16.cu:26`

Baseline BF16 tiled GEMM, `C[M,N] = A[M,K] · B[N,K]^T`. smem A + B tiles,
scalar/MMA path per the tile config. Not on the V4-Flash decode path;
used by `crates/spark-model/src/layers/dspark_head.rs:192` for the drafter.
Weight reuse: **REUSED via smem tile.**

## dense_gemm_bf16_f32out

`kernels/gb10/common/dense_gemm_bf16.cu:85`

Same as above with an FP32 `C`. Used where the consumer needs full precision
(logits/routing).

## dense_gemm_f32in_f32out

`kernels/gb10/common/dense_gemm_bf16.cu:131`

FP32 in, FP32 out. Registered at
`crates/spark-model/src/layers/moe/init.rs:82` (`dense_gemm_f32in`) — the
FP32 routing/gate path (`ATLAS_FP32_ROUTING` / `ATLAS_FP32_GATE`).
At V4-Flash the router is 144×4096 FP32 = 2.36 MB → 0.0086 ms/layer,
×43 = **0.37 ms/step**. Halving it to BF16 saves ~0.19 ms but changes routing
numerics — not recommended.

## dense_gemm_bf16_pipelined

`kernels/gb10/common/dense_gemm_bf16.cu:331`

`cp.async` multi-stage BF16 GEMM, the production large-M dense path.
Call sites `crates/spark-model/src/layers/dflash_head/forward_block_layer_paged.rs:233,943,988`.
Weight reuse: **REUSED via smem tile.**

## fused_silu_mul

`kernels/gb10/common/dense_gemm_bf16.cu:475`

Elementwise `silu(gate) * up` → BF16. Pure bandwidth: 3 × M·I × 2 B.
At the V4-Flash shared expert (I=2048, M=1): 12 KB → 0.045 µs. Launch-bound.
Superseded on the GEMV path by the `*_silu_input` fused kernels.

## dense_gemm_bf16_mtile16

`kernels/gb10/common/dense_gemm_bf16_mtile16.cu:77`

Small-M (M ≤ 16) BF16 tensor-core GEMM: `N_TILE=64`, ring-buffered `cp.async`.
Written for the DFlash/DSpark drafter head where M = γ+1.
Launcher `crates/spark-model/src/layers/ops/gemm_dense.rs:149`;
registered `crates/spark-model/src/layers/dflash_head/from_weights.rs:127`.

Weight reuse: **REUSED via smem tile** — but note it still runs a **16-row MMA
tile**: at M=6 that is 2.7× compute over-provision (much better than the 128-row
kernels' 21×, which is the whole point).

## dense_gemm_bf16_mtile16_n128

`kernels/gb10/common/dense_gemm_bf16_mtile16.cu:222`

Wide-stream variant: `W128_N_TILE=128`, 8 warps (block 256), `W128_K_STEP=32`,
**`W128_STAGES=4`** ring, `W128_STRIDE=40`.
smem = A 4×16×40×2 = 5120 B + B 4×128×40×2 = 40960 B = **46080 B/CTA →
2 CTAs/SM**. Grid `(ceil(N/128), 1, 1)`, Block `(256,1,1)`.

The header states the LPDDR5X-specific rationale precisely, and it generalises
to every kernel in this document:

> "on LPDDR5x, MANY concurrent 64-row B streams (96+ CTAs) thrash DRAM page
> locality; a 128-row slice per CTA halves the number of concurrent streams and
> doubles each stream's contiguity"

`float acc[2][4]` only (16 rows → 1 MMA m16 tile, 2 N-subtiles/warp), so register
pressure is low and the smem tile is the occupancy limiter.
Same ascending-K m16n8k16 chain → **bit-identical output** to the 64-wide variant.
Launcher `gemm_dense.rs:186`; call site `crates/spark-model/src/layers/dspark_head.rs:968`.

**This DRAM-page-locality argument is worth testing on the GEMV family.** The
GEMVs launch `ceil(N/4)` CTAs — 8192 concurrent streams for `wq_b`. If page
thrashing is real at that count, raising `N_PER_BLOCK` from 4 to 8 or 16 for the
large-N projections would cut the stream count 2-4× at the cost of fewer waves.
`w4a16_gemv_sw` (`N_PER_BLOCK_SW=8`) already exists to test this.

## dense_gemm_splitk_partial

`kernels/gb10/common/dense_gemm_splitk.cu:27`

K-split GEMM: each CTA computes a partial sum over a `K/split_k` slice into an
FP32 workspace `[split_k, M, N]`. Grid carries the split factor in `blockIdx.z`.
Registered `crates/spark-model/src/layers/qwen3_attention/init.rs:439`.

**This is the existing machinery that Finding F5 wants applied to the low-N
GEMVs** (`wkv_a` N=512 → 2.7 waves). A split-K GEMV would turn 128 CTAs into
128×split_k CTAs.

Weight reuse: **REUSED via smem tile** within each K-slice; across slices the
weight is *partitioned*, not duplicated, so total DRAM traffic is unchanged.

## dense_gemm_splitk_reduce

`kernels/gb10/common/dense_gemm_splitk.cu:84`

Sums the `[split_k, M, N]` FP32 workspace down to `[M, N]` BF16.
Extra traffic: `split_k · M · N · 4` read + `M·N·2` write.
At `wkv_a` (N=512, M=6, split_k=8): 98 KB — 0.36 µs, negligible against the
2.1 MB weight stream. **The split-K trade is clearly favourable at low N.**
Registered `qwen3_attention/init.rs:444`.

## dense_gemm_tc

`kernels/gb10/common/dense_gemm_tc.cu:23`

Pure BF16 tensor-core GEMM with `ldmatrix` fragment loads.
Grid `(ceil(N/64), ceil(M/16), 1)`, Block `(128,1,1)`
(confirmed `examples/dense_gemm_microtest.rs:154`).
Launcher `crates/spark-model/src/layers/ops/lora_delta.rs:180,191` — the LoRA
delta path. Also the documented fallback when `m > DENSE_GEMV_BATCHM_MAX_M` (16).

Contains the "`ldmatrix` broken on sm_121" note that `w4a16_fp8_ldmab.cu`
contradicts — see that section.

Weight reuse: **REUSED via smem tile.**

## dense_gemv_bf16

`kernels/gb10/common/dense_gemv_bf16.cu:33`

Plain BF16 GEMV, `C[1,N] = A[1,K] · B[N,K]^T`. `BLOCK_SIZE=256`,
`N_PER_BLOCK=4`, `threads_per_out=64`. `uint4` (8 BF16) vectorised weight loads.
Grid `(ceil(N/4),1,1)`, Block `(256,1,1)` (`gemm_quant.rs:110`).
Registered `crates/spark-model/src/model/impl_a1.rs:72`; also used by
`crates/spark-model/src/layers/dspark_head.rs:193`.

**Documentation bug (F6):** the comment at `dense_gemv_bf16.cu:40-41` says
`threads_per_out = 32` and "which of 8 outputs"; the code computes
`256/4 = 64` over **4** outputs. The same wrong text is copied into
`gemm_quant.rs:108`.

**Bytes:** 2 B/weight. At the V4-Flash `wo_b` shape (4096×8192) that is
67.1 MB → **0.246 ms** — i.e. **2× the FP8 cost and 3.6× the NVFP4 cost**.
`crates/spark-model/src/weight_loader/laguna/load_layers.rs:445` notes that
"o_proj is kept BF16: MEASURED, o-NVFP4's `w4a16_gemv_batch4/batch8` arms are…"
— worth re-measuring for V4-Flash given F1 (the batch2/batch3 arms are on the
old loop; the batch4/batch8 arms are not, so an old measurement may be stale).

**AI:** 1.0 FLOP/byte. Hardest bandwidth-bound kernel in the set.

## dense_gemv_bf16_fp32out

`kernels/gb10/common/dense_gemv_bf16.cu:120`

Same with FP32 output. Registered at `impl_a1.rs:73` as
`dense_gemv_fp32out_kernel` but the source comment states it is
**permanently `KernelHandle(0)`**: "the FP32 logits path required an FP32
residual stream, which no longer exists, so this stays KernelHandle(0) and the
BF16 path is always taken." **Dead on the V4-Flash path.**

## dense_gemv_bf16_batch2

`kernels/gb10/common/dense_gemv_bf16_batch2.cu:32`

**VERDICT: weight REUSED.** One `uint4` (8 BF16 weights) load per iteration,
unpacked once, MACed into `acc0` and `acc1`. Bit-identical to two `dense_gemv`
calls (the per-row K-iteration and reduction order are unchanged).
Grid `(ceil(N/4),1,1)`, Block `(256,1,1)`, `out_stride` argument
(`gemm_quant.rs:189`). Written for the K=2 MTP verify GDN `in_proj_qkvz`.

**Bytes at `wo_b` (N=4096, K=8192):** weight 67.1 MB (M-independent) +
2×8192×2 activations + 2×4096×2 output = 67.15 MB → **0.246 ms**, vs 0.246 ms at
M=1. Free batching.

## dense_gemv_bf16_batchm

`kernels/gb10/common/dense_gemv_bf16_batchm.cu:40`

`template <int MAX_M>` with `MAX_M = 16` (mirrored by
`pub const DENSE_GEMV_BATCHM_MAX_M: u32 = 16;` at `gemm_quant.rs:177`).
**VERDICT: weight REUSED.** The weight unpack is **hoisted above the M loop**
into `float bf[8]`; the M loop then only reads activations and MACs.

Two deliberate details worth preserving:
- It uses an `active` flag with **no early `return`**, so every thread reaches
  `__syncthreads()` — a divergent return would deadlock the cross-warp reduce.
- The M loop uses `#pragma unroll` + predication, **not `break`** — the source
  comment explains a data-dependent exit makes `acc[t]` a dynamic index and
  spills the accumulator array to local memory.

Grid `(ceil(N/4),1,1)`, Block `(256,1,1)`, with explicit `a_stride`/`out_stride`
(`gemm_quant.rs:145-175`). The doc-comment states the motivating shape exactly:
"at the DFlash verify shape (M = gamma+1 = 7, N = num_heads = 48) [the prefill
tensor-core GEMM] launches a single CTA and drags the whole weight through one SM."

**That N=48 case is worth flagging:** `ceil(48/4) = 12 CTAs` on 48 SMs =
**0.25 waves** — 75 % of the GPU idle. For the DSpark drafter head this GEMV is
latency-bound, not bandwidth-bound, and split-K (F5) is the fix.

## dense_gemv_fp8w

`kernels/gb10/common/dense_gemv_fp8w.cu:131`

**Per-row** (not block) FP8: `B[N,K]` E4M3 + `row_scale[N]` FP32, scale applied
**after** the full-K accumulation (`result = acc * row_scale[n]`), so there is
no in-loop scale traffic at all — the cleanest inner loop of the FP8 family.
`uint4` = 16 FP8 weights + 2 × `uint4` = 16 BF16 activations per iteration.
`decode4_fp8` unpacks a `uint32` into 4 floats via `__nv_fp8_e4m3` casts (or
`scl_fp8()` bit math under `__SCALE__`/`__HIP_PLATFORM_AMD__`, because SCALE's
built-in narrow-format cast is non-standard on gfx1151).
Grid `(ceil(N/4),1,1)`, Block `(256,1,1)` (`gemm_quant.rs:~215`).
Registered `crates/spark-model/src/model/impl_a1.rs:88` — **this is the
V4-Flash lm_head kernel** (`--lm-head-dtype fp8`, the config default).

**Bytes at the lm_head (N=129280, K=4096):** 529.5 MB weight + 517 KB row_scale
+ 8 KB activation → **1.942 ms per decode step**, 4.1 % of the 47.6 ms step.
Grid = 32320 CTAs = **673 waves** — perfectly bandwidth-bound.

**AI:** 2.0 FLOP/byte.

## quantize_bf16_to_fp8

`kernels/gb10/common/dense_gemv_fp8w.cu:65`

BF16 → per-row-scaled FP8 E4M3 quantiser (computes the row absmax then encodes).
Uses `scl_enc_fp8()` bit-math encode under `__SCALE__`. Load-time / calibration
only; not on the decode path.

## dense_gemv_fp8w_batch2

`kernels/gb10/common/dense_gemv_fp8w_batch2.cu:72`

**VERDICT: weight REUSED.** Header states it explicitly: "each block streams an
expert/output column's weight row **ONCE** and applies it to both activation
rows, **halving FP8 weight bandwidth** vs two separate M=1 GEMV launches", and
"the result is bit-identical to running `dense_gemv_fp8w` twice (the per-token
reduction order is unchanged)".

Structure: `decode4_fp8(w32, f0..f3)` unpacks once per `uint32`, then
`mac4(acc0, ...)` and `mac4(acc1, ...)` consume the same `wf0..wf3`.
`BLOCK_SIZE=256`, `N_PER_BLOCK=4`, `VEC_SIZE=16`.
Grid `(ceil(N/4),1,1)`, Block `(256,1,1)` (`fp8_gemv_batch.rs:32`).
Registered `crates/spark-model/src/model/impl_a1.rs:~92`.

**Bytes at the lm_head (N=129280, K=4096), M=2:** 529.5 MB + 517 KB + 16 KB
activation + 517 KB output = 530.5 MB → **1.943 ms** vs 1.942 ms at M=1.
**Batching the verify lm_head is essentially free** — it saves a full
1.94 ms/step over two serial M=1 launches.

**Gap:** there is no `dense_gemv_fp8w_batchm` (M≤8). At γ=6 (n=7) the lm_head
must either fall back to a per-token loop (7 × 1.94 = **13.6 ms**) or use one of
the `fp8_gemm_row_scaled_mtile8/m16` tiled kernels registered at
`impl_a1.rs:~100`. **Writing a `dense_gemv_fp8w_batchm<8>` — a 30-line copy of
`w8a16_gemv_batchm_impl` with the row-scale epilogue — would be worth
~11.7 ms/step at γ=6 if the current path is the per-token loop.** This is
potentially the largest single win in this document; verify which path γ=6 takes
first.

---

## nvfp4/w4a16_gemm.cu — the V4-Flash-specific module

`kernels/gb10/deepseek-v4-flash/nvfp4/w4a16_gemm.cu` (1399 lines).
**This file SHADOWS `common/w4a16_gemm.cu`** (the build's `collect_cu_files`
dedups by file stem), so for DeepSeek-V4-Flash these are the `w4a16_gemm` /
`w4a16_gemm_t` symbols actually loaded.

### w4a16_gemm (nvfp4)

`nvfp4/w4a16_gemm.cu:32`. cp.async-pipelined NVFP4 dequant + BF16 MMA.
smem per the header at `:142-146`: A 2×64×40×2 = 10240 B, Bp 2×16×144 = 4608 B,
Bs 2×2×144 = 576 B, LUT 64 B ≈ **15.5 KB → ~6 CTAs/SM** (register-limited).
Uses `prmt` for BF16 packing and a `BP_PAD` bank-conflict fix.
Registered `crates/spark-model/src/model/impl_a1.rs:79`.
Weight reuse: **REUSED via smem tile.**

### w4a16_gemm_t (nvfp4)

`nvfp4/w4a16_gemm.cu:206`. Transposed NVFP4 layout, `K_STEP_T=32`,
`N_TILE_LG=128`, `M_TILE=64`. smem ≈ 19.6 KB (header `:184`).
Grid `(ceil(N/128), ceil(M/64), 1)`, Block `(128,1,1)`.
Weight reuse: **REUSED via smem tile.**

### fp8_gemm_t (nvfp4)

`nvfp4/w4a16_gemm.cu:371`. BF16 A × FP8 B. The A fragments are converted
**on the fly in the MMA macro** via `bf16x4_to_e4m3x4(&sA[...])`, feeding
`mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32`. Double-buffered
`cp.async` with the `LOAD(nxt) ‖ COMPUTE(cur) → wait → sync` structure.
`float acc[16][4]` = 64 acc registers, 16 N-subtiles per warp.
Weight reuse: **REUSED via smem tile.**

### predequant_nvfp4_to_fp8

`nvfp4/w4a16_gemm.cu:490`. One-time NVFP4 → FP8 E4M3 conversion at model load.
One thread per packed byte → 2 FP8 values via
`cvt.rn.satfinite.e4m3x2.f32`. Grid `ceil(N*K/2 / 256)`, Block 256.
Launcher `crates/spark-model/src/layers/ops/gemm_fp8_prefill.rs:166`;
call sites `weight_map/quantized.rs:310`, `qwen3_ssm/init.rs:360`,
`qwen3_attention/prefill_weights.rs:228`, `moe/helpers_c.rs:21`.

**Note the memory cost:** this *doubles* the weight footprint (0.5625 → 1.0
B/weight) for whatever it converts, and on a 120 GB machine holding a 162B model
that matters. It exists so prefill can use the fast FP8 MMA path; **decode should
never touch the pre-dequantised copy** (it would pay 1.78× the DRAM traffic).

### bf16_to_fp8

`nvfp4/w4a16_gemm.cu:529`. BF16 → FP8 E4M3 activation conversion, 2 elements per
thread via `cvt.rn.satfinite.e4m3x2.f32`. Grid `ceil(M*K/2 / 256)`, Block 256.
Prefill only (decode activations stay BF16 for the GEMV path).

### fp8_fp8_gemm_t

`nvfp4/w4a16_gemm.cu:560`. Pure FP8 × FP8 — **no conversion in the inner loop**.
`smem_Af[2][64][32]` (2 KB/buf, half the BF16 variant) + `smem_Bf[2][128][32]`.
`M_TILE=64`, `N_TILE_LG=128`, `K_STEP_T=32`. Grid `(ceil(N/128), ceil(M/64))`,
Block `(128,1,1)`. `float acc[16][4]`.
Weight reuse: **REUSED via smem tile.**

### w4a16_gemm_t_k64

`nvfp4/w4a16_gemm.cu:684`. `K_STEP_T64=64`, `PAD_T64=8` (144 B rows, 16-B
aligned). **Halves the outer K-loop** — 32 iterations instead of 64 at K=2048 —
with two m16n8k32 MMAs per N-tile per step. K must be divisible by 64.
smem: A 2×64×72×2 = 18432 B, Bp 2×32×144 = 9216 B, Bs 2×4×144 = 1152 B,
B_fp8 128×80 = 10240 B, LUT 64 B ≈ **38.4 KB**. `B_fp8` row stride 80 = 64+16
avoids 4-way bank conflicts.
Registered `qwen3_attention/init.rs:509`, `qwen3_ssm/init.rs:109`,
`dense_ffn.rs:314`.
Weight reuse: **REUSED via smem tile.**

### w4a16_gemm_t_m128

`nvfp4/w4a16_gemm.cu:900`, `__launch_bounds__(128, 3)`, Block `(128,1,1)`,
Grid `(ceil(N/128), ceil(M/128), 1)`. Two consecutive 64-row M-chunks per CTA
(`acc0[16][4]` + `acc1[16][4]` = 128 acc registers, hence the 3-CTA bound).

The header quantifies the whole point of M-tiling, and it is the clearest
statement of the weight-re-stream economics in the tree:

> For large-M prefill (ISL=1016, N=12288):
> `M_TILE=64`: grid=(96,16,1)=1536 blocks, **16 weight re-reads → 227 MB B DRAM**
> `M_TILE2=128`: grid=(96,8,1)=768 blocks, **8 weight re-reads → 114 MB B DRAM**

smem ≈ 29.8 KB → 3 blocks/SM. "~2× speedup at ISL>128 vs `w4a16_gemm_t`" for qkvz.
Registered `nemotron_moe.rs:186`, `nemotron_mamba2.rs:111`.
Weight reuse: **REUSED via smem tile**, and *doubly* so across the two M-chunks.

### fp8_gemm_t_m128

`nvfp4/w4a16_gemm.cu:1106`, `__launch_bounds__(128, 3)`. M128 variant of
`fp8_gemm_t` (BF16 A × FP8 B). smem: A 2×128×40×2 = 20480 B + B 2×128×32 = 8192 B
≈ 28.7 KB → 3 blocks/SM. Grid `(ceil(N/128), ceil(M/128), 1)`, Block `(128,1,1)`.
Purpose (header): for `out_proj` (K=2048, N=2048) and paged Q/K/V, "halves the
number of times B is read from DRAM (8 m-tile groups vs 16 at M=1015)".
Registered `qwen3_ssm/init.rs:236`, `qwen3_attention/init.rs:697`.

### fp8_fp8_gemm_t_m128

`nvfp4/w4a16_gemm.cu:1260`, `__launch_bounds__(128, 3)`. M128 FP8×FP8.
smem: Af 2×128×32 = 8192 B + Bf 2×128×32 = 8192 B ≈ 16 KB, 3 blocks → 48 KB/SM.
Header notes the register reasoning explicitly: "dual acc0+acc1 need ~145
regs/thread; 3 blocks allows 170 regs/thread" — i.e. the occupancy bound is
chosen to *avoid spilling*, not to maximise warps.
Registered `qwen3_attention/init.rs:698`.

---

## Summary: where the 47.6 ms plain-decode step goes (dense GEMM/GEMV only)

| component | kernel | bytes/step | t_roof | share |
|---|---|---|---|---|
| MLA attention projections (FP8) | `w8a16_gemv*` | 4.60 GB | 16.85 ms | 35.4 % |
| MoE routed experts (NVFP4, 6/token) | `w4a16_gemv*` | 3.65 GB | 13.38 ms | 28.1 % |
| lm_head (FP8) | `dense_gemv_fp8w` | 530 MB | 1.94 ms | 4.1 % |
| MoE shared expert (NVFP4) | `w4a16_gemv_dual` + `_silu_input` | 609 MB | 2.23 ms | 4.7 % |
| router (FP32) | `dense_gemm_f32in_f32out` | 101 MB | 0.37 ms | 0.8 % |
| **dense GEMM/GEMV subtotal** | | **9.49 GB** | **34.77 ms** | **73.0 %** |
| remainder (attention kernels, norms, KV cache, elementwise, launch overhead) | | | 12.8 ms | 27.0 % |

The dense path is **73 % of the decode step and is running at ~100 % of its
roofline** given the current weight dtypes. The two levers that actually move it:

1. **Attention projections FP8 → NVFP4**: 16.85 → 9.48 ms = **−7.4 ms/step**
   (already implemented behind `nv4_ok` at `mla.rs:531`; verify it is live).
2. **lm_head FP8 → NVFP4**: 1.94 → 1.09 ms = **−0.85 ms/step**
   (`w4a16_gemv_logits` already exists).

Everything else in this document is worth 0.1-1.5 ms each. There is no
weight-re-streaming bug to fix in the batched GEMVs — they are already optimal
on bandwidth.
