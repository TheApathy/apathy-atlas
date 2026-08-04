# 04 — Elementwise, Norm, Cache and mHC Kernels

Target: DeepSeek-V4-Flash-162B, one NVIDIA GB10 (DGX Spark), sm_121, **273 GB/s** LPDDR5X, 120 GB, **48 SMs**.
Model: 43 layers, `hidden = 4096`, `vocab = 129280`, MLA compressed KV = 576, KV cache FP8, `hc_mult = 4`, `hc_sinkhorn_iters = 20`.
Decode measured at ~47.6 ms/token ⇒ ~1.1 ms per layer. CUDA graphs measured to save ~0%, so **launch count is not the bottleneck — memory passes are.**

Roofline constants used throughout: **1 MiB = 3.84 µs**, **1 KiB = 3.75 ns**, **1 B = 3.66 ps** at 273 GB/s.

---

## Findings

The scope of this document is the elementwise / norm / cache / quantise / mHC kernel surface. Reading all of it against the real launchers gives five load-bearing conclusions.

**1. mHC dominates this entire category, and it does so through one matrix.** Every HC site streams `hc_fn`, a `[mix_hc = 24, hc*H = 16384]` FP32 matrix = **1.5 MiB**. There are two HC sites per layer (pre-attn, pre-ffn) × 43 layers = **86 sites per decode step** ⇒ **129 MiB of weight traffic per token**, or **0.495 ms/step at the 273 GB/s roofline**. That is the single largest number in this document and it is pure weight streaming that no other kernel in this category comes close to. The kernel source states this itself at `hyper_connection.cu:192-211`, including the measurement that in the *fused* one-block form it runs at **~8 GB/s** (single-SM bandwidth), i.e. **tens of ms per token**. The split path (`hc_pre_mix` + `hc_pre_finish`) is the fix and is on by default; `ATLAS_HC_SPLIT=0` restores the pathology for A/B.

**2. The FP32 mHC highway doubles all residual traffic.** `hc_streams` is `[T, 4, 4096]` **FP32** (`sizes.rs:448-487`), not BF16 — 64 KiB at T=1 instead of 32 KiB. It is read+written by every `hc_post` (86/step) and read again by every `hc_pre_mix` (86/step). That is **~16.5 MiB/step ≈ 63 µs** of highway traffic that would be 31 µs in BF16. BF16 was tried and collapsed generation, so this is a correctness-constrained cost, not a bug — but it means any further HC work should minimise *round trips*, not element width.

**3. There is a live asymmetry bug in `hc_post` sharding.** `decode_inner.rs:505-509` computes `post_shards = 16` and the post-FFN site at `:757-772` correctly uses `hc_post_sharded(..., post_shards)`. But the post-attention site at **`decode_inner.rs:664-677` calls the unsharded `ops::hc_post(...)`**, which hard-codes `shards = 1` (`ops/hyper_connection.rs:152-166` → `:189-199`). At decode T=1 that is a grid of **(1,1,1)** — one block, one SM out of 48 — moving 64 KiB of FP32 streams + 8 KiB hidden. The same unsharded call appears three times on the verify path: **`multi_seq/mod.rs:355`, `:490`, `:543`**. Fixing this is a one-line change per site with no numerical consequence (`hc_post` is a pure per-`d` map; sharding over `gridDim.y` is already implemented and used by the sibling site).

**4. Grid underfill is systemic at decode, not incidental.** Every one-block-per-token kernel here runs on 1/25 of the GPU when T=1: `rms_norm` (grid `(1,1,1)`, block `(1024,1,1)` — `ops/norm.rs:33-41`), `argmax_bf16` (grid `(1,1,1)` over the whole **129280**-entry vocab — `argmax_bf16.cu:14`), `hc_head` (`ops/hyper_connection.rs:220-232`), `hc_mean` (`:249-256`), `hc_expand` (`:25-32`). `rms_norm` fires ~86×/step at 8 KiB in + 8 KiB out + 8 KiB weight = ~2.06 MiB/step ≈ 7.9 µs at roofline, but at single-SM bandwidth it is closer to 250 µs. The short-row warp fast path never engages: `rms_norm_short_row_eligible` requires `hidden_size <= 256 && num_rows >= 1024` (`ops/norm.rs:75-80`) and H=4096, T=1 fails both.

**5. Sinkhorn is 20 sequential iterations executed by a single thread.** `hyper_connection.cu:114-181` (`hc_pre`) and the duplicated block at `:317-379` (`hc_pre_finish`) run the whole normalisation inside `if (tid == 0)`. Per site that is 20 × (4 row sums + 4 row divides + 4 col sums + 4 col divides over a 4×4) ≈ **20 × 64 serial FLOPs plus 20 × 32 divides**, with 255 of 256 threads parked at `__syncthreads()`. It happens **86× per decode step**. It is latency, not bandwidth — but it is unhidden latency on the critical path of every layer, twice. Note `hyper_connection.cu:168-171`: dropping the final exact column projection was A/B-tested (portv4b11) and **regressed coherence onset from ~150 to ~90 tokens**, so the last half-iteration must stay. Reducing `sinkhorn_iters` is the only safe lever and it is a numerics question, not a code question.

Secondary findings: `residual_add.cu` is 11 kernels of **entirely scalar BF16** loads (1 element/thread) where `uint2`/`uint4` would give 4-8× fewer transactions; `quant_rowwise_fp8.cu:38` and `per_token_group_quant_fp8.cu:39` both **read the activation twice** (absmax pass, then quantise pass) instead of caching the row in registers; `kv_cache_append.cu:20` and `e2m1_branchless.cu:48` are non-vectorised; and `rope_forward_proportional`, `rope_forward_mrope_interleaved*` and `fused_k_norm_rope_cache_*` still do a per-thread **FP64 `pow()`**, which runs at 1/64 rate on SM121 (the `rope_forward` comment at `rope.cu:29` records this leaving that kernel ~13× above its bandwidth floor before the shared-memory freq cache was added).

---

# mHC — `kernels/gb10/deepseek-v4-flash/nvfp4/hyper_connection.cu`

Constants (`:18-24`): `HC_BLOCK 256`, `HC_MAX_MULT 4`, `HC_MAX_MIX 24`, `HC_MIX_BLOCK 512`. With `hc = 4`: `mix_hc = (2 + hc) * hc = 24`, `hc_dim = hc * H = 16384`.

Shapes per token at decode (T=1):

| Buffer | Shape | dtype | Bytes |
|---|---|---|---|
| `hc_streams` | `[1, 4, 4096]` | **FP32** | 65536 (64 KiB) |
| `hidden` / `x` | `[1, 4096]` | BF16 | 8192 (8 KiB) |
| `hc_fn` | `[24, 16384]` | FP32 | **1572864 (1.5 MiB)** |
| `hc_scale`, `hc_base` | `[24]` | FP32 | 96 each |
| `hc_mix` (`mix_out`) | `[1, 25]` | FP32 | 100 |
| `hc_post` (p) | `[1, 4]` | FP32 | 16 |
| `hc_comb` | `[1, 4, 4]` | FP32 | 64 |

## hc_block_reduce

`hyper_connection.cu:27` — device helper, not a `__global__`. Warp-shuffle `__shfl_down_sync` within each warp, then a `__shared__` array indexed by warp id, then a final warp reduce. Standard two-stage. Used by `hc_pre`, `hc_pre_mix`, `hc_pre_finish`, `hc_mean`.

## hc_expand

`hyper_connection.cu:38`. Broadcasts the BF16 hidden state into all `hc` FP32 stream planes at the start of the model — `streams[i*H + d] = (float)x[d]` for every `i < hc`. Launched once per forward from `ops/hyper_connection.rs:25-32` with grid `[num_tokens,1,1]`, block `[256,1,1]`.

- dtypes: in BF16, out FP32. No shared memory, no reduction.
- Bytes/call M=1: read 8 KiB + write 64 KiB = **72 KiB ⇒ 0.27 µs**. M=2: 144 KiB ⇒ 0.54 µs. M=6: 432 KiB ⇒ 1.62 µs.
- Fires **1× per decode step** (model entry, not per layer). **Per-step: 0.27 µs.** Negligible.
- Inefficiency: scalar `float` stores, not `float4`; grid `(1,1,1)` at decode so one SM. Both irrelevant at 0.27 µs.

## hc_pre

`hyper_connection.cu:58`. The **fused** HC-pre path, used only when `ATLAS_HC_SPLIT=0`. One block per token. Pass 1 computes `sum(x^2)` over `hc_dim = 16384` stream elements for the RMS. Pass 2 loops `m = 0..mix_hc` and, for each of the 24 rows, streams the entire `hc_fn` row (`fn_row = hc_fn + m*hc_dim`, 64 KiB) and dot-products it against the normalised streams — **the whole 1.5 MiB matrix through one block**. Then `tid == 0` builds the 4×4 `comb` from the mix vector and runs 20 Sinkhorn iterations in registers. Finally it collapses the streams into the BF16 `y_out` using the width-`hc` weights.

Launcher `ops/hyper_connection.rs:57-72`: grid `[num_tokens,1,1]`, block `[256,1,1]`. **At decode that is grid (1,1,1).**

- dtypes: streams FP32, `hc_fn`/`hc_scale`/`hc_base` FP32, `y_out` BF16, `comb`/`post` FP32.
- Reduction: `hc_block_reduce` (warp shuffle + shared) for pass 1 and each of the 24 dot products. Sinkhorn: **single thread, no reduction, 20 sequential iterations** (`:145`).
- Bytes/call M=1: 1.5 MiB `hc_fn` + 64 KiB streams (read twice, but cached poorly at 256 threads) + 8 KiB `y_out` ≈ **1.57 MiB ⇒ 6.0 µs at roofline**.
- Fires **2× per layer × 43 = 86× per decode step**. **Per-step at roofline: 0.52 ms.** **Per-step at the measured 8 GB/s single-SM rate: ~16.9 ms** — i.e. more than a third of the entire 47.6 ms budget.
- Inefficiencies: (a) grid `(1,1,1)` — 1/25 of the GPU streams 129 MiB/token; (b) scalar `float` loads of `fn_row`, not `float4`; (c) Sinkhorn serialises 255/256 threads. This kernel is superseded by the split path below and should be considered A/B-only.

## hc_pre_mix

`hyper_connection.cu:228`. Pass-1 half of the split HC-pre. Grid is **2-D over the mix rows**: block `(t, m)` computes one output scalar. For `m < mix_hc` it dot-products `hc_fn` row `m` against the streams; for `m == mix_hc` it computes `sum(x^2)`. Output is `mix_out[t, m]` — the raw unscaled dot product, `[T, mix_hc+1]` FP32.

Launcher `ops/hyper_connection.rs:105-132`: grid `[num_tokens, mix_hc+1, 1]` = **(1, 25, 1)** at hc=4, block `[512,1,1]`.

The comment at `:218-226` is the design rationale: at 256 scalar-loading threads the kernel measured **19 GB/s** even with all 48 SMs busy; 512 threads issuing **`float4` 16-byte loads** (`:261`) puts 8× more bytes in flight. The `float4` reassociation means decode is no longer bit-identical to prefill (~1e-7 drift) — an accepted trade. Note `:251-259`: the RMS block (`m == mix_hc`) is deliberately given its own block rather than riding on block 0, because with `mix_hc+1 == 25` blocks unbalancing any one block doubles the whole kernel's wall time. **The comment justifies this with "25 blocks and 25 SMs" — that premise is false.** `cudaGetDeviceProperties` on this box reports `multiProcessorCount = 48`, matching `crates/atlas-core/src/device.rs:16` (`NUM_SMS = 48`). A 25-block grid leaves **23 of 48 SMs idle**. Splitting each `hc_fn` row over 2 blocks (`gridDim.z = 2` split-k on `nvec` + a trivial finalize) would put the kernel on 50 blocks and roughly halve it.

- dtypes: streams FP32 read as `float4`, `hc_fn` FP32 read as `float4`, `mix_out` FP32.
- Reduction: `hc_block_reduce` — warp shuffle then shared, per block.
- Shared memory: the `red[]` array inside `hc_block_reduce` (32 floats = 128 B).
- Bytes/call M=1: 1.5 MiB `hc_fn` + 64 KiB streams (24 blocks each read the full 64 KiB stream vector — this is **24 × 64 KiB = 1.5 MiB of stream re-reads**, though L2-resident) + 100 B out. HBM-unique ≈ **1.57 MiB ⇒ 6.0 µs**.
- Fires **86× per decode step**. **Per-step: 0.51 ms at roofline.** This is *the* number for mHC. Far better than `hc_pre`'s single block, but the 25-block grid covers only **52% of the 48 SMs**, so realistic is ~1.0 ms/step.
- Remaining inefficiency: two levers. (a) **Split-k the grid to 50 blocks** to cover all 48 SMs (see the SM-count correction above) — worth ~0.5 ms/step. (b) Make `hc_fn` smaller (BF16 storage for the mix matrix) — halving it saves ~0.25 ms/step directly.

## hc_pre_finish

`hyper_connection.cu:291`. Pass-2 half of the split HC-pre. Reads the 25 scalars from `mix_in`, applies `rsqrt(mi[mix_hc]/hc_dim + norm_eps)` to scale all 24 (`:318-320`), builds `comb` and `post`, runs **the same 20 Sinkhorn iterations in `tid == 0`** (`:317-379`, a verbatim duplicate of the `hc_pre` block), then collapses streams → `y_out` BF16 across a sharded grid.

Launcher `ops/hyper_connection.rs:105-132`: grid `[num_tokens, hidden_size.div_ceil(256), 1]` = **(1, 16, 1)** at H=4096, block `[256,1,1]`.

- dtypes: `mix_in` FP32, streams FP32, `y_out` BF16, `comb`/`post` FP32.
- Reduction: none across blocks for the collapse (pure map); Sinkhorn is single-thread.
- Bytes/call M=1: 100 B `mix_in` + 64 KiB streams + 8 KiB `y_out` ≈ **72 KiB ⇒ 0.27 µs**.
- Fires **86× per decode step**. **Per-step: 23 µs at roofline.**
- Inefficiency: **the Sinkhorn block is recomputed identically in all 16 grid-y shards**, 16× redundant single-threaded work per site, 1376× per decode step. It could be computed once into `comb`/`post` by shard 0 (or by a tiny separate 1-block kernel) — but the shards would then need a grid-wide sync, so the current redundancy is the cheaper choice given it is register-only work. The real lever remains lowering `sinkhorn_iters`.

## hc_post

`hyper_connection.cu:398`. The expand side: writes the sublayer output `x` back into the `hc` stream planes as `o[j*H+d] = p[j]*x[d] + Σ_i comb[i][j] * res[i*H+d]`. The core loop at `:418-427` deliberately loads **all `hc` residual values into registers before writing any output**, so `out` may safely alias `residual` (in-place highway update).

Launchers: `ops/hyper_connection.rs:152-166` (`hc_post`, **hard-codes `shards = 1`**) delegating to `:189-199` (`hc_post_sharded`, grid `[num_tokens, shards.max(1), 1]`, block `[256,1,1]`).

Call sites:
- `decode_inner.rs:664-677` — post-attention — **uses unsharded `ops::hc_post` ⇒ grid (1,1,1)**.
- `decode_inner.rs:757-772` — post-FFN — uses `hc_post_sharded(..., 16)` ⇒ grid (1,16,1). Correct.
- `multi_seq/mod.rs:355`, `:490`, `:543` — verify path — **all unsharded ⇒ grid (2,1,1) at M=2, (6,1,1) at M=6**.

- dtypes: `x` BF16 in, `res`/`o` FP32, `comb`/`p` FP32.
- Bytes/call M=1: read 8 KiB `x` + 64 KiB `res` + write 64 KiB `o` = **136 KiB ⇒ 0.51 µs**. M=2: 272 KiB ⇒ 1.02 µs. M=6: 816 KiB ⇒ 3.06 µs.
- Fires **2× per layer × 43 = 86× per decode step**. **Per-step: 44 µs at roofline** — but the 43 unsharded post-attention calls run single-SM, so the real cost is several hundred µs.
- Inefficiencies: (a) **the shards=1 bug above — highest-value one-line fix in this document**; (b) scalar `float` loads/stores rather than `float4` (unlike `hc_pre_mix`, which was already converted); (c) reads and writes `hc_streams`, which `hc_pre_mix` re-reads immediately at the next site — see the fusion table.

## hc_head

`hyper_connection.cu:433`. Final collapse of the `hc` FP32 stream planes into a single BF16 hidden vector for the LM head, applying the learned head weights. Launcher `ops/hyper_connection.rs:220-232`: grid `[num_tokens,1,1]`, block `[256,1,1]`. Called at `decode_inner.rs:791-809` (`is_last_layer`) and `multi_seq/mod.rs:400`, `:579`.

- dtypes: streams FP32 in, BF16 out.
- Bytes/call M=1: 64 KiB + 8 KiB = **72 KiB ⇒ 0.27 µs**. M=6: 432 KiB ⇒ 1.62 µs.
- Fires **1× per decode step. Per-step: 0.27 µs.** Negligible.
- Inefficiency: grid `(1,1,1)`, scalar loads. Not worth fixing at this magnitude.

## hc_mean

`hyper_connection.cu:499`. Mean of the `hc` stream planes into a single hidden vector (alternative head collapse). Grid `[num_tokens,1,1]`, block `[256,1,1]` (`ops/hyper_connection.rs:249-256`). Uses `hc_block_reduce`. Not on the V4 decode hot path (the model uses `hc_head`). Bytes/call: 72 KiB ⇒ 0.27 µs. **Per-step: 0 (unused).**

---

# RMS Norm — `kernels/gb10/common/rms_norm.cu`

17 `__global__` kernels. Shared infrastructure: `unpack_bf16x2` `:18`, `pack_bf16x2` `:24`, `warp_reduce_sum` `:31` (`__shfl_xor_sync` butterfly). All variants use `__shared__ float warp_sums[32]` for the cross-warp stage and process BF16 **two-wide via `uint32`** loads. One block per token throughout.

Common launcher shape (`ops/norm.rs:33-41`): grid `[num_tokens,1,1]`, block `[hidden_size.min(1024),1,1]`. **At decode M=1, H=4096: grid (1,1,1), block (1024,1,1) — 4 elements per thread, ONE block on 48 SMs.**

## rms_norm

`rms_norm.cu:45`. Classic RMSNorm: `y = x * rsqrt(mean(x^2) + eps) * w`. Two-wide BF16 via `uint32`, FP32 accumulate, warp-shuffle then shared reduce.

- Called at `decode_inner.rs:581-591` (pre-attn, `hidden → normed`) and `:725-735` (pre-ffn, `hidden → normed2`); `multi_seq/mod.rs:~315` for the verify path with `n` rows.
- Bytes/call M=1: 8 KiB in + 8 KiB weight + 8 KiB out = **24 KiB ⇒ 90 ns**. M=2: 40 KiB ⇒ 150 ns. M=6: 104 KiB ⇒ 390 ns.
- Fires **2× per layer × 43 = 86× per decode step**. **Per-step: 7.7 µs at roofline.**
- Inefficiencies: (a) grid `(1,1,1)` — 1/25 utilisation, so the real cost is ~25× the roofline number, ~190 µs; (b) `rms_norm_short_row_eligible` (`ops/norm.rs:75-80`) requires `hidden_size <= 256 && num_rows >= 1024`, so the warp-row fast path can never fire for this model; (c) **the input is written by `hc_pre_finish` immediately before and read straight back** — a fusion candidate (see table).

## rms_norm_f32

`rms_norm.cu:117`. Same as above with FP32 input. Not on the V4 BF16 decode path. Bytes M=1: 16 KiB in + 8 KiB w + 8 KiB out = 32 KiB ⇒ 120 ns. **Per-step: 0.**

## rms_norm_residual

`rms_norm.cu:177`. Fuses `x += residual` then norms, writing both the updated residual and the normed output — saves one full round trip vs. `bf16_residual_add` + `rms_norm`. Launcher `ops/norm.rs:101-110`, grid `[num_tokens,1,1]`. Bytes M=1: 8+8+8 in/res/w + 8+8 out = **40 KiB ⇒ 150 ns**. Not used by the mHC decode path (mHC replaces the plain residual stream). **Per-step: 0 for V4.**

## rms_norm_residual_vanilla

`rms_norm.cu:253`. Non-two-wide reference variant for bit-exactness contracts. Not in the decode path. **Per-step: 0.**

## residual_add_rms_norm

`rms_norm.cu:300`. Ordering variant of the above (add first, then norm the sum, keeping the sum as the new residual). Launcher `ops/norm.rs:134-144`. Same 40 KiB / 150 ns. Not on the V4 mHC path. **Per-step: 0.**

## residual_add_rms_norm_vanilla

`rms_norm.cu:386`. Scalar reference. **Per-step: 0.**

## residual_add_rms_norm_gatef32

`rms_norm.cu:435`. Adds an FP32 gate multiply into the fused residual+norm. Launcher `ops/norm.rs:157+`. **Per-step: 0 for V4.**

## rms_norm_residual_f32 / residual_add_rms_norm_f32

`rms_norm.cu:525` and `:604`. FP32-residual variants of the two fused forms, for architectures keeping an FP32 highway. Bytes M=1: 16 KiB res in + 8 KiB x + 8 KiB w + 16 KiB res out + 8 KiB out = 56 KiB ⇒ 210 ns. Not called on the V4 mHC decode path (mHC owns the FP32 highway). **Per-step: 0.**

## rms_norm_residual_f32_abs / residual_add_rms_norm_f32_abs

`rms_norm.cu:706` and `:765`. As above plus an absmax side-output (for downstream dynamic quantisation), avoiding a separate absmax pass. Reduction uses the same warp-shuffle + shared pattern, with a second reduction tree for the max. **Per-step: 0 for V4 decode.**

## rms_norm_f32_in_abs

`rms_norm.cu:845`. FP32 input, BF16 output, absmax side-output. **Per-step: 0.**

## f32_residual_add

`rms_norm.cu:905`. Bare FP32 `a += b`. Grid ceil(n/256), block 256. Bytes for n=4096: 16+16+16 = 48 KiB ⇒ 180 ns. **Per-step: 0 for V4.**

## gated_rms_norm

`rms_norm.cu:921`. RMSNorm followed by a sigmoid/SiLU gate multiply, fused. Saves one read+write of the hidden vector vs. separate kernels. Bytes M=1: 8 in + 8 gate + 8 w + 8 out = 32 KiB ⇒ 120 ns. Used by GDN paths, not MLA decode. **Per-step: 0 for V4 MLA.**

## gated_rms_norm_f32_input

`rms_norm.cu:1019`. FP32-input variant. **Per-step: 0.**

## gated_rms_norm_prefill

`rms_norm.cu:1098`. Multi-row prefill variant with a wider grid. Prefill only. **Per-step: 0.**

## l2_norm_bf16

`rms_norm.cu:1197`. `y = x / ||x||_2`, warp-shuffle reduce. Used for query/key L2 normalisation in some attention variants. Bytes M=1 over 4096: 16 KiB ⇒ 60 ns. Not on the V4 MLA decode path. **Per-step: 0.**

---

# `kernels/gb10/common/rms_norm_vanilla.cu`

## rms_norm_vanilla

`rms_norm_vanilla.cu:38`. Scalar (one BF16 element per thread) RMSNorm, the bit-exact reference against which the two-wide `rms_norm` is validated under `--fmad=false`. Grid `[num_tokens,1,1]`, block `[min(H,1024)]`. Bytes M=1: 24 KiB ⇒ 90 ns, but with 4× the load instructions of `rms_norm`. **Per-step: 0** (reference only).

## rms_norm_vanilla_warp_row

`rms_norm_vanilla.cu:120`. One **warp** per row instead of one block — the fast path for short, numerous rows. Gated by `rms_norm_short_row_eligible` (`ops/norm.rs:75-80`): `hidden_size <= 256 && hidden_size % 2 == 0 && num_rows >= 1024`. **H=4096 decode never qualifies.** **Per-step: 0.**

---

# `kernels/gb10/common/residual_add.cu`

11 kernels, all **scalar (one element per thread)**, all grid `ceil(n/256)` block `256`, no shared memory, no reductions. At n = 4096 that is grid (16,1,1) — adequate SM coverage, but 2-byte transactions instead of 16-byte. Converting these to `uint4` (8×BF16) would cut instruction count 8× and let the memory pipe issue full-width; individually each is sub-microsecond, but they fire often.

## bf16_residual_add

`residual_add.cu:10`. `a[i] += b[i]` in BF16. Bytes M=1 at n=4096: 8+8+8 = **24 KiB ⇒ 90 ns**. On the V4 MoE path it is **superseded by `moe_weighted_sum_blend_residual_batchn`** (see `fused_verify_elemwise.cu:236`), which absorbs it. Where still used, ~43×/step ⇒ **3.9 µs**.
- Inefficiency: scalar BF16 loads; and it is a pure elementwise pass over a buffer its producer just wrote — always a fusion candidate.

## bf16_to_f32

`residual_add.cu:25`. Width conversion. 8 KiB in + 16 KiB out = 24 KiB ⇒ 90 ns. Occasional. **Per-step: ~0.**

## silu_mul_separate

`residual_add.cu:41`. `out = silu(gate) * up` with gate and up in separate buffers. For a 4096-wide intermediate: 8+8+8 = 24 KiB ⇒ 90 ns. On the V4 dense-FFN path only; the MoE path fuses this. **Per-step: small.**

## bf16_scaled_add

`residual_add.cu:60`. `a += s * b`, scalar `s`. 24 KiB ⇒ 90 ns.

## bf16_sigmoid_blend

`residual_add.cu:78`. `out = sigmoid(g) * a + (1 - sigmoid(g)) * b`. 32 KiB ⇒ 120 ns.

## sigmoid_gate_mul

`residual_add.cu:96`. `out = sigmoid(g) * x`. 24 KiB ⇒ 90 ns.

## sigmoid_gate_mul_batched

`residual_add.cu:119`. Batched-row form of the above for the verify path (M=2/M=6). M=6: 144 KiB ⇒ 540 ns.

## bf16_concat

`residual_add.cu:142`. Copies two BF16 buffers into one contiguous output. Pure data movement — **a prime fusion victim**: whatever produced the two halves could have written them into the concatenated layout directly, eliminating a full read+write. For an MLA 576-wide concat at M=1 that is ~2.3 KiB ⇒ 8 ns; small, but the launch and round trip are pure waste.

## sigmoid_gate_mul_head_broadcast

`residual_add.cu:171`. Per-head scalar gate broadcast across `head_dim`.

## softplus_gate_mul_head_broadcast

`residual_add.cu:192`. Same with softplus. Used by GDN.

## bf16_sigmoid_blend_device

`residual_add.cu:212`. As `bf16_sigmoid_blend` but the blend scalar is read from device memory (graph-replayable — no host readback needed). **This pattern is worth copying anywhere a host-side scalar currently forces a sync.**

---

# `kernels/gb10/common/bf16_add.cu`

## bf16_add_inplace

`bf16_add.cu:8`. Two-rank all-reduce accumulate, `a += b` in BF16. Single-GB10 decode is rank-1, so **not launched. Per-step: 0.**

---

# `kernels/gb10/common/vector_add.cu`

## vector_add

`vector_add.cu:6`. Test-only smoke kernel. **Per-step: 0.**

---

# `kernels/gb10/common/relu_squared.cu`

## relu_squared

`relu_squared.cu:12`. `y = max(x,0)^2`, the DeepSeek FFN activation. Grid ceil(n/256), block 256. For a 4096-wide FFN intermediate at M=1: 8 KiB in + 8 KiB out = **16 KiB ⇒ 60 ns**. On the V4 MoE path the activation is fused into the expert GEMM, so this fires only for dense FFN layers. **Per-step: bounded by ~2.6 µs if it fired every layer.**
- Inefficiency: scalar loads; and it is a read-modify-write over a buffer the up-projection just wrote — should be an epilogue of that GEMM, not a kernel.

## relu_squared_inplace

`relu_squared.cu:26`. In-place variant: 8 KiB read + 8 KiB write = 16 KiB ⇒ 60 ns. Halves the traffic of the out-of-place form by reusing the buffer.

## bias_add_bf16_f32

`relu_squared.cu:43`. BF16 tensor + FP32 bias broadcast. Should be a GEMM epilogue.

## moe_weighted_sum_scale

`relu_squared.cu:57`. Scales expert outputs by their router weight and accumulates. On the V4 path this is largely displaced by `moe_weighted_sum_blend_residual_batchn` (`fused_verify_elemwise.cu:236`). Covered in detail in the MoE document.

## convert_f32_to_bf16

`relu_squared.cu:80`. Width narrowing. 16 KiB in + 8 KiB out ⇒ 90 ns for 4096 elements. Occasional.

---

# RoPE — `kernels/gb10/common/rope.cu`

Six kernels. Common launcher shape: grid `(num_q_heads + num_kv_heads, ceil(seq_len / pos_per_block), batch)`, block **128**.

The header comment at `:29` records the key SM121 fact: **FP64 `pow()` runs at 1/64 rate on sm_121**, and computing the inverse frequency per thread left `rope_forward` **~13× above its bandwidth floor (1.16 ms for 4.4 M elements)**. The fix in `rope_forward` was to cache the frequencies in `__shared__ float s_freq[128]`, computed once per block. **Three of the six kernels here — and both mRoPE kernels, and all three `fused_k_norm_rope_cache` kernels — never received this fix and still call FP64 `pow()` per thread.**

## rope_forward

`rope.cu:29`. Standard split-half RoPE. `__shared__ float s_freq[128]` caches the per-block inverse frequencies. Q and K rotated in one launch (grid.x spans both head sets).
- Bytes/call M=1 for MLA-sized q+k rope (`(nq+nkv) * head_dim` BF16 read+write): with 128 q heads × 64 rope dims that is ~16 KiB in + 16 KiB out ⇒ **0.12 µs**.
- Not the V4 MLA variant (V4 uses the interleaved YaRN form). **Per-step: 0 for V4.**

## rope_forward_proportional

`rope.cu:143`. Proportional / NTK-scaled RoPE. **Still computes `pow()` in FP64 per thread — the shared-memory frequency cache was never applied here.** At 1/64 FP64 rate this is the same ~13× overhang the base kernel had. Not on the V4 path, but it is a latent trap for any config that selects it. **Per-step: 0 for V4.**

## rope_forward_yarn

`rope.cu:221`. YaRN interpolation/extrapolation ramp with attention-factor `mscale`. Split-half layout. **Per-step: 0 for V4** (V4 uses interleaved).

## rope_forward_yarn_scaled

`rope.cu:290`. YaRN with an explicit output scale. **Per-step: 0 for V4.**

## rope_forward_yarn_interleaved

`rope.cu:347`. **This is the V4 MLA RoPE.** Rotates adjacent pairs `(2i, 2i+1)` rather than split halves, and folds the YaRN `mscale` directly into the cos/sin tables so no separate scale kernel is needed. FP32 internal, BF16 in/out.
- Grid `(nq + nkv, ceil(seq/pos_per_block), batch)`, block 128. At decode M=1 with MLA's 64 rope dims and 128 q heads + 1 latent k: grid ≈ (129, 1, 1) — good SM coverage.
- Bytes/call M=1: q rope part `128 heads × 64 × 2 B` = 16 KiB read + 16 KiB write, plus k 128 B, plus cos/sin tables (small, L2-resident) = **~32 KiB ⇒ 0.12 µs**. M=6: ~192 KiB ⇒ 0.72 µs.
- Fires **1× per layer × 43 = 43× per decode step**. **Per-step: 5.2 µs at roofline.**
- Inefficiency: it is a pure elementwise pass over the Q tensor that the Q up-projection GEMM just wrote and that attention immediately reads — the classic epilogue-fusion candidate. `fused_verify_elemwise.cu:78` already proves the fusion works for the verify path.

## rope_forward_yarn_interleaved_inv

`rope.cu:417`. The **conjugate** form (negated sin) used for DeepSeek-V4 eq. 26 output **de-rotation** — undoing the query rotation on the attention output. Identical cost profile to the forward: **~32 KiB ⇒ 0.12 µs**, **43×/step ⇒ 5.2 µs**.
- Inefficiency: same — it reads and writes the attention output buffer between attention and the o-projection. `rope_forward_yarn_interleaved` and `rope_forward_yarn_interleaved_inv` bracket attention, and each is a standalone HBM round trip that a fused attention epilogue would eliminate.

---

# `kernels/gb10/common/rope_mrope_interleaved.cu`

## rope_forward_mrope_interleaved

`rope_mrope_interleaved.cu:34`. Multimodal RoPE (separate t/h/w position sections) in the interleaved-pair layout. **Per-thread FP64 `pow()` — un-fixed.** Vision path only. **Per-step: 0 for text decode.**

## rope_forward_mrope_interleaved_k_only

`rope_mrope_interleaved.cu:109`. K-only variant for cache writes. **Per-thread FP64 `pow()` — un-fixed. Per-step: 0 for text decode.**

---

# `kernels/gb10/common/fused_k_norm_rope_cache.cu`

The single most valuable existing fusion in this file set: it collapses **K RMSNorm → RoPE → paged-cache write** into one kernel, keeping everything FP32 internally and performing **exactly one BF16 rounding at the write**. The header notes this eliminated two intermediate rounding stages that produced a documented quality cliff at layers 35-39.

Common shape: grid `(num_tokens, num_kv_heads, 1)`, block `(head_dim)`. `__shared__ float smem_normed[256]`, `__shared__ float warp_sums[8]`; `warp_reduce_sum_fkv` at `:31`.

## fused_k_norm_rope_cache_write_bf16

`fused_k_norm_rope_cache.cu:53`. BF16 KV cache target. Reads the raw K projection, norms it, rotates it, writes it to the paged slot.
- Bytes/call M=1 for MLA (1 latent KV head × 576 dims): ~1.2 KiB in + 1.2 KiB out ⇒ **~9 ns**. M=6: ~54 ns.
- Would fire 43×/step if the model used BF16 KV. **V4 uses FP8 KV ⇒ per-step: 0** (the `_fp8` variant runs instead).
- Inefficiency: **per-thread FP64 `pow()` for the inverse frequency** — never got the `rope_forward` shared-memory cache treatment. At 1/64 FP64 rate on sm_121 and only `head_dim` threads per block, this is latency-bound, not bandwidth-bound. Precomputing the frequency table once (as `rope.cu:29` does) is a direct win.

## fused_k_norm_rope_mrope_cache_write_bf16

`fused_k_norm_rope_cache.cu:153`. mRoPE variant. Vision only. **Per-step: 0.**

## fused_k_norm_rope_cache_write_fp8

`fused_k_norm_rope_cache.cu:252`. **The V4 decode path.** Same fusion with an FP8 E4M3 cache write, so the output is 1 byte/element instead of 2.
- Bytes/call M=1: ~1.2 KiB BF16 in + ~0.6 KiB FP8 out ⇒ **~7 ns**. M=2: 14 ns. M=6: 42 ns.
- Fires **43× per decode step**. **Per-step: 0.3 µs.** Bandwidth-trivial.
- Inefficiency: the same un-fixed FP64 `pow()`. With grid (1, 1, 1) at M=1 decode — one token, one latent KV head — **this kernel runs a single block of `head_dim` threads on one SM, 43 times per step**, and each of those blocks pays FP64 transcendental latency. This is a pure-latency kernel that the roofline analysis makes look free but which the launch-serialised critical path does not.

---

# `kernels/gb10/common/argmax_bf16.cu`

## argmax_bf16

`argmax_bf16.cu:14`. Greedy sampling: argmax over the full **129280**-entry BF16 logit vector. **Grid `(1,1,1)`, block `(1024,1,1)`.** Each thread strides over ~127 vocab entries accumulating a local max, then a pure **shared-memory tree reduction** over `__shared__ float s_val[1024]` + `__shared__ unsigned int s_idx[1024]` = **8 KiB of shared memory**. No warp shuffle at all.
- Bytes/call: 129280 × 2 B = **252.5 KiB ⇒ 0.95 µs at roofline**.
- Fires **1× per decode step. Per-step: 0.95 µs at roofline.**
- Inefficiencies, and they are severe relative to that number: (a) **grid `(1,1,1)` — the whole 252 KiB vocab streams through ONE SM**, so at the ~8-19 GB/s a single SM achieves this is **13-31 µs, not 0.95 µs**; a two-stage split (256 blocks → 256 partials → 1 block final) would recover ~25×; (b) the reduction is a shared-memory tree with `__syncthreads()` at every level where `__shfl_down_sync` within warps would remove 5 of the 10 barriers and 8 KiB of shared pressure; (c) BF16 loaded scalar, not two-wide via `uint32`.

## top2_bf16_rows

`argmax_bf16.cu:66`. Top-2 per row, for speculative-decode acceptance checks that need the runner-up logit. Same single-block-per-row structure. Fires once per verify step on the M rows.
- Bytes/call M=6: 6 × 252.5 KiB = **1.48 MiB ⇒ 5.7 µs at roofline**; grid (6,1,1) so 6/48 SMs.
- **Per-step (verify): ~5.7 µs at roofline, realistically ~25 µs.** Same fix applies: shard over vocab.

## argmax_fp32

`argmax_bf16.cu:124`. FP32 logit variant. 129280 × 4 B = 505 KiB ⇒ 1.9 µs. Used when logits are kept FP32. Same single-block pathology.

---

# `kernels/gb10/common/embed_from_argmax.cu`

## embed_from_argmax

`embed_from_argmax.cu:17`. Gathers one embedding row (`hidden = 4096` BF16 = 8 KiB) from the embedding table using a **device-resident** token id — no host round trip, so the whole decode step stays graph-replayable.
- Bytes/call M=1: **8 KiB ⇒ 30 ns**. Fires **1× per decode step. Per-step: 0.03 µs.** Negligible.

## batched_embed

`embed_from_argmax.cu:41`. M-row gather for the verify path. M=6: 48 KiB ⇒ 180 ns. **Per-step: 0.18 µs.**

## batched_embed_f32

`embed_from_argmax.cu:57`. FP32 output variant — relevant because the mHC highway is FP32, so this can feed `hc_expand` directly. 6 × 16 KiB = 96 KiB ⇒ 360 ns.

## embed_from_argmax_f32

`embed_from_argmax.cu:73`. Single-row FP32 variant. 16 KiB ⇒ 60 ns.
- **Fusion note:** `embed_from_argmax_f32` writes an FP32 hidden vector that `hc_expand` then reads and broadcasts into 4 planes. These two are adjacent, tiny, and trivially fusible into one kernel that gathers straight into all `hc` planes — saving one 16 KiB round trip and one launch per step.

---

# `kernels/gb10/common/metadata_fill.cu`

## fill_slots_from_block_table

`metadata_fill.cu:5`. Computes the `slot_mapping` (i64, `-1` = skip) for each token from the paged block table and sequence lengths, **on device**, so the decode loop never needs a host-side page computation. Grid/block sized to `num_tokens`.
- Bytes/call M=1: a handful of i32/i64 — **< 1 KiB ⇒ ~4 ns**. Fires 1×/step. **Per-step: ~0.**
- This kernel is pure enablement for CUDA-graph capture; its cost is a rounding error and it should not be touched.

---

# `kernels/gb10/common/reshape_and_cache.cu`

Common shape: grid `(num_tokens, 1, 1)`, block `256`. All copies use **`uint2` 8-byte vectorised** loads/stores. `slot_mapping[i] < 0` early-returns the block (padding tokens).

## reshape_and_cache_flash_v_only

`reshape_and_cache.cu:30`. Writes only the V half into the paged cache (for architectures where K is written by the fused K kernel). MLA V=512 dims: ~1 KiB in + 1 KiB out ⇒ ~8 ns.

## reshape_and_cache_flash

`reshape_and_cache.cu:67`. BF16 K and V into paged layout. MLA 576 dims: ~2.3 KiB round trip ⇒ ~9 ns. **V4 uses FP8 ⇒ per-step: 0.**

## reshape_and_cache_flash_fp8

`reshape_and_cache.cu:152`. **The V4 decode cache write** (for the V/NoPE half not covered by the fused K kernel). BF16 in, FP8 E4M3 out with per-tensor or per-head scale.
- Bytes/call M=1: 576 × 2 B in + 576 × 1 B out ≈ **1.7 KiB ⇒ ~6 ns**. M=6: ~38 ns.
- Fires **up to 43× per decode step. Per-step: ~0.3 µs.** Bandwidth-trivial.
- Inefficiency: grid `(1,1,1)` at M=1 — one block. Latency, not bandwidth. Merging it into the fused K kernel (which already writes the same page) would remove 43 launches and one page-table walk per step.

## reshape_and_cache_flash_nvfp4

`reshape_and_cache.cu:293`. NVFP4 cache variant (E2M1 + per-group FP8 scale). Halves cache bytes again vs FP8 at the cost of a per-group absmax. **Per-step: 0 unless NVFP4 KV is selected.**

## bf16_absmax_per_head

`reshape_and_cache.cu:395`. Per-head absmax for dynamic cache scaling. Warp-shuffle reduce. Small.

## bf16_absmax

`reshape_and_cache.cu:443`. Whole-tensor absmax. Warp-shuffle + `atomicMax` on the float's uint bit pattern (monotonic for non-negative floats — a valid and cheap trick). Small.
- Inefficiency: absmax over a buffer implies **an extra full read of a tensor someone just wrote**. Where the producer is an RMSNorm, `rms_norm_*_abs` variants (`rms_norm.cu:706`, `:765`, `:845`) already fuse it — those should be preferred over a standalone absmax pass everywhere the producer is a norm.

---

# `kernels/gb10/common/reshape_and_cache_turbo.cu`

Thirteen TurboQuant cache-write kernels. All share `GROUP_SIZE = 16` codebook quantisation with a **matched-norm L2 correction** — after quantising a group, the kernel rescales it so the group's L2 norm matches the original, costing 16 extra FMAs per group and buying ~0.5% PPL. Grid `(num_tokens, ...)`, block 256.

| Kernel | Line | K format | V format |
|---|---|---|---|
| `reshape_and_cache_flash_turbo4` | 179 | turbo4 | turbo4 |
| `reshape_and_cache_flash_turbo8` | 276 | turbo8 | turbo8 |
| `reshape_and_cache_flash_turbo3` | 358 | turbo3 | turbo3 |
| `reshape_and_cache_flash_turbo2` | 470 | turbo2 | turbo2 |
| `..._bf16k_turbo3v` | 576 | BF16 | turbo3 |
| `..._bf16k_turbo4v` | 689 | BF16 | turbo4 |
| `..._bf16k_turbo2v` | 787 | BF16 | turbo2 |
| `..._fp8k_turbo3v` | 889 | FP8 | turbo3 |
| `..._fp8k_turbo4v` | 999 | FP8 | turbo4 |
| `..._fp8k_turbo2v` | 1100 | FP8 | turbo2 |
| `..._turbo4k_turbo3v` | 1210 | turbo4 | turbo3 |
| `..._turbo4k_turbo8v` | 1313 | turbo4 | turbo8 |
| `..._turbo3k_turbo8v` | 1406 | turbo3 | turbo8 |

Bytes/call M=1 for MLA 576 dims: ~1.2 KiB BF16 in, 0.15-0.6 KiB out depending on bit width ⇒ **< 6 ns**. If a turbo variant is selected it fires 43×/step ⇒ **< 0.3 µs/step**. These kernels exist to shrink the *attention* kernel's cache reads (covered in the MLA document), not to save time in themselves. **Per-step in this category: negligible.** The real payoff is downstream: a turbo3 V cache cuts the attention kernel's V traffic by ~5× versus BF16.

---

# `kernels/gb10/common/kv_cache_append.cu`

## kv_cache_append

`kv_cache_append.cu:20`. Appends new K/V into a contiguous (non-paged) cache. Grid `(num_new_tokens, num_kv_heads, batch)`, block 256.
- **Non-vectorised: scalar BF16 element copy**, unlike `reshape_and_cache.cu` which uses `uint2`. 2-byte transactions where 8- or 16-byte would do.
- Bytes/call M=1 MLA: ~2.3 KiB ⇒ ~9 ns. Used only on the contiguous-cache path; V4 decode is paged. **Per-step: 0 for V4.**
- Fix if ever used: `uint4` copies, matching `kv_block_indirect_copy`.

---

# `kernels/gb10/common/kv_block_indirect_copy.cu`

## kv_block_indirect_copy

`kv_block_indirect_copy.cu:25`. Copies KV pages between block slots (used for prefix-cache reuse, beam fork, and page defragmentation). Grid `(chunks, MAX_PAIRS, 2)`, block 256, **`uint4` 16-byte vectorised**. Reads `meta[0] = n_pairs` **on device** and early-returns blocks beyond it — so the launch geometry is constant and the kernel is **CUDA-graph replayable for any `n_pairs <= MAX_PAIRS`**. This is the right pattern and should be the template for any other variable-count kernel in the decode loop.
- Bytes/call: `n_pairs × block_size × 576 × 1 B` for FP8. Zero on a steady-state decode step (no page moves). **Per-step: 0.**

---

# `kernels/gb10/common/transpose_u8.cu`

## transpose_u8

`transpose_u8.cu:15`. Byte-granular tiled transpose. `__shared__ unsigned char tile[32][33]` — the **+1 pad removes shared-memory bank conflicts**. Grid `(ceil(cols/32), ceil(rows/32))`, block `(32,8)` (each thread handles 4 rows).
- Used at weight-load time to reorient FP8/NVFP4 weight scale tensors. **Per-step: 0.**

---

# `kernels/gb10/common/widen_block_scale_f32.cu`

## widen_block_scale_f32

`widen_block_scale_f32.cu:20`. Expands a compact block-scale tensor to the widened layout the GEMM kernels expect. **Load-time only. Per-step: 0.**

---

# `kernels/gb10/common/dequant_fp8_blockscaled_bf16.cu`

## dequant_fp8_blockscaled_bf16

`dequant_fp8_blockscaled_bf16.cu:89`. Dequantises block-scaled FP8 weights to BF16. Uses `__constant__ float E4M3_LUT_DEQ[256]` (`:22`) — a **256-entry constant-memory lookup table**, so decoding an E4M3 byte is one broadcast constant load rather than a bit-manipulation sequence. Grid `(ceil(K/64), ceil(N/4))`, block `(64,4)`.
- **Load-time only** (weights are dequantised once at startup, or consumed in quantised form by the FP8 GEMV kernels). **Per-step: 0.**

---

# `kernels/gb10/common/dequant_nvfp4_bf16.cu`

## dequant_nvfp4_to_bf16

`dequant_nvfp4_bf16.cu:50`. NVFP4 → BF16. `__constant__ float DQ_E2M1_LUT[16]` (`:26`) for the 4-bit mantissa/exponent decode, plus `dq_fp8_e4m3_decode` (`:34`) for the per-group FP8 scale, times a global FP32 scale. Grid `(N,1,1)`, block 256.
- **Load-time only. Per-step: 0.**

---

# `kernels/gb10/common/e2m1_branchless.cu`

## e2m1_quantize

`e2m1_branchless.cu:48`. Float → E2M1 (4-bit) quantisation. The device helper `branchless_float_to_e2m1` (`:17`) does the conversion with **7 unsigned integer comparisons and zero branches**, so there is no warp divergence regardless of input distribution — the right design for a quantiser that sees arbitrary activations. `pack_8xe2m1` (`:31`) packs 8 nibbles into a `uint32`.
- Inefficiency: **8 scalar `float` loads per thread**, not two `float4` loads. Since each thread already produces exactly one packed `uint32` from 8 consecutive floats, converting to `float4 ×2` is a mechanical change that would halve the load instruction count and issue full-width transactions.
- Used at quantisation/load time and by NVFP4 activation quantisation. **Per-step: 0 on the V4 FP8-weight decode path.**

---

# `kernels/gb10/common/quantize_bf16_to_nvfp4.cu`

## f32_to_bf16_trunc

`quantize_bf16_to_nvfp4.cu:27`. Truncating (not round-to-nearest) FP32→BF16, matching the reference quantiser's rounding contract exactly.

## nvfp4_global_absmax

`quantize_bf16_to_nvfp4.cu:131`. Whole-tensor absmax for the NVFP4 global scale. `__shfl_down_sync` within warps → `__shared__ float smem[8]` across warps → **`atomicMax` on the reinterpreted uint bits** of the block result. Three-stage, minimal shared memory, no second kernel launch for the final reduce.
- Inefficiency: it is a **full extra read of the tensor** before the quantise kernel reads it again — the classic two-pass quantiser problem. Unavoidable for a *global* scale (you cannot know the max without seeing everything), which is exactly why per-group scales are preferable.

## quantize_bf16_to_nvfp4

`quantize_bf16_to_nvfp4.cu:176`. BF16 → NVFP4 with per-group FP8 scales and the global scale from above. Grid `(N,1,1)`, block 256.
- **Load-time / calibration only for V4 decode. Per-step: 0.**

---

# `kernels/gb10/common/quant_rowwise_fp8.cu`

## quant_rowwise_fp8

`quant_rowwise_fp8.cu:38`. Per-row dynamic FP8 activation quantisation: absmax the row, derive the scale, quantise. Grid `(R,1,1)`, block 256. `__shared__ float smem_warp_max[8]` + `__shared__ float smem_scale`; `__shfl_down_sync` within warps then shared across.
- **Inefficiency (the important one): the kernel reads X twice** — once for the absmax pass, once for the quantise pass. For a 4096-wide row that is 8 KiB read twice + 4 KiB written = 20 KiB where 12 KiB would suffice.
- Bytes/call M=1, R=1, K=4096: **20 KiB ⇒ 75 ns** (12 KiB ⇒ 45 ns if single-pass). M=6: 120 KiB ⇒ 450 ns.
- Fires once per quantised GEMV input. If that is 1× per layer: **43×/step ⇒ 3.2 µs**, of which **~1.3 µs is the redundant second read**.
- Fix: at block 256 over a 4096 row each thread owns 16 elements — they fit in registers. Hold them across the reduction and the second global read disappears entirely. (This is exactly what `rms_norm_*_abs` does for the norm case.)

---

# `kernels/gb10/common/per_token_group_quant_fp8.cu`

## per_token_group_quant_fp8

`per_token_group_quant_fp8.cu:39`. Per-token, per-128-element-group FP8 quantisation (the DeepSeek block-scaled GEMM input format). Grid `(M, K/128, 1)`, block 128. `__shared__ float smem_warp_max[4]` + `smem_scale`.
- **Same two-pass inefficiency: reads A twice.** But here each block owns exactly 128 elements and has 128 threads, so **each thread holds exactly one element** — caching it in a single register across the reduction is a one-line change that removes the entire second read.
- Bytes/call M=1, K=4096 (32 groups): 8 KiB read twice + 4 KiB + 128 B scales = **20.1 KiB ⇒ 75 ns**. M=6: 121 KiB ⇒ 450 ns.
- Fires per quantised GEMM input; at 1×/layer: **43×/step ⇒ 3.2 µs**, **~1.3 µs of it redundant**.
- Grid `(1, 32, 1)` at M=1 — 32 blocks, good SM coverage. The problem is purely the duplicate read.

---

# `kernels/gb10/common/fused_verify_elemwise.cu`

The two kernels here are the model of what the rest of this file set should look like.

## fused_qkv_norm_rope_cache_write_bf16

`fused_verify_elemwise.cu:78`. Collapses **Q norm + K norm + RoPE(Q) + RoPE(K) + paged cache write** for all `n` verify rows into a **single launch**, replacing what was `8n` separate launches. Grid `(num_q_heads + 2*num_kv_heads, n, 1)`, block `(head_dim)`. `__shared__ __nv_bfloat16 normed_bf[256]`, `__shared__ float warp_sums[32]`.
- Everything stays FP32 between the norm and the rotation; one BF16 round at the end.
- Bytes/call M=6: q ~96 KiB + k ~14 KiB round trip ≈ **110 KiB ⇒ 0.42 µs**.
- Fires **43× per verify step. Per-step (verify): 18 µs at roofline** — and it removes ~2064 launches (8 × 6 × 43).
- This is the template: **the M=1 decode path does not use it** and still issues the unfused sequence (`rms_norm` → `rope_forward_yarn_interleaved` → `fused_k_norm_rope_cache_write_fp8`). Extending this kernel to n=1 is a direct win on the serial decode path.

## moe_weighted_sum_blend_residual_batchn

`fused_verify_elemwise.cu:236`. Fuses the MoE expert weighted sum, the shared-expert sigmoid blend, and the residual add into one pass. Grid `(ceil(hidden/256), num_tokens, 1)`, block 256. `__shared__ float s_warp_sums[8]`, `__shared__ float sigmoid_val`.
- The source comment states it plainly: *"Removes one launch and one `[n, hidden]` BF16 read+write round-trip per layer."*
- Bytes saved per layer at M=1: 8 KiB read + 8 KiB write = 16 KiB ⇒ 60 ns; **43×/step ⇒ 2.6 µs saved.** At M=6: 96 KiB ⇒ 360 ns/layer ⇒ **15.5 µs/step saved.**
- Grid at M=1 is (16,1,1) — 16 of 48 SMs, acceptable.

---

# `kernels/gb10/common/wht_bf16.cu`

## wht_bf16_inplace

`wht_bf16.cu:51`. In-place Walsh-Hadamard Transform over BF16 head vectors, the rotation that makes TurboQuant's codebook quantisation near-optimal (it spreads outlier energy uniformly across the head dimension). Grid `(num_heads,1,1)`, block **`(32,1,1)` — one warp per head**. The device helper `wht256_warp_bf16` (`:17`) does the butterfly network entirely with **`__shfl_xor_sync` — zero shared memory, zero `__syncthreads()`**. Supports head dims 128 / 256 / 512.
- Bytes/call M=1 for MLA 576 (one latent head): ~1.2 KiB read + 1.2 KiB write ⇒ **~9 ns**.
- Fires only when a TurboQuant KV variant is active, at most 43×/step ⇒ **< 0.4 µs/step.**
- The one-warp-per-head shape means at M=1 with 1 latent KV head the grid is `(1,1,1)` — a single warp on the whole GPU. Pure latency; irrelevant at these byte counts but it does serialise into the critical path.

## wht_bf16_inplace_inv

`wht_bf16.cu:183`. The inverse. The WHT is **self-inverse up to a `1/sqrt(N)` scale**, so this is the same butterfly with a different normalisation. Same cost profile.

---

# `kernels/gb10/common/tq_plus_innerq.cu`

**This file contains no `__global__` kernels.** It holds `__device__` state — `d_innerq_scale`, `d_innerq_scale_inv`, `d_innerq_sq_accum`, `d_innerq_count`, `d_innerq_active`, `d_innerq_calibrating` — and two host-side controllers, `turbo_innerq_start_calibration` (`:64`) and `turbo_innerq_finalize` (`:81`), which move that state with `cudaMemcpyToSymbol` / `cudaMemcpyFromSymbol`.

Because the flags live in device symbols rather than kernel arguments, the *apply* kernels below can be unconditionally launched inside a captured CUDA graph and cheaply self-disable. That is the right design — but note `cudaMemcpyToSymbol` is **not graph-capturable**, so calibration must run outside graph capture. **Per-step: 0.**

---

# `kernels/gb10/common/tq_plus_innerq_apply.cu`

## tq_plus_innerq_apply_q

`tq_plus_innerq_apply.cu:20`. Applies the calibrated inner-quantisation scale to the query vectors. Grid `(num_heads,1,1)`, block `(32,1,1)` — one warp per head. **Early-returns immediately when `d_innerq_active == 0`**, so the steady-state cost when the feature is off is a launch plus a constant-cache read.
- Bytes/call when active: ~1.2 KiB round trip ⇒ ~9 ns. **Per-step: < 0.4 µs.**

## tq_plus_innerq_apply_k

`tq_plus_innerq_apply.cu:46`. Same for keys, plus — **during calibration only** — an `atomicAdd` into `d_innerq_sq_accum` from head 0 to accumulate the running sum of squares. Restricting the atomic to head 0 avoids 128-way contention on a single address.
- **Per-step: < 0.4 µs** when active, ~0 when not.

---

# `kernels/gb10/common/lora_bgmv.cu`

Constants: `BGMV_BLOCK_SIZE 256`, `BGMV_N_PER_BLOCK 4`, `BGMV_VEC_SIZE 8` (`uint4` = 128-bit loads). `__shared__ float smem[8]`. Both kernels are documented as **bit-identical to `dense_gemv_bf16`** — the same accumulation order — so enabling LoRA cannot change base-model numerics.

## lora_bgmv_shrink

`lora_bgmv.cu:50`. Batched grouped matrix-vector: `x @ A` down to `max_rank`. Grid `(ceil(max_rank/4), N, 1)`, block 256.

## lora_bgmv_expand_fold

`lora_bgmv.cu:139`. `(x @ A) @ B` expanded back to `n_out` and **folded into the existing output** (accumulate, not overwrite), so no separate add kernel is needed. Grid `(ceil(n_out/4), N, 1)`, block 256.

**Neither is in the V4 decode path unless LoRA adapters are loaded. Per-step: 0.**

---

# Ranking — estimated ms per decode step (M=1, 43 layers)

Roofline column is bytes ÷ 273 GB/s. "Realistic" adjusts for grid underfill where a kernel runs on ≪48 SMs (using the ~8 GB/s single-SM figure measured in `hyper_connection.cu:192-211`, or ~19 GB/s for the 256-thread scalar case).

| # | Kernel | file:line | Fires/step | Bytes/call M=1 | Roofline ms/step | Realistic ms/step |
|---|---|---|---|---|---|---|
| 1 | **`hc_pre`** (fused, `ATLAS_HC_SPLIT=0`) | `hyper_connection.cu:58` | 86 | 1.57 MiB | **0.517** | **~16.9** (1 SM) |
| 2 | **`hc_pre_mix`** (split, default) | `hyper_connection.cu:228` | 86 | 1.57 MiB | **0.517** | **~0.55** (48 SMs, float4) |
| 3 | **`hc_post`** | `hyper_connection.cu:398` | 86 | 136 KiB | **0.044** | **~0.35** (43 calls unsharded) |
| 4 | `hc_pre_finish` | `hyper_connection.cu:291` | 86 | 72 KiB | 0.023 | ~0.03 |
| 5 | `rms_norm` | `rms_norm.cu:45` | 86 | 24 KiB | 0.0077 | **~0.19** (grid 1) |
| 6 | `rope_forward_yarn_interleaved` | `rope.cu:347` | 43 | 32 KiB | 0.0052 | ~0.006 |
| 7 | `rope_forward_yarn_interleaved_inv` | `rope.cu:417` | 43 | 32 KiB | 0.0052 | ~0.006 |
| 8 | `bf16_residual_add` (where unfused) | `residual_add.cu:10` | ≤43 | 24 KiB | 0.0039 | ~0.005 |
| 9 | `quant_rowwise_fp8` | `quant_rowwise_fp8.cu:38` | ~43 | 20 KiB | 0.0032 | ~0.004 |
| 10 | `per_token_group_quant_fp8` | `per_token_group_quant_fp8.cu:39` | ~43 | 20.1 KiB | 0.0032 | ~0.004 |
| 11 | `relu_squared` (dense layers only) | `relu_squared.cu:12` | ≤43 | 16 KiB | 0.0026 | ~0.003 |
| 12 | `argmax_bf16` | `argmax_bf16.cu:14` | 1 | 252.5 KiB | 0.00095 | **~0.020** (grid 1) |
| 13 | `wht_bf16_inplace` (+ inv) | `wht_bf16.cu:51`, `:183` | ≤86 | 2.4 KiB | 0.0008 | ~0.001 |
| 14 | `tq_plus_innerq_apply_q` / `_k` | `tq_plus_innerq_apply.cu:20`, `:46` | ≤86 | 2.4 KiB | 0.0008 | ~0.001 |
| 15 | `fused_k_norm_rope_cache_write_fp8` | `fused_k_norm_rope_cache.cu:252` | 43 | 1.8 KiB | 0.0003 | ~0.001 (FP64 pow) |
| 16 | `reshape_and_cache_flash_fp8` | `reshape_and_cache.cu:152` | 43 | 1.7 KiB | 0.0003 | ~0.001 |
| 17 | `reshape_and_cache_flash_turbo*` | `reshape_and_cache_turbo.cu:179+` | ≤43 | 1.8 KiB | 0.0003 | ~0.001 |
| 18 | `hc_expand` | `hyper_connection.cu:38` | 1 | 72 KiB | 0.00027 | ~0.001 |
| 19 | `hc_head` | `hyper_connection.cu:433` | 1 | 72 KiB | 0.00027 | ~0.001 |
| 20 | `embed_from_argmax` | `embed_from_argmax.cu:17` | 1 | 8 KiB | 0.00003 | ~0.0001 |
| 21 | `fill_slots_from_block_table` | `metadata_fill.cu:5` | 1 | < 1 KiB | ~0 | ~0 |

**Category total (split path, default config): ~0.61 ms/step at roofline, ~1.15 ms/step realistically — about 2.4% of the 47.6 ms budget.**
**With `ATLAS_HC_SPLIT=0`: ~17.5 ms/step, ~37% of budget.** The split path is doing most of the work already.

Verify-path deltas (M=6): all mHC and norm kernels scale ~linearly in bytes but the **grid stays 1-wide in `num_tokens`** for the unsharded `hc_post` calls (`multi_seq/mod.rs:355`, `:490`, `:543`), so a 6× byte increase lands on 6 of 48 SMs. `top2_bf16_rows` (`argmax_bf16.cu:66`) adds ~5.7 µs roofline / ~25 µs realistic per verify step. `fused_qkv_norm_rope_cache_write_bf16` (`fused_verify_elemwise.cu:78`) contributes ~18 µs but removes ~2064 launches.

Zero-cost on the V4 decode path (documented for completeness): all `dequant_*`, `widen_block_scale_f32`, `transpose_u8`, `quantize_bf16_to_nvfp4`, `e2m1_quantize`, `lora_bgmv_*`, `bf16_add_inplace`, `vector_add`, `kv_cache_append`, `kv_block_indirect_copy`, all `*_vanilla` and mRoPE kernels, `hc_mean`, and the FP32/gated `rms_norm` variants.

---

# Fusion opportunities — adjacent kernels touching the same buffer

Ordered by value. Each row names two kernels that are launched back-to-back and where the second reads exactly what the first wrote.

| # | Buffer | Producer | Consumer | Saving/step | Note |
|---|---|---|---|---|---|
| 1 | `hc_streams` `[1,4,4096]` FP32, 64 KiB | `hc_post` `hyper_connection.cu:398` (writes) | `hc_pre_mix` `hyper_connection.cu:228` (reads) | **~5.5 MiB ⇒ 21 µs** | 86 sites × 64 KiB. `hc_post` writes the highway, the next site's `hc_pre_mix` immediately reads all of it. A fused `hc_post_pre_mix` would keep the streams in L2/registers across the sublayer boundary. Blocked by the sublayer sitting between them — but the *pass-1 RMS* (`sum(x^2)` over the streams, `m == mix_hc`) can be computed **inside `hc_post`** for free, removing one of the 25 mix blocks and its 64 KiB read. |
| 2 | `y_out` / `hidden` `[1,4096]` BF16, 8 KiB | `hc_pre_finish` `hyper_connection.cu:291` | `rms_norm` `rms_norm.cu:45` via `decode_inner.rs:581`, `:725` | **~1.3 MiB ⇒ 5 µs** + 86 launches | `hc_pre_finish` writes the collapsed hidden vector and `rms_norm` reads it back on the very next launch. Both are one-block-per-token-shard kernels over the same 4096 lanes. `hc_pre_finish` already has the streams in registers; folding the RMS reduction in gives a `hc_pre_finish_norm` that emits `normed` directly. **Highest-confidence structural fusion in this document.** |
| 3 | Q tensor `[1,128,64]` BF16, 16 KiB | Q up-projection GEMV | `rope_forward_yarn_interleaved` `rope.cu:347` | **~1.4 MiB ⇒ 5.2 µs** + 43 launches | Pure elementwise pass over a GEMV output. `fused_verify_elemwise.cu:78` already proves the fusion for the verify path; the M=1 decode path (`decode_inner.rs`) does not use it. Extending `fused_qkv_norm_rope_cache_write_bf16` to n=1 collapses `rms_norm` + both RoPEs + the cache write into one kernel. |
| 4 | Attention output `[1,128,64]` BF16, 16 KiB | MLA attention kernel | `rope_forward_yarn_interleaved_inv` `rope.cu:417` | **~1.4 MiB ⇒ 5.2 µs** + 43 launches | The eq.26 de-rotation is a per-head elementwise map on the attention output, immediately before the o-projection. It belongs in the attention epilogue (or the o-projection prologue), not as a standalone HBM round trip. |
| 5 | Activation row `[1,4096]` BF16, 8 KiB | — (self) | `quant_rowwise_fp8.cu:38` / `per_token_group_quant_fp8.cu:39` | **~0.7 MiB ⇒ 2.6 µs** | Not a two-kernel fusion but a **within-kernel double read**: both quantisers read the tensor once for absmax and again to quantise. At block 256 / 4096 (16 elems/thread) and block 128 / 128 (1 elem/thread) respectively, the values fit trivially in registers across the reduction. One-line fix in each. |
| 6 | FFN intermediate BF16 | up-projection GEMV | `relu_squared` `relu_squared.cu:12` | **~0.7 MiB ⇒ 2.6 µs** | Activation as GEMV epilogue. Only applies to dense-FFN layers; the MoE path already fuses. |
| 7 | MoE output + residual | expert GEMV | `bf16_residual_add` `residual_add.cu:10` | **~1 MiB ⇒ 3.9 µs** | **Already solved** by `moe_weighted_sum_blend_residual_batchn` (`fused_verify_elemwise.cu:236`) — verify only that the M=1 decode path actually routes through it. |
| 8 | Hidden FP32 `[1,4096]`, 16 KiB | `embed_from_argmax_f32` `embed_from_argmax.cu:73` | `hc_expand` `hyper_connection.cu:38` | 16 KiB ⇒ 0.06 µs | Trivial in bytes, but removes a launch and lets the embedding gather write straight into all 4 stream planes. |
| 9 | K/V page | `fused_k_norm_rope_cache_write_fp8` `:252` | `reshape_and_cache_flash_fp8` `:152` | 43 launches | Both write the *same page* for the same token in the same step. Merging removes 43 launches and one page-table walk per step. Bytes are negligible; this is a latency/critical-path win. |
| 10 | Logits `[1,129280]` BF16, 252 KiB | LM-head GEMV | `argmax_bf16` `argmax_bf16.cu:14` | ~20 µs realistic | Not fusible (needs the full vector), but **shardable**: split the argmax over 256 blocks producing 256 partials, then one block finishes. Recovers ~25× on the dominant single-SM cost. |

---

# Concrete fix list

1. **`decode_inner.rs:664-677`** — change `ops::hc_post(...)` to `ops::hc_post_sharded(..., post_shards, ...)`, matching the sibling at `:757-772`. Grid goes (1,1,1) → (1,16,1) for 43 calls/step. No numerical change.
2. **`multi_seq/mod.rs:355`, `:490`, `:543`** — same substitution on the verify path.
3. **`quant_rowwise_fp8.cu:38`** and **`per_token_group_quant_fp8.cu:39`** — cache the row/element in registers across the absmax reduction; delete the second global read.
4. **`argmax_bf16.cu:14`** — two-stage sharded reduction over the vocab; also switch the shared-memory tree to `__shfl_down_sync` within warps.
5. **`fused_k_norm_rope_cache.cu:53/153/252`**, **`rope.cu:143`**, **`rope_mrope_interleaved.cu:34/109`** — replace per-thread FP64 `pow()` with the `__shared__ float s_freq[]` precompute already used at `rope.cu:29`.
6. **`hyper_connection.cu:398`** (`hc_post`) — convert the scalar `float` loop at `:418-427` to `float4`, matching the `hc_pre_mix` conversion at `:261`. Note this changes accumulation order (the `hc_pre_mix` comment records ~1e-7 drift as acceptable).
7. **`residual_add.cu`** (all 11 kernels), **`e2m1_branchless.cu:48`**, **`kv_cache_append.cu:20`** — vectorise to `uint4`/`float4`.
8. **Extend `fused_qkv_norm_rope_cache_write_bf16`** (`fused_verify_elemwise.cu:78`) to the n=1 serial decode path in `decode_inner.rs`.
9. **Shrink `hc_fn`** from FP32 to BF16 (or FP8 with a per-row scale). It is 1.5 MiB × 86 = 129 MiB/step; halving it saves **~0.25 ms/step** directly and is the largest remaining lever in this category. Requires a numerics A/B — the FP32 highway itself was found necessary, but `hc_fn` is a *weight*, not the highway, and the dot products already accumulate in FP32.
10. **Re-tune `hc_sinkhorn_iters`** below 20. Pure critical-path latency, 86×/step, single-threaded. Constraint: `hyper_connection.cu:168-171` — the final exact column projection must be retained (removing it regressed coherence onset ~150 → ~90 tokens).
