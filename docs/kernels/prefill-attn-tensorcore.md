# Prefill attention → tensor cores: the 2.6× that gets prefill to ~950 tok/s

Status: **specified, not implemented.** Everything below is measured on GB10
with DeepSeek-V4-Flash-162B unless marked as arithmetic.

## Why this is the whole job

Current prefill budget, N=911, 43 layers, `ATLAS_PROFILE=1`
(NVFP4 attention + `ATLAS_V4_ATTN_RELEASE_BF16=1`, 385 tok/s clean):

| stage | ms/pass | %   | state |
|---|---|---|---|
| **ATTN core attention** | **864.6** | **45%** | **scalar FP32, 2.2 TFLOPS** |
| MoE grouped_gate_up | 369.6 | 19% | 149 of 183 GB/s — at ceiling |
| ATTN q_latent_expand | 261.1 | 14% | GEMM at ~11 TFLOPS |
| MoE grouped_silu_down | 214.1 | 11% | at ceiling |
| ATTN kv_proj | 119.0 | 6% | |
| MoE shared_expert | 56.9 | 3% | |

`prefill_attn_compressed` does **1.90 TFLOP in 864.6 ms = 2.2 TFLOPS**. GB10
tensor cores do 250–500 TFLOPS on BF16. Even at 10% tensor-core efficiency
that is an 11× cut on 45% of prefill.

Arithmetic to the target:

```
today                            1921 ms →  474 tok/s (profiled basis)
attention → tensor cores (11×)   1134 ms →  803 tok/s
+ q_latent_expand GEMMs (3×)      960 ms →  949 tok/s
```

**~950 tok/s is reachable without touching weights.** The remaining gap to the
reference's ~1010–1055 is quant-dependent: their D2R MoE GEMM dequants
IQ2_XXS/Q2_K expert weights straight into MMA fragments, ~2-bit against our
4-bit MXFP4. Our MoE is already at its bandwidth ceiling, so there is no
kernel win left there — only a requantization, which is out of scope.

## What NOT to redo

The memory side of this kernel is already fixed (commits 848e1e5f, c2bca29e):
K/V staged in shared memory, conflict-free interleaved lane→dim mapping,
uint4 loads, Q hoisted to registers. That took it 139 → 20.7 ms/layer (6.7×).
Global K/V traffic is now ~7% of its time. **What remains is purely the scalar
FP32 math.** Do not re-profile the memory path; profile the MMA pipeline.

## The shape

Per block today: 128 threads = 16 q-rows × 8 dim-lanes, grid
`(num_q_heads, ceil(S/16), 1)`. head_dim = 512, sliding_window = 128,
CSA ratio 4 → a row attends ~144 raw + ~227 compressed keys at N=911.

Two GEMMs per key tile, both `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`
— the same instruction `moe_w4a16_grouped_gemm.cu:164` already uses, so copy
its fragment loading and `asm volatile` block rather than writing new PTX:

1. **S = Q·Kᵀ** — `[16 rows × 512] × [512 × KT] → [16 × KT]`.
   M=16, N=8, K=16 ⇒ 32 k-steps of the 512-dim contraction, KT/8 n-tiles.
   Accumulate in f32. Q is A (row-major), K is B (col-major — K is already
   stored key-major in smem, which IS the col-major operand layout).
2. **O += P·V** — `[16 × KT] × [KT × 512] → [16 × 512]`.
   P is the softmax-weighted S, converted to bf16 for the A operand.

Online softmax between them is unchanged in structure: running `m`, `l`, and
the `o_acc` rescale by `eo`.

## The three traps

These cost real time in the memory-side rewrite; they apply again.

1. **Compile-time loop bounds.** A runtime chunk/tile count makes per-thread
   arrays dynamically indexed; ptxas cannot register-allocate them and they
   land in local memory. A 768-byte stack frame was measured and it erased the
   win. Keep `NCHUNK`/`KT` as `#define`s and check `--resource-usage` shows
   `0 bytes stack frame, 0 bytes spill`.
2. **Fragment alignment.** `__shared__ __nv_bfloat16 x[]` is 2-byte aligned by
   default; `ldmatrix`/`uint4` reads need 16 bytes. Keep `__align__(16)` on
   every staged tile.
3. **Masking must stay an exact no-op.** Rows walk the tile union and mask
   out-of-window keys with `score = -INFINITY`, which is exact:
   `m_new = max(m, -inf) = m` ⇒ `eo = exp(0) = 1`, `en = exp(-inf) = 0`.
   **Never use a large negative constant** — at the initial `m = -1e30` a score
   of `-1e30` gives `en = exp(0) = 1` and folds a spurious key. With MMA the
   mask must be applied to the S fragment AFTER the mma, before the softmax.

## Validation

The interleaved mapping already reorders summation, so this is not bit-exact
against BF16 and must be validated behaviourally:

- `tool-eval-bench --short` is the gate. **90/100, 12 pass / 3 partial / 0
  fail** is the current bar; it runs in ~330 s. TC-14 is the known-borderline
  scenario (it flips on numerics); the other 14 must not move.
- `scratchpad/prefill_scan.py 64 256 512 1024 1800` — the curve must RISE with
  length and plateau. A flat curve means batching regressed.
- `scratchpad/quality_probe.py` for a fast smoke test (the bat-and-ball
  question is a good trap: correct answer is 5 cents).

## Instrumentation that works here

- `ATLAS_PROFILE=1` + the `aprof!` probes in `prefill/cache_skip_v4.rs`
  attribute 91% of prefill wall. This is the reliable instrument.
- **nsys cannot profile prefill on this model.** The ~6-minute weight load
  overruns its buffers and the prefill trace is silently dropped — a run
  captured only 2 load-time quantize kernels, and `--delay` did not help.
  Do not spend time on it.

---

# Concrete design (fragment layouts extracted from the working in-tree MMA)

`moe_w4a16_grouped_gemm.cu:148-168` is a VERIFIED m16n8k16 block on this
target. Its index math is reproduced here so the kernel can be written in one
pass instead of rediscovering PTX operand layouts.

With `group_id = laneid >> 2` and `tid = laneid & 3`:

```
A [16 x 16] row-major, a_stride = row length:
    fr0 = warp_m_offset + group_id;  fr1 = fr0 + 8
    fc0 = tid*2;                     fc1 = fc0 + 8
    a0 = A[fr0][fc0..fc0+1]   a1 = A[fr1][fc0..fc0+1]
    a2 = A[fr0][fc1..fc1+1]   a3 = A[fr1][fc1..fc1+1]   (2 bf16 packed per u32)

B [16 x 8], indexed sB[k][n], b_stride = n length:
    nc = nt*8 + group_id;  k0 = tid*2;  k1 = k0 + 8
    b0 = B[k0..k0+1][nc]      b1 = B[k1..k1+1][nc]

D/C [16 x 8] f32, 4 per thread:
    acc[0],acc[1] -> row group_id,     cols tid*2, tid*2+1
    acc[2],acc[3] -> row group_id + 8, cols tid*2, tid*2+1
```

## Block mapping — CORRECTED: head_dim=512 forces a dim split

An earlier revision of this doc said "give each warp its own 16 q-rows so the
softmax never crosses warps". **That is infeasible here and would have been
found only after writing the kernel.** Checked arithmetic:

```
O accumulator for a [16 x 512] tile in ONE warp:
    512/8 = 64 n-tiles of m16n8 x 4 f32/thread = 256 f32 REGISTERS per thread
    (budget is ~166 total, measured on the current kernel)
Q held in registers across the key loop:
    512/16 = 32 k-steps x 8 bf16 = 128 u32 per thread — on top of the above
Q staged in smem instead, 4 warps x 16 rows:
    sQ 64 KB + sKT 16 KB + sV 16 KB = 96 KB  (48 KB static limit)
```

head_dim = 512 is 4-8x the 64-128 that flash-attention layouts assume, so a
warp cannot hold an output tile spanning the full head_dim. This is precisely
why the current scalar kernel splits dims across 8 lanes (64 dims and 64 f32
`o_acc` per thread).

**Correct mapping: the BLOCK owns 16 q-rows; warps split the head_dim.**
- 4 warps x 128 dims each ⇒ 16 n-tiles ⇒ 64 f32 accumulators per thread,
  which matches today's register footprint and is known to fit.
- QK^T: every warp needs the same S, so either compute it redundantly per warp
  (cheap — it is 16x KT) or compute once and share via smem.
- **The softmax row max/sum DOES cross warps** and must go through shared
  memory with a `__syncthreads()`. Budget for it; it is unavoidable at this
  head_dim, not a design smell.
- P then round-trips through smem for the PV A-operand anyway (the S output
  fragment layout does not match the A input layout), so the same smem
  staging serves both.

## The two GEMMs

**1. S = Q·Kᵀ → [16 rows x 8 keys], contraction over head_dim = 512 (32 k-steps).**
A = Q tile, row-major `[16][head_dim]`, natural layout.
B must be indexed `sB[k][n]` = `[dim][key]` — **K must be staged TRANSPOSED**
(`sKT[dim][key]`), not in today's `sK[key][dim]`. Do the transpose while
staging from global; pad the key stride to dodge bank conflicts.

**2. O += P·V → [16 rows x 8 dims], contraction over keys.**
B is indexed `[key][dim]`, which IS today's natural `sV[key][dim]` layout —
no transpose needed here.
A = P (softmax-weighted S) — the **S output fragment layout does not match the
A input fragment layout**, so P must round-trip through shared memory. This is
standard for flash attention; budget smem for it.

## Online softmax on the fragment

A row's 8 columns live across the 4 threads sharing a `group_id` (consecutive
lanes), so the row max/sum is `__shfl_xor_sync` over lanes 1 and 2 within that
group of 4, then combined with the running `m`/`l` exactly as the scalar
version does. Masking is applied to the S fragment AFTER the mma and BEFORE
the softmax — `-INFINITY`, never a large negative constant (see trap 3).

## Expected

Core attention 864.6 -> ~78 ms/pass at 10% tensor-core efficiency; prefill
474 -> ~800 tok/s, and ~950 with the q_latent_expand GEMMs. Anything under
~5x on this stage means the MMA pipeline is stalling — check smem bank
conflicts on the transposed K tile first.
