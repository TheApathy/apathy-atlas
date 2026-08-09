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
