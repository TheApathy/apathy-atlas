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

---

# Round 2 — `prefill_attn_compressed_tc2`

Round 1 (`prefill_attn_compressed_tc`) landed a uniform **4.1-4.2x** over the
scalar kernel: 7.03 ms/call at the CSA config (S=2176, window=128, ratio=4)
and 18.30 ms/call at the HCA config (window=0, ratio=128, full-causal),
~490 ms/pass at N=2410. That is 5x short of the "10% of tensor-core peak"
target above, and the reason turns out to be that **the design's own diagnosis
was wrong twice**. Both errors are corrected below and in the kernel.

## The measurement that reframes everything

At S=2176 full-causal the grid is 64 heads x 136 row-blocks = 8,704 blocks,
each walking ~68.5 key tiles on average = 596,224 block-tiles. Each block-tile
retires 128 MMAs (4 warps x 32). At ~1.5 GHz, 18.30 ms = 27.45e6 cycles:

```
MMA cycles     = 596,224 x 128 / (48 SM x 4 quadrants) x 4 cyc = 1.59e6
wall cycles                                                    = 27.45e6
                                                       MMA share = 5.8%
per warp per key tile: 27.45e6 x 192 / (596,224 x 4)   = 2210 cycles
```

**94% of the kernel is not tensor-core work.** Round 1's issue mix, per warp
per key tile, all at 2-byte granularity (confirmed in SASS):

| what | count | conflict |
|---|---|---|
| `LDS.U16` rebuilding both B fragments | 128 | 2-way |
| `STS.U16` building the TRANSPOSED K tile | 64 | 4-way |
| `LDS.64` reading the 4 warps' S partials | 32 | 4-way |
| `STS`/`LDS` round-tripping P through smem | 16 | — |
| `HMMA.16816` | 32 | — |

169 shared-memory instructions per 32 MMAs, most of them replayed, against
8 warps/SM of latency hiding. This is an instruction-issue and occupancy
problem, not an MMA-pipeline problem.

## Correction 1: K never needed to be transposed

The doc above says (twice) that the `.col` B operand forces `sKT[dim][key]`.
It does not. `mma...row.col` means the B operand's **contraction index is the
contiguous one**, and the b register packs `B[k][n], B[k+1][n]` — two
consecutive **dims** of one key. That is exactly `sK[key][dim]`, the natural
layout. Round 1 paid 64 4-way-conflicted 2-byte stores per warp per tile to
build a transpose that was never required.

The 4-way conflict was not fixable by padding either: consecutive staging
lanes are 8 dims apart, so the transposed store's lane-to-lane bank stride is
`4*PAD mod 32`, always a multiple of 4 ⇒ at most 8 distinct banks for 32
lanes ⇒ a 4-way floor for any pad. Deleting the transpose is the only fix.

## Correction 2: P does not round-trip through smem

The doc says "the S output fragment layout does not match the A input fragment
layout, so P must round-trip through shared memory". For **this** mapping it
already matches. The replicated softmax reconstructs, per thread, column
`c = (cix/2)*8 + 2t + (cix%2)` for rows `g` and `g+8`, i.e.

```
en0[0..1] = P[g  ][2t, 2t+1]      = a0
en1[0..1] = P[g+8][2t, 2t+1]      = a1
en0[2..3] = P[g  ][2t+8, 2t+9]    = a2
en1[2..3] = P[g+8][2t+8, 2t+9]    = a3
```

which is verbatim the m16n8k16 A fragment. (This is the general fact that two
adjacent n-tiles' D fragments concatenate into the A fragment of the following
16x16 GEMM.) `sP`, its 16 smem ops and **the barrier that guarded it** are all
deleted.

## What tc2 does

| | tc | tc2 |
|---|---|---|
| K tile layout | transposed, `sKT[512][17]` | natural, `sKV[16][520]` |
| V tile | separate `sV[16][520]` | **same buffer** (V==K) |
| QK^T B operand | 64 `LDS.U16`, 2-way | 8 `ldmatrix.x4`, conflict-free |
| PV B operand | 64 `LDS.U16`, 2-way | 8 `ldmatrix.x4.trans`, conflict-free |
| S partials | `[warp][row][key]` f32, 32 loads 4-way | `[row][key]` float4, 8 loads conflict-free |
| P | bf16 via smem + barrier | stays in registers |
| smem / block | 38,720 B | **20,992 B** |
| CTAs/SM | 2 (smem-bound) | **3** (register-bound at 167) |
| barriers / key tile | 4 | **3** (4 only when V ≠ K) |
| smem instrs / warp / tile | 169 | **40** |

SASS confirms the intent: `16 LDSM.16.M88.4 + 16 LDSM.16.MT88.4 + 16 LDS.128
+ 32 STS.128 + 16 STS` and **zero** `LDS.U16`, 0 spills, 20,992 B smem.

### Why stride 520 makes both ldmatrix forms conflict-free

A row is `520 * 2 = 1040 B = 260 words ≡ 4 (mod 32)`. An ldmatrix phase reads
8 rows x 16 B; row `k` starts at bank `4k + c`, spanning banks `4k+c ..
4k+c+3`, so `k = 0..7` tiles all 32 banks exactly once. The same 4-word
row-stride trick sizes `sSp` at 17 float4 (`68 words ≡ 4 mod 32`).

### The aliasing question

`V == K` is always true at the model call site (MLA keeps V in K's buffer,
rope in the tail), and that is what lets one buffer feed both GEMMs. When it
is **not** true (microtest only) tc2 re-stages the same tile with V between
the QK^T and the PV — legal because barrier (3) has already retired every
warp's QK^T reads, and free of extra barriers on the aliased path. This keeps
smem at 20,992 B **unconditionally**, so no dynamic-smem opt-in and no
host/kernel size contract that can be got wrong. (The Rust launcher *can* set
`CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` — `KernelLaunch::shared_mem`
→ `AtlasRegistry::launch_on_stream` does it automatically above 48 KB — it
simply is not needed here.)

## Levers REJECTED, with the arithmetic

- **KT=32.** Halves barriers per key traversed, but *every* per-key cost
  (staging, both B-fragment streams, sSp) scales linearly with KT, so nothing
  improves per key. Cost: `sKV 32x520x2 = 33.3 KB + sSp 32x17x16 = 8.7 KB =
  42 KB` ⇒ 2 CTAs/SM, down from 4-by-smem. Trading 33% of the occupancy for
  2 fewer barriers per 32 keys is the wrong direction when the stall is
  latency. (KT=24 dodges the smem cliff at 31.9 KB / 3 CTAs but makes KT no
  longer 2 n-tiles, complicating the softmax reconstruction for no new win.)
- **cp.async.cg double buffering.** Needs a second 16.6 KB tile ⇒ 37.6 KB ⇒
  back to 2 CTAs/SM, the same trade as KT=32. And the global side is not the
  problem: K/V is 2.5 MB at N=2410 and is served from L2, which is why the
  measured HCA time (18.3 ms) is less than half the 42 ms that 11.6 GB of
  DRAM re-streaming would cost. With 3 CTAs/SM the remaining global latency is
  covered by the other blocks.
- **`window == 0` compile-time specialization.** The per-row window bound is
  `v0 && c < COUNT && kp >= LO && kp < HI` — ~32 ALU ops per thread per tile
  against ~2200 cycles. Unmeasurable.
- **BR=32 q-rows per block** (the other half of the HCA lever: halve the K
  re-streaming and reuse each B fragment across two m-tiles; per-16-row smem
  traffic drops ~35%). Rejected for round 2 on register arithmetic:
  `o_acc 16 n-tiles x 2 m-tiles x 4 f32 = 128` plus
  `qa 8 k-steps x 2 m-tiles x 4 u32 = 64` = 192 registers before any
  transient, so ~220-234 live. That is under the 255 cap but pins occupancy
  at `65536/(2*128) = 2` CTAs/SM — 2 CTAs x 32 rows = 64 rows/SM, the same
  row throughput as tc2's 3 CTAs x 16 = 48 rows/SM only if nothing spills,
  and the doc's own trap 1 (a 768-byte stack frame erased an earlier win)
  says the downside is worse than the upside is good. Splitting BR=32 over 8
  warps instead keeps registers flat but turns the S partial into an 8-way
  reduction (sSp read traffic 16 KB → 128 KB per tile). **Revisit only with
  the warp-0 softmax reduction below in place.**
- **`__launch_bounds__(128, 4)`** to buy the 4th CTA the smem now allows:
  ptxas hits 128 registers but spills (40 B stack frame, 80 B spill stores,
  68 B spill loads). Left off; 167 registers / 3 CTAs with 0 spills is the
  shipped configuration. This is a one-line A/B if the hardware says
  otherwise.

## Next lever, if tc2 is not enough

The remaining smem traffic per block-tile is roughly `16 KB stage + 16 KB
QK-B + 16 KB PV-B + 4 KB sSp write + 16 KB sSp read ≈ 68 KB`, i.e. the S
partials are now the single largest non-fragment term at ~29%. Having **warp 0
alone** reduce the partials and broadcast `eo`/`P` would take that to ~10 KB
(≈ -15% of all smem traffic) and remove three quarters of the redundant
`__expf` MUFU work, at the cost of serializing the softmax in one warp — which
is cheap at 3 CTAs/SM because the other blocks fill the idle warps. That is
also the precondition that makes BR=32 affordable.

## Validating round 2

```
cargo run --release -p spark-model --example prefill_attn_tc_microtest \
    --features cuda,gpu-examples -- 7C21 2176 128 4     # CSA
cargo run --release -p spark-model --example prefill_attn_tc_microtest \
    --features cuda,gpu-examples -- 7C21 2176 0   128   # HCA, full causal
```

The microtest gates all three kernels. Because tc2 is a data-movement-only
rewrite — same MMA operand *values*, same sSp summation order, same softmax
terms, same bf16 rounding of P — it is held to `cos >= 0.9999999` against tc
(only FMA re-association may separate them) on top of the standard
`0.999 / 0.995` bar against the scalar reference, at S=253 aliased and S=160
NOT aliased (which drives the V re-stage path), crossed with CSA and HCA.
A tc2-vs-tc failure therefore points at an ldmatrix fragment mapping, not at
numerics.

In the engine, `ATLAS_V4_PREFILL_TC2=1` swaps tc2 in at both call sites
(compressor and full-attention); unset keeps tc; `ATLAS_V4_PREFILL_TC=0`
still falls all the way back to the scalar kernel.
