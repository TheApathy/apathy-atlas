# exl3_gemv — EXL3 trellis 3.0 bpw decode GEMV (GB10, sm_121)

S4 of `docs/EXPERT-3BPW-PLAN.md`: the offline bring-up of the M=1 decode GEMV
that consumes EXL3 trellis-coded expert weights, validated bit-exactly against
a CPU oracle on SYNTHETIC trellis data (no checkpoint required — the format
has no metadata, so random i16 words are a valid payload).

- Kernel: `kernels/gb10/common/exl3_gemv.cu` (module `exl3_gemv`, entries
  `exl3_gemv_m1`, `exl3_dequant_dump`)
- Oracle: `crates/spark-model/examples/exl3_gemv_microtest.rs`
- Decode logic ported from ExLlamaV3 (MIT, Turboderp 2025), vendored at
  `/home/flocka/sparkinfer-upstream/b12x/gemm/trellis_linear/csrc/vendor/quant/`
  (`codebook.cuh`, `exl3_dq.cuh`, `hadamard_inner.cuh`,
  `exl3_gemm_inner.cuh`); vendored rev 704aefd7, checkpoint rev 787d1582.

## 1. Format, as verified against the reference source

Per-matrix tensors (safetensors, e.g. w1 K=4096 N=2048):

| tensor | dtype/shape | meaning |
|---|---|---|
| `trellis` | I16 `[K/16, N/16, 48]` | 48 u16 = 96 B per 16×16 tile = exactly 3.000 bpw |
| `suh` | F16 `[K]` | input-side random-sign vector |
| `svh` | F16 `[N]` | output-side random-sign vector |
| `mcg` | I32 scalar | 3INST cb=1 multiplier `0xCBAC1FED` (compile-time in all kernels) |

**Bit stream.** The 48 u16 of a tile, read as 24 little-endian u32, form one
CIRCULAR 768-bit stream. Bit `g` of the stream lives in u32 word `g/32` at
bit position `31 - g%32` (words fill MSB→LSB). Weight `t` (t = 0..255 in tile
linear order) is the 16-bit window **ending** at bit `((t+257)*3) mod 768` —
windows overlap and advance 3 bits per weight; the first weights' windows wrap
around the end of the stream. Verified equivalent to the vendored `dq8`
(bits=3, align=4) extraction over 256 K random windows, 0 mismatches.

**Decode (3INST, cb=1 "mcg").** Per 16-bit window `w`:

```
x = w * 0xCBAC1FED                    // u32 wrap mul
x = (x & 0x8fff8fff) ^ 0x3b603b60     // lop3 immLut 0x6a == (a&b)^c
val = fp16(x.lo16) + fp16(x.hi16)     // IEEE fp16 add, RN
```

Decoded values lie in (−4, 4). There are **no group scales, zero points, or
in-memory codebook tables** — the effective payload is exactly 3.0 bpw plus
0.39 % for suh/svh.

**Tile linear order → (k, n).** The quantizer packs tiles in the
`mma.m16n8k16` B-fragment order consumed by
`dq_dispatch(shb, lane_id << 3, frag_b[n2], frag_b[n2+1])`. With
`lane = t/8`, `s = t%8`:

```
n_in_tile = 8*(s/4) + lane/4
k_in_tile = 2*(lane%4) + (s%2) + 8*((s%4)/2)
```

Tile (kb, nb) of the tensor holds B[k, n] for k in kb*16.., n in nb*16..
(B = Wᵀ; the GEMM computes C = A·B).

**Hadamard / sign vectors — SCOUT CORRECTION.** The rotation is a
**blockwise-128 Sylvester–Hadamard transform** (`H[i][j] = (−1)^popcount(i&j)`
per aligned 128-chunk, normalized 1/√128, computed in fp32 via a 4-element
in-register stage + 32-lane warp-shuffle stage), **not** a 16-point per-tile
transform. At inference time (nothing folded into stored weights):

```
x' = H128( diag(suh) · x ) / sqrt(128)        // input pass, along K
y0 = Bᵀ x'                                     // trellis GEMV
y  = diag(svh) · H128( y0 ) / sqrt(128)       // output pass, along N
```

`suh` is multiplied BEFORE the input Hadamard; `svh` AFTER the output
Hadamard (reference: `had_hf_r_128_inner<true,false>` on input,
`had_ff/fh_r_128_inner<false,true>` on output in `exl3_gemm_kernel` /
`output_had_sh_gl`).

## 2. Kernel design (`exl3_gemv_m1`) — tuning rounds 2–3 (2026-08-10)

Grid `(N/128, SPLIT_K)`, block 256 threads (8 warps), static smem ~16.6 KB,
`__launch_bounds__(256, 4)` → **4 CTAs/SM** (reg cap 64, exactly met, no
spills; bring-up build was 52 regs at 42.5 KB smem = 2 CTAs/SM).

- **Strip ownership**: block owns a 128-wide output strip = 8 tile-columns,
  which makes the output Hadamard-128 block-local. Warp w owns tile-column w;
  each lane accumulates 2 outputs (`n = 16w + lane/4`, `+8`).
- **Phase 1 — input pass**: x' for one SUPERBLOCK of the K-slice
  (≤ 8 chunks = 1024 k), computed in fp32 via the warp-shuffle Hadamard,
  stored as packed `__half2` (k, k+1) pairs (2 KB smem — round 4), refilled
  per superblock inside the stage loop — the per-slice K cap is gone. At the
  production geometry a CTA runs 4–6 stages, so the refill never fires after
  the prologue. Issued AFTER the first trellis cp.async stages so the DRAM
  stream starts immediately.
- **Phase 2 — weight stream**: **3-stage** cp.async (`.cg`, 16 B per thread)
  ring; each stage = 8 tile-rows × 8 tiles = 6 KB = exactly one 128-k
  chunk, fetched as contiguous 768-B runs per tile-row (stride N/16·96 B
  between rows), unpredicated (every stage is provably full). Every trellis
  byte is read exactly once, as 128-bit transactions; the copy geometry gives
  ~427 B per warp-request (2 of 3 warps issue one 512-B run, the third two
  256-B runs) — see §3c for why that width is load-bearing. Stage `s+2` is
  re-armed at the TOP of iteration `s`, which makes the trailing barrier
  redundant: **ONE `__syncthreads()` per 6 KB stage** (was two, rounds 1–3)
  and a fetch window of two compute periods (was one). A warp decodes one
  96-B tile per iteration: `dq8` gives 8 weights/lane from two u32 smem
  words. The dot runs in `__half2` HFMA2 chains within each 128-k chunk — 4
  independent chains ({acc0,acc1} × even/odd tile-row, depth 4 each) —
  combined into fp32 in FIXED order once per chunk (round 3; numerics tier
  in §3b). The row loop is fully unrolled (8 tiles per stage per warp).
- **Phase 3 — reduce + output**: quad shuffle-reduce → 128 fp32 partials in
  smem. `SPLIT_K = 1`: the block applies Hadamard-128 + svh and stores bf16.
  `SPLIT_K > 1`: each split publishes its raw partial to `ws[split][N]`; an
  atomic counter elects the LAST split, which combines partials in fixed
  split order (deterministic), then does the output pass. Counters self-reset
  → back-to-back launches need no host memset.

Constraints: `N % 128 == 0`, `K % 128 == 0`; any `SPLIT_K ≥ 1` (the x'
superblock loop removed the old ≤4096-K slice cap). `SPLIT_K` is a
GRID-FILLING knob only: at 4 CTAs/SM the GB10 has 192 slots, and the FUSED
launches multiply by `2·top_k` / `top_k` groups, so the production policy
(`Exl3MoeState::split_for`) targets ~96 CTAs per group and then rounds up to
the first split making `strips·split·groups` an exact multiple of 192. Note
that filling the grid by splitting harder does NOT buy bandwidth — see the
measured sweep in §3c.

Shared-memory budget: `s_x` 2 KB + `s_stage` 3×6 KB + `s_y`/`s_elect`
~0.6 KB = **20,996 B** (ptxas) → 4 × (20,996 + 1 KB driver reserve) = 88 KB
on the 100 KB GB10 SM; registers are the binding limit (64 regs × 256 thr ×
4 CTAs = the full 64 K file), so the third stage buffer costs no occupancy.

## 3. Dequant instruction budget

**Measured diagnosis (2026-08-10 hardware round)**: the bring-up kernel
plateaued at ~156 GB/s across splits 4–12 at both production shapes
(~1.95 GB/s per CTA × 80 CTAs), against a 229 GB/s ceiling that sibling
GEMVs reach. A fully-contiguous diagnostic shape ran ~6.7 GB/s per CTA, so
the DRAM pattern was not the constraint — the kernel was
**issue-latency-bound**: the dequant was one DEPENDENT chain (serial window
shifts → u32 mul → lop3 → hadd), effective IPC ≈ 1, ~19 µs per 3.158 MB
matrix ≈ 166 GB/s-equivalent. Fix = break the chains (ILP) + more warps
(occupancy):

Per 96-B tile (256 weights) per warp, per lane (8 weights), round-2 sequence:

| stage | lane-ops | depth |
|---|---|---|
| 2 × LDS.32 (tile words a, b) | 2 | 1 |
| span align: `mlo = SHF.R(b,a,s2)`, `mhi = a>>s2` (s2 ∈ {0,8,16,24}) | 2 | 1 |
| odd windows: 3 × immediate funnel shift (w5,w3,w1; w7 = mlo free) | 3 | 1 |
| even windows: 4 × independent `>>3` | 4 | 1 |
| mask: 8 AND | 8 | 1 |
| 3INST (8 IMAD + 8 LOP3 + 8 PRMT lo/hi + 4 HADD2) | 28 | 4 |
| half2→float2 (4 × cvt) | ~8 | 1 |
| x' loads (2 × LDS.64, k-pairs shared between both n-halves) | 2 | 1 |
| dot (8 FFMA, 4 accumulator chains: acc{0,1} × even/odd row) | 8 | 2 |

≈ **65 lane-ops / 8 weights ≈ 8.1 ops/weight**, but now as **four
independent window-pair chains** (each window is a pure function of
`(mlo, mhi)`) instead of one depth-7 serial chain — the vendored align=4
`dq8` derives w6..w4 by serial `>>3` from w7, which is what capped IPC.
The even/odd-row accumulator split removes the serial FFMA tail across
tiles. Expected effective IPC ≈ 2 per warp; combined with 4 CTAs/SM
(32 warps vs 16) the issue side stops binding: the dependent-chain time
~19 µs drops under the 13.8 µs DRAM floor → expected **~200–229 GB/s**
(gate 3 verifies on hardware).

The lane geometry guarantees the restructure is safe: for `t = 8·lane`,
`b0 % 32 ∈ {19, 11, 3, 27}`, so the 37-bit window span never crosses three
u32 words and `s2 ∈ {0, 8, 16, 24} < 32` (single funnel-shift alignment).
Bit-exactness is unchanged — every window is masked to 16 bits, and bits
`[s2+3j, s2+3j+16)` of the 64-bit pair are identical whether reached by one
funnel shift or truncate-then-shift.

**Round-2 post-mortem by SASS (2026-08-10, no-GPU round): why it only
gained 8%.** Both the bring-up and round-2 kernels were compiled with
`nvcc -arch=sm_121a -cubin --resource-usage` and nvdisasm'd:

- **No spills either round** (r1: 52 regs, r2: 64 regs exactly — the
  `__launch_bounds__(256,4)` target was met cleanly, ptxas did not
  serialize).
- **The expected IPC win never existed**: ptxas had ALREADY parallelized
  the bring-up kernel's "serial" `>>3` window chain — the r1 loop extracts
  windows with independent `SHF.R.U64` funnel shifts straight from the
  (a, b) word pair. Measured loop bodies: r1 = 262 SASS instr / 4 tiles,
  r2 = 269 / 4 tiles — the SAME ~65–67 instructions per 96-B tile, with a
  well-interleaved schedule in both. Round 2's +8% (156→168) was the
  occupancy doubling (2→4 CTAs/SM), not ILP.
- **The real bound is warp-issue throughput, not LDS**: per-tile LDS
  traffic was 2×LDS.32 (tile words) + 2×LDS.64 (x', 2 wavefronts each) =
  6 LSU wavefronts and 768 B read per 96-B tile. At the 229 GB/s target
  that is ~38 GB/s/SM of shared traffic vs the ~218 GB/s/SM (128 B/cycle)
  shared ceiling — 17%, not binding. What binds is the ~67-op stream per
  tile: the r2 opcode census per 4 tiles is 32 FFMA + 32 HADD2.F32 (the
  fp32 dot tail = 24% of all issue) + 64 LOP3 + 36 SHF + 39 IMAD +
  32 PRMT + 16 LDS + ~17 HADD2/HFMA2.

**Round 3 (this round): shrink ops/byte — half2 dot + half2 x'.** The
only lever that moves an issue-bound kernel is fewer instructions per
trellis byte:

| change | per-tile effect |
|---|---|
| accumulate `d·x` in `__half2` HFMA2 chains, cvt to fp32 once per 128-k chunk | −8 HADD2.F32, −8 FFMA, +4 HFMA2 (+~1 amortized chunk-combine) |
| store x' as packed `__half2` pairs | 2×LDS.64 → 2×LDS.32 (6→4 LSU wavefronts, 768→384 LDS B/tile, s_x 8→4 KB) |
| full unroll of the 8-row stage loop | loop overhead → 0, whole-stage scheduling window |

Verified in the round-3 SASS: the stage loop is **447 instr / 8 tiles =
55.9/tile** (−17% vs 67), FFMA count in the loop is 0, HADD2.F32 dropped
32→~4 per stage (the per-chunk combine), all LDS are 32-bit, still 64 regs
/ 0 spills / 16.9 KB smem / 4 CTAs/SM. Expected from pure issue scaling:
168 × 67/55.9 ≈ **~200 GB/s**, plus whatever the shorter FMA chains and
halved LSU wavefronts recover of the 1.4-of-4 effective IPC — estimate
**~195–215 GB/s** (gate 3 on hardware decides; ceiling 229).

## 3c. Round 4 (2026-08-12, no-GPU round): the SASS CYCLE budget

Rounds 1–3 all reasoned in *instructions per tile* and went
156 → 168 → 168 GB/s. Round 3's null result was not bad luck: the budget
below shows instruction count was never within 3× of binding. This section
replaces the "warp-issue-bound" claim in §3.

### Inputs (all measured or derived, nothing assumed)

`nvcc -arch=sm_121a -cubin -O3 --resource-usage` + `nvdisasm -c`, round-3
source:

| quantity | value |
|---|---|
| `exl3_gemv_m1` registers / spills / smem | 64 / **0** / 16,900 B |
| CTAs/SM (both limits) | 4 — regs bind exactly: 64 × 256 × 4 = the 64 K file |
| stage loop | 618 SASS instr |
| compute region (8 tiles, 1 warp) | **446 instr = 55.75 / 96-B tile** |
| compute opcode census / stage / warp | 131 LOP3, 73 SHF, 70 IMAD, 64 PRMT, 49 HFMA2, 32 LDS, 21 HADD2 (338/446 = 76 % integer ALU) |
| `BAR.SYNC` in the steady stage loop | **2** (+1 per x' superblock boundary) |
| stage issue | 48 instr, `LDGSTS.E.BYPASS.128`, predicated |
| GB10 SM clock (`nvidia-smi`) | 2,398 MHz current, 3,003 MHz max → budget at **2.4 GHz** |

Shape = the microtest GATE3 production case, N=2048 K=4096, at the split that
measured best (8):

- payload = N·K·3/8 + (N+K)·2 = **3,158,016 B**; measured 166 GB/s ⇒
  **19.02 µs**
- CTAs = 16 strips × 8 splits = **128** of 192 slots ⇒ 67 % fill, ONE wave,
  2.67 CTAs/SM = 21.3 warps of the SM's 48 = **44 % warp occupancy**
- per CTA: 32/8 = **4 stages** × 6,144 B = 24,576 B trellis + ~2,048 B
  (A + suh slice) + 512 B ws

### The budget, per SM, over the 19.02 µs kernel

| term | derivation | cycles @2.4 GHz | % of wall |
|---|---|---:|---:|
| wall clock | 19.02 µs × 2.4 GHz | 45,650 | 100 % |
| **issue** | 2.67 CTA × 8 warp × 4 stage × ~460 instr ≈ 41.8 K warp-instr ÷ 4 slots/cycle | 10,450 | **22.9 %** |
| LDS / shared pipe | 32 LDS-instr/warp/stage → 2,731 wavefronts × 128 B ÷ 128 B/cycle | 2,731 | 6.0 % |
| `BAR.SYNC` fixed cost | 2/stage × 4 × 2.67 = 21 barriers × ~30 cyc | 640 | 1.4 % |
| **memory stall (residual)** | — | ~35,200 | **≈ 78 %** |

Inverting the issue term gives the ceiling directly: 6,144 B per
(8 warps × 460 instr) = **1.67 B per warp-instruction**, so at 4 instr/cycle/SM
the issue-throughput ceiling is 6.68 B/cycle/SM × 48 SM × 2.4 GHz =
**≈ 770 GB/s** — 3.4× the 229 GB/s DRAM ceiling (481 GB/s even at a throttled
1.5 GHz). Round 3's −17 % instructions moved that ceiling from 640 to
770 GB/s and the wall clock by under 4 %, which is exactly the "no measurable
change" that was observed. **Instruction count is closed as a lever.**

LDS is likewise closed: 32 × 128 B / 6,144 B = 5.33 shared bytes per DRAM
byte, i.e. 25 GB/s/SM at the 229 GB/s target against a 307 GB/s/SM shared
ceiling (8 %).

### Testing the doc's three suspects against that budget

**(b) GRID / WAVE QUANTISATION — checked first, and it is MEASURED-DEAD.**
CTA counts at N=2048 (16 strips), against 192 slots:

| SPLIT_K | 1 | 2 | 3 | 4 | 6 | 8 | 12 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CTAs | 16 | 32 | 48 | 64 | 96 | 128 | **192** |
| wave fill | 8 % | 17 % | 25 % | 33 % | 50 % | 67 % | **100 %** |
| stages/CTA | 32 | 16 | 10.7 | 8 | 5.3 | 4 | 2.67 |
| measured GB/s | 135 | …151 → | | 156 | | 166–168 | 168 |

Only SPLIT_K = 12 fills a wave exactly (N=4096: 32 strips, SPLIT_K = 6) — and
**the sweep already ran it**. Going from 8 % fill to 100 % fill is 12× the
CTAs each doing 12× less work, and it moved the aggregate 135 → 168 GB/s
(+24 %) with per-CTA rate collapsing 8.4 → 0.87 GB/s. That is an aggregate
saturating well below the ceiling, not a wave sawtooth: wave quantisation
would show a jump at S=12 and it does not. **(b) does not explain the
plateau.**

Production is already wave-exact by construction: CTAs =
`strips · split · groups` with `split = 96/strips` ⇒ **96 · groups**.
gate/up carries `groups = 2·top_k` ⇒ 192·top_k = an exact multiple of a wave
for ANY top_k. `down` carries `groups = top_k` ⇒ 96·top_k, exact only when
top_k is **even** — the one real exposure, now that adaptive top-K can route
an odd width. Guarded in `Exl3MoeState::split_for` (round 4): the split is
walked up to the first value making `strips·split·groups % 192 == 0`, which
is a no-op at even top_k and lifts `down` from 3 to 6 at odd top_k. Both arms
take the FUSED group count so the per-slot fallback keeps the same SPLIT_K
and stays bit-identical (GATE8).

**(c) SPLIT-K COMBINE TAX — real, ~9 % in the microtest regime, second-order.**
At S=8: ws traffic = 128 × 512 B written + 16 elected × 8 × 512 B read =
128 KB against 3.146 MB = **+4.1 % DRAM**. Serial tail = `MEMBAR.SC.GPU`
(three in the SASS) + a global `atomicAdd` RTT + the elected block's ws
re-read + a **single-warp** output Hadamard/store on 16 CTAs ≈ 2,000–2,500
cycles ≈ 0.9 µs ≈ **5 %** of 19.0 µs, with the machine ~99 % idle for it.
K-slice evenness is clean at S=8 (32/8 = 4) but **not at the production
splits**: 32 chunks over 6 = 5,5,6,5,5,6 and 16 over 3 = 5,5,6, so a strip's
elected combine waits on a CTA doing 12.5 % more work. That is a straight
12.5 % loss in a one-wave launch but costs only in the LAST wave of the 8–16
wave production launch, which is why the policy is left alone rather than
forced onto divisors of `chunks_total` (doing so would trade the straggler
for a 12.5 % tail wave — a measured wash at best).

**(a) BARRIER CADENCE — what the budget implicates.** 78 % of the wall clock
is memory stall while the CTA's DRAM depth is capped at the double buffer
(12 KB, 6 KB at the wait point) and the whole 8-warp block is brought to a
stop-the-world checkpoint **every 6,144 bytes**. That is 1 barrier per
3,072 B per CTA. The comparison that settles it (all same GPU, same
directory, measured):

| kernel | barriers in the main loop | bytes per barrier | measured |
|---|---|---:|---:|
| `moe_shared_expert_fused_t` gate_up M=1 | **0** (0 `__syncthreads` in the whole kernel) | ∞ | 194–206 GB/s |
| `moe_shared_expert_fused_t` down M=1 | 0 (1 per CTA, before the K loop) | 32,768 | 194–206 GB/s |
| `dense_gemv_bf16`, `w4a16/w8a16_gemv`, `moe_expert_gemv` | 0 | ∞ | at ceiling |
| **`exl3_gemv_m1` (rounds 1–3)** | **2 per stage** | **3,072** | **166** |

`exl3_gemv.cu` is the ONLY GEMV in `kernels/gb10/common/` that uses cp.async
+ smem weight staging, and the only one stuck at 166. The fast ones are
thread-private register loops: each thread owns `VEC` adjacent output columns
for the whole K slice, so there is nothing to synchronise and each warp keeps
its own `GS/2` independent LDGs in flight.

**Why round 4 does NOT copy the sibling shape.** The same file records a
measured GB10 law (`moe_shared_expert_fused_t.cu:65-75`): *a 32-B per-warp
request pins a GEMV at ~130 GB/s no matter how many warps are resident; a
128-B request has no such ceiling (67 → 118 → 194 purely on added warps).*
**Width sets the ceiling, warp count sets how close you get.** The EXL3 tile
is a 96-B 16k × 16n blob, so a warp-private stream would fetch 96 B per
request (one tile-column, stride N/16·96 between k-steps) — under the width
the law says matters. The block-cooperative stage is exactly what buys the
current **~427 B average warp-request** (2 of every 3 warps issue one 512-B
contiguous run; the third issues two 256-B runs). Trading that for
barrier-freedom would very likely lose more than it gains. Round 4 therefore
keeps the stage and attacks the barrier.

### What round 4 changed

| change | effect |
|---|---|
| cp.async ring 2 → **3 buffers**; stage `s+2` re-armed at the **TOP** of iteration `s` | 2 barriers/stage → **1**; fetch window 1 → **2 compute periods** |
| `exl3_load_stage` tail predication removed | every stage is provably full (splits are chunk-aligned, one stage == one chunk, `nstages == c_hi − c_lo`); `LDGSTS` is now unpredicated, the branchy issue block is gone |
| x' superblock 16 → **8 chunks** (4 KB → 2 KB) | keeps smem at 20,996 B so 4 × (20,996 + 1 KB driver reserve) = 88 KB ≤ 100 KB — the third stage buffer is free |

The legality argument for the single barrier: at iteration `s` the re-arm
target is buffer `(s+2) % 3 == (s−1) % 3`, which held the stage consumed at
iteration `s−1`; the one barrier at the top of iteration `s` already orders
every warp past that consume, so the trailing barrier of rounds 1–3 is
redundant. The ring still holds exactly two groups in flight, so the
`cp.async.wait_group` immediates (1 steady, 0 on the last stage) are
unchanged.

ptxas after the change: **64 registers, 0 spills, 1 barrier, 20,996 B smem** —
4 CTAs/SM held. SASS: stage loop 618 → 612 instr, compute census unchanged
(131 LOP3 / 70 IMAD / 64 PRMT / 32 LDS / 50 HFMA2), exactly **one BAR.SYNC in
the steady path**.

**Expected: 185–205 GB/s, hardware decides.** Stated as a range on purpose —
round 3 published a point estimate for a lever that turned out to be closed.
If barrier cadence + fetch-window depth is the whole story the kernel should
approach the 194–206 GB/s the barrier-free siblings reach. **A null result is
itself the next datum**: if GATE3/GATE9 still land ≤ 175 GB/s after this, the
only surviving suspect is the DRAM request pattern — 768-B islands at a
12 KB (N=2048) / 24 KB (N=4096) stride — and the next experiment is a strip
width sweep (`EXL3_NSTRIP` 128 → 256, doubling the contiguous run to 1,536 B
at the cost of 2 tile-columns per warp), NOT more instruction tuning and NOT
more split tuning.

### Numerics: unchanged, bit-identically

Round 4 moves no arithmetic. The same 8 tiles are decoded from the same bytes
in the same order, the fp16 chains are still grouped per 128-k chunk (= per
stage), and the per-chunk fp32 combine and the split combine keep their fixed
order. **For a fixed grid the output is bit-identical to round 3**, so §3b
stands as written and GATE1 (bitdiff == 0), GATE2 (cos ≥ 0.99999 + relaunch
byte-identity) and GATE8 (fused == per-slot) are unchanged in meaning.

### New: GATE 9 — the number the tok/s arithmetic should use

GATE3 times **one matrix**: grid = (N/128, SPLIT_K) = 128–192 CTAs, i.e. a
SINGLE wave, so the per-CTA prologue and the split-K epilogue are fully
exposed and amortised over only 32/SPLIT_K stages. **Production never runs
that shape** — the fused launch carries `2·top_k` / `top_k` groups on
`blockIdx.z`, so gate+up is 192·top_k CTAs = `top_k` FULL waves and a
retiring CTA's tail overlaps the next CTA's prologue. Tuning against GATE3
alone optimises a regime the model does not run in. GATE9 (added round 4,
inside `fused_decode_gate`) times the two fused launches at exactly the
production geometry and reports µs, GB/s, CTA count, waves-of-192 and
chunks/CTA for each. Cache: gate+up streams `2·top_k` distinct matrices
(50 MB at top_k = 8) cyclically against a 24 MB L2, so the inter-iteration
hit rate is ~0 without needing a ring.

## 3b. Numerics tier (changed in round 3 — recorded per the microtest law)

- **Dequant is still bit-exact** (gate 1 unchanged): `exl3_dq8` and the
  3INST decode are untouched; `exl3_dequant_dump` must still show
  bitdiff == 0.
- **The GEMV accumulation tier moved** from "fp32 throughout" to:
  x' quantized to fp16 after the fp32 Hadamard (rel. 2⁻¹¹/element,
  incoherent across k), products+accumulate in fp16 within each 128-k
  chunk (≤8 fp16 FMAs deep per half2 slot), fp32 across chunks and splits.
  Expected end-to-end relative output error ~1e-3 ⇒ cosine ≈ 1−1e-6 —
  passes the unchanged **cosine ≥ 0.99999** gate with ~10× margin.
  fp16 range is safe: |w| < 4 and x' is a normalized Hadamard mix, so
  chunk partials sit far below 65504.
- **Determinism is preserved**: the fp16 chains and the per-chunk fp32
  combine run in a fixed order, and the split combine remains fixed-order
  → bit-identical relaunches for a fixed grid (gate 2's relaunch
  byte-identity probe applies as-is). Chunks are 128-k aligned and never
  straddle a split boundary, so the fp16 grouping itself is even
  SPLIT_K-invariant; only the fp32 cross-chunk summation order varies
  with the grid, exactly as in rounds 1–2.

## 4. Roofline arithmetic per expert matrix

Bytes per matrix = `N·K·3/8 + (N+K)·2`:

| shape | trellis bytes | @229 GB/s | @192 GB/s (today's achieved MoE BW) |
|---|---:|---:|---:|
| N=2048 K=4096 (w1/w3) | 3.158 MB | 13.8 µs | 16.4 µs |
| N=4096 K=2048 (w2) | 3.158 MB | 13.8 µs | 16.4 µs |

Per expert triplet ≈ 9.47 MB → 41 µs @229. Routed 6 experts/layer ≈ 56.9 MB
≈ 248 µs @229 (vs 80.2 MB MXFP4) — the −5.2…−7.7 ms/token of plan §3.

## 5. Validation (run on the GPU box, server killed)

```
cargo run -p spark-model --release --example exl3_gemv_microtest \
    --features cuda,gpu-examples
```

Gates (exit code enforced):

1. **Dequant bit-exact**: `exl3_dequant_dump` vs the CPU oracle decode,
   u16-compare over all N·K weights; must be 0 diffs. The CPU oracle's
   window extraction was verified against a verbatim host port of the
   vendored `dq8` (256 K windows, 0 mismatches) and its fp16 rounding is
   bit-identical to native IEEE fp16 hardware over all 65536 windows
   (FNV `be697083fb057234`), so a dump mismatch convicts the GPU kernel.
2. **GEMV cosine ≥ 0.99999** vs the f64 full-pipeline reference, at
   SPLIT_K=1 and the production split, plus a relaunch byte-identity probe
   (determinism of the split combine). Round 3 changed the accumulation
   tier (fp16 x' + per-chunk fp16 accumulate, §3b) — the gate value is
   unchanged, but this run is the tier's acceptance test.
3. **Cold-rotation GB/s** through a ≥512 MB weight ring (defeats the 24 MB
   L2) at both expert shapes; judged against the 229 GB/s ceiling. NOTE
   (§3c): this is a ONE-WAVE launch (128–192 CTAs) and is therefore *not*
   the production regime — read it together with GATE9.

Gates 4–7 cover the P1 prefill leg (§6). Gate 8 covers the fused decode
dispatch:

8. **Fused == per-slot, byte-identical.** 8 synthetic experts, 6 routed slots
   with a deliberately non-slot-ordered index list, pushed through both the
   per-slot chain (4·top_k launches) and the fused pair (3 launches) at the
   same production SPLIT_K. All three output buffers must byte-match; the
   buffers are poisoned with 0x00 before path A and 0xFF before path B so an
   unwritten slot/strip cannot pass. GATE8b re-runs the fused path and
   requires byte-identity with itself (split-K election determinism with 12
   concurrent groups sharing one allocation); GATE8c asserts the launch
   counts (24 → 3).

9. **Production-geometry GB/s** (informational, round 4): times
   `exl3_gemv_m1_fused_gate_up` and `exl3_gemv_m1_fused_down` at exactly the
   grid the serving path launches (`2·top_k` / `top_k` groups on `blockIdx.z`)
   and reports µs, GB/s, CTAs, waves-of-192 and chunks/CTA for each plus the
   per-layer routed-FFN total. This — not GATE3 — is the number the
   tok/s arithmetic should be built on; see §3c.

## 6. Open items toward S6 (serve integration)

LANDED (combined-residency, loader/dispatch legs — GPU-unvalidated):

- `exl3_gemv_m1_idx` — device-indexed twin of `exl3_gemv_m1` (pointer tables
  + on-device `indices[slot]` read; graph-safe, no D2H of the routing).
- Loader: `weight_map/exl3.rs` (`Exl3Weight`/`Exl3ExpertWeight`, shape/dtype/
  `mcg` validation), store I16/I32/F16 passthrough (suh/svh stay native F16 —
  `load_fns::exl3_keep_f16`), `assemble_moe` EXL3 arm (auto-detected from
  `…rank0.trellis`, `ATLAS_EXPERT_EXL3=0` refuses), expert count from config
  (216 on the reference REAP checkpoint). NO transpose pass — tiles load as-is.
- Decode M=1 dispatch: `layers/moe/exl3_decode.rs` — per routed slot
  gate/up (idx-GEMV) → clamped SwiGLU → down (idx-GEMV); NVFP4 shared expert
  via `w4a16_gemv` + unclamped SwiGLU. Routed format tag `Exl3Trellis` fences
  every legacy NVFP4/E8M0 path.
- **Fused decode dispatch** (supersedes the per-slot chain as the default):
  `exl3_gemv_m1_fused_gate_up` + one flat `moe_silu_mul` +
  `exl3_gemv_m1_fused_down`. The (slot, projection) pair rides `blockIdx.z`
  (SPLIT_K keeps `blockIdx.y`), mirroring
  `moe_expert_gate_up_shared_bf16`'s slot-on-y / proj-on-z organization; each
  CTA still resolves its expert from `indices[slot]` on device, so the arm
  stays graph-safe. Routed launches per layer drop 4·top_k → 3 (36 → 3 at
  top_k = 8; 1548 → 129 per token over 43 layers, +172 for the unchanged
  4-launch NVFP4 shared expert). Bit-identical to the per-slot chain at equal
  SPLIT_K — gated by microtest GATE8. `ATLAS_EXL3_FUSED=0` restores the
  per-slot chain for A/B.
  Split-K scratch is now **per launch group**: `ws + group·gridDim.y·N`
  (fp32) and `counters + group·N/128` (i32), sized at load for `2·top_k`
  groups. This is load-bearing — per-slot launches serialized on the stream
  and could share one region; fused groups run concurrently and would corrupt
  each other's partials without it. The self-resetting counter logic is
  unchanged and stays correct per group, so replays still start all-zero.

LANDED (P1 prefill leg — GPU-unvalidated, gates 4-7 of the microtest):

- Prefill / M>1 (plan §3 P1): `forward_prefill` now routes EXL3 through
  `run_routed_grouped_gemm_exl3` (forward_prefill_exl3.rs) — design
  option (a): rotations ride on the ACTIVATIONS, scratch holds the RAW
  decoded weights. Kernels (all in this module):
  * `exl3_h128_pre_rows` — expands the token-major input into the sorted
    layout with `A_rot[r] = H128(diag(suh_e)·A[tok_r])/√128` per row (suh
    is per EXPERT, so a token routed to k experts gets k distinct rows —
    the grouped GEMM then runs with `sorted_token_ids = NULL`). One warp
    per 128-chunk via `exl3_had128`; in-place legal with identity gather
    (used for the down-input rotation over the post-SiLU intermediate).
  * `exl3_h128_post_rows` — in-place `Y[r] = diag(svh_e)·H128(Y[r])/√128`.
  * `exl3_dequant_chunk_bf16` — decodes experts `[e0, e0+count)` into
    slot-major BF16 `[N,K]` scratch (fp16→bf16 RN tail on the bit-exact
    dump path); consumed by `moe_bf16_grouped_gemm` launched per chunk
    SUB-RANGE (`weight_ptrs = static slot table`, `expert_offsets + e0`,
    `num_experts = count` — offsets are absolute rows, so sub-range
    launches read/write the correct global rows).
  Scratch: `ATLAS_EXL3_PREFILL_CHUNK` (default 8) × 16.78 MB = 134 MB —
  one slot size serves all three projections (each is inter×h elements).
  Host reads `expert_offsets` once per layer (prefill-only D2H, same
  pattern as the exact-tiles grid sizing) for exact per-chunk m-tiles +
  empty-chunk skip. NOT graph-capture-legal (prefill never captures).

STILL OPEN:

- ~~m-row (γ-verify) MROW variant — `forward_km` declines for EXL3~~ LANDED,
  GPU-unvalidated — see §8. `exl3_gemv_mrow_fused_{gate_up,down}_m{1,2,4,6,8}`
  + `MoeLayer::dispatch_exl3_verify`; `forward_km` no longer declines.
- Real-checkpoint spot-check: run the dump gate against tp1 tiles.
- Perf tuning after first GPU measurement: stage depth, `SPLIT_K` policy
  (dispatch default fills ~96 CTAs; `ATLAS_EXL3_SPLIT` overrides).
  Half2-accumulate landed in round 3 (§3/§3b).
- ~~Fused gate+up riders to cut the 3·top_k+3 launch count~~ LANDED (§6
  LANDED list): `exl3_gemv_m1_fused_gate_up` / `exl3_gemv_m1_fused_down`,
  4·top_k+4 → 3+4 launches/layer, bit-identical (GATE8). A silu-in-GEMV
  rider is no longer worth it — the SwiGLU is now ONE flat elementwise
  launch per layer over `[top_k, inter]`.

- ~~m-row (γ-verify) MROW variant~~ LANDED — see §8.
- Real-checkpoint spot-check (plan option a): run the dump gate against
  tiles from `/home/flocka/sparkinfer-ref/data/tp1` once readable.
- ~~Perf tuning after first GPU measurement~~ round 2 done (ILP restructure
  + 4 CTAs/SM, +8%); ~~half2-accumulate variant~~ round 3 done (§3/§3b):
  half2 dot + half2 x' + full unroll, 67→55.9 SASS instr/tile — measured
  NO CHANGE (166–168). ~~remaining suspects: barrier cadence, grid-size
  occupancy~~ **resolved by the round-4 SASS cycle budget (§3c)**: issue is
  22.9% of the wall clock and LDS 6%, so ~78% is memory stall; grid/wave
  quantisation is measured-dead (the sweep already ran the exactly-filling
  split); the implicated term is barrier cadence + fetch-window depth, and
  round 4 ships 3 buffers / 1 barrier per stage. NEXT, only if round 4 comes
  back ≤175 GB/s: the DRAM request pattern (`EXL3_NSTRIP` 128 → 256 widens
  the contiguous run 768 → 1,536 B).

  the guarded per-row path); mandatory S6 scope per plan §3/§4.7
  (partial-exactness law: the verify chain flips as a whole).
- Real-checkpoint spot-check (plan option a): run the dump gate against
  tiles from `/home/flocka/sparkinfer-ref/data/tp1` (world-readable now).
- ~~Perf tuning after first GPU measurement~~ round 2 done (§2/§3): ILP
  dequant restructure + 4 CTAs/SM. If gate 3 still lands under ~200 GB/s,
  the next lever is the half2-accumulate variant (`EXL3_GEMM_H_ACC`-style)
  to shed the 8 cvt + fp32 FFMA per tile — quality re-gate required.
- Shared-expert rider (grid.y expert slot) when wiring into
  `moe_shared_expert_fused_t` dispatch.
- ~~Fused gate+up riders to cut the 3·top_k+3 launch count~~ LANDED (§6
  LANDED list): `exl3_gemv_m1_fused_gate_up` / `exl3_gemv_m1_fused_down`,
  4·top_k+4 → 3+4 launches/layer, bit-identical (GATE8). A silu-in-GEMV
  rider is no longer worth it — the SwiGLU is now ONE flat elementwise
  launch per layer over `[top_k, inter]`.
- Prefill P2 (plan §3): grouped trellis GEMM decoding straight to MMA
  fragments, to recover the P1 dequant traffic below.

## 7. P1 prefill cost arithmetic (honest, supersedes the plan's estimate)

Per MoE layer, per prefill chunk, once the chunk is long enough that all
216 experts are routed (N ≳ 1024 at top-6):

| traffic | bytes |
|---|---:|
| trellis read (dequant) | 216 × 9.44 MB = 2.04 GB |
| BF16 scratch write | 216 × 3 × 16.78 MB = 10.87 GB |
| GEMM scratch re-read | 10.87 GB × ceil(rows/expert/64) (1× at N=1024, 2× at N=2410) |
| activations (A_rot ×2 + in-place passes) | ~0.4 GB |

≈ 24–35 GB/layer ≈ 120–175 ms/layer @ ~200 GB/s ≈ **5–7.5 s per 43-layer
pass over a 2410-token prompt** — NOT the ~0.2 s a per-pass reading of the
bytes suggests (the dequant repeats per LAYER, ×43). Under chunked prefill
the cost multiplies again by the number of chunks (plan §6.3's flagged
risk): a 2410-token prompt at `--max-prefill-tokens 1024` = 3 chunks ≈
15–20 s prefill. Acceptable for the bring-up smoke; P2 is the fix.

## 8. m-row (γ-verify) path — `exl3_gemv_mrow_fused_*` (2026-08-12)

GPU-UNVALIDATED. Compiles for `gb10/deepseek-v4-flash` (zero spills), gated by
GATE9 of `exl3_gemv_microtest`, which has NOT been run on hardware yet.

### 8.1 Why this file is the gate on everything above 28 tok/s

MXFP4 plain 21.89 / EXL3+fused plain 23.58 tok/s, measured back-to-back. But
speculation did not work on EXL3 at all: `forward_km` declined for trellis
layers, and its fallback `forward_batched` *hard-errors* on EXL3
("forward_batched (M>1) not wired"). So arming DSpark on EXL3 was not merely
slower — it was unavailable.

The economics (docs/DECODE-WATERFALL-2026-08-10.md §6, docs/SPEC-3X-PLAN.md):
the γ=5 verify step is ~113 ms eager, of which the m=6 expert union is 54.1 ms
— the largest bucket and the only one that scales with verify width. At 3.0 bpw
that bucket goes to ~38.4 ms, dropping the verify:plain step ratio 3.8 → ~2.85.
At the committed 3.46 tok/step that turns speculation from a 0.91x LOSS into a
1.21x win. The EXL3 byte cut helps VERIFY more than it helps plain.

### 8.2 The contract, mirrored from MXFP4 `exp_splitk_m_t`

Studied end to end (`moe_shared_expert_fused_t.cu` `mrow_gather_slots` +
`gate_up_shared_t_m_impl` + `silu_down_shared_t_m_impl`,
`forward_phase.rs::dispatch_splitk_m_t`, `forward_km.rs`). Reproduced exactly,
so the scheduler above the dispatch is unchanged:

| aspect | MXFP4 `_m` | EXL3 `_mrow` |
|---|---|---|
| flat routing | `indices[num_tokens*top_k]`, slot `y` ⇒ token `y/top_k` | same |
| grid.y / z | `y` on grid.y, `proj*SPLIT+ks` on grid.z | SPLIT on grid.y (EXL3 needs it there), `2*y+proj` on grid.z |
| dedup | first slot holding an id is LEADER, computes every slot routed to it; later duplicates exit before touching memory | same, `exl3_mrow_gather` |
| gather bound | `M = min(count, MROW)`, surplus rows alias row 0 | same |
| ladder | `_m{1,2,6,8}` + count-bucketed arms; host picks `MROW >= num_tokens` | `_m{1,2,4,6,8}`; host picks the smallest rung `>= num_tokens` |
| shared expert | computed in-kernel on the `y == total_routed` block-set | NOT in-kernel — EXL3 shared weights are NVFP4, so it is the same per-row `w4a16_gemv` chain plain decode runs, `num_tokens` times |
| output layout | routed slots flat in `expert_{gate,up,down}_out`, shared rows in the shared scratch | identical (blend untouched) |
| split-K partials | `partial[2][SPLIT][rows][N]`, separate `*_finalize_m` launch | `ws` keyed by OUTPUT ROW (`2*slot+proj` / `slot`), `[S][N]` each; combine is in-kernel via the existing last-split election, so there is NO finalize launch |
| graph safety | expert ids read on device, geometry independent of routing | same |

The one contract that is *not* a mirror is the partial layout, and it is forced:
the m=1 EXL3 kernels already do the split-K combine in-kernel (last split to
arrive is elected by a self-resetting atomic), so a separate finalize kernel
would change the accumulation structure and break bit-identity. Keying `ws` by
the flat routed slot instead of by launch group is what makes that safe under
dedup — slots are unique across leaders, so concurrent leaders never collide,
while `counters` stays keyed by launch group exactly as at m=1.

### 8.3 The exact-GEMV law, and how per-row bit-identity is guaranteed

The binding constraint, learned expensively (memory
`oproj-grouped-kernels-ab-2026-08-09`): a PARTIALLY exact verify chain is WORSE
than either extreme — o-proj-only exactness measured 2.54 tok/step against 2.83
for none and 2.92–3.01 for full. So each verify row's expert output must be
bit-identical to the m=1 fused path's output for that same token.

This is guaranteed structurally, not statistically:

1. **x'** is produced by the SAME `exl3_input_pass` device function on the same
   activation row and the same per-expert `suh`. The Hadamard is per aligned
   128-chunk, so a chunk's x' bits depend only on that chunk — the smaller m-row
   superblock (`EXL3_M_XCHUNKS = 8` vs `EXL3_MAX_XCHUNKS = 16`, a pure
   smem/refill trade) cannot move a bit.
2. **K-slice**: same `chunks_total*split/S` formula at the same `S`. The host
   passes the m=1 `split_for(N)` — `dispatch_exl3_verify` deliberately does not
   re-tune it, because re-slicing K is exactly how bit-identity would be lost.
3. **Per-output op order**: same `r` order over the 8 tile-rows, same
   `exl3_dq8` decode of the same trellis bytes, same four HFMA2 chains with the
   same even/odd tile-row split, same fixed-order `__hadd2` +
   `__half22float2` + four fp32 adds per 128-k chunk, same quad shuffle-reduce,
   same fixed split-order combine (`p = 0..S-1`), same output
   `exl3_had128` + `svh` + `*RSQRT128` + `__float2bfloat16`.
4. **Rows never interact arithmetically.** Each row owns a private accumulator
   chain; the ladder's surplus rows alias x' slice 0 and are dropped at emit.
   Dedup changes only WHICH CTA evaluates a row — the same argument that makes
   the m=1 → fused collapse bit-identical (GATE8).
5. FP results are invariant under instruction scheduling, so the different
   register allocation (110 regs / 2 CTAs per SM at MROW=6, vs 56–64 / 4 at
   m=1) cannot move them either.

The chain AROUND the GEMV is held exact the same way: the SwiGLU is the same
elementwise `moe_silu_mul` over a wider flat extent, and the shared expert runs
the SAME single-row `w4a16_gemv` chain once per row rather than a batched
`w4a16_gemm` (different accumulation order ⇒ partial exactness ⇒ the law). That
costs `4*num_tokens` small NVFP4 launches per layer — ~24 at γ=6, ~1000 per
step across 43 layers, ~3 ms against a ~110 ms verify step. Batching it needs a
bit-exact `w4a16_gemv_batchm`; that is the next lever here, not a shortcut.

### 8.4 Launch and occupancy budget

Per MoE layer at `num_tokens = 6`, `top_k = 8`:

| stage | per-row fallback (today) | m-row |
|---|---:|---:|
| routed gate+up | 6 | 1 |
| routed SwiGLU | 6 | 1 |
| routed down | 6 | 1 |
| shared (NVFP4) | 24 | 24 |
| **total** | **42** | **27** |

and, far more importantly, the routed trellis stream drops from
`num_tokens * top_k` expert reads to `|union|` — the measured DSpark union is
far below `6*top_k` (hash-routed layers pick the identical top-6 for every row;
learned-gate layers measured 1.28x overlap at K=2).

ptxas, sm_121, zero spills / zero stack on every arm:

| arm | regs | smem | CTAs/SM |
|---|---:|---:|---:|
| `_m1` | 71–78 | 14 988 | 2 |
| `_m2` | 86–89 | 17 680 | 2 |
| `_m4` | 114–120 | 23 064 | 2 |
| `_m6` | 109–110 | 28 448 | 2 |
| `_m8` | 108–110 | 33 832 | 2 |

`__launch_bounds__(256, 2)`, the MXFP4 `MOE_M_LB` lesson applied: the m-row arm
carries MROW accumulator chains and is load-latency bound, so it wants
registers to keep loads in flight, and smem already caps it near 3 CTAs/SM. The
m=1 entries are UNTOUCHED (56/63/64 regs, 16 900 B smem — unchanged).

Split-K scratch grew: `exl3_ws_floats` now sizes for the widest of the four
claims (m=1 gate+up / down, m-row gate+up / down) using the ACTUAL splits
rather than `EXL3_MAX_SPLIT`. At the V4 shapes the binding term is the m-row
gate+up, 2·64·6·2048 f32 = 6.3 MB/layer (was 3.1 MB).

### 8.5 What is NOT done

- **Never run on a GPU.** GATE9 is the acceptance test; run it first.
- The shared expert is `4*num_tokens` launches (see §8.3). A bit-exact
  `w4a16_gemv_batchm` would take it to 4.
- `EXL3_MROW_ARMS` tops out at 8 (= `MOE_DECODE_MAX_ROWS`). Past that
  `verify_ffn_is_batched` declines, `forward_km` returns false, and
  `forward_batched` hard-errors on EXL3 — the loud pre-existing failure, kept
  deliberately over silently-wrong output. Widen the ladder before widening the
  drafter.
- No `GROUP_UNIQUE` / count-bucketed partition arms (the MXFP4 family's
  `_m1u` / `_m6c56` tier). Whether the M==1-heavy leader distribution wants its
  own arm here is a measurement, not a guess.
- `forward_k2` (the n==2 fused K=2 path) still has no EXL3 arm;
  `k2_verify_ffn_is_batched` therefore stays `_t`-only and EXL3 n==2 batches
  through `forward_km` instead.
