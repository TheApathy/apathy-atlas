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
- ~~Prefill P2 first rung~~ LANDED, GPU-unvalidated: the env-gated
  `exl3_grouped_prefill` decodes one trellis tile per warp directly into BF16
  register fragments and feeds the M64 tensor-core MMA body. It removes both
  global BF16 weight materialization and shared decoded-weight staging (§9).

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
elementwise `moe_silu_mul` over a wider flat extent. With
`ATLAS_VERIFY_GEMV_V2=1`, the shared expert uses a grouped-batch kernel whose
per-row FMA and reduction order is copied from single-row `w4a16_gemv`; it
streams each NVFP4 weight once and reduces `4*num_tokens` launches to three
GEMVs plus one flat SwiGLU. Compile-time V2 entries cover M=4/5/6/8/16; other
widths through eight use the exact runtime-M incumbent. Missing symbols or
ineligible dimensions retain the per-row exact fallback.

The no-GPU SASS gate reports the M6 V2 entry at 904 instructions, 80 registers,
three branches, and zero spills versus 1,232 instructions, 76 registers, and 31
branches for the runtime-M incumbent. Its FFMA/FADD/shuffle counts scale exactly
six-to-eight with the number of compiled rows. This is arithmetic-order and
compiler-resource evidence, not a device-time or tok/s result.

### 8.4 Launch and occupancy budget

Per MoE layer at `num_tokens = 6`, `top_k = 8`:

| stage | per-row fallback (today) | m-row |
|---|---:|---:|
| routed gate+up | 6 | 1 |
| routed SwiGLU | 6 | 1 |
| routed down | 6 | 1 |
| shared (NVFP4), per-row fallback | 24 | 24 |
| shared (NVFP4), exact batch enabled | 24 | 4 |
| **total, exact batch enabled** | **42** | **7** |

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

### 8.5 Fixed-K2 decode and SM121-loadable M16

The serving checkpoint's K2 routed projections now resolve decode and m-row
handles from `exl3_gemv_k2`. That module includes the same source with
`EXL3_FIXED_BITS=2`: every entry rejects a mismatched runtime bitrate, then
passes the literal K2 width into the inlined lane geometry, trellis staging,
and decoder. K3 remains on the generic module;
`ATLAS_EXL3_FIXED_K2=0` is the exact generic fallback.

Side-by-side `sm_121` cubins, with zero stack/local/spills throughout:

| arm | complete SASS instructions, generic → K2 | registers, generic → K2 |
|---|---:|---:|
| fused gate+up M1 | 1,800 → 1,520 (-15.6%) | 64 → 63 |
| fused down M1 | 1,784 → 1,520 (-14.8%) | 63 → 63 |
| m-row gate+up M2 / M6 | 2,840 → 2,576 / 6,224 → 5,976 | 88 → 88 / 107 → 104 |
| m-row down M2 / M6 | 2,744 → 2,488 / 5,992 → 5,704 | 86 → 80 / 107 → 104 |
| m-row gate+up M16 | 14,664 → 14,416 | 128 → 128 |
| m-row down M16 | 13,984 → 13,736 | 128 → 128 |

This cubin gate also caught a pre-existing M16 failure hidden by the normal PTX
build: its eight-chunk x' superblock requested `0xd848` bytes of static shared
memory, above ptxas's `0xc000` SM121 limit. M16 alone now uses a four-chunk
storage window. The identical 128-k chunks are computed and accumulated in the
same order, with one refill after chunk four; only scratch residency changes.
Both M16 entries now assemble at 40,008 B shared memory, 128 registers, and
zero stack/local/spills. The smaller rungs retain their eight-chunk window.

These are compiler and loadability facts, not device timings. Before default
performance claims, byte-compare generic K2 and fixed K2 for M1 and every
selected m-row rung, including an M16 case whose split owns more than four
128-k chunks.

### 8.6 What is NOT done

- **Never run on a GPU.** GATE9 is the acceptance test; run it first.
- The current m-row grid still launches against every routed slot; duplicate
  leaders exit only after CTA creation, route staging, and leader election. A
  compact device worklist is the next structural verify lever.
- `EXL3_MROW_ARMS` tops out at 16 (the DFlash2 verify width). Past that
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

## 9. P2 grouped prefill first rung — direct trellis to tensor cores

`kernels/gb10/common/exl3_grouped_prefill.cu` is an opt-in replacement for
P1's global BF16 dequant scratch. Set `ATLAS_EXL3_PREFILL_DIRECT=1` to select
it; unset remains the P1 baseline until the GPU gates below pass.

Direct mode is frozen at model load, omits P1's roughly 134 MB decoded-weight
scratch allocation, and uses a full compact grid by default: one CTA per exact
`(expert, N64 strip)`. Each CTA reads that expert's device offsets once, then
walks only its live M tiles. The kernel retains grid-stride support for the
undersubscribed parity case, but production exposes every strip to the hardware
scheduler. This removes the per-layer
offsets D2H and stream synchronization without either a heuristic expert-load
cap or the old rectangular mostly-empty M-tile universe. Set
`ATLAS_EXL3_PREFILL_PERSISTENT=0` only to retain the exact 3-D grid as a parity
fallback.

`ATLAS_EXL3_PREFILL_FUSED_POST=1` additionally replaces the separate gate
H128-post, up H128-post, SwiGLU, and down H128-pre kernels with
`exl3_h128_post_silu_pre_rows`. The fused kernel explicitly rounds both
rotated operands to BF16 before clamping/activation, then rounds the SwiGLU
result to BF16 before applying the down rotation. Those are the legacy
store/load barriers. At 2,410 tokens and the current top-6 config, it reduces
these four passes from nine BF16 buffer transactions to three: about 533 MB to
178 MB per MoE layer, or 15.3 GB across 43 layers. That is traffic arithmetic,
not an end-to-end timing claim.

`ATLAS_EXL3_PREFILL_FUSED_UNPERMUTE=1` replaces the final down H128-post and
indexed weighted unpermute with `exl3_h128_post_unpermute_rows`. One warp owns
one token/128-column chunk, visits slots in the legacy top-k order, explicitly
rounds each rotated expert value to BF16, then weights and accumulates it. At
the same geometry this avoids materializing the rotated routed output: roughly
355 MB becomes 118 MB per layer, saving another 10.2 GB across 43 layers.
Together the two tail fusions remove about 25.5 GB of activation traffic per
prefill pass. These values exclude unchanged final-output writes.

For the exact K2 `H=4096` shape, fused unpermute automatically selects
`exl3_h128_post_unpermute_rows_h4096`; set
`ATLAS_EXL3_HROW_FIXED_SHAPE=0` for the generic fallback. The entry point
shares one force-inlined arithmetic body with the generic kernel, keeps top-k
as the same runtime-ordered loop, and rejects any nonexact H, grid, or block
shape. On SM121 its numeric body is 168 instructions rather than the generic
176 and uses 36 rather than 39 registers. The fail-closed block-shape entry
guard brings the complete fixed function to 184 static instructions without
changing that 36-register footprint, BF16/Hadamard/load/store/accumulation
counts, or zero shared/stack/local/spill memory. This is compiler evidence; it
does not replace byte parity or CUDA-event timing.

`ATLAS_EXL3_PREFILL_M128=1` selects a separately compiled 256-thread direct
kernel. Warp pairs share one cooperatively staged trellis strip: warps 0–3
produce rows 0–63 and warps 4–7 produce rows 64–127.
On a tail of at most 64 rows, the upper four warps skip decode, MMA, and stores
while still reaching every CTA barrier. Consequently M128 executes the same
`ceil(rows/64)*64` padded MMA rows as M64. Its remaining tradeoff is a
256-thread block and different occupancy/scheduling. The current serving
checkpoint is 256 experts/top-6: a balanced 2,410-token prompt gives 56–57
rows/expert, so M128 does not even reduce the tile count versus M64 and remains
an opt-in skew experiment. Only a real histogram and same-boot timing can
select this rung.

`ATLAS_EXL3_PREFILL_K64=1` selects a separate M64 kernel that stages four
consecutive K16 activation/trellis slices before synchronization. It executes
the four decode/MMA slices in the original K order, so FP32 accumulation order
is unchanged, but pays two stage barriers per K64 instead of eight. Across one
gate/up/down routed tile at the current K=4,096/4,096/2,048 shapes, that is 320
dynamic stage barriers instead of 1,280. K64 is mutually exclusive with M128;
both remain opt-in until device parity and timing pass.

The K64 modules additionally use SM121 asynchronous global-to-shared copies.
Activation rows keep the load-bearing `+2` BF16 bank-dispersion pad, so their
shared addresses are only four-byte aligned: four `cp.async.ca` words replace
the register-mediated 16-byte load/store. Naturally aligned trellis vectors use
one 16-byte `cp.async.cg`. One commit/wait covers both streams before the same
CTA barrier and the MMA sequence is untouched. On the fixed-K2 cubin this
replaces the static 6 LDG + 12 STS mix with 4 LDG + 4 STS + 8 LDGSTS and one
dependency barrier; function text shrinks from 784 to 736 16-byte SASS slots
while staying at 64 registers, 10,496 B ELF shared memory, 32 HMMA, and zero
stack/local/spills. The generic K64 cubin also assembles with 63 registers.

The same experiment reduced K16 text from 512 to 488 slots but raised its
register count from 56 to 64, so it was rejected there. The shipping K16 module
remains exactly 512 slots / 56 registers. This selection is compiler evidence,
not a device-timing claim; the P2 byte-parity and same-boot K16/K64 CUDA-event
gates below remain mandatory.

A two-buffer K64 `cp.async` pipeline was also kept out of production. Its
sentinel-stage form can preserve K/HMMA order and safely barrier asynchronous
copies plus synchronous tail zero-fill, but static text grew from 640 to 664
instructions and ptxas shared memory doubled from 9,472 to 18,944 B. On a
100-KiB SM that changes the shared-memory ceiling from nine to five resident
CTAs. Only GPU parity, sync/race checking, and occupancy-timed A/B evidence
could justify that latency-hiding trade, so the single-buffer arm remains.

All direct variants stage logical BF16x8 activation vectors. K16/M128 use one
aligned `uint4` global load followed by four 32-bit stores into the padded
shared row; K64 uses the four asynchronous words described above. Current K
dimensions, K-stage bases, and vector source columns are 16-byte aligned. Both
forms are bitwise copies and leave arithmetic unchanged.

The serving checkpoint's K2 projections automatically use fixed-bit decoder
cubins inside direct mode. `ATLAS_EXL3_PREFILL_FIXED_K2=0` restores the generic
K2/K3 decoder for A/B and fallback; K3 projections always remain generic. The
fixed kernels keep the same trellis word order and reject a mismatched runtime
bit width. On SM121, fixed K2 reduces the complete K16 cubin from 592 to 512
SASS instructions and 64 to 56 registers; K64 drops from 880 to 784
instructions and reduces declared shared memory from 9,984 B to 9,472 B. All
four cubins have zero stack, spills, and local memory. These are compiler
effects, not device timings.

For the checkpoint-proven K2 projection shapes, persistent mode also selects
fixed `N/K` entry points: gate/up use `2048x4096`, and down uses `4096x2048`.
The kernels reject mismatched runtime dimensions, bitrate, or persistence;
`ATLAS_EXL3_PREFILL_FIXED_SHAPE=0` restores the shape-generic fixed-K2 arm.
On SM121, fixing the shape reduces K16 from 512 to 424 SASS instructions
(-17.2%) at the same 56 registers and 2,560 B ptxas shared memory. K64 falls
from 736 to 656 instructions (-10.9%) and from 64 to 56 registers at the same
9,472 B shared memory. Both gate/up and down compile identically, retain the
same 8/32 HMMA counts and copy mix, and use zero stack/local/spill memory.
This is static compiler evidence; GPU byte parity and timing remain required.

Those four fixed-shape entries also compile the production direct-mode row
mapping as identity. The host already passes a null `sorted_token_ids`; the
specialized kernels reject any non-null pointer, while every generic entry
retains the optional gather. K16 consequently shrinks again from 424 to 384
instructions (-9.4%) at the same 56-register/2,560-B resource footprint. K64
stays at 656 instructions and 56 registers but removes the sorted-index global
load. Executed activation copies, HMMAs, barriers, shared memory, and arithmetic
order are unchanged. This remains compiler evidence pending the GPU gates.

Finally, the fixed entries require the exact production one-CTA-per-strip
1-D grid and compile out the generic undersubscribed grid-stride loop. Runtime
guards check both auxiliary grid dimensions and the full
`num_experts*(N/64)` extent; host sizing uses checked multiplication. N-tile
counts are compile-time power-of-two constants, so the strip maps to its
expert and N64 tile with a shift and mask. K16 shrinks from 384 to 376
instructions and 56 to 55 registers; K64 shrinks from 656 to 640 instructions
at the same 56 registers. Both retain their shared-memory footprints, copy and
HMMA counts, barriers, and zero stack/local/spill use. The generic fixed-K2
modules preserve grid-stride support for undersubscribed parity and opt-out.

A full compiler-visible fixed-128-thread experiment (C++ runtime block guard,
constant load stride, and fixed M-warp mapping) was rejected. It reduced K16
from 376 to 344 instructions and K64 from 640 to 624, but raised registers to
64 from 55/56. Bisection retained `__launch_bounds__(128)` and the existing
runtime mapping. A uniform inline-PTX entry guard now rejects any non-128x1x1
block without exposing the fixed size to ptxas: registers remain 55/56 and the
steady loops are unchanged. The guard adds 16 static slots to K16 and eight to
the final K64 body. Generic and M128 kernels carry neither the fixed launch
bound nor this exact-block ABI guard.

K64 fixed-shape entries additionally pack each adjacent BF16 output pair with
two independent round-to-nearest conversions and one aligned 32-bit store.
Every pair starts on an even BF16 column and stays within its N2048/N4096 row;
the two M-tail row guards remain independent. Combined with launch bounds this
reduces K64 from 640 to 600 instructions, replaces 32 scalar 16-bit stores with
16 pair stores, and cuts BF16 conversion instructions from 48 to 32 at the
same 56 registers, shared memory, HMMAs, copies, barriers, and zero spills.
K16 keeps scalar stores because its packed candidate raised registers 55→56.

The K64 fixed-K2 decoder also simplifies its three shifted 18-bit windows.
After the first funnel shift, K2 consumes only `lo` bits 4..21, 8..25, and
12..29, so `lo >> 4/8/12` is exactly equivalent to funneling in a high word.
An all-lane CPU oracle covers deterministic edge and seeded random words. The
change reduces the unguarded K64 body from 600 to 592 instructions and its hot
loop from 413 to 409 (256 fewer dynamic instructions per M tile at K=4,096).
With the fail-closed entry guard, the complete function is 600 instructions at
the same 56 registers and unchanged HMMAs, async copies, dependencies,
barriers, stores, shared memory, and zero spills. K16 retains generic windows
because its candidate raised registers 55→56; its guarded complete function
is 376 instructions / 55 registers.

`ATLAS_EXL3_PREFILL_N128=1` adds an exact K64/K2 M64xN128 rung for the
DeepSeek gate/up/down shapes. Eight column warps share the same staged M64
activation tile instead of four, so the N2048 projections use 16 rather than
32 strips per expert and N4096 uses 32 rather than 64. Across three routed
projections this halves the exact outer grid from 32,768 to 16,384 CTAs per
layer while preserving the number and order of HMMAs and trellis-tile reads.
At 2,410 tokens and top-6, it removes 5.69 GB/layer of logical BF16 activation
loads (244.5 GB across 43 layers); these are instruction-level bytes and may
hit cache, not a claim of equivalent DRAM traffic or elapsed-time savings.

Both N128 shapes assemble for SM121a at 600 static instructions, 62 registers,
10,496 B ptxas shared memory, one barrier, and zero stack/local/spills. The N64
comparison is 600 instructions, 56 registers, and 9,472 B ptxas shared memory.
A CPU owner model proves every M64xN128 output element is written exactly once,
and the host enables this rung only on the exact persistent fixed-shape K64/K2 path.
The increased block width and register footprint make same-boot N64/N128 byte
parity and CUDA-event timing mandatory before promotion.

```bash
ATLAS_TARGET_MODEL=deepseek-v4-flash cargo run --release -p spark-model \
  --example exl3_prefill_n128_microtest --features cuda,gpu-examples
```

The harness uses both production matrix shapes and 1/63/64/65/129 rows per
expert, compares generic N64, exact N64, exact N128, and exact N256 output
bytes, poisons every destination independently, verifies that wrong N128/N256
block sizes leave the poison untouched, and then prints same-boot CUDA-event
timings.

`ATLAS_EXL3_PREFILL_N256=1` is the next exact K64/K2 rung. Sixteen column
warps in one 512-thread CTA share the same staged M64 activation tile while
retaining each warp's existing 16-column accumulator and K order. Relative to
N128, this halves the exact outer grid from 16,384 to 8,192 CTAs per layer and
removes 2,842,951,680 logical activation bytes/layer, or 122,246,922,240 bytes
(113.85 GiB) across 43 layers at 2,410 tokens/top-6. Trellis reads, HMMAs,
output stores, and accumulation order are unchanged.

Both N256 wrappers assemble for SM121a at 62 registers, 12,544 B ptxas shared
memory, 32 static HMMAs, one barrier, and zero stack/local/spills/atomics. The
raw register arithmetic (`2*512*62`) is below 64 K, but allocation granularity
and the one-argument launch bound do not prove two resident CTAs. A GB10
occupancy query and device timing remain promotion gates. The CPU owner model
covers every M64xN256 output element once and the host keeps N256 mutually
exclusive with the N128 experiment.

The traffic-only upper bound is still insufficient for the 2,000 tok/s goal:
charging every saved byte to 273 GB/s removes at most about 0.448 s from the
historical 2.66-second TTFT, or roughly 1,089 prefill tok/s. Cache reuse makes
the actual gain smaller. N256 is therefore a meaningful measurement rung, not
a claim that the target is reached.

```bash
bash scripts/check-exl3-prefill-n256-sass.sh
```

The next numeric candidate is W2A8, not a wider exact tile. The isolated
`kernels/gb10/experiments/exl3_w2a8_grouped_prefill.cu` component keeps EXL3
trellis weights compressed, pairs two decoded K16 fragments into one native
E4M3 K32 MMA, and retains FP32 accumulators. It consumes sorted post-H128 A8
from the existing per-token/per-K128 quantization contract. For every K128
group, `a_scale[row,group]=max(maxabs/448,1e-12)`; after decoded weights are
scaled by 16, that group's FP32 inner accumulator is folded into the outer
accumulator with `a_scale/16`, before the existing BF16 output boundary. The
component now has a default-off, model-specific serving integration behind
`ATLAS_EXL3_PREFILL_W2A8=1`. It is eligible only for the exact DeepSeek K2
gate/up `(2048,4096)` and down `(4096,2048)` shapes with top-6 routing, TP1,
no communicator or EP, direct persistent fixed-shape prefill, the fixed-K2
dual-pre/fused-post chain, no M128/N128/N256 selector, all four CUDA handles,
no graph capture, and sufficient checked arena
capacity. Any W2A8 mismatch retains the incumbent path before the first W2A8
write; EXL3 graph capture keeps its pre-existing fail-closed rejection. Once
the five-launch same-stream chain starts, errors propagate instead of entering
the incumbent alias schedule. Its raw down BF16 result receives the existing
H128/SVH post unless fused post-unpermute owns that boundary. The compact
A8/scales reuse the expert output
arenas; the token-major `fp8_act` buffer is not used. The CPU prerequisite
exhaustively enumerates all 65,536 codebook windows: 10,746 finite binary16
values in
`[-3.94921875, 3.94921875]`; an exact power-of-two weight scale of 16 avoids
E4M3 saturation and loss of nonzero codebook values. It also proves the
M64xK32 A, K32xN16 B, and M64xN64 accumulator fragment bijections. In
particular, the incumbent BF16-K16 lane fragments are not already a native FP8
B fragment: an intra-quad shuffle/repack must produce four consecutive K bytes
per `b0`/`b1` register. The component implements that mapping with two direct
index shuffles per packed pair; the CPU model pins the byte order independently.
A compact M3xK256xN8 projection oracle crosses two K128 scale groups, includes
an all-signed-zero floor group, and bounds the mathematical W2A8 result against
the BF16 incumbent at cosine above 0.99 and normalized RMSE below 0.15. That is
an offline operator-model check, not native-MMA parity.

```bash
ATLAS_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 cargo test -p spark-model \
  --test exl3_w2a8_numeric_model --test exl3_w2a8_component_model \
  --test exl3_w2a8_dispatch_model --test exl3_w2a8_emitter_probe_model
bash scripts/check-exl3-prefill-w2a8-sass.sh
bash scripts/check-exl3-prefill-w2a8-probe-build.sh
bash scripts/check-exl3-prefill-w2a8-emitter-probe-build.sh
```

Both exact GU `(N=2048,K=4096)` and down `(N=4096,K=2048)` instantiations
compile for SM121a at 103 registers, 7,168 B cuobjdump shared memory, one
ptxas barrier slot, and zero stack/local/spills. The retained K64-stage body
contains 16 static E4M3 K32 QMMA sites, 16 `SHFL.IDX` sites, and 16 native
saturating E4M3 conversion sites; it executes twice per K128 scale group.
These are compile/resource facts, not numeric or timing evidence. Raw register
arithmetic caps residency at four 128-thread CTAs on a 64K-register SM, and
allocation granularity can lower that, so a GB10 occupancy query remains a gate.

The adjacent producer source
`kernels/gb10/experiments/exl3_w2a8_h128_emit.cu` removes the producer-side
BF16 sidecar without skipping its numeric boundary. Its dual H4096 entry
rounds and re-expands each gate/up H128 result as BF16, then emits row-major
E4M3 plus one FP32 scale per K128. Its H2048 entry preserves the gate/up
post-H128 and SwiGLU BF16 boundaries before down H128 and quantization. The
reduction remaps each warp's four values per lane into the standalone
quantizer's four contiguous 32-value trees, including the same ordered `fmaxf`
fold. SM121a compilation reports 36 registers for the dual entry and 40 for
the down entry, 5,120 B cuobjdump shared memory (4,096 B ptxas static shared)
for both, and zero stack, local memory, spills, atomics, or BF16 global stores.
The component tests pin checked sidecar layouts at expanded-row counts 6,144,
6,150, and 14,460, plus arithmetic overflow rejection. These remain static
component facts: model-specific wrappers make the emitters available only to
the explicit serving opt-in, and native numeric, occupancy, quality, and
end-to-end timing gates are still required before promotion.

The standalone emitter admission harness compares the dual-H4096 candidate to
the fixed dual BF16 transform plus the standard quantizer, and the down-H2048
candidate to the incumbent fused post/SwiGLU/pre transform plus that quantizer.
FP8 data and FP32 scales must match byte-for-byte for nonidentity and null
identity routing across row boundaries 1/31/32/33/63/64/65/129. It also checks
two output poisons, inactive tails, guard regions, and each malformed grid,
block, row, and fixed-width launch independently. Build and retain the
provenance-bound runner with:

```bash
W2A8_EMITTER_PROBE_OUTPUT_DIR=/tmp/atlas-w2a8-emitter-probe \
  bash scripts/check-exl3-prefill-w2a8-emitter-probe-build.sh
/tmp/atlas-w2a8-emitter-probe/run-emitter-probe.sh
```

The receipt binds the harness, emitter, incumbent EXL3 source, quantizer,
build script, tools, repository state, host binary, and extracted cubins. An
offline compile does not satisfy native parity; the strict generated runner
must finish on GB10 with zero mismatches and clean guards.

The isolated
`kernels/gb10/experiments/exl3_w2a8_grouped_prefill_n128.cu` component is the
next W2A8 strip-width experiment. It doubles the N64 component to an exact
M64xN128, 256-thread/eight-warp CTA while retaining the K128 scale groups,
two K64 stages, native E4M3 K32 MMA, FP32 accumulation, and final BF16
boundary. It is compile-only: no production registry, environment selector,
serving dispatch, or fallback can reach it. At 2,410 tokens, top-6 routing,
and 43 layers, the CPU work model halves the logical activation rereads from
the N64 component, removing 122,246,922,240 bytes (113.85 GiB) and 704,512
CTAs across gate, up, and down. Those are structural counts, not a cache,
latency, or throughput prediction.

```bash
ATLAS_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 cargo test -p spark-model \
  --test exl3_w2a8_n128_model
bash scripts/check-exl3-prefill-w2a8-n128-sass.sh
```

Both production-shape instantiations assemble for SM121a at 1,016 static
instructions, 96 registers, 8,192 B resource shared memory, one barrier, and
zero stack/local/spills/atomics. The trellis-stage regression proves the 128
cooperative `uint4` loads are a bijection over the four K16 rows by 32 N128
words and that all 512 scalar words consumed by the eight warps are initialized
exactly once. GB10 numeric parity, occupancy, component timing, and end-to-end
timing remain required before any serving integration.

The isolated
`kernels/gb10/experiments/exl3_w2a8_fused_gu_down_emit_n128.cu` component
tests a different N128 lever: collapse the two gate/up consumers and the
post-H128/SwiGLU/down-pre A8 producer into one exact-shape CTA. This is an
Atlas-specific implementation of the conceptual mega-fusion precedent in
Entrpi/ds4 commit `da027a1`; Atlas retains EXL3 trellis decode, explicit BF16
round/re-expand boundaries, and its standalone per-K128 A8 reduction order.
Each 256-thread CTA owns one expert/N128 tile and loops M64 rows. Gate and up
are computed sequentially into two 64x128 BF16 shared tiles; the dead GEMM
staging storage is then reused by the down-A8 reduction. The component emits
the down GEMM's A8 input and FP32 scales, not the final down projection.

```bash
ATLAS_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 cargo test -p spark-model \
  --test exl3_w2a8_fused_gu_down_emit_n128_model
bash scripts/check-exl3-prefill-w2a8-fused-gu-n128-sass.sh
bash scripts/check-exl3-prefill-w2a8-fused-gu-n128-probe-build.sh
```

The exact SM121a entry assembles at 3,344 static instructions and 128
registers/thread (32,768 nominal registers/block). ptxas reports 39,936 B
static shared memory; `cuobjdump --dump-resource-usage` reports 40,960 B,
including its additional 1 KiB accounting. Stack, local memory, spills, and
atomics are all zero. The source/SASS gate also pins 32 E4M3 QMMA, 52 BF16
conversion, 36 saturating E4M3 conversion, 4 `MUFU.EX2`, 166 FFMA, 193
shuffle, and 7 `BAR.SYNC` sites. At 2,410 tokens/top-6 it structurally replaces
three launches with one and removes 236,912,640 logical BF16 gate/up
write-plus-read bytes per layer: 86 launches and 10,187,243,520 bytes
(9.488 GiB) across 43 layers. These are compile-only resource and structure
facts, not measured traffic or time. There is no serving symbol registration
or dispatch, native parity, occupancy result, timing result, or throughput
claim; all remain promotion gates.

Build an immutable GB10 runner and supply its only admission threshold
explicitly:

```bash
W2A8_FUSED_GU_PROBE_OUTPUT_DIR=/tmp/atlas-w2a8-fused-gu-probe \
  bash scripts/check-exl3-prefill-w2a8-fused-gu-n128-probe-build.sh
/tmp/atlas-w2a8-fused-gu-probe/run-fused-gu-n128-probe.sh 1.01
```

The probe byte-compares all 14,460 expanded rows and their K128 scales across
256 experts, in addition to eight row-boundary cases, one empty-expert case,
dual output poisons, and 15 malformed geometries. Timing is same-process ABBA
over a balanced synthetic 256-expert histogram; it is not the checkpoint's
real routing histogram or an end-to-end result. `1.01` is an explicit
admission threshold, not a measured speedup or throughput claim.

The isolated
`kernels/gb10/experiments/exl3_w2a8_grouped_prefill_n256.cu` component extends
the same mapping to M64xN256 with 512 threads and sixteen N16-owning warps. The
first 256 threads bijectively stage the 256 activation `uint4` vectors and the
four-by-64 trellis `uint4` vectors; each warp consumes one disjoint 16-word
trellis fragment and writes one N16 strip. Relative to N128, this halves the
outer grid from 16,384 to 8,192 CTAs/layer and halves logical A8 activation
rereads. At 2,410 tokens/top-6 that structural difference is 1,421,475,840
bytes/layer, 61,123,461,120 bytes (56.92 GiB) and 352,256 CTAs across 43 layers.
These are logical instruction-level counts, not DRAM, latency, or throughput
predictions.

```bash
bash scripts/check-exl3-prefill-w2a8-n256-sass.sh
```

Both exact GU/down N256 instantiations compile for SM121a at 1,024 static
instructions, 100 registers/thread, 51,200 nominal and 53,248
allocation-granularity-rounded registers/block, 10,240 B cuobjdump shared
memory (9,216 B ptxas static shared), one barrier slot, and zero
stack/local/spills. Each retains 16 static E4M3 QMMA, 16 `SHFL.IDX`, and 16
saturating E4M3 conversion sites. This establishes static resource feasibility
only: N256 has no serving wrapper or registry entry, and no GB10 numeric,
occupancy, timing, or throughput result.

The standalone promotion probe independently compiles both production shapes
at N64, N128, and N256 and emits six provenance-bound binaries. On GB10, persist them
and pass
explicit, review-approved numeric thresholds (the harness intentionally has no
defaults):

```bash
W2A8_PROBE_OUTPUT_DIR=/tmp/atlas-w2a8-probes \
  bash scripts/check-exl3-prefill-w2a8-probe-build.sh
/tmp/atlas-w2a8-probes/run-gu-pair-probe.sh \
  <min_cosine> <max_abs_error> <min_end_to_end_speedup>
/tmp/atlas-w2a8-probes/run-down-pair-probe.sh \
  <min_cosine> <max_abs_error> <min_end_to_end_speedup>
/tmp/atlas-w2a8-probes/run-gu-three-width-probe.sh \
  <min_cosine> <max_abs_error> <min_end_to_end_speedup>
/tmp/atlas-w2a8-probes/run-down-three-width-probe.sh \
  <min_cosine> <max_abs_error> <min_end_to_end_speedup>
```

Each process compares the BF16 incumbent and W2A8 component at
M=1/63/64/65/127/128/129 plus four-expert cases with leading/internal and
trailing empty experts. Distinct per-K128 magnitudes and zero groups exercise
scale indexing. Prefix/suffix and inactive-tail guards, two different output
poisons, and wrong block/grid/N/bits/persistence launches detect partial writes
or escaped guards. PASS requires finite numeric metrics and a review-selected
minimum speedup for A8 quantization plus W2A8, averaged in baseline/candidate/
candidate/baseline order; kernel-only and quantization times remain diagnostics.
The paired runners execute the receipted N64 and N128 binaries sequentially,
write exclusive synthetic-output dumps in a private temporary directory, and
require the concatenated BF16 bytes to compare exactly. They also fail unless
all nine W2A8 output hashes plus the input and trellis identities match.
Individual `run-{gu,down}-n{64,128,256}-probe.sh` runners remain
available for isolated timing; `run-{gu,down}-probe.sh` aliases N64 for existing
workflows.
The existing pair runners remain supported and are not replaced by the
compile-only N256 component. The generated
`run-{gu,down}-three-width-probe.sh` sequences receipted N64, N128, and N256
binaries with the same inputs and explicit thresholds, then requires all nine
hashes, input/trellis identities, and raw BF16 dumps to match exactly across
the three widths before admitting any timing comparison. Until it runs
successfully on GB10, N256 has no runtime or throughput evidence.
The probe prints the accepted thresholds, device/driver/runtime, deterministic
input/trellis/output hashes, FP8 saturation/floor-scale counts, and all times.
The build receipt binds source, named build tools, flags, commit/status, exact
host-binary hashes, and stable
extracted cubins for all six variants. Host hashes bind the binaries actually
run; extracted cubins
provide the reproducible device-code identity because nvcc embeds unstable
temporary names in the host ELF. It also hashes the build script, records exact
GU/down compile commands, and embeds the manifest-derived build ID in each probe.
The generated runners verify the receipt and binary hashes before launch, then
print both identities; the probe prints the same build ID in its runtime log. A
nonempty retained-output directory is rejected instead of overwritten. The
standard `NVCC_PREPEND_FLAGS` and `NVCC_APPEND_FLAGS` injection paths are
also rejected because they would make the recorded command incomplete. Every
source, named tool, commit, and status digest is revalidated after GU/down
compilation and before emitting a receipt or runner. These
component times still do not substitute for five-run end-to-end TTFT.

W2A8 intentionally changes operand precision and K32 reduction. It requires a
native-RNE conversion dump, tagged fragment and one-hot MMA dumps, activation
saturation telemetry, CPU-quantized projection comparisons, layer-logit/top-k
agreement, quality gates, occupancy, and same-boot timing before promotion.
The current host wiring is experimental and default-off; those gates remain
prerequisites for promotion or a default change.
Even an ideal 2x reduction
of the historical ~1.25-second MoE bucket saves at most ~0.625 seconds, still
short of the 1.455-second reduction required by the rounded 2.66-second baseline
for 2,000 tok/s.

`ATLAS_EXL3_PREFILL_DUAL_PRE=1` adds an exact H4096 gate/up pre-rotation
entry. It widens each gathered BF16 activation lane once, then calls the same
multiply, H128, and BF16-rounding body twice with independent gate and up SUH
tables. The host uses this only with the persistent fixed-shape K2 path and
`ATLAS_EXL3_PREFILL_FUSED_POST=1`; all other paths retain the two legacy
launches. The entry rejects any K other than 4096 or grid/block other than
`[rows,4,1]` / `[256,1,1]`.

The safe alias schedule writes gate's rotated H4096 rows into
`expert_down_out` and up's into `expert_up_out`. Gate consumes its rotation
first; up then reads its distinct rotation while writing H2048 projected rows
over the now-dead gate rotation. The fused post consumes that result before
the down projection reuses `expert_down_out`. Dispatch checks the full
`rows*4096*2` capacity with checked arithmetic.

At N=2,410/top-6, this removes one launch and one 118,456,320-byte
(112.969-MiB) logical input read per layer: 43 launches and 5,093,621,760
bytes (4.744 GiB) across 43 layers. DeepSeek-V4 buffer sizing reserves the
second H4096 rotation even while execution is opt-in, adding 59,228,160 bytes
(56.484 MiB) at N=2,410 and at most 96 MiB for a 4,096-token arena relative
to the former H2048 buffer. Residency must therefore be checked even before
promotion.

The dual-only writer packs each naturally aligned adjacent BF16 pair into one
32-bit store. Both values still receive independent round-to-nearest BF16
conversion in their original lane order; the generic fallback retains its
scalar stores. With CUDA 13.0.88 for SM121a, the dual entry assembles at 224 SASS
instructions and 33 registers versus 144 instructions and 19 registers for
one single entry (288 instructions across two launches). Both have zero
shared/local/stack/spill memory and no barriers. It issues four packed stores
versus eight scalar stores across two legacy launches while preserving 48
FFMAs, 40 shuffles, and 12 multiplies. This is compiler evidence,
not device timing or a TTFT/tok/s claim.

```bash
bash scripts/check-exl3-dual-pre-sass.sh
```

The offline checker recompiles SM121a, validates the symbol/resource ceilings,
load/store and arithmetic instruction counts, and prints the cubin SHA-256.

```bash
ATLAS_TARGET_MODEL=deepseek-v4-flash cargo run --release -p spark-model \
  --example exl3_dual_pre_microtest --features cuda,gpu-examples
```

That GPU promotion gate byte-compares the two-launch oracle with the dual
entry at N=1/17/65/256/2,410, checks the NULL identity gather, uses distinct
signs and poison values, guards outputs with canaries, exercises every K/grid/
block guard, and only then reports 31-sample ABBA CUDA-event timing. It
compiles offline but has not been run on a GPU in this worktree.

`ATLAS_EXL3_PREFILL_FUSED_BLEND=1` adds a separate exact DeepSeek-V4 tail
which combines fixed-H4096 down-post/unpermute with the shared-expert blend.
One 256-thread CTA owns a token: its eight warps first reproduce the existing
gate dot/reduction, then each warp processes four H128 chunks. The standalone
and combined kernels include one common helper for the gate reduction and
final blend. The combined kernel also calls the same ordered top-k/Hadamard
body as the existing unpermute entry and explicitly retains both BF16
materialization boundaries.

The host enables this arm only for the exact persistent fixed-shape K2 path,
top-6 routing, a present shared expert, non-EP execution, no graph capture,
overlap disabled, and dumps disabled. Every mismatch retains the existing
unpermute followed by `moe_batched_blend`; EP still blends only after its
all-reduce. The environment variable is opt-in until hardware parity and
timing promote it.

At N=2,410 the structural saving is one launch and one 18.828-MiB routed BF16
write plus one 18.828-MiB reread per layer: 39,485,440 logical bytes/layer,
or 1,697,873,920 bytes (1.581 GiB) across 43 layers. These are eliminated
instruction-level bytes, not measured DRAM traffic or a TTFT/tok/s claim.
The SM121a fused entry assembles at 40 registers, 32 B ptxas shared memory,
and zero stack/local/spills. The pre-existing standalone blend now uses the
same helper and assembles at 15 registers, 32 B shared memory, and zero
stack/local/spills with CUDA 13.0.88. A compiler-opaque block guard makes its
shared-scratch precondition fail closed without changing that register count.

```bash
ATLAS_TARGET_MODEL=deepseek-v4-flash cargo run --release -p spark-model \
  --example exl3_fused_blend_microtest --features cuda,gpu-examples
```

That bounded gate byte-compares the legacy two-launch chain against the fused
entry at N=1/17/65/256/2,410, with a real gate and the legacy NULL-gate=1
semantics. It builds a stable expert-sorted top-6 inverse map over all 256
experts, guards both outputs with canaries, checks wrong H/top-k/grid/block
fail-closed behavior, and only then reports 31-sample ABBA CUDA-event timing.
Its 2026-08-27 GB10 run failed byte parity at the first gated N=1 case, so the
exact max profile forces this arm off and retains fused unpermute plus the
incumbent shared-expert blend.

The host-histogram fallback grid is expert-major
`(N/64, ceil(max_rows/M_TILE), num_experts)`. Persistent mode instead has
`num_experts*(N/64)` compact CTAs and loops over
`ceil(expert_rows/M_TILE)` inside the assigned CTA. For each M tile, one CTA:

1. stages one sorted `64x16` BF16 activation tile;
2. reads one contiguous four-tile EXL3 strip (`256 B` at K2, `384 B` at K3);
3. assigns one `16x16` trellis tile to each warp and decodes it directly into
   BF16 B-fragment registers using the same window extraction, 3INST
   operation, FP16 result and FP16-to-BF16 rounding as P1;
4. reuses that fragment across eight `m16n8k16` FP32-accumulating MMAs and
   stores the same sorted output layout as `moe_bf16_grouped_gemm`.

This preserves P1's surrounding H128 rotations and eliminates both the BF16
global scratch write and its grouped-GEMM reread. At the 2,410-token geometry,
the routed expert stream is approximately the resident trellis payload times
`ceil(rows_per_expert/64)` (one pass in the balanced 256-expert/top-6 model;
skew can add tiles), rather than P1's decoded-weight scratch traffic. This is a
traffic model, not a speed claim.

Offline gates currently passed:

- CPU tile-order bijection: all 256 `(k,n)` positions covered exactly once;
- CPU warp/accumulator model: every cell of the `64x64` output tile has one
  owner and each warp consumes only its assigned 16-column trellis fragment;
- CPU expert-offset model: every sorted routed row covered exactly once,
  including empty experts and 64-row boundaries;
- CPU BF16x8 staging model covers every activation element exactly once and
  proves 16-byte alignment at K=2,048/4,096 for K16/K64 stages;
- the targeted Rust/source models pass and the GB10 kernel set compiles;
- CPU expert-strip model covers every live M/N tile once. At current config
  dimensions (256 experts, top-6, H=4,096, MoE intermediate=2,048), it replaces
  7,405,568 M64 rectangular slots per modeled layer (3,702,784 for M128) with
  32,768 exact outer strips. Gate/up each launch 8,192 CTAs (171 waves on 48
  SMs) and down launches 16,384 (342 waves), instead of 96 CTAs serially walking
  85 or 171 strips apiece. The counts are scheduler arithmetic, not timing;
- persistent M64 `sm_121` cubin: 64 registers, 2,688 B declared kernel shared
  memory, zero stack, zero spills, eight BF16 HMMA instructions per K tile,
  and no local load/store or atomic instructions;
- K64-stage M64 `sm_121` cubin: 64 registers, 9,984 B declared kernel shared
  memory, zero stack/spills/local memory/atomics. Its generated body contains
  32 static HMMA instructions per four-slice stage versus eight in K16;
- fixed-K2 K16/K64 cubins preserve the CPU trellis-layout model and have
  explicit generic fallback plus GPU-gated byte-parity cases;
- hybrid M128 `sm_121` cubin: 64 registers, 4,992 B declared shared memory,
  zero stack/spills; its CPU ownership model covers every `128x64` output cell
  once and proves tail masking matches M64 padded-MMA work;
- fused post/SwiGLU/down-pre `sm_121` entry: 39 registers, no shared/local
  memory, barriers, stack, or spills; its CPU model proves both intermediate
  BF16 boundaries are load-bearing;
- fused down-post/unpermute `sm_121` entry: 39 registers, no shared/local
  memory, barriers, stack, spills, or atomics; CPU models cover warp ownership,
  post BF16 rounding, and top-k accumulation order.
- exact H4096 shared-tail entry: 40 registers, 32 B shared memory, zero
  stack/local/spills; CPU/source models cover its one-CTA ownership, ordered
  top-k body, two BF16 barriers, common gate tree, opt-in host fallback, and
  43-launch/1.581-GiB logical-traffic model.
- exact dual H4096 pre entry: 33 registers and zero shared/local/stack/spill
  memory; CPU/source models cover independent transforms, single input loads,
  exact geometry, checked capacity, safe alias lifetime, its GPU promotion
  harness, and the 43-launch/4.744-GiB logical-traffic model.
- exact shared-expert V2 M6 entry: 904 instructions, 80 registers, three
  branches, and zero spills versus 1,232/76/31 for the runtime-M incumbent;
  `scripts/check-exl3-shared-v2-sass.sh` proves equal per-row
  FFMA/FADD/shuffle counts and resolves every M=4/5/6/8/16 entry.

### Shared-expert load-time FP8 A/B

`ATLAS_EXL3_SHARED_PREFILL_FP8=1` predequants only the three NVFP4 shared-expert
matrices to persistent E4M3 at model load. The routed EXL3 weights and router
gate remain unchanged. Its dedicated launcher retains BF16 activations and one
`fp8_gemm_t` launch per projection rather than inheriting the default LDMAB
activation-quantization launch, so the experiment changes one precision arm.
Missing/wrong-format shared weights fail closed, partial mirror construction is
freed before returning an error, and the original NVFP4 weights remain the
default fallback.

At H=4,096/shared-intermediate=2,048 the mirrors cost 25,165,824 bytes per MoE
layer, or 1,082,130,432 bytes (1.008 GiB) across 43 layers. At N=2,410, the
38 M64 tiles otherwise revisit 41,120,956,416 FP4 values across those layers.
This is operation-count and allocation arithmetic, not DRAM traffic or time.
The existing historical waterfall bounds the whole shared-expert opportunity
far below the 1.455-second reduction needed from the rounded 2.66-second
baseline for 2,000 tok/s.

Unlike the exact routed fusions, this arm is not byte-identical: the current
baseline uses BF16 MMA while the arm stores E4M3 weights and uses FP8 MMA. GPU
promotion therefore requires memory-residency proof; baseline-versus-arm
cosine/logit checks at M=1/63/64/65/2,410 for both production shapes; full
output hashes and tool/prose quality; then five fresh zero-cache TTFT samples.

### M6/M16 device-built verify worklists

`ATLAS_EXL3_VERIFY_WORKLIST=1` is an experimental exact-K2/top-6 decode arm
for exactly 6 or 16 verify rows. A one-CTA same-stream builder compacts the
routed slots into one record per distinct expert. M6 retains its original
32-byte `{expert,count,slots[6]}` ABI and 36-record capacity. M16 uses a
separate 80-byte `{expert,count,slots[16],reserved[2]}` ABI and 96-record
capacity, with the reserved tail required to be zero. A single maximum-sized
7,680-byte allocation serves either exact-width arm without changing the M6
stride. Every other row count stays on the incumbent ladder.

Fixed 96-CTA gate/up and down kernels grid-stride the compact logical work and
call the same exact-width trellis/Hadamard/split-combine body as the incumbent.
For either width, if `U` is the union of routed experts, the compact path runs
`192U` gate/up and `96U` down body tasks. M16 therefore removes
`288*(96-U)` logical tasks per layer versus its 27,648-task incumbent. Full
overlap (`U=6`) removes 25,920 tasks per layer; no overlap (`U=96`) removes
none. This is scheduling arithmetic, not device timing or a decode tok/s
claim. The corresponding M6 hash case still removes 8,640 tasks per layer.

The CPU oracle checks both exact wire layouts, canonical M16 padding, strict
decoding, full-overlap and no-overlap routes, 512 M6 plus 128 M16 adversarial
routes, once-only coverage of all 36/96 slots, capacity failures, late
duplicate/out-of-range rejection, and the full 0..capacity task/counter
bijection for every split from 1 through 12. The CUDA builders validate the
complete route before emitting any record and bound every slot write.

SM121a compilation reports the M16 builder at 20 registers and no shared
memory. Both M16 persistent consumers use 128 registers and 37,972 B shared,
versus 128 registers and 40,020 B shared for their incumbent M16 twins. All
five M16 entries have zero stack, local memory, and spills. The retained M6
measurements remain 18 registers for the builder, 109/104 registers and
28,708 B shared for gate/up and down, and zero stack/local/spills. These are
compile-time resource gates; actual GB10 occupancy and timing remain device
promotion gates.

```bash
bash scripts/check-exl3-persistent-worklist-sass.sh
CUDARC_CUDA_VERSION=13000 cargo test -p spark-model \
  --test exl3_persistent_worklist_model
```

Malformed input records a nonzero device status; without a device-to-host
synchronization or dynamic fallback, each consumer deterministically zeros all
36 or 96 routed output rows instead of blending stale scratch. This does not
surface a host request error: the shared expert still contributes, so the
contract is memory/stale-output containment rather than full request-failure
propagation. The layer-local worklist, counter, and split scratch are safe only
under the server's current single-scheduler/same-stream contract; future
overlapping execution needs per-stream scratch or an explicit lease. Promotion
still requires M6 and M16 malformed-route canaries, duplicate-heavy/no-overlap/
full-overlap byte parity, repeated graph replay to expose stale split counters,
and CUDA-event timing. Until those GPU gates pass, the environment flag must
remain off.

Required GPU gates before defaulting or quoting tok/s:

1. run the compiled `exl3_gemv_microtest` P2 gate, which byte-compares generic
   and fixed-K2 K16/K64 exact-grid, undersubscribed grid-stride, and full
   compact-grid output with P1 on K2/K3 synthetic trellis at
   1/63/64/65/129 rows per expert; the harness is present but has not been
   executed on hardware;
2. run its fused-post/pre gate, which byte-compares the one-pass result with
   the legacy post+post+SwiGLU+pre composition using distinct per-expert sign
   vectors;
3. run its fused-tail gate, which byte-compares down-post/unpermute against the
   legacy composition with a nontrivial reverse map and top-k weights;
4. run `exl3_fused_blend_microtest` for gated/NULL-gate production and short-
   tail byte parity, fail-closed guards, and legacy-vs-fused CUDA-event timing;
5. run `exl3_dual_pre_microtest` for sorted and identity-gather byte parity,
   output canaries, every exact-geometry guard, and two-launch-vs-dual timing;
6. byte-compare both exact and persistent M128 against P1 at the existing
   K2/K3 row-boundary cases, then time M64 versus M128 at the real histogram;
7. real-checkpoint layer parity and full logits/output-hash parity;
8. same-boot P1/P2 and K16/K64 CUDA-event timing at the production expert
   histogram;
9. five-run median TTFT plus the mandatory tool/prose quality gates, including
   memory residency with the enlarged DeepSeek-V4 expert arena.

The next optimization rung is deliberately deferred until the parity and
timing gates identify a measured bottleneck; offline resource counts alone do
not establish that this kernel moves end-to-end TTFT.
