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
  (≤ 16 chunks = 2048 k), computed in fp32 via the warp-shuffle Hadamard,
  stored as packed `__half2` (k, k+1) pairs (4 KB smem — round 3), refilled
  per superblock inside the stage loop — the per-slice K cap is gone.
  Issued AFTER the first trellis cp.async stages so the DRAM stream starts
  immediately.
- **Phase 2 — weight stream**: 2-stage cp.async (`.cg`, 16 B per thread)
  pipeline; each stage = 8 tile-rows × 8 tiles = 6 KB = exactly one 128-k
  chunk, fetched as contiguous 768-B runs per tile-row (stride N/16·96 B
  between rows). Every trellis byte is read exactly once, as 128-bit
  transactions. A warp decodes one 96-B tile per iteration: `dq8` gives 8
  weights/lane from two u32 smem words. The dot runs in `__half2` HFMA2
  chains within each 128-k chunk — 4 independent chains ({acc0,acc1} ×
  even/odd tile-row, depth 4 each) — combined into fp32 in FIXED order once
  per chunk (round 3; numerics tier in §3b). The row loop is fully unrolled
  (8 tiles per stage per warp).
- **Phase 3 — reduce + output**: quad shuffle-reduce → 128 fp32 partials in
  smem. `SPLIT_K = 1`: the block applies Hadamard-128 + svh and stores bf16.
  `SPLIT_K > 1`: each split publishes its raw partial to `ws[split][N]`; an
  atomic counter elects the LAST split, which combines partials in fixed
  split order (deterministic), then does the output pass. Counters self-reset
  → back-to-back launches need no host memset.

Constraints: `N % 128 == 0`, `K % 128 == 0`; any `SPLIT_K ≥ 1` (the x'
superblock loop removed the old ≤4096-K slice cap). Use `SPLIT_K` for
occupancy: at 4 CTAs/SM the GB10 has 192 slots (N=2048 → 16 strips →
SPLIT_K=12 fills exactly; N=4096 → 32 strips → SPLIT_K=6).

Shared-memory budget: `s_x` 4 KB + `s_stage` 2×6 KB + `s_y`/`s_elect`
~0.6 KB ≈ 16.6 KB → 4 CTAs/SM on the 100 KB GB10 SM; registers are the
binding limit (64 regs × 256 thr × 4 CTAs = the full 64 K file).

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
   L2) at both expert shapes; judged against the 229 GB/s ceiling.

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

STILL OPEN:

- Prefill / M>1 (plan §3 P1): `forward_prefill` / `forward_batched` fail
  loudly. Design: per-expert `exl3_dequant_dump` to F16/BF16 scratch + an
  M-row H128 activation pre-pass (`x' = H128(diag(suh)·x)`) and post-pass
  (`y = diag(svh)·H128(y0)`), feeding the existing grouped BF16 GEMM.
- m-row (γ-verify) MROW variant — `forward_km` declines for EXL3 (falls to
  the guarded per-row path); the exact-verify twin is mandatory S6 scope.
- Real-checkpoint spot-check: run the dump gate against tp1 tiles.
- Perf tuning after first GPU measurement: stage depth, `SPLIT_K` policy
  (dispatch default fills ~96 CTAs; `ATLAS_EXL3_SPLIT` overrides).
  Half2-accumulate landed in round 3 (§3/§3b).
- Fused gate+up / silu-in-GEMV riders to cut the 3·top_k+3 launch count.

- m-row (γ-verify) MROW variant — mandatory from day one per plan §3/§4.7
  (partial-exactness law: the verify chain flips as a whole).
- Real-checkpoint spot-check (plan option a): run the dump gate against
  tiles from `/home/flocka/sparkinfer-ref/data/tp1` once readable.
- ~~Perf tuning after first GPU measurement~~ round 2 done (ILP restructure
  + 4 CTAs/SM, +8%); ~~half2-accumulate variant~~ round 3 done (§3/§3b):
  half2 dot + half2 x' + full unroll, 67→55.9 SASS instr/tile, expected
  ~195–215 GB/s — needs the gate-2 quality re-gate on hardware. If gate 3
  still lands under ~200 GB/s after round 3, the remaining suspects are
  the per-stage barrier/cp.async wait overhead (measure with a
  barrier-free single-stage probe) and grid-size occupancy at the
  production split, NOT the instruction stream.
- Shared-expert rider (grid.y expert slot) when wiring into
  `moe_shared_expert_fused_t` dispatch.
