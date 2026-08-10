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

## 2. Kernel design (`exl3_gemv_m1`)

Grid `(N/128, SPLIT_K)`, block 256 threads (8 warps), static smem 42.5 KB,
52 registers, no spills (sm_121a).

- **Strip ownership**: block owns a 128-wide output strip = 8 tile-columns,
  which makes the output Hadamard-128 block-local. Warp w owns tile-column w;
  each lane accumulates 2 outputs (`n = 16w + lane/4`, `+8`) in fp32.
- **Phase 1 — input pass**: x' (fp32) for the block's 128-aligned K-slice
  into smem (up to 32 chunks = 16 KB); one 128-chunk per warp per iteration
  via the warp-shuffle Hadamard. Issued AFTER the first trellis cp.async
  stages so the DRAM stream starts immediately.
- **Phase 2 — weight stream**: 2-stage cp.async (`.cg`, 16 B per thread)
  pipeline; each stage = 16 tile-rows × 8 tiles = 12 KB, fetched as
  contiguous 768-B runs per tile-row (stride N/16·96 B between rows). Every
  trellis byte is read exactly once, as 128-bit transactions. A warp decodes
  one 96-B tile per iteration: `dq8` gives 8 weights/lane from two u32 smem
  words; products and accumulation in fp32 (`__fmaf_rn`).
- **Phase 3 — reduce + output**: quad shuffle-reduce → 128 fp32 partials in
  smem. `SPLIT_K = 1`: the block applies Hadamard-128 + svh and stores bf16.
  `SPLIT_K > 1`: each split publishes its raw partial to `ws[split][N]`; an
  atomic counter elects the LAST split, which combines partials in fixed
  split order (deterministic), then does the output pass. Counters self-reset
  → back-to-back launches need no host memset.

Constraints: `N % 128 == 0`, `K % 128 == 0`, per-block K-slice ≤ 4096
(`EXL3_MAX_XCHUNKS`); use `SPLIT_K > 1` for larger K and for occupancy
(N=2048 → 16 strips → SPLIT_K=3 → 48 blocks ≙ 48 SMs).

## 3. Dequant instruction budget

Per 96-B tile (256 weights) per warp, per lane (8 weights):

| stage | lane-ops |
|---|---|
| 2 × LDS.32 (tile words a, b) | 2 |
| window extraction (2 × 64-bit funnel shift + 6 derived `>>3` + 8 AND) | ~18 |
| 3INST (8 IMAD + 8 LOP3 + 8 PRMT lo/hi + 4 HADD2) | 28 |
| half2→float2 (4 × cvt) | ~8 |
| x' loads (2 × LDS.64, k-pairs shared between both n-halves) | 2 |
| dot (8 FFMA) | 8 |

≈ **66 lane-ops / 8 weights ≈ 8.3 ops/weight** (the plan's 4.3 counted only
extraction+decode; the fp32-exact dot and conversions add the rest).

Roofline: at the 229 GB/s ceiling the stream demands 229/0.3765 B ≈
608 Gweights/s → ~5.0 T lane-ops/s against ~23 T lane-op issue capacity
(48 SM × 2.5 GHz × ~192 lanes) → **~22 % issue utilization. Decode is not
the bottleneck**; the kernel is DRAM-bound as intended.

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
   (determinism of the split combine).
3. **Cold-rotation GB/s** through a ≥512 MB weight ring (defeats the 24 MB
   L2) at both expert shapes; judged against the 229 GB/s ceiling.

## 6. Open items toward S6 (serve integration)

- m-row (γ-verify) MROW variant — mandatory from day one per plan §3/§4.7
  (partial-exactness law: the verify chain flips as a whole).
- Real-checkpoint spot-check (plan option a): run the dump gate against
  tiles from `/home/flocka/sparkinfer-ref/data/tp1` once readable.
- Perf tuning after first GPU measurement: stage depth, `SPLIT_K` policy,
  possible half2-accumulate variant (`EXL3_GEMM_H_ACC`-style) if issue-bound.
- Shared-expert rider (grid.y expert slot) when wiring into
  `moe_shared_expert_fused_t` dispatch.
