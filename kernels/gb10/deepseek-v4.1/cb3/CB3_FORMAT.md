# CB3 expert pack — the bit-level format, K-order now SETTLED

Derived from the shipped pack (`k154-cb3/layers/layer-NN.safetensors`, layout
`cb3-v2`) and cross-checked against the PTX the Python prefill path runs
(`dsv41-prefill-work/tools/cb3_moe.py::_cb3_asm`).

## Shapes, per layer shard

| tensor | shape | meaning |
|---|---|---|
| `w1_lo`, `w3_lo` | [154, 2304, 1280] | 2 low bits per weight, 4 per byte. K = 5120 |
| `w1_hi`, `w3_hi` | [154, 2304, 640] | 1 high bit per weight, 8 per byte |
| `w1_cb`, `w3_cb` | [154, 2304, 8] | 8-entry codebook PER OUTPUT ROW, e2m1 in the low nibble |
| `s1`, `s3` | [154, 2304, 160] | UE8M0 scale per 32 weights (5120/32 = 160) |
| `w2_*` | [154, 5120, ...] | the down projection: K = 2304, 2304/32 = 72 scales |

DIM 5120, INTER 2304, 154 experts per shard, 40 layer shards, 83 GB total.

## The index reconstruction

A weight is a 3-bit codebook index: two low bits from `lo`, one high bit from
`hi`, selecting one of the row's 8 codebook entries, then scaled by its group's
UE8M0 exponent.

**THE HIGH-BIT INTERLEAVING IS NOT SEQUENTIAL, AND THIS IS THE PART THAT LOOKS
OBVIOUS AND IS WRONG.** The natural reading — hi bit *k* belongs to weight *k* —
does not hold. `_cb3_tile128` splits a 128-weight tile as `lo[0:32]` + `hi[0:16]`
and makes four decoder calls whose `(shift, hi-bit)` pairs are
`(0,0) (4,2) (0,4) (4,6)`, each expanding to two halves. Working that through:

> `hi` byte *j*'s LOW nibble (bits 0-3) supplies the high bits for `lo` byte *j*'s
> four weights; its HIGH nibble (bits 4-7) supplies them for `lo` byte *j+16*.

Verified: reconstructing a real row both ways — semantically, and by running the
PTX model over the same bytes — gives **equal multisets** only with this mapping.
The naive sequential reading gives a different multiset, i.e. genuinely different
weights, not merely a permutation.

## What is settled, and what is not

**SETTLED, and pinned by tests:**
- the decoder itself, `cb3_decode.cuh`, 256/256 vectors bit-exact against an
  independent numpy model of the same PTX (`cb3_decode_vectors.txt`)
- the shapes, the codebook-per-row layout, the 32-element scale grouping
- the high-bit interleaving above

**NOT SETTLED: the K-ORDER.** The PTX emits its 128 weights field-major within a
u32 lane (shift 0 across four `lo` bytes, then shift 2, …); the byte-major
reading gives the same multiset in a different order. Which one indexes K is
decided by how `_cat2` reassembles the halves and what `tl.dot_scaled` expects
along K — and that is a Triton layout question, not something the byte format
answers.

**THIS CANNOT BE SETTLED BY READING.** An ordering that is wrong pairs every
weight with the wrong activation and produces output that is finite,
correctly-shaped, and meaningless — precisely the "plausible garbage" the V4.1
loader's hard stop exists to prevent. It needs a CAPTURED REFERENCE from the
Python engine: one expert's inputs and its output, against which a candidate
ordering either reproduces the bytes or does not.

`make_gemm_fixture.py` builds a float64 reference from the real pack and is
correct up to that ordering; it is deliberately NOT presented as ground truth
for a GEMM until the capture exists.

## SETTLED 2026-09-21: the K-order, empirically

The section above said this could not be settled by reading, and that was right
about the FORMAT — but the tree already contained the answer in code:
`dsv41-prefill-work/tools/cb3.py`. Two things the format notes above missed:

1. **K is blocked.** `block_plan(K)` splits a row into 512-weight blocks, with a
   256 tail where 512 does not divide: K = 5120 (w1/w3) -> 10x512 + 0; K = 2304
   (w2) -> 4x512 + 1x256. The 256 tail exists so w2's hi tile is not 32 B wide
   for the whole row, which would cap the kernel near 100 GB/s.

2. **`_v2_fields(block_w)` is the mapping**, and it is unambiguous. Within one
   block, for scale-group `g` (0..block_w/32) and sub-position `r` (0..1):

       K index  = g*32 + lane*2 + r         (lane 0..15)
       lo byte  = g // 2,  lo_shift = 4*(g%2) + 2*r
       hi byte  = (g//2) // 2,  hi_bit = ((g//2)%2)*4 + (g%2)*2 + r

   This also explains this decoder's four PTX specialisations
   `(sh,hb) = (0,0) (4,2) (0,4) (4,6)`: they are exactly the `r = 0` cases, each
   call covering its `r = 1` partner as the odd nibble. 4 x 2 = the 8 combinations.

### The measurement, with its negative control

Shipped Triton kernel (`moe_forward_v2`) vs a float64 CPU reference built from
`dequant_cb3_v2`, one expert from the real `layer-00.safetensors`, 4 tokens:

| ordering | rel_l2 | cosine |
|---|---|---|
| TRUE (`unpack_cb3_v2`) | **2.32e-03** | **+0.999997** |
| `r` sub-position swapped (xor 1) | 1.35e+00 | +0.088 |
| field-major <-> byte-major in block | 1.35e+00 | +0.094 |
| random permutation of K | 1.33e+00 | +0.120 |

2.32e-03 is bf16 rounding (the kernel accumulates fp32 from bf16 activations),
not a permutation. The three controls are permutations of the SAME multiset, so
each is "finite, correctly-shaped and meaningless" — the exact failure this file
warned about — and the two the file named as genuinely ambiguous (the r-swap and
the byte-major reading) both fail by ~580x. **The check has been watched FAIL**,
which is what makes the agreement evidence rather than a gate that cannot fail.

Captured reference for the Rust/CUDA decoder to validate against:
`korder_ref_W1.npy` (dequantized [16, 256] bf16->f32) with the exact input
planes beside it (`korder_ref_planes_{lo,hi,cb,s}.npy`). Reproduce with
`korder_capture.py`; re-run `korder_control.py` before trusting any change.

STILL OPEN for the Rust lane, and neither is implied by this: the MoE GEMM, and
GPU residency for the 83 GB K154 pack.
