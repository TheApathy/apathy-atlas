# CB3 expert pack — the bit-level format, and the one thing still unsettled

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
