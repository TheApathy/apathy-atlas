# Mia to Atlas DeepSeek performance port map

Source baseline: `tpurtell/ds4-mia-exl3-k2-1spark` commit
`f20b97dfd7666c00c316f29542e2e53f33cabb19`, with B12X commit
`28e083482fd18ca3ce0e2553cd533102be85552f` in the locally rebuilt image.

## What Mia actually does differently

### Expert-major route packing

Mia's B12X W4A16 path converts `topk_ids` into four persistent device buffers:

- packed route indices grouped by expert;
- one expert ID per padded route block;
- a device-side packed-route count;
- `num_experts + 1` expert offsets.

For DeepSeek verification width six, the host planner selects an 8-row expert
block. Routing is padded per expert and the projection kernels stream each
selected expert's weights over its grouped rows. Buffers are allocated before
CUDA graph capture and warmed for the exact DSpark widths.

Atlas already owns most prerequisites: `moe_sort_by_expert`, device expert
offsets, `moe_build_tile_worklist`, persistent FP8 grouped GEMM, and exact EXL3
multi-row kernels. The missing connection is an EXL3 decode path driven by a
compact device worklist. The existing EXL3 verifier instead launches against
the routed slot geometry and performs leader election inside each projection.

This means the next implementation should reuse Atlas's routing SSOT rather
than port Triton or vLLM:

1. Build compact `(expert, route-mask or row-list)` records from the existing
   gate output entirely on device.
2. Preallocate worklist/count scratch for the maximum 16-row native DFlash2
   verify so CUDA graphs see stable addresses.
3. Launch a persistent EXL3 gate/up kernel over work records, not routed slots.
4. Reuse the same records for down projection, preserving per-row reduction
   order and routing weights.
5. Keep the current MROW implementation as the exact oracle and fallback.

The historical promotion floor remains at least 213 GB/s cold-weight
microbenchmark bandwidth plus exact output parity. Merely sorting expert IDs
without a persistent worklist previously regressed and is not this design.

### DSpark proposal path

Mia preallocates stable input, hidden, position, slot-index, and output buffers,
then captures the whole draft forward in a CUDA graph. It also skips draft
logits and the confidence head unless a configured sampling or diagnostic path
needs them. Atlas already batches the LM head and keeps the Markov chain on
device, but still exposes separate projection/attention launches inside each
of three stages.

Atlas port order:

1. graph the fixed five-row `propose_block` using its existing owned scratch;
2. preserve the single final synchronization used for tokens/confidence;
3. add a no-confidence fast path only after proving the scheduler does not use
   confidence gating;
4. warm every exact width before the first measured request.

### Adaptive depth and probabilistic drafting

Mia contains threshold and hardware-aware confidence schedulers, but its
default published one-Spark recipe uses five probabilistic proposals. Atlas
already supports confidence prefix truncation and adaptive fallback. These are
secondary until proposal plus six-row verification beats serial decode; they
cannot repair an expensive verifier.

## Non-GPU completion versus live gates

Completed without GPU:

- exact Mia image rebuild and lineage record;
- local-checkpoint launch contract with busy-GPU refusal and dry-run output;
- locked streaming benchmark with a tested decode-only clock;
- 1M-capable circular DSpark capture;
- exact offline DFlash2 corpus validator;
- source-level mapping from B12X route packing to Atlas primitives.

Still requires a free GPU:

- Mia boot and kernel compilation;
- locked same-prompt Mia/Atlas timings and hashes;
- circular-capture wrap parity;
- dispatch proof and bandwidth measurement for a persistent EXL3 worklist;
- proposal CUDA-graph parity and timing.
