# F15: upstream Flash-Next prefill transfer

2026-09-16, `perf/qwen38-flash-next`, HEAD
`11e76f29a68ed5e22be8fd14cc72e318bdcc6111` plus preserved dirty source.
This is a pinned source comparison and a CPU-tested PLE I/O change. The active
DeepSeek GB10 reservation was not disturbed, so no new Atlas model timing,
GPU parity, server, vision, or 2,000 tok/s claim is made here.

## Pinned inputs

- `bilikaz/qwen38-flash-next-recipe` at
  `b570be67ef20093f71b8aad0a4c8bcfaa9109d87` (v3, 2026-09-14).
- `deathbyorderfill/Qwen3.8-Flash-Next-NVFP2-SD-Quant` at
  `d26bae0f4e8d3669bfe7c36b1425167cef6c601b`.
- official `vllm-project/vllm` v0.29 source at
  `98dff2a81d747d1dba01a47f939f48c3526d4206`.

The bilikaz serving image's eight private vLLM patches are referenced through
`bilikaz/myllmbox-runner`, but that repository is not anonymously readable.
Their exact source therefore cannot be reviewed or represented as transferred.

## What transfers

| Upstream mechanism | Atlas relevance | F15 disposition |
| --- | --- | --- |
| Parallel, delayed PLE population from checkpoint-backed storage | Reduces serialized cold sparse-table stalls without permanently allocating the 26.9 GiB table on GPU | Transfer the bounded parallelism, but keep Atlas's O_DIRECT sidecar and no private cache by default |
| Adaptive large-row PLE prefetch only after observed major faults | The NVFP2 repo reports cold gather 6,516 ms to 437 ms, while unconditional warm prefetch regresses 35 ms to 150 ms | Do not copy its mmap/process_madvise code into Atlas's different 8 KiB O_DIRECT format; retain as a future PageCache-only candidate with measured fault gates |
| Rows-per-expert MoE tile selection | Atlas's 2,048-token profile spends 3.033 s in compact expert GEMM | Port the policy, not Triton constants: specialize the existing CUDA kernel by observed active rows/expert and prove routing/output parity |
| Chunk-64 GDN prefill with fused preparation and a final recurrent state | Replaces serial recurrence and is required for scalable long prompts | Highest-value architectural port, but it requires new CUDA source, state parity, split-chunk continuation, and an isolated GPU lane |
| Profile-time GDN kernel warmup before KV allocation | Prevents first-request autotune allocation spikes in vLLM | Not directly applicable to Atlas's precompiled kernel dispatch |
| NVFP4 output head, native MTP K=3 | Improves bilikaz decode, not the dominant Atlas prefill path | Keep in the decode lane; do not credit it to prefill |
| `vm.compaction_proactiveness=0` | bilikaz attributes removal of periodic multi-second stalls to this host policy | Runbook A/B only; never bake a host-global sysctl into model correctness or a kernel speed claim |

## Implemented in F15

The opt-in whole-prompt PLE path previously opened a 32-entry io_uring and
manually submitted selections in 32-row chunks. At 2,048 tokens and 16 heads,
that forced 1,024 submit/wait windows even when the device could sustain a
larger queue.

`PleOffloadReader::read_rows_windowed` is now the single batching authority.
It preserves caller order and delegates every bounded window to the existing
page-coalescing, validation, cache, and error path. The Qwen4 PLE reader uses a
256-entry queue: 2 MiB of registered 8 KiB buffers and 128 maximum windows for
32,768 selections. Default O_DIRECT and zero private-cache policy are unchanged.

Tests first failed because the planner did not exist. After implementation,
the focused two planner tests pass, and the full `spark-storage` library suite
passes 19 tests with seven GPU tests explicitly ignored. Rustfmt and diff
whitespace checks pass. Gated Qwen4 runtime syntax is rustfmt-parsed, but no
native CUDA build or actual-model performance qualification was run.

## Why this is not yet 2,000 tok/s

F12's 2,048-token diagnostic remains 117.7331 effective input tokens/s, not
isolated GPU prefill. Its provisional trace attributes 5.440 s to exact M32
projections, 3.033 s to compact expert GEMM, 4.644 s across the listed
attention projection/core groups, 0.895 s to dense GEMV, and only 0.659 s to
the observed GDN recurrence group. PLE is not a reported leading kernel group.

Consequently, eliminating all observed GDN time alone cannot approach 2,000
tok/s on this short path. The target needs a stack change: tensor-core-safe
projections, shape-specialized compact MoE, correct batched dense-window/QSA
attention, and chunked GDN. PLE I/O matters chiefly for cold and longer prompts.

## Ordered qualification ladder

1. Reclaim a fresh isolated Flash-Next GB10 lane and A/B F15 at 256, 2,048,
   8,192, and 16,384 input tokens. Record cold and warm PLE time, major faults,
   per-request TTFT, isolated GPU prefill, output hash, cache state, and memory.
   Reject F15 if warm TTFT or tail latency regresses.
2. Reprofile 2,048 and 16,384 with complete CUDA-event capture. Optimize the
   largest measured group, not a presumed upstream bottleneck.
3. Complete the F13 projection numerical/quality gate before timing its `out`
   or `all` modes. Never promote a wording-only comparison.
4. Implement F14's race-free dense-window attention plan and continuation
   tests before any QSA batching claim.
5. Add rows-per-expert variants to the current compact NVFP4 MoE kernel, with
   exact router/worklist and output comparisons for sparse through dense bins.
6. Port official vLLM's chunk-64 GDN algorithm as a new default-off Atlas path.
   Compare hidden output, final recurrent state, convolution state, resets,
   split chunks, and next-token continuation against the scalar path.
7. Only after each local gate passes, run same-binary ABBA performance and
   text-plus-vision product qualification. External cache-warm numbers are
   targets, not Atlas receipts.

## Pinned source links

- https://github.com/bilikaz/qwen38-flash-next-recipe/blob/b570be67ef20093f71b8aad0a4c8bcfaa9109d87/README.md
- https://github.com/bilikaz/qwen38-flash-next-recipe/blob/b570be67ef20093f71b8aad0a4c8bcfaa9109d87/recipe.yaml
- https://github.com/deathbyorderfill/Qwen3.8-Flash-Next-NVFP2-SD-Quant/blob/d26bae0f4e8d3669bfe7c36b1425167cef6c601b/model/qwen4_exp.py
- https://github.com/deathbyorderfill/Qwen3.8-Flash-Next-NVFP2-SD-Quant/blob/d26bae0f4e8d3669bfe7c36b1425167cef6c601b/sglang_patched/fused_moe.py
- https://github.com/vllm-project/vllm/blob/98dff2a81d747d1dba01a47f939f48c3526d4206/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py
- https://github.com/vllm-project/vllm/tree/98dff2a81d747d1dba01a47f939f48c3526d4206/vllm/model_executor/layers/mamba/ops/gdn_chunk_cutedsl
