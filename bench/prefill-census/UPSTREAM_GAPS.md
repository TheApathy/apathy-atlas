# F9 upstream prefill findings

2026-09-05, source audit only. No imported runtime, donor weights or new timing.

## Pinned references and metric limits

Mia repository HEAD was freshly verified as
`203834ca88000c8192112e396b80d886b522caa0`:
[reference repository](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark/tree/203834ca88000c8192112e396b80d886b522caa0).
Its reported roughly2265 tok/s is client-TTFT-derived prefill, using a different
mixed MXFP8/NVFP4 checkpoint and workload. The inspected sparkDash client mostly
uses repeated filler,8 outputs, one trial per size and UUID prefixes; it does not
establish Atlas's exact-count/cache0/quality qualification. Its fallback can use
character estimates if token usage is missing. Do not compare that directly with
our factual-summary five-trial server-TTFT census or call it on-GPU prefill.

Actual official vLLM source was retrieved at merged PR53896 commit
`e126687a9a828d513c01a07cd69f025f27d63280`. This is an available successor reference,
**not proof of the exact source inside Mia's benchmark image**. The image was
absent locally and was not pulled or executed. Local reference checkout:
`/var/tmp/atlas-vllm-f9-reference.OT8RdkPv`.

## Material architectural gaps

1. **Chunked recurrent prefill.** The reference uses whole-query convolution and
   `fla_chunk_gated_delta_rule` with initial/final state and precomputed sequence
   metadata. GB10 is not admitted by its SM90/SM10.x FlashInfer prefill selector;
   do not claim that backend runs here. Atlas F8 still invokes the shipping
   scalar recurrent core for each token in all36 SSM layers. At2048 tokens that
   is73,728 scalar GDN launches before convolution/norm/HC/projections. This is a
   source-derived count, not a measured profile. Our existing WY helper is not
   automatically the same chunked tensor-core algorithm.
2. **Batched hyperconnections and projections.** Reference decoder forward
   processes the whole hidden-state tensor. HC merges down/injection projections,
   uses unquantized BF16 linears and fuses preceding residual combination with
   normalization. Atlas's shipping HC uses per-row quantized projections. A port
   needs correct original weights and same-input state/output comparisons; it is
   not a precision-preserving flag change.
3. **Native FP4 MoE and scheduling.** Ordinary ModelOpt NVFP4 supports dynamic
   FP4 activations/static FP4 weights with block-scaled tensor-core grouped GEMMs.
   It builds actual expert problem sizes and fuses gate/up and SiLU/requantization.
   Atlas F8's original routed path expands packed weights to BF16 and uses BF16
   MMA, with separate gate/up launches and a conservative expert-grid ceiling.
   First eliminate empty tile work while retaining arithmetic. A later W4A4
   backend needs packed merged layout, swizzled scales, alpha conventions,
   calibrated activations and independent quality gates.
4. **Precision is per component.** Reference router/shared gate are unquantized;
   shared MLP follows the mixed checkpoint configuration. Atlas predequants
   router/shared weights to FP8, and its prefill helper converts BF16 inputs to
   unscaled E4M3 before FP8 MMA. It is not W8A16 arithmetic. Do not treat F8 FFN
   grouping as automatically numerically identical to serial decode.
5. **Sparse attention.** Reference QSA actually consumes selected top-k indices.
   Atlas's broader legacy batching can update the indexer yet run ordinary dense
   attention. Long-context retrieval/state correctness is a separate prerequisite.

Source anchors at the pinned vLLM commit:

- `vllm/models/qwen4_exp/nvidia/model.py` and `hyperconnection.py`.
- `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`.
- `vllm/model_executor/layers/quantization/modelopt.py`.
- `vllm/model_executor/layers/fused_moe/experts/cutlass_moe.py`.
- `csrc/libtorch_stable/quantization/fp4/nvfp4_blockwise_moe_kernel.cu`.
- `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py`.

The last file explicitly disables automatic FlashInfer B12X selection for an
SM121 guard issue. TRTLLM/CuteDSL candidates target family100; CUTLASS candidates
admit family120. Source support does not prove Mia's actual selected MoE backend.

## Smaller lessons, correctly attributed

The matched32K chunk-only example is2133→2366 tok/s, about10.9%, not an
explanation for our roughly42 tok/s baseline. PLE-prefetch attribution is not an
isolated whole-model A/B; Atlas's earlier same-binary PLE-only experiment barely
changed TTFT. FP8 KV's main benefit there is capacity, not demonstrated faster
prefill. Draft vocabulary slicing and verify graph widths primarily affect
decode, and are not directly transferable to Atlas's existing head/layout.

Mia's MXFP8 patch falls back to BF16 for unsupported small projection geometry
and vision, rather than dequantizing every matrix. This is a useful loader
lesson, not permission to relabel Atlas's NVFP4 checkpoint as the same model.

F11 subsequently completed F8's native build and8/8 short outputs matching the
same-binary baseline, with48-layer receipts. Same-input FFN/full-state
qualification remains before uncached ABBA timings. That smoke result is not
a measured win or2K-prefill qualification.
