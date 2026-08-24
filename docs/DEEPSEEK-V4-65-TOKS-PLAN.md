# DeepSeek-V4 Flash: path to 65 tok/s on one Spark

This branch targets 65 generated tokens per second without treating the Qwen
result as directly transferable. Qwen3.8's largest gains came from its hybrid
GDN/attention layout; DeepSeek-V4 is a 43-layer MLA/MoE model and needs a
DeepSeek-specific target path plus useful speculative acceptance.

## Evidence and current gap

| Runtime/profile | Single-stream decode |
| --- | ---: |
| Atlas plain EXL3 K2 | 22.90-22.98 tok/s |
| Atlas warm residency profile | 24.73-28.40 tok/s |
| Mia/vLLM K2 profile | about 65-75 tok/s, workload-dependent |

The Atlas warm result is not a quality-neutral baseline: its lossy NVFP4
attention-residency path changed three of four output hashes. The older Atlas
speculative measurement also used raw gamma 5, which means four proposals in
Atlas, despite the checkpoint containing a five-token DSpark block. Commit
`be97db83` corrects the launch contract to five proposals / six verify rows.

The Mia result establishes that the model/checkpoint can cross 65 tok/s on
Spark-class hardware. It does not establish that DFlash2 weights exist for
DeepSeek or that Qwen DFlash2 weights can be reused. Mia serves the embedded
DeepSeek DSpark/MTP path with its own vLLM kernel stack.

## Immediate Atlas lane

The default K2 speculative launcher now enables the portable, already-present
DeepSeek fast paths:

- fused V4 decode;
- compile-time verify GEMV;
- MoE multi-row partitioning;
- adaptive speculation with low-gear fallback;
- DSpark capture and the checkpoint-native five proposal rows.

These are the relevant pieces of the Qwen performance recipe. Qwen's GDN
kernels are not applicable to DeepSeek MLA and should not be copied into this
branch.

The next controlled GPU run should first establish a byte-locked plain K2
baseline, then measure the corrected DSpark launch with the same prompt,
generation length, clocks, server build, and sampling settings. Record target
decode time, draft time, verify time, average accepted tokens, and fallback
rate. A headline tok/s result without those counters cannot distinguish a
target-kernel problem from a low-acceptance drafter.

## DeepSeek-native DFlash2 checkpoint ABI

Atlas's block-diffusion DFlash head currently shares the target embedding and
LM head. A compatible checkpoint therefore needs all of the following:

- target hidden size `4096`;
- target vocabulary size `129280` and the same tokenizer/token IDs;
- capture IDs using HF hidden-state indexing in the range `1..=43`;
- an `fc.weight` trained for
  `4096 * number_of_capture_layers` input columns;
- a valid mask token inside the 129280-token vocabulary;
- a declared trained block/verify width and matching attention semantics;
- DeepSeek-trained weights. Qwen3.6 DFlash2 weights are structurally invalid.

The factory now rejects incompatible pairings before model scratch or KV-cache
construction. Cross-vocabulary `d2t` remapping is intentionally rejected
because Atlas detects such tables but does not yet execute the remap.

For a future 16-row drafter, the DeepSeek verify path also needs a measured
multi-row kernel ladder beyond the current small-row arms. Widening the kernels
without parity hashes and dispatch proof is not a performance result.

## Promotion gates

1. Plain baseline is reproducible across at least three warm trials.
2. Speculation uses the trained proposal count and reports nonzero dispatch.
3. Greedy outputs match the locked plain baseline when the path claims exactness.
4. Each approximate/quantized arm gets a separate quality score; hashes alone
   do not promote it.
5. The speculative arm beats plain on median decode tok/s and does not rely on
   adaptive fallback for the reported headline.
6. The 65 tok/s claim names prompt length, output length, concurrency, context
   residency, quantization, and acceptance rate.

Upstream implementation reference:
<https://github.com/tpurtell/ds4-mia-exl3-k2-1spark>
