# Next attention lane: audited plan, not implemented

2026-09-05, source fingerprint
`f589a434108efd39bf69495a82c2f405617c181bdd85d9b80120f8a3b469a037`.
Root and prefill_recipe_audit read-only review. No F14 GPU/native/source claim.

## Why the legacy batch flag cannot be promoted

1. `kernels/gb10/common/qwen4_qsa.cu:128` stages all current tokens into only four
   ring rows per physical page. Tokens0 and4 in one page target the same row;
   all groups subsequently pool that overwritten ring. Separate launches do
   not recover overwritten current-chunk keys. This is a concrete write race
   for pages wider than four tokens, not a measured F12 regression: F12 retains
   the scalar QSA update, not this legacy batch flag.
2. `qwen3_attention/prefill/cache_skip.rs:230` and `paged.rs:259` omit the
   `qwen4_yarn_inv_freq`/`rope_yarn_scaled` branch present in scalar
   `decode/attention_forward.rs:391`.
3. `qwen3_attention/trait_impl/prefill_inner.rs:141` updates QSA but invokes
   generic dense attention without using the selected sparse indices.

QSA selects512 compressed groups of four tokens. Scalar sparse execution starts
at position2048; ranking becomes necessary beyond512 completed groups. This is
not a512-token dense window. Future continuation can expose bad history even
when a short dense-window answer looks correct.

## Bounded implementation sequence

Keep F12/F13 production paths separate. First add exact synthetic QSA-history
tests that fail the old staging: multiple groups in one page, permuted physical
pages, start modulo4, partial groups, split chunks and next-token continuation.
Correct pooling reads current-chunk raw keys directly, consulting old ring only
for an incomplete leading group; update the ring deterministically afterward.
Preserve the scalar key projection/reduction and raw/pooled normalization.

Then add a new default-off dense-window helper before the F8 attention row loop
in `qwen3_attention/trait_impl/prefill_moe_only.rs`. Require canonical
H2560/Q24/KV2/HD256, eager C1, BF16 KV, start0 and M2..2048, no prefix reuse or
alternate metadata. Validate both HC objects, all handles, extents and aliases
before any effects. QG width12288, K/V each512, attention output6144.

Use explicit original-layout `w4a16_gemm` for QG/K/V/O, ordinary per-head
`rms_norm`, batched YaRN, KV append, `inferspark_prefill_64`, sigmoid gate and
saved-HC injection. No generic FP8 fallback or extra weight copy. Do not assume
the existing fused normalization has the scalar reduction order. QSA key-only
`dense_gemv_batchn` is the exact-reference candidate; `dense_gemm_bf16` is not
automatically a tensor-core implementation merely because of its name.

Capture and compare Q/K/V projections, normalized/rotated Q/K, KV bytes, QSA
ring/compressed keys, attention result and injected hidden on identical inputs.
Keep explicit numerical gates separate from byte-exact state gates. Model
continuation must cover2048/2049/2051/2052, resets and chunk boundaries before
same-binary cache-zero timing. Preserve the old path when off.

Full long-context performance requires a new multi-row sparse attention path
that actually consumes each row's selections, not merely raising the guard.

## Performance limit and upstream comparison

F12's provisional profile still observes substantial MoE and HC cost. Attention
batching alone cannot establish2,000tok/s; reprofile the changed path before
choosing the next kernel. Do not treat incomplete trace totals as hard bounds.

The upstream Mia README, refreshed on September5, reports2,265tok/s at16K/32K
with2,048-token chunks. It is not a2,048-token prompt measurement, and uses its
own checkpoint, FP8 KV and vLLM recipe. Its larger-chunk and PLE-prefetch results
are useful mechanisms, not local Atlas qualification. Benchmark matched
context/quantization/cache/output settings before calling the result reproduced.
Source: https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark
