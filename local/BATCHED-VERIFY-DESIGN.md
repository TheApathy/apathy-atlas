# Cross-Seq Batched DFlash Verify — design sketch (2026-07-08)

THE LEVER: DFlash concurrency currently verifies PER SEQUENCE (step_mtp loops; c=8 = eight K=17
forwards = eight full 14GB weight reads per round; per-seq tok/s collapses 76.8→~26 at c=8 — the
serialization signature). ONE batched M=17c verify reads the weights ONCE for all c sequences.
Projected aggregate: 208 → 400+ at c=8. This is Phase-D's core and the multi-user endgame.

PREREQUISITES ALREADY LANDED:
- GEMM dispatch for 32<M<=256: ATLAS_FFN_KGAMMA_WIDE (committed 2026-07-08) → w4a16_gemm_t_m128.
- Per-seq SSM state indirection at fixed addresses: ssm_multi_seq_ptr_scratch + PAD_SLOT_ID
  (piecewise-graph work, validated lossless c=1..8 incl. mid-batch compaction).
- Attention per-seq metadata refresh pattern: decode_a2_piecewise.rs (fixed scratch, re-upload eager).
- Chunked-WY handles arbitrary K windows (K-flex fix validated); wy17 stays per-seq K=17 inside the
  batched step (SSM layers iterate seqs within one kernel launch family — see multi_seq variants).

DESIGN (v1, eager attention like piecewise):
1. Scheduler: new step_verify_dflash_batched(active: &mut [ActiveSeq]) — gather each seq's
   verify_input_tokens (bonus + drafts, all K=17), build one [c*17] token batch. Per-seq propose stays
   as-is initially (propose is small; echo/async kills it later).
2. Model: decode_verify_dflash_batched — ONE forward over M=c*17 rows:
   - Embed c*17 rows; FFN/norm/lm_head batched (M=136 @ c=8 → the WIDE window).
   - Attention: per-seq paged decode with per-row seq metadata (each row's kv = its seq's blocks +
     its position among the 17) — reuse the multi_seq attn metadata layout; EAGER first (no graphs).
   - SSM/GDN: per-seq wy17 over its 17 rows with slot-indirected h/conv state (the multi_seq ptr
     table); the 48 layers loop c sub-batches per layer OR use a c-batched wy17 wrapper (kernel
     already fills 48 SMs at c=1 — batching across seqs in one launch is the perf follow-up, v1 can
     loop inside the layer WITHOUT re-reading FFN weights, which is where the 14GB lives).
   KEY INSIGHT: the WEIGHT-heavy parts (FFN 9.6GB + attn/qkv projections + lm_head) batch trivially
   (pure GEMM M growth); only attention-KV and SSM-state are per-seq — and both already have
   indirection machinery. The 14GB read amortizes even if SSM loops per-seq inside the step.
3. Accept/commit: per-seq accept loops unchanged (each seq's 17-row logit slice → its own
   num_accepted/bonus/rollback/commit). commit_verify_state_async per seq as today.
4. Gates: token-exactness per seq vs single-stream md5 (e3a39829 counting), incl. mixed max_tokens
   compaction case; then aggregate curve. Env ATLAS_DFLASH_BATCHED_VERIFY=1 default OFF.

RISKS: logits buffer sizing ([c*17, vocab]); per-row RoPE positions (each seq's own seq_len+t);
graph capture deferred (eager v1 — the weight amortization dwarfs launch overhead at M=136);
scheduler fairness (all seqs must be in verify phase together — align step loop like step_mtp does).

COST ESTIMATE: v1 ~2-4 agent-days. VALUE: ~2x aggregate on top of 2.71x → ~5x total vs single-stream;
plus it composes with batched-propose later. After echo + this, the box serves ~400 tok/s multi-user
at 95% HumanEval quality.
