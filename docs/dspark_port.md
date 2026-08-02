# DSpark drafter port — design (2026-08-02)

## Why (measured)

Atlas serves DeepSeek-V4-Flash-162B at **21.0 tok/s plain and 21.0 tok/s with
K=2 MTP speculation** — speculation contributes nothing. Marginal cost of the
second verify row ≈ 27 ms (MoE expert streaming 17.9 ms dominates; attn per-row
~3; draft fwd ~5); at 63% accept, 1.63 tok/step × 0.61 efficiency = 1.00×.
K=3 recurrent MTP is *worse* (16.5 tok/s): depth-2 conditional accept is 37%
because the nextn=1 head was never trained on its own output.

The ds4/ds4-on-spark reference (the "DSpark 27.7 suite mean" we're chasing)
says the same in its own docs: "upstream MTP is single-token, net-negative
single-stream." Its entire win is **tok/step ≈ 2.6–4.0 from the DSpark block
drafter** — its per-verify-row marginal cost (~15 ms) is the same ballpark as
ours. Beat the drafter, beat the number.

## The DSpark drafter (authoritative: official 0731 `inference/model.py`)

Artifacts on disk:
- Weights: `/home/flocka/models/DeepSeek-V4-Flash-0731-drafter/model-000{46,47,48}-of-00048.safetensors`
  (~10.9 GB, pure `mtp.*`, verified headers; from `deepseek-ai/DeepSeek-V4-Flash-0731`).
- Reference impl: same dir, `inference-model.py` (classes `DSparkAttention`,
  `DSparkBlock`, `DSparkMarkovHead`, `DSparkConfidenceHead`), `inference-generate.py`,
  `config-0731.json`.
- ds4 C implementation (secondary reference): `/tmp/ds4-src` (re-clone
  `Entrpi/ds4` if gone).

Config: `dspark_block_size=5`, `dspark_target_layer_ids=[40,41,42]`,
`dspark_markov_rank=256`, `dspark_noise_token_id=128799`, drafter
`sliding_window=128`. 3 stages (`mtp.0/1/2`), each a full V4 layer:
MLA attention (wq_a [1024,4096], wq_b [32768,1024], wkv [512,4096],
wo_a [8192,4096] grouped, wo_b [4096,8192], q_norm/kv_norm, attn_sink[64]),
mHC (hc_attn/hc_ffn fn [24,16384] F32, hc_mult=4), MoE **256 experts**
(w1/w3 [2048, 2048 packed] + w2 [4096, 1024 packed] I8 = native MXFP4 nibbles,
E8M0 scales per-32 — Atlas's `_e8m0` kernels run this natively) + shared expert
(FP8 E4M3). Extras: `mtp.0.main_proj` [4096,12288] FP8+E8M0-scale,
`mtp.0.main_norm` [4096] BF16, `mtp.2.norm`, `mtp.2.hc_head_{fn,base,scale}`,
`mtp.2.markov_head.markov_w{1,2}` [129280,256] BF16,
`mtp.2.confidence_head.proj` [1,4352] BF16. Embed + lm_head are **shared with
the target** (not in the shards).

### Capture (target side)

At target layers 40/41/42 (0-indexed, of 43), after the layer completes:
`capture_i = mean over hc_mult streams of hc_streams` → one [4096] BF16/F32
vector per layer. `main_hidden = concat(capture_40, capture_41, capture_42)`
[12288]. Needed at every position (prefill) and at the last committed
position (decode/verify — capture verify rows too: accepted rows provide the
multi-accept catchup for free).

### Drafter state

Per stage: a **128-entry ring buffer of `main_kv`** rows only, where
`main_kv = kv_norm(wkv(main_x))` (+rope on last 64 dims, FP8-quantized latent
— same MLA cache layout as the target, 576/entry). `main_x =
main_norm(main_proj(main_hidden))` [4096], the same vector fed to all 3
stages. **Draft rows are never written to the cache** — rejected drafts can't
pollute it. Prefill: target prefill's captured hiddens → main_x per position →
ring rows (batch). Decode: one ring write per committed position (ring index
`pos % 128`).

### Propose forward (per step, after target commits token t at pos)

1. `main_x = main_norm(main_proj(concat captures))`; write `main_kv` per stage
   at ring[pos % 128].
2. Block = 5 rows: `ids = [t, noise, noise, noise, noise]`, `x =
   embed(ids)` replicated to hc_mult streams. Positions: block row i at
   `pos + i` (rope).
3. For each stage 0..2: full V4 layer forward on the 5 rows, EXCEPT
   attention = queries from the 5 rows over [128-window ring ∥ 5 block-row
   KV] with **no causal mask** (all 5 see all 5), attn_sink, softmax scale,
   MLA inverse-rope on output, grouped wo_a einsum + wo_b. mHC mixing +
   256-expert MoE as normal (5-row batchN path).
4. Head (stage 2): hc_head collapse → rms_norm(mtp.2.norm) → **target's FP8
   lm_head** → logits [5, 129280].
5. Sequential Markov chain (greedy): `out[0]=t`; for i in 0..4:
   `logits[i] += markov_w2 @ markov_w1[out[i]]`; `out[i+1] = argmax(logits[i])`;
   confidence[i] = sigmoid(confidence_proj · [hidden_row_i | markov_w1[out[i]]])
   (proj computed in F32). Truncate the chain at the first
   confidence < 0.9 (`DS4` default; make it an env knob).
6. Emit up to 5 drafts → existing K=γ verify (`step_verify_dflash`, dispatch
   floor already ≥4 drafts).

### Verify + controller

Verify = existing γ-block path (1 committed + up-to-5 draft rows). Acceptance:
longest matching prefix (existing). Quench controller (from ds4, port later):
regret `guard − tok/step` accumulated per request, guard ≈ 2.10 plain-steps,
terminally disable speculation for the request when deficit > ~4 plain-steps.

## Port plan (tasks #10–#14)

1. **Loader** (`weight_loader/deepseek_v4/dspark.rs`): detect
   `mtp.0.main_proj.weight` in the drafter store; drafter ModelConfig = target
   config clone with `num_experts=256`; build 3 stage weight-sets (reuse the
   `assemble_layer`/MoE-building sub-paths with prefixes `mtp.{0,1,2}`) + the
   extras. Ride the `--dflash --draft-model <dir>` plumbing
   (`load_dflash_drafter`), branching on store contents.
2. **Capture**: 3 tiny hc-mean kernels per step at layers 40/41/42, fixed
   buffers (graph-friendly); capture in decode, prefill (batched), and the
   γ-verify forward (accepted rows = catchup).
3. **Propose forward**: new module `layers/dspark_head/`. The MoE + mHC of
   each stage reuse existing 5-row batched paths; the windowed bidirectional
   attention is the only genuinely new kernel work (5 queries × ≤133 KV — tiny;
   one warp per (head,row) is plenty).
4. **Verify/cache/quench wiring** in the scheduler.
5. **Memory**: drafter ≈ 10.2 GB on-GPU. Free the ~8.6 GB BF16 prefill
   fallbacks and/or trim KV pool; measure.
6. **Bench**: decode_bench + 9-workload suite vs 21.0 baseline; gates: smoke,
   quality_probe, longgen (0 regressions). Risk: drafter trained against
   0731 non-REAP hiddens, our target is REAP-162B — acceptance may sag;
   measure per-workload accept before optimizing further.

## Measured: offline acceptance (2026-08-02)

The make-or-break risk is resolved. `bench/deepseek-v4/dspark_probe/`
replays ATLAS_DSPARK_DUMP captures (hc-mean at layers 40/41/42, greedy token
stream from the Atlas server on the REAP-162B target) through the OFFICIAL
reference implementation with the target's shared embed/lm_head. On 634
propose points across 3 workloads (math steps / python code / networking
explanation):

    draft[0..4] match: 72.6 / 65.3 / 57.0 / 49.4 / 38.9 %
    chain hist (0..5 accepted): [174, 74, 77, 61, 65, 183]
    mean accepted chain 2.50  →  tok/step 3.50 (ungated)
    confidence@0.9: tok/step 2.12 — under-keeps 319/634; recalibrate
    against OUR verify cost model, not ds4's (chain profile is bimodal:
    27% accept zero, 29% accept all five → adaptive depth pays)

Compare 1.63 tok/step for the shipping K=2 MTP path. The drafter, trained
against non-REAP 0731 hiddens, transfers to the REAP target at full
strength.

## Expected outcome

Reference law: `speedup = tok/step ÷ ~2.05` per DSpark verify step (theirs).
Our verify rows are cheaper than 2.05×-plain would imply
(1+5 rows ≈ 1 + 5×0.56 ≈ 3.8 plain-steps eager today — the MROW=2-style dedup
generalized to 6 rows and the batched attention bring this toward ~2.3). At
their measured accept (52–91%/workload, tok/step up to 4.0), even 2.5×
step-cost yields suite-mean ≈ 26–30 tok/s on our 21.0 base — at parity or
above the reference's 27.7, with headroom from kernel work they don't have
(NVFP4 target, split-K GEMV, MROW dedup).
