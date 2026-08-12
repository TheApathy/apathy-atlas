# Self-speculation (early-exit drafting) on DeepSeek-V4-Flash — NO-GO

Analysis of 2026-08-12. **No new GPU runs.** Every number below is quoted from
a named in-tree doc, log, kernel census, or constant, or is arithmetic on
those. Companion to `docs/SPEC-3X-PLAN.md` (the tok/s = committed / step_time
frame) and `docs/DECODE-WATERFALL-2026-08-10.md` (the byte census and the
verify-step budget).

**Verdict up front: NO-GO at every early-exit depth, on every workload
regime, under every favorable assumption tested.** The decisive number is a
single line:

> The existing DSpark drafter produces a draft token for **3.15 ms**.
> Early-exiting the target after **one single layer** — 1 of 43 — costs
> **3.26 ms**. Early-exit drafting is more expensive per draft token than the
> whole 10.12 GiB drafter it was proposed to replace, before it has executed
> a second layer or predicted anything useful.

Nothing downstream of that line can be rescued by acceptance, because
acceptance is bounded above by 100% and the arithmetic below shows that even
100% loses at most depths.

---

## 0. Survey — what `use_self_speculative` already does, honestly

**It is fully built, it is not early-exit drafting, and it is a provable
no-op on this model.**

The in-tree mechanism is **layer-*type* skipping, not layer-*depth*
truncation**. `TransformerModel::decode_draft`
(`crates/spark-model/src/model/impl_b1.rs:487-589`) runs the full eager layer
loop and skips exactly one thing:

```rust
for (i, layer) in self.layers.iter().enumerate() {
    if self.config.layer_type(i) == LayerType::LinearAttention {
        continue; // Skip SSM layers
    }
    layer.decode(...)?;
}
```

It was written for **hybrid SSM/attention** models (Qwen3-Next, Nemotron-H
shapes), where dropping the Mamba2/GDN layers removes real work *and* leaves
the recurrent state untouched, which is why the caller can rewind with a bare
`seq_len -= 1` and why `checkpoint_ssm_states` / `rollback_ssm_states` bracket
the verify.

DeepSeek-V4-Flash has **zero** `LinearAttention` layers:

| evidence | value |
|---|---|
| `kernels/gb10/deepseek-v4-flash/MODEL.toml:8-9` | `layers_total = 43`, `layers_full_attention = 43` |
| `crates/atlas-core/src/config/parsers/deepseek_v4.rs:138` | `config.layer_types = vec![LayerType::FullAttention; config.num_hidden_layers]` |
| ⇒ `ModelConfig::num_ssm_layers()` (`config/methods.rs:51`) | **0** |

So on V4-Flash `decode_draft` skips **nothing**. It is a full 43-layer eager
forward. The startup banner even prints the truth and nobody reads it:
`impl_a1.rs:259-267` logs *"Self-speculative decoding: ENABLED (skipping **0**
SSM layers, keeping **43** attention layers)"*.

**The capability predicate is dishonest.** `has_self_speculative_dispatch()`
(`crates/spark-model/src/model/trait_impl/speculative.rs:81-83`) returns the
raw `self.self_speculative` flag with no model check, so
`serve.rs:611`'s `args.self_speculative && scheduler_model.has_self_speculative()`
arms the path for a model where it cannot work. See §5 for the one-line fix
this analysis shipped.

What the flag delivers today on V4-Flash, arithmetically (γd = 4 drafts,
verify-side cost held at today's measured value):

| regime | propose = 4 × full forward | T_step | **ceiling at 100% acceptance** | plain decode |
|---|---:|---:|---:|---:|
| matched-protocol code | 181.2 ms | 367.3 ms | **13.6 tok/s** | 21.9 |
| repeat | 181.2 ms | 275.6 ms | **18.1 tok/s** | 21.9 |

`--self-speculative` on this model is strictly worse than plain decode at
*any* acceptance rate, including a perfect oracle drafter.

`use_ngram_speculative` is a genuinely different and genuinely working thing:
a CPU 4-gram proposer (`NgramProposer::new(4)`, `mod.rs:227`) driving a
CUDA-graphed **K=2** verify (`spec_step.rs:176-367`). Zero weights, zero GPU
propose cost. Not early exit; not in scope here, but see §6 — it is the only
zero-weight drafter whose propose cost the arithmetic actually permits.

**Is early-exit drafting partially built? No.** Nothing in the tree taps an
intermediate residual, no head is trained or wired for a depth-N hidden, and
the only "skip" primitive keys on `LayerType`, not depth. Building it would
be from scratch. §3 explains why the mHC residual makes even the tap point
non-obvious.

---

## 1. The cost model (all inputs measured, all sources named)

| symbol | value | source |
|---|---:|---|
| full forward bytes | **6.674 GB/token** | `DECODE-WATERFALL` §1 census: MoE 4.02 + MLA 2.125 + lm_head 0.5295 |
| plain graphed step | **45.3 ms** = 21.9 tok/s | `DECODE-WATERFALL` §1 (LOOP_TRACE + `decode_ab_probe`, uniform across workloads) |
| `lm_head.fp8` M=1 | **2.26 ms** (529.5 MB @ 235 GB/s) | `DECODE-WATERFALL` §2 kernel census |
| ⇒ per-layer amortized cost `t_layer` | **(45.3 − 2.26)/43 = 1.001 ms** | derived |
| ⇒ per-layer bytes | **0.1429 GB** | derived |
| DSpark propose, γd=4 | **12.6 ms total = 3.15 ms/draft** | `SPEC-3X-PLAN` §1 step table (post grouped-GEMV; the 63–74 ms in `docs/kernels/00-index.md` is the superseded γ=2-era figure) |
| today's spec, matched protocol | **3.46 committed / 198.7 ms = 17.41 tok/s**, accept 61.6% | `SPEC-3X-PLAN` matched-protocol addendum |
| today's spec, repeat | **3.59 committed / 107 ms = 33.55 tok/s**, accept 64.8% | `SPEC-3X-PLAN` §1 |

Early-exit draft token cost at depth N:

```
bytes(N)  = N × 0.1429 + 0.5295 GB          (the head does NOT shrink with N)
t_draft(N)= N × 1.001  + 2.26   ms
T_step(N) = verify_side + γd × t_draft(N)   (drafting is autoregressive: γd sequential passes)
```

`verify_side` = today's step minus today's propose: **186.1 ms** (matched
protocol) / **94.4 ms** (repeat). Holding it fixed is the correct A/B — this
analysis swaps only the proposer.

### Assumptions, all chosen in early exit's favor

1. `t_layer` is the **graphed** plain-decode rate. Today's `decode_draft` is
   eager (`graph_capture: false`, `impl_b1.rs:565`), which would be ~1.21
   ms/layer. Early exit gets the graphed number for free here.
2. The draft's KV writes, block-table plumbing, per-draft launch overhead and
   host round-trip are charged at **zero**.
3. The MoE union savings that make deep γ cheap on the *verify* side
   (`SPEC-3X-PLAN` §2) do **not** apply to drafting: each draft pass is m=1
   and reads its own top-6 experts per layer, serially.
4. Committed tokens follow the standard geometric chain
   `committed = 1 + Σ_{i=1..γd} p^i`, which is optimistic (it assumes
   per-position independence; measured chains die faster).

---

## 2. THE ARITHMETIC TABLE

γd = 4 drafts/step (today's CLI γ=5). `need_p` = the per-token draft
acceptance required merely to **match today's throughput** — not to beat it.
The rightmost columns give the tok/s early exit would actually deliver at
fixed acceptance levels; **p = 0.62 is today's measured acceptance**, and
p = 1.00 is a perfect oracle.

### 2a. Matched-protocol code generation — baseline 3.46 / 198.7 ms = 17.41 tok/s

| N | draft bytes | % of full fwd | ms/draft | propose | T_step | need committed | **need_p** | p=0.62 | p=0.75 | p=0.85 | p=0.95 | **p=1.00** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1.673 GB | 25.1% | 10.27 | 41.1 | 227.2 | 3.96 | **0.883** | 10.5 | 13.4 | 16.3 | 19.9 | **22.0** |
| 12 | 2.244 GB | 33.6% | 14.27 | 57.1 | 243.2 | 4.23 | **0.917** | 9.8 | 12.5 | 15.3 | 18.6 | **20.6** |
| 16 | 2.816 GB | 42.2% | 18.27 | 73.1 | 259.2 | 4.51 | **0.949** | 9.2 | 11.8 | 14.3 | 17.5 | **19.3** |
| 20 | 3.388 GB | 50.8% | 22.28 | 89.1 | 275.2 | 4.79 | **0.979** | 8.7 | 11.1 | 13.5 | 16.4 | **18.2** |

### 2b. Repeat / favorable content — baseline 3.59 / 107 ms = 33.55 tok/s

| N | ms/draft | propose | T_step | need committed | **need_p** | p=0.62 | p=0.75 | p=0.85 | p=0.95 | **p=1.00** |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 10.27 | 41.1 | 135.5 | 4.55 | **0.952** | 17.6 | 22.5 | 27.4 | 33.4 | **36.9** |
| 12 | 14.27 | 57.1 | 151.5 | 5.08 | **IMPOSSIBLE** (>5 max) | 15.8 | 20.1 | 24.5 | 29.9 | 33.0 |
| 16 | 18.27 | 73.1 | 167.5 | 5.62 | **IMPOSSIBLE** | 14.3 | 18.2 | 22.1 | 27.0 | 29.9 |
| 20 | 22.28 | 89.1 | 183.5 | 6.16 | **IMPOSSIBLE** | 13.0 | 16.6 | 20.2 | 24.7 | 27.2 |

For N ≥ 12 on repeat, the maximum achievable committed count at γd = 4 is 5,
and break-even needs 5.08–6.16. **A perfect oracle drafter loses.**

### 2c. The most favorable configuration conceivable

Stack every optimization anyone could propose: N = 8 (only 18.6% of the
model), an NVFP4 draft head (265 MB → 1.13 ms; legitimate here because a
draft head is acceptance-gated, not quality-gated — `SPEC-3X-PLAN` R6), *and*
the free first draft (the verify pass already computes layer-N hiddens for
its accepted row, so draft #1 costs only its head).

| regime | propose | T_step | need committed | **need_p** |
|---|---:|---:|---:|---:|
| matched code | 28.5 ms | 214.6 ms | 3.74 | **0.854** |
| repeat | 28.5 ms | 122.9 ms | 4.12 | **0.904** |

**85–90% per-token acceptance from an untrained 8-layer tap, just to break
even.** Our fully-trained, purpose-built, 10 GiB DSpark drafter — which sees
the target's layer-40/41/42 hiddens — achieves 55–65%.

### 2d. The bound that ends the discussion: the whole propose budget

| regime | today | if propose cost were **exactly zero** | the entire propose phase is worth |
|---|---:|---:|---:|
| matched code | 17.41 | 18.59 | **+1.18 tok/s** |
| repeat | 33.55 | 38.03 | **+4.48 tok/s** |

Propose is 6.3% (matched) / 11.8% (repeat) of the step. Early-exit drafting
at N=8 costs **41.1 ms — 3.3× the entire propose budget** — and at N=20 it
costs 89.1 ms, **7.1×**. There is no acceptance number that buys back a
proposer which is a multiple of the phase it lives in, when the current
proposer already occupies less than an eighth of the step.

---

## 3. Why it fails *here* specifically — three structural reasons

This is not a general refutation of early-exit drafting. LayerSkip-style
self-speculation is credible on dense models. Three properties of *this*
model invert the economics:

**(1) The vocab head costs 3.71 layers.** `lm_head` is 0.5295 GB; a layer is
0.1429 GB. The head is **7.9%** of a forward while a layer is **2.14%**.
Contrast a dense LLaMA-2-13B, where the head is ~1.3% of the forward and a
layer ~2.5% — a ratio of 0.5 layers, 7× smaller. The reason is sparsity: an
MoE layer reads only 6 of 256 experts, so *layers are cheap and the head is
not*. Truncating a body whose layers are already cheap, while paying a fixed
head tax per draft token, is the wrong trade. This is what produces the N=1
floor of 3.26 ms — a floor that is *already above* the incumbent drafter.

**(2) Drafting is autoregressive; the incumbent is not.** DSpark drafts all γ
tokens in **one** block pass (bidirectional in-block attention over γ+1 rows,
`dspark_head.rs:1-17`), so it pays its lm_head **once per step**. Early exit
must run N layers *and* a head **γ times**, serially. The head tax multiplies
by γ. This asymmetry alone is ~7 ms/step at γd=4.

**(3) There is no residual stream to tap.** V4-Flash uses manifold-constrained
hyper-connections (`hc_mult = 4`). The inter-layer state is `[T, 4, H]`
streams, not a vector; `hc_expand` opens them at layer 0 and `hc_head` — *"the
final collapse before the LM head: a single learned sigmoid-weighted sum"*
(`layers/ops/hyper_connection.rs:253-256`) — closes them at layer 42 only.
`hc_head`'s weights are trained for depth-43 streams. Tapping at layer N means
either (a) running `hc_head` out of distribution, or (b) the `hc_mean` plain
average that DSpark uses for its capture — which is trained-for at layers
40/41/42 and nowhere else. **Every existing head in the tree expects a
full-depth hidden**: the target `lm_head`, the V4 MTP head (whose body is
itself a *complete extra V4 layer* with MLA + mHC + 256-expert MoE, so it is
not a cheap projection — `deepseek_v4_mtp.rs:1-30`), and the DSpark head
(`main_proj` trained on layer-40/41/42 `h.mean(dim=2)`). None can serve a
depth-N tap without training, and training is out of scope.

Two further mechanism obstacles, recorded so nobody rediscovers them:

- **KV sharing.** A depth-N draft writes KV for layers 0..N−1 at the draft
  position; the verify pass then rewrites the same slots from a *batched*
  (m = γ+1) kernel. Those are not bit-identical to the m=1 draft writes —
  the measured batched-vs-plain hidden gap is 2–3% cosine at identical tokens
  (memory: *DSpark verify numerics gap*). Under **the exact-GEMV law**
  (*partial exactness is worse than none*, `verify-exactness.md`,
  `DECODE-WATERFALL` §6 closing constraint) a half-shared cache is the worst
  configuration available.
- **FP8-KV + compressed pool.** Draft-time writes perturb the FP8-KV
  calibration window (`fp8_kv_calibration_tokens = 256`) and the V4 compressed-KV
  pool, whose verify/decode asymmetry is an already-diagnosed acceptance bug
  (memory: *DSpark verify compressor asymmetry*).

---

## 4. What about the memory payoff?

The premise is correct: **the drafter and the fast-prefill BF16 mirrors do not
both fit.**

| resident item | size | source |
|---|---:|---|
| DSpark 0731 drafter (256 experts) | **10.12 GiB** | `REF-DRAFTER-ANALYSIS` §3, from safetensors headers |
| V4 attention BF16 mirrors (needed by the cuBLASLt prefill path, 856 → 1062 tok/s) | **~8.06 GiB** | `ATLAS_V4_ATTN_RELEASE_BF16` A/B; `prefill/v4_fp8_proj.rs:37-43` |

An early-exit drafter is genuinely **0 GB of new weights** (it reuses the
target's layers and head). That payoff is real. It is also irrelevant,
because §2 shows the mechanism loses throughput — buying 10 GiB by giving up
the speculative multiplier is not a trade anyone wants.

**The memory conflict has a cheaper resolution that is already in-tree.**
`ATLAS_DSPARK_REF_DRAFT=1` + `ATLAS_DSPARK_DRAFT_EXPERTS=64` resolves a
64-expert REAP subset at load time, freeing **7.18 GiB** of the drafter with
no second checkpoint on disk (`weight_loader/deepseek_v4/dspark_reap.rs`;
`REF-DRAFTER-ANALYSIS` §4). 2.94 GiB + 8.06 GiB fits where 10.12 + 8.06 does
not.

**Honest caveat that must not be lost:** the subset needs a REAP ranking
artifact (`REAP_K216_PLAN.json`), which exists only for the reference tp1
checkpoint. The `DeepSeek-V4-Flash-0731` drafter that `scripts/dspark-serve.sh`
actually points `--draft-model` at carries no such plan, and
`load_dspark_drafter` **hard-errors** rather than guessing a subset (guessing
would load real weights under wrong expert ids and draft silently wrong
tokens). So realizing the 7.18 GiB requires switching `--draft-model` to the
reference tp1 `mtp.*` shards. That is a config change and an A/B, not a
research project — and it is the correct next move for the memory problem
that motivated this investigation.

---

## 5. What this analysis shipped

One honesty fix, no new mechanism. `has_self_speculative_dispatch()` now
requires the model to actually have SSM layers to skip, so
`--self-speculative` on a pure-attention model degrades to plain decode
(21.9 tok/s) instead of arming a path whose *perfect-acceptance* ceiling is
13.6–18.1 tok/s. This mirrors the existing `has_recurrent_state()` SSOT idiom
in `config/methods.rs:62-70`: a capability predicate must be derived from the
config, never from the request.

Files: `crates/spark-model/src/model/trait_impl/speculative.rs`,
`crates/spark-model/src/cli`-adjacent banner in
`crates/spark-server/src/main_modules/serve.rs` (unchanged — it already logs
the fallback correctly).

---

## 6. NO-GO — what would have to change for this to become viable

Each item below is necessary; the first two are jointly necessary and still
not obviously sufficient.

1. **Train the model for early exit.** Layer-dropout + early-exit loss
   (LayerSkip-class continual pretraining) is the only known way to make a
   depth-N tap agree with depth-43 at the 85–90% the arithmetic demands. This
   is a pretraining-budget item on a 284B checkpoint, not an inference change.
   Without it, an untrained tap at N=8 will not reach even 62%.
2. **Make the vocab head cheap.** The head must fall from 3.71 layers to well
   under 1 for a truncated body to matter. That means a *trained* low-rank
   draft head — exactly the shape the DSpark drafter already ships
   (`drafter lm: [129280, 256] = 66.2 MB, 273 µs` in the `DECODE-WATERFALL`
   §2 census, **8× cheaper than the target head**). Note what this implies:
   the thing that would make early exit affordable is *a trained drafter
   head*, i.e. the very component early exit was proposed to eliminate.
3. **Make drafting non-autoregressive.** The γ-fold head tax dies only if the
   γ drafts are produced in one pass — which is the DSpark/DFlash block
   formulation, not early exit.
4. **Hardware:** a machine where the drafter and the BF16 mirrors both fit
   removes the *motivation* entirely (payoff (a) evaporates above ~145 GB of
   usable VRAM). Nothing about a faster GPU helps payoff (b) — the ratios in
   §2 are dimensionless, so early exit loses identically on any device.
5. **Model:** a dense (non-MoE) target, or one without mHC, restores the
   textbook economics. On V4-Flash, sparsity is precisely what breaks it.

### Where the multiplier actually is (unchanged by this analysis)

`SPEC-3X-PLAN` §3 already ranks it: **draft acceptance is worth +19 tok/s;
all plain-decode kernel work is worth ~+6.** Early-exit drafting attacks
neither — it attacks propose cost, a phase worth **+1.2 to +4.5 tok/s in
total** (§2d) and already down to 12.6 ms. The open experiments that do move
acceptance are R2(a) (finish the exact-GEMV chain: head-gate
`dense_gemv_batchm` + batched attention/rope exactness) and R2(b)
(probabilistic draft sampling). The one open experiment that moves *memory*
is the 64-expert REAP draft in §4.

---

## 7. Reproduction

```bash
# The arithmetic in §2 is pure derivation from the anchors in §1. To re-derive:
#   t_layer   = (45.3 - 2.26) / 43                    # DECODE-WATERFALL §1, §2
#   t_draft   = N * t_layer + t_head
#   T_step    = (T_today - propose_today) + gamma_d * t_draft
#   need_p    : solve  1 + sum_{i=1..gamma_d} p^i  =  rate_today * T_step
#
# To re-measure the anchors instead of quoting them:
scripts/dsflash-serve-bench.sh <name> 5 ... ATLAS_LOOP_TRACE=1 ATLAS_STEP_TIMING2=1
ATLAS_TARGET_MODEL=deepseek-v4-flash cargo run --release -p spark-model \
  --example decode_gemv_audit --features cuda,gpu-examples

# To confirm the survey claim (V4-Flash has zero skippable layers) without a GPU:
grep -n 'layer_types' crates/atlas-core/src/config/parsers/deepseek_v4.rs   # => vec![FullAttention; 43]
```
