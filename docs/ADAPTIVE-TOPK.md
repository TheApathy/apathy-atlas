# Adaptive top-K — skipping negligible routed experts

**Status: implemented, default OFF, UNMEASURED.** Nothing in this document is a
result. The distribution the whole idea rests on has not been measured yet; §2
is the instrument that measures it and §5 is my prediction, which exists so it
can be falsified, not quoted.

**This is a quality-affecting change.** It deletes model computation. Read §6
before enabling it anywhere near an output anyone will read.

## 1. The idea, and the arithmetic that makes it interesting

From `docs/DECODE-WATERFALL-2026-08-10.md`, DeepSeek-V4-Flash-162B on one GB10:

| quantity | value |
|---|---:|
| plain graphed decode step | 45.3 ms = 21.9 tok/s |
| MoE expert stream | 4.02 GB/token = 20.9 ms (46% of the step) |
| achieved MoE expert-GEMV bandwidth | 192 GB/s |
| MoE layers | 43 |
| routed experts per token | top-6 (+1 always-on shared) |

Per routed expert, per layer (MXFP4-E8M0, per-32 scales, `moe_intermediate_size`
2048 × `hidden_size` 4096, three projections):

```
gate  2048×4096 → 4,194,304 B packed + 262,144 B scale = 4.46 MB
up    2048×4096                                        = 4.46 MB
down  4096×2048                                        = 4.46 MB
                                        per expert      13.37 MB
```

`13.37 MB × 6 × 43 = 3.45 GB` routed + `~0.58 GB` shared = **4.03 GB/token** —
the census closes on the waterfall's 4.02 GB, so the model below is sound.

**One routed slot dropped for every token, at every layer, is worth
575 MB/token = 2.99 ms/step = 45.3 → 42.3 ms = +1.7 tok/s.** Two slots:
+3.5 tok/s. That is the entire prize, and it scales exactly linearly in the mean
number of slots dropped. The shared expert is never a candidate.

Absolute ceiling, for calibration only: pruning to top-1 would leave
`43 × (13.37 + 13.4) MB = 1.15 GB/token`, a 15 ms saving → 33 tok/s. Nobody is
proposing that; it is here so the size of the whole lever is on the record.

## 2. Measure first — `ATLAS_MOE_GATE_HIST=1`

### What is logged, and what "gate weight" means on this model

DeepSeek-V4-Flash routes with **sqrtsoftplus + correction bias**
(`scoring_func: "sqrtsoftplus"`, `topk_method: "noaux_tc"`), not softmax and not
sigmoid. `kernels/gb10/common/moe_topk_sqrtsoftplus.cu`:

```
score[e]     = sqrt(log(1 + exp(logit[e])))       # NOT a probability
selection[e] = score[e] + correction_bias[e]      # bias steers SELECTION only
idx[0..k]    = argtopk(selection)                 # descending SELECTION order
w[t]         = score[idx[t]] / sum_j score[idx[j]] # norm_topk_prob = true on V4
w[t]        *= routed_scaling_factor              # V4: 1.5
```

So the weights the logger reads are **post-normalization and post-scaling**, and
on V4 they sum to exactly `routed_scaling_factor = 1.5`. The scale-free quantity
— what the logger reports and what the prune thresholds on — is the **gate-mass
fraction** `mass[t] = w[t] / sum_j w[j]`, which equals
`score[idx[t]] / sum_j score[idx[j]]` whether or not the router normalized.
Uniform routing is `1/6 = 0.1667`.

### The rank subtlety (this trips people)

Slots come out in descending **selection** order (`score + bias`). The correction
bias is per-expert, so **slot order is not weight order** — the last emitted slot
is not reliably the smallest weight. The logger reports both: `slot_mass` in
emitted order and `sorted_mass` descending by weight. The prune thresholds on the
*weight*, so `sorted_mass` is the column that predicts what gets dropped, and
`sorted_mass[-1]` is "rank 6" in the sense §1 cares about.

### Invocation

```bash
# Plain decode (GAMMA=-), eager so the logger is not skipped under graph capture.
scripts/dsflash-serve-bench.sh gatehist - \
  ATLAS_MOE_GATE_HIST=1 \
  ATLAS_MOE_GATE_HIST_PATH=/tmp/gate_hist.jsonl \
  ATLAS_MOE_GATE_HIST_MAX=400000 \
  ATLAS_PROFILE=1

# drive it: prose + code + repeat + quote, the same four workloads the decode
# probe uses, so the distribution belongs to the workloads we quote tok/s on
python3 scripts/decode_ab_probe.py gatehist 8977 1

python3 scripts/moe_gate_hist.py /tmp/gate_hist.jsonl            # aggregate
python3 scripts/moe_gate_hist.py /tmp/gate_hist.jsonl --by-layer # per layer
python3 scripts/moe_gate_hist.py /tmp/gate_hist.jsonl --kind hash
```

Knobs: `ATLAS_MOE_GATE_HIST_PATH` (default `moe_gate_hist.jsonl`),
`_MAX` (JSONL line cap, default 200000 — the rolling `tracing` summary keeps
counting past it), `_EVERY` (summary period in fires, default 4096).

It synchronizes and reads back once per MoE fire, so it is a **measurement mode**:
decode slows several-fold, and it is skipped entirely under CUDA-graph capture.
Run it eager.

Layers are identified by their router-gate device pointer, assigned ids in
first-seen order — which is model order, because layers fire in order. Hash-routed
layers (`num_hash_layers`) are tagged `"kind":"hash"`; their expert selection is a
static `tid2eid[token_id]` table, so they are a different population and the
analyzer defaults to `--kind gate`.

### The three numbers to come back with

1. Mean gate mass at weight-rank 5 (the smallest of six) and rank 4.
2. Fraction of **(layer, token)** pairs whose smallest slot is below
   0.02 / 0.05 / 0.10.
3. Fraction of all **droppable slots** (everything but the arg-max) below each
   threshold — this one converts directly to bytes: `frac × 6 × 575 MB/token`.

## 3. The implementation

`ATLAS_MOE_ADAPTIVE_TOPK=<threshold>`, a gate-mass fraction in `(0, 1)`.
**Unset = off = bit-identical to current behaviour, and not one extra kernel node
in the graph.** Out-of-range or unparseable values disarm loudly
(`tracing::error!`) rather than silently.

Flow, per MoE fire, all on device:

```
router (moe_topk_sqrtsoftplus)  →  indices[6], weights[6]
moe_adaptive_topk_prune         →  rewrites both IN PLACE
expert GEMVs (grid.y = top_k+1) →  pruned slots hit the NULL guard and return
moe_weighted_sum_blend          →  pruned slots carry weight 0
```

### Graph safety — the crux

The constraint is that a CUDA graph freezes launch geometry at capture, so a
data-dependent *expert count* is not available to us: it would change `grid.y`
and force re-capture per token. This design never changes the count.

* **`moe_adaptive_topk_prune`** (`kernels/gb10/common/moe_adaptive_topk.cu`)
  launches at a compile-time-constant `grid (1,1,1)` / `block (32,1,1)` and reads
  every input from device memory. No host readback, no dynamic parallelism, no
  data-dependent geometry — it captures and replays like any other node.
* **The expert GEMVs keep `grid.y = top_k + 1` exactly as today.** The expert
  *count* is unchanged; only *which slots do work* changes, and that is decided
  by a device-side load inside an already-launched block.
* **The skip is a NULL-pointer early-out, and it happens before any weight byte
  is read.** The prune writes the sentinel expert id `num_experts` into a pruned
  slot. Every expert pointer table now carries one extra, all-NULL entry at that
  index (`ptr_table_build::SENTINEL_SLOTS`), so the slot's block loads a NULL
  `B_packed` and takes the guard that already exists for EP-remote experts:

  ```cuda
  const unsigned int expert_id = expert_indices[expert_slot];
  B_packed = (const unsigned char*)gate_packed_t_ptrs[expert_id];
  ...
  if (B_packed == 0) { emit_zero(); return; }   // ← before the K loop
  ```

  `emit_zero()` writes the slot's `N` bf16 zeros and returns. The K loop — the
  only thing that streams the 13.37 MB — never runs. **That guard is where the
  bytes are saved; the prune kernel only decides.** This is the "keep the grid,
  make skipped slots no-op" option from the design brief, and it only pays
  because the early-out is ahead of the streaming loop rather than inside it.

The sentinel entry is added unconditionally, not under the env var. A table whose
sentinel exists only when a knob is set is a table that faults the first time
someone flips the knob on a path nobody re-tested.

### Which paths may arm (`helpers_b::adaptive_topk`)

The guard is verified present, ahead of any weight read, in
`moe_shared_expert_fused_t.cu` (`_t`, split-K and `_m` multi-row — they share
`gate_up_shared_t_impl` / `silu_down_shared_t_impl`) and in
`moe_shared_expert_fused.cu` / `moe_expert_gemv_fused.cu` (the non-transposed
NVFP4 fused decode pair). Those arm. The **EXL3, BF16-dequant, FP8 and W3** routed
paths are not verified and refuse to arm with an error line. That is a
correctness gate, not taste: without the guard the sentinel would be
dereferenced as a weight pointer.

### Renormalization: **yes on this model**, and exactly why

sigmoid/sqrtsoftplus routing is **not** softmax, so "the weights sum to 1" is not
a property of the scoring function and cannot be assumed. It is a property of the
router having divided by the sum of the **selected** set — which is what
`norm_topk_prob` controls, and which is `true` on DeepSeek-V4-Flash (the six
weights sum to `routed_scaling_factor = 1.5`).

Because the normalization is over the selected set, renormalizing over the
survivors reproduces *exactly what the router would have emitted had it selected
the smaller set*. That is the model's own semantics. Not renormalizing would
attenuate the entire routed branch by the dropped mass — at 5% dropped mass, in
43 layers, that is a systematic `0.95^43 ≈ 0.11` pull on the routed contribution
relative to the (unattenuated) shared expert and residual. Silent, compounding,
and nothing like "we skipped a small expert".

So the rule implemented is: **renormalize iff `norm_topk_prob`.** When the router
did *not* normalize, the weights are raw scores whose sum carries meaning and
rescaling would invent magnitude; there we drop and leave the rest alone. The
call site passes `ctx.config.norm_topk_prob` straight through; both branches are
unit-tested in `moe/mod_tests.rs`.

The arg-max slot is **never** dropped, whatever the threshold — a token always
reaches at least one routed expert.

### Known gap: the batched verify routing sites

The prune is wired at the single-row decode routing site (`moe/forward.rs`),
which serves plain decode **and** the per-row speculative-verify loop. The
batched `_m` verify paths (`forward_km`, `forward_kn`, `forward_k2`,
`forward_k3`, `forward_batched`) do their own routing and are **not** pruned:
the sentinel's interaction with those kernels' cross-row expert *dedup* is
unverified, and shipping an unverified path is worse than shipping none.
Prefill is likewise unpruned.

**Consequence: measure with plain decode (`GAMMA=-`) and do not quote a
spec-armed number.** A run where decode prunes and the batched verify does not is
measuring two different models against each other, and any acceptance-rate
change it shows is an artifact of that, not of the threshold.

## 4. Threshold sweep and decision rule

Sweep **0.01 / 0.02 / 0.05 / 0.10**, plain decode, one variable at a time:

```bash
for T in 0.01 0.02 0.05 0.10; do
  scripts/dsflash-serve-bench.sh atk$T - ATLAS_MOE_ADAPTIVE_TOPK=$T
  python3 scripts/decode_ab_probe.py atk$T 8977 2     # run twice; first is warmup
done
# baseline arm, same sitting, same binary:
scripts/dsflash-serve-bench.sh atk-off -
python3 scripts/decode_ab_probe.py atk-off 8977 2
```

### Decision rule

Accept a threshold only if **all** of these hold. Any single failure rejects it;
there is no "close enough" on a lever that deletes computation.

1. **It pays for itself.** Measured mean dropped slots per fire ≥ **0.34**
   (= 1 ms/step ≈ +0.5 tok/s). Below that the prune's own 43 launches/token are
   in the same order as the saving and the arm is noise. Reject without even
   running the quality gates.
2. **`tool-eval-bench` ≥ 90/100.** Standing gate; the stack currently holds 97,
   so this is a ≤7-point budget, not a fresh threshold to negotiate.
   `bench/laguna/full_eval.sh`-shape invocation: all 69 scenarios,
   `--error-rate 0 --seed 1234`, `PASS+FAIL+PARTIAL == 69` or the score is a
   partial run and unquotable.
3. **Frost verbatim, temp 0.** Prompt: *"Quote the first stanza of 'Stopping by
   Woods on a Snowy Evening' by Robert Frost."* Must come back exactly:

   > Whose woods these are I think I know.
   > His house is in the village though;
   > He will not see me stopping here
   > To watch his woods fill up with snow.

   This is the test that caught the NVFP4 lm_head fabricating recall when prose,
   code and repeat all looked fine (waterfall §4, item 4b). Verbatim recall is
   the first capability a byte-cutting change destroys and the last one a fluency
   read will notice. **Any misquote is an immediate reject**, regardless of every
   other number.
4. **`decode_ab_probe.py` hashes.** Two checks, and they are not the obvious one:
   * **Determinism:** two runs of the *same* arm must produce *identical* hashes
     on all four workloads. A differing hash is a non-determinism bug in the
     prune, not a quality cost — stop and fix it.
   * **Non-inertness:** the arm's hashes must *differ* from the `atk-off`
     baseline whenever the measured drop rate is > 0. Identical hashes with a
     claimed tok/s gain means the feature did not fire and the gain is
     measurement noise.

   Text divergence from baseline is *expected* and is not itself a failure —
   routing changed, so the text changes. The hashes exist to prove the arm is
   deterministic and live, not to prove the text is unchanged. Quality is judged
   by 2 and 3.
5. **Warm/cold consistency.** Because prefill routes unpruned and decode routes
   pruned, the same token gets different MoE output depending on which pass
   produced it. Run `bench/warm_cold_chat_diff.py` on the winning threshold; a
   warm-hit divergence that the baseline does not show is a reject, and it is the
   symptom that this asymmetry needs closing before ship.
6. **Report the honest tok/s.** Quote graphed steady-state (`ATLAS_LOOP_TRACE=1`),
   not an eager `ATLAS_PROFILE=1` run — see the memory note "Plain decode graphs
   were the artifact". A 256-token bench can silently run eager.

If two thresholds pass, take the **smaller** one. The byte saving is linear in
drop rate but the quality risk is not, and the tail of the mass distribution is
where the experts that matter for rare, precise tokens live.

## 5. Expected saving per threshold — a PREDICTION, not a result

**The real numbers come from §2. These exist so §2 can prove them wrong.**

My prediction is that this lever is **weaker on DeepSeek-V4 than the softmax
intuition behind the idea suggests**, for two structural reasons:

* **sqrtsoftplus is strongly compressive.** `sqrt(log(1+e^x))` squashes logit
  spread twice over. Two experts with logits 3.0 and 1.0 give scores 1.746 and
  1.146 — a **1.52×** weight ratio. Softmax on the same logits gives `e² =`
  **7.39×**. A routing function that cannot produce a 10× spread cannot produce a
  6th expert carrying 2% of the mass.
* **`noaux_tc` exists to flatten load.** The correction bias promotes under-used
  experts into the top-6. The slot it promotes is one whose *pre-bias score* is
  mid-pack by construction — precisely the opposite of a negligible expert.

Predicted mean gate mass by weight-rank (top-6, sums to 1.0; uniform = 0.167):

| rank | r0 | r1 | r2 | r3 | r4 | r5 |
|---|---:|---:|---:|---:|---:|---:|
| predicted mean mass | 0.25 | 0.19 | 0.17 | 0.16 | 0.14 | 0.12 |

which implies:

| threshold | predicted mean slots dropped /fire | GB/token saved | ms/step | tok/s (from 21.9) |
|---|---:|---:|---:|---:|
| 0.01 | ~0.00 | ~0.00 | ~0.00 | 21.9 (+0.0) |
| 0.02 | ~0.01 | 0.00 | 0.02 | 21.9 (+0.0) |
| 0.05 | ~0.03 | 0.02 | 0.09 | 22.0 (+0.1) |
| 0.10 | ~0.40 | 0.23 | 1.20 | 22.5 (+0.6) |
| *(reference)* 1 slot always | 1.00 | 0.575 | 2.99 | 23.6 (+1.7) |
| *(reference)* 2 slots always | 2.00 | 1.15 | 5.98 | 25.4 (+3.5) |

Read against the brief's hypothesis — "a 15–20% byte cut, ~3–4 ms/step,
~2 tok/s" — that requires a mean of **1.0–1.4 slots dropped**, which on this
predicted distribution needs a threshold **above 0.10**. At 0.10 you are already
deleting an expert carrying a tenth of the gate mass, which is not negligible by
any reading of the word. **If the prediction holds, the byte saving and the
quality cost are the same knob and there is no favourable setting.** If §2 shows
r5 near 0.03, the prediction is wrong, the lever is real, and the sweep decides
where.

Netted out of every row: the prune's own cost, 43 extra launches/token at ~2 µs
eager ≈ 0.09 ms/step, ≈ 0 graphed. At thresholds 0.01 and 0.02 that is larger
than the saving — those rows are net-negative eager.

## 6. Files

| file | role |
|---|---|
| `kernels/gb10/common/moe_adaptive_topk.cu` | the prune kernel |
| `crates/spark-model/src/layers/moe/gate_hist.rs` | `ATLAS_MOE_GATE_HIST` logger |
| `crates/spark-model/src/layers/moe/helpers_b.rs` | `adaptive_topk()` — knob parse + path gate |
| `crates/spark-model/src/layers/moe/forward.rs` | dispatch (prune, then log) |
| `crates/spark-model/src/layers/moe/ptr_table_build.rs` | `SENTINEL_SLOTS` |
| `crates/spark-model/src/layers/ops/moe_gate.rs` | launch wrapper |
| `crates/spark-model/src/layers/moe/mod_tests.rs` | host reference + renormalization rules |
| `scripts/moe_gate_hist.py` | histogram analyzer |
