# REST draft store — Phase 2 integration plan

Phase 1 built the store, the lookup library, and a CPU-only eval harness.
Nothing is wired into the scheduler. This document says exactly where it
would plug in, and — because the Phase 1 numbers came back the way they
did — what has to be true before it is worth doing.

## 1. What Phase 1 measured

Replaying 1096 real Qwen3.8 generations (`regen_qwen38.jsonl`,
777 338 decode steps) against a 16.9 M-token store built from
`<repo-root>` + `qwen38/benchmark`:

| min_match | engagement | accepted lookahead / engaged step | / decode step |
|---|---|---|---|
| 4 | 12.35 % | 0.63 | 0.077 |
| 6 | 2.88 % | 0.95 | 0.027 |
| 8 | 0.87 % | 1.25 | 0.011 |

**That corpus is worthless for that workload.** At the default gate the
store engages on fewer than 1 % of steps and, when it does, half the
engagements accept nothing. The generations are Java/Python answers to
Magicoder-Evol prompts; the corpus is the Atlas Rust tree. There is no
verbatim overlap to retrieve.

Rebuilding the store from an *in-domain* corpus (932 generations from the
same distribution, evaluated on 157 held-out generations with near-duplicate
rows removed — 8 of the original 165 held-out rows were exact duplicates of
corpus rows and were inflating every number by roughly 2×) changes the
verdict completely:

| min_match | engagement | accepted lookahead / engaged step | / decode step |
|---|---|---|---|
| 4 | 26.02 % | 2.02 | 0.524 |
| 6 | 11.40 % | 3.05 | 0.348 |
| 8 | 5.82 % | 4.40 | 0.256 |
| 10 | 3.64 % | 5.51 | 0.201 |
| 12 | 2.57 % | 6.38 | 0.164 |

Lookup latency is a non-issue at either corpus size: p50 3.5–10 µs,
p99 20–65 µs, against a decode step of roughly 65 ms. The store costs
nothing to consult.

**Two results drive the design below.**

*Domain match is the entire lever.* Same code, same gate, same tokenizer:
0.011 vs 0.256 accepted tokens per decode step, a 23× swing, purely from
what went into the corpus. Any Phase 2 that indexes a generic code dump
will measure as noise.

*Branching is nearly worthless here.* Across every node budget the
frequency-weighted tree beats its own single highest-count chain by only
3–7 %:

| max_nodes = depth | tree | spine | gap |
|---|---|---|---|
| 2 | 1.038 | 1.020 | 1.8 % |
| 4 | 1.630 | 1.576 | 3.4 % |
| 8 | 2.320 | 2.202 | 5.4 % |
| 16 | 3.050 | 2.840 | 7.4 % |

Retrieved continuations that match at all tend to match as one long
unbranched span. Value scales with *depth*, not with width.

## 2. What that means for the integration

The obvious design — REST proposes a `DDTreePayload` and rides the
tree-verify path — is the wrong one, on the evidence.

A tree payload buys 3–7 % over a flat chain, and costs: the `ATLAS_DDTREE_UNCAP`
budget trap (`DflashDraftBudget::new`, `crates/spark-model/src/layers/dflash_head/draft_budget.rs:29-46`
— by default `tree_nodes == flat`, so branch builders place zero nodes and
DDTree measures as an exact no-op), the `dispatch_verify` narrowing
(`scheduler/mtp_step.rs:88-91` drops the payload for any non-`Generic`
dispatch), the thinking-mode drop (`mtp_step.rs:182-184`), the
content-policy drop (`mtp_step.rs:190-212`), the EP drop (`mtp_step.rs:216-218`),
and `retain_prefix` converting any grammar-truncated frame back to flat
(`scheduler/proposal_lifecycle.rs:193-200`).

**Ship REST as a flat draft chain.** It gives up under a tenth of the
value and skips every one of those gates.

### Where the chain goes

`ActiveSeq.pending_drafts` (`scheduler/types.rs:197`), installed through
the pairing authority in `scheduler/proposal_lifecycle.rs` so drafts and
tree stay consistent — never by direct assignment. The existing ngram path
(`scheduler/spec_step.rs:208,294,327`) *does* assign `a.pending_drafts`
directly; that is legal only because it can never produce a tree, and it
should not be copied.

### The call site

`scheduler/mtp_step.rs:350-351`, the Phase A bootstrap propose:

```rust
let _mtp_grammar_mask = mtp_grammar_mask_for(a);
match propose_and_install(model, a, tok, num_drafts, _mtp_grammar_mask.as_deref()) {
```

REST is not a replacement for that call. **It must not displace the DFlash
proposal.** DFlash on this target runs around p ≈ 0.90 per-token acceptance;
REST at its best operating point delivers 2.0 accepted tokens on 26 % of
steps. Swapping a REST chain in for a DFlash chain is a straight loss on
74 % of steps and a coin flip on the rest.

So the integration is **conditional pre-emption, gated hard**:

1. In the Phase A bootstrap, before `propose_and_install`, call
   `crate::rest_store::propose(&ctx)` where `ctx` is the sequence's recent
   token history (`a.seq` tokens + `a.output_tokens`, last `max_k`).
2. Take the REST chain **only** when its match length clears a gate set
   well above the eval's break-even — the `min_match ≥ 10` rows, where
   wasted engagements fall to 18 % — *and* the resulting spine is at least
   `num_drafts` long, so it fills the same verify width DFlash would have.
3. Otherwise fall through to `propose_and_install` unchanged.

Skipping the DFlash forward pass on a REST hit is where the win actually
comes from: it removes the drafter's GPU work from those steps entirely,
on top of the accepted tokens. At 3.6 % engagement that is a small win, and
it is the *only* win — which is why §4's gate matters more than the code.

### Types and functions involved

| Concern | Item |
|---|---|
| Draft slot | `ActiveSeq.pending_drafts` — `scheduler/types.rs:197` |
| Tree slot (left `None`) | `ActiveSeq.pending_tree_payload` — `scheduler/types.rs:204` |
| Install | `proposal_lifecycle::install` / `install_collected` — `scheduler/proposal_lifecycle.rs:78,103` |
| Frame accessor | `SchedulerProposalFrame::parts` for `ActiveSeq` — `scheduler/proposal_lifecycle.rs:58` |
| Propose entry | `step_mtp` — `scheduler/mtp_step.rs:121` |
| Verify | `dispatch_verify` → `step_verify_dflash`; a flat chain of width ≥ 4 routes `Generic` — `scheduler/mtp_step.rs:20,75` |
| Grammar truncation | `truncate_drafts_at_grammar_boundary` — `scheduler/spec_step.rs:463` |
| REST lookup | `crate::rest_store::propose` — this module |

A REST chain needs no new capability from the model: it is `Vec<u32>`
into `pending_drafts`, exactly the shape MTP already produces. No
`DraftProposer` impl, no `physical_verify_k` negotiation
(`spark-model/src/speculative.rs:245`), no `is_dflash` claim.

### If the tree is wanted later

`DraftTree::to_payload_parts` already emits `(tree_token_ids, parent_indices)`
satisfying `DflashDraftBudget::validate_tree` — every parent is `-1` or
strictly less than its child's index, because the trie's best-first prune
admits parents before children. `crate::rest_store::validate_tree_shape`
asserts it at the producer. Wiring it would mean setting
`pending_tree_payload` via `install`, and setting `ATLAS_DDTREE_UNCAP=1`
or the budget silently drops every branch. Worth 3–7 %; do it last, if ever.

## 3. Startup wiring

- `main.rs`: add `mod rest_store;` next to `mod ngram;` (line 63). Phase 1
  declares the module in `lib.rs` instead, because a `mod` with no consumer
  trips `deny(warnings)` in the binary crate.
- `main_modules/serve_phases/preflight.rs`: call
  `rest_store::init(&tokenizer_json_bytes)` after the tokenizer loads. It
  validates the store's tokenizer fingerprint against the tokenizer the
  server actually loaded and returns `Ok(false)` when `ATLAS_REST_STORE` is
  unset. A fingerprint mismatch must be fatal at startup, not silently
  ignored — token ids from another tokenizer are not lower-quality drafts,
  they are noise.
- No pool-geometry change. A flat REST chain occupies verify slots that
  `checked_ssm_speculative_geometry`
  (`spark-model/src/model/ssm_pool_geometry.rs:90`) already sized for
  `num_drafts`.

## 4. The gate on doing Phase 2 at all

Phase 2 is worth writing **only** if a corpus can be assembled that stands
in the same relation to production traffic as the in-domain store did to
its held-out split. Concretely, before any scheduler code is written:

1. Build a store from the actual target domain — the repository being
   served, or a corpus of the target's own recent generations.
2. Run `rest-store-eval` against held-out real traffic, **decontaminated**.
   The 2× inflation from 8 duplicate rows in 165 is exactly how this
   measurement lies if nobody checks.
3. Require ≥ 0.25 accepted tokens per decode step at `min_match ≥ 10`.
   Below that, REST cannot pay for the DFlash steps it pre-empts.

The `--jsonl` / `--files` eval takes seconds and needs no GPU. Run it
before writing any scheduler code, not after.

## 5. Second tier: self-context retrieval

A static store can only draft text that resembles something indexed
offline. §1 measured what that costs on the wrong corpus: 0.011 accepted
tokens per decode step. The tier added alongside it indexes nothing —
its corpus is the sequence itself, prompt plus everything generated so
far — so it needs no file, no network, and no prior sight of the prompt.

Structure: an incremental suffix automaton (`rest_store::sam`) per
sequence, in `ActiveSeq.self_context`. Appending is O(1) amortized and
the longest-suffix match is maintained as a side effect, so the per-step
cost does not grow with history. It replaces the O(n·k)-per-step rescan
in `spark-server::ngram`, which now delegates to it rather than carrying
a second implementation of the same query.

### The gate is set by the drafter, not by usefulness

This is the result that decides the whole tier. Pre-empting DFlash means
giving up a frame worth `p(1-p^γ)/(1-p)` ≈ **7.15 accepted tokens** at
γ=15, p≈0.90. A retrieval chain has to beat that number, not zero.

Replaying 120 real AEON generations (63,710 decode steps; the 31
harness-error rows in `aeon_dedup.jsonl` excluded — a repeated error
template is trivially self-similar and fakes a good result):

| min_match | engagement | accepted/engaged | accepted/decode step | verdict |
|---|---|---|---|---|
| 6 | 13.52 % | 4.64 | 0.627 | loses to the drafter |
| 8 | 8.83 % | 5.54 | 0.489 | loses |
| 10 | 6.23 % | 6.41 | 0.399 | loses |
| 12 | 4.59 % | 7.26 | 0.333 | break-even |
| 16 | 2.86 % | 8.77 | 0.250 | **wins by 1.6 tok/engagement** |
| 20 | 2.06 % | 9.71 | 0.200 | wins, at half the engagement |

Hence `ATLAS_SELF_CONTEXT_MIN_MATCH` defaults to 16. Note that the
accepted-tokens-per-decode-step column *falls* as the gate rises while
the verdict improves — that column is the right measure for a tier that
adds drafts, and the wrong one for a tier that replaces them.

A code workload agrees: 1,096 regenerated Magicoder answers, 777k decode
steps, 9.40 accepted per engaged step at min_match 16.

### Value concentrates late in a generation

Within each AEON generation, comparing the first quarter of the output
against the last, at min_match 12: 5.71 → 7.64 accepted per engaged step,
and 106 → 1,370 engagements. Self-similarity is something a generation
accumulates, which is the argument for the tier on long outputs. The
available corpora cannot test that directly — the longest generation in
`aeon_dedup.jsonl` is ~3.5k tokens and `regen_qwen38.jsonl` is truncated
at 768 — so the position breakdown is the evidence, not the length
breakdown.

### Eval

`rest-store-selfctx --tokenizer <tokenizer.json> --jsonl <generations>`,
CPU-only, seconds to run. There is no holdout and none is needed: at step
`t` the drafter can only see `tokens[..t]`, which the target had already
emitted before producing `tokens[t]`, so there is nothing a partition
could hold out.

### The saturation problem, and the gate that answers it

Retrieval engages on repetitive text. So does the neural drafter — the
engine has been observed logging `accepted=15/15 (100%)` on stretches of
repetitive parser code. Pre-empting a frame like that costs the
difference, and the offline eval cannot see it, because it has no
drafter. It can only show how much the answer matters. On the three long
generations at min_match 16 (14.42 % engagement, 9.69 accepted/engaged):

| assumed drafter acceptance at the SAME positions | net per engagement | tok/step |
|---|---|---|
| 7.15 — its unconditional mean | +2.54 | +4.5 % |
| 11.07 — elevated on repetitive text | -1.39 | -2.5 % |
| 15.00 — saturated | -5.31 | -9.4 % |

The sign of the whole feature lives in that column, so the scheduler
does not bet on it. `ActiveSeq.last_verify_accepted` records what the
drafter actually achieved on its most recent γ-block for that sequence,
and `retrieval_chain` skips pre-emption when it is at or above the
tier's own measured yield (`ATLAS_SELF_CONTEXT_MAX_DRAFTER_ACCEPT`,
default 10; `ATLAS_REST_MAX_DRAFTER_ACCEPT`, default 13). Retrieval
therefore fires only where the drafter has just demonstrated it was
leaving tokens on the table.

Value concentrates in long matches, which is where the gate should be
tuned if it is retuned: at min_match 16 on the long generations, the
32-64 band carries 31.7 % of engagements at 11.04 accepted (57.5 % of
them filling the whole frame), while the gate..20 band carries 20.4 % at
7.91.
