# The reference stack's compact 64-expert DSpark draft

Addendum to `docs/SPEC-3X-PLAN.md` §"Drafter swap". Resolves the apparent
contradiction between the reference recipe's `draft_experts: 64` and the
216-expert `mtp.*` tensors in the checkpoint we already load, and records what
landed in Atlas as a result.

Sources, all on this box:

| What | Path |
|---|---|
| Their serve recipe | `$SPARKINFER_REF/config/recipe.json` |
| Their draft builder | `$SPARKINFER_REF/scripts/build_dspark_draft.py` |
| Their entrypoint | `$SPARKINFER_REF/scripts/entrypoint.sh` |
| Their vLLM patch | `$SPARKINFER_REF/patches/vllm-dspark-compact-draft.patch` |
| Target checkpoint | `$SPARKINFER_REF/data/tp1/` |
| REAP provenance map | `$SPARKINFER_REF/data/tp1/REAP_K216_PLAN.json` |
| Their built draft | `$SPARKINFER_REF/data/dspark-draft-k64/` |

## 1. How the 64 experts are selected — the crux

**Not at runtime, and not "the first 64".** The reference `entrypoint.sh`
runs `build_dspark_draft.py` once before the server starts, which repacks the
checkpoint's `mtp.*` tensors into a separate 2.94 GiB directory. The vLLM
patch does nothing but let the draft blocks *construct* with the draft
config's `n_routed_experts` — it is not the selector.

The selector is this, over `REAP_K216_PLAN.json`:

1. `keep_maps.mtp_keep` is the ascending list of **original** (0..255) expert
   ids that REAP kept. Its *position* is the **current** (0..215) id the
   checkpoint tensors are actually named for. This is the id space that
   matters; everything else in the plan is expressed in original ids.
2. Walk the structured-output calibration categories **in order** —
   `agentic_tool_trajectory`, then `tool_calling` — taking the first
   `structured_experts_per_category` (32) entries of
   `keep_maps.structured_ranked_by_category[cat]["42"]`. `"42"` is
   `keep_maps.mtp_keep_from_layer`: the observation dataset covers the 43 main
   routed-MoE layers, so the MTP stages borrow layer 42's ranking. Skip ids
   REAP pruned, and skip ids already chosen.
3. **Top up from the global REAP ranking** `keep_maps.mtp_ranked` until the
   budget of 64 is full, same availability and dedup rules.
4. Map the chosen original ids back to current ids, **sort ascending**, and
   renumber densely 0..63. The sort is what lets a row-slice of
   `ffn.gate.{weight,bias}` line up with the renumbered experts.

### The trap

Step 2 does **not** produce 64. The two structured rankings overlap heavily,
so on the shipped plan only **39** distinct experts survive it — the other
**25 come from the global REAP fill**. Any implementation that reads
`draft_experts: 64` + `structured_experts_per_category: 32` and concludes
"32 × 2 categories" picks a different expert set, loads real weights under
wrong ids, and drafts quietly-wrong tokens. This is the single fact most
likely to be got wrong on a reimplementation.

### Verification

`crates/spark-model/src/weight_loader/deepseek_v4/dspark_reap.rs` ports the
algorithm and reproduces the reference's own
`DSPARK_DRAFT_PLAN.json:selected_current_expert_ids` **bit-exactly** — all 64
ids and the 39/25 split — locked by
`dspark_reap::tests::matches_shipped_reference_selection`. The shard the
reference built from that selection hashes to
`d72dd9d92abe2cfd2d90931072ae3b920a8f0be09465a88c839072a16d7e5cd5`, matching
their published `sha256`.

The 64 checkpoint (current) ids:

```
6 7 8 12 16 17 20 22 23 26 30 32 35 47 53 59 62 67 68 69 72 82 87 90 96 97
101 103 105 114 119 122 123 126 128 129 137 138 139 141 146 147 149 153 154
155 158 161 163 172 178 181 184 185 186 188 192 196 198 200 202 205 206 208
```

## 2. Architecture comparison — it is the SAME drafter

Tensor schema of their compact draft vs the checkpoint's full `mtp.*`, from
the safetensors headers of `carried-00{3,4,5}.safetensors` and their
`dspark-draft-k64/model.safetensors.index.json`:

| Tensor family | full `mtp.*` | their k64 draft | differs? |
|---|---:|---:|---|
| `attn.{wq_a,wq_b,wkv,wo_a,wo_b}.{weight,scale}` | 3 ea. | 3 ea. | no |
| `attn.{q_norm,kv_norm}.weight`, `attn.attn_sink` | 3 ea. | 3 ea. | no |
| `{attn_norm,ffn_norm}.weight` | 3 ea. | 3 ea. | no |
| `hc_{attn,ffn}_{fn,scale,base}` | 3 ea. | 3 ea. | no |
| `hc_head_{fn,scale,base}` (stage 2) | 1 ea. | 1 ea. | no |
| `ffn.shared_experts.w{1,2,3}.{weight,scale}` | 3 ea. | 3 ea. | no |
| `ffn.gate.{weight,bias}` | 3 ea. `[216,·]` | 3 ea. `[64,·]` | **row-sliced** |
| `ffn.experts.{E}.w{1,2,3}.{weight,scale}` | 648 ea. (216×3) | 192 ea. (64×3) | **subset** |
| `main_proj.{weight,scale}`, `main_norm.weight` | 1 ea. | 1 ea. | no |
| `markov_head.markov_w{1,2}.weight` | 1 ea. | 1 ea. | no |
| `confidence_head.proj.weight`, `norm.weight` | 1 ea. | 1 ea. | no |
| **total tensors** | 3985 | 1249 | |

Their draft `config.json` is a byte-copy of the target `config.json` with
**exactly one key changed**: `n_routed_experts: 216 → 64`.

Conclusions that follow:

- **Same stage count** (3), same `dspark_block_size: 5`,
  `dspark_target_layer_ids: [40,41,42]`, `dspark_markov_rank: 256`,
  `sliding_window: 128` — identical to `DsparkParams::V4_FLASH_0731`.
- **Same MLA attention weights.** `draft_attention_backend: B12X_MLA_SPARSE`
  is a *kernel* selection, not a different weight set. The proof is negative
  and decisive: the target's layers carry `attn.compressor.*` (41 layers) and
  `attn.indexer.*` (21 layers) — the DSA sparse-attention machinery — and the
  `mtp.*` stages carry **neither**. There are no draft indexer weights to
  load, so the draft cannot be doing learned sparse selection; `B12X_MLA_SPARSE`
  is the short-context/windowed MLA path, which is what our `dspark_head`
  already does over its 128-entry `main_kv` ring.
- **Same expert format** (I8-packed MXFP4 `.weight` + `F8_E8M0` `.scale`), so
  our existing E8M0 expert kernels serve it unchanged.
- The `mtp.*` stages carry **no `ffn.gate.tid2eid`** (the target's 3 hash-routed
  layers do), consistent with our loader already taking the learned-gate path
  for the drafter.

**There is no structural difference. The compact draft is our drafter with a
smaller `n_routed_experts`.**

## 3. Memory arithmetic

Measured from safetensors headers, not estimated:

| | bytes | GiB |
|---|---:|---:|
| Full `mtp.*` (216 experts, reference ckpt) | 9,257,533,500 | **8.62** |
| — of which routed experts (3888 tensors) | 8,663,334,912 | 8.07 |
| — everything else (attn, mHC, shared, heads) | 594,198,588 | 0.55 |
| One expert across all 3 stages | 40,108,032 | 0.037 |
| Compact 64-expert draft (their shard) | 3,157,375,260 | **2.94** |
| **Freed by the subset** | **6,100,158,240** | **5.68** |

(Our predicted 64-expert size from the per-expert cost is 3,161,112,636 B; the
3,737,376 B difference is exactly the 3 × 152 dropped gate rows at
4096·2 + 4 bytes each — the arithmetic closes to the byte.)

The unpruned 0731 drafter (256 experts) is 594,198,588 + 256 × 40,108,032 =
10,861,854,780 B = **10.12 GiB**, matching the ~10.68 GB figure we carry for
the current drafter. Subsetting *that* to 64 frees **7.18 GiB**.

Atlas takes the saving **without a second checkpoint on disk**: the subset is
resolved at load time and unselected expert tensors are never read off disk,
never counted in the OOM pre-flight, and never allocated. The reference pays
2.94 GiB of extra disk and a one-off repack pass; we pay neither.

## 4. What landed

Default **OFF**. `ATLAS_DSPARK_REF_DRAFT=1` turns it on.

| File | Change |
|---|---|
| `crates/spark-model/src/weight_loader/deepseek_v4/dspark_reap.rs` | **new** — the selection algorithm + 6 tests incl. the bit-exact golden lock |
| `.../deepseek_v4/dspark.rs` | `ref_draft_enabled`, `resolve_ref_draft_subset`, `compact_draft_skip_fn`; `load_dspark_drafter` takes the subset; `DsparkDrafterModule.num_experts` |
| `.../deepseek_v4/assemble.rs` | `assemble_moe_subset` — expert-id indirection, router row-gather, correction-bias gather. `assemble_moe` delegates with `None` (target layers unchanged) |
| `crates/spark-runtime/src/weights.rs` | `SafetensorsLoader::extra_skip` (`TensorSkipFn`), ORed into `should_skip_tensor` |
| `crates/spark-server/.../serve_phases/weights.rs` | resolves the subset from the drafter dir, installs the skip filter |
| `crates/spark-model/src/factory{,/build}.rs` | `DflashBuildArgs::dspark_expert_subset`; **bug fix** below |
| `examples/dspark_{loader_smoke,engine_probe}.rs` | honour the same gate, so the offline oracle can A/B without the server |

Knobs, mirroring theirs:

| Atlas | reference | default |
|---|---|---|
| `ATLAS_DSPARK_REF_DRAFT` | (implied by `MODE=dspark`) | `0` (off) |
| `ATLAS_DSPARK_DRAFT_EXPERTS` | `DSPARK_DRAFT_EXPERTS` | 64 |
| `ATLAS_DSPARK_STRUCTURED_PER_CATEGORY` | `DSPARK_STRUCTURED_EXPERTS_PER_CATEGORY` | 32 |

The expert count flows from the data at every step — the gate's row count, or
the resolved subset's length — and is never a literal.

### Bug fixed on the way

`factory/build.rs` passed a hardcoded `256` as `drafter_num_experts` to
`DsparkDraftHead::new`, while `load_dspark_drafter` correctly derived the true
count from the gate shape. On the reference 216-expert checkpoint the proposer
was therefore configured for 256 experts against a 216-expert MoE. It now
reads `module.num_experts`. **This is live on the default path** — worth
re-baselining plain DSpark acceptance against it independently of the compact
draft, since it may move the current 55-62% number on its own.

### Which checkpoints the flag applies to

`REAP_K216_PLAN.json` is a 0xSero REAP artifact of the **reference tp1
checkpoint**. The official `DeepSeek-V4-Flash-0731` drafter shards
(`$DRAFT_MODEL_DIR`, 256 experts, what
`scripts/dspark-serve.sh` points `--draft-model` at today) do **not** carry
one, and the gate is a hard error there — guessing a subset would load real
weights under wrong ids and draft silently wrong tokens, so the loader refuses
rather than falling back.

Copying the plan next to a different drafter does not work either, and is
explicitly rejected: the selected ids are *positions in that plan's kept set*.
Every reference id is < 216 and would therefore "fit" a 256-expert drafter
while naming completely different experts — a bounds check cannot catch it.
`load_dspark_drafter` compares `DraftExpertSubset::source_experts` against the
drafter's actual gate rows and refuses on mismatch. Subsetting the unpruned
256-expert drafter would need its own ranking artifact, which does not exist.

## 5. Acceptance-measurement protocol

The claim under test is that the reference's ~75%-implied acceptance comes
from the draft *weights*, not the draft *size*. The compact draft has strictly
fewer experts than ours, so the honest prior is **neutral-to-slightly-worse
acceptance for meaningfully less memory** — the win, if any, is that 5.68 GiB
buys KV depth or drafter headroom elsewhere. Measure it, don't assume it.

Hold everything constant except the drafter: same binary, same target weights,
same prompt, same γ, `temperature 0`, same `max_tokens`. Two runs.

```bash
# A — baseline: full-expert drafter (today's default)
ATLAS_DSPARK_CAPTURE=1 ATLAS_DSPARK_ACCEPT_LOG=1 \
  ./scripts/dspark-serve.sh 2>&1 | tee /tmp/accept-full.log

# B — compact draft: identical invocation + the gate
ATLAS_DSPARK_CAPTURE=1 ATLAS_DSPARK_ACCEPT_LOG=1 \
ATLAS_DSPARK_REF_DRAFT=1 \
  ./scripts/dspark-serve.sh 2>&1 | tee /tmp/accept-k64.log
```

Against each server, the matched-protocol workload from
`docs/SPEC-3X-PLAN.md` — the same 512-token code-generation prompt at
`temperature: 0`, `max_tokens: 512`, concurrency 1, discarding the first
request so CUDA-graph capture and FP8-KV calibration do not land in the
measurement (see `plain-decode-graphs-were-the-artifact` — short benches
silently run eager).

Read off, per run:

- `draft accept %` and `tok/step` from the `ATLAS_DSPARK_ACCEPT_LOG` lines.
- End-to-end median tok/s across ≥5 post-warmup requests.
- The `DSpark drafter loaded: …` line, to confirm B says
  `64-expert MoE, compact draft subset of 216` and A does not.
- The `DFlash drafter store: … (compact draft: 64 routed experts)` line and
  the reported store bytes — B should be ~5.68 GiB lighter than A. This is the
  cheapest end-to-end proof the subset actually took effect.

Decision rule: the compact draft ships as default only if acceptance is within
noise of baseline **and** the freed memory is converted into something
measurable (KV depth, or a lever the memory was blocking). A pure memory win
with an acceptance regression is not a win — acceptance is the 2× gap.

Offline, GPU-cheap pre-check before touching the server (same captures, same
replay, only the expert set changes, so the acceptance delta is attributable
to the subset alone):

```bash
cargo run --release -p spark-model --example dspark_engine_probe                    # A
ATLAS_DSPARK_REF_DRAFT=1 \
  cargo run --release -p spark-model --example dspark_engine_probe                  # B
```

Loader-only sanity (no propose, no verify — checks the subset resolves, the
skip filter bites, and the MoE assembles):

```bash
ATLAS_DSPARK_REF_DRAFT=1 \
  cargo run --release -p spark-model --example dspark_loader_smoke -- <drafter-dir>
```

And with no GPU at all, the selection itself:

```bash
cargo test -p spark-model --lib dspark_reap -- --nocapture
```
