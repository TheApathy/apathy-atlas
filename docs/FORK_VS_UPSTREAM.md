# Fork vs upstream — what we changed and what it bought

A curated engineering record of how `TheApathy/apathy-atlas` (branch
`qwen38/gb10-perf-and-packaging`) differs from `Avarok-Cybersecurity/atlas`,
grouped by subsystem, with the measured effect where one exists and an explicit
"no measured effect" where one does not.

If you only want the headline: **the fork turned a working dense inference
engine into a speculative-decoding engine tuned for one machine (GB10) and one
model family (Qwen3.x-27B). Single-stream decode on the MinHeap probe went from
~48 tok/s at campaign start to 63.8–64.0 tok/s today, roughly 4× the
no-speculation floor of 12.9 tok/s. The price is that the fork no longer commits
the scalar-FMA oracle's tokens: the tensor-core verify path is a deliberate
re-reference, frozen 2026-08-17 at content hash `12e0c0ad`, and the bit-exact
path is still reachable but costs 27.2 vs 31.2 tok/s.**

Everything else in this document is the detail behind those two sentences, plus
the parts that did *not* work, which are a larger fraction of the diff than the
parts that did.

---

## 1. Scope, and how to read this

This document describes the branch **as of 2026-08-23**. It is a snapshot, and
the branch moves faster than the snapshot does — the shas below went stale twice
on the day they were written. Treat the shape as current and the counts as
historical; `git log` and the commands in §11 are the authority.

| | |
|---|---|
| Snapshot taken | 2026-08-23 |
| Upstream `main` at the time | `00cf2c41` |
| Merge base | `ddc7080f` (2026-05-24) |
| Commits ahead, at the time | ~296 |
| Commits behind, at the time | ~201 |
| Diff, at the time | 594 files, +126,455 / −3,283 |

Three caveats govern every number below.

**1.1 — The line count overstates authored engine code.** Of the +126,455
lines, roughly 49,000 are not engine source: `research/ddtree_port` and
`research/dflash_port` (33,381 lines) are *vendored upstream vLLM files* checked
in as reference material for the DDTree and DFlash2 ports, not code we wrote or
compile. `local/` (8,281), `tests/SINGLE_GPU_RESULTS.md` (4,669), `bench/`
(1,450) and `tools/` (954) are harness and audit material. The engine change is
the ~74,000 lines in `crates/` and `kernels/`.

**1.2 — There are two disjoint measurement eras, and mixing them produces
wrong conclusions.** Era A (2026-05 → 2026-07) measured against the AEON-Q36-27B
target with the v5-goheavy drafter, on `counting` / `coding` / `prose` probes
under md5 constitution `91a6ff90`. Era B (2026-08-14 → present) measures against
Qwen3.8-27B with the qwen38-v2 drafter on the MinHeap probe. Almost every
speedup asserted in a fork commit message is an Era-A number. Where Era B
re-measured an Era-A claim, the result was usually null, smaller, or opposite —
see §7. This document labels the era of every figure.

**1.3 — Comparability broke on 2026-08-15.** The binary stopped compiling a
`qwen3.6-27b` kernel target and now ships `qwen3.8-27b` only. Any absolute
measured on a pre-2026-08-15 binary is a different regime, including the
champion's documented acceptance figure. This is written down at
`qwen38/benchmark/arms/atlas-fork.sh:26-34`.

---

## 2. The divergence at a glance

| Cluster | Path | Files | Lines | Character |
|---|---|---:|---|---|
| Model / verify orchestration | `crates/spark-model` | 156 | +35,877 / −2,251 | Speculative decode, SSM, MLA, loaders |
| CUDA kernels | `kernels/gb10` | 124 | +29,751 / −378 | 67 new kernels, 57 modified |
| Scheduler + HTTP | `crates/spark-server` | 78 | +7,192 / −315 | Propose/verify loop, API surface |
| Vendored vLLM reference | `research/` | 112 | +33,381 | Not compiled; port source material |
| Bench + eval harness | `local/`, `bench/`, `tools/` | 86 | +10,685 | Arms, sweeps, evals |
| Correctness audit log | `tests/SINGLE_GPU_RESULTS.md` | 1 | +4,669 | ~55 independent audit passes |
| Runtime / storage / core | `crates/spark-{runtime,storage}`, `atlas-{core,kernels}` | 17 | +1,494 / −102 | Buffers, KV sizing, HSS, build |
| Docs | `docs/turboquant-plus.md` | 1 | +536 | Only tracked doc added |

The shape is lopsided on purpose: `spark-model` and `kernels/gb10` are 88% of
the engine diff, and within them the DFlash speculative drafter (+7,906), the
DDTree tree-verify port (+3,364) and the TurboQuant+ KV compression kernels
(~9,000) are the three largest single investments.

---

## 3. Numerics — the bit-exactness ledger

This is the section that matters more than any other. Read it before citing any
throughput number from this fork.

### 3.1 The constitution the fork used to hold, and retired

Upstream's contract is that the dense verify step commits exactly the tokens a
scalar-FMA forward pass would commit, so speculation is a pure latency
optimization with no output consequence. The fork held that contract through
Era A and **deliberately retired it on 2026-08-17**.

`ATLAS_FFN_TC=1` and `ATLAS_SSM_PROJ_TC=1` route the verify FFN and the SSM
QKVZ/out projections through tensor-core transposed-weight GEMMs instead of the
bit-exact scalar GEMV. MMA reduction order, BF16 weight rounding, and split-K
FP32 partials all differ from the serial FMA oracle, so token output changes.
Rather than treat that as a bug, the fork re-established the reference:

| Reference hash | What it is | tok/s (γ=6, 1500-tok MinHeap) |
|---|---|---:|
| `f376a16e…` | The scalar-FMA oracle. Still reachable via `ATLAS_FFN_TC=0 ATLAS_SSM_PROJ_TC=0` | 27.2 |
| `b8249fb9…` | FFN-only TC | 29.8 |
| **`12e0c0ad…`** | **The REFREEZE — TC FFN + TC SSM projections. This is what "bit-exact" means everywhere in the fork's later record** | **31.2** |

Declared at `qwen38/benchmark/arms/atlas-fork.sh:54-62` and profiled in
`qwen38/analysis/TC-REFREEZE-VERIFY-PROFILE-20260817.md:111-116`, which states
plainly that the zero-serial-oracle-mismatch gate "is deliberately superseded."

**Verdict: the fork bought 4.0 tok/s (+14.7%) at γ=6 by giving up token
identity with upstream.** That was a decision, it is written down, and the
bit-exact path is one env var away. But it means every subsequent "byte-identical"
claim is relative to `12e0c0ad`, not to upstream.

A related correction landed 2026-08-23: comments in
`ops::w4a16_gemm_n64_m32_splitk`, `layers::attn_qkv_splitk` and both
`w4a16_gemm.cu` copies had claimed split-K was "lossless … token-exact". It is
not. FP32 partials avoid mid-accumulation BF16 rounding, but the kernel restarts
its accumulator at each slice boundary and FP32 addition is not associative. The
witness in `qwen38/SPEED-70.md:3005-3033` is `[2^24, 1, 1, −2^24]`: left-to-right
sums to 0, a 2+2 split sums to 1. **"Lossless FP32 partials" ≠ "token-exact."**

### 3.2 What else changes output bits, beyond the deliberate refreeze

These were found by reading the kernel diffs, and several are not in the fork's
own bit-exactness table.

| Change | Location | Nature |
|---|---|---|
| Softmax `exp`: degree-3 Taylor → `__expf` | `kernels/gb10/common/prefill_paged_compute{,_512}.cuh` | Changes **every** paged-prefill kernel vs upstream. The Taylor path was advertised at 1e-4 but measured at 5.1e-3 max relative error, compounding to ~5% cosine drift over 18,920-token rows. Old path survives behind `ATLAS_FAST_SOFTMAX_EXP` |
| P×V MMA BF16 → FP16 | `prefill_paged_compute.cuh` | Softmax probabilities get a 10-bit mantissa instead of 7. ~10% slower, taken deliberately. Revert via `ATLAS_DISABLE_FP16_PV` |
| GDN gate clamp `[1e-6, 1−1e-6]` → `[0, 1]` | 5 files, ~15 sites, incl. `common/gated_delta_rule{,_wy,_wy3,_wy4}.cu` | Bits change whenever a gate saturates. Note the upstream comment justified the ε-clamp as preventing MTP verify drift that "flips single-token argmax decisions" |
| TurboQuant sparse-V threshold `1e-3` | `common/paged_decode_attn_turbo{3,4,8}*.cu` | V load+dequant skipped and zeroed below threshold. An approximation, not a reassociation |
| Turbo K/V write scale: amax → matched-L2-norm | `common/reshape_and_cache_turbo.cu` | **Every quantized KV byte's scale differs from upstream.** Compensates centroid-rounding shrinkage, ~0.5% PPL |
| Turbo LUT `float` → `__half` | same files | Lloyd-Max codebook constants re-rounded |
| MLA softmax scale `1/√128` → `1/√320` | `qwen3_attention/prefill/paged_mla.rs` + 2 more | **Bug fix.** Old scale over-sharpened softmax by ≈0.63× |
| MLA HDIM=256 → `mla_fused_prefill` / HDIM=128 | `prefill/paged_mla.rs`, `cache_skip_mla.rs` | **Bug fix.** HDIM=256 over-read adjacent K heads when hd≤128 |
| FP8 prefill K/V dequant: one scale → per-side | `common/inferspark_prefill_paged_fp8.cu` | **Bug fix** of an upstream FIXME |
| Sliding-window mask hoisted out of `causal` branch | `common/inferspark_prefill{,_h128}.cu` | **Bug fix** for `causal=0` (DFlash bidirectional) only |
| Vision auto-upscale to `ATLAS_VISION_MIN_DIM` (default 768) | `vision_preprocess.rs:63-95` | Different pixels for sub-768px images — that was the intent |
| DFlash capture-layer offset default −1 → 0 | `factory/build.rs` | Changes drafter conditioning; affects acceptance, not committed tokens |

### 3.3 What is genuinely bit-exact

Two categories, and the distinction is worth keeping straight.

**Bit-exact by construction — propose-only.** The dense verify is the oracle, so
anything that only changes *what gets proposed* cannot change committed output no
matter how wrong it is. This covers retrieval/SAM (`dflash_head/retrieval.rs:23-30`),
echo (`echo.rs:16-18`), Markov (`markov.rs:29-32`), early-exit self-draft
(`model/early_exit.rs:11-17`), PLD, recycle, portfolio, DDTree branching, and the
approximate sparse-column draft GEMV (`ops/sparsity.rs:11-14`). Drafter swaps are
therefore free to test: every drafter arm ever measured produced an identical
`content_sha256`.

**Bit-exact restructurings — same math, verified.**

| Change | Mechanism, and what validated it |
|---|---|
| **GDN lazy commit** (`ATLAS_SSM_GDN_LAZY`) | Changes *what is written to DRAM*, not what is computed. Per-token output comes from rolling H registers and never depended on intermediate H being in DRAM; `lazy_j` only decides which of 16 snapshot slots get stored. Partial-accept replays the identical FP32 recurrence in the same order from the untouched root state — `gated_delta_rule_wy17_replay` re-seeds from ROOT, not from a nearer checkpoint, precisely because the WY correction is computed against the full prefix. Validated by 7/7 identical output hashes on the MinHeap probe |
| **`wy17_lazy_vsplit`** | Each CTA owns a disjoint V-column band — a split over independent outputs, not a reduction |
| **`tree_wy` ≡ wy17 chain** (commit `c2005662`) | Two fixes. (a) *Predicate*: the linear fast path fired on `parent[t]==t-1` alone, but branch tail nodes sit at contiguous slots behind their fork, so WY cross-terms were summed over non-ancestor slots — an algebraic corruption, 21–27% of tail-row bytes wrong. Fixed by tracking `prev_contig`. (b) *Rounding order*: the ancestor walk built `gprod` newest-first while wy17 multiplies oldest-first — same product, 1–2 ULP apart. Reversed to match. Validated by `local/bench/bench_tree_wy_equiv.cu`: **0 mismatched bytes** across chain K=17/K=32, fork-leaf, fork+tail, 3-branch K=32. Note `--fmad=false` is what makes source order the only remaining rounding freedom, i.e. what makes bit-exactness achievable at all |
| **Half-warp shuffle masks** (`0b899884`, `ebe5b364`) | Replaces `0xFFFFFFFF` with lane-half masks in MLA reductions. The reduction was already confined to 16 lanes; the mask merely named threads that had already returned — CUDA §B.15 UB. Same arithmetic, same operand order. **Removes UB; does not re-reference** |
| **Fused `gateup_silu`** | Two independent accumulator sets, same per-output K-order. Eliminates an activation round-trip and one launch |
| **`ldb`-stride lm_head** (`8ef523ed`) | Pads B-row stride to 64 with a zero tail — identical layout for 64-divisible callers. Also fixed a real bug: the old predicate scheme silently skipped the last 13 vocab columns per tile row |
| **Prefill FFN pipe kernel** | `crates/spark-model/examples/w4a16_gemm_pipe_parity.rs` does a raw byte comparison of the full output buffer across M ∈ {3,18,64,130}, both FFN shapes, `bail!` on any differing byte. **8/8 PASS** |
| **Cross-seq batched verify** | Pure M-growth of a deterministic GEMM; measured md5-exact at c=8 |
| **Async propose** | Same kernels, same inputs, different stream |
| **DDTree KV relocation** | Byte-lossless permutation of committed KV; test `kv_plan_three_cycle_lossless` at `ddtree.rs:3036` |
| **`dense_gemm_tc` A-tile cooperative load** (`f67d149b`) | Was a *correctness* bug — half the A tile was never loaded for M>8 |

### 3.4 The gap in the bit-exactness story

Three things a reader should not be allowed to assume.

1. **The Era-A md5 constitutions (`91a6ff90` for counting, `e3a39829` for the
   batched arms) are asserted in ~27 commit messages and encoded in no
   CI-runnable test.** They live in shell benches under `local/` and in commit
   prose. `qwen38/analysis/LOSSLESSNESS.md:30-55` states the `91a6ff90` contract
   is model-specific to AEON-27B and was **never re-established for Qwen3.8**,
   that coding was never under a bitwise contract at all, and that
   `ATLAS_THINK_SPEC=1` — which the champion arm sets — is explicitly not
   bit-lossless.
2. **Most parity gates compared spec-on to spec-on and were structurally
   incapable of catching a batched-vs-M=1 divergence.** The one test that could
   is `SPEED-70.md:2628-2640`: same prompt, temp 0, spec-ON vs spec-OFF, both
   sides `sha256=84f9abd377efd2d2 bytes=362`. **PASS.** Cite that one.
3. **`qwen38/benchmark/results/lossless/` contains a single 35-byte file holding
   empty chat scaffold, and `results/gamma-invariance/` is empty.** Neither
   directory validates anything.

Finally, two attention reduction paths ship behind gates named
`ATLAS_UNSAFE_UNVERIFIED_FP8_KGAMMA_EXACT` and
`ATLAS_UNSAFE_UNVERIFIED_BF16_KGAMMA_EXACT` (`layers/mod.rs:1221-1242`). The
alarming names exist specifically to prevent accidental promotion. Good hygiene,
but they are untested on device.

---

## 4. Speculative decoding — the fork's reason to exist

Upstream ships MTP speculation. The fork replaced it with DFlash and then built a
portfolio around it.

### 4.1 What the drafter became

`crates/spark-model/src/layers/dflash_head/` grew from a single flat-chain neural
drafter into **eight competing proposal sources**, each independently gated: the
neural block-diffusion drafter (default), retrieval/SAM, prompt-lookup (PLD),
echo (recycling the target's own rejected verify logits), recycle, a Markov
bigram repair head, early-exit self-draft, and DDTree tree speculation.
`forward_block` was decomposed so one denoise pass (`noise_pass.rs`) can be
iterated for multi-step denoise drafting — and that extraction is documented as
reproducing prior behavior bit-for-bit.

### 4.2 What the verify loop became

`crates/spark-server/src/scheduler/verify_dflash_step.rs` went from 44 lines to
~1,500. Upstream's shape was linear: sync, build `[last_token, ...drafts]`,
verify, exact-match accept loop, commit, save hidden, re-propose. The fork's
shape is a dispatch tree with a four-way accept selection (thinking accept /
grammar-masked accept / DDTree walk / flat), tree-metadata upload before verify,
a think-mask patch that rewrites raw `<think>` argmaxes before the accept walk,
non-contiguous emission under tree commit, a KV compaction pass, and a commit
plan that derives the canonical SSM slot from the accepted path rather than from
the accepted count.

Two changes there are load-bearing correctness work, not performance:

- **`k_verify = tokens.len()`** (was `drafts.len()+1`). With a wide tree the old
  form put the canonical SSM state at the wrong intermediate slot.
- **The ancestor-exactness guard** (`verify_dflash_step.rs:327-341`, commit
  `9e445bcf`). `ATLAS_DFLASH_TREE_COMMIT=1` alone is unsound: the full tree
  walker consumes branch-row argmaxes, which are only valid if verify gave every
  node ancestor-exact attention. Under prefix-read/DFS metadata it now **degrades
  to the flat-safe walker with a warning** rather than corrupting output. This
  prevents md5-level corruption and is a real fix, not a mask.

### 4.3 The measured progression — Era B, and the only citable one

All rows: MinHeap 400-token probe, single stream, greedy, thinking off, 5–7 reps,
deterministic. Measured drift band ±0.25 tok/s.

| Step | Change | tok/s | Δ | Output bits |
|---|---|---:|---:|---|
| entry | `optimized-qwen` target, drafter v2, γ=12 | 48.15 | — | ref `49caed9e` |
| 1 | Target build → `optimized-qwen-unsloth-official` | 52.63 | +4.48 | **changes** — `49caed9e` → `f3b4b2ce` |
| 2 | γ=12 → γ=15 | 54.53 | +1.90 | unchanged |
| 3 | `ATLAS_SSM_GDN_SEQ_PERSISTENT=1` | 56.23 | +1.70 | unchanged |
| 4 | `ATLAS_ATTN_QKV_FUSED=1` | 57.02 | +0.79 | unchanged |
| 5 | `ATLAS_SSM_GDN_LAZY=1` | **62.18** | **+5.16** | unchanged, 7/7 hashes identical |
| 6 | + drafter split-K (this session) | **63.8–64.0** | +1.6–1.8 | see §3.1 — split-K is a re-reference |

Source: `qwen38/SPEED-70.md:1845-2827`, raw JSON in
`qwen38/benchmark/results/minheap-one-lazy{on,base}.json` (62.183 vs 57.050),
reconfirmed at `minheap-reconfirm.json` (62.016). Current-session figures:
63.96 / 63.77 with drafter split-K, 62.86 / 62.92 without; historical baseline on
the same probe 51.26.

**Step 1 is a model swap, not an engine improvement, and it is explicitly
un-gated on quality** (`SPEED-70.md:1928-1934`). Steps 2–5 are engine and config
and are hash-stable against `f3b4b2ce`.

Against the no-speculation floor: `ATLAS_DISABLE_SPECULATION=1` measures
**12.90 tok/s** (= 77.5 ms/forward), and an independent frozen-binary pair gives
13.348 vs 55.929 — **4.19×**. That ratio, not any individual kernel, is what the
fork bought.

### 4.4 Why the wins are all traffic wins

The roofline explains the entire result and should be quoted whenever someone
proposes a FLOP-side optimization. One verify at γ=15 moves **16.33 GB** and does
**0.52 TFLOP** — 32 FLOP/byte against a 238–1190 balance point, so **the GPU idles
roughly 90% of every verify** (`SPEED-70.md:2272-2290`).

Consequently the three wins that survived Era B are all bandwidth wins: lazy
commit eliminates 15 of 16 snapshot writes (+5.16), persistent-H eliminates H
re-reads (+1.70), QKV fusion eliminates launches and memory round-trips (+0.79).
Every FLOP knob measured null *by construction*.

Current cycle census at γ=15: verify 105.57 ms, propose 22.16 ms, cycle
131.25 ms, 5.81 accepted of 15. The cost model is
**verify(k) = 75.53 + 1.890·k ms** (3-point fit at γ = 15/11/7), against a
bandwidth floor of **70.37 ms** for the 16.33 GB sweep at 232 GB/s. An earlier
fit of `78.74 + 2.996·k` appears in `SPEED-70.md:98` — that was measured on a
different binary and is superseded.

The remaining gap to 70 tok/s is a single scalar: per-token accept probability
**p = 0.9011 today, 0.9204 needed** at the current cycle time. That gap is a
drafter property, not an engine one — the kernel terms are at their bandwidth
floor. Drafter work is out of tree.

---

## 5. CUDA kernels

67 new files, 57 modified, zero deleted. The modified set is where numerics move
(§3.2); the new set is where the speed came from.

| Cluster | Files | ~Lines | What it does |
|---|---|---:|---|
| **TurboQuant+ KV compression** | ~40 files under `kernels/gb10/common/` | ~9,000 | Decode + prefill attention over asymmetrically quantized paged KV: K and V independently at bf16 / FP8-E4M3 / 4-3-2-bit Lloyd-Max. `prefill_paged_compute_asym.cuh` (752) takes two tile-loader macros instead of one, which is what makes a 22-variant matrix expressible without 22 hand-written kernels. Pure bandwidth play: 2-bit is 6.4× smaller than bf16, and K-side score fidelity matters more than V-side |
| **K=γ verify attention** | `paged_decode_attn_kgamma_nvfp4.cu` | 1,365 | FlashAttention-v2-style Q-tile fusion: all 17 draft rows attend in one kernel instead of 17 decode launches |
| **Tree / multi-seq state advance** | `tree_kv_scatter.cu`, `gated_delta_rule_tree{,_wy}.cu`, 6 `*_multi_seq.cu` files, `causal_conv1d_chunk3_l2norm.cu` | ~1,300 | Fold `num_seqs` into the grid instead of one launch per sequence — the old path left a 48-SM GB10 running a 256-thread grid. `causal_conv1d_chunk3_l2norm` collapses 288 launches/token into one |
| **WY-chunked GDN** | `gated_delta_rule_wy17.cu` (471), `_vsplit.cu` (376), `qwen3.8-27b/nvfp4/gated_delta_rule.cu` (1,316) | ~2,200 | K=17 WY-chunkwise GDN: q/k loaded to SMEM once, 136 inter-token k-dots precomputed, one fused state-update pass, replacing 17 sequential per-token kernels. Carries the lazy-commit variant. `vsplit` is an occupancy play — wy17 launches exactly 1 CTA/SM, vsplit halves the V-dim to get 2 |
| **Verify-shaped GEMMs** | `qwen3.8-27b/nvfp4/w4a16_gemm.cu` (3,432), `moe_w4a16_grouped_gemm.cu` (1,369), `cutlass_nvfp4_gemm.cu` (735) | ~6,500 | `w4a16_gemm_t_m32_n64` is purpose-built for the K=17 verify FFN shape; plus split-K variants and four fused `gateup_silu` forms. The qwen3.6 copy received the same additions **append-only** — existing symbols untouched |
| **W3 3-bit FFN lane** | `common/w3a16_gemv.cu`, `w3a16_gemm.cu` | 562 | Selected FFN layers drop NVFP4 → 3-bit (LUT `{0,±1,±2,±4}`, 8 weights → 3 bytes). −25% weight bytes on a weight-bound decode. Explicitly ABBA-eval-gated, **not** md5-gated — it cannot be byte-identical |
| **MLA long-context prefill** | `inferspark_prefill_128.cu` (760), `mla_prefill_paged_320.cu`, `inferspark_prefill_h128.cu` (934) | ~1,900 | HDIM=128 and paged absorbed-form HDIM=320 flash-attention for Mistral Small 4 |
| **Vision encoder** | `qwen3.{6,8}-27b/nvfp4/vision_encoder.cu` | 626 | ViT for Qwen3-VL; BF16 storage, f32 accumulate. Self-described "simplicity > performance" — runs once per prefill |
| **Prototypes, explicitly not wired** | `w4a16_dequant_prmt_proto.cu`, `w4a16_gemv_sparse_cols.cu`, `ffn_sparsity_measure.cu` | 588 | All three carry COMPILE-ONLY banners |

One structural note: `qwen3.8-27b/` is largely a copy-fork of `qwen3.6-27b/`
rather than shared code. `cutlass_nvfp4_gemm.cu`, `vision_encoder.cu`,
`w4a16_gemm_v2.cu` and `gated_delta_rule_wy17_vsplit.cu` are byte-identical
duplicates between the two targets. That is ~3,000 lines of guaranteed drift.

---

## 6. Everything else, briefly

### 6.1 Weight loading and mixed precision

The Qwen3.8 checkpoint is not uniformly quantized — FP8-E4M3 islands sit beside
compressed-tensors NVFP4 in the same file — so each projection must be classified
from its own keys rather than from the checkpoint-wide declaration. The tensor
inventory behind that finding was audited against a local, non-shipped
checkpoint and is not published. Alongside
it: a dense MTP head loader for the AEON re-quants, TurboQuant+ weight
pre-rotation folding S2·H·S1/√d into Q/K/V/O at load time (saves 160 kernel
launches per token), a W3 sidecar loader, and transposed FFN copies built at load
when `ATLAS_FFN_M16_TRANSPOSED=1`.

`crates/spark-runtime/tests/fast_weights_parity.rs` asserts byte-identical
weights against the reference loader. The weight cache (`docs/weight-cache.md`)
has a *load-time* measurement — 45–60 s → 17 s, 704 slot comparisons, 0 failures
— and **no throughput claim**; do not attribute one to it.

### 6.2 Memory sizing on unified memory

GB10's unified memory turns GPU over-allocation into a host kill, and the fork
has been bitten. Commit `28d56e40` is the real fix: `model/ssm_pool.rs:119-145`
now predicts the pool footprint and hard-`bail!`s *before* allocating, where
previously the size was only logged after every `gpu.alloc` had already
succeeded. Prediction accuracy 0.4%. KV blocks gained a reachability clamp at
`factory/build.rs:344-348` (silent clamp + `info!`, no error path), and the same
commit found a **13.7× KV over-allocation** worth ~+6% throughput on its own.

**Honest limits.** The window was narrowed, not closed. `PagedKvCache::new`
(`spark-runtime/src/kv_cache/paged_impl.rs:24-28`) still does `2 × num_layers`
allocations with no free-memory check and logs the total afterward — exactly the
pattern `28d56e40` fixed for SSM, left in place for KV. `buffers/sizes.rs` has no
clamp on any arena field. `ATLAS_MAX_BATCH_TOKENS` has no upper bound. And
`gpu.free_memory()` returns `max(cuMemGetInfo.free, /proc MemAvailable)` — most
of host RAM — so at large `--max-seq-len` the surviving bound is not protective.
The SSM guard itself is fail-open (`free = …unwrap_or(0)`) with zero margin.

The single most aggressive change in this area is not a KV path:
`serve_phases/weights.rs:15-36` replaces upstream's 1.3× weight-load safety factor
with per-format multipliers including **NVFP4 → 1.02**, justified by one
empirical data point on one checkpoint on one machine. That is a 28-point margin
cut. It is the line to revisit first if a sizing incident recurs.

### 6.3 Scheduler and API surface

The MTP gate was widened so concurrency ≥2 goes through `step_mtp` instead of
collapsing to the sequential per-seq SSM path (cited: 28 → 14 tok/s aggregate at
c=2), and so thinking spans no longer disqualify when THINK_SPEC is on.
`ATLAS_THINK_SPEC` replicates the plain-decode thinking interventions
(reflection suppression, confidence early-stop, tool-call mask, forced `</think>`)
as a post-verify CPU accept filter, reusing the *same* `process_seq_logits`
function so the two paths cannot drift; it truncates, never rewrites.

API-side: TCP_NODELAY on accepted connections (Nagle + delayed-ACK had stretched
the observed inter-token window ~2.7×, so a true 85.7 tok/s read as ~32–35),
streaming guards that now *cancel* the scheduler rather than suppress output
while the model runs to `max_tokens`, `finish_reason=length` on tool-loop caps,
`--kernel-target` / `--mtp-vocab` / `--default-chat-template-kwargs` CLI flags,
and the `reasoning_effort` fix (`5903be90`) — it had been hardcoded to `"high"` as
a Mistral-template workaround, and Qwen3.8's template `raise_exception`s on it,
so **every thinking request 400'd**.

Correctness fixes worth naming individually:

1. **SSM decode-ring slot reuse after rollback** (`ssm_decode_ring.rs:102-120`,
   commit `1f2658ee`) — the best fix in the diff. `record()` assumed round-robin
   and did `entries.remove(0)`. After a rollback truncation the cursor and live
   set desync, so debug builds panicked and **release builds left two live
   entries sharing one snapshot slot**, silently restoring overwritten SSM state
   on a later rollback.
2. **BF16 KV not applied to MLA layers** (`427104f4`) — FP8 applied to Mistral
   Small 4's 320-dim MLA KV latent produced gibberish past ~600 input tokens.
3. **Drafter ctx drift on bootstrap** (`14f8239f`) — `seq_len` advanced without a
   trim, drifting the drafter's RoPE conditioning. 83 drift warnings → 0,
   non-deterministic → deterministic.
4. **Verify dispatch by `drafts.len()` not `num_drafts`** — the old dispatch
   forced K=2 verify and discarded valid drafts.

### 6.4 Host-side performance hygiene — unmeasured

mimalloc as global allocator, LTO/`codegen-units=1`/`panic=abort`, OnceLock
caching of per-token `env::var` reads, an O(n) stop-string scan replacing an
O(n²) `find` over the whole accumulator, and pre-allocated `output_tokens`.

**Verdict: no citable measurement.** Five of these cite a common baseline
(`28.07 ± 0.60 tok/s`, 13 runs, code_long) that exists only in commit prose —
`grep -rl "28.07"` returns nothing in the repo. Each of those five reports a
*negative* mean delta rescued by a variance-reduction argument. Treat the whole
cluster as hygiene, not as speed.

One item is actively wrong and should be fixed: **`panic = "abort"`**. Its
justification comment says "we never `catch_unwind` across the GPU dispatch
boundary", and no `catch_unwind` appears in any `.rs` — but
`main_modules/serve_router.rs:42` installs `tower_http::catch_panic::CatchPanicLayer`,
which is implemented with `catch_unwind`. Under `panic="abort"` that layer is a
dead no-op, so one panicking request handler now takes down a multi-tenant
server. The comment directly above it states the intent this destroys: "with ~500
production unwraps still in the codebase post-audit, this is cheap insurance."

---

## 7. The refutation bank — what we built that did not work

This is a larger fraction of the diff than the wins, and recording it is the
point. Everything here is measured, with the numbers.

| Thing | Result |
|---|---|
| **DDTree branch speculation** — the largest single port (+3,364 lines), documented in Era A as the champion's biggest coding lever | **The thesis is false on this architecture.** It does not engage on Qwen3.8 and no env flag makes it: `draft_budget.rs:32` makes `tree_nodes == flat` so `remaining` is always 0. Even after a 3-defect fix that built real 15-node trees, verify still ran k=13 and acceptance was unchanged at 6.15. The economics: each node costs 3.0 ms, so a 27-node tree needs **+35% acceptance to break even**. Port A/B: all four arms within 0.08 tok/s with byte-identical acceptance |
| **FP16 GDN h-state** — the campaign's largest projected win, ~200 lines of ported plumbing | **44.42 → 9.18 tok/s**, acceptance 4.78 → 0.20, content hash differs, and **verify time did not move** (100.87 → 100.95 ms). There was never a speedup to trade. The decode f16 GDN kernels do not exist in this tree |
| **DFlash2 (incoai drafter)** — kernels ported, microtest-validated, execution proven | Worse than v2 on both trunks: 36.68 vs 48.15, and 37.92 vs 51.93. p=0.831 vs v2's 0.9011. The `is_causal` fix bought +1.7% |
| **Adaptive draft width / TPS routers** | Every router loses to fixed γ: 41.43 (fixed) vs 38.12 (bandit) / 36.02 (adaptive-γ) / 32.22 / **21.99** (climbdrop-2). Climbdrop cut verify by exactly 25.4 ms = 8 positions × 3.0 ms and still lost |
| **Async propose overlap** | Null twice. The first "null" was itself invalid — it read `async_engaged`, a hardcoded literal 0. Re-measured with real telemetry: 57.01 baseline vs 56.85 async vs 55.50 async-fused, and **no telemetry line ever appeared — the overlap never engaged.** The commit is candid that full propose‖verify overlap is architecturally impossible, since the drafter conditions on verify-time hidden captures |
| **SAM / retrieval drafting** | Null on Qwen3.8: best arm +0.08 against 0.13 drift. Cause is a default — `ATLAS_RETRIEVAL_HYBRID_MIN = l_max = 16` requires a 16-token exact suffix match. Lowering it is monotone loss (hyb10 42.88 → hyb4 41.33). Era-A claim was +12% |
| **FP8 KV cache** | −3.24 tok/s, and verify time *identical* — it saves nothing, because 48 of 64 layers are GDN with constant-size state. Costs acceptance 5.25 → 4.79 |
| **TC QKV path** | −23 to −30% **and lossy**. 51.48 → 39.73 strided / 37.12 nostrided, both with new hashes. Independent repro: 62.76 → 40.21. "Do not re-run it" |
| **FFN gate+up split-K** | Loss: verify 117.88 (fused) → 123.32 (splitk=2) → 121.54 (splitk=4) |
| **`cp.async.ca` → `.cg`** — the textbook L1-bypass for a stream-once weight sweep | **−4.2%**, bit-exact. L1 is earning its keep. Reverted |
| **Big-core CPU pinning** (reported +2–7% elsewhere) | Null. 51.38 vs 51.33, 6 runs, 0.15 spread — the tightest measurement in the corpus |
| **Drafter epoch selection** | Null. Spread 0.10 tok/s (0.2%) with byte-identical output. The prior sweep's 9% spread was measured against a different target and does not transfer |
| **Echo drafting** | Refuted as standalone: 0.85/16 salvage accept. Kept only as a free draft under async-overlap composition |
| **Prompt-lookup drafting (PLD)** | No-win on prose (Era A); Qwen3.8 re-measure 43.118 vs 43.179 baseline |
| **Sparse-column GEMV** | Refutes the "fewer bytes" thesis — single-stream 150 tok/s is not reachable losslessly |
| **Batched QKV v1 / v2** | v1 is completely inert on Qwen3.8 — `ms_phase_qkv` returns `ExactM17` for all n ∈ 4..17, pinned by test `gated_nvfp4_never_falls_through_to_the_batched_route`. v2 collapses accept rate to 0% with an unidentified root cause; quarantined |
| **Software prefetch in the exact FFN** | +1% (27.28 → 27.59). Kernel is issue-bound (IPC 2.64), not latency-bound |
| **`ATLAS_TARGET_LMHEAD_VOCAB=131072`** | Broke output — real code tokens reach ID 131343, so truncated argmax caused a repetition spiral and 0% accept |
| **`ATLAS_TC_NVFP4_M16`** | 48.40 → 38.57 **and the output hash changes** |

Two contested items to handle carefully rather than cite:

- **`ATLAS_LM_HEAD_TC` has three verdicts**: a loss (47.6 vs 51.3, 400-tok probe),
  a win (+1.03, verify −4.0 ms, on 5 coding tasks with census), and "dead; do not
  promote" — where the third also found the output sha drifting with a
  **1419 tok / finish=stop** vs 1500/length outcome, i.e. an MMA rounding flip
  causing early EOS. The first two are reconciled by workload mix; the third is
  not reconciled anywhere and is an output-correctness finding, not a throughput
  one. **Do not present this as a clean win.**
- **Three `drafter-e*-r1.json` files report identical acceptance at ~45.9 tok/s
  because the arm's own drafter was never loaded** — all three record the same
  `draft-model` path. Their r2 replicates land at 37.6 / 36.3 with distinct
  acceptance. Do not cite the r1 rows.

**The harness was itself the thing being measured on at least five occasions**,
each time producing a plausible number: `--disable-thinking` on a reasoning model
zeroing an entire category; a 512-token default cap that 77% of cases hit;
unbounded reasoning consuming 44% of wall clock; a self-inflicted concurrency
probe reading 16 tok/s; and an arm that benchmarked the *incumbent* server and
reported its result as its own. The guards added in response — live
`/proc/<pid>/cmdline` model-identity assertion, flag-presence checks, lock
holder-PID liveness — are themselves a fork deliverable.

---

## 8. Local workarounds and debt

Changes that are pinned to our machine, our checkpoint, or our schedule rather
than being general fixes. Each is a candidate for either upstreaming properly or
deletion.

| Area | Item |
|---|---|
| **Kernels** | The **M_TILE=16 / K=17 corruption is flag-isolated, not fixed** (`b95f9998`, `a8855631`). `w4a16_gemm_t_m16` produces subtly wrong logits on its second tile at M=17, collapsing acceptance 15.9/16 → ~1.5/16, and masqueraded as a drafter-quality problem for a whole session. A/B narrowed the culprit to the `ms_phase_qkv` `TC_NVFP4_M16` attention path with two named suspects — **neither investigated**. Serve scripts carry a `DO-NOT-ENABLE` comment. A live latent bug behind a default-off flag |
| | `gated_delta_rule_wy17.cu` hardcodes `K_TOKENS 17` / `BLOCK_SIZE 128` with the SMEM budget computed for exactly that shape; `gated_delta_rule_tree_wy.cu` hardcodes `K_MAX 32` for static SMEM sizing. `w3a16_gemm.cu`'s format must match `local/tools/repack_w3.py` byte-for-byte with **no version check in the kernel** |
| **Model** | A **committed merge artifact**: `layers/dflash_head/async_propose.rs.orig`, 513 lines, tracked in git, a stale near-duplicate that will drift |
| | `ATLAS_DFLASH_ZERO_LATE_LAYERS` is self-described as "a workaround for SSM kernel numerical drift" (cosine similarity falls to 0.86 by L61 vs HF). Its companions read ctx hidden states **from a file on disk** written by a Python sidecar, or fetch them over **HTTP per prefill**. Debugging scaffolding for an unfixed kernel drift, living in the production forward path (gated, default off) — and if enabled it issues ~8k blocking host syncs per propose step |
| | `model/trait_impl/verify_csk.rs` documents itself as "scaffolding shipped, REGRESSION" with the losing numbers in its own header. ~384 lines of gated-off dead path |
| | Capture layers `[1,16,31,46,61]`, "48 SSM layers, c=4" sizing arithmetic, and the W3 LUT justified empirically on AEON-Q36-27B weights are all single-checkpoint tuning. Notably there are **no absolute filesystem paths anywhere in the crate** |
| **Server** | `anthropic/handlers.rs:339` — `let skip_template_tools = false;` stubs off an upstream feature because `ModelBehavior` lacks the field. The MODEL.toml opt-in is silently ignored |
| | Qwen3.8 is identified by `model_type == "qwen3_5"` plus two magic template substrings — content sniffing |
| | `--kernel-target` **bypasses the `(model_type, hidden_size)` compatibility check**, so a wrong value loads mismatched kernels instead of being rejected |
| | `--mtp-vocab` defaults to truncating the LM-head GEMV to low token IDs on the assumption BPE is frequency-ordered; malformed `--default-chat-template-kwargs` JSON is silently dropped |
| | `ATLAS_SSM_MULTI_SEQ_BATCHED` warns at boot that it is **"currently a no-op"** — a documented-inert env var, shipped |
| | `--kv-cache-dtype turbo2` is user-selectable and documented to **"produce garbage at runtime"** (write kernel exists, decode kernel does not) |
| | `f6175ea0` added `post_think_gate_steps` to `ActiveSeq`/`SwappedSeq` and threaded it through six construction sites. It has one write (`= 0`) and **no reads**. Dead |
| **Storage** | The `qd > 64` bound was weakened to `qd > 256` and the guarding test edited to match with the comment "assertion bound was stale". `stream_sync` was **deleted** from `io_uring read()`, with correctness now resting on an in-comment assertion true only for today's two callers |
| **Build** | `atlas-kernels/build.rs` **removes** the macOS auto-skip, the `metallib_modules()` stub, and the `ComputeTarget` source-extension plumbing that exist at the merge base. **Metal / non-NVIDIA build support present upstream is gone on this branch** — this will conflict on any re-sync |
| **Process** | Three commits — `f81ae296 snapshot: full uncommitted tree pre-rebase`, `90192848` and `80147254 wip: pre-existing uncommitted work` — mean a nontrivial slice of the 35,877 `spark-model` lines never passed through a reviewed change. One casualty is visible: the native FP8 SSM prefill path present at the merge base **and still present upstream today** is gone from HEAD (`grep fp8_ssm_prefill` returns nothing). Its removed comment documented it as a root-cause fix with `tokens_to_first_degeneration` going 1,196 → 16,968. **Either that removal was deliberate and needs a note, or it is a silent quality regression on every FP8-on-disk checkpoint** |

---

## 9. Re-syncing with upstream

We are **201 commits behind**. The merge base is 2026-05-24; upstream's last
three months are concentrated in areas we did not touch, which is fortunate.

**Low conflict risk — take these.** Upstream's recent work is dominated by CI and
site infrastructure: the CodeRAG corpus pipeline (`fa198bef`, `2d3bb04d`,
`ad87e219`, `5af5bc41`), the governance intent-ledger harvest workflow
(`5690b19b`, `5aa9f779`, `167d8b12`, `73544705`), CLA allowlisting, and the
"Ask the codebase" site feature (`33d283cb`). None of it touches `crates/` or
`kernels/`.

**Genuine engine work upstream that we should evaluate.**

- `04726b14` — a concurrency cycle stacking five PRs, adding a decode-floor gate,
  a concurrency instrument, accept-stats, and per-model BFCL floors. This overlaps
  our scheduler work directly and is the most important item to read before any
  merge.
- `da56736d` — restores shared-memory LUT staging in the decode GEMV partials.
  Check whether our `w4a16_gemv` additions conflict or subsume it.

**Things we should send upstream rather than carry.** The MLA fixes are general
bug fixes, not GB10 tuning, and belong upstream: the absorbed-space softmax scale
`1/√320`, the HDIM=256 over-read replacement, `kv_write_start` in cache-skip MLA
writes, the half-warp shuffle masks, the `dense_gemm_tc` A-tile cooperative load
for M>8, and the FP8 prefill per-side K/V scale. Same for the SSM decode-ring
rollback fix and the batched-MoE dynamic `s_act`.

**Things that must not be merged as-is.** The `atlas-kernels/build.rs` Metal
removal (§8) would regress upstream's non-NVIDIA support. The NVFP4 1.02 memory
multiplier is machine-specific. The `panic="abort"` profile is wrong on its own
terms.

---

## 10. Coverage of this document

Honest accounting of what was characterised and what was not.

**Covered by reading the actual diff:** `crates/spark-model` (top ~15 files by
size plus every numerics-relevant site), all 57 modified files under
`kernels/gb10` and the largest new ones, `crates/spark-server` scheduler and API
surface, `crates/spark-runtime` buffers and KV sizing, `crates/spark-storage`,
`crates/atlas-kernels/build.rs`, and the full measurement corpus
(`qwen38/SPEED-70.md`, `qwen38/benchmark/results/`, `qwen38/analysis/`,
`tests/SINGLE_GPU_RESULTS.md`, `results.md`).

**Sized but not read line by line:** the 67 new `kernels/gb10` files were grouped
by purpose and spot-read, not audited individually. `research/ddtree_port` and
`research/dflash_port` (33,381 lines) were confirmed to be vendored upstream vLLM
source and were not reviewed as our code. The ~86 harness files under `local/`,
`bench/` and `tools/` were surveyed for what they measure, not reviewed for
quality. The `xgrammar` crate changes and the ~55 audit passes in
`SINGLE_GPU_RESULTS.md` were not characterised.

**Deliberately out of scope:** per-commit attribution, and any claim that would
have required a GPU run to verify (a benchmark was running).

---

## 11. How to regenerate this document

Re-run these rather than rewriting from scratch. All are read-only.

```sh
cd /path/to/apathy-atlas

# 1. Re-establish the frame.
git fetch upstream
git rev-list --count upstream/main..HEAD          # commits ahead
git rev-list --count HEAD..upstream/main          # commits behind
git merge-base upstream/main HEAD
git diff --shortstat upstream/main...HEAD

# 2. Re-size the clusters (regenerates the §2 table).
git diff --numstat upstream/main...HEAD \
  | awk '{split($3,a,"/"); k=a[1]"/"a[2]; add[k]+=$1; del[k]+=$2; n[k]++}
         END {for (i in add) printf "%8d %8d %5d  %s\n", add[i], del[i], n[i], i}' \
  | sort -rn | head -30

# 3. Re-read the commit narrative, then re-cluster.
git log --oneline upstream/main..HEAD

# 4. Per-area sizing.
for p in crates/spark-model kernels/gb10 crates/spark-server \
         crates/spark-runtime crates/spark-storage crates/atlas-core \
         crates/atlas-kernels docs tests bench local research tools; do
  printf "%-24s %s\n" "$p" "$(git diff --shortstat upstream/main...HEAD -- $p)"
done

# 5. The numerics axis — modified files matter more than added ones.
git diff --diff-filter=M --name-only upstream/main...HEAD -- kernels/gb10
git diff --diff-filter=A --name-only upstream/main...HEAD -- kernels/gb10

# 6. Re-check the env-gate surface and its defaults.
grep -rn 'env::var("ATLAS_' crates/ | wc -l
grep -rn 'env::var("ATLAS_' crates/ | grep -v 'Some("1")'   # candidate default-ON gates

# 7. Re-check upstream recency before claiming a re-sync plan (§9).
git log upstream/main --oneline -25
```

Measurement sources to re-read, in priority order. **None of these ship in this
repository** — they are working files on the measurement box, listed so the
provenance of the numbers above is nameable rather than anonymous. An outside
reader cannot follow this list; the published extract is §4 and §7.

1. `SPEED-70.md` — the primary log. Section 41–49 is the current Era-B
   progression; §53 is the split-K re-reference correction.
2. `benchmark/results/*.json` — raw arm results. Always check
   `spec_env.draft-model` and `deterministic` before citing a row.
3. `analysis/` — per-lane closure documents. The TC-REFREEZE and
   bandwidth-ledger docs supersede earlier framings and say so.
4. `benchmark/arms/atlas-fork.sh` — the champion configuration, with the
   REFREEZE declaration at lines 54-62. The published equivalent is
   `bench/qwen38-gb10/serve.sh`.

When updating: keep the era label on every number, keep the "no measured effect"
rows, and keep the refutation bank in §7. The refutations are the most expensive
knowledge in this repository and the easiest to lose.
