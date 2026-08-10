# SPEC-3X — where the speculative multiplier is lost, and the ranked road to 60–80 on favorable content

Analysis of 2026-08-10 (no new GPU runs; every number below is quoted from a
named log, doc, or in-tree constant, or is arithmetic on those). Companion
instrumentation shipped in this commit: `STEP_TIMING2` now prints
`tok_step=` and `spec_tok_s=` in the same 64-step window as the phase wall
times, so acceptance × step-time never has to be stitched from two
differently-windowed logs again.

## 0. Semantics first — the numbers people quote are off-by-one in two places

1. **CLI γ is not drafts.** `serve.rs:616`: `num_drafts = dflash_gamma - 1`.
   The bench default `--dflash-gamma 5` runs **4 drafts/step**, verify width
   K = 5 (confirmed: "Captured CUDA graph for K=γ verify (slot=0 **K=5**)"
   in `serve-v2-verify.log`, and `drafted = 4·steps` in every accept line).
   The reference runs `num_speculative_tokens: 5` = **5 drafts**, graph
   capture size 6. **We speculate one token shallower than the stack we are
   chasing.** Older docs' "γ=5 (m=6)" describe the 5-draft config; today's
   serve is m=5.
2. **The accept log's "X tok/step" is committed tokens** (accepted drafts
   + the always-emitted bonus), per `accept_log.rs:52`. "Accept 3.59" on
   repeat = 2.59/4 drafts accepted (64.8%) + 1.

So the correct frame is:

```
tok/s = committed / T_step,   committed = accepted_drafts + 1 ≤ (γd + 1)
γd = drafts/step = CLI-γ − 1;  verify rows m = γd + 1
```

## 1. Today, measured (2026-08-10 logs, graphed verify, V2 kernel stack)

Config: `dsflash-serve-bench.sh` γ=5 (γd=4) + `ATLAS_DFLASH_ADAPTIVE`
`ATLAS_DFLASH_LOW_GEAR` `ATLAS_MTP_GATE_FORCE` `ATLAS_MOE_GEMV_V2`
`ATLAS_V4_DECODE_FUSED` `ATLAS_V4_PROJ_FP8MMA` `ATLAS_VERIFY_GEMV_V2`,
NVFP4 attention. Plain baseline 21.9 tok/s (45.7 ms/step, graphed).

| workload | committed tok/step | draft accept | zero-accept steps | histogram (accepted:steps) | tok/s | implied T_step | source log |
|---|---:|---:|---:|---|---:|---:|---|
| repeat | **3.59** | 64.8% | **7.8%** | 0:5 1:12 2:11 3:12 4:24 /64 | **33.5–33.8** | **~107 ms** | serve-verify-wf (15:29), serve-reprobe1k, serve-drafter-gemv |
| mixed bench (code/quote/prose 300-tok gens) | 2.74–2.81 | 43.6–45.3% | 26.6–28.9% | 0:74 1:55 2:43 3:31 4:53 /256 | 20.2–22.7 | — (adaptive-suspend mixes serial steps in) | serve-v2-verify (16:15) |
| prose under adaptive | suspends (mean accepted 0.6–1.9 < 2 over 12) → serial **eager** decode | | | | ~19.8 vs plain 21.9 | | serve-gate-final3 |

Step decomposition (γd=4→5 era, `DECODE-WATERFALL-2026-08-10.md` §6 +
today's MLAPROF):

| bucket | ms/step | floor | note |
|---|---:|---:|---|
| MoE expert union (`exp_splitk_m_t`) | 54.1 | ~35 (MXFP4 @205 GB/s) | 135 GB/s vs m=1 sibling's 206 |
| MLA `A_proj` | 10.5 (245 µs × 43, 15:29 log) | ~4.5 | weights read once ⇒ m rows ≈ m=1 cost; runs 2.4× that |
| MLA `C_oproj` | 11.3 (262 µs × 43) | ~7.5 | 1.5× weight-bound floor |
| `B_attn` (rope/cache/paged) | 4.4 | ~4 | fine |
| MoE gate | 6.2 | — | GEMV lever measured a WASH (routing flips), do not re-chase |
| HC/norms/glue | ~16 eager | ~8 | partially probe-sync artifact; graphed truth unmeasured |
| route/blend | 1.7 | 1.7 | fine |
| **verify total** | **~113 eager / ~90–92 graphed** (inferred: 107 − propose 12.6 − host ~3) | ~60 | K=5 graph IS captured now — but **re-captured every request** (15 captures / 16 requests in serve-v2-verify: cache key does not survive request boundaries) |
| propose | 12.6 | ~9 | per stage×3: a_hc_pre 1.7–2.0, b_qkv 0.42–0.45, c_attn 0.25, d_o_proj 0.6–0.68; + stage_moe 1.7, lm_head 2.7 (top item) |
| host (loop_gap/walk/emit/commit) | ~2–4 | — | now directly measurable: `ATLAS_STEP_TIMING2=1` prints all buckets + `tok_step`/`spec_tok_s` |

Cross-check: 3.59 / 0.107 = 33.6 ✓ (1.54× plain).

**What closed acceptance from "~1.0" to 3.59 on repeat** (so we don't
re-litigate it): the "pinned at 1.0" era was a workload-mix artifact + gate
probe-mixing (memory: the 3.79-on-plain-captures "capture-quality gap" was
DEMOLISHED 2026-08-07 — same-prompt plain captures give 2.29, online 2.38);
the real gains were the boundary-slot fix (1.02→1.10), the FULL exact-GEMV
verify chain (2.83→2.92–3.01, zero-accept 20.3→17.7%, and THE LAW: partial
exactness is worse than none), and the drafter grouped-GEMV (propose
19.4→12.6 ms). Remaining zero-accepts are mostly CONTENT (7.8% on repeat vs
27–29% mixed) with a bounded numerics slice (~3–10 pp — the two unconverted
legs below).

## 2. The refined arithmetic — what 3× and 4× actually require

3× plain = 65.7 tok/s; 4× = 87.6.

| γd (drafts) | max committed | T_step for 3× at PERFECT accept | T_step for 3× at 90% chain (committed ≈ 0.9·max) |
|---:|---:|---:|---:|
| 4 (today) | 5 | 76 ms | 68 ms |
| 5 (reference depth) | 6 | 91 ms | 82 ms |
| 7 | 8 | 122 ms | 110 ms |
| 9 | 10 | 152 ms | 137 ms |

Today's step is ~107 ms at γd=4 ⇒ **3× is arithmetically impossible at
today's depth even with perfect acceptance.** Both levers are mandatory and
a third (depth) besides:

- **accept-at-depth**: repeat survival today P(≥1..≥4) = 0.92/0.73/0.56/0.38
  (conditionals 0.92/0.80/0.77/0.67). The reference's repeat (58.6 tok/s at
  K5, T_step ≈ 100 ms) implies committed ≈ 5.5–6/6 — near-perfect chains
  (p≈0.95/depth). **Our repeat chains die at p≈0.7–0.8/depth. This, not
  step time, is the largest single multiplier gap** (committed 3.59 vs ~5.9
  at the same depth+step ⇒ 33.6 vs ~55).
- **T_step**: 107 → ≤ 91 ms at γd=5 (buckets above show ~28 ms recoverable
  before any new idea).
- **γ**: 60–80 needs committed 6–8, i.e. γd = 8–10 with the p≈0.9 chains.

### Verify cost vs γ — the MoE union grows much slower than the binomial bound

Independence bound: E[unique experts] = 144·(1−(1−6/144)^m):

| m | 1 | 2 | 4 | 5 | 6 | 8 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|
| E[unique] | 6.0 | 11.8 | 22.5 | 27.6 | 32.4 | 41.6 | 49.9 | 53.8 |
| ratio vs m=1 | 1.0 | 2.0 | 3.8 | 4.6 | 5.4 | 6.9 | 8.3 | 9.0 |

But the MEASURED anchor refutes independence in our favor: m=1 MoE moves
4.02 GB in 20.9 ms (192 GB/s); the m=6-era union kernel measured 54.1 ms at
135 GB/s ⇒ **~7.3 GB unique bytes = only ~1.8× the m=1 stream**, not the
5.4× the formula predicts (5.4× would be 19+ GB — physically impossible
under 54.1 ms at any bandwidth ≤ 229). Adjacent verify tokens reuse experts
heavily. Consequence: **deep γ is cheap on the MoE side.** Projected MoE
union at constant routing locality: m=8 ≈ 60–65 ms, m=10 ≈ 70–80 ms at
today's 135 GB/s; ~40/48 ms at 205 GB/s; ~28/34 ms with EXL3 3.0 bpw on
top. MLA A/B/C are flat-to-linear-small in m. A γd=9 step with the R3+R4
levers lands ≈ 95–105 ms — inside the 3× window at committed ≥ 6.3.

Protocol to nail the real curve (one run): the union logger already exists
(`chore(moe): sample the expert union densely enough to log in one run`) —
sweep CLI γ ∈ {5,6,8,10} on repeat+code, log E[unique] per layer per step.

## 3. Ranked plan

Expected numbers are for REPEAT (the favorable-content target); the
across-the-board floor is carried by adaptive + the EXL3 plain lift.

| # | item | cost | expected effect | measurement protocol |
|---|---|---|---|---|
| **R0** | **Joined measurement pass** (this commit's `tok_step`/`spec_tok_s` in STEP_TIMING2) + γ-sweep CLI γ ∈ {5,6,8,10} × {repeat,code,quote,prose} with `ATLAS_DSPARK_ACCEPT_LOG=1 ATLAS_STEP_TIMING2=1 ATLAS_MTP_GATE_FORCE=1` | 1 serve session | the accept-vs-depth survival curve + true graphed T_step decomposition per workload — gates everything below | one summary line per 64 steps now contains wall, phases, tok_step, spec_tok_s |
| **R1** | **Match reference depth: CLI γ=6 (γd=5, m=6)** | config only | repeat +0.3–0.5 committed for ~+5–7 ms MoE marginal: 33.6 → **~35–37** | A/B γ5 vs γ6, same prompts, accept log + spec_tok_s |
| **R2** | **Accept-at-depth on favorable content** — (a) finish the exact-GEMV chain: head-gate `dense_gemv_batchm` (third family, unverified) + batched attention/rope exactness (the layer-diff harness convicts non-GEMV stages); (b) A/B probabilistic draft sampling (reference `draft_sample_method: "probabilistic"`) vs our Markov-biased greedy chain | days, in-tree harnesses exist | the single biggest multiplier: repeat draft-accept 65% → 85%+ ⇒ committed 3.59 → ~4.5–5.5 at γd=5: **→ 45–55 at today's step time**. Mixed zero-accept 27.6% has a proven-compressible numerics slice (first two exactness legs bought 20.3→17.7%) | accept histogram per leg; obey THE LAW (flip all legs together); quality gate 90/100 |
| **R3** | **Verify step 107 → ~85–90 ms** — (a) A_proj 10.5→~4.5 (weights-read-once bound), (b) C_oproj 11.3→~7.5, (c) HC/glue graphed-truth then fusion ~16→~8, (d) MoE union 135→~205 GB/s (54→36 ms; expert-major streaming — the one bucket that still scales with m) | kernel campaign, oracles exist | repeat at committed 3.59: **→ 40–42**; stacked on R2: **→ 50–58** | microtest oracle per kernel + bit-exactness gate (the exact-GEMV law binds every verify kernel), then spec_tok_s |
| **R4** | **EXL3 3.0 bpw experts** (bring-up in flight; quality-gated) | separate campaign | verify MoE ×0.71 bytes (36→26 ms @205); **plain 21.9→27–29** — lifts the adaptive prose floor, which is what makes ">28 across the board" true; makes γd=8–10 affordable | EXPERT-3BPW-PLAN.md gates; re-run R0 sweep on the EXL3 build |
| **R5** | **Deep γ (γd=8–10)** once R2 chains hold p≥0.85 | config after R2–R4 | committed 6.5 / step ~100 ms ⇒ **65**; committed 7.5 / ~105 ⇒ **71**. This is where 60–80 lives. 4× (88+) needs γd≥10 at p~0.9 AND step ≤ 90 at m=11 — only with full R3+R4 | γ-sweep again; stop at the knee of the survival curve |
| R6 | Host/glue cleanups: graph-cache keys that survive requests (15 recaptures/16 requests measured — each recapture is an eager step + capture cost), graph the adaptive-SUSPENDED serial path (+2 tok/s on prose), drafter lm_head NVFP4 (2.7→~1.4 ms, acceptance-gated not quality-gated), propose floor 12.6→~9 | small, parallel | +1–3 tok/s each in their regime | LOOP_TRACE + STEP_TIMING2 loop_gap/collect buckets |

### Drafter swap (EXL3 ships the reference K64 draft in-checkpoint) — sized, and small

Their draft = the model's own mtp.* stages with experts REAP-sliced 256→64
(purpose: MEMORY for 262K KV, per the mined stack analysis). Our propose MoE
costs **1.7 ms of a 12.6 ms propose** — a 64-expert drafter saves ≤ ~1.3 ms/step
(~1%) and risks acceptance (fewer experts, and our accept parity with their
draft family is already proven: we are AHEAD on prose, 20.0 vs 19.5). The
drafter is NOT the multiplier gap. A/B it only because it's free once the
EXL3 loader lands; expect ~neutral.

### DDTree — re-run at today's numbers: still dead

Marginal verify row today ≈ 7 ms of a 107 ms step (6.5%) vs 10.5/173 (6.1%)
on 2026-08-04 — the economics did NOT change. Break-even needs
+0.23 committed/step from 2 branch rows; measured branch win rate was 4.5%
of steps = +0.045. Still ~5× short; death-depth placement buys ≤3×. Verdict
unchanged. One cheap salvage: the per-depth deaths/top2/margin probe already
in `verify_dflash_step.rs` (~line 316) — read it during the R0 sweep; only
if deaths concentrate at ONE depth with top2-hit ≥ 50% does a single-branch
tree deserve one more A/B.

## 4. The ceiling, stated honestly

- **With today's drafter** (0731 3-stage V4, 256-expert, greedy chain):
  content-driven acceptance — engine-probe ceilings ~2.3–2.4 committed on
  prose, ~3.8+ on code/math at γd≈5. The reference's same-family K5 draft
  spans 2.18 (adversarial prose, 0.96× plain — THEY lose on prose too) to
  4.00 (code). Prose speculation is structurally ≤ plain; the prose floor
  must come from adaptive + the EXL3 plain lift (27–29), not from spec.
- **3× across the board is not a real target** — the reference's own median
  multiplier is 1.43–1.9×. The reachable shape is: **prose 27–30 (plain
  floor), code ~45–55, repeat/templated 60–80** at γd 8–10 with R2 chains
  and R3+R4 step times. 4× (80+) exists only on repeat-class content at
  γd≥10, p≈0.9, step ≤ 90 ms — every lever, no slack.
- Waterfall sanity: at γd=9 the per-committed-token weight traffic is
  ~(MoE union 2.5×4.02×0.71 + MLA 2.1 + head 0.5)/6.5 ≈ 1.5 GB/token —
  4.5× less than plain's 6.7 GB/token. Speculation is how this hardware
  beats its own 34 tok/s plain byte ceiling; there is no other road.

## 5. Immediate next actions (in order)

1. Run the R0 sweep (one session, ~30 min serve time) — γ×workload grid
   with the new joined summary line; extract survival curves + graphed
   T_step + expert-union curve.
2. Flip CLI γ to 6 in `dsflash-serve-bench.sh` if R1's A/B confirms (it
   should — the reference ships exactly this depth).
3. Start R2(a): head-gate `dense_gemv_batchm` exactness leg — the harness
   and the law are already in-tree; this is the highest-EV work item in the
   entire plan.

---

## Matched-protocol measurement vs the reference (2026-08-10)

Ran the reference's **published protocol verbatim** — 5 independent 512-token
code generations, concurrency 1, steady state excluding TTFT and the first
generated token, min/median/mean — against our own stack. This supersedes
every prior cross-stack decode comparison in this repo, which used our
`decode_ab_probe` workloads and is NOT the same measurement.

| | min | median | mean | draft accept |
|---|---:|---:|---:|---:|
| ours, γ=5 | 16.06 | **17.41** | 17.22 | 61.6% (473/768) |
| ours, γ=8 | 17.13 | **18.80** | 18.55 | 54.7% (525/960) |
| **reference** | 34.30 | **38.12** | 39.49 | ~95% (implied) |

**Gap: 2.0×, and it is draft acceptance, not engine speed.**

Arithmetic: at γ=8 we commit 3.73 tok/step at an 18.80 tok/s rate ⇒ 198 ms
step. For the reference to reach 38.12 on comparable hardware and a
comparable K5 verify step, it must commit ~5.7 of a maximum 6 per step ⇒
~95% per-token draft acceptance. Ours is 55-62%. Same GPU, same model size,
same verify math — their drafter is right nine times in ten where ours is
right roughly one time in two.

Corollary: **deeper γ helps us slightly (γ=8 > γ=5 here) because our
acceptance is poor**, which is the opposite of the reference's regime; with
95% acceptance the shallower K5 wins on step time. Optimal γ is a function of
draft quality, so any γ policy must be re-derived after a drafter change.

### Context on the reference's other published numbers

- Their **concurrency-1 general** decode (`results/c1-c2-c4.json`) is
  **18.94 tok/s** — the 38.12 is a curated code scenario. Against the general
  figure we are at parity or ahead.
- Their **1055 tok/s prefill** is a single **252,047-token** prompt. Ours is
  1062 (n=20 median) at **2,410** tokens — two orders of magnitude shorter and
  much harder to saturate. Prefill is not a gap.
- They explicitly do not claim the decode number is settled: "a steady
  35 tok/s floor remains an open optimization gate… not claimed as complete";
  the prefill figure is labelled "tuning evidence".

### What this does to the ranked plan

Kernel work on plain decode (the 28-tok/s ladder in DECODE-WATERFALL §6) is
real but bounded at roughly +6 tok/s. **Draft acceptance is worth +19.** The
reference's 64-expert compact drafter with sparse MLA attention ships inside
the EXL3 checkpoint already loaded and served successfully on this box, so
the drafter comparison is now the highest-value open experiment.
