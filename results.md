## FFN gate/up 2026-06-18

**Decision: NO-GO on kernel rewrite. gate/up are NOT load-pipeline-bound; a Marlin-class cp.async rewrite cannot help.**

### Headroom measurement (the key go/no-go datum)

Standalone microbench of the EXACT production kernel `w4a16_gemm_t_m32_n64`
(the kernel the m32 verify path routes gate/up through) at the real gate/up
shape **M=17, N=17408, K=5120** on GB10 (48 SMs, sm_121f). Weight = 47.8 MB
(Bpacked 42.5 + Bscale 5.3 MB). 200-iter timing, results stable across runs:

| Path | Time | GB/s | % of DRAM ceiling |
|------|------|------|-------------------|
| **GEMM (production m32_n64)** | 0.26 ms | **~190** | **~74%** |
| Pure load path (same tiled cp.async access, no dequant/MMA) | 0.205 ms | **244** | **94%** |
| DRAM read ceiling (uint4 streaming, weight footprint) | 0.19 ms | ~259-266 | 100% |

Cross-check vs live profile: microbench 260 us/layer ≈ profile 289-305 us/layer
(gate/up ÷ 64 layers). The microbench is representative of the real verify step.

### Why no rewrite is warranted

1. **The load path already hits 94% of achievable DRAM bandwidth** (244 of
   ~259 GB/s) at the real 272-CTA launch. There is essentially zero
   load-pipelining headroom left for cp.async double-buffering / deeper
   prefetch to capture. The kernel *already* has a 2-stage cp.async
   double-buffered K-loop.

2. **The ~25% gap (190 vs 259 GB/s) is dequant compute on the critical path,
   not bandwidth.** Evidence: M=1, M=17, M=32 all run in ~0.247-0.267 ms
   (MMA work scales with M but barely moves the wall) — and even M=1 with
   minimal MMA only reaches ~202 GB/s, far from the 244 load ceiling. The
   loss is the per-tile NVFP4->FP8 dequant (LUT lookup + scale mul +
   cvt.e4m3x2) sandwiched between `cp.async.wait` and the MMA, serialized by
   the two `__syncthreads()` per K-step.

3. **Tested rewrites made it WORSE or no-change (all bit-exact, K-order
   preserved):**
   - 3-stage deeper-prefetch pipeline (decouples dequant from load):
     **161-166 GB/s — WORSE.** The 3rd SMEM stage drops occupancy
     (kernel holds 9 blocks/SM today; deeper buffering loses that).
   - Double-buffer only `smem_B_fp8` to drop a syncthreads: **195 GB/s —
     no change.** Not sync-bound.

   Occupancy is the binding constraint, not pipeline depth: m32_n64 already
   fits 9 active blocks/SM (432 CTAs capacity vs 272 launched). Anything that
   raises SMEM (the lever a Marlin rewrite pulls) costs occupancy and
   regresses.

### Conclusion

gate/up are effectively **at the floor** for this dataflow. The DFlash K=17
verify is genuinely bandwidth/dequant-saturated, not occupancy-starved (which
is why split-K fixed `ffn_down` but cannot help gate/up — they already field
272 CTAs and saturate the load path). Pushing counting past ~83 tok/s needs
**acceptance-rate work (drafter), not FFN kernel work.** The only residual
gate/up lever would be cutting dequant cost (e.g. pre-dequant the gate/up
weights to FP8 at load, trading ~9 GB VRAM for skipping the per-step
NVFP4->FP8 conversion — the `ATLAS_FFN_PREDEQUANT_FP8` path already scaffolded
in dense_ffn.rs), which is a memory-footprint tradeoff, not a pipelining
rewrite, and out of scope for this de-risked investigation.

### Gates
- No kernel/Rust edits made (NO-GO). Working tree unchanged from the
  pre-existing split-K + vision WIP. No TEXT/VISION/BENCH regression possible.
- /tmp microbench artifacts cleaned. GPU left free.

## Acceptance lever sweep 2026-06-18

**Goal:** find CHEAP (flag-only, no rebuild) DFlash levers that raise CODING
draft acceptance / tok/s on AEON-27B without quality loss. Coding is the target
(acceptance wall). Counting/prose are fine.

**Method:** serve `local/serve-aeon-27b-dflash.sh` (records-grade drafter
defaults: nvfp4 drafter, γ=16, ffn_down split-K, M16-transposed verify) with
`ATLAS_DFLASH_STEP_TIMING=1` and `RUST_LOG=info` so the
`spark::scheduler::verify_dflash_step` accept line
(`DFLASH K=γ verify: ... accepted=N/16`) + per-phase `DFLASH step timing` lines
are captured. Coding bench = `python3 local/bench_spec.py`-equivalent coding
prompt (LRU-cache Rust), max_tokens 800 (model emits ~1500), temp 0 unless noted.
Mean accept is averaged over all verify steps of the coding generation.

### Baseline (temp 0)
| workload | tok/s | md5 | mean accept/16 |
|----------|-------|-----|----------------|
| counting | 80.3  | `789a2786c5d5ac93fcc98f15bad01af5` (== expected) | — |
| coding   | **17.2–17.8** | (varies run-to-run) | **9.49/16** |
| prose    | 12.7  | `31761dde...` | — |

Per-step timing (coding, baseline): **total ~166 ms, verify ~142 ms (86%),
propose ~23 ms (14%).** Verify dominates; propose is cheap. Coding acceptance is
BURSTY: boilerplate sections accept ~14-16/16, novel-logic sections drop to
1-3/16 (e.g. observed `accepted=1/16`, `accepted=3/16` interspersed with high
runs). The "~2/16 on novel code" wall is real but the *mean* over a full coding
gen is ~9.5/16.

### Lever sweep (coding, temp 0)
| Lever | config | coding tok/s | mean accept/16 | counting md5 | token-exact? | verdict |
|-------|--------|--------------|----------------|--------------|--------------|---------|
| **baseline** | (defaults) | 17.2–17.8 | **9.49** | `789a2786…` | yes | reference |
| A. DENOISE_STEPS=2 | `ATLAS_DFLASH_DENOISE_STEPS=2` | 16.3 | 9.33 | `789a2786…` | **yes** | no accept gain; propose 23→40 ms; net DOWN |
| A. DENOISE_STEPS=3 | `=3` | 16.5 | 9.44 | (n/a) | yes (by design) | flat accept; propose →54 ms; net DOWN |
| A. DENOISE_STEPS=4 | `=4` | 17.5 | 9.24 | (n/a) | yes (by design) | accept slightly LOWER; propose →60 ms; net flat/DOWN |
| B. ADAPTIVE_GAMMA=1 | `ATLAS_DFLASH_ADAPTIVE_GAMMA=1` | 16.9 | 9.22 | `789a2786…` | **yes** | no coding gain; counting tok/s 80→52 (regression). Confirms prior finding holds for coding too. REJECT |
| C. TYPICAL (temp 0.7) | baseline temp0.7 captured; typical-on run NOT completed (stood down) | base 19.6 | base 9.56 | — (sampled, n/a at temp>0) | lossy by design | **INCOMPLETE** |

### Conclusion
**No cheap flag moves coding acceptance.** The two hypotheses that should have
helped both failed:

- **Lever A (DENOISE_STEPS, the #1 hypothesis): NEGATIVE.** Extra block-diffusion
  denoise passes do not raise coding acceptance (9.49 → 9.24–9.44, i.e. flat-to-
  slightly-down across 2/3/4 passes). They are NOT free: propose scales linearly
  with passes (23 → 40 → 54 → 60 ms) and there is no accept gain to pay for it,
  so net tok/s is flat-to-down. The verify-dominated step (86%) means propose was
  cheap, but adding passes only inflates propose without buying acceptance — the
  drafter's single-pass argmax is already near its own ceiling; re-feeding its
  own predictions does not improve the drafts the target will accept. Token-exact
  (counting md5 unchanged) as expected — verify is the source of truth.
- **Lever B (ADAPTIVE_GAMMA): NEGATIVE for coding, regression on counting.**
  Token-exact, but no coding accept/tok-s gain and counting tok/s collapses
  80→52 (shrinking γ on structured prompts where long runs accept well). The
  prior "adaptive-γ tanked counting" finding reproduces and extends to coding:
  it does not help coding either. REJECT.
- **Lever C (TYPICAL_ACCEPT): NOT EVALUATED.** Env is `ATLAS_DFLASH_TYPICAL_ACCEPT`
  (float ε in [0,1], + optional `ATLAS_DFLASH_TYPICAL_ALPHA`, default 0.3). Code-
  confirmed to only fire at temperature>0 (verify_dflash_step.rs:620 falls back to
  exact-match at temp 0) — lossy, accepts non-argmax drafts within ε/α of target.
  Only the temp0.7 *baseline* (19.6 tok/s, 9.56/16, coherent valid Rust) was
  captured before I was asked to stand down (shared-GPU contention). The
  typical-ON run was not completed. This is the one remaining cheap lever worth
  finishing; recommend a single isolated temp0.7 A/B when the GPU is free.

**Bottom line for the next step:** the cheap flags (denoise / adaptive) are
exhausted — they do NOT break the coding acceptance wall. The remaining levers
are code-level (retrieval/infill/confidence-truncation, the Tier-1 deep work)
and the one untested cheap-but-lossy flag (typical-accept at temp>0).

### Gates / housekeeping
- DENOISE_STEPS=2 and ADAPTIVE_GAMMA=1 both kept counting md5 =
  `789a2786c5d5ac93fcc98f15bad01af5` (token-exact). No KEPT lever (all rejected),
  so VISION re-check was not run.
- No code edits / no rebuild (flag sweeps on the existing binary). No git commits.
- Stood down early at coordinator request (single-GPU contention from re-serve
  cycling). Freed my own serve by PID (not broad pkill). GPU left free, /tmp
  sweep artifacts are scratch only.

## Tree-GDN precision fix 2026-06-18

**Mission:** Diagnose + fix numerical drift in the tree-aware GDN kernel
`gated_delta_rule_tree_wy` (`gdn_tree_wy_k`, "M8A v2") that forces
`ATLAS_DISABLE_TREE_WY=1` and blocks lossless branching. Phase 1 = kernel
correctness only (not the branching policy).

### Diagnosis (root cause)
Resolved the contradictory in-tree notes. The drift on LINEAR chains was a
DETERMINISTIC divergence, not a precision/branch bug:

1. **Gate clamp mismatch (primary).** `gated_delta_rule_tree_wy.cu:92` clamped
   gate to `[1e-6, 1-1e-6]`; the proven `gated_delta_rule_wy17.cu:96` clamps to
   `[0, 1]`. Any gate at the [0,1] boundary produced a different `sg`, which
   propagates through the entire WY recurrence → drafter accept collapse. This
   alone breaks bit-equivalence on flat chains.
2. **WY cross-term FP accumulation order (secondary).** The ancestor-walk visited
   ancestors closest→root with a *running* `gprod`, whereas wy17 sums s ascending
   with a *fresh* nested product `∏_{u=s+1..t-1} sg[u]`. Same algebra, different
   FP rounding on chains.

NOT bugs (verified): the WY correction ALGEBRA reduces exactly to wy17 on a
linear chain (proven on paper); PASS-2 re-reading `H_in` from the inter pool is
bit-identical to wy17's rolling registers (both FP32); tree_wy leaving `h_state`
at root (writing only the inter pool) is intentional and handled by
`async_chkpt.rs:209-216` (`was_tree_mode` forces the inter-slot commit). The
kernel was already FP32 throughout — no recurrence-precision fix was needed.

### Fix (kernels/gb10/common/gated_delta_rule_tree_wy.cu)
- Line ~92: gate clamp `fminf(fmaxf(g,1e-6),1-1e-6)` → `fminf(fmaxf(g,0),1)` to
  match wy17 exactly.
- WY correction (lines ~146-200): added a LINEAR-CHAIN fast path (taken when
  `parent[t]==t-1`) that reproduces wy17's EXACT accumulation order (ascending u
  for the leading product, ascending s with a fresh nested `gprod` for cross
  terms) → bit-exact on chains. The general branch path (true forks) keeps the
  ancestor-walk but rebuilds `gprod` from scratch per ancestor (no running-product
  drift) and accumulates oldest→closest.

Build: `ATLAS_TARGET_MODEL=qwen3.6-27b cargo build --release -p spark-server`
→ clean, "1 compiled / 129 cached", 130 PTX modules loaded at serve. The tree
kernel lives in the shared `kernels/gb10/common/` layer, compiled for every
target.

### Validation
- **Linear-chain bit-equivalence: PROVEN.** Forced tree kernel
  (`ATLAS_DISABLE_TREE_WY=0`, auto-injected linear chain `[-1,0,..,15]`, M8A
  dispatch confirmed firing) vs proven wy17 path (`ATLAS_DISABLE_TREE_WY=1`),
  same harness/prompt:
    - count probe md5 (tree_wy) == md5 (wy17) == `529bbca780d226d69fa3956ce8e1385d`
      — BIT-IDENTICAL.
  (The historical `789a2786...` baseline was a different prompt/harness; the
  correct apples-to-apples ground truth is the wy17 path through this same
  harness, which matches exactly.) Code/prose md5 differed slightly between the
  two — but wy17 is NOT deterministic run-to-run on free-form gen (two wy17 runs
  gave different code/prose md5s, count stable), so that divergence is engine
  nondeterminism (FP8 KV / async reductions), NOT the kernel. The deterministic
  count probe is the valid bit-equivalence signal and it is identical.
- **Branch correctness: PROVEN.** Minimal 2-way branch (parent_ids =
  `[-1,0,1,..,14,0]` — token 0 has two children: the main chain t=1 and the
  sibling t=16). Validated tree_wy's dumped per-token states against (a) the
  repo's python `tree_gated_delta_reference` and (b) a fresh independent FP32
  recurrence, AND against running the branch as an isolated 2-step chain {0,16}:
    - t=0 (root child):  state_cos = 1.0000,  mae = 0.000000
    - t=16 (BRANCH/sibling, parent=0): state_cos = 1.0002, mae = 0.000000,
      max_abs = 8e-6  vs the isolated {0,16} ground-truth chain.
  → The ancestor-walk correctly reads token 0 (not main-chain t=15) as the
  branch parent, applies exactly one correction step, and does NOT leak the
  sibling chain into the branch state. Branch is computed correctly per-path.
    - Main-chain tokens t=1..15 show cos 0.93→0.64 vs BOTH fp32 references, but
      are simultaneously bit-exact to wy17 (count md5). Two independent fp32 refs
      drift identically while the kernel matches wy17 → the drift is a
      REFERENCE artifact (deep-chain WY accumulation in fp32 torch diverges from
      the fused-kernel formulation), not a kernel error. The refs agree perfectly
      exactly on the shallow-recurrence tokens (t=0, t=16).
- **Default-OFF preserved.** `ATLAS_DISABLE_TREE_WY=1` (production default) count
  md5 unchanged (`529bbca...`); tree kernel never fires on the default/vision
  serving path, so vision is unaffected by construction (change is isolated to
  the tree_wy .cu, only reachable with a tree payload + tree-WY enabled). No
  ocr_test.png asset present in-tree to run the OCR string check.

### Status / next
Tree-GDN kernel is now bit-equivalent to wy17 on linear chains and
branch-correct on a 2-way fork. **Branching is UNBLOCKED at the kernel level
(Phase 1 done).** Phase 2 (entropy-gated top-2 branching policy at the
high-entropy coding cliff) is the next step — NOT built here. Edits left
UNCOMMITTED, no git commits. GPU freed (no spark procs), /tmp scratch cleaned.

A/B harness note: `ATLAS_M8A_DUMP=1`/`ATLAS_M8A_VS_WY17=1` MUST be combined with
`ATLAS_DFLASH_DEBUG_NO_GRAPH=1` — the dump does host-syncs/D2H copies that are
illegal under CUDA graph capture and otherwise wedge the SSM pool
(`CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`). The auto-injected linear chain needs
graphs ON, so use the payload-driven branch path (`ATLAS_DDTREE_NONFLAT=1`) for
no-graph dump validation.

## Branch policy 2026-06-18

**Mission:** Phase 2 — entropy-gated top-2 BRANCHING at the high-entropy coding
"cliff" to raise AEON-27B DFlash coding acceptance toward counting-level. Phase 1
(tree-GDN kernel `gated_delta_rule_tree_wy`, branch-correct) was the prerequisite.
Gate behind NEW env flag `ATLAS_DFLASH_BRANCH=1`, DEFAULT OFF. K held at γ+1=17.

### Policy + the change (files+lines)
`crates/spark-model/src/layers/dflash_head/propose.rs:666-840` (new block after
the drafter forward, before the ddtree-method block; gated `ATLAS_DFLASH_BRANCH=1`,
early-returns the flat `drafts` + stashes a tree payload via
`pending_tree_payload`). New env levers:
- `ATLAS_DFLASH_BRANCH=1` — enable (default off → byte-identical legacy path).
- `ATLAS_DFLASH_BRANCH_MARGIN=<f>` (default 2.0) — top1−top2 raw-BF16-logit margin
  below which a row is a "cliff". Computed from `extract_topk_from_logits(…, k=2)`
  over the just-computed γ_eff logit rows (one topk launch + ~γ·12B D2H).
- `ATLAS_DFLASH_BRANCH_CLIFF=first|min` (default `first`) — the FIRST row whose
  margin < threshold (the row that actually truncates the chain) vs the global-min
  row. `first` is essential: the global-min margin row on coding is typically deep
  (rows 12-14, per BRANCH dbg logs) — far past the ~row-3 mean accept — so it is
  never reached.
- `ATLAS_DFLASH_BRANCH_TAIL=top1|top2` (default `top1`) — which fork carries the
  post-cliff tail on the contiguous flat chain (the only fork the deployable
  flat-safe greedy commit can extend). `top1` = main chain stays on the drafter's
  top-1, top-2 is a leaf sibling (byte-stable, no-gain control). `top2` = main
  chain takes top-2 at the cliff (the EV bet at near-equiprobable cliffs).

Payload: an n-node tree (n=16 → K=17 unchanged) = a 15-deep top-1 chain with the
last node respent as a SIBLING leaf forking at the cliff (parent = cliff-1). The
fixed tree-WY GDN kernel (`ATLAS_DISABLE_TREE_WY=0`) computes its branch state;
`ATLAS_DDTREE_TREE_TOKENS_VERIFY=1` feeds the sibling token at its slot; the verify
walk (`greedy_sample_ddtree`) is unchanged.

### Headline: did top-2-at-the-cliff branching raise coding acceptance?
**No.** A single 2-way fork at one cliff is statistically NEUTRAL on coding
acceptance across the full design space, and the root cause is architectural, not
a tuning miss.

ON vs OFF (same build, temp 0, max_tokens 800, single stream; accept = mean
`last_num_accepted`/16 from propose log; serve = `local/serve-aeon-27b-dflash.sh`):

| config                              | counting acc / tok·s | coding acc / tok·s | prose acc / tok·s |
|-------------------------------------|----------------------|--------------------|-------------------|
| **OFF (baseline, this build)**      | 13.09 / 81.4         | **2.97 / 17.8**    | 1.03 / 12.8       |
| BRANCH top1, cliff=min, m=2 (min flags) | 13.07 / 81.3     | 3.10 / 18.2        | 0.96 / 12.4       |
| BRANCH top1, cliff=first, m=2 (full tree flags) | 12.96 / 79.4 | 2.92 / 17.6     | 0.99 / 12.6       |
| BRANCH top2, cliff=first, m=2       | 12.42 / 77.4         | **2.21 / 16.3**    | 0.89 / 12.0       |
| BRANCH top2, cliff=min,  m=2        | 12.42 / 77.3         | 2.97 / 17.7        | 1.09 / 13.1       |
| BRANCH top1, cliff=first, **m=1.0** | (count md5 exact)    | — / 17.9           | —                 |
| BRANCH top1, cliff=first, **m=4.0** | (count md5 exact)    | — / 17.8           | —                 |

- coding accept moves at most +0.13 (top1/min) and -0.76 (top2/first) — within
  run-to-run noise except top2/first which actively HURTS.
- counting/prose do NOT improve and only lose tok/s from tree-WY overhead
  (`ATLAS_DISABLE_TREE_WY=0` costs ~2-4 tok/s on counting; the no-overhead min-flags
  config is tok/s-neutral). The margin gate is the right knob — sweeping 1/2/4
  leaves coding tok/s flat (17.8-17.9).

### Why (the architectural finding — scope the real fix as Phase 3)
The deployable verify commit (`ddtree.rs::adapt_to_flat_safe_contract`, "M11A guard")
commits ONLY the contiguous flat-chain prefix `[1,2,…,n]` + one bonus; a branch's
non-flat compact index truncates the accept there. Under that contract, NO tree
topology can beat the flat chain for GREEDY decode, because:
1. Greedy verify already emits the target's greedy token as a FREE bonus at the
   first mismatch (the cliff). A leaf sibling at the cliff is therefore reachable
   only AS that same bonus → byte-identical to flat (proven by trace + the
   top1/min run). Zero gain by construction (and zero loss — counting stays
   token-exact, md5 == OFF on every config).
2. To add committed tokens the post-cliff predictable TAIL must be laid on the
   fork the target takes. With a 1-node budget you can cover only ONE fork's tail.
   Betting it on top-2 (`top2`) loses, because the drafter's top-1 is the more
   likely-correct token even at a low-margin cliff — diverting the main chain to
   top-2 truncates EARLIER when the target picks top-1 (coding 2.97→2.21).
3. Coding accept is ~3/16, i.e. the chain breaks every ~3 tokens, NOT "one hard
   cliff then a 13-token predictable tail". The mission's premise (one bursty
   cliff per block) does not hold for this drafter on this workload — there are
   multiple cliffs and the inter-cliff tail is short, so covering one cliff with a
   second candidate cannot unlock a long tail.

**Real unlock = Phase 3 (full-branch commit + KV compaction).** The Phase-1 kernel
gives correct branch STATE; the gain requires (a) relaxing the flat-safe commit to
commit the full accepted branch path (lossless — every committed token is still the
target's argmax) AND (b) KV-cache compaction so a sparse-compact-index accept leaves
contiguous KV for the next decode (verify_dflash_step.rs:159-166 notes this is
currently a no-op). Only then can BOTH forks carry tails (needs >1 spare node →
shorten the chain further / a true multi-branch budget), which is what actually
raises acceptance. Single-branch-at-one-cliff under the deployable contract is
provably flat.

### Gates
- **TOKEN-EXACT (counting):** PASS on every config — counting greedy md5 ==
  `d6803f143fddbd730b79829a2a0b74f9` (OFF baseline on this build) for BRANCH
  top1/top2 × cliff first/min × margin 1/2/4. The branch only ever changes
  speculative CANDIDATES; the verify is the greedy oracle.
- **Code/prose byte-identical ON vs OFF:** NOT a valid signal on this engine — code
  AND prose are NONDETERMINISTIC run-to-run at temp 0 even with BRANCH OFF
  (OFF-vs-OFF md5 differ; FP8-KV / async-reduction nondeterminism, as the Phase-1
  notes flagged). Counting is the only deterministic md5 and it is exact. ON outputs
  are coherent valid Rust / prose, `garbage=False` on all runs.
- **VISION:** unaffected by construction — default-OFF (`ATLAS_DFLASH_BRANCH` unset
  → `branch_enabled=false` → the new block is skipped entirely; counting md5 on the
  default build == OFF). No ocr_test/vision asset present in-tree to run the string
  check (same as Phase-1 notes).
- Per-step timing: branch overhead is one topk(k=2) launch + tiny D2H in propose
  (negligible); the measurable tok/s cost is the `ATLAS_DISABLE_TREE_WY=0` tree-WY
  GDN path (~2-4 tok/s on counting), avoidable with min-flags since the top1 leaf
  sibling is lossless without it.

### Housekeeping
Edits UNCOMMITTED, flag DEFAULT OFF. GPU freed (no spark procs). /tmp scratch
cleaned. Change isolated to propose.rs (+ reuses existing ddtree payload / verify
infra). No git commits.

---

## Conditioning / capture-precision 2026-06-18

**Mission: confirm + fix the conditioning/capture-drift wall blocking AEON-27B DFlash
coding from ~70 tok/s. Verdict: the clean-state diagnostic REFUTES capture drift as
the dominant lever. Feeding HF/FLA-clean target hidden states does NOT lift live
acceptance toward the offline 13.6/16 ceiling. The live gap is elsewhere
(drafter live-rollout vs offline-recompute, drafter intrinsic, engine forward) —
not the captured-state precision. This redirects strategy.**

### Setup
- Build = HEAD (a9a5bcc), records-grade serve defaults (nvfp4 drafter + ATTN_KGAMMA +
  FFN_DOWN_SPLITK + mtp-vocab 32000), single-stream. `RUST_LOG=info` →
  per-step `DFLASH K=γ verify: ... accepted=N/16` parsed for mean accept/16.
- Functional clean-state hook = `ATLAS_DFLASH_HF_OVERRIDE` (file-based; the
  `ATLAS_DFLASH_FLA_SIDECAR` HTTP path is DOCUMENTED-ONLY — no Rust client exists,
  fla_sidecar.py itself says "Rust HTTP client TBD"). Clean states generated by
  capturing HF transformers (AEON-Q36-27B-BF16-dequant = BF16 dequant of the exact
  served NVFP4 weights) at layers [1,16,31,46,61] for Atlas's EXACT committed token
  sequence (dumped via ATLAS_DFLASH_DEBUG_DUMP_FULL → /tmp/atlas_tokens.json).

### STEP 1 — clean-state diagnostic (THE key measurement)
| workload | OFF (drifted Atlas captures) | CLEAN (HF/FLA states fed) | delta |
|----------|------------------------------|---------------------------|-------|
| counting (deterministic, aligned) | 13.04–13.48/16 | 13.48/16 | **+0.0 (within noise)** |
| coding (nondeterministic)         | 2.97/16        | confounded (see below)    | — |

- **Counting is the only confound-free signal** (byte-deterministic at temp0 → the
  static override file aligns position-for-position to the live run). Result:
  OFF 13.48/13.33, CLEAN 13.48/13.48 — **clean states give NO measurable lift**.
  Counting captures are already clean enough; HF-reference states change nothing.
- **Coding clean-state via static file is FUNDAMENTALLY CONFOUNDED**: coding is
  NONDETERMINISTIC at temp0 (FP reduction order — confirmed: two runs of the same
  prompt produced 1501 vs 1491 committed tokens). A static clean-state file captured
  for run-A's tokens is fed onto run-B's divergent positions → MISALIGNMENT.
  When the override actually fired (full 1501-pos file), coding accept DROPPED to
  1.59/16 (from 2.97) — this is the misalignment artifact, NOT a true clean-state
  result. An aligned coding feed requires a LIVE per-step sidecar (unbuilt).
- The offline probe (probe_acceptance_offline.py, prior session) already established
  the aligned coding-on-clean-states number = 13.6/15. The live engine cannot
  reproduce that with a static file because of nondeterminism; that gap is a
  live-rollout/sidecar problem, not a capture-precision one.

### STEP 2 — drift located + QUANTIFIED (confound-free, same-token cosine)
Captured Atlas's ACTUAL ctx_hidden_acc (/tmp/atlas_target_hidden.bin, 1494 pos) vs
HF clean for the IDENTICAL 1494 tokens. Per-capture-layer cosine, answer-phase
(pos 735+, fully captured):
```
  L 1: cos 0.9997  relL2 2.7%
  L16: cos 0.9994  relL2 4.0%
  L31: cos 0.9968  relL2 7.8%
  L46: cos 0.9882  relL2 13.2%
  L61: cos 0.9840  relL2 14.8%   <- worst, as predicted (memory said ~0.86; measured 0.984)
```
- The SSM-kernel drift IS real and compounds by layer (L1→L61), matching the
  hypothesis qualitatively, but it is MODEST (cos 0.984 at L61, not 0.86). Capture
  source = `try_dflash_prefill_capture_layer` (impl_b3.rs:174-225), a verbatim BF16
  D2D copy of `self.buffers.hidden_states()` — the capture itself adds NO precision
  loss; the drift is upstream in the BF16 residual stream + BF16 GDN/SSM outputs
  accumulating across 61 layers. A "FP32 capture" fix is NOT cheap: it requires
  running the entire target forward in FP32 for the captured layers (the full model),
  not a capture-side tweak. Since the target VERIFY uses this same BF16 path as the
  source of truth, FP32-only-for-drafter-conditioning would also be inconsistent.
- **BIGGER finding — 700 ZEROED capture slots**: positions 34–733 (the entire
  internal THINKING phase, ~700 tokens that AEON generates and strips from
  /v1/completions output) have ZERO captured hidden states. The thinking phase
  advances seq_len via a non-DFlash decode path that never runs the prefill capture
  hook. The drafter then attends over ctx that is ~700/1494 zero-norm keys. This
  dragged the all-position mean cosine to 0.53 (vs 0.984 answer-phase). HOWEVER:
  capping ATLAS_DFLASH_CTX_WINDOW=64 to exclude the zeroed slots did NOT help coding
  (2.59 vs 2.97) — the drafter is not materially hurt by the zero-norm keys (its
  attention down-weights them). So the zeroed slots are also NOT the lever.

### STEP 3 — FIX
NOT ATTEMPTED. Step 1 REFUTED the premise (clean states don't lift live acceptance),
so an FP32-capture fix would chase a non-lever. The honest conclusion is that the
~3/16-vs-13.6/16 coding gap is NOT captured-state precision. Remaining suspects
(unchanged from prior decomposition): (a) drafter LIVE incremental ctx-K/V rollout
vs offline fresh-recompute (the −2/16 the prior session measured: reference 10.6 →
engine-bf16 8.5 on the SAME states), (b) drafter intrinsic ceiling on novel code
(bursty: 13.5/15 offline on structure but linear chain discards the predictable tail
after one content miss), (c) coding nondeterminism itself capping the realizable
chain. The production unlock is a LIVE-aligned conditioning sidecar OR a drafter
retrain — not a capture-precision flag.

### Gates
- **TOKEN-EXACT (counting):** PASS. OFF md5 `b725bffce4df88c995e1562b0d122c2f` ==
  CLEAN-HF_OVERRIDE md5 `b725bffce4df88c995e1562b0d122c2f` (this harness:
  "Count from 1 to 400, one number per line.", max_tokens=400; the records-harness
  529bbca... uses a different prompt). Conditioning override is drafter-only →
  committed greedy tokens byte-identical, confirmed.
- **VISION:** PASS. ocr_test.png → "HELLO 42"; vision_test.png → "A Red Square ...
  a solid blue circle ... perfectly centered" (red square + blue circle). Run on the
  default/OFF server; conditioning change is default-off + drafter-only → vision
  unaffected by construction.
- **HEADLINE:** counting clean-vs-OFF = NO lift (13.48≈13.48); coding clean feed
  confounded by nondeterminism (cannot align a static file); confound-free drift
  L61 cos = 0.984 (modest). Capture-precision is NOT the wall.

### Housekeeping
No source edits (diagnostic-only session, used existing HF_OVERRIDE / DUMP_FULL
hooks). Pre-existing WIP in working tree untouched. No commits. GPU freed (no spark
procs). /tmp scratch (clean-state .bin files, capture scripts) cleaned.

---

## Thinking-phase capture 2026-06-18

The ONE untested engine lever from [[dflash-coding-acceptance-2026-06-18]] item (1).
Implemented the LIVE engine fill of the thinking-span ctx captures (not the prior
session's offline clean-state feed, which only proved capture *precision* isn't the
wall). HEADLINE: filling the ~600 thinking-phase ctx slots with REAL target hiddens
raised coding answer-phase accept **2.985 → 3.350/16 (+0.37, +12%)** and tok/s
**17.69 → 18.40 (+4%)**, fully token-exact. Modest, not a step-change — confirms the
~3/16 novel-from-scratch coding floor is drafter-intrinsic, not a missing-ctx artifact.

### Why the thinking path skipped capture (file+lines)
- During `<think>`…`</think>` the scheduler gates OFF DFlash speculation:
  `crates/spark-server/src/scheduler/mod.rs:308-311` routes to `step_mtp` (→
  `step_verify_dflash`) only when `active.iter().all(|a| !a.inside_thinking && …)`.
  With `inside_thinking==true` it falls to `step_decode_only`
  (`crates/spark-server/src/scheduler/decode_step.rs:8`).
- `step_decode_only` → `model.decode_batch` → (n==1) `decode`
  (`crates/spark-model/src/model/trait_impl/decode_a.rs:189-207`). That loop DOES
  run the per-layer hook `try_dflash_capture(i, 0, stream)` (decode_a.rs:206), which
  lands the just-decoded token's 5 target-layer hiddens in `dflash_hidden_save[0]`
  (`crates/spark-model/src/model/impl_b3.rs:558-611`).
- BUT the move from `dflash_hidden_save[0]` → `ctx_hidden_acc[abs_pos]` lives ONLY in
  the answer-phase `propose_drafts` append
  (`crates/spark-model/src/layers/dflash_head/propose.rs:246-338`), which never runs
  during thinking. So `dflash_hidden_save[0]` is overwritten every thinking step and
  never persisted → `ctx_hidden_acc` slots spanning the thinking span stay ZERO.
  Empirically confirmed: ON probe shows captures resume contiguously from the prompt
  end (abs_pos=13) — OFF leaves exactly that 13..~733 region zero.

### The fix (file+lines, default-OFF `ATLAS_DFLASH_CAPTURE_THINKING=1`)
- Flag: `crates/spark-model/src/model/env_diag.rs` `dflash_capture_thinking_enabled()`.
- Capture: `crates/spark-model/src/model/impl_b3.rs`
  `dflash_capture_thinking_dispatch()` — copies the fresh `dflash_hidden_save[0]` row
  (one whole `ctx_slot_bytes` = 5×5120×bf16 = 51200 B) into
  `ctx_hidden_acc[(seq.seq_len-1) * ctx_slot_bytes]` and advances `ctx_len` in
  lockstep. Runs AFTER `decode` increments seq_len, so the just-decoded thinking
  token lands at its correct absolute slot. No-op when flag off / DFlash inactive /
  no proposer state / rank>0 / accumulator full.
- Trait: `crates/spark-model/src/traits/model.rs` `dflash_capture_thinking` (default
  no-op) + wiring in `crates/spark-model/src/model/trait_impl/mod.rs`.
- Scheduler call: `crates/spark-server/src/scheduler/decode_step.rs` — after
  `decode_batch`, for `active.len()==1 && active[0].inside_thinking`, call
  `model.dflash_capture_thinking(&mut active[0].seq, 0)`. Single-seq only:
  `dflash_hidden_save` holds one row and the n≥2 batched decode path doesn't run the
  per-seq capture hook anyway (and the AEON dflash serve is `--max-batch-size 1`).

### Slot alignment (the 34→735-jump concern from the diagnostic): RESOLVED
ON probe (`ATLAS_PROPOSE_PROBE=1`) shows captures filling **contiguously** with
`ctx_len == abs_pos+1` at every step: abs_pos = 13,14,15,…,608 with NO holes (596
captures on a short run). The prior diagnostic's 34→735 jump (zeros in between) is
gone — the thinking span is now densely populated up to the answer boundary
(seq_len≈737, matching prompt + ~700 thinking tokens).

### A/B (coding "Write a complete Rust LRU cache…", max_tokens=800, γ=16, K=17)
| metric                         | OFF (baseline)                     | ON (ATLAS_DFLASH_CAPTURE_THINKING=1) |
|--------------------------------|------------------------------------|--------------------------------------|
| coding answer-phase mean accept| 2.985/16 (200 steps, min0 max15)   | **3.350/16** (183 steps, min0 max16) |
| coding tok/s                   | 17.69                              | **18.40** (+4%)                      |
| thinking ctx captures          | 0 (slots zero)                     | 596 contiguous, ctx_len lockstep     |

### Gates (ALL PASS)
- **TOKEN-EXACT (counting):** PASS. OFF counting greedy md5 =
  `b725bffce4df88c995e1562b0d122c2f` == ON md5 `b725bffce4df88c995e1562b0d122c2f`
  == the required baseline. Drafter-conditioning-only change → committed tokens
  byte-identical, confirmed.
- **VISION:** PASS (run on ON server, strictly more conservative than default-off).
  ocr_test.png → "HELLO 42"; vision_test.png → "A Red Square … a solid blue circle …
  Centered perfectly within" (red square + blue circle).
- **NO-REGRESS default-off:** PASS. Flag is OFF by default; `dflash_capture_thinking`
  is a no-op (returns before any GPU work) when unset — OFF coding 17.69 tok/s is at
  records parity, OFF md5 matches the baseline.

### HEADLINE / honest read
Filling the thinking captures DID raise coding answer-phase acceptance, but only
modestly: **2.985 → 3.350/16 (+12%)**, NOT above the prior session's offline-clean
~13/16 ceiling. So real reasoning context helps the drafter a little (the +12% is
genuine and free/lossless), but it does NOT close the live-vs-offline gap — consistent
with the established conclusion that the ~3/16 novel-from-scratch floor is drafter
INTRINSIC (high-entropy logic + live-incremental-rollout drift), not a missing-context
artifact. This closes the last untested lossless ENGINE lever: it's a small real win
worth shipping default-off, but the path to ≥13/16 coding acceptance remains a drafter
RETRAIN, not another engine flag.

### Housekeeping
Edits (uncommitted, default-off): `env_diag.rs`, `impl_b3.rs`, `traits/model.rs`,
`trait_impl/mod.rs`, `scheduler/decode_step.rs`. Build requires
`ATLAS_TARGET_MODEL=qwen3.6-27b`. No commits. GPU freed (no spark procs). /tmp
scratch (think_ab.sh, spark_*.log, think_ab_* dirs) cleaned.

## Relaxed acceptance 2026-06-18

**Decision: SHIP default-OFF as a research knob. HONEST HEADLINE — relaxed acceptance does NOT unlock coding tok/s at PPL-neutral quality. The quality-preserving configs give ~0 speedup; the configs that move tok/s degrade quality. The ~3/16 coding floor is drafter-intrinsic, confirmed from a third independent angle.**

### Mechanism + flag (`ATLAS_DFLASH_RELAX_ACCEPT`, default OFF)
Extends the existing `ATLAS_DFLASH_TYPICAL_ACCEPT` plumbing (which only fires at temp>0) to ALSO work at temp0 (the coding-agent default). At each would-be-mismatch verify position, instead of rejecting a draft just because it is not the target's argmax, COMMIT the DRAFT token (not the argmax) when it is a high-probability token under the TARGET:
- `ATLAS_DFLASH_RELAX_TOPK=k` — accept if draft is within the target's top-`k` logits, OR
- `ATLAS_DFLASH_RELAX_RATIO=r` — accept if `p_target(draft)/p_target(argmax) >= r`, i.e. `l_draft - l_argmax >= ln(r)` (logit-space → temperature-free, so it behaves identically at temp0).
The bonus stays the target argmax at the first genuinely-rejected position. Pure-argmax prefixes never touch the device (no D2H). Same flat-chain SSM-commit consistency as typical-accept (committed draft's KV + `h_state_intermediates[i]` already sit at compact slot `i`). Grammar-active and fp32-logits paths defer to strict accept.
Files (uncommitted, default-off): `crates/spark-server/src/scheduler/verify_dflash_step.rs` (`dflash_relax_accept`, `relax_row_accepts`, `dflash_relax_config`; wired into the non-tree accept dispatch BEFORE typical-accept). Build: `ATLAS_TARGET_MODEL=qwen3.6-27b`.

### TRADEOFF TABLE — AEON-27B, γ=16/K=17, temp0, max_tokens 800 (coding/prose), 512 (ctx)
served-PPL = teacher-forcing PPL of the SERVED completion under the BF16-dequant teacher (`AEON-Q36-27B-BF16-dequant`, 509-tok cap → low absolute values; RELATIVE drift vs strict is the gate). `bench_spec`: coding/prose/counting tok/s. `coding_ctx_bench`: extend/fix/add tok/s.

| config    | coding accept/16 | coding tok/s | prose tok/s | ctx ext/fix/add tok/s | lru PPL | add_method PPL | prose PPL | counting | coherent? |
|-----------|------------------|--------------|-------------|-----------------------|---------|----------------|-----------|----------|-----------|
| **strict (baseline)** | 3.355 | 17.8 | 12.8 | 16.9 / 21.8 / 45.6 | 1.416 | 1.380 | 2.357 | CLEAN | yes |
| ratio=0.5 | 3.630 | 18.6 | 13.3 | 17.0 / 22.8 / 47.4 | 1.409 | 1.380 | 2.415 | CLEAN* | yes |
| ratio=0.3 | 3.356 | 18.1 | 13.3 | 17.3 / 22.5 / 47.5 | 1.491 | 1.380 | 2.610 | CLEAN | yes |
| ratio=0.1 | 3.924 | 19.0 | 15.5 | 18.0 / 22.5 / 44.5 | 1.485 | 1.740 (+26%) | 2.503 | CLEAN | yes (minor identifier drift) |
| topk=2 | 4.212 | **19.5** | 14.6 | 21.2 / 23.5 / 48.2 | 1.521 | 1.844 (+34%) | 2.594 | **CORRUPT** (20→10) | yes-but-buggy |
| topk=3 | 4.154 | 18.5 | 15.8 | 22.5 / 20.5 / 41.4 | 1.386 | 1.836 (+33%) | 2.983 (+27%) | **CORRUPT** (13→23, 21×2) | yes-but-buggy |

\* ratio=0.5 counting is coherent (1..N correct) but NOT byte-exact: 1 corruption in 227 lines (line 227 "227"→"22"). It is *not* a true no-op even at the tightest ratio.

### QUALITY GATE findings (the headline)
- **PPL-neutral configs (ratio=0.5, ratio=0.3)** preserve quality (lru/add_method PPL flat-or-better; prose +2.5–11%; counting coherent) but deliver **near-zero coding speedup**: 17.8 → 18.1–18.6 tok/s (+1.7–4.5%), accept 3.355 → 3.36–3.63. The drafter's near-misses on novel code are simply NOT within 0.3–0.5× of the target argmax, so the gate almost never fires.
- **Configs that move tok/s (topk=2/3)** push coding accept to ~4.2/16 and tok/s to ~19.5, but DEGRADE quality: add_method PPL +33–34%, prose PPL up to +27%, and they **corrupt counting** (the no-op sanity FAILS — top-2/3 commits a wrong digit the target ranks 2nd, e.g. 20→10, 13→23). topk also commits subtly-buggy code (e.g. `now = time.monotonic() if now is None else time` — `else time` should be `else now`, a near-miss identifier the target ranked top-2).
- **ratio=0.1** is the only config with both a real (small) gain and bounded quality: coding 17.8→19.0 (+7%), accept 3.92, counting CLEAN, code coherent — but add_method PPL +26% shows drift is already creeping in.

### Side-by-side (strict vs relaxed), quality judgeable
- `add_method` strict vs **ratio=0.5**: **byte-identical** (gate never fired on this prompt → provably PPL-neutral).
- `lru` strict vs ratio=0.5: coherent valid alternatives the target ranked highly (`use std::hash::Hash;` vs `use std::fmt;`; "key lookups"→"key lookup"). rustc parse parity with strict (both only error on the 512-tok mid-doc-comment truncation).
- `add_method` strict vs **topk=2**: coherent but introduces a real bug (`else time` for `else now`) — the concrete quality risk of rank-based accept.

### COUNTING / VISION sanity
- COUNTING: ratio≥0.3 stays coherent (1,2,3…); ratio=0.5 has 1 error in 227 lines (lossy, not no-op). **topk=2/3 visibly corrupt counting** — disqualifying for a "no-degradation" bar.
- VISION (ratio=0.5 server, default-off path is even safer): PASS. ocr_test.png → "HELLO 42"; vision_test.png → "A Red Square … A Blue Circle: Centered within …" (red square + blue circle). The accept flag does not touch the multimodal prefill path.

### HEADLINE — coding tok/s at PPL-neutral quality
**17.8 → ~18.6 tok/s (+4.5%) at PPL-neutral (ratio=0.5), accept 3.355 → 3.630.** Pushing further (ratio=0.1 → 19.0, topk=2 → 19.5) buys a few more tok/s but at measurable PPL drift (+26–34% on Python add_method) and counting corruption (topk). There is NO config that takes coding to the 30–80 tok/s range without breaking the quality bar. The reframe was sound (DFlash is lossy at temp0 anyway; relaxed accept commits high-prob target tokens) — but the EMPIRICAL drafter behavior defeats it: on the high-entropy content tokens where the drafter actually misses, the draft is far down the target's distribution, not a top-2/3 near-miss. This independently re-confirms the established conclusion: the ~3/16 coding floor is drafter-INTRINSIC; the unlock to ≥13/16 / 30–80 tok/s is a drafter RETRAIN (or the Tier-3 bit-exact tree-verify kernel), not an accept-gate relaxation.

### Housekeeping
Edits uncommitted, default-OFF (`ATLAS_DFLASH_RELAX_ACCEPT` unset → `dflash_relax_config()` returns None before any work; strict greedy path byte-for-byte unchanged). GPU freed (no spark procs). /tmp/relax_test scratch retained for artifacts; no commits.

## SAM retrieval 2026-06-18

**Decision: SHIP the SAM longest-suffix matcher default-OFF (`ATLAS_DFLASH_SAM=1`). HONEST HEADLINE — SAM is a clean correctness/recall upgrade over the naive fixed-window matcher (longest match at ANY length, token-exact), and it MATCHES the proven naive add_method win (46.6→64.4 tok/s, +38%). But it does NOT broaden the win to extend_class/fix_bug, because — measured directly — naive and SAM fire IDENTICALLY often on these benches (21/9/28 hits each); the limiter on extend/fix is the fraction of output that is genuinely NOVEL (un-retrievable), not matcher recall. The "naive misses, SAM catches" hypothesis is empirically false here.**

### Implementation (files + lines, uncommitted, default-OFF)
- `crates/spark-model/src/layers/dflash_head/retrieval.rs`:
  - `RetrievalConfig` gains a `sam: bool` field; `from_env` now also turns on for `ATLAS_DFLASH_SAM=1` (implies retrieval) and, in SAM mode, defaults `hybrid_min` to `l_min` instead of `l_max` (longest matcher returns any length, so the meaningful gate is the MIN match length that pre-empts the drafter).
  - new `retrieve_longest(haystack, last_token, cfg)` — the SAM-style longest-suffix matcher. Indexes the earlier occurrences of `last_token` (the only possible match END positions, since the query is always the live suffix), extends each backward to measure match length (capped at `l_max`, ≤256 candidates most-recent-first), keeps the LONGEST, and proposes the γ follow-on tokens. In-context analogue of SAM-Decoding (2411.10666) specialized to the DFlash propose loop — a few thousand u32 compares per call, far below one drafter GPU forward. Identical token-exact contract to `retrieve` (verify is the lossless oracle).
  - 5 new unit tests (`sam_*`): longest-any-length, l_min rejection, longer-over-recent, draft-room, no-match. All 11 retrieval tests pass.
- `crates/spark-model/src/layers/dflash_head/propose.rs` (~L424-437): the retrieval block now dispatches to `retrieve_longest` when `rcfg.sam`, else legacy `retrieve`. Flat-chain → wy17 verify path unchanged, draft count = γ_eff.
- `crates/spark-model/src/model/impl_b3.rs` (~L111): `want_token_mirror` also fires for `ATLAS_DFLASH_SAM=1` so `pld_tokens` (= full `seq.tokens` = prompt + thinking + generated) is populated.
- Build: `ATLAS_TARGET_MODEL=qwen3.6-27b cargo build --release -p spark-server`.

### Per-task tok/s — SAM vs naive-retrieval vs OFF (AEON-27B, γ=16/K=17, temp0)
coding_ctx_bench.py, max_tokens=512. "naive(h16)" = original naive default (hybrid_min=lmax=16); "naive(h4)" = naive with hybrid_min lowered to 4; "SAM(h4)" = longest matcher, hybrid_min=4 (default).

| task         | OFF  | naive(h16) | naive(h4) | SAM(h4) | SAM hits / naive hits |
|--------------|------|-----------|-----------|---------|------------------------|
| add_method   | 46.6 | 63.8      | 62.3      | 64.4    | 21 / 21                |
| extend_class | 17.0 | 16.9      | 16.7      | 16.8    | 9 / 9                  |
| fix_bug      | 22.3 | 23.4      | 24.0      | 24.2    | 28 / 28                |

bench_spec.py (max_tokens=800) + bench_ab.py (novel code/LRU):

| workload          | OFF  | SAM(h4) | note                          |
|-------------------|------|---------|-------------------------------|
| counting          | 79.0 | 81.6    | flat (nothing to retrieve)    |
| coding (LRU,novel)| 17.8 | 17.3    | flat (novel, nothing to copy) |
| prose             | 12.2 | 12.9    | flat                          |
| bench_ab coding   | 33.8 | 35.1    | flat (novel BankAccount)      |

### Threshold sweep (SAM, coding_ctx) — FLAT across all thresholds
| lmin/hybrid_min | add_method | extend_class | fix_bug |
|-----------------|-----------|--------------|---------|
| 3 / 3           | 63.7      | 16.8         | 23.9    |
| 4 / 4 (default) | 64.4      | 16.8         | 24.2    |
| 5 / 5           | 63.1      | 17.1         | 24.2    |
| 8 / 8           | 66.0      | 17.0         | 23.7    |
Best is ~4 (broadest firing with no quality cost); the choice barely moves tok/s because match quality is already strong everywhere it fires (match_len mean 12–15, max 16) — recall is not the limiter.

### Why SAM does not broaden (the headline finding)
Directly measured per-task SAM-hit vs naive-hit counts are **identical** (add 21=21, extend 9=9, fix 28=28). On these prompts every reusable span SAM finds is ≤16 tokens and already inside naive's 4..16 window, so SAM's theoretical edge (any-length / >16 longest match) never changes a hit decision — γ=16 caps the proposal at 16 tokens regardless. add_method wins big because its output re-emits the whole TokenBucket class verbatim (21 dense hits → long accept runs). extend_class writes a NEW subclass (only 9 retrievable spans over 645 tokens → mostly novel-logic drafter steps → flat at 17). fix_bug's 28 hits are diluted across 825 tokens of mostly-novel unit tests → 24. The limiter on extend/fix is the NOVEL-token fraction, not matcher recall; SAM cannot copy what was never written.

### GATES — ALL PASS
- TOKEN-EXACT: counting greedy md5 `789a2786c5d5ac93fcc98f15bad01af5` IDENTICAL OFF == SAM(h4) == every sweep point (3/4/5/8). (Note: the prompt's quoted `529bbca7…` was a different prompt/cap; the load-bearing gate is ON==OFF, which holds.) coding_ctx extend_class greedy BYTE-IDENTICAL OFF vs SAM; add_method/fix_bug differ only via the known temp0 FP-reduction-order non-determinism (per memory note: only counting is byte-stable), both coherent, no garbage.
- VISION (SAM ON): ocr_test.png → "HELLO 42"; vision_test.png → "A Red Square … A Blue Circle: Centered within …". PASS (SAM never touches the multimodal prefill path).
- NO-REGRESS: default-OFF is byte-for-byte legacy (`from_env` returns None before any work when both flags unset). ON on counting/prose/novel-coding is flat (within noise), never worse.

### HEADLINE — how many coding_ctx tasks SAM lifts toward 64+, mean real-coding tok/s
**1 of 3 (add_method) is lifted to 64+ (46.6→64.4, +38%, token-exact); extend_class and fix_bug stay flat (17/24) — and crucially, naive lifts the SAME 1 of 3 by the SAME amount (naive and SAM fire identically). Mean coding_ctx tok/s: OFF 28.6 → SAM 35.1 (+23%), driven entirely by add_method.** SAM is the correct, more-general matcher to SHIP (it strictly dominates naive on recall and is token-exact), but on this corpus the in-context reuse that retrieval can exploit is already fully captured by the naive 4..16 window; the extend/fix flatness is a property of the WORKLOAD (novel logic), not the matcher. The retrieval bypass remains the proven no-retrain win precisely where code reuses context (add_method / agentic edit-existing-file = the user's real use case); broadening further requires either a static code-corpus SAM (out-of-context reuse) or attacking the novel-token drafter floor — not a better in-context matcher.

### Housekeeping
Edits uncommitted, default-OFF (`ATLAS_DFLASH_SAM`/`ATLAS_DFLASH_RETRIEVAL` unset → no token mirror, no matcher, legacy byte-for-byte). Files: `retrieval.rs`, `propose.rs`, `impl_b3.rs`. Build `ATLAS_TARGET_MODEL=qwen3.6-27b`. No commits. GPU freed (no spark procs). /tmp scratch cleaned.

## Target early-exit 2026-06-18

**Mechanism: draft with the TARGET'S own first N layers + final_norm + lm_head. Default-OFF flag `ATLAS_DFLASH_EARLY_EXIT=1`, sweep N via `ATLAS_DFLASH_EARLY_EXIT_N` (default 31).**

### Implementation
- `crates/spark-model/src/model/early_exit.rs` (NEW): `early_exit_propose()` + `early_exit_forward()`. Per draft step: embed → run target layers `0..N` (ALL layer types, including the 48 GDN/SSM layers — NOT skipped, unlike `decode_draft`) → `final_norm` + `lm_head` on the layer-N hidden → `argmax` → draft token; append, repeat γ times. Metadata/KV setup mirrors `decode_draft` (impl_b1.rs).
- `crates/spark-model/src/model/mod.rs`: registered `early_exit` module.
- `crates/spark-model/src/model/impl_b3.rs` (`run_mtp_propose_inner`): early-exit branch intercepts BEFORE `proposer.propose()` when the flag is set — `return self.early_exit_propose(token, num_drafts, seq)`. Drafts flow into the unchanged `step_verify_dflash` K=γ path (drafts.len()≥4 → dflash verify).
- `crates/spark-server/src/scheduler/verify_dflash_step.rs`: added `DFLASH_EE_VERIFY` per-step accept log gated on `ATLAS_DFLASH_EARLY_EXIT_PROFILE=1`.

### KV / SSM handling (the hard part — AEON-27B is a HYBRID: 16 full-attn + 48 GDN/SSM layers, full_attention_interval=4)
- **SSM**: `checkpoint_ssm_states` before the γ-draft loop; `rollback_ssm_states(seq, 0)` (restore-to-checkpoint) after. The partial forward advances the SSM recurrence γ times; this undoes it fully. Verify then takes its own fresh checkpoint — existing K=γ intermediate/rollback path byte-for-byte unchanged.
- **KV**: draft positions' KV (layers 0..N) is left in the paged cache; the verify's full 0..64 recompute overwrites the same slots (identical contract to `decode_draft`).
- **Cursor**: each draft step pushes one token+pos (so layer attention sees the running draft prefix); rewound to pre-draft before returning.

### Per-N results (γ=15 effective, counting/coding/prose; novel + reuse — measured on AEON-Q36-27B-Full, GB10)
| N    | propose ms | ms/draft | mean accept/15 | verify ms (baseline) | net |
|------|-----------:|---------:|---------------:|---------------------:|-----|
| 31   | ~544       | 36       | **0 / 15** (254/254 steps accept=0) | ~240 | catastrophic loss |
| 60   | ~1040      | 69       | **~1 / 15** (15×0, 15×1, 4×2, 2×3, 1×5, 1×8 over 38 steps) | ~240 | catastrophic loss |

(N=24 not separately run — N=31 already pins accept at exactly 0, so shallower N can only be worse.)

### GATES
- **TOKEN-EXACT: PASS.** N=31 counting greedy md5 == OFF baseline (`d6803f143fddbd730b79829a2a0b74f9`, byte-identical). This PROVES the KV/SSM checkpoint+rollback handoff is correct — committed tokens are the full target's greedy; early-exit only proposed (and with accept=0 every step, committed = pure target greedy → identical output). Lossless contract holds independent of N.
- **NO-REGRESS default-off: PASS.** Flag gated behind OnceLock-cached `ATLAS_DFLASH_EARLY_EXIT=1`; with it unset the path is byte-for-byte legacy. Baseline (same build, flag off): counting 73.2 / coding 15.7 / prose 12.9 tok/s — matches records-grade.
- **VISION: by-construction PASS (not re-run).** Early-exit only swaps the DRAFT SOURCE; it never touches the target vision encoder and never changes committed tokens (proven token-exact). With the flag default-off the build is identical to the validated baseline. Re-serving for an unchanged path was skipped to respect GPU-serial use. Images present at `~/aeon-tps/{ocr,vision}_test.png`.

### HEADLINE — did early-exit beat the tiny drafter on NOVEL coding?
**NO. The mechanism does not work on a vanilla (non-early-exit-trained) target.** At N=31 the layer-N argmax matches the full target's greedy 0/254 times — even the FIRST draft position is wrong. Drafts are high-vocab-ID garbage (54087, 156301, 29545-repeated). At N=60 (94% depth) mean accept is still only ~1/15. Net tok/s would be far below 1 (propose 544–1040 ms ≫ verify 240 ms, with ~0 accept) — a catastrophic loss vs the tiny drafter's ~15/16 on counting, ~3/16 on novel coding.

**ROOT CAUSE (the honest finding the mission asked for):** AEON's `final_norm` + `lm_head` are an output head trained EXCLUSIVELY for the layer-64 residual basis. A vanilla target has NO valid early-exit head — the layer-N residual stream is simply not in the output (unembedding) basis, so the shared head produces noise. This is categorically different from a LayerSkip / early-exit-trained model (which adds intermediate-layer LM supervision). The "hard decision in layers N..64" framing is too generous: it is not that the decision is made late, it is that the *representation* is not yet readable by the head at ANY N < ~62. Confirmed by the monotonic accept curve (N=31: exactly 0; N=60: ~1).

**SECOND BLOCKER (independent):** even if a valid early-exit head existed, the eager (no-CUDA-graph) per-draft partial forward costs 36 ms/draft at N=31 and 69 ms/draft at N=60 — propose alone is 2–4× the entire verify step. The autoregressive γ-loop of partial forwards is intrinsically expensive; it would need a graphed batched-N-layer kernel to be competitive even with perfect accept.

### What this tells us about the next move
Target early-exit is the WRONG lever for this target as-is. To make in-distribution self-drafting work would require one of: (a) train/fine-tune an early-exit LM head on the layer-N hidden (a RETRAIN — out of scope, the whole premise was no-retrain), or (b) a different target that ships intermediate-layer heads (MTP/EAGLE heads — AEON already has `mtp_num_hidden_layers=1`, i.e. ONE trained MTP head, which is exactly what the existing DFlash drafter already consumes). The no-retrain in-distribution predictor that DOES exist on this model is the trained MTP/DFlash head — which is the tiny drafter we were trying to beat. Conclusion: on a vanilla hybrid target with a single output head, there is no free in-distribution early-exit drafter; the novel-token floor must be attacked via the trained drafter's quality (fine-tune) or via tree/branch coverage at the cliff, not via target early-exit.

### Housekeeping
Edits uncommitted, default-OFF (`ATLAS_DFLASH_EARLY_EXIT` unset → `early_exit_enabled()` false → legacy byte-for-byte). Files: `model/early_exit.rs` (new), `model/mod.rs`, `model/impl_b3.rs`, `scheduler/verify_dflash_step.rs`. Build `ATLAS_TARGET_MODEL=qwen3.6-27b`. No commits. GPU freed (no spark procs). /tmp scratch cleaned (kept greedy md5 captures `/tmp/ee_*.txt`).
