# Model Migration Playbook — porting a new model to Atlas on GB10

This is the distilled optimization methodology from the DeepSeek-V4-Flash-162B
port (branch `combined-residency`, ~25 perf commits, prefill 40 → 905 tok/s,
decode 6.6 → 33 tok/s on one GB10). It is written for the NEXT port — a model
this repo has never seen — so everything below is stated as technique, with the
DeepSeek numbers kept only as evidence that the technique pays.

Per-topic depth lives in the linked docs; this file is the map, not the
territory:

- `docs/REPRODUCE-PREFILL.md` — the deterministic reproduction protocol + bisect table
- `docs/PREFILL-CAMPAIGN-2026-08-10.md` — a full campaign ladder with its NO-GO table
- `docs/SESSION-STATE.md` — the state/targets/dead-ends doc pattern
- `docs/atlas-on-the-table.md` — the open/closed optimization ledger pattern
- `docs/2026-08-09-bandwidth-frontier.md` — the hardware-ceiling measurements
- `docs/kernels/00-index.md` — hardware ground truth + per-kernel roofline docs
- `docs/kernels/prefill-attn-tensorcore.md` — a kernel design doc with MMA fragment math
- `docs/FIELD-NOTES.md` — serve-side gotchas and fast iteration oracles
- `docs/verify-exactness.md` — byte-level plain-vs-speculative diff methodology

---

## 0. The shape of a port

Every port so far (Qwen3-Next-80B, Laguna-S-2.1, DeepSeek-V4-Flash-162B) has
gone through the same phases, and the expensive mistakes came from doing them
out of order:

1. **Bring-up**: make it load, serve, and answer. Fit is usually a checkpoint
   question (pruned/requantized variants), not a kernel question.
2. **Correctness before speed**: chat template, tool-call parser, thinking
   markers, reasoning split. On V4 the tool-eval-bench went 23 → 90 purely on
   jinja/template/parser fixes (`f6c7ff47`, `b1728ee1`, `0194185d`) — no kernel
   was touched. A port that benchmarks before its template renders tool turns
   is measuring garbage.
3. **Instrument**: waterfall probes + GEMM shape log (§2, §3). Do not write a
   kernel before this exists.
4. **Dispatch audit** (§1): find the fast kernels already in-tree that the new
   model's paths fail to reach. This is historically the largest single win.
5. **Kernel campaigns**: only for buckets the audit leaves standing, each with
   a microtest oracle (§4) and an explicit numerics tier (§5).
6. **Speculation**: last, because its economics depend on the plain step cost
   (break-even C = spec_step/plain_step; measure both first).

## 1. The central disease: fast kernels already in-tree, wired to slow ones

The single most valuable lesson of the V4 port. **Five of the eight prefill
ladder steps were dispatch fixes, not new kernels** — 18–30× per site
(`0e368e6f`, `415734fa`, `0fbef449`, `889abbfe`, `af92018a`); the same disease
produced the biggest decode wins (`b4413260`, `2a957f1c`, `8998780f`,
`d7aff34d`, `6a46a4c1`). A new model is *maximally* exposed to it, because
every dispatch predicate in the tree was written for a previous model.

How a path ends up on the slow kernel — the recurring mechanisms, all observed:

- **Stale workaround pins**: kv_proj was pinned to the scalar GEMM by a NaN
  note that named a *different* kernel (`415734fa`). Re-read every "disabled
  because X" comment against the kernel it actually names.
- **Model-specific override files REPLACE the common file**: a kernel that only
  exists in another model's override resolves `KernelHandle(0)` on yours and
  the caller silently falls back to a per-row loop (`8998780f`,
  `fp8_gemm_t_row_scaled_mtile8`). Diff your model's kernel set against the
  donor model's.
- **Width predicates written for the old speculation width**:
  `(2..=4).contains(&n)`, `MOE_DECODE_MAX_ROWS = 2/6` — one row past the cap is
  not "slightly slower", it is the per-row fallback re-streaming all weights
  per row (~82 ms/row on V4, `3b123825`, `b4413260`).
- **Guards for an unrelated model's crash** disabling your batched path
  entirely (`force_seq_ffn`, `2a957f1c` — which also hid a real routing bug:
  the per-token entry read `token_ids[0]` for every row).
- **Per-token loops where a batched entry exists**: the BF16 wo_a arm ran
  n × groups GEMV launches — 313k launches per prefill, 49% of all prefill GPU
  time (`9f000654`).
- **Small-M GEMM on a tile kernel**: `dense_gemm` (16×16 tile) at M=5 wastes
  11 of 16 lanes and runs at 27 GB/s; `dense_gemv_batchm` at the same shape is
  6.8–8× (`d7aff34d`).

**The audit procedure** (turns archaeology into a checklist, `cf1838d3`):

1. `ATLAS_GEMM_SHAPE_LOG=1` on a single serve run covering prefill AND decode.
   Every GEMM/GEMV wrapper logs its unique `(kernel, M, N, K)` once — all 52
   wrappers are instrumented; keep it that way when adding wrappers.
2. For every logged tuple, ask: *is this the fastest in-tree kernel at this
   exact shape?* Answer with the microtest, never by reading headers:
   `dense_gemm_microtest -- <kernel> <M> <N> <K>` times any kernel at any
   shape against a CPU oracle. Kernel headers lie in both directions —
   `w8a16_gemm_pipelined`'s documented 12 TFLOPS was a small-shape artifact;
   it does 27 at real shapes (`21705a81`).
3. Log which branch won at dispatch sites with fallbacks (a one-shot
   `eprintln!` is enough). "This bug class is invisible without it"
   (`8998780f`).

## 2. The waterfall method (profiling without nsys/ncu)

**On GB10, nsys cannot trace prefill** (the ~6-minute weight load overruns its
buffers, `--delay` included, trace silently dropped) **and ncu is not
installed.** The replacement is in-tree bucket instrumentation, and it found
every target the campaign hit:

- `prof!` / `aprof!` / `hprof!` / `pprof!` macros: CUDA-event or
  sync-and-time buckets around phases, enabled by `ATLAS_PROFILE=1` (or the
  path-specific vars: `ATLAS_DSPARK_PROPOSE_PROF`, `ATLAS_PROFILE_VERIFY`,
  `ATLAS_DFLASH_STEP_TIMING`).
- `scripts/prefill_waterfall.py` parses a profiled serve log into the
  per-bucket waterfall.

Rules that make the numbers trustworthy:

1. **Close the waterfall.** Instrument until wrapper totals equal the sum of
   their interior buckets (<1% — the V4 FFN wrapper matched at 1365.9 vs
   1366 ms) and the total attributes to within ~1% of measured TTFT/step.
   An unattributed lump is where the bug lives: the "4_core_attention" bucket
   hid the entire 890 ms compressor block until it was split (`1b4bd7f2`).
2. **Drain the stream before starting a timer.** A `prof!` that starts with
   launches still in flight bills them to its first bucket: the MoE gate read
   221 µs/layer against a true 143 and sent a whole investigation after the
   wrong stage (`8b2e65f1`, `f3dbdabb`). Sub-bucket splits whose probe sits
   inside a loop absorb neighboring stages — read the total as exact, the
   split as indicative (`5f9d3174`).
3. **Sub-split before believing a bucket.** Bucket-level cost ≠ the kernel you
   assume: the "compressor" bucket was ~100% its two GEMMs, not the compress
   kernel — a 20–80× misestimate that had −0.25 s budgeted against the wrong
   code (`0e368e6f`).
4. **Probes must be graph-safe and opt-in.** Always-on D2H probes cost real
   tok/s and hard-wedge CUDA graph capture (CUDA 901) (`0ea19689`). And don't
   read env vars per invocation on a hot path — `std::env::var` takes the
   process env lock and allocates; cache in a `OnceLock` (`af92018a`,
   `f6c7ff47`).
5. Where nsys *does* work (decode), `scripts/nsys-verify.sh` /
   `scripts/nsys-plain.sh` give kernel-level attribution.

## 3. The microtest oracle pattern

Every kernel change ships with a standalone example binary that answers both
"is it right?" and "is it fast?" in seconds, without a server. This is the
fast iteration loop — serve round-trips take ~10 minutes; the oracle takes
~30 s. Exemplars to copy:

| exemplar | demonstrates |
|---|---|
| `crates/spark-model/examples/dense_gemm_microtest.rs` | any-kernel-any-shape GEMM comparison vs CPU oracle |
| `crates/spark-model/examples/moe_unified_t_m_microtest.rs` | bit-identity gates + routing-overlap parameter sweep + per-stage attribution |
| `crates/spark-model/examples/w4a16_gemv_grouped_microtest.rs` | grouped-vs-N-launch bit-identity + L2-defeating timing |
| `crates/spark-model/examples/w4a16_parity_microtest.rs` | strict bitdiff column + `-- M N K` shape override |
| `crates/spark-model/examples/prefill_attn_tc_microtest.rs` | behavioral (cosine) gate, parameterized over S/window/ratio configs |

The invariant structure:

- **Deterministic inputs**: splitmix64 PRNG, no `rand` dependency — inputs are
  reproducible across runs and machines.
- **A reference**: CPU f32 oracle for numerics claims; the *incumbent kernel*
  for replacement claims (run both, diff bytes).
- **Correctness gate first, timing second**, and the process **exit code
  reflects the gate** so scripts and agents cannot ignore a failure.
- **Gate type matches the claim** (§5): `bitdiff == 0` for data-movement-only
  rewrites; cosine ≥ 0.999 overall / ≥ 0.995 worst-row for reassociating
  rewrites; max|Δ|/RMS bounds for split-K style reductions where near-zero
  dot products make raw ULP meaningless (`4a43f2d0`).
- **L2-defeating weight rotation for M=1/decode timing**: allocate ≥256 MB of
  weight instances and rotate per iteration (`ROTATION_BYTES` in the grouped
  microtest) — a decode GEMV timed on a cached weight measures L2, not DRAM,
  and DRAM is what decode pays.
- **CUDA-event timing** over many iterations, reported as achieved GB/s or
  TFLOP/s against the known ceiling (§6), so a result is judged in absolute
  terms, not just "faster than before".
- **Parameterize by the production-relevant axis** (routing overlap for MoE
  dedup, S/window/ratio for attention, M for batch width) and check the
  microtest reproduces production numbers before trusting it as the fast
  oracle (the MoE microtest reproduced serve at 137 vs 135 GB/s — from then
  on kernel iteration never needed the server).

## 4. The bit-identity ladder

Every optimization claims a numerics tier, and **the claim is proven, not
asserted**. Both batched-GEMV kernel headers that claimed bit-identity were
wrong, with measured byte counterexamples (`757492de`) — a cosine test cannot
prove bit-identity.

**Tier 1 — bit-identical.** Same arithmetic ops, same operands, same order;
only data movement changed (wider stores, smem re-banking, grid shrink,
launch merging). Proof required, either or both of:

- **SASS opcode-histogram diff**: compile both variants (`cuobjdump -sass`),
  histogram opcodes; every *arithmetic* opcode count unchanged, only
  load/store widths move; registers/smem/spills stated. A compile-out toggle
  (`-DATLAS_..._NO_X`) must reproduce the pre-change SASS *exactly* so the A/B
  is honest (`3178855c`, `d8ebbd90`, `4ac91ed4`). This tier can land from an
  agent without touching the GPU.
- **Parity microtest bitdiff column**: byte-compare against the incumbent at
  the production shapes, across seeds/configs (96-config sweep in
  `f3dbdabb`).

Watch for silent tier breaks: a packed store needs *all* address components
even (row index may be data-dependent — N even is a separate condition from
column even, `ebe5bf83`); reading a value from registers instead of re-loading
does not change bytes, but changing *reduction order* does.

**Tier 2 — cosine-close.** Reassociation (tensor-core tiling, online-softmax
tile order, split-K, float4 lanes). Gate at **cosine ≥ 0.999 overall and
≥ 0.995 worst-row** against the scalar/CPU reference, at edge configs (partial
tiles, non-aliased V, tail rows) — and then the end-to-end quality gate (§7)
decides. A tier-2 change that moves the quality score is a regression
regardless of speed. When a rewrite claims "same math, different staging" on
top of an existing tier-2 kernel, hold it to ~1.0 against that kernel
(cos ≥ 0.9999999, `512fb819`) — only FMA re-association may separate them.

**Tier 3 — behaviorally gated.** Precision changes (FP8 head, NVFP4
transcode, requant). These are *model-quality* decisions: same-session A/B on
the quality suites, with a recorded full-precision baseline so a reduced
config is judged on *regressions vs baseline*, not absolute pass/fail
(`cd82378c` longgen baseline pattern). Precision cuts can be quality-POSITIVE
(FP8-native wq_a raised the gate 90 → 93, `0fbef449`; NVFP4 attention 93 vs
90) — measure, don't assume the direction.

**The speculation corollary — the partial-exactness law.** For a verify path
that must reproduce plain decode: making *one* stage bit-exact while others
still reorder is WORSE than making none exact (accept 2.54 vs 2.83 vs 2.92+
with the full chain, `090830de`, `6d80d0e6`). Exactness of a chain is
all-or-nothing; flip the whole chain together behind one flag. And byte-hash
layer diffs (`docs/verify-exactness.md`, `scripts/exactdiff.py`) are the
instrument — printed norms at 4 decimals hid divergence that FNV byte hashes
caught immediately (`634a3d77`).

## 5. Known GB10 (DGX Spark, sm_121) hardware facts

Collected the expensive way; do not re-derive. Ground truth in
`docs/kernels/00-index.md`.

| fact | value / rule |
|---|---|
| SMs | **48** (several old kernel comments claim 25 and size grids to it — wrong by 1.9×) |
| Compute capability | 12.1 (`sm_121` / `sm_121a` / `sm_121f`) |
| Shared memory | **102400 B/SM** (100 KiB); **49152 B static per-block cap** — above it you must opt in to dynamic smem |
| Threads/SM | 1536 |
| DRAM | 256-bit LPDDR5X, **273 GB/s theoretical**; **~229 GB/s achieved ceiling** (streaming read 228.9, contiguous 1-GiB GEMV 225.5, d2d 219.9); per-expert-sized GEMV 168; 64-B random gather 129 (`scripts/bw_ceiling.py`) |
| L2 | 24 MB (`persistingL2CacheMaxSize` 18 MB) — small KV pools are L2-resident; "redundant re-reads" of them cost issue slots, not DRAM |
| Launch overhead | eager launch floor ~4–6 µs → any kernel moving <~1.5 MB is launch-bound |
| `cp.async.cg` | works (sm_80+ semantics, correct on sm_121) |
| **TMA / `cp.async.bulk`** | **silently corrupt data on sm_121 — do not use** (`w8a16_gemm_pipelined.cu` header). No wgmma either; the MMA is `mma.sync` m16n8k16 (BF16) / m16n8k32 (FP8) |
| `ldmatrix` | `.x4` and `.x4.trans` validated bit-exact in the tc2 prefill attention kernel (`512fb819`); one older kernel comment claims it broken — trust the newer measured kernel, re-verify per use |
| Bank conflicts | 32 banks × 4 B. Conflict degree = 32 / gcd(lane-to-lane stride *in words* mod 32, 32)… in practice: compute the store/load word index as a function of lane, take gcd with 32. A 32-B row stride ⇒ gcd 4 ⇒ 8-way; 128-B lane spacing ⇒ all lanes one bank. Padding only helps if it makes the stride odd-word (co-prime with 32): 48-B (12-word) and 520-word strides permute all 32 banks; a transposed 2-byte store with lane stride `4*PAD mod 32` can NEVER be padded conflict-free (`4ac91ed4`, `512fb819`, `c2bca29e`) |
| Register/occupancy cliffs | runtime-bounded per-lane arrays get demoted to local memory (measured 768-B stack frames) — loop bounds and chunk counts must be compile-time; `__align__(16)` is REQUIRED for `uint4` loads from bf16 arrays (2-byte default alignment) (`c2bca29e`) |
| Masking | `-INFINITY` only, never a large negative constant — at running-max init −1e30, a −1e30 score contributes exp(0)=1 (`848e1e5f`) |
| MoE dequant-GEMM FLOP ceiling | the w4a16 dequant family tops at ~35 TFLOP/s regardless of arithmetic intensity — instruction-issue-bound, not DRAM-bound; more FLOP/byte does not help without fewer instructions per weight byte |
| **FP8-KV calibration suppresses CUDA graphs** | `fp8_kv_calibration_tokens = 256` forces eager until seq_len > 266 — **every short-prompt bench silently measures the eager path**. Use ~450+ token prompts and grep the log for both `FP8 calibration frozen` and `CUDA graph captured`. (On V4, graphs then measured worth ~0% — but you cannot know that until they actually engage.) |
| Cold vs hot | SM clocks hold 2.4–2.6 GHz cold and hot (`scripts/clock_ramp_probe.py`) — clock ramp does NOT explain serve-vs-standalone gaps here |

## 6. The validation gate sequence

Every change walks the same ladder, in order; a change that skips a rung gets
reverted by a later measurement.

1. **Microtest oracle** (§3): correctness gate + standalone timing, GPU
   otherwise idle (a resident server OOMs/starves the microtests — kill it).
2. **Serve measurement**, same binary, env-flag A/B:
   - prefill: **5-run median TTFT** at a fixed prompt (~±1% spread; warm up
     one short request first — the first request pays lazy-init);
   - decode: steady-state tok/s over fixed workloads (the four-workload
     probe: prose/code/repeat/quote — repeat-only numbers overstate);
   - speculation: `ATLAS_MTP_GATE_FORCE=1` always when measuring acceptance,
     otherwise the throughput gate parks losing speculation and masks it;
     lossless configs must byte-hash identical text at temperature 0.
3. **`tool-eval-bench --short`**: the bar is **90/100, 0 failures**. One
   scenario is the known numerics-borderline canary; the others must not
   move. A speed win that drops the score is a regression, full stop.
4. **Commit with the measurements in the message**: before/after, run count,
   spread, quality score, and the flag that reverts it. The commit log IS the
   lab notebook — this playbook was reconstructed from it.
5. **Every change env-gated with a `=0` opt-out**, default ON once validated.
   The set of flags forms a deterministic bisect table
   (`docs/REPRODUCE-PREFILL.md` §2) — any regression can be bisected by
   flipping flags on one binary instead of rebuilding N times. Compile-time
   toggles (`ATLAS_EXTRA_NVCC_FLAGS=-D...`) serve the same role for SASS-level
   changes.
6. **Build-flag trap**: the kernel set is selected at build time
   (`ATLAS_TARGET_MODEL=<model>`); a default build compiles another model's
   kernels and the server dies at boot — but microtests can mask it because
   `kernels/gb10/common/` compiles into every target.

## 7. The agent/worktree pattern

Kernel work parallelizes; the GPU does not. The pattern that worked:

- **Each kernel campaign runs in an isolated git worktree** on its own agent
  branch (`moe-p0`, `shared-k64`, `tc-round2`, `w8a8-wire`, …), with a brief
  naming the target kernel, the claim tier (§4), and the oracle to extend.
- **Agents never run the GPU.** They deliver: kernel + dispatch wiring
  (env-gated, default OFF), the microtest extension, and *compile-time
  evidence* — `--resource-usage` output, SASS opcode-histogram diffs, smem/
  register/spill accounting (`d8ebbd90` landed three bit-identical
  micro-optimizations "SASS-verified, no GPU used").
- **GPU validation is serialized in the main session**: run the oracle, run
  the serve A/B, run the quality gate, then default the flag ON and merge the
  worktree branch (`96e0fb8d`, `612619c1`, `e66088e9`).
- **Re-verify the brief's claims in-code.** Agent briefs and even design docs
  are hypotheses: the tc2 kernel disproved two claims of its own design doc
  (K never needed transposing; P never needed smem), and the moe-p0 merge
  carried three corrections to its brief (`d8ebbd90`, `512fb819`,
  `2859c39a`). The receiving session owns the truth, not the brief.

## 8. NO-GO ledger discipline

**Record refuted hypotheses with the refuting measurement, in a tracked doc,
or you will pay for the same experiment twice.** The V4 port keeps them in
`docs/SESSION-STATE.md` ("Measured dead ends"), the campaign doc's NO-GO
table, and `docs/atlas-on-the-table.md` (open/closed ledger). Rules:

- A row moves to CLOSED **only with a measurement, never with an argument**.
- When a *baseline assumption* is corrected, re-quote every verdict that was
  measured against it (the "no MoE kernel win left" verdict died when the
  183 GB/s "ceiling" was measured to be 229 — the verdict was an artifact of
  a false constant).
- Supersede loudly: corrections get their own ledger section ("do not regress
  to the old beliefs") because a stale number in a doc re-derails the next
  session ("online acceptance ~1.0" drove weeks of work; it was 2.68 all
  along — the mean hid a bimodal histogram, `5da8a922`).
- Keep the negative kernels and probes in-tree as measured negatives — the
  microtest that times the losing variant is the proof the door is closed.

The big closed doors from this port (measurements in the linked docs — listed
here so a new port recognizes the *shapes* of dead ends):

| refuted hypothesis | refuting measurement |
|---|---|
| SM clock ramp explains serve-vs-standalone gaps | cold == hot, 2.4–2.6 GHz throughout |
| per-stage sync calls cost real time | `=0` changed nothing end-to-end |
| the non-GEMM tail of a bucket is the cost | bucket was ~100% its GEMMs (20–80× misestimate) |
| register-resident re-reads beat L1 re-reads | bit-identical but slower — occupancy loss beats L1 savings |
| copy/gather traffic dominates a fragmented projection | in-place +1%; the cost was launch count (or the GEMM shape) |
| launch count dominates a slow small-M path | 240→48 launches: noise; the small-M kernel shape was the cost |
| lossless expert-weight compression (any form) | flat spectra, orthogonal experts, ~zero MI, 1.028× entropy ceiling |
| CUDA graphs are a big decode lever here | graphed vs eager: ~0% (once they actually engaged) |
| batched GEMVs re-read weights per row | verified false for every kernel in the tree; M=6 cost = union size |
| a 6.4× kernel win must show end-to-end | gate GEMV: ~0 end-to-end; near-tied logits also flipped routing |
| async propose overlap flag overlaps | 19.96 vs 20.00 tok/s |
| manual double-buffer prefetch on a reg-heavy kernel | regressed 12% — ptxas already pipelines at 128 regs; the `__launch_bounds__` sweep already in-tree predicted it |

## 9. Bring-up checklist for the next model

Condensed order of operations, with the trap each step disarms:

1. **Fit**: find/prepare a single-node checkpoint (expert pruning or requant
   beats more bits of the full model). Record the reference stack's numbers
   on the same box first — they are the target and the sanity check.
2. **Build** with `ATLAS_TARGET_MODEL=<model>`; add the model's kernel dir;
   check which kernels exist only in *other* models' override dirs (§1).
3. **Smoke + quality probes**: coherence, a small accuracy set, a
   long-generation gate with a full-precision baseline recorded *before* any
   precision experiments.
4. **Template/parser correctness**: tool role rendering, native tool-call
   syntax parser (streaming included), thinking markers from the checkpoint's
   own encoding files — capability detection must consult the tokenizer, not
   the architecture (`0194185d`).
5. **Instrument**: wire `prof!`/`aprof!` buckets for the new model's layer
   structure until the waterfall closes (§2); wire `ATLAS_GEMM_SHAPE_LOG`
   into any new GEMM wrappers.
6. **Dispatch audit** (§1) — expect this to be the biggest single lever.
7. **Kernel campaigns** on the surviving buckets, agent-parallel (§7), each
   through the gate sequence (§6).
8. **Speculation last**, with the acceptance instrumentation
   (`ACCEPT_LOG` histograms, propose/verify phase profilers, gate-force) and
   the exactness harness from day one — the V4 acceptance chase burned weeks
   on a mis-measured mean and non-bit-exact verify kernels (§4).
9. **Keep the ledger** (§8) from the first measurement: state doc, open/closed
   table, reproduction doc with the bisect flag table.
