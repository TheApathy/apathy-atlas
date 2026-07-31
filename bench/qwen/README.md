# Qwen3.6-27B champion — DFlash reproduction harness

This directory reproduces the **champion** speculative-decoding configuration for
a dense Qwen3.6-27B target with a DFlash drafter, measured single-stream on
GB10-class hardware (Grace Blackwell, `sm_121f`, unified LPDDR5x).

It is the sibling of `bench/laguna/` on the `laguna` branch and is deliberately
the same shape: one `env.sh` that owns every path and setting, one self-verifying
launcher, one deterministic decode benchmark. It is **not** a re-parameterization
of that harness. The two stacks disagree on γ, on KV dtype, on which log lines
carry the accept counter, and on which gates are safe to enable. Settings do not
transfer between them; see [Traps](#5-five-things-that-will-cost-you-a-day).

---

## 1. Prerequisites

| What | Notes |
|---|---|
| A CUTLASS source checkout | `CUTLASS_HOME` must point at it. If you have built vLLM from source you already have one under its `.deps/`. |
| Target checkpoint | Public: `AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP`, apache-2.0, ungated. **Pin the revision** — see [Checkpoint availability](#checkpoint-availability). |
| DFlash drafter checkpoint | A `DFlashDraftModel` snapshot for that target. Ours is an in-house retrain and is **not published** — this is the one thing you cannot obtain. |
| CUDA toolkit + a Blackwell-class GPU | ~50 GB of free memory at launch; γ=16 inflates the SSM-MTP pool and KV by roughly 24 GB over a non-speculative serve. |

Nothing here has a machine-specific default. Every path comes from the
environment, and the scripts refuse to guess:

```bash
export QWEN_MODEL=/path/to/Qwen3.6-27B-target
export QWEN_DRAFT=/path/to/Qwen3.6-27B-DFlash-drafter
export QWEN_BIN=/path/to/spark          # optional; defaults to ./target/release/spark
```

### Checkpoint availability

**The target is public. The drafter is not, and that limit is on us.**

**Target — get the exact revision.**

```bash
hf download AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP \
    --revision 6f31471e2e1c3462420f14af9a9a0046bae3d9eb \
    --local-dir ./qwen36-27b-target
export QWEN_MODEL=$PWD/qwen36-27b-target
```

`--revision` is not optional decoration. The repo has moved on since we
downloaded it: `model.safetensors` at the current head has the same byte length
as ours but a **different content hash**, so an unpinned `hf download` gets you
weights we never measured. Same size is not same file. Everything published here
is against `6f31471e`, which still resolves.

**Drafter — you cannot get ours, but you can get the one we started from, and it
is not worse.** Ours is a `DFlashDraftModel` we retrained in-house and have not
released. An earlier version of this section said no public drafter was a
drop-in. **That was wrong, and it was wrong in our favour**, so it is corrected
here in full.

Hookup matters more than weights: serving a drafter whose wiring the target does
not expect does not fail loudly, it produces a plausible but much lower accept
rate that reads exactly like "the other drafter is worse". Read from each
checkpoint's own `config.json`:

| checkpoint | layers | `target_layer_ids` | `mask_token_id` | drop-in? |
|---|---|---|---|---|
| ours (in-house retrain) | 6 | `[1,10,18,27,35,44,52,61]` | 248077 | — |
| `z-lab/Qwen3.5-27B-DFlash` @ `25ee0025` | 6 | `[1,10,18,27,35,44,52,61]` | 248077 | **yes** |
| `KingsonHO/Qwen3.6-27B-DFlash` | 5 | `[1,16,31,46,61]` | 248070 | no |
| `deepsweet/Qwen3.6-27B-DFlash-FP16` | 5 | `[1,16,31,46,61]` | 248070 | no |
| `z-lab/Qwen3.6-27B-DFlash` | — | — | — | not checked — do not assume either way |

So `z-lab/Qwen3.5-27B-DFlash` at `25ee0025` **is** a structural drop-in. Pin the
revision, for the same reason the target is pinned:

```bash
hf download z-lab/Qwen3.5-27B-DFlash \
    --revision 25ee0025ff950496a634e100b75c2db4515e9824 \
    --local-dir ./qwen-dflash-drafter
export QWEN_DRAFT=$PWD/qwen-dflash-drafter
```

We measured it against ours with `bench/qwen/drafter_ab.sh` — three arms in one
sitting, the third a repeat of the first so the noise floor spans the same
elapsed time as the comparison it judges:

| pooled | ours | `z-lab` 3.5 | A/A repeat |
|---|---|---|---|
| token-weighted tok/s | 36.1 | **38.0** | 37.0 |
| mean accept /16 | 4.85 | **5.23** | 4.84 |

A/A floor: 2.36% on tok/s, 0.35% on accept. Ours came in **−4.8% tok/s and −7.2%
accept**, clearing the accept floor by 21×. On this suite the public drafter is
*better* than ours.

> **Correction, and a trap worth the paragraph.** An earlier version of this
> table said −3.6% on accept, pooling each arm over *its own* verify steps —
> which is what a serve's `mean=` line gives you, and what our first harness
> copied. That is confounded. A speculative step commits `accept+1` tokens, so a
> drafter that accepts *more* on a cell takes *fewer* steps there, and its own
> average therefore gives that cell *less* weight. Every arm gets scored on a
> mix tilted toward the cells it is worst at, and the better drafter is
> penalised for being better. It is not a rounding effect: `prose` alone carried
> 41% of the step weight purely because `prose` is the row both drafters accept
> least on. Holding the weights fixed across arms — same workload, two drafters,
> which is the question actually being asked — the same six rows move from −3.6%
> to −7.2%. The A/A floor barely moves (0.34% → 0.35%), because A and its own
> repeat share a step distribution by construction. `eval_verdict.py` pools with
> fixed weights and documents why; the numbers above are the re-pooled ones.

Read the verdict narrowly. This suite is five Python prompts and one prose
prompt. Our retrain was warm-started to raise the **Go** share of its training
corpus, and the A/B that justified shipping it reported Go +4.6% with Python
neutral — an axis this suite does not contain a single row of. The honest
statement is "ours loses on Python and prose here", not "the retrain was
worthless". That gap is why section 4 now defaults to a coverage-gated matrix
instead of this suite.

So be clear about what this branch does and does not offer:

* **Reproducible from this branch with public checkpoints only:** the build
  recipe, the serve configuration and its gate-parity assert, the benchmark
  harness, the single-stream measurement protocol, every trap in section 6, and
  — using the `z-lab` drafter above — the section 5 tok/s to within a few
  percent, in our measurement slightly better.
* **Not reproducible without our drafter:** the completion hashes in
  `reference_hashes.json`.

With a substituted drafter, `check_repro.py` will report `MISMATCH` on every
stable row and exit 1. **That is the correct result for a different drafter and
is not a build failure** — do not go hunting a numerics bug you do not have.

**A caveat we owe you, because it is our bug and not yours.** On a greedy target
at temperature 0 the committed tokens are the target's own argmax, so they ought
not to depend on which drafter proposed them. On this stack they do. In the A/B
above, 4 of 6 rows produced different completion text under the two drafters, and
3 of those (`common-algo`, `novel-logic`, `math`) are byte-stable when the
drafter is held fixed across two runs — so the drafter swap, not run-to-run
nondeterminism, is what moved them. The divergences are substantive, not
tie-breaks: the `math` row returns a structurally different implementation and a
different length. The mechanism is not yet identified and we are not going to
guess at one here. Two consequences:

* Do not use hash equality as a correctness gate when comparing drafters on this
  stack. Grade acceptance; report hashes without grading them.
* The `prose` row is separately nondeterministic run-to-run even with the drafter
  held fixed, and `check_repro.py` already excludes it from the stable set.

If you want a decode result you can check against a published reference end to
end, use the `laguna` branch instead: its target
(`poolside/Laguna-S-2.1-NVFP4`) and drafter
(`poolside/Laguna-S-2.1-DFlash-NVFP4`) are both public, so
`bench/laguna/reference_hashes.json` is checkable all the way through.

## 2. Build

```bash
CUTLASS_HOME=/path/to/cutlass bash bench/qwen/build_cutlass.sh
```

A bare `cargo build --release` is **not** equivalent. It produces a `spark` that
defaults to a different kernel target, cannot load this model, and fails only at
serve time. The build script sets the four `ATLAS_TARGET_*` variables that select
the `kernels/gb10/qwen3.6-27b/nvfp4` kernel set, and asserts that set exists
before starting a twenty-minute compile.

## 3. Serve

```bash
bash bench/qwen/serve_champion.sh
```

This launcher asserts its own invariants and refuses to report a healthy serve
unless all of them hold. Every assert exists because its absence once produced a
clean-looking but wrong measurement:

1. **Gate parity (pre-launch).** `env.sh`'s `qwen_champion_env()` is a second copy
   of the configuration in `local/serve-aeon-27b-dflash.sh`, the script the
   published numbers were measured on. `verify_gate_parity.sh` re-derives the
   gate set from that launcher on every run and refuses to serve on any
   difference. It runs *before* the preflight kill, so a config error does not
   cost you a running serve. This one is not hypothetical: eleven gates were
   missing from `env.sh` for a while, because the check that cleared it had been
   pointed at a stale copy of the launcher and so agreed with itself. Run
   `bash bench/qwen/verify_gate_parity.sh --self-test` to confirm the comparator
   fires on a planted divergence before you trust it passing.
2. **Gate guard (pre-launch).** Every `ATLAS_*` gate the stack exports must exist
   as a string in the binary. A gate the binary cannot read is a silent no-op,
   and the serve is then healthy while running a configuration nobody asked for.
   The gate list is derived from the function body at runtime, never from a
   hand-maintained copy.

   One wrinkle worth knowing before it fails you: not every `ATLAS_*` variable is
   read from the environment. `ATLAS_DFLASH_QUANT` is consumed by the shell and
   forwarded as `--dflash-quantization`, so nothing in the binary looks that name
   up and its absence from `strings` is correct. Such a variable is not exempted
   — that would turn a genuinely too-old binary into a pass — it is redirected:
   the guard requires the *flag* it feeds to be present instead. The forwarding
   map is re-derived from the launcher and `env.sh`, like everything else here.
3. **Kernel target.** The serve prints a three-field tuple with the SM arch
   first — `Selected kernel target: (sm_121, qwen3.6-27b, nvfp4)` — so the model
   is matched as a whole *field* inside it, not as a prefix after the colon.
   Pinning the arch or the quant suffix would produce a false failure the day
   either changes for an unrelated reason; matching a bare prefix, which is what
   this assert used to do, fails a serve that loaded the right target.
4. **Determinism.** The same temperature-0 prompt twice must be byte-identical.
   Without this, every hash comparison in the harness is measured against a
   moving target.
5. **No SSM corruption signature.** See trap 1 below — the failure mode is a run
   of `!` in otherwise valid output, not a crash.
6. **Speculation actually ran, at width 16.** A serve that silently decodes
   serially is fast enough to look plausible and produces numbers that are not
   about DFlash at all.

## 4. Measure

```bash
python3 bench/qwen/decode_bench.py --tag champion \
    --log bench/qwen/ab/serve-champion.log \
    --json-out bench/qwen/ab/champion.json \
    --dump-text bench/qwen/ab/text
```

Six prompts ordered easy → hard, temperature 0, thinking off. Each row is
classified DFlash-or-serial **before** it is aggregated, and a row that cannot be
graded is reported as `UNGRADED` rather than folded into either bucket. A count
of zero must never be readable as "nothing wrong".

### Checking your run against ours

```bash
python3 bench/qwen/check_repro.py bench/qwen/ab/champion.json
```

`reference_hashes.json` holds the completion hash of every prompt from our own
cold-clone run, so a reproduction can be checked against *our* result rather
than only against itself. Matching tok/s is weak evidence — two stacks can agree
on throughput to a percent and still emit different tokens. The hashes are what
constrain the computation.

Three per-row states, deliberately not two: `MATCH`, `MISMATCH`, and
`KNOWN-UNSTABLE` for the `prose` row, which we publish as nondeterministic. A
legitimately unstable row must not read the same as a broken one, and must not
be quietly dropped either. The checker also refuses to pass a run that is
missing prompts, since that would report zero mismatches and look exactly like a
clean result.

Exit codes: `0` pass, `1` a stable row differs, `2` incomplete run, `3` nothing
to compare. A `MISMATCH` almost always means a configuration difference rather
than a numerics one — check the `serve_champion.sh` asserts first.

### Evaluating a change (not the same thing as reproducing a number)

`decode_bench.py` answers "did I rebuild your stack correctly". It is the wrong
instrument for "is my change an improvement", and using it that way is how we
shipped a drafter on evidence that never touched the workload it was built for.
For that question:

```bash
export QWEN_DRAFT_A=/path/to/baseline     QWEN_DRAFT_B=/path/to/change
export QWEN_CLAIMS=lang:go                # the workload the change targets
bash bench/qwen/drafter_ab.sh
```

Three arms — `a`, `b`, and `a` repeated last so the noise floor spans the same
elapsed time as the comparison it judges. Each arm runs `eval_matrix.py`: five
matched algorithm tasks (`binsearch`, `mergesort`, `modinv`, `intervals`,
`boilerplate`) in **C, Python and Go**, phrased identically apart from the
language name, plus two non-code anchors. Then `eval_verdict.py` scores it. Four
things it does that a pooled A/B does not:

1. **Coverage is a gate.** `QWEN_CLAIMS` names the workload the change is
   supposed to help. If the matrix grades zero cells there, the scorer prints
   `REFUSING TO SCORE` and exits **3** — before any number, so it cannot be read
   past. "We never tested the thing it was built for" is a distinct outcome from
   "no significant difference", and the exit codes keep them distinct (`0`
   scored, `3` uncovered claim, `4` too few comparable cells).
2. **Every cell is judged against its own repeat**, not against a floor borrowed
   from another sitting. Cell floors are n=1 — a lower bound on spread, not a
   tolerance — and `TIE` therefore means "not distinguishable here", never
   "proven equal".
3. **The pooled number is printed next to the mix it depends on**: weighted by
   tokens, by cell, and by language. If those disagree in *sign*, the report says
   so, because "B beats A" is then a claim about your prompt mix. Weights are
   held fixed across arms — see the correction in section 1 for what happens
   when they are not.
4. **Hashes are reported, never graded.** See the caveat in section 1.

Accept leads the report. It is the only quantity a drafter alone controls, and
its floor here came out ~7× tighter than tok/s (0.35% vs 2.36%). `tok/s` moves
for reasons that have nothing to do with the drafter.

`QWEN_SUITE=decode` falls back to the original six Python-and-prose prompts.
That is the arm `reference_hashes.json` is keyed on, so it stays available for
reproduction — but it is not adequate evidence for a change, and the default no
longer pretends otherwise.

The scorer is pure — it reads JSON and touches no GPU — so its refusal paths are
tested offline against fixtures, including a replay of the real 2026-07-31
drafter A/B that asserts this harness would have refused it:

```bash
python3 bench/qwen/test_eval_verdict.py   # ~1s, no serve required
python3 bench/qwen/test_census.py
```

A guard tested by running the real thing is only tested on the path where it
does nothing; if the refusal is broken, the live test's failure mode *is* the
event the refusal exists to prevent.

### Files

| File | Purpose |
|---|---|
| `env.sh` | Every path, port, gate and geometry setting for the harness. |
| `verify_gate_parity.sh` | Re-derives the gate set from `local/serve-aeon-27b-dflash.sh` and fails on drift. That launcher, not `env.sh`, is the source of truth. |
| `build_cutlass.sh` | Release build with the correct kernel target. |
| `serve_champion.sh` | Self-verifying launcher (the six asserts above). |
| `benchenv.py` | Log scraping. Owns the accept anchor — read it before editing a regex. |
| `decode_bench.py` | The six-prompt deterministic decode benchmark. Reproduction, not evaluation. |
| `check_repro.py` | Checks a `decode_bench.py` run against `reference_hashes.json`. |
| `drafter_ab.sh` | Three-arm A/B driver. Refuses on hookup mismatch, gate drift, or a missing arm. |
| `eval_matrix.py` | One arm of the evaluation matrix: matched tasks × C/Python/Go. |
| `eval_verdict.py` | Scores the matrix. Pure; owns the coverage gate and the pooling rules. |
| `test_eval_verdict.py` | Offline truth table for `eval_verdict.py`, including the real-A/B replay. |
| `test_census.py` | Offline truth table for the contamination census. |

---

## 5. Measured numbers

Every figure below was produced by `decode_bench.py` in this directory, on the
six prompts it actually ships, from a **fresh clone of this branch and a cold
build** — two passes through a single serve. Nothing here is carried over from a
prior measurement record; see [Provenance](#provenance-of-these-numbers) for why
that distinction cost a rewrite.

They are **content-mix dependent**: throughput spans 4.3× from the most
predictable prompt to the least, so a single headline number would be a fiction.
The token-weighted row is the honest end-to-end figure.

| Prompt | pass 1 | pass 2 | mean accept /16 |
|---|---|---|---|
| `repetitive` | 70.1 | 71.5 | 10.43 |
| `easy-code` | 53.8 | 56.5 | 9.13 |
| `common-algo` | 61.4 | 62.7 | 9.78 |
| `novel-logic` | 38.7 | 38.9 | 5.25 |
| `math` | 39.0 | 38.4 | 5.28 |
| `prose` | 16.5 | 16.5 | 1.60 |
| **suite mean** | **46.6** | **47.4** | |
| **token-weighted** | **36.5** | **36.8** | |

All six rows are classified DFlash (none fell back to serial, none `UNGRADED`),
and every verify step ran at width 16.

**These are single-stream figures**, and that is a property of the stack rather
than a shortcut in the harness: `decode_bench.py` issues one blocking request at
a time, and the serve is launched with `--max-batch-size 1` / `--max-num-seqs 1`.
Speculative decoding is a latency optimization — it spends extra compute per step
to shorten the critical path — so its advantage shrinks as soon as more than one
sequence is active and the GPU is no longer idle-waiting. Do not compare these
against a concurrent aggregate throughput number; that is a different regime with
a different optimum, and it will be higher.

**What reproduces exactly, and what does not.** Five of the six prompts are
byte-identical across both passes *and* across a second, independently built
binary — and the per-prompt mean accept agrees to the last digit on both builds.
That is the strong evidence here; the tok/s figures are the weak evidence.

`prose` is the exception and is **intermittently nondeterministic**: across four
observed passes it produced one output three times and a different one once.
Both outputs are fluent and on-topic — they diverge mid-sentence into different
phrasings, not into corruption. It is also by far the lowest-acceptance row
(1.60/16). This is *not* measurement contamination: under `max_batch=1` no
co-batching is possible, and in the run that diverged the only foreign request in
the log arrived 12 ms *after* the last prompt completed. Treat the `prose` row as
an open item rather than a stable reference point. Note that
`serve_champion.sh`'s determinism assert passes throughout — it probes a short
code prompt, which is in the stable set.

**Run-to-run spread on timing is wider than the output stability suggests.** An
earlier run of this same suite, on a different build of the same tree, measured
38.1–38.5 token-weighted against this run's 36.5–36.8 — about 4.4% higher, with
*byte-identical output and identical accept histograms*. Nothing about the
computation changed, so that gap is wall-clock only. Budget **±5% on tok/s**
between runs and do not read a difference that size as a regression. GPU clocks
are not locked here, and this particular serve was started shortly after a heavy
compile.

Acceptance on the two hardest code rows sits at roughly **5.2 of 16** drafted
tokens, and on prose at **1.6 of 16**. Raising it was attempted repeatedly and is
documented as a dead end:

> the ~3/16 coding floor is drafter-**INTRINSIC**; the unlock to ≥13/16 /
> 30–80 tok/s is a drafter **RETRAIN** (or a bit-exact tree-verify kernel), not
> an accept-gate relaxation.
> — `results.md`

Two levers were measured and are worth knowing about because they look promising
and are not:

- **Relaxed accept gate.** PPL-neutral settings buy +1.7–4.5% on coding and
  nothing more; pushing harder reaches ~19.5 tok/s but at measurable perplexity
  drift and counting corruption. On the high-entropy tokens where the drafter
  actually misses, the draft is far down the target's distribution — not a
  top-2/3 near-miss the gate could rescue.
- **Adaptive γ.** No coding gain, and counting collapses from ~80 to ~52 tok/s.

Quality gates for the two non-lossless gates in the stack, from the launcher's
own record: `ATLAS_THINK_SPEC` scored HumanEval thinking-mode pass@1 **97.5%
with** the flag versus **95.0% without** (CI `[0, +0.075]`, "not worse" → ship),
and the shipped drafter checkpoint scored **95.0% vs 95.0%**, delta CI `[0,0]` —
identical outputs on every problem.

### Provenance of these numbers

This section used to quote a table from `results.md` — the measurement record the
scripts were built around — rather than from the scripts. When the suite was
finally run from a clean clone, only one of those three rows survived contact
with it. `results.md` reported counting at ~73 tok/s, which matches `repetitive`
at 70–73. It also reported novel coding at ~17.8 and prose at ~12.9, where this
harness measures 38.7 and 16.5. And it put coding acceptance at ~3.35/16 where
the two hardest code rows measure 5.25 and 5.28.

Those older figures are not necessarily wrong; they were taken on a
configuration that cannot now be reconstructed from what is in this repository.
But a reproduction harness whose headline table cannot be produced by running it
is worse than one with no table at all, because the gap reads as a broken build
to whoever runs it first. So the numbers were replaced rather than reconciled.
The prose immediately above is kept because it is *analysis* — the reason the
accept ceiling is drafter-intrinsic, and why two attractive levers do not pay —
and that reasoning is unaffected by the absolute figures moving.

The quoted "~3/16 coding floor" in the block above is left verbatim as a
quotation from that record. Read it as directional: this harness measures ~5/16
on the equivalent rows, and the conclusion it supports — that a retrain, not a
gate relaxation, is the unlock — is what carries forward.

---

## 6. Five things that will cost you a day

**1. γ=16 is a correctness constraint, not a tuning knob.**
γ=16 plus the prefix gives K=17 verify tokens, and K=17 is the only width routed
through the fused `gdn_wy17_k` SSM kernel that saves all 17 intermediates. Every
K in 5..16 falls through to a sequential per-token SSM path that produces NaN at
positions K-3..K-1 and corrupts the SSM rollback. It does not crash — the target
emits a correct first token followed by a run of `!`. `env.sh` refuses to launch
a `DRAFT_CAP != γ` pair, and `serve_champion.sh` separately checks the *output*
for the signature, because the first guard only checks what was set.

**2. The M_TILE=16 NVFP4 attention path corrupts at K=17.**
`ATLAS_TC_NVFP4_M16` and `ATLAS_TC_NVFP4_M16_MS_ATTN` are exported as `0` on
purpose and are not dead entries to tidy away. With them on, verify's deep-slot
argmax repeats earlier digits, greedy determinism breaks across requests, and
acceptance collapses from ~15.6/16 to ~1.5/16 — which reads exactly like a
drafter-quality regression and sends you after the wrong thing entirely. Every
other combination of the KGAMMA/TRANSPOSED gates is token-exact. It bought about
20 ms of verify time.

**3. The drafter must be NVFP4; bf16 is not a slower fallback, it is unusable.**
With `ATLAS_DFLASH_QUANT=bf16` every drafter projection runs a dense GEMM over
the full context and propose time scales to ~1.75 s per step at sequence length
800 — coding prompts simply time out. Drafter precision moves *acceptance*, never
committed output: the target's logits are the source of truth either way.

**4. A bare `accepted=` regex double-counts every step.**
Three different log families contain that substring, and two of them are emitted
once per step when `ATLAS_DFLASH_STEP_TIMING=1`. Counting both leaves the
histogram's shape and the mean accept unchanged and merely doubles `steps` — so
the token-accounting guard reads ~2.0 instead of ~1.0 and every row is
misreported as "not really DFlash". `benchenv.py` anchors on the `N/M (P%)` shape
that uniquely identifies the ungated verify line, and the anchor is tested
against a fixture containing all three families.

**5. Phase timings here are microseconds, and only exist when you ask for them.**
The champion config does not set `ATLAS_DFLASH_STEP_TIMING`, so `verify_ms` and
`propose_ms` are legitimately absent on a normal run and render as `-`, never as
`0.0`. An unmeasured phase must not be readable as a measured zero. Set
`QWEN_STEP_TIMING=1` for the breakdown, and note that doing so is a deviation
from the configuration these numbers describe.
