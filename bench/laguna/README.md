# Reproducing the Laguna DFlash decode numbers

This directory is the harness for the `laguna` branch: a single-stream
speculative-decoding (DFlash) configuration for **poolside/Laguna-S-2.1-NVFP4**
on GB10-class hardware (Grace Blackwell, `sm_121f`, 48 SMs, unified LPDDR5x).

Everything here is designed so that someone else can get the same numbers, or
find out why they can't. Nothing is hardcoded to a particular machine — set
three environment variables and the scripts run from a fresh clone.

---

## 1. Prerequisites

```bash
# CUTLASS source checkout (any recent release; the grouped MoE GEMMs need it)
export CUTLASS_HOME=/path/to/cutlass

# Gated checkpoints -- download with `hf download`, then point at the snapshot dirs
export LAGUNA_MODEL=/path/to/models--poolside--Laguna-S-2.1-NVFP4/snapshots/<sha>
export LAGUNA_DRAFT=/path/to/models--poolside--Laguna-S-2.1-DFlash-NVFP4/snapshots/<sha>
```

Both checkpoints are gated on Hugging Face and need a token. Keep it in the HF
CLI's own store or an untracked `.env` and read it as `$HF_TOKEN`. Never inline
it in a script — the repo's `.gitignore` blocks the whole class of files that
hold credentials, but it cannot un-publish one you commit.

## 2. Build

```bash
bash bench/laguna/build_cutlass.sh
export LAGUNA_BIN=$PWD/target/release/spark
```

**A bare `cargo build --release` is not equivalent.** It produces a `spark` that
defaults to a different kernel target, cannot load Laguna, and fails only at
serve time. The four `ATLAS_TARGET_*` variables in `build_cutlass.sh` are what
select the Laguna NVFP4 kernel set.

Two related traps worth knowing before you debug a "working" build:

- Editing a `.cu` does not reliably trigger a kernel rebuild, and the
  `compiled N kernels` line is itself cached, so it will report success over
  stale PTX. `LAGUNA_FORCE_KERNELS=1` clears the fingerprint.
- A clean `cargo build --release` never compiles `examples/` or `benches/`, so a
  stale kernel caller there survives a green build. Use
  `--features="cuda gpu-examples"` when that matters.

## 3. Reproduce the table

```bash
bash bench/laguna/repro_table.sh
```

**This is the entry point for the decode table below.** It runs all four arms
back to back on one box, grades every completion hash against
`reference_hashes.json`, and prints the table. Budget ~25 minutes.

| arm | what it is |
|---|---|
| `serial` | no speculation at all — the denominator |
| `nogproj` | DFlash, `ATLAS_VERIFY_GPROJ_GEMV` off |
| `gproj` | DFlash, full production stack — the headline arm |
| `gproj-p2` | `gproj` again, unchanged — your box's A/A noise floor |

The fourth arm is the one people skip and shouldn't. A reproduction that lands
3% off ours means nothing until you know what 3% costs on *your* machine, and
that number is not a constant: it belongs to a harness plus a config plus a
session, not to the hardware. Ours came out at 1.1%. Measure your own rather
than inheriting it.

The driver refuses to print a table if any arm produced no output. Scoring
whichever arms survived is how a sweep whose arms had *all* died still printed a
confident verdict — see trap 4.

Run a single arm with `repro_arm.sh <tag> [ENV=VAL ...]` when a reproduction
disagrees and you are bisecting which arm moved.

## 4. Serve interactively

```bash
bash bench/laguna/serve_prod.sh
```

For poking at the model by hand, not for reproducing the table. It is
self-verifying — it refuses to report a healthy serve unless the chat template
resolved to a native Laguna template (never the ChatML fallback), CUTLASS
scale-factor blocks were built, every `ATLAS_*` gate it exports actually exists
as a string in the binary, and the same temperature-0 prompt twice returns
byte-identical output. That last one is load-bearing: every hash comparison in
this directory is meaningless if the baseline is not deterministic.

> **`serve_prod.sh` is a different regime from the table.** It serves
> **bf16 KV at 3072** context; the published numbers were measured at **fp8 KV
> at 8192**, which is what `env.sh` defaults to and what `repro_table.sh` uses.
> Benchmarking against a `serve_prod.sh` serve and comparing to the table below
> is comparing two different configurations, and it will look like a
> reproduction failure when nothing is wrong.

To measure against a serve you launched yourself:

```bash
python3 bench/laguna/decode_bench.py --tag mine --log bench/laguna/ab/serve-prod.log
```

Six prompts ordered easy → hard, temperature 0, thinking off. Reports per-prompt
decode tok/s, the suite mean, the token-weighted mean, and the accept
distribution scraped from the log window this run opened.

Other entry points:

| script | what it answers |
|---|---|
| `repro_table.sh` | **the published decode table, all four arms, hash-graded** |
| `repro_arm.sh` | one arm of the above, for bisecting a disagreement |
| `lang_bench.py` | does coding throughput hold across C / Python / Go |
| `prefill_cublas.sh` | prefill throughput, and the cuBLASLt prefill route A/B |
| `conc_capacity.sh` | throughput and latency vs concurrency |
| `capacity_table.sh` | serial vs DFlash capacity: weights, KV pool, prefill, decode |
| `gate_run.sh` | speed **and** quality gate for one config on one serve |
| `gate_ab.sh` | two-arm A/B that refuses an arm whose binary can't read its gates |
| `full_eval.sh` | the 69-case tool-eval |

### Checking your run against ours

```bash
python3 bench/laguna/check_repro.py --arm gproj bench/laguna/ab/repro/gproj.json
```

`reference_hashes.json` holds the completion hash of every prompt in all three
arms from our own cold-clone run, so a reproduction can be checked against *our*
result rather than only against itself. Matching tok/s is weak evidence — two
stacks can agree on throughput to a percent and still emit different tokens.

`--arm` is required (`serial`, `nogproj`, `gproj`) and is not guessed from the
tag, because **the arms legitimately disagree with each other**. DFlash is meant
to be output-identical to serial and is on 5 of 6 rows, but the g_proj GEMV arm
differs from both on 4 rows *by design*: it removes a ~1 ulp BF16 error present
in the baseline GEMM, so some argmax ties break the other way. Checking one arm
against another's reference therefore produces four very convincing mismatches
that mean nothing.

Three per-row states, deliberately not two: `MATCH`, `MISMATCH`, and
`KNOWN-UNSTABLE` for `prose` on the `nogproj` arm, which we publish as
nondeterministic (see trap 1 — it suspends to serial by design, and its text is
not stable there; the same prompt *is* stable on `gproj`). The checker also
refuses to pass a run that is missing prompts, per trap 4.

Exit codes: `0` pass, `1` a stable row differs, `2` incomplete run, `3` nothing
to compare.

---

## What the numbers look like

Measured single-stream on GB10-class hardware, token-weighted across the
6-prompt suite:

| stack | weighted tok/s | vs serial |
|---|---|---|
| serial decode | 21.3 | — |
| + DFlash (γ=6) | 29.9 | +40.0% |
| + g_proj verify GEMV | 32.7 | +53.4% |

Prefill, fitted over a length sweep: ~2160 tok/s, rising ~50% with
`ATLAS_PREFILL_CUBLAS=1`. Quality: 82/100 on the 69-case tool-eval.

**Reproduction record.** The table above was re-measured from a cold clone of
this branch at `6b070c15` — fresh checkout, fresh build (158/158 nvcc
invocations, 0 cache hits), no warm `target/`:

| stack | published | cold-clone re-measure | delta |
|---|---|---|---|
| serial decode | 21.3 | 20.9 | −1.9% |
| + DFlash (γ=6) | 29.9 | 29.7 | −0.7% |
| + g_proj verify GEMV | 32.7 | 32.4 | −0.9% |

Expect **±2% on tok/s** and treat anything inside that as agreement. The
stronger check is the text: 17 of the 18 prompt outputs were byte-identical to
the original run, on a different binary three days later, and the shipping arm
reproduced byte-for-byte across two separate serve launches in the same
session. That is what `--dump-text` and the `sha=` column are for — a tok/s
number that lands on target while the text moved is not a reproduction.

Two things to expect rather than treat as breakage:

- **The `prose` row suspends.** `ATLAS_DFLASH_ADAPTIVE=1` drops a row to serial
  decode when measured speedup falls under `ATLAS_DFLASH_ADAPTIVE_MIN=1.2`, and
  prose does that every time (`adapt_suspend: 1`, ~20.6 tok/s — the serial
  rate). `decode_bench.py` marks it `** SUSPENDED->serial` and prints a
  `DFlash-only` mean beside the headline. The headline *includes* the suspended
  row on purpose: it is a content-mix figure, not a peak.
- **`prose` text differs between the serial and DFlash arms**, and the DFlash
  arm's prose is the one cell that is not stable run-to-run. Committed tokens
  are supposed to be drafter-independent, so this is a known open defect, not a
  configuration error. It predates this branch and is confined to that row —
  every other prompt agrees byte-for-byte across arms and across runs.

**Decode throughput is a content-mix property, not one number.** On the same
stack, the spread across prompts is far larger than any lever in this repo:

| content | tok/s | accept |
|---|---|---|
| repetitive | 46.3 | high |
| easy code / common algorithms | ~40–44 | ~3.2 |
| novel logic | ~33–37 | ~2.6 |
| prose | 20.7 | 0.58 |

Per language, on genuinely speculative rows: Python 43.9, C 42.3, Go 39.5. The
spread per *task* (1.7×) dwarfs the spread per *language* (1.11×). Note that
the prediction going in was the opposite — that C and Go boilerplate would be
more predictable and therefore accept more. It is recorded as falsified in
`lang_bench.py` rather than quietly dropped, which is the only reason the rest
of that docstring is worth trusting.

---

## Five things that will cost you a day if you don't know them

These are the failure modes that produced clean-looking, wrong measurements.
Each one is now a guard somewhere in this directory.

### 1. A suspended request is a *serial* request

`ATLAS_DFLASH_ADAPTIVE` suspends speculation when the rolling 12-step mean
accept falls below the threshold, and serial-decodes until a re-probe. This is
correct behaviour — it is what keeps DFlash from ever being much slower than
plain decode.

The problem is the reporting. `accepted=` is logged on **speculative steps
only**, so `mean_accept` is an average over the speculative *subset* of a
decode. A suspended request still reports a small, entirely plausible accept
figure describing the ~7% of its decode that speculated, and reads exactly like
a low-accept DFlash row. Average it into a DFlash table and the mean drops for a
reason the table does not name.

The guard is nearly free. Each speculative step commits `accepted + 1` tokens,
so for a row that really ran DFlash end to end:

```
steps * (mean_accept + 1) / completion_tokens  ≈  1
```

Measured over a 15-row sweep: 13 genuine rows at 0.93–0.98, 2 suspended rows at
0.17 and 0.20. No ambiguous middle. `benchenv.spec_fraction` computes it and
`is_dflash` cross-checks it against the serve's own `adaptive spec: SUSPENDED`
tracing, because one signal alone is not evidence.

The same error has a second form: speculative decode is disabled whenever more
than one sequence is active, so **every row of a concurrency table above C=1 has
DFlash off**. Two engines in one table, again.

### 2. `/health` says ready on somebody else's serve

The port has a shared default. Another serve bound to it answers `/health`
identically, so readiness proves the port is answering — never that it is
answering with *your* binary. Foreign traffic co-batched into your measurement
window can fake a −60% regression.

`laguna_lock` is what makes the port yours. It flocks fd 9, and the serve
inherits the descriptor deliberately: an orphaned serve keeps the lock, so the
next arm refuses to start rather than racing a process it cannot see. A stale
lock means "kill the serve", not "delete the lock file".

**The lock stops foreign *serves*. It does nothing about foreign *clients*.**
These are separate threats and only the first one has a mutex. Nothing prevents
another process on the box from sending requests to a port you legitimately
hold, and under `--max-batch-size 1` the scheduler is FIFO, so those requests
queue *ahead* of yours. `decode_bench.py` times from request to response, so
the queueing is billed to the model as decode time.

This is not hypothetical and it is not loud. On 2026-07-31 nine foreign
requests (five capped at 16 tokens, ~1/sec) landed between two prompts of the
`nogproj` arm. One row reported **15.8 tok/s where the serve's own `Done:` line
said 33.6** — and its completion hash was byte-identical, so every correctness
check passed. Re-running reproduced it. Same output, half the speed, nothing
flagged. The token-weighted arm total read 25.6 instead of 30.5.

Two defenses, both now in the harness:

- `decode_bench.py` censuses the serve log and refuses to write a scoreable
  JSON unless the serve completed *exactly* the requests it issued (one warmup
  plus the suite). A contaminated arm is diverted to `<arm>.json.contaminated`,
  which makes `repro_table.sh` refuse to score the campaign rather than print a
  plausible table. Diverting matters more than a nonzero exit: a printed table
  is read as a result no matter what follows it.
- `export LAGUNA_PORT=<something private>` if you share the box. `8890` is a
  default that a lot of unrelated scripts also default to, which is precisely
  why it keeps happening. Moving ports is the only fix that does not depend on
  the other process cooperating.

Census the *requests*, not a content feature of the traffic. An earlier guard
keyed on whether requests were thinking and was wrong in both directions: our
own traffic thinks by default, and the serve prints `Thinking start token` at
boot whether or not any client ever connects.

`assert_decode_clean.sh` is the standalone form of the same census, for logs
this bench did not write — `gate_run.sh` logs, where the tool-eval phase adds
~38 more requests and the count has to be cut at the decode/eval boundary
first. It is what `gate_ab.sh` calls (advisorily, with `|| true`). It expects a
hardcoded `WANT_DONE=7`; the in-bench census derives the same number from
`len(SUITE)`, so adding a prompt cannot silently desynchronise it.

One case a completion census cannot see: a foreign request that started before
your window and had not finished inside it emits no `Done:` line to count. It
still shows up as an unexplained slow row — which is what sent us looking.

### 3. ARMED is not DISPATCHED

A gate string present in the binary proves only that the export is not a silent
no-op. It does not prove the code path ran. Kernels can be built and wired and
still never dispatched — because a precondition in the model config vetoes them,
or because the model has zero layers of the type the kernel optimises, or
because a `min` threshold sits above the observed range so the branch never
fires.

Log *built* and *took-it* separately, and check the phase counts in a profile
before crediting a layer-specific lever.

The inverse trap is worse and less checked: a launcher exporting a gate its
binary predates, or a binary that never had it. `serve_prod.sh` derives the gate
list from its own `export` lines and greps the binary for each one, rather than
trusting a name from memory — checking a *remembered* gate name is its own
failure mode, and a gate that does not exist fails into a false bug hunt rather
than an error.

Note the implementation detail: it uses `grep -cF … || true`, never `grep -q`.
Under `set -o pipefail` a `-q` match makes `strings` take SIGPIPE and the
pipeline report failure — which would mark *every* gate absent. The guard
against silent no-ops getting its own silent failure.

### 4. Zero differences is not zero comparisons

A count of zero that means "nothing to see" must never be readable as "nothing
wrong". The accept-histogram scrape in this harness once required an
`accepted=3/5` spelling while the serve emitted `accepted=3)`. It matched
nothing for an entire campaign: every arm's distribution came back empty, every
arm therefore looked histogram-identical, and the comparison that was supposed
to discriminate a numerics divergence silently never happened.

Assert non-empty **inputs**, keep "ran but ungraded" as a state distinct from
"graded and clean" (`is_dflash` returns `None`, not `False`), and test a guard
against a deliberately broken input — a guard that has never fired is not a
guard, it is a comment.

### 5. Know your noise floor, and know what it belongs to

An A/A floor belongs to a specific harness, config and session — it is not a
constant you can carry between benchmarks. On the 69-case tool-eval, same binary
and same environment across two independent serves, the real floor is **zero**:
82/82, not one case of 69 flipped. On decode tok/s under the lock it is about
±1%.

A widely-quoted "~5-point eval floor" turned out to be a misattribution: two
*different binaries* had been declared equivalent because six short greedy
prompts produced matching decode hashes. Six prompts do not prove equality over
69 multi-turn scenarios.

---

## Configuration notes

`γ=6` is a swept optimum and both directions were measured to lose: γ=8 costs
−6.7% and γ=10 costs −38.1%, as verify time is superlinear in the verify width
with a knee between K=9 and K=11, while accept saturates around 3.35. γ≤4 does
not degrade gracefully toward serial decode — it breaks the block size the
drafter was trained for and lands ~40% *below* it.

`--max-batch-size 1` is likewise measured, not a default. Because speculation is
off whenever more than one sequence is active, concurrency 2–3 collapses to
~0.6× single-stream, and batched serial decode only overtakes single-stream
DFlash somewhere between C=5 and C=6. Higher concurrency buys aggregate
throughput at several times worse per-request latency, which is the wrong trade
for interactive use and the right one for batch.

Two precision mirrors are on and both pay: FP8 attention is worth ~16% tok/s and
an FP8 `lm_head` ~6%, roughly additive, and accept moves by +0.004 and −0.122
respectively — the wrong sign for "quantization hurts". Every byte cut that
*did* hurt (3-bit experts, drafter FP8, shared-expert FP8) was measured and never
shipped. There is nothing here to buy quality back by restoring precision.

## Housekeeping

Run artifacts land under `bench/laguna/ab/` (override with `LAGUNA_OUT`) and are
gitignored: serve logs, profiles and result JSON embed absolute paths and device
topology, are regenerated per run, and are useless to anyone else. The scripts
are the deliverable; what they emit is not.
