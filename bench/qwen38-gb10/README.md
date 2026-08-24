# Qwen3.8-27B single-stream decode on GB10 — reproduction

Reproduces the **63.9 tok/s** median single-stream decode result for
Qwen3.8-27B on a DGX Spark (GB10).

```bash
git clone https://github.com/TheApathy/apathy-atlas.git
cd apathy-atlas
git checkout perf/qwen38-gb10-dflash
```

Related work: Avarok-Cybersecurity/atlas#648 integrates DFlash2 for the same
model on the same hardware and reports 54.5 tok/s. That PR is open and
unmerged, and this branch does not contain its commits — the two are
concurrent, independent integrations against the same DFlash2 release. It is
the most useful external reference point for the numbers below.

## Hardware and build

GB10 / DGX Spark, unified memory, `sm_121f`. The result is bandwidth-bound at
roughly 273 GB/s, so it does not transfer to discrete-GPU parts.

**Build the kernels for the right target.** This is the single most common way
to get a wrong result here:

```bash
# nvcc must be ON PATH. cudarc's build script resolves the CUDA version by
# shelling out to a bare `nvcc --version`; it does NOT consult CUDA_HOME. A
# clean checkout has no cached build-script output, so this fails there even
# though it succeeds on a tree that has built before.
export PATH=/usr/local/cuda/bin:$PATH

touch crates/atlas-kernels/build.rs
ATLAS_TARGET_MODEL=qwen3.8-27b ATLAS_TARGET_QUANT=nvfp4 \
  cargo build --release -p spark-server
```

Without those two variables the build **silently** defaults to
`qwen3-next-80b-a3b` and produces a binary that will not serve this model. The
build log line is the check:

```
atlas-kernels: compiled N kernels for target 0 (gb10, qwen3.8-27b, nvfp4)
```

If that line names any other model, the binary is wrong. `target/release/spark`
is rebuilt in place by any other build in the tree, so pin or re-check it
immediately before measuring rather than trusting an earlier build.

## Weights and drafter

- **Target**: `unsloth/Qwen3.8-27B-NVFP4` @ `7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108`.
  Upstream super-squashed the repo on 2026-08-15, so older revision hashes
  (including ones cited by earlier submissions) now 404. Pin this revision.
- **Drafter**: `drafter-qwen38-v2-epoch4-step24852` — DFlash family, 69 tensors,
  3.96 GiB BF16, 6 layers, `hidden_size` 5120, `vocab_size` 248320,
  **`block_size: 16`**. The engine quantises 6 layers x 7 dense + fc to NVFP4 at
  load; the BF16 sources are retained.

  `block_size` determines the usable draft width: `trained_drafts =
  block_size - 1`, so `--dflash-gamma 15` is both optimal and maximal, and
  `--dflash-gamma 20` is refused by the loader. Point a different drafter at
  this and you must re-derive gamma from *its* `block_size` — the clap default
  is wrong for any drafter and silently degrades acceptance rather than failing.

  **This drafter is not currently published.** It is a 4 GB local artifact, so
  the exact 63.9 figure is not reproducible from a clean checkout by an outside
  party without it; ask and we will publish it. The public
  [`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)
  drafter runs in this configuration and is the honest substitute, but it is a
  different drafter and acceptance — hence tok/s — will differ. Everything else
  in this harness is exact.

## Run

```bash
MODEL_DIR=/path/to/Qwen3.8-27B-NVFP4 \
DRAFT=/path/to/dflash-drafter \
./bench/qwen38-gb10/serve.sh

# wait for the health endpoint, then:
python3 bench/qwen38-gb10/weschera_minheap_repro.py \
  --endpoint http://127.0.0.1:8896/v1/chat/completions \
  --output /tmp/minheap.json \
  --repetitions 5 --max-tokens 400
```

The probe is single stream, greedy (temp 0), thinking off, and reports a
median. It also hashes the completion so runs can be compared byte-for-byte;
at temp 0 the hash should be identical across repetitions, and a differing hash
means something in the configuration is non-deterministic and the timing
numbers should not be trusted.

## Expected

| Configuration | 400-token probe | 1500-token probe |
|---|---:|---:|
| Historical reference | 51.26 | 41.22 |
| This configuration | **63.96 / 63.77** | **45.94** |
| Same, `ATLAS_DFLASH_DRAFT_SPLITK` unset | 62.86 / 62.92 | — |

Two figures are quoted for the 400-token production row because they were
measured interleaved in a single session; the spread is run-to-run drift, not
two distinct configurations.

The 1500-token figure was **re-measured on 2026-08-24** at 45.94 median
(44.54 / 45.94 / 46.21 / 46.12 / 45.92, 5 reps, deterministic, container-served
from `serve.sh`). It replaces an earlier 43.33 that was taken before `serve.sh`
was corrected to the full measured environment, and therefore described a
configuration that no longer ships.

Decode rate falls with generation length because speculative acceptance falls
with it. Measured 2026-08-24 on the configuration in `serve.sh`, from the
engine's own `SPEC_CYCLE_V2` per-cycle telemetry (`ATLAS_DFLASH_SPEC_CYCLE_V2=1`,
which `serve.sh` sets), 3 repetitions per length:

| max_tokens | decode tok/s | cycles | accepted / γ=15 | acceptance |
|---:|---:|---:|---:|---:|
| 400 | 63.98 | 144 | 7.02 | **46.8%** |
| 800 | 54.55 | 343 | 5.89 | **39.3%** |
| 1500 | 46.33 | 753 | 4.94 | **32.9%** |

### The per-position hazard is not flat

An earlier revision of this file claimed the per-position conditional match rate
was "flat at 0.87 over 11k cycles". **That is wrong in shape**, and only
coincidentally close in magnitude. Measured over 1,240 pooled cycles:

| position | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| hazard | .90 | .83 | .85 | .83 | **.78** | .84 | .84 | .92 | .91 | .90 | .97 | .91 | .99 | .92 | .90 |

The curve is U-shaped: it declines to a minimum around position 5, then **rises**
through the tail, reaching 0.9–0.99 past position 8. The same shape appears
independently at all three generation lengths; only the level shifts (mean
hazard 0.913 at 400 tokens, 0.895 at 800, 0.872 at 1500). The "0.87" in the old
claim is approximately the *1500-token mean*, quoted as if it were a universal
constant.

The rising tail is the interesting part and it is survivor bias in the useful
sense: cycles that survive past position 7 are disproportionately the
structurally predictable ones (indentation, closers, boilerplate), so continuing
gets *easier* conditional on having got that far. Acceptance is therefore
bimodal rather than uniformly mediocre — a large population of cycles that die
in the first few positions, and a smaller one that runs nearly to the full
γ. That distinction matters for drafter work in a way "flat at 0.87" does not:
the win is in the cycles that die early, not in extending the ones that already
run long.

Method note: the engine emits an `accepted` count per cycle, not per-position
match flags. Because flat DFlash accepts a prefix up to the first mismatch, the
accepted-count distribution determines the discrete hazard exactly —
`h_i = P(accepted ≥ i) / P(accepted ≥ i−1)` — so this is a derivation from the
emitted data, not a fit.

Every `ATLAS_*` variable the profile sets is listed in [`FLAGS.md`](FLAGS.md),
with its value and the description from the source that reads it.

## The headline number is workload-specific

**63.9 tok/s is the MinHeap probe, and the MinHeap probe is close to the best
case.** Speculative decoding pays off in proportion to how predictable the next
tokens are, so the decode rate is a property of the *workload*, not of the
engine alone. Measured on the published container, temp 0, 400 tokens,
thinking off, three repetitions each (spread under 0.1 tok/s):

| Workload | tok/s |
|---|---:|
| MinHeap class + complexity (the probe) | 58.5 |
| Arithmetic word problem with algebra | 50.9 |
| SQL query + explanation | 35.8 |
| Rust IPv4 parser | 35.1 |
| Multi-constraint logic puzzle | 34.4 |
| Security explanation (JWT `alg:none`) | 24.9 |
| Open prose, three paragraphs | **18.2** |

Median across these is ~35 tok/s and the spread is 3.2x. Boilerplate-heavy code
drafts extremely well; open prose barely drafts at all and runs near the
no-speculation floor.

This is why the probe is a fixed prompt: it is a *comparison* instrument, and
every figure in this repo and in the upstream PRs it is compared against uses
the same prompt. It is not a promise about your workload. If you are sizing for
prose or chat, plan against the low end of that table, not the headline.

(The 58.5 above is the same configuration as the published 63.95; the difference
is the request protocol — the probe sends `reasoning_effort: none`, this table
sends `enable_thinking: false`.)

## Measurement notes

- **Do not A/B sequentially on a warm box.** Step time `S = tokens_per_step /
  overall_tok_s`; a sequential A/B produces phantom regressions from thermal and
  cache state. Interleave the arms.
- `ATLAS_DFLASH_DRAFT_SPLITK=8` is worth +0.98 tok/s but is **not bit-exact** —
  it changes reduction order in the drafter GEMMs. Everything else in
  `serve.sh` is bit-exact. Drop it if you are validating numerics rather than
  speed.
- `ATLAS_WEIGHT_CACHE=1` caches post-transform weights (~13 GB, LRU-bounded).
  It affects load time only (17 s vs 45-60 s), not decode.
- Watchdogs are disabled in this profile because each was measured terminating
  healthy output. They are unrelated to the speed result; leave them on for
  quality work.

Full per-flag evidence, the cycle decomposition, and the levers that measured
null are in [`docs/QWEN38_PERFORMANCE_RECIPE.md`](../../docs/QWEN38_PERFORMANCE_RECIPE.md).
