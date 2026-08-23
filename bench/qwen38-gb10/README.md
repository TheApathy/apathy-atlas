# Qwen3.8-27B single-stream decode on GB10 — reproduction

Reproduces the 63.9 tok/s median single-stream decode result on a DGX Spark
(GB10). This builds directly on the DFlash2 integration from #648; that PR is
the baseline this one is measured against.

## Hardware and build

GB10 / DGX Spark, unified memory, `sm_121f`. The result is bandwidth-bound at
roughly 273 GB/s, so it does not transfer to discrete-GPU parts.

**Build the kernels for the right target.** This is the single most common way
to get a wrong result here:

```bash
touch crates/atlas-kernels/build.rs
ATLAS_TARGET_MODEL=qwen3.8-27b ATLAS_TARGET_QUANT=nvfp4 cargo build --release
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
- **Drafter**: a DFlash-family drafter with `block_size: 16`. `block_size`
  determines the usable draft width: `trained_drafts = block_size - 1`, so
  `--dflash-gamma 15` is both optimal and maximal. `--dflash-gamma 20` is
  refused by the loader.

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
| This configuration | **63.96 / 63.77** | 43.33 |
| Same, `ATLAS_DFLASH_DRAFT_SPLITK` unset | 62.86 / 62.92 | — |

Two figures are quoted for the production rows because they were measured
interleaved in a single session; the spread is run-to-run drift, not two
distinct configurations.

Decode rate falls with generation length because speculative acceptance falls
with length (measured 48.2% at 400 tokens, 38.7% at 800, 32.5% at 1500). The
per-position hazard is flat at 0.87 over 11k cycles, so this is a property of
the drafter, not a leak or a degradation over the run.

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
