# Qwen3.8-27B single-stream decode on GB10

Serve configuration and decode probe for Qwen3.8-27B (NVFP4) on a DGX Spark
(GB10) with DFlash speculative decoding.

## Build

The kernel target must be set explicitly:

```bash
touch crates/atlas-kernels/build.rs
ATLAS_TARGET_MODEL=qwen3.8-27b ATLAS_TARGET_QUANT=nvfp4 cargo build --release
```

Without those two variables the build silently defaults to
`qwen3-next-80b-a3b` and produces a binary that will not serve this model.
Check the build log line:

```
atlas-kernels: compiled N kernels for target 0 (gb10, qwen3.8-27b, nvfp4)
```

If it names another model, the binary is wrong. `target/release/spark` is
rebuilt in place by any other build in the tree, so re-check it immediately
before measuring.

## Weights and drafter

- **Target**: `unsloth/Qwen3.8-27B-NVFP4` @ `7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108`.
  Pin the revision — upstream super-squashed the repo on 2026-08-15, so older
  hashes 404.
- **Drafter**: any DFlash-family drafter. `block_size` determines the usable
  draft width (`trained_drafts = block_size - 1`), so derive `--dflash-gamma`
  from *your* drafter's `block_size`; the CLI default is wrong for any drafter
  and degrades acceptance silently rather than failing. A `block_size: 16`
  drafter takes `--dflash-gamma 15`.
  [`incoai/Qwen3.8-27B-DFlash2`](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2)
  is a public drafter that runs in this configuration.

## Run

```bash
MODEL_DIR=/path/to/Qwen3.8-27B-NVFP4 DRAFT=/path/to/dflash-drafter \
  ./bench/qwen38-gb10/serve.sh

python3 bench/qwen38-gb10/weschera_minheap_repro.py \
  --endpoint http://127.0.0.1:8896/v1/chat/completions \
  --output /tmp/minheap.json --repetitions 5 --max-tokens 400
```

Single stream, greedy, thinking off, reports a median. The probe hashes each
completion: at temperature 0 the hash should be identical across repetitions,
and a differing hash means the configuration is non-deterministic and the
timings should not be trusted.

## Measurement notes

- Decode rate falls with generation length, because speculative acceptance
  falls with length. Compare like with like — same `--max-tokens`.
- Don't A/B sequentially on a warm box. Step time is
  `S = tokens_per_step / overall_tok_s`; sequential arms produce phantom
  regressions from thermal and cache state. Interleave the arms.
- `ATLAS_DFLASH_DRAFT_SPLITK` is not bit-exact (it reassociates the K loop).
  Unset it when validating numerics rather than speed.
- `ATLAS_WEIGHT_CACHE=1` affects load time only, not decode.
- The watchdogs disabled in `serve.sh` are unrelated to the speed result and
  should be left on for quality work.
