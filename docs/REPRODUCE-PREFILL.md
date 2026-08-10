# Reproducing the DeepSeek-V4-Flash prefill numbers (792 tok/s @ N=2410)

Everything below runs from a clean checkout of `combined-residency` on one
GB10 (DGX Spark, sm_121). No hidden state: the only inputs are the model
directory and the env flags listed here.

## 0. Prerequisites

- One GB10, 128 GB unified memory, no other process holding the GPU
  (a resident server will OOM the microtests — kill it first).
- `DeepSeek-V4-Flash-162B` at `/home/flocka/models/DeepSeek-V4-Flash-162B`
  and the 0731 drafter at `.../DeepSeek-V4-Flash-0731-drafter`.
- `tool-eval-bench` (the quality gate) — any recent build.

## 1. Build — the flag that is NOT optional

```bash
ATLAS_TARGET_MODEL=deepseek-v4-flash cargo build --release
```

**The default target is `qwen3-next-80b-a3b`.** A plain `cargo build --release`
compiles the wrong kernel set and the server dies at boot with
`No compiled kernel target matches model_type 'deepseek_v4'`. Microtests can
mask this because `kernels/gb10/common/` compiles into every target set.

## 2. Serve

```bash
scripts/dsflash-serve-bench.sh <run-name> 5 \
  ATLAS_V4_ATTN_NVFP4=1 ATLAS_V4_ATTN_RELEASE_BF16=1 \
  ATLAS_MTP_GATE_FORCE=1 ATLAS_DFLASH_LOW_GEAR=1
```

Both attention flags are required TOGETHER (`RELEASE_BF16` alone aborts with
"requires successful NVFP4 transcodes"). The pair is quality-positive and
frees ~8 GiB. Boot takes ~6 minutes (weight load + NVFP4 transcode); wait for
`/health` rather than a fixed sleep.

Flags that select the optimizations under test (all DEFAULT ON — listed so a
bisect can turn them off one at a time):

| flag | `=0` restores | landed in |
|---|---|---|
| `ATLAS_V4_PREFILL_TC` | scalar prefill attention | `6d8216a3`, `889abbfe` |
| `ATLAS_V4_COMP_GEMM_TC` | scalar compressor GEMMs | `1b4bd7f2`, `0e368e6f` |
| `ATLAS_V4_KV_PIPELINED` | scalar kv_proj GEMM | `415734fa` |
| `ATLAS_V4_WOA_INPLACE` | wo_a gather/scatter | `af92018a` |
| `ATLAS_HC_TILED` | one-block-per-token `hc_pre` | `17481eaa` |
| `ATLAS_VERIFY_EXACT_GEMV` | drifting batched verify GEMVs | `090830de` |

## 3. Measure prefill

```bash
python3 scripts/prefill_probe.py 8977        # TTFT at ~1k/2k/3k tokens
```

or, for the exact 2410-token figure quoted in the campaign doc, the
five-run median (warm up with one short request first — the first request
after boot pays lazy-init):

```
prompt: "Summarize in one sentence: " + 120 x
        "Fact {i}: division {i} reported revenue of {1000+i*7} units at
         margin {10+i%20} percent."
max_tokens=4, temperature=0, stream=true
prefill tok/s = prompt_tokens / TTFT
```

Expect **TTFT 3.03–3.10 s → 780–795 tok/s**. Run-to-run spread is ±1%;
report the median of 5, not a single run.

## 4. The quality gate — mandatory before believing any number

```bash
tool-eval-bench --base-url http://127.0.0.1:8977/v1 --short --no-live \
  --json-file /tmp/teb.json
```

The bar is **90/100 with 0 failures** (12 pass / 3 partial). TC-14 is the
known-borderline scenario that flips on numerics; the other 14 must not move.
A speed change that drops this is a regression regardless of tok/s.

## 5. Profiling (how the campaign found its targets)

```bash
scripts/dsflash-serve-bench.sh prof 5 <flags...> ATLAS_PROFILE=1
# run one prefill, note the wall-clock start time, then:
python3 scripts/prefill_waterfall.py serve-prof.log <seconds-since-midnight>
```

Prints the per-bucket waterfall. Sanity check that validates the method: the
`xw_ffn_block` wrapper must equal the sum of its interior buckets to <1 ms.

```bash
ATLAS_GEMM_SHAPE_LOG=1   # every unique (kernel, M, N, K) logged once
```

This is the tool that found most of the campaign's wins — it shows which
kernel each projection ACTUALLY dispatches, which repeatedly was not the
fastest available one. Cross-reference each logged shape against
`dense_gemm_microtest`.

**`nsys` cannot trace prefill on this model** — the ~6-minute weight load
overruns its buffers and the trace is silently dropped, `--delay` included.
`ncu` is not installed. The in-tree probes above are the instrument.

## 6. Kernel-level oracles (run with the GPU free)

```bash
# GEMM kernel comparison at any shape (cosine vs CPU + CUDA-event timing)
cargo run --release -p spark-model --example dense_gemm_microtest \
  --features cuda,gpu-examples -- <kernel> <M> <N> <K>
#   kernels: dense_gemm_bf16 | dense_gemm_tc | dense_gemm_bf16_pipelined | ...

# The campaign's headline comparison:
#   dense_gemm_tc            2410 1024 4096 -> 8.07 ms
#   dense_gemm_bf16_pipelined 2410 1024 4096 -> 0.44 ms   (cosine 1.000000)

cargo run --release -p spark-model --example prefill_attn_tc_microtest \
  --features cuda,gpu-examples -- 0x7C21 <S> <window> <ratio>
cargo run --release -p spark-model --example hc_pre_microtest --features cuda,gpu-examples
cargo run --release -p spark-model --example w8a8_gemm_microtest --features cuda,gpu-examples
cargo run --release -p spark-model --example w4a16_gemv_grouped_microtest --features cuda,gpu-examples
```

## 7. Artifacts

Run output (`serve-*.log`, `probe-*.json`, `*.bin`) is **gitignored** — it
differs every run and must never gate a reproduction. Curated results that
back a claim in the docs are committed under `docs/probes/`.

## 8. What the numbers are NOT

- Not a single-run best: everything quoted is a median with the spread stated.
- Not comparable to the reference stack's `1055 tok/s` — that figure is at
  252,047 tokens; ours is at 2,410. The prefill curve here is nearly flat with
  length (590 @ 1525 vs 610 @ 3065 measured mid-campaign) because the HCA
  layers attend full-causally, so length is not an escape hatch in either
  direction.
- Not decode: decode numbers live in `docs/SESSION-STATE.md` and are measured
  with `scripts/decode_ab_probe.py`.
