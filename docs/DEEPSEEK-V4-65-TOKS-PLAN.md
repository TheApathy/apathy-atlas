# DeepSeek-V4 Flash: path to 65 tok/s on one Spark

This branch targets 65 generated tokens per second without treating the Qwen
result as directly transferable. Qwen3.8's largest gains came from its hybrid
GDN/attention layout; DeepSeek-V4 is a 43-layer MLA/MoE model and needs a
DeepSeek-specific target path plus useful speculative acceptance.

## Evidence and current gap

| Runtime/profile | Single-stream decode |
| --- | ---: |
| Atlas plain EXL3 K2 | 22.90-22.98 tok/s |
| Atlas warm residency profile | 24.73-28.40 tok/s |
| Mia/vLLM K2-v1 controlled short decode | 58.1 tok/s |

The Atlas warm result is not a quality-neutral baseline: its lossy NVFP4
attention-residency path changed three of four output hashes. The older Atlas
speculative measurement also used raw gamma 5, which means four proposals in
Atlas, despite the checkpoint containing a five-token DSpark block. Commit
`be97db83` corrects the launch contract to five proposals / six verify rows.

The current public Mia evidence does not establish a general 65 tok/s result.
K2-v1 reached 58.1 tok/s on its controlled short decode; content medians ranged
from 23.91 tok/s for creative prose to 53.37 tok/s for code. The 80+ tok/s
repetition row failed its correctness target, and C6 figures are aggregate
throughput rather than the single-stream number targeted here. Mia still proves
that B12X/Trellis small-M MoE kernels plus the native DSpark path can more than
double Atlas's plain result. It does not establish that DFlash2 weights exist
for DeepSeek or that Qwen DFlash2 weights can be reused.

The exact upstream commit is reproducible with
`scripts/build-mia-deepseek-runtime.sh`. The resulting image is GPU-free to
build. `scripts/mia-deepseek-serve.sh` requires an explicit `--allow-gpu`,
refuses a busy GPU by default, mounts the audited local K2 checkpoint, and uses
Mia's one-Spark 1M-context settings. `bench/deepseek-v4/mia_decode_sweep.py`
records per-content decode medians and output hashes once a GPU window exists.

## Immediate Atlas lane

The default K2 speculative launcher now enables the portable, already-present
DeepSeek fast paths:

- fused V4 decode;
- compile-time verify GEMV;
- MoE multi-row partitioning;
- adaptive speculation with low-gear fallback;
- DSpark capture and the checkpoint-native five proposal rows.

These are the relevant pieces of the Qwen performance recipe. Qwen's GDN
kernels are not applicable to DeepSeek MLA and should not be copied into this
branch.

The next controlled GPU run should first establish a byte-locked plain K2
baseline, then measure the corrected DSpark launch with the same prompt,
generation length, clocks, server build, and sampling settings. Record target
decode time, draft time, verify time, average accepted tokens, and fallback
rate. A headline tok/s result without those counters cannot distinguish a
target-kernel problem from a low-acceptance drafter.

## DeepSeek-native DFlash2 checkpoint ABI

Atlas's block-diffusion DFlash head currently shares the target embedding and
LM head. A compatible checkpoint therefore needs all of the following:

- target hidden size `4096`;
- target vocabulary size `129280` and the same tokenizer/token IDs;
- capture IDs using HF hidden-state indexing in the range `1..=43`;
- an `fc.weight` trained for
  `4096 * number_of_capture_layers` input columns;
- a valid mask token inside the 129280-token vocabulary;
- a declared trained block/verify width and matching attention semantics;
- DeepSeek-trained weights. Qwen3.6 DFlash2 weights are structurally invalid.

The factory now rejects incompatible pairings before model scratch or KV-cache
construction. Cross-vocabulary `d2t` remapping is intentionally rejected
because Atlas detects such tables but does not yet execute the remap.

The EXL3 launcher accepts a separate checkpoint through `DRAFTER=/path` and
infers the native DFlash2 capture contract. Embedded K2 continues to use
`DRAFTER_KIND=dspark`; a native DFlash2 launch does not allocate or update the
unrelated three-row DSpark HC-mean capture buffer.

The branch now compiles an EXL3 MROW=16 target-verify arm and sizes DFlash
scratch independently of the legacy `ATLAS_DFLASH_BATCH_MOE` experiment flag.
This removes the previous row-9 cliff where a 16-row drafter fell back to
re-streaming every routed expert once per row. It is implementation readiness,
not a performance claim: GPU parity hashes and dispatch proof remain promotion
gates. The separate MXFP4 `_t` ladder remains capped at its proven eight rows.

With `ATLAS_VERIFY_GEMV_V2=1`, the EXL3 shared expert now uses Atlas's exact
grouped-batch NVFP4 GEMV at the six-row K2 verify width instead of three GEMVs
per row. A compile-time M=16 sibling serves the DFlash2 width without inflating
the K2 kernel's register footprint. The arithmetic order matches the single-row
kernel by construction; live byte-parity and timing are still required before
promotion.

The next verifier experiment's 48-byte persistent expert-major work record is
now a production module rather than a microbenchmark-local struct. CPU gates
pin its 16-byte alignment, little-endian round trip, legal metadata bits,
six-row ceiling, uniform top-k shape, unique experts within each row, and expert
bounds. The microbenchmark consumes that shared ABI and still compiles. This is
host-planning readiness only: production dispatch remains disabled until the
live kernel clears the 213 GB/s microbenchmark floor and exact per-row parity.

The DSpark hc-mean capture is now a 256-row circular serve history rather than
`max_seq_len` rows. The checkpoint attends to only 128 capture positions, so
the 1M-context allocation falls from 24.6 GB to 6 MiB without changing the
drafter's visible history. Offline `ATLAS_DSPARK_DUMP` remains linear and
absolute-positioned. The boundary-slot workaround also expires once that
position leaves the attention window; previously it kept zeroing a reused ring
slot after 128 generated tokens.

## Promotion gates

1. Plain baseline is reproducible across at least three warm trials.
2. Speculation uses the trained proposal count and reports nonzero dispatch.
3. Greedy outputs match the locked plain baseline when the path claims exactness.
4. Each approximate/quantized arm gets a separate quality score; hashes alone
   do not promote it.
5. The speculative arm beats plain on median decode tok/s and does not rely on
   adaptive fallback for the reported headline.
6. The 65 tok/s claim names prompt length, output length, concurrency, context
   residency, quantization, and acceptance rate.
7. The circular capture path passes long-context wrap parity on live GPU before
   the speculative 1M profile is promoted.

The two-phase release harness keeps those gates usable on a single GPU. Run
`bench/deepseek-v4/dflash_release_gate.py run` once against plain and once
against DFlash2, using identical `--max-tokens` and `--reps`, then compare the
JSON files. Comparison requires exact per-prompt output hashes, defaults to a
65 tok/s median single-run decode floor, rejects identical plain/candidate
implementation identities, and requires at least 3.0 committed tokens per verify
step from an `ATLAS_DSPARK_ACCEPT_LOG=1` server log. It never labels aggregate
throughput as single-stream decode. Both runs must name the same immutable
`--model-identity` and separately record their `--implementation-identity`.
Reasoning deltas participate in both decode timing and output hashing, avoiding
an inflated result when DeepSeek emits a long reasoning stream before content.
Acceptance is parsed only from server-log bytes appended during that run, so a
stale high-acceptance summary cannot promote a candidate; log truncation or
rotation during measurement also fails closed. Result files are
atomic and refuse overwrite unless `--overwrite` is explicit.

## Long context contract

The K2 checkpoint already declares 1,048,576 positions with 16x YaRN from an
original 65,536-token window. Plain K2 and embedded DSpark therefore default to
the full 1,048,576-token window (leaving generation room above a 1,000,000-token
prompt); native DFlash2 remains at 131,072 by default because its
five-layer context accumulator is not yet windowed.

`bench/deepseek-v4/context_sweep.py --plan-only` validates that config and
locks exact 8K, 128K, 250K, 512K, and 1M prompt hashes. The live pass places a
unique retrieval needle near the midpoint, records retrieval success, TTFT,
decode-only throughput, token counts, and output hashes. A memory-capacity
claim does not pass the 1M gate without retrieval success. Plan records pin the
config and tokenizer SHA-256 digests, refuse to overwrite by default, and are
published atomically; pass `--overwrite` only for an intentional rerun. Every
plan/live record requires immutable model and implementation identities, and a
live run fails if the server's prompt-token count drifts, decode is not
measurable, or retrieval misses.

Upstream implementation reference:
<https://github.com/tpurtell/ds4-mia-exl3-k2-1spark>
