# Qwen3.8-27B single-stream decode on GB10 — reproduction

Reproduces measured single-stream decode profiles for Qwen3.8-27B on a DGX
Spark (GB10). The former **72.2518 tok/s** v3/NVFP4-KV headline is retained
below only as quarantined historical evidence: its Rust launcher passed 13
arguments to an 11-argument contiguous-prefill attention PTX. The inserted
`query_start=0` was decoded as `num_q_heads=0`, so every chunk-0 attention CTA
returned without writing output. That trajectory and its `f51d8358...` hash
are not valid performance controls.

The matched-ABI clean-HEAD Weschera baseline is **55.3824 tok/s** over five
deterministic 1,500-token runs. The current corrected tree reaches about
**58.012 tok/s** with all new prefill routes disabled. Its default-off M17
attention-QKV activation-staging candidate reaches **59.4598 tok/s** with the
same corrected-tree output hash, a 2.25% gain. Reaching 72 from the valid
58.012 control needs another 24.1%; reaching 85 needs 46.5%. Verification and
drafter acceptance against the corrected target trajectory are the decode
targets; prefill-only kernels cannot close that gap.

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

  **Both drafters are published.**

| Drafter | `--dflash-gamma` | tok/s |
|---|---:|---:|
| `onewhosighs/Apathy-Qwen3.8-27B-DFlash-drafter-v2` (ours) | 15 | 63.9 |
| `incoai/Qwen3.8-27B-DFlash2` (public, different family) | 7 | 43.9 |
| `onewhosighs/Apathy-Qwen3.8-27B-DFlash-drafter-v3` (ours) | 15 | 58.012 corrected-ABI control; historical 72.2518 is invalid |

Derive gamma from your drafter's `block_size`; the DFlash2 drafter refuses 15
and needs `GAMMA=7`. Everything else in this harness is exact.

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

Target-prefill and V3 decode qualification use the dedicated strict harness.
Record the no-spec control first:

```bash
# Restart with enough admitted context plus continuation headroom for the
# largest sweep row.
unset DRAFT
CONTROL_NONCE="$(openssl rand -hex 32)"
CONTROL_LOG="$(mktemp /tmp/qwen38-no-spec.XXXXXX.log)"
ATLAS_QUALIFICATION_RUN_NONCE="$CONTROL_NONCE" \
MAX_SEQ_LEN=65536 MAX_PREFILL_TOKENS=8192 \
MODEL_DIR=/path/to/Qwen3.8-27B-NVFP4 \
./bench/qwen38-gb10/serve-no-spec.sh >"$CONTROL_LOG" 2>&1 &
CONTROL_PID=$!

python3 bench/qwen38-gb10/prefill_decode_repro.py \
  --print-process-attestation "$CONTROL_PID" \
  --expected-binary target/release/spark \
  > /tmp/no-spec-process-attestation.json

python3 bench/qwen38-gb10/prefill_decode_repro.py \
  --endpoint http://127.0.0.1:8896/v1/completions \
  --provenance-manifest /tmp/no-spec-provenance.json \
  --route-log "$CONTROL_LOG" \
  --write-prefill-reference /tmp/no-spec-prefill-reference.json
```

Then stop the control and restart the same source, binary, kernel bundle,
target, tokenizer, and target-side flags with DFlash V3:

```bash
CANDIDATE_NONCE="$(openssl rand -hex 32)"
CANDIDATE_LOG="$(mktemp /tmp/qwen38-v3.XXXXXX.log)"
ATLAS_QUALIFICATION_RUN_NONCE="$CANDIDATE_NONCE" \
MAX_SEQ_LEN=65536 MAX_PREFILL_TOKENS=8192 \
MODEL_DIR=/path/to/Qwen3.8-27B-NVFP4 \
DRAFT=/path/to/Apathy-Qwen3.8-27B-DFlash-drafter-v3 \
./bench/qwen38-gb10/serve-v3-72tps.sh >"$CANDIDATE_LOG" 2>&1 &
CANDIDATE_PID=$!

python3 bench/qwen38-gb10/prefill_decode_repro.py \
  --print-process-attestation "$CANDIDATE_PID" \
  --expected-binary target/release/spark \
  > /tmp/dflash-v3-process-attestation.json

python3 bench/qwen38-gb10/prefill_decode_repro.py \
  --endpoint http://127.0.0.1:8896/v1/completions \
  --provenance-manifest /tmp/dflash-v3-provenance.json \
  --route-log "$CANDIDATE_LOG" \
  --prefill-reference /tmp/no-spec-prefill-reference.json \
  --output /tmp/qwen38-prefill-decode.json
```

The candidate manifest uses `runtime_mode: "dflash-v3"` and must include the
drafter index digest. Both manifests bind the source revision plus SHA-256
identities for the binary, compiled kernel bundle, model index, tokenizer, and
effective launch environment; the candidate adds `draft_index_sha256`. The
harness rejects missing/malformed identities and any common target identity
that differs between no-spec and V3. Each v5 manifest also contains the exact
canonical `effective_environment` object, its matching
`effective_environment_sha256`, a hash of the full process environment named
`full_environment_sha256`, the 64-hex `server_run_nonce`, and
`command_line_sha256`, plus `same_mode_environment_delta` (empty for
no-spec-to-V3). The structured object
enumerates every `ATLAS_*` variable except the qualification nonce; the full
hash covers all remaining variables without storing possible secrets. Keep
these manifests beside the result. The `--print-process-attestation` output
provides those six process-derived fields without printing non-Atlas environment
values. It waits until that PID has exec'd the expected binary and owns the
exact loopback listener, closing the background-launch race. Merge it with the
source, kernel bundle, model, tokenizer, runtime, route-marker,
comparison-delta, and (for V3) drafter identities.

`serve-no-spec.sh` and the legacy-named `serve-v3-72tps.sh` source the same versioned target
profile, so model name, NVFP4 KV policy, vocabulary, target kernels, and all
other target arguments remain identical. For the no-spec-to-V3 comparison, the
harness additionally requires the complete structured `ATLAS_*` maps to match,
then compares normalized command lines after removing only the canonical
`--dflash --draft-model ... --dflash-gamma ... --dflash-quantization nvfp4`
tuple. Its normalized full-environment comparison removes only the nonce,
`DRAFT`, and `RUNTIME_MODE`. Any other CLI, CUDA, thread, model, context, memory,
or kernel drift invalidates attribution to DFlash. The `72tps` filename is a
compatibility name only; it is not a current throughput claim.

Each manifest must also contain `required_route_markers`, an object mapping
every exact engagement-message fragment expected from that fresh server process
to its exact positive count. The corrected target-side prefill profile starts
with this set (all counts are one):

```json
{
  "required_route_markers": {
    "ENGAGED ATLAS_PREFILL_FFN_PIPE: ordinary exact pipe route": 1,
    "ENGAGED ATLAS_PREFILL_PROJ_PIPE: attention_q": 1,
    "ENGAGED ATLAS_PREFILL_PROJ_PIPE: attention_k": 1,
    "ENGAGED ATLAS_PREFILL_PROJ_PIPE: attention_v": 1,
    "ENGAGED ATLAS_PREFILL_PROJ_PIPE: attention_o": 1,
    "ENGAGED ATLAS_PREFILL_PROJ_PIPE: ssm_qkvz": 1,
    "ENGAGED ATLAS_PREFILL_PROJ_PIPE: ssm_out": 1,
    "ATLAS_SSM_PREFILL_PACK engaged: direct QKV/gates plus fused conv/Z replaces": 1
  }
}
```

The V3 manifest additionally declares the FC and dual-K/V fragments
`DFLASH_PREFILL_PIPE engaged for V3 FC context ingestion` and
`DFLASH_PREFILL_PIPE engaged dual V3 context K/V ingestion`, each once. Add any
enabled candidate's engagement marker as well. The harness reads the complete
log after the sweep, rejects a missing, repeated, ambiguous, or undeclared
engagement line, and stores only the full-log and engagement-line SHA-256 values
plus counts in the v5 reference/report. It resolves the exact 127.0.0.1 listener
socket and binds its PID/start time, executable, command line, environments,
stdout/stderr log inode, and nonce before and after model discovery and every
request. Use direct `>"$LOG" 2>&1` redirection: a `tee` pipeline gives the server
pipe file descriptors and is rejected. Use `tail -F "$LOG"` in another terminal
for live visibility. Every run needs a fresh `mktemp` log and nonce.

For an implementation-only M128-to-M256 check, compare fresh processes in the
same runtime mode. Add `--comparison-kind same-mode --runtime-mode no-spec` to
both commands (or `dflash-v3` to both). The reference writer records the M128
environment; the candidate command reads it with `--prefill-reference` after
enabling M256. Source, binary, kernel bundle, model, and tokenizer must match;
V3 also requires the same drafter index. Both manifests declare the same
non-empty `same_mode_environment_delta`: exactly one known kernel flag changing
from string `"0"` in control to `"1"` in candidate. That declaration must equal
the complete observed `ATLAS_*` difference, while the command line and the hash
of every other environment entry remain identical. A legacy no-spec-to-V3
reference cannot be reused for this gate.

For the current Q/K-norm plus RoPE candidate, both manifests use:

```json
{
  "same_mode_environment_delta": {
    "ATLAS_PREFILL_QKNORM_ROPE": {"control": "0", "candidate": "1"}
  }
}
```

The gate requires every 2K/8K/32K repetition to reach 2,000 effective prompt
tok/s with explicit zero cached tokens, produce at least 32 continuation
tokens, and hash-match the corresponding no-spec continuation. It then requires
deterministic 85 tok/s Weschera-style decode. A fast TTFT with corrupted
long-prompt state is therefore a failure. Same-mode candidates must also beat
control TTFT at both median and nearest-rank p90 in the 2K and 8K bins; merely
remaining above 2,000 tok/s while regressing the control is a failure.
`ATLAS_PREFILL_FFN_FUSED_EPILOGUE=1` is a compiled but
unqualified kernel candidate; leave it off for the control, then enable it for
the paired GPU run. It is not part of the corrected control profile yet.
`ATLAS_PREFILL_FFN_DUAL_FUSED=1` is the stronger default-off candidate: it
replaces both pipe projections plus SiLU with one exact large-M kernel. Test it
only after the control and up-only candidate; when both fusion flags are set,
the dual route takes precedence.

`ATLAS_PREFILL_FFN_FLASHINFER=1` is the qualified exact-Qwen3.8 alternative
for full M2079 and M8192 FFN chunks. Set the same absolute
`FLASHINFER_SM121_LIB` path in both arms and change only the enable bit. The
runtime requires the compiled-in native-library SHA-256 and loads a verified,
fully sealed memfd copy, not mutable path bytes. Candidate manifests must
replace the control E2M1 markers with these exact one-shot receipts:

```text
ATLAS_PREFILL_FFN_FLASHINFER ENGAGED layer=0 M=2079 tactics=4/4/4
ATLAS_PREFILL_FFN_FLASHINFER ENGAGED layer=0 M=8192 tactics=2/2/4
```

The latest five-repetition paired medians were 1,314.085 prompt tok/s at actual
M2079 and 1,050.050 at actual M8223, with all continuation/decode hashes equal.
The relative gate passed; the 2,000 prompt tok/s absolute gate did not. This
route affects prefill only and is not evidence for speculative decode above the
corrected 58.012 tok/s control.

The subsequent immutable-checkpoint projection gate passed exact BF16 parity
for attention QGKV, attention O, and SSM O at both M2079 and M8192. Isolated
parent/candidate medians were 3.5425/0.9029 and 13.0573/8.0888 ms for QGKV,
1.3994/0.4320 and 4.7317/1.4968 ms for attention O, and 1.3885/0.4384 and
4.7387/1.5408 ms for SSM O. The device-only dynamic activation chain needed by
runtime-generated SSM QKVZ also passed exact packed/scale/scalar parity at both
lengths. None of these rows is part of the production launcher yet; require a
same-binary zero-cache TTFT gate and identical continuation hashes after the
routes are wired.

`ATLAS_PREFILL_KV_DUAL=1` reuses that exact shared-input kernel in independent
output mode for target attention K/V in both the first cache-skip chunk and
later paged chunks. Keep it off for the control, then test it after the FFN
candidates while requiring the same output hash. Its candidate manifest must
require exactly one of each post-launch marker:

```text
ENGAGED ATLAS_PREFILL_KV_DUAL: attention_kv_cache_skip
ENGAGED ATLAS_PREFILL_KV_DUAL: attention_kv_paged
```

Those two entries replace the control manifest's `attention_k` and
`attention_v` projection-pipe markers for every dual-eligible chunk. Before a
live run, calculate the exact tokenizer/chunker row sizes; retain a legacy K/V
marker only if an actual final chunk has `M<=32` and therefore deliberately
uses the serial fallback. Do not copy the control marker set blindly: v5
rejects both missing declared markers and undeclared observed ENGAGED lines.

Before any server A/B, run the raw parent-versus-dual gate on the reserved
device. It checks every BF16 K and V byte, distinct unwritten-output sentinels,
4-KiB redzones, immutable A/weights/scales, M=33/63/64/65/127/128/129/2048/8192,
and cancellation-sensitive inputs. Optional timing is allowed only after all
parity cases pass:

```bash
ATLAS_TARGET_MODEL=qwen3.8-27b ATLAS_TARGET_QUANT=nvfp4 \
  cargo run --release -p spark-model --features cuda \
  --example w4a16_attention_kv_dual_microgate

ATLAS_PREFILL_KV_DUAL_MICROGATE_TIMING=1 \
ATLAS_TARGET_MODEL=qwen3.8-27b ATLAS_TARGET_QUANT=nvfp4 \
  cargo run --release -p spark-model --features cuda \
  --example w4a16_attention_kv_dual_microgate
```

`ATLAS_PREFILL_ATTN_GATE_FUSED=1` is a separate default-off dense-Qwen3.8
chunk-0 candidate. It replaces BR64 contiguous attention followed by
`sigmoid_gate_mul_batched` with one ABI-separate kernel. The attention value is
still rounded to BF16 and widened before the unchanged scalar
`1/(1+expf(-g))`, multiply, and final BF16 conversion. An eligible incomplete
bundle fails before Q/K/V projection or KV-cache mutation. The exact same-mode
manifest delta is:

```json
{
  "same_mode_environment_delta": {
    "ATLAS_PREFILL_ATTN_GATE_FUSED": {"control": "0", "candidate": "1"}
  }
}
```

Keep the control's complete route-marker map unchanged. Add exactly this one
entry to the candidate map; it does not replace another marker because the
parent attention and gate did not emit an ENGAGED line:

```text
ENGAGED ATLAS_PREFILL_ATTN_GATE_FUSED: cache-skip-br64
```

The ordered sweep emits that process-global marker on its first eligible 2K
chunk. Both 2K and 8K rows use the fusion. The 32K row is deliberately mixed:
only its first cache-skip chunk is fused and paged continuation chunks retain
the parent route. Do not report the complete 32K request as fused. Before a
server A/B, run the complete raw gate; optional timing requires that full
parity matrix and must win at both median and p90:

```bash
ATLAS_PREFILL_ATTN_GATE_MICROGATE_FULL=1 \
ATLAS_TARGET_MODEL=qwen3.8-27b ATLAS_TARGET_QUANT=nvfp4 \
  cargo run --release -p spark-model --features cuda \
  --example qwen38_prefill_attn_gate_microgate

ATLAS_PREFILL_ATTN_GATE_MICROGATE_FULL=1 \
ATLAS_PREFILL_ATTN_GATE_MICROGATE_TIMING=1 \
ATLAS_TARGET_MODEL=qwen3.8-27b ATLAS_TARGET_QUANT=nvfp4 \
  cargo run --release -p spark-model --features cuda \
  --example qwen38_prefill_attn_gate_microgate
```

The timing gate uses same-stream CUDA events, with every sample covering eight
consecutive launches normalized to per-layer time. It uses 21 alternating
pairs after three balanced warmups. Its
paired median saving must remain positive after a `3*MAD` noise bound and must
model at least 0.5 ms across 16 layers at 2K and 2.0 ms at 8K. Parity is run
twice with different output fills and opposite launch order so unwritten bytes
cannot accidentally compare equal. The raw binary rejects any embedded target
other than exact `(sm_121,qwen3.8-27b,nvfp4)` and any runtime device other than
a 48-SM GB10 at compute capability 12.1. Its verdict records target, device,
case count, and matrix checksum. The default two-case run ends in `SMOKE PASS
ONLY — NOT QUALIFIED`; only the explicit full run can emit a full-parity
verdict, and only full plus timing can emit `FULL PARITY+TIMING PASS`.

Use fresh directly redirected logs and nonces for `PREFILL_ATTN_GATE_FUSED=0`
control and `=1` candidate processes. For the coupled goal, run both as
`--comparison-kind same-mode --runtime-mode dflash-v3` with the same DRAFT,
binary, kernel bundle, model, tokenizer, endpoint, context, and every other
environment/CLI input. The raw gate or either strict 2K/8K TTFT non-win stops
promotion even if absolute prefill remains above 2,000 tok/s.

Selection occurs before Q, so a large aligned explicit request with a stale
kernel, conflicting fast layout, or incompatible K/V weights fails before any
Q/K/V projection. The promoted
`ATLAS_DFLASH_PREFILL_PIPE=1` route already uses the independent-output mode for
V3's bulk context K/V ingestion; this shortens speculative setup after TTFT and
does not accelerate the target's first token.

The target also has two default-off native W4A4 experiments. Start with
`E2M1_GEMM_DOWN_ONLY=1`: it leaves gate/up on W4A16 and quantizes only the down
activation. Test `E2M1_GEMM=1` last; it changes all three FFN projections, but
now prepares the shared gate/up activation once instead of twice. Both alter
activation precision, so neither belongs to the promoted V3 profile until its
output-quality and TTFT gates pass. If both are set, the full route wins.
Add `E2M1_STATIC_SCALE=1` only as a paired candidate: it uses the ModelOpt
checkpoint's calibrated input scales and removes the remaining per-projection
runtime absmax/host synchronization. The launcher refuses it without one of
the W4A4 routes. This is the scaling contract used by the independently
reproduced SGLang GB10 recipe, but its 2.17K prompt tok/s result measures the
whole SGLang stack—not this flag's isolated benefit.
Pair `E2M1_KMAJOR=1` with either W4A4 arm to address Atlas's measured wide-N
weight-layout bottleneck. It consumes the transforms already retained for
decode and stages one coalesced K-major weight tile for 128 prompt rows. This
keeps the current W4A4 numerical contract while replacing the row-major M64
implementation; it remains off until the canonical hash and TTFT sweep prove
both equivalence and a real production-shape gain.
The M128 kernel double-buffers both its staging and compute-form B tiles. It
can therefore transpose the next tile while all eight warps finish the current
tile, retaining only the final full-CTA visibility/lifetime barrier per K64
step. This is a compile-only property until the raw three-way parity gate and
production-shape timings pass.
For long-prefill qualification, add `E2M1_KMAJOR_M256=1` only after that M128
row passes. Its sixteen-warps share the same staged tile across 256 rows,
halving executed transformed-weight loads at 2K/8K/32K while leaving A/output
traffic and per-warp compute unchanged. It dispatches only from M=2048 and
falls back to M128 for shorter/tail calls. Modeled total kernel traffic falls
about 24%, not 2x; the flag remains a compile-only candidate until bitwise
M128 parity and per-projection GPU timing pass.
The M256 compute-form B tile is likewise double-buffered: this keeps the current
tile live while the next tile is transposed and reduces the source-level
full-CTA barrier sequence from two to one per K64 step. The SM121 host compile
must remain at or below 128 registers with zero spills after this trade.
After the static-scale row passes, add `E2M1_SILU_QUANT=1` to fuse only the
SwiGLU-to-down-quantization boundary. It preserves the standalone SiLU BF16
round trip in registers, then writes the exact packed/scaled input consumed by
the selected W4A4 down kernel. This removes one launch and the temporary BF16
activation's global write/read; it remains default-off until the same quality
and TTFT sweep passes. The launcher refuses it without `E2M1_STATIC_SCALE=1`.

For the v3/full-vocabulary/NVFP4-KV profile, use the dedicated legacy-named
wrapper instead of changing historical defaults:

```bash
MODEL_DIR=/path/to/the/exact/optimized-qwen-target \
DRAFT=/path/to/Apathy-Qwen3.8-27B-DFlash-drafter-v3 \
./bench/qwen38-gb10/serve-v3-72tps.sh
```

The following measurements are quarantined ABI-defect evidence, not expected
performance. Five measured rates were 70.6411, 72.1877, 72.2574, 72.0858, and 72.1689
tok/s (median 72.1689). A subsequent unchanged-server ten-run gate measured
72.3210 median with 0.987% coefficient of variation; all fifteen responses
shared stable output SHA-256
`f51d8358ea2a5c63353ca00a29208ae2cccd3039b070043cad514cc4af9761c4`.
A 2026-08-25 requalification after the upstream TUI port measured 72.3670,
72.4972, 72.2518, 72.2105, and 72.0784 tok/s (median 72.2518), retaining that
same output hash across all five repetitions. All of those binaries skipped
chunk-0 attention through the 13-host/11-PTX mismatch; do not reproduce or
promote them. NVFP4 KV changes output versus BF16 and remains separately
quality-qualified. See the historical record in
[`docs/QWEN38_WESCHERA_72TPS.md`](../../docs/QWEN38_WESCHERA_72TPS.md).

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
tokens are, so decode rate is a property of the *workload*, not of the engine
alone. Measured on the published container, temp 0, 400 tokens,
`reasoning_effort: none`, median of 3:

| Workload | tok/s (server) |
|---|---:|
| MinHeap class + complexity (the probe) | **63.9** |
| Arithmetic word problem with algebra | 55.1 |
| SQL query + explanation | 37.4 |
| Rust IPv4 parser | 37.0 |
| Multi-constraint logic puzzle | 36.1 |
| Security explanation (JWT `alg:none`) | 25.7 |
| Open prose, three paragraphs | **19.2** |

Median across these is ~37 tok/s and the spread is 3.3x. Boilerplate-heavy code
drafts extremely well; open prose barely drafts at all and runs near the
no-speculation floor.

This is why the probe is a fixed prompt: it is a *comparison* instrument, and
every figure in this repo and in the upstream PRs it is measured against uses
the same prompt. It is not a promise about your workload. If you are sizing for
prose or chat, plan against the low end of this table, not the headline.

### Which rate you are reading

Two figures differ by ~8% and it is worth knowing which is which:

- **`usage.response_token/s`** — the server's own decode counter. This is what
  the probe reports and what every published number here uses.
- **client wall-clock** (`completion_tokens / elapsed`) — includes request
  overhead and time-to-first-token, so it reads lower. On the MinHeap probe:
  63.9 server vs 58.6 client.

Compare like with like. A client-side number is not a regression against a
server-side one.

## Measurement notes

- **Do not A/B sequentially on a warm box.** Step time `S = tokens_per_step /
  overall_tok_s`; a sequential A/B produces phantom regressions from thermal and
  cache state. Interleave the arms.
- `ATLAS_DFLASH_DRAFT_SPLITK=8` is worth +1.13 tok/s (measured 64.07 vs 62.94).
  It reassociates the K loop, so bit-exactness is **not guaranteed** — though on
  this probe it produced a byte-identical completion either way. Treat it as
  unproven rather than unsafe, and drop it when validating numerics —
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
