# Weight cache — fast engine recovery

Persist the post-transform weight artifacts so a restart skips the host-side
work that dominates cold start.

If you only want the headline: **on Qwen3.8-27B NVFP4 the weight-load phase
drops from ~45–60 s to 17 s, of which ~4 s is applying 12.74 GiB / 352 cached
slots. Verified byte-for-byte with 704 slot comparisons, 0 failures. Every flag
is off by default; with no flags set the loader is byte-identical to the
pre-feature code.**

---

## 1. Why it exists

Cold-start weight loading on GB10 is not disk-bound. Raw NVFP4 weights are
mmap'd zero-copy and the page cache stays warm across a restart, so re-reading
them is nearly free. The time goes into rebuilding derived artifacts on the
host:

- `QuantizedWeight::transpose_for_gemm` — D2H the packed weight, transpose it
  byte-by-byte in a scalar CPU loop, H2D the result. Runs ~9× per layer.
- Runtime requantization (FP8 → BF16 → NVFP4) on checkpoints that need it.
- Predequant FP8 buffers for the prefill fast paths.

None of that depends on anything that changes between restarts. The cache
writes the *outputs* of the transpose family to disk once, keyed by a
fingerprint of the model and build, and on later starts mmaps them and pushes
them straight to the device.

### Measured (Qwen3.8-27B NVFP4, GB10)

| | |
|---|---|
| Cached artifact size | 12.74 GiB across 352 slots |
| Applying the cache | ~4 s |
| Weight-load phase, cached | 17 s |
| Weight-load phase, uncached | ~45–60 s |
| One-time build cost (first run) | +18 s over an uncached load |
| Verify run | 704 slot comparisons, **0 failures** |

352 slots = 64 attention (16 full-attention layers × Q/K/V/O) + 96 SSM
(48 linear-attention layers × qkvz/out_proj) + 192 FFN (64 layers × gate/up/down,
present because `ATLAS_FFN_M16_TRANSPOSED=1`). Two buffers per slot — packed
weight and scale — hence 704 comparisons under verify.

The operational point: a crash-restart or a config-flag iteration costs seconds
instead of minutes. This pairs with an external health-check supervisor that
relaunches the serve command — with a 17 s weight load the engine is back before
a caller times out, which is what makes automatic relaunch worth wiring up at
all.

---

## 2. Environment contract

**The feature as a whole is off by default. With `ATLAS_WEIGHT_CACHE` unset the
weight-load path is byte-identical to the pre-feature loader** — no cache is
read, none is written, nothing is deleted.

Once `ATLAS_WEIGHT_CACHE=1` is set, eviction is **on**: the cache keeps itself
inside a size budget rather than growing without bound.

| Env var | Default | Effect |
|---|---|---|
| `ATLAS_WEIGHT_CACHE` | off | `1` enables the cache. Nothing below has any effect without it. |
| `ATLAS_WEIGHT_CACHE_DIR` | `~/.cache/atlas-weight-cache` | Cache root. One subdirectory per fingerprint. |
| `ATLAS_WEIGHT_CACHE_VERIFY` | off | `1` runs both paths on every hit and byte-compares samples. Slower than an uncached load; a correctness gate, not a serving mode. |
| `ATLAS_WEIGHT_CACHE_EVICT` | **on** | Set to `0` (or `false`/`off`/`no`) to opt out, after which no directory is ever deleted and the cache accumulates without limit. Any other value leaves eviction on. |
| `ATLAS_WEIGHT_CACHE_MAX_GIB` | `32` | Total budget across the root. 32 GiB holds two Qwen3.8-27B variants with headroom. |
| `ATLAS_WEIGHT_CACHE_KEEP` | `0` | `0` = size-based only. `>0` additionally caps the directory count. |

The opt-out is matched leniently on purpose. Being generous about what counts as
"off" can only ever produce *fewer* deletions, so someone who writes `=false`
meaning "stop deleting my caches" gets that rather than a surprise reclaim.

`_EVICT`, `_MAX_GIB`, `_KEEP` and `_DIR` are deliberately **not** part of the
cache key — they change where caches live and which ones survive, never what a
cached buffer contains. Toggling any of them will not invalidate a valid cache.

---

## 3. Correctness model

### Fingerprint

The cache directory name is a 128-bit digest over everything that could change
a transform's output:

- `CACHE_FORMAT_VERSION` (bump to invalidate every cache globally)
- `CARGO_PKG_VERSION`, plus build-time `ATLAS_TARGET_MODEL` / `ATLAS_TARGET_QUANT`
  (a binary built for a different kernel target can transform differently)
- Detected weight format, NVFP4 variant, and runtime quant format
- An explicit list of `ModelConfig` fields — dimensions, head counts, layer
  types, TP rank/size, weight prefix
- 14 transform-affecting env vars, including `ATLAS_FFN_M16_TRANSPOSED`,
  `ATLAS_FFN_PREDEQUANT_FP8`, `ATLAS_FORCE_NVFP4_MOE`, `TQ_PLUS_WEIGHT_ROTATION`,
  `ATLAS_ATTN_SLIDING_WINDOW`
- The full sorted `(name, dtype, shape)` table of the weight store
- A **content sample**: leading 2 KiB of 24 evenly-spread tensors, read D2H

The content sample is what stops an abliterated re-quant from colliding with the
official checkpoint. They have identical tensor names, shapes and dtypes;
metadata alone is not a safe key. The digest is non-cryptographic — it is a
cache key, not a security boundary.

Adding a `ModelConfig` field or an env var that changes a transform means adding
it to the fingerprint **and** bumping `CACHE_FORMAT_VERSION`.

### Fallback discipline

Every failure path degrades to *recompute*, never to *serve something*. Wrong
part count, wrong part length, blob shorter than the index claims, mmap failure,
H2D failure — all log a warning and fall back to the real transform. Write-side
failures disable the writer and skip publishing.

The buffer lengths are derived independently from model geometry
(`transposed_lens`) and cross-checked against the index on every hit. If the
transposed layout ever drifts without a format-version bump, the mismatch shows
up as a slow load, not a corrupt weight.

### Torn-cache commit marker

`blob.bin` is appended during the build; `index.json` is written last, to a temp
file, then renamed. The index is the commit marker — a blob interrupted by a
crash, an OOM, or a failed load has no index and is ignored on the next start.
The index is only published after every layer has loaded successfully.

### Log lines to grep for

| Line | Meaning |
|---|---|
| `weight cache MISS: building at <dir>` | No usable cache for this fingerprint; this run will build one. |
| `weight cache WRITTEN: <N> slots, <X> GiB at <dir>` | Build committed. Without this line, nothing was published. |
| `weight cache HIT: <N> slots, <X> GiB at <dir>` | Serving from cache. |
| `weight cache: <N> hits, 0 misses (read-only, <dir>)` | End-of-load summary. Non-zero misses on a hit run means a slot key changed — investigate. |
| `weight cache VERIFY summary: <N> slots checked, 0 failures` | The correctness gate passed. |
| `weight cache VERIFY FAIL: <slot>[<i>] <window> window ... differs` | **Hard stop.** Names the exact slot and byte window. The run still serves the freshly recomputed buffer, so output stays correct. |
| `weight cache EVICT: <fp> (<X> GiB, last used <when>)` | A directory was deleted. |
| `weight cache: eviction ON — budget <X> GiB, currently <N> dirs / <Y> GiB` | Pruning is active; states the budget in force. |
| `weight cache: eviction OFF (ATLAS_WEIGHT_CACHE_EVICT opts out) — <N> dirs, <X> GiB retained` | Why nothing is being cleaned. |

---

## 4. Operations

### Disk footprint

12.74 GiB per variant for Qwen3.8-27B, measured, with
`ATLAS_FFN_M16_TRANSPOSED=1`. The 192 FFN slots are the bulk of that; without
the flag the footprint is roughly 3–4 GiB (estimated from layer geometry, not
measured). Each distinct fingerprint gets its own directory, so a checkpoint
swap, a `--tp-size` change, or flipping a transform-affecting env var each
produce a new one.

### Multi-variant behavior

By default a pass runs at startup once the active directory is committed, so
the root stays inside `ATLAS_WEIGHT_CACHE_MAX_GIB` on its own. At 12.74 GiB per
variant the 32 GiB default holds two with headroom; a third start on a new
fingerprint reclaims the least-recently-used one.

The pass:

- The **active** fingerprint is never evicted, at any budget.
- Directories with no valid index (torn builds, or caches from an older
  `CACHE_FORMAT_VERSION`) go first, and are reclaimed even when under budget —
  they can never produce a hit.
- Remaining directories are evicted least-recently-used, by a `last_used`
  timestamp file, until the root fits `ATLAS_WEIGHT_CACHE_MAX_GIB` and
  `ATLAS_WEIGHT_CACHE_KEEP`.
- **Concurrency guard**: a directory touched within the last 10 minutes is
  skipped, because another server may be running against it. An index-less
  directory gets a full hour instead — that is exactly what a build in
  progress looks like, and a cold build writes GiB before publishing.

Setting `ATLAS_WEIGHT_CACHE_EVICT=0` disables all of the above; directories then
accumulate indefinitely (two variants ~25 GiB, five ~64 GiB) and nothing cleans
up. `last_used` is still stamped on every hit under the opt-out, so recency data
stays accurate for whenever pruning is switched back on.

### Forcing invalidation

- One variant: `rm -rf ~/.cache/atlas-weight-cache/<fingerprint>`
- Everything: `rm -rf ~/.cache/atlas-weight-cache`
- Globally, in code: bump `CACHE_FORMAT_VERSION` in
  `crates/spark-model/src/weight_loader/transform_cache.rs`. Required whenever
  the on-disk layout, the fingerprint inputs, or a slot key's meaning changes.

### Verify workflow after an engine upgrade

The fingerprint includes `CARGO_PKG_VERSION` and the build-time kernel target,
so a rebuild that changes either invalidates the cache automatically. It does
**not** catch a change to transform *logic* within the same package version.
After touching anything in the transform path, run once with:

```
ATLAS_WEIGHT_CACHE=1 ATLAS_WEIGHT_CACHE_VERIFY=1 <serve command>
```

and confirm `weight cache VERIFY summary: <N> slots checked, 0 failures`. Keep
the transform-affecting env flags identical across cache-build and verify runs —
they are part of the key, and changing one silently gives you a cold build
instead of a verification.

---

## 5. Limitations and roadmap

**Transpose family only.** Today the cache covers
`QuantizedWeight::transpose_for_gemm` outputs. Still recomputed on every start:
`quantize_to_nvfp4` (a GPU kernel, comparatively cheap),
`predequant_for_prefill` for attention and FFN (`ATLAS_FFN_PREDEQUANT_FP8` alone
is ~17 GB of FP8 buffers), MoE `transpose_for_prefill`, and the FP8-native
attention transposes. Each of those mutates a layer object in place across
several buffers, so they need a different wrapper shape than the one-in-one-out
fit `transpose_for_gemm` had.

**No cross-process sharing.** The cache is per-process and disk-mediated; two
concurrent servers each pay their own H2D. Phase 2 is an IPC daemon that holds
the transformed buffers and hands them to a restarting engine directly,
following the SGLang fast-recovery design — that removes the H2D as well as the
transform.

**Graph capture now dominates.** With weight load at 17 s, CUDA graph capture is
the larger remaining term in time-to-first-token. Further work on weight loading
has less headroom than the capture path.

**No background reaper.** Eviction only runs inside a server start that has the
cache enabled. If `ATLAS_WEIGHT_CACHE` is turned off, whatever is on disk stays
there untouched — the budget is enforced at startup, not continuously.

---

## 6. Source map

| Path | Contents |
|---|---|
| `crates/spark-model/src/weight_loader/transform_cache.rs` | Fingerprint, index, blob writer, mmap reader, verify, process-wide handle |
| `crates/spark-model/src/weight_loader/transform_cache/evict.rs` | LRU policy, directory scan, `last_used` marker, grace-period guards |
| `crates/spark-model/src/weight_map/quantized.rs` | `transpose_for_gemm_cached` — the cached wrapper |
| `crates/spark-model/src/weight_loader/qwen35_dense.rs` | Qwen3.5/3.8 dense call sites (attention, SSM, FFN) |
| `crates/spark-model/src/weight_loader/qwen35/load_layers/attention_arms.rs` | Qwen3.5 MoE attention call sites |
