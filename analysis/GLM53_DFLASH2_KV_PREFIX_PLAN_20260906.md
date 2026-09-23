# GLM DFlash2 committed-prefix KV reuse: RED implementation contract

Status: NEW source-only tests and plan. The production helper and runtime adapter
do not exist. No tests, compiler, CUDA, model, HTTP, or benchmark execution by the
author. Root must read and execute actual RED before GREEN authorization.

## Existing real work to remove

`dflash2_runtime.rs` projects the entire `projected_target` prefix through each
of five layers' K/V weights on every proposal (358–374). `attention_plan` passes
`past_retained_tokens=0`. Existing attention execute then normalizes, applies
absolute RoPE, and overwrites the entire committed cache. The runtime already
owns five K/V pairs; this proposal needs no new device allocation or weights.

The first proposal still processes the full prompt. Subsequent proposals process
only target-confirmed rows not yet represented by completed per-layer KV cursors.
This is a drafter candidate, not proof of target/scalar arithmetic parity or a
performance result. Existing policy-before-commit and rollback remain unchanged.

## Proposed new pure production helper

`src/model/glm53/dflash2_kv_prefix.rs`:

```text
parse_kv_prefix_flag(Option<&str>) -> Result<bool>  # absent/0 off; 1 on; else error
KvPrefix::new(layer_count: usize, max_context: u32) -> Result<KvPrefix>
begin(&mut self, context: u32, target_position: u32, stream: u64, capturing: bool)
enqueue_layer(&mut self, index: usize, io: &mut dyn KvPrefixIo) -> Result<()>
finish(&mut self, io: &mut dyn KvPrefixIo) -> Result<()>
abort(&mut self, io: &mut dyn KvPrefixIo) -> Result<()>
reset(&mut self, io: &mut dyn KvPrefixIo) -> Result<()>
completed_rows(layer) -> Result<u32>; pending() -> bool
pending_stream() -> Option<u64>; poisoned() -> bool
KvPrefixIo::{enqueue_layer(layer, KvTail, stream), synchronize(stream)} -> Result<()>
KvTail::{source_row, retained_rows, new_rows, committed_end} -> u32
```

`KvTail` is immutable, constructed only by the owner, and Copy for borrowed kernel
binding/receipts. Actual architecture limits remain authoritative in the existing
runtime/config: pass loaded `weights.layers.len()` and `MAX_CONTEXT_TOKENS=2047`.
The helper validates nonzero dimensions, checked offsets/extents, identity,
monotonicity and its supplied capacity; it does not create another model validator.

Begin validates all inputs before creating an in-flight update. Cursor zero means
bootstrap: `[0, context)`. Normal update uses `[completed, context)` and retains
the earlier rows. Repeated same context refreshes only the last committed row:
`retained=source=context-1,new=1`. Existing AttentionPlan rejects zero target rows;
this explicit refresh avoids changing that kernel contract or re-normalizing an
already-normalized cached K. A delayed proposal may legitimately catch up more
than eight committed rows after ordinary/adaptive decode; retain the same capacity
bound instead of inventing an eight-row cache-advance limit.

All layer receipts are ordered and recorded only after actual layer enqueue
succeeds. Completed cursors do NOT move on enqueue. Finish requires every layer
receipt and a successful same-stream fence, then atomically advances all cursors.
Record in-flight status before foreign I/O, including callbacks that panic after
submission. Failed enqueue, caught panic, failed completion or explicit abort
blocks reuse. Abort drains but leaves the cache poisoned; reset drains any pending
work on its original stream (including stream zero), then invalidates every
cursor. Failed reset retains pending ownership and does not clear poison/cursors.
The helper owns no GPU allocation; runtime ownership must survive failed fencing.

## Real runtime integration, after RED and review

1. Register the helper and NEW `dflash2_kv_prefix_runtime.rs` child from the
   current runtime. Keep a runtime-owned tracker (e.g. Mutex<KvPrefix>) across
   proposals. Parse `ATLAS_GLM53_DFLASH2_KV_PREFIX` strictly; reject non-UTF8.
   Default-off `propose` preserves the existing body. Opt-in dispatches to
   `propose_with_kv_prefix` before anchor upload or any proposal effect.
2. Cached wrapper binds actual `context_tokens == target.position()`, same
   runtime/stream and current cursor. Begin before effects; an attempt guard
   poisons the target on error/abandonment. Retain the tracker on panic and recover
   its poisoned mutex only for explicit drain/reset, never optimistic reuse.
3. A real KvPrefixIo layer adapter projects `projected_target + source_row*8192`
   through that layer's original K/V weights for `new_rows`. Query/noise remain
   eight rows. Bind existing attention with `new_context_tokens=new_rows`,
   `past_retained_tokens=retained_rows`, `absolute_context_end=context_tokens`.
   Use original scratch K/V for the unnormalized tail and existing per-layer
   cache pair. Noise staging offset is `new_rows*1024*2`, NOT context*1024*2.
   The layer callback encompasses both projections and actual attention/cache
   enqueue; do not manufacture a layer receipt before normalization/RoPE/cache.
4. Current fixed 2047 context means no eviction/compaction is needed: kept past
   equals retained rows. Provisional noise occupies `[context,context+8)` but
   never increments the committed KV cursor. Next target-confirmed delta
   overwrites those slots with true committed representations. No speculative
   target input or rejected draft token grants cache validity.
5. Finish only after the complete five-layer proposal, head/selector and existing
   `read_proposal` validation succeed. Any error aborts/drains and poisons; no
   automatic ordinary fallback or prefix-publication-on-enqueue. `finish`'s fence
   is also the proof that prior queued capture FC/norm completed. Do not confuse
   existing logical `context_tokens` advancement with completed cache validity.
6. Explicit cached mode is eager-only. Reject `propose_graph_probe` BEFORE anchor
   upload/begin_capture; captured host-side construction cannot publish a GPU
   cache receipt. Default-off original graph probe remains unchanged. General
   graph replay caching requires a separate generation/launch completion contract.
7. `reset_context` drains tracker first, then invalidates logical context. `free`
   drains before any KV/arena/weight free; if completion fails, retain the entire
   runtime owner. Existing consuming `free(self)` and proposal stack H2D/D2H
   lifetime guarantees require root review: adding a cursor cannot make those
   pre-existing resource boundaries safe. Do not hide failed drain by only
   clearing the cursor or dropping raw device-pointer metadata.

## RED commands for root

Use the root-reserved existing offline CPU target; no foreign model/tokenizer.

```sh
ATLAS_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 cargo test --offline -p spark-model --test glm53_dflash2_kv_prefix
ATLAS_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 cargo test --offline -p spark-model --test glm53_dflash2_kv_prefix_wiring
```

The first should compile-fail on the missing production helper. The second
should fail six actual-runtime wiring gates. Deferred tagged storage tests prove
cursor/ordering/poison behavior only, NOT floating-point or attention correctness.

Before performance credit, root must compare frozen full-recompute versus cached
same-input raw K/V after norm/RoPE, attention output, selected hidden and draft
IDs across first/repeat context, 1–8 advances, 16-slot boundaries, partial accept,
reset and 2047 end. Changing GEMM M can change rounding: do not call this exact
without evidence. Then rerun actual policy/rollback tests and matched text/image/
bias/repetition/grammar/long-copy requests with cache0 and output hashes, retaining
the existing case10 and P6c failures. Only isolated repeated measurements can
establish a speed win; source FLOP reduction is not measured throughput.
