# P10: actual two-runtime GLM DFlash2 KV-prefix raw comparison

Status: parent actual missing-helper RED and wiring4FAIL completed, followed by
installed-candidate seam RED1FAIL. Source GREEN hooks and executable staged;
parent compile/format/GPU evidence pending. No child test,
compiler, formatter, GPU, model, server, HTTP, or foreign code execution.

## Fixed comparison contract

One real EXL3 target with capacity2048, an installed candidate plus one external
reference (two separately loaded/zero-initialized trained DFlash2 runtimes),
one exact target token/capture history. The reference
uses the ORIGINAL full-recompute branch in `dflash2_proposal.rs`; the candidate
uses the actual P9 committed-prefix adapter. Never route both through the new
adapter or restore reference caches over candidate caches. No global env toggle.

Explicit diagnostic modes: `FullRecompute` and `CachedPrefix`. Neither changes
the HTTP production selector. Normal and graph calls pass `observer=None`.
The diagnostic API rejects graph capture before upload or callback effects.

Observe each layer immediately after attention, before its output buffer is
reused: full K pool, full V pool, attention output. Then selected hidden before
the EXL3 head, head logits before top-k, and chosen draft IDs. Five layers give
18 ordered stages:44,824,092 bytes per completed proposal. Byte extents come
from existing admitted runtime geometry, with a64MiB total capture cap.

Full pools use the actual NHD identity block-table layout (`reshape_and_cache.cu`):
rows `[0,context)` committed, `[context,context+8)` provisional noise, remaining
pool bytes unused/stale. Keep full raw files and separately report these three
regions; do not count provisional tokens as committed cache authority.

## New files and minimal seams after actual RED

- `dflash2_probe_contract.rs`: mode/stage/derived extents, ordered layout, checked
  NHD region interpretation. No second checkpoint architecture validator.
- `dflash2_probe_capture.rs`: actual observer's owned readback over existing
  `OwnedReadback`. Validate stage/order/source/extent/stream before copy; set
  terminal failure before foreign I/O; copy or fence error/panic never publishes
  a frame. Explicit drain uses original stream, including0. Keep prior frames.
- `dflash2_probe_runtime.rs`: explicit public diagnostic entrypoint and borrowed
  observer. Share current proposal arithmetic and owned anchor/path/status.
  Reference resets its tracker before proposal, enqueues with `prefix=None`,
  and drains/resets afterwards without publishing cached cursors. Candidate
  uses normal begin, ordered receipts, final selector validation and finish.
- `dflash2_proposal.rs`: optional observer only; emit real device spans at the
  named reuse boundaries. Default `None` performs no diagnostic transfers.
- `glm53/mod.rs`, `dflash2_runtime.rs`: explicit type/module registrations and
  new optional argument; no kernel, arithmetic, policy or server changes.
- `examples/glm53_dflash2_kv_parity.rs` plus small `session.rs`/artifact helpers:
  actual loaders, two drafters, real capture ingestion, raw output/receipt
  persistence, exact comparisons, exit nonzero on any mismatch or failure.
- `target_dflash2_probe.rs`: installed candidate state/layout inspection and
  actual cached diagnostic proposal. Holds its lock only during proposal, never
  across target verify/reset. Actual production commit/replay observers remain
  authoritative. The policy verifier's installed-drafter guard is unchanged.

## Ownership and stop conditions

Keep target, both runtimes, and both observation owners inside a persistent
session envelope before proposal work. Catch unwinds while that envelope is
still alive. Same-stream fences must succeed before host/source reuse or any
runtime/model/backend teardown; failed/panicking drain quarantines the entire
envelope under the existing consuming shutdown contract. A borrowed observer
callback alone is not an ownership guarantee. The retained `OwnedReadback`
buffer, source devices and backend lease must all survive uncertain completion.

Loader-before-return ownership is a separate existing boundary: inspect exact
constructor semantics before claiming that the envelope covers loader internals.
Do not silently broaden a runtime-only guarantee to those legacy constructors.

Raw frames validate finite BF16 and in-vocabulary U32 IDs. First mismatches are
reported with stage/byte or element position and complete hashes. Any mismatch
prevents exact qualification; finite closeness is not a substitute. Artifact
receipt failure must not drop pending host transfers. These runs have no speed
credit. Subsequent production timing uses an uninstrumented frozen binary.

## Executable scenarios

Same-capture context sweep:1,1,2,4,7,11,15,16,17,22,28,35,43,60,2039,2047.
This includes repeated context, deltas1..8, catch-up17, physical16-slot edges,
and final2047 capacity. Fill gaps with actual scalar target forward/capture
observations, never fabricated projected_target tensors. A requested reset
must reset the real target and both drafter contexts before a fresh repeat.

This executable also exercises actual target verifier full acceptance and first
rejection. An explicitly labeled deterministic diagnostic policy may force
these state paths (not a claim of natural raw-model acceptance). Full acceptance
leaves all8 wide capture rows available; first rejection replays only the anchor,
leaving its row0 available. Candidate ingestion occurs inside the real installed
observer; the independent reference consumes8/full or1/first-reject afterwards.
Partial acceptance>0 overwrites row0 during serial replay, so it cannot be
claimed covered without an additional target commit-observer seam. Root decides
whether that seam is required now; do not pretend final taps contain every row.

## Parent RED command

```sh
ATLAS_SKIP_BUILD=1 CUDARC_CUDA_VERSION=13000 cargo test --offline -p spark-model \
  --test glm53_dflash2_probe_contract --test glm53_dflash2_probe_capture \
  --test glm53_dflash2_probe_wiring
```

Those RED failures were observed by parent before GREEN. Current tree has new
helpers and real hooks/example; the installed-candidate adjustment adds a fifth
wiring gate. Test behavior is unchanged; only module-scoped dead-code attributes
allow standalone includes to exercise a subset of the public helper API.

## Executable CLI and immutable admission

```sh
cargo build --offline --release -p spark-model --features gpu-examples \
  --example glm53_dflash2_kv_parity
glm53_dflash2_kv_parity TARGET DRAFT TOKENS_JSON FRESH_OUTPUT PROVENANCE_JSON [CONTEXTS_CSV]
```

Provenance JSON requires exactly source_sha256, binary_sha256,
target_config_sha256, draft_config_sha256. The latter three are checked against
actual files; source is the root-supplied frozen-source manifest identity. Tokens
are an explicit U32 JSON array covering the selected contexts (default full16
context sweep above). Optional contexts are validated/nondecreasing and reported;
a short subset is not full-window qualification. Both frame sets and comparison
are written before stopping at the first raw mismatch. Case receipts include the
actual complete committed token history, anchor, raw hashes and separate
committed/noise/unused comparisons. Forced policy/reset cases run only after the
selected sweep is exact. No natural partial acceptance or speed qualification.
