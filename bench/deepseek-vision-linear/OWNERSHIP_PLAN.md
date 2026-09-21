# DeepSeek encoder diagnostic ownership: NEW RED boundary

Authority: physical `TEAM_INBOX.md` entry 2026-09-06 20:20:15 UTC,
DS-OWNED-BOUNDARY-RED+PREFILL-AUDIT. Only new tests and this plan are staged.
No helper, encoder, example, model, backend, loader or numerical change.
Root executes RED. Prior pure plan/completion helpers are frozen: root reports
5+4 PASS and format check PASS; that does not establish real resource retention.

## Actual lifetime findings

| Boundary | Current source | Concrete issue |
| --- | --- | --- |
| Host upload | `spark-runtime/src/cuda_backend/gpu_impl.rs:120` | `cuMemcpyHtoDAsync` precedes a fallible stream fence. An Err can still leave the host source referenced. |
| Host readback | same backend `copy_d2h` | Destination Vec has the same failed-fence lifetime obligation. |
| Encoder | `layers/deepseek_vision/forward.rs:40` | Local BF16 pixels drop on upload Err; outer fence is skipped. Run+fence failure hides the fence error. |
| Arena | encoder `mod.rs:156` | Consuming `release(self)` loses the object on fence/free Err. Raw arena has no automatic owning Drop. |
| Isolated probe | `examples/deepseek_vision_probe/gpu.rs:95` | Selected weights are freed even after encoder release fails. Output/stage readbacks are local Vecs. |
| Request images | `model/deepseek_vision.rs:87,118,219` | Pending buffers clear before a proven drain; error cleanup ignores fence outcome; `drain(..)` loses remaining entries on first failed free; teardown takes state before success. |
| Full model Drop | `model/drop.rs:30`, `impl_a2.rs:54` | Pinned host staging is freed after the ignored vision teardown result. |

`WeightStore` and `WeightTensor` are pointer metadata, not device allocation
owners (`spark-runtime/src/weights.rs`, including `filtered_view` documentation).
The model's DenseWeight/EXL3 references and BufferArena likewise contain raw
pointers; no relevant automatic freeing Drop was found. Cloning a store does
not create allocation ownership. The probe needs a checked unique allocation
ledger, retaining original pointers and extents, with no weight payload copy.

`AtlasCudaBackend` has no Drop; its context/stream/module registry is held by a
process-global OnceLock (`atlas-core/src/registry.rs`). Dropping its wrapper
neither synchronizes nor destroys the CUDA context. A future cuBLAS handle and
library lifetime must be explicitly retained with the session. No destructor
may destroy/unload that owner after uncertain completion.

## Smallest owning helper and concrete encoder integration

NEW `layers/deepseek_vision/owned_session.rs` will own `Option<R>` by value.
`OwnedSession::new`, `run(body, fence)`, nonconsuming
`try_release(fence, release)`, `is_poisoned`, and `is_released` are the proposed
API. A short borrowed adapter should reuse the existing CompletionIo/OwnerIo
policy; callbacks let the encoder borrow its existing GpuBackend per call,
without inventing a cloneable GPU owner or extending its lifetime unsafely.
The successful release path removes R only after fence and all releases succeed.
Failed fence/free retains R and disallows automatic reuse/recovery. Drop of an
unreleased capsule deliberately quarantines R without running resource Drop.
This is bounded terminal retention, not successful cleanup or a recovery API.
No unsafe access to the retained bytes is needed. Result handling does not claim
panic recovery; an unreleased capsule must also never free on ordinary unwind.

NEW `encoder_ownership.rs` will be the actual resource state under the existing
forward mutex, replacing Mutex<()> with Mutex<EncoderOwnership>. Its resource
record holds the existing arena and `pending_upload: Option<Vec<u8>>`. Move the
existing BF16 Vec into that field before copy_h2d; do not clone its payload.
The completion closure clears it only after a successful real stream fence.
`with_owned_upload` wraps upload, unchanged run and observer. Nonconsuming
encoder `try_release(&mut self, gpu)` routes through the capsule; failed free
retains arena metadata. Geometry determines the upload cap (currently 4,064,256
bytes), and scratch remains 206,275,584 bytes. Numerical order stays unchanged.

Behavior RED calls the missing PRODUCTION owning capsule with real Vec payloads
and Drop counters. It verifies allocation identity, deferred upload/readback,
both errors, all five retained resource categories even after scope exit,
partial release, poison/released admission, and no implicit FFI/device cleanup.
This proves the capsule once GREEN, not its still-RED concrete CUDA wiring.

## Isolated probe owner before borrowed GemmEx

NEW example `owner.rs` supplies `ProbeResources`: backend/context lifetime,
selected-weight ledger/store, encoder, optional diagnostic linear backend
(handle/library/workspace), and pending readback. The entire post-load operation
and teardown stays inside one owning capsule, including early metadata/encoder
construction failures. Remove a ledger entry only after its free succeeds;
check duplicate/overlapping entries before accepting ownership. Release after
one proven shared-stream drain, with handle before workspace/arena/weights and
backend last. Partial release records retain failed and later entries.

`readback_owned` allocates/registers the destination before copy_d2h, never reads
or resizes it after an uncertain fence, and takes/publishes bytes only after a
successful fence. Both final output and stages must use it. A normal observer
callback error also drains; a D2H fence error must not drop local staging while
propagating the error. Keep at most one pending stage readback (current maximum
47,775,744 bytes), not a second corpus or full model. The native path stays the
default; later explicit diagnostic modes reuse this exact session.

## Full model and pre-session limits: do not overclaim

Minimal full-model integration requires a separate reviewed source extension:
nonconsuming `try_release_deepseek_vision` keeps its Option until success, drains
before clearing pending rows, removes pointers only after successful frees, and
returns the real outcome to model Drop. On failed drain, retain/quarantine vision
state and skip pinned-host free. Ordinary prepare/splice must reject poisoned
state. Record every stream that used pending rows, or validate default-stream
equality before effects; a default-only fence does not prove arbitrary-stream
completion. Local original/safe token-ID uploads in splice need the same retained
host ownership. Merely changing the teardown function does not fix those paths.
The wiring RED deliberately keeps this unsolved boundary visible.

The generic safetensors loader has an earlier, separate gap: its mmap/converted
host bytes can drop after copy_h2d's failed fence before any WeightStore returns
(`weights/loader/load_fns.rs:129,218`). A post-load probe owner cannot repair
that. No generic loader/backend/whole-model ownership fix is authorized here.
The diagnostic claim therefore starts only after successful selective loading,
and loader-failure safety remains unqualified. Backend allocation leaks are not
proof of host-buffer retention. Do not claim complete process panic recovery,
server lifecycle correctness, or numerical/performance qualification from RED.

## Root-only RED commands and next gate

Use a fresh root-reserved output directory; preserve each exit status. Behavior
should fail to compile because owned_session.rs is absent. The independent
wiring executable should compile and fail four real integration seams.

```sh
rustc --edition=2024 --test /home/flocka/atlas/apathy-deepseek/crates/spark-model/tests/deepseek_vision_owned_boundary.rs -o ROOT_RESERVED_DIR/owned-boundary
CARGO_MANIFEST_DIR=/home/flocka/atlas/apathy-deepseek/crates/spark-model rustc --edition=2024 --test /home/flocka/atlas/apathy-deepseek/crates/spark-model/tests/deepseek_vision_owned_wiring.rs -o ROOT_RESERVED_DIR/owned-wiring
ROOT_RESERVED_DIR/owned-wiring
```

After actual RED, request narrow pure owning-helper GREEN first. Actual encoder,
probe and full-model seams remain separately held until root source review and
fault-injected current-tree CPU coverage. Only then wire borrowed linear backend,
build one frozen example, and run native/default/full encoder qualification with
the unchanged final comparator thresholds and separately hashed backend receipt.
