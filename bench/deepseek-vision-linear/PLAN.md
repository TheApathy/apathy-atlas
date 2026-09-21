# DeepSeek full-encoder linear diagnostic: RED-only proposal

Root owns execution. The three new standalone tests exercise absent production
helpers; no helper, encoder, probe, CUDA, dependency or runtime change is made.

## RED commands

Use a fresh root-owned output directory and retain each actual exit status.
Compile independently; the contract/completion files should fail because their
helper is absent. Compile and execute the wiring test separately: native control
already exists; explicit diagnostic routing/completion is expected to fail.

```sh
rustc --edition=2024 --test /home/flocka/atlas/apathy-deepseek/crates/spark-model/tests/deepseek_vision_linear_diagnostic_contract.rs -o ROOT_RESERVED_DIR/contract
rustc --edition=2024 --test /home/flocka/atlas/apathy-deepseek/crates/spark-model/tests/deepseek_vision_linear_diagnostic_completion.rs -o ROOT_RESERVED_DIR/completion
rustc --edition=2024 --test /home/flocka/atlas/apathy-deepseek/crates/spark-model/tests/deepseek_vision_linear_diagnostic_wiring.rs -o ROOT_RESERVED_DIR/wiring
ROOT_RESERVED_DIR/wiring
```

## Intended pure helpers, not implemented

`layers/deepseek_vision/linear_diagnostic_plan.rs`:

- `Dimensions` is populated from existing admitted `Geometry`; no new pixel,
  grid, padding, checkpoint loader or model capability validator. Tests pin the
  canonical fixture. `EncoderPlan::new(dimensions, patches, aligned_rows, mode)`
  derives seven projection families and their 131 ordered instances. Actual
  existing `run()` remains the computation/iteration owner.
- Strict required `BackendMode` spellings are `scalar/default/full`. The
  `scalar` CLI spelling reports **native-wmma**, accurately describing the
  shipping fused-bias kernel; it is not a scalar reduction claim. Default/full
  report `gemmex-default`/`gemmex-full`, math modes 0/16. Missing/typo/non-UTF8 fails.
- `bind(slot, ActualLinear, Buffers)` compares the actual callback dimensions,
  bias presence and packed `ldc=N` with the next expected call before I/O.
  `Span` and immutable bound calls retain checked byte extents, pointer ends,
  alignment and non-aliasing. Native mode needs no workspace/GemmEx call.
- GemmEx uses original resident BF16 W[N,K], arguments T,N and (N,M,K),
  lda=ldb=K, ldc=N, dtype14/compute68/algorithm99, alpha1. No transposes or
  resident weight copies. K588 and all real M tails remain legal.
- Biased calls first copy each exact BF16 bias row into C using ordered
  `GpuBackend::copy_d2d_async`, then beta1. Bias-free calls use beta0 and no
  broadcast. This preserves the single output cast boundary; there is no
  post-GEMM bias add. Row-copy overhead makes this diagnostic-only, not a speed
  candidate. Workspace is exactly 8,519,680 bytes, separate from encoder scratch.

`layers/deepseek_vision/linear_diagnostic_completion.rs`:

- `with_completion(io, body)` runs the real upload/run/observer body, always
  attempts its completion fence, retains both operation/fence errors, and returns
  a borrowed output only after completion. `CompletionIo` supplies the fence,
  poison query and quarantine through the existing owning GPU abstraction.
- A failed completion poisons the session; no future forward or teardown may
  optimistically reuse/free anything. `OwnerIo`/`release_owned` require a fresh
  successful fence before explicit cleanup. All four owners (weights, encoder
  arena, cuBLAS workspace, handle/library lifetime) survive failed completion.
- Tests perform real deferred host-buffer copies for lifecycle visibility;
  they do not replace the real encoder or claim numerical GPU parity.

## Later narrow integration, requiring separate GREEN authority

1. Add `linear_diagnostic_plan` and `linear_diagnostic_completion` module wiring
   plus an explicit borrowed `VisionLinearBackend` interface in model-local code.
   Its `linear(actual_call) -> Result<()>` receives existing input/weight/bias/
   output pointers and checked dimensions. One backend is borrowed exclusively
   for an entire image; its cursor advances only after successful submission,
   and a successful completion receipt requires all 131 calls in order.
2. Add `forward_observed_with_linear_backend` and thread an optional borrowed
   backend through the SAME `forward_inner/run/linear` path. The seven actual
   linear call sites pass it; original `forward` and `forward_observed` supply
   None. Candidate handling returns without native fallthrough. No env selector,
   server flag, image cache, speculation guard or default numerical change.
3. Wrap upload as well as run/observer in completion handling. Preserve existing
   forward lock; retain a poison bit under that ownership. No successful output
   may hide simultaneous operation and fence errors.
4. Extend only the existing isolated `deepseek_vision_probe` example with a
   strict explicit diagnostic CLI and example-local GemmEx owner/FFI helpers.
   The existing frozen operator bench/admission stays untouched. Reuse existing
   GPU/KernelLaunch abstractions; no cuBLASLt global workspace or autotune.
   Retain one handle and the admitted workspace, bind gpu.default_stream(), then
   set workspace AFTER stream, host pointer mode and verified math mode. Use
   CUDA-feature-gated cuBLAS C ABI without adding a dependency. Record the
   effective library identities/version and stream/workspace ownership.
5. Probe teardown must not keep its current free-weights-after-release-error
   pattern. Destroy/free only after successful completion; otherwise quarantine
   all owners and terminate the isolated diagnostic as failed. Never copy the
   earlier operator adapter's destroy-after-failed-sync behavior.
6. Run the actual selective 267-tensor encoder (932,786,176 weight bytes), same
   original scratch (206,275,584 bytes), same pinned dyadic inputs and all three
   grids 3x3/4x5/54x54. Two reset-independent repetitions and existing stage taps
   must be byte stable; native/default/full are separately labeled. Sidecar
   backend receipts preserve the strict existing comparator manifest schema.
   Compare unchanged final thresholds .999 cosine/.995 worst row/.05 relative
   L2 and inspect first divergent stages. Do not infer FC1 is the sole cause or
   claim full-model/vision semantics/performance from a passing operator.

Pinned isolated evidence: /var/tmp/atlas-deepseek-gemmex-p5.F4ODgxdu/run-v2/result.json
SHA256 1c121864855074568e345517b6660af98fc51d1cfabd8f6047f8201835114dff.
This is a test/plan scaffold only; no actual RED or GREEN execution by this agent.
