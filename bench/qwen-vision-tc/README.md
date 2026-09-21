# Qwen vision tensor-core microgate scaffold

Status: the current formatted, mechanically split helper snapshot passed all
15 CPU tests (contract 7, numerics 3, execution 5) and the formatting check.
Current logs: `/var/tmp/atlas-qwen-vision-tc-green.WgQqPiO3/formatted-split-green.log`.
The contract split preserves its API and the 250-line cap. No GPU runner,
CUDA kernel, production encoder edit, benchmark result, numerical promotion,
or runtime default change exists here.

Parent design: `qwen38/analysis/QWEN_VISION_TC_PLAN_20260906.md`.
Root alone executes tests/builds/GPU work after reserving the applicable lane.

## CPU commands (root only)

These standalone files need only the Rust standard library; they do not build
Atlas, initialize CUDA, import a donor, or read model weights. Run each as its
own command and retain the compiler's actual exit status. Initial REDs failed
because `src/contract.rs` was absent; the helpers now exist.

```sh
rustc --edition=2024 --test /home/flocka/atlas/src/bench/qwen-vision-tc/tests/contract.rs -o /var/tmp/ROOT_RESERVED_DIR/qwen-vision-contract
rustc --edition=2024 --test /home/flocka/atlas/src/bench/qwen-vision-tc/tests/numerics.rs -o /var/tmp/ROOT_RESERVED_DIR/qwen-vision-numerics
rustc --edition=2024 --test /home/flocka/atlas/src/bench/qwen-vision-tc/tests/execution.rs -o /var/tmp/ROOT_RESERVED_DIR/qwen-vision-execution
```

`ROOT_RESERVED_DIR` is a placeholder for a fresh root-owned directory, not a
literal directory to create. After compilation, root runs those exact test
binaries. Alternatively the standalone Cargo package is
dependency-free; `cargo test --offline --manifest-path .../Cargo.toml` still
needs a root-owned target directory and execution window.

## Implemented pure helper interfaces

- `contract`: strict explicit `Mode` parser (`scalar`, `upstream-separate`,
  `fused-bias`), seven `Family` shapes, checked `Plan`, non-null bounded
  `Region`, immutable `BoundPlan` and `Launch` receipts. Shape/byte formulas
  have one source here; no image parser, grid/pixel validation or model loader.
  Buffer/launch types reside in private `launch` (`contract_launch.rs`) and
  are re-exported through the unchanged contract API.
- A microkernel's `rows` is its actual M, including odd/tail M. Image requests
  are admitted by the existing production `ImageLayout`, not by a second
  validator in this bench. Full-image tests later go through that existing
  model entry point. Layer output widths 1152/4304 and merger 5120/2560 are pinned.
- Scalar ABI is `(A,B,bias,C,M,N,K)`, grid ceil(N/32),ceil(M/32), block32x32.
  Upstream ABI is `(A,B,C,M,N,K)`, grid ceil(N/128),ceil(M/128), block256,
  followed by `(C,bias,M,N)` with ceil(M*N/256) blocks. Proposed fused-bias
  candidate uses the scalar argument order and tensor-core geometry.
- `numerics`: finite BF16 RNE boundary functions and the explicit counterexample.
  This does not emulate tensor-core MMA order or define a cosine pass threshold.
- `execution`: `ProbeIo` isolates enqueue/fence/immutable-operands-and-guards/
  readback/poison from CPU orchestration. A failed enqueue is drained before
  return. A failed fence poisons the owner and retains allocations; no result
  is published or allocation freed. Successful drain permits normal cleanup.
  Execution tests write actual deferred host buffers, not merely log strings;
  their synthetic writes test ownership, not matrix arithmetic.

## Later numerical integration (not authorized in this scaffold)

1. Retain the passing pure-helper gates with every candidate change. Missing-handle,
   extent, alias and alignment rejection occurs before any I/O. Tensor-core
   A/B addresses and K-row strides require 16-byte alignment (K multiple 8);
   output/bias require BF16 alignment. Single-projection data stays below 96 MiB.
2. A new explicit benchmark adapter may use `GpuBackend` and `KernelLaunch`
   against a root-pinned standalone candidate PTX. Use one owned stream for
   initialization, uploads, both kernels, fence and readback. Retain buffers
   until completion; no production source change or full model allocation.
3. Pin actual resident-equivalent BF16 operands and weights from optimized-qwen
   config 267be2125ee2ec272555748c87cc636b25a96107946f05491bd6043151c7fe4e,
   or FlashNext config e765305daba0951974308f4d32c075b52a6a45974730d273f2216718a994d624.
   Pin every selected payload and source/PTX; never clone/dequantize full weights.
4. Compare shipping fused scalar, pinned upstream separate-bias and explicitly
   labeled fused-bias candidate on the SAME input. Include actual K4304/N4304
   tails, M1/4/36/127/128/129/1024, positive/zero/negative biases, raw outputs,
   finite checks, guard/operand preservation and Compute Sanitizer.
5. Full encoder and image semantics use existing `ImageLayout`/aggregate/splice
   ownership, original preprocessing, patch/all 27 blocks/final receipts, independent
   pinned reference and uncached image tests. Keep scalar default until root's
   numerical and quality gates pass. Image TTFT is not text-prefill/decode speed.

No launch, model, context, speculation, image-cache or quality guard is opened.
