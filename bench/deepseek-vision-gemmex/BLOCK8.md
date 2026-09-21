# Block-8 FC1 same-input diagnostic

This separate executable tests the actual P14 `grid-4x5` block-8 norm2 input
with the original native BF16 `vision.blocks.8.mlp.w1.weight`. It does not run
a model or encoder, and no authoritative block-8 FC1 teacher payload or hash
is available. Candidate differences are reported against the captured native
control, never promoted to teacher, full-encoder or performance parity.

Input: BF16 row-major `[20,1024]`, 40,960 bytes. Weight: BF16 row-major
`[5632,1024]`, 11,534,336 bytes. Output: BF16 `[20,5632]`, 225,280 bytes.
No bias, activation or RoPE is included. P14 manifest and stage hashes,
selected raw input/control hashes, config, index, shard header and exact
selected weight range are admitted before CUDA. `inspect` reads only this
weight payload, not the full 7.6-GB shard, and stores its fresh SHA receipt.

## Root-controlled qualification sequence

CPU tests/build and GPU execution require separate coordination claims.
The following commands are recipes, not evidence that they have run.
Use the explicit binary name; the historical block-0 executable is unchanged.

```sh
cargo test --offline --manifest-path bench/deepseek-vision-gemmex/Cargo.toml \
  --test block8_contract --test block8_lt --test block8_report --test block8_wiring
cargo build --offline --release --manifest-path bench/deepseek-vision-gemmex/Cargo.toml \
  --bin deepseek_vision_block8
```

The build is Rust-only and loads existing pinned PTX at run time; no new CUDA
compunit or production changes are introduced. Set a separately reserved
`CARGO_TARGET_DIR` if needed. The executable is under that target's `release`.
First inspect into an explicitly NEW directory whose parent already exists:

```sh
/ABS/TARGET/release/deepseek_vision_block8 inspect \
  --model /home/flocka/models/DeepSeek-V4-Flash-Vision-EXL3-K2-c171bea5 \
  --corpus /var/tmp/atlas-glm-p14-run.iH0Tjxxn/deepseek-block8 \
  --out /ABS/NEW-ADMISSION
```

Review `admission.json` and its printed SHA, all selected source/library
receipts and weight selection before claiming GPU. Run into another NEW
directory; the caller must pin ELF/PID/start time/effective environment and
exclude concurrent GPU/compiler/model/benchmark activity.

```sh
/ABS/TARGET/release/deepseek_vision_block8 run \
  --admission /ABS/NEW-ADMISSION/admission.json \
  --admission-sha REVIEWED_64_HEX_SHA \
  --out /ABS/NEW-RUN
```

`run` reconstructs admission and verifies the saved weight bytes against a new
bounded selected read before CUDA initialization. No cached `passed` field is
accepted as proof. There is no network, Python, model construction or inference.
The read-only input receipt helpers assume trusted immutable local files;
they are not a comprehensive hostile-filesystem isolation boundary. Requested
CUDA/cuBLAS/Lt libraries are hashed, not every transitive dependency.

## Arithmetic and ordering

1. Replay production `deepseek_vision_linear` twice with reset outputs on an
   owned nondefault stream. Both raw outputs must equal the pinned P14 native
   control hash and bytes exactly before any candidate GEMM executes.
2. GemmEx default math `0`, then full math `16`, two reset repeats each. Reuse
   the tested T/N, BF16 A/B/C, FP32 compute `68`, algorithm `99` ABI. The
   8,519,680-byte workspace is set after the stream; math is restored to `0`
   even after an attempted-call failure.
3. Lt baseline (reduction preference left unset), then compute-type-only
   (preference attribute `3`, mask `2`), two reset repeats each. The baseline
   means the pinned CUDA13 default preference, not the historical Torch path.
   Request one heuristic, with no autotuning or algorithm mutation. Record
   actual algorithm bytes/hash, ID, tile, split-K, reduction and workspace.
   Compute-only rejects output-type reduction before matmul. Allowed reported
   schemes are no reduction (`0`) or compute-type reduction (`2`); all data
   operands/output remain BF16.

There are nine guarded device allocations: 88,334,848 bytes total against an
explicit 96-MiB cap. The Lt workspace is fixed at 64 MiB. The cap excludes
context/library-internal allocations. Host payloads, per-repeat outputs and
workspace-reset buffers are bounded by these fixed shapes. Inputs, weights,
native control and 256-byte allocation redzones must remain unchanged.

Every available raw output is saved before finite, repeated-output, native
or numeric validation, including after attempted API failures. Each attempt
records operation/readback/save errors. A failed read produces no invented
payload hash. Files use create-new publication. Partial runs preserve output
and attempt files; `result.json` reports errors after outer guard/cleanup
attempts. Missing completion artifacts never imply success.

The inherited Driver and diagnostic Lt owners attempt completion before
teardown and aggregate errors, but still free/destroy after a failed fence.
This is **not quarantine** or a production-safe recovery claim. Any API,
drain, cleanup, native-control, finite, reset or redzone error exits `1` and
ends the probe; do not continue using that process. Initialization failures
have only the inherited best-effort destructor, not a complete cleanup receipt.

Exit `0` means only `DIAGNOSTIC_COMPLETE`: both native replays exact and all
four candidates completed finite, repeat-stable comparisons. It does not mean
candidate equality. Reports contain FP64 global/row cosine, native-denominator
relative L2, max absolute error and F32 exact fraction at the true `[20,5632]`
shape. No adjustable tolerance or teacher substitute is introduced. The real
three-grid encoder gates remain unchanged: `.999 / .995 / .05`.
