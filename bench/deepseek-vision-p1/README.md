# DeepSeek Vision P1: bounded same-input numerical gates

Standalone diagnostic only. This does not load Atlas, run a model, modify encoder
code, qualify the full encoder, or enable speculation. Root owns every compile,
test and GPU execution window. `inspect` and the synthetic CPU tests do not load
CUDA or cuBLAS. No Python/ATen runtime or new reference generation is involved.

## Purpose and retained evidence

The retained V7 4x5 and 54x54 encoder outputs fail the unchanged full-encoder gate
(cosine >= .999, worst-row cosine >= .995, relative L2 <= .05). These are focused
boundary diagnostics, not a claim that either candidate closes that gate.

- RoPE: all three same-QKV host-angle controls must reproduce their retained
  native Q/K/V hashes before the candidate angle kernel is loaded. The candidate
  uses CUDA FP32 pow/divide/multiply/cos/sin and the *same old native RoPE PTX*.
  Candidate Q/K/V must match the retained official hashes exactly. Save both angle
  tables and all Q/K/V bytes, even when the candidate hash does not match.
- fc2: original BF16 [20,2816] SwiGLU and selected original-layout BF16
  `vision.blocks.0.mlp.w2.weight` [1024,2816] must replay the retained native WMMA
  fc2 hash. Only then run one cuBLASLt first-heuristic [20,1024,2816] candidate.
  Require its hash to match the retained official fc2 hash exactly. No autotune,
  weight transpose/copy, or relaxed tolerance. cuBLASLt COMPUTE_32F is recorded;
  it is not relabeled as the exact PyTorch BF16-reduction algorithm.

Official source revision: `6821d6ad3681a4b137b066b76094fa82ebd0a380` from
`deepseek-ai/DeepSeek-V4-Flash-Vision-Exp`. Actual quantized checkpoint revision:
`c171bea574201ff25530256fbd63626c7fd20f3c`. `src/pins.rs` pins the local official
source, checkpoint config/index, retained manifests, reports, and native PTX.
Reports used Torch 2.10.0+cu130, CUDA math SDPA with TF32 disabled; the default and
full BF16-reduction reports agree on all selected Q/K/V and fc2 reference hashes.
They are independent operators on shared native inputs, not chained execution.
The TF32 assignment is evidenced by the separately pinned retained script, not
a field in the stage reports. That retained script later gained the full-reduction
option; its byte identity with the historical default run is not established.

**Missing evidence:** official angle, Q/K and fc2 payload bytes were not retained.
There is no angle-reference hash either. This harness compares available official
output hashes; it cannot calculate elementwise errors against missing payloads
or call its candidate angle table an exact reference angle table.

## CPU build and admission (root only)

Initial GREEN compilation exited 101 before tests ran: the device-name scratch
used `i8` instead of platform `c_char` (unsigned on this AArch64 host), and a device
base pointer added a `usize` guard. The revised candidate uses exact C ABI typing,
checked u64 spans/usize totals, and a focused overflow/cap test. That failure is
retained evidence; it is not a claim that the pending test rerun has passed.

The first GPU RoPE control run stopped before candidate execution at
`/var/tmp/atlas-production-p1-ds-numeric.kZSPdWZW/rope/result.json` (exit 1).
The 54x54 value output retained nonfinite initialization sentinels despite intact
guards and successful cleanup. That artifact is preserved. The revised driver
fences default-stream memset and pageable copies with `cuCtxSynchronize` before
return, including attempted completion on operation errors and during cleanup.
An owned nonblocking kernel-stream drain alone is not a completion fence for
those operations. The revised source requires a fresh CPU ELF and admission;
this source change does not establish that the GPU control now passes.

```sh
CARGO_TARGET_DIR=/var/tmp/ROOT_CHOSEN_CPU_TARGET cargo test --offline --locked --manifest-path /home/flocka/atlas/apathy-deepseek/bench/deepseek-vision-p1/Cargo.toml
CARGO_TARGET_DIR=/var/tmp/ROOT_CHOSEN_CPU_TARGET cargo build --offline --locked --release --manifest-path /home/flocka/atlas/apathy-deepseek/bench/deepseek-vision-p1/Cargo.toml
/var/tmp/ROOT_CHOSEN_CPU_TARGET/release/deepseek-vision-p1 inspect --model /home/flocka/models/DeepSeek-V4-Flash-Vision-EXL3-K2-c171bea5 --corpus /var/tmp/atlas-deepseek-vision-parity-v7-20260905T0440Z --out /var/tmp/ROOT_CHOSEN_FRESH_ADMISSION
```

Admission hashes selected bounded payloads and current libraries, but does not
load them. Only the single 5,767,168-byte fc2 weight is read from a checkpoint
payload; its selected shard size/header are bound to the retained receipt. The
full shard and all 263 vision weights are not rehashed. The historical canonical
263-tensor hash remains a *retained reference receipt*, not new verification of
all current checkpoint bytes. Source files and the copied selected weight are
rehash-checked before every GPU run. Output directories must not already exist.

## Candidate build receipt (root only, reserved compiler window)

```sh
/usr/local/cuda/bin/nvcc -ptx -O3 -arch=sm_121f --fmad=false --ftz=false --prec-div=true --prec-sqrt=true /home/flocka/atlas/apathy-deepseek/bench/deepseek-vision-p1/angles.cu -o /var/tmp/ROOT_CHOSEN_FRESH_PTX/angles.ptx
```

Root creates and reviews a build receipt with these exact fields (substitute
measured hashes and actual compiler version, never infer them):

```json
{
  "schema": "atlas-dsv-p1-angle-build-v1",
  "source_sha256": "64-lowercase-hex",
  "ptx_sha256": "64-lowercase-hex",
  "compiler_sha256": "64-lowercase-hex",
  "compiler_version": "Cuda compilation tools, release 13.0 ...",
  "flags": ["-ptx", "-O3", "-arch=sm_121f", "--fmad=false", "--ftz=false", "--prec-div=true", "--prec-sqrt=true"]
}
```

Pass independently reviewed admission, PTX and build-receipt SHA256 values:

```sh
/var/tmp/ROOT_CHOSEN_CPU_TARGET/release/deepseek-vision-p1 run-fc2 --admission /var/tmp/ROOT_CHOSEN_FRESH_ADMISSION/admission.json --admission-sha REVIEWED_SHA --out /var/tmp/ROOT_CHOSEN_FRESH_FC2
/var/tmp/ROOT_CHOSEN_CPU_TARGET/release/deepseek-vision-p1 run-rope --admission /var/tmp/ROOT_CHOSEN_FRESH_ADMISSION/admission.json --admission-sha REVIEWED_SHA --candidate-ptx /var/tmp/ROOT_CHOSEN_FRESH_PTX/angles.ptx --candidate-ptx-sha REVIEWED_SHA --build-receipt /var/tmp/ROOT_CHOSEN_FRESH_PTX/build.json --build-receipt-sha REVIEWED_SHA --out /var/tmp/ROOT_CHOSEN_FRESH_ROPE
```

The CUDA path owns a fresh context and one nonblocking stream on ordinal 0,
requires SM12.1, and caps owned device allocations at 128 MiB including guards
and the 64 MiB cuBLASLt workspace. Copies follow an owned-stream drain; all
default-stream initialization and copies receive an explicit context completion
fence before returning. Cleanup also fences the context before releasing resources.
All allocations have 256-byte redzones; operands must remain byte-identical.
Outputs are prefilled with nonfinite sentinels. Handles, modules, workspace,
stream and context are explicitly released, with cleanup failures reported.

Exit 0 means all tested reference hashes exact and cleanup successful; exit 2
means a completed reference mismatch (not permission to relax a gate); exit 1
means admission, control, nonfinite, preservation, redzone or execution failure.
Read `result.json`, `native-controls.json`, `invocation.json`, and raw payloads.
No TTFT, prefill or decode performance is measured or claimed by this tool.
