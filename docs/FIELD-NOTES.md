# Field notes — how this engine operates (the time-cutting map)

The operational knowledge that makes work on this codebase fast. Every entry
earned its place by costing time to discover. Update it when you learn
something that would have cut an hour off.

## Where the fast iteration loops are

| task | oracle | runtime |
|---|---|---|
| MoE decode/verify kernels | `cargo run --release -p spark-model --example moe_unified_t_m_microtest -- 32 24 0xD54 6 -1 6` | ~40 s incl. build |
| w4a16 GEMV variants | `--example w4a16_gemv_grouped_microtest --features cuda,gpu-examples` | seconds |
| dense BF16 GEMM | `--example dense_gemm_microtest` | seconds |
| raw achieved bandwidth | `scripts/bw_ceiling.py` (torch) | ~1 min |
| decode end-to-end | `scripts/dsflash-serve-bench.sh <name> 5 <ENV=V ...>` + streaming probe | ~6 min load + run |
| quality gate | tool-eval-bench `--short` (bar: 90/100, 0 fail; TC-14 is numerics-borderline) | ~330 s |

**Never debug kernel bandwidth in serve first.** The microtests reproduce
production behavior (dedup microtest 137 vs production 135) and iterate in
seconds. Serve is for composite validation only.

## Kernel plumbing (10-second version)

- Kernels live in `kernels/gb10/{common,deepseek-v4-flash/nvfp4}/*.cu`.
  **Registration is automatic by file stem** — add an `extern "C" __global__`
  entry point to an existing `.cu` and it is immediately loadable as
  `gpu.kernel("<file-stem-or-KERNEL.toml-alias>", "<entry-name>")`.
- `KERNEL.toml [modules]` maps file stems to module aliases.
- Optional kernels: `try_kernel(...)` → `KernelHandle(0)` on miss; dispatch
  tests `.0 != 0` and falls back. This is the standard pattern for opt-in
  kernel rollouts.
- Wrappers live in `crates/spark-model/src/layers/ops/quant_dispatch.rs`
  (grid/block/args in one place, doc-comment the kernel signature).
- The kernel build banner says which model target the kernel set was built
  for — a microtest failing with `cuModuleGetFunction ... NOT_FOUND` usually
  means the kernel set on disk is for a different MODEL.toml target, not a
  code bug.

## The o-projection map (where the decode milliseconds live)

- Plain decode Step 6 (`decode/attention_forward_v4.rs`): wo_a is
  BLOCK-DIAGONAL (o_groups=8 × [o_lora=1024, group_in=4096]) → historically
  8 serial launches/layer; now one `w4a16_gemv_grouped` launch (bit-identical,
  153→228 GB/s). wo_b is a plain [4096, 8192] GEMV — single launch was always
  fine (225 GB/s).
- Verify Phase C (`trait_impl/multi_seq/mla.rs`): the batched `_ld` kernels
  reduce K in a DIFFERENT order than single-row → ulp drift → 43-layer
  amplification → the 2-3% capture drift that collapses drafter acceptance.
  `ATLAS_OPROJ_EXACT=1` was the slow per-row workaround; the default is now
  `w4a16_gemv_grouped_batchm` (bit-exact per row AND 3.07× the per-row
  speed). `ATLAS_OPROJ_BATCH_EXACT=0` restores the drifty `_ld` path.
- The drafter (`dspark_head.rs` ~line 816) has the SAME 8-launch disease on
  dense BF16 `dense_gemm` — unfixed; it is propose's `d_o_proj` 7.39 ms.

## Bandwidth truths (measured 2026-08-09, do not re-derive)

- Real kernel ceiling ~229 GB/s (streaming read); contiguous GEMV 225;
  64B random gather 129; per-expert-sized working sets ~168.
- Anything quoted against "183 GB/s achievable" predates the ceiling probe.
- Expert weights are maximally incompressible (entropy + SVD + MI all
  measured shut). Byte reductions = lossy requant only.

## Reference stack (0xSero SparkInfer), same-box measured

- Decode 5×512 code: min 32.5 / median 37.4 / mean 38.7 (their ≥35 gate
  FAILS here too). Prefill ~654 tok/s at 6K uncached (1,055 was 252K).
- **Prose ~19.5 tok/s — below our plain.** Repeat ~58.6. Their fixed-K5 has
  no adaptive fallback; they lose ~26% vs their own floor on prose. The gap
  to close is structured content only.
- Stack: `$SPARKINFER_REF` (compose; `docker compose start` to
  re-boot, ~15 min: weights cached in ./data). Patches are loading/format
  only — kernels are stock SparkInfer (`$SPARKINFER_SRC`).
- Their artifact: 3.0 bpw EXL3/Trellis experts + REAP-216. Implied plain
  floor ~26.2 on this box.

## Serve-side gotchas that cost sessions

- **`cargo build --release` DEFAULTS TO THE QWEN3 KERNEL TARGET.** Serve
  dies with "No compiled kernel target matches model_type 'deepseek_v4'".
  Build with `ATLAS_TARGET_MODEL=deepseek-v4-flash cargo build --release`.
  Microtests can mask this: `common/` kernels compile into every target set.

- `ATLAS_DSPARK_ADAPTIVE` was a phantom flag (nothing reads it); the real
  one is `ATLAS_DFLASH_ADAPTIVE`. Fixed in the scripts 2b55e285 — but check
  any old launcher you copy.
- `ATLAS_MTP_GATE_FORCE=1` is REQUIRED when measuring acceptance — the
  throughput gate otherwise serial-parks speculation and hides it.
- Propose in `ATLAS_DFLASH_STEP_TIMING` is inflated (carries catch-up
  seeding); use `ATLAS_DSPARK_PROPOSE_PROF=1` for the true split.
- nsys cannot trace prefill on this model (weight load overruns buffers).
  `ATLAS_PROFILE=1` + in-tree `aprof!`/`prof!` probes are the instrument.
- Only ONE server at a time — single shared GB10; the reference container
  and Atlas cannot both hold weights (95+ GiB each in 128 GiB unified).
