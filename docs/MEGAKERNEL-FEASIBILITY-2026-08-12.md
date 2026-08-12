# Persistent-Megakernel Decode on GB10 — Feasibility Study

**Verdict: NO-GO.** Cooperative launch *works* on sm_121 — that part of the idea is
sound and is established below with compiled evidence. But the thing it would buy
is **0.7–1.2 ms/step**, and the `grid.sync()` that replaces each kernel boundary
**costs 2.1–4.3 ms/step**. Expected net is a **loss of 1.0–3.6 ms**. Even
positing a free barrier, the ceiling is **≤1.3 ms (2.9%, +0.6 tok/s)** — under the
3 ms bar this study was told to stop at.

The premise that motivated the idea — "45.3 ms of which only ~32 ms is byte
movement, so ~13 ms is non-byte residue" — does not survive decomposition. Only
~1 ms of that 13 ms is kernel-boundary residue. The rest is bytes moving at their
real per-kernel rates, and it is already itemised in
[`DECODE-WATERFALL-2026-08-10.md`](DECODE-WATERFALL-2026-08-10.md), which closes
to <1% using kernel durations alone. There is no unexplained time to go hunting.

No megakernel was built. Per the brief's instruction, the study stopped at the
arithmetic. What is committed is the evidence: a compile-verified cooperative
probe and a resource-census script, both under `bench/megakernel-feasibility/`
(outside the `kernels/` tree that `build.rs` walks, so neither costs a byte of
build time or JIT).

---

## 1. Cooperative launch on sm_121 / CUDA 13 — VERDICT: SUPPORTED, but unwired

### Toolchain

```
/usr/local/cuda -> CUDA 13.0.88 (nvcc build cuda_13.0.r13.0/compiler.36424714_0)
driver           580.126.09 (NVIDIA Open Kernel Module, aarch64)
device           NVIDIA GB10, compute_cap 12.1, 48 SMs
```

`nvcc --list-gpu-code` includes `sm_121`. `CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH`
(=95) and `cudaLaunchCooperativeKernel` are both present in the CUDA 13 headers;
only the *multi-device* cooperative API is deprecated.

### `this_grid().sync()` compiles clean through the tree's own build path

The tree compiles kernels with `nvcc --ptx -arch=sm_121f -O3` (`build_target.rs:60`,
arch from `kernels/gb10/HARDWARE.toml`) and JITs the PTX text at startup via
`cuModuleLoadData` (`atlas-core/src/registry.rs:200`). The critical question was
whether grid sync needs `-rdc=true` + a `cudadevrt` device link, which that
pipeline cannot do. **It does not.**

`cooperative_groups/details/driver_abi.h::get_grid_workspace()` resolves the grid
barrier by reading `%envreg1`/`%envreg2` — an address the *driver* writes into the
launch environment. The barrier itself
(`cooperative_groups/details/sync.h:68-108`) is inline PTX:
`atom.add.release.gpu.u32` to arrive, `ld.acquire.gpu.u32` to spin. No external
symbols, no relocatable device code.

Verified for both arches:

```
$ nvcc --ptx -arch=sm_121f -O3 --fmad=false -DTQ_PLUS_SIGNS \
      bench/megakernel-feasibility/cg_grid_sync_probe.cu -o /tmp/probe.ptx
$ grep -c envreg /tmp/probe.ptx
2

$ nvcc -arch=sm_121a -cubin --resource-usage \
      bench/megakernel-feasibility/cg_grid_sync_probe.cu -o /tmp/probe.cubin
ptxas info : Compiling entry function 'cg_probe' for 'sm_121a'
             0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
             Used 14 registers, used 1 barriers
```

### What the barrier actually costs — read off the SASS

This is the number the whole study turns on, so it is measured from the emitted
instructions rather than assumed:

```
$ cuobjdump -sass /tmp/probe.cubin
  BAR.SYNC.DEFER_BLOCKING 0x0                       <- full CTA drain
  @P1 MEMBAR.ALL.GPU                                <- device-scope fence
  @P1 ERRBAR ; @P1 CGAERRBAR
  @P1 ATOM.E.ADD.STRONG.GPU PT, R7, desc[UR10][R2.64+0x4], R7
  ... LD.acquire.gpu spin ...
  BAR.SYNC.DEFER_BLOCKING 0x0
```

**`grid.sync()` does not eliminate the kernel-boundary drain — it re-implements
it in software, and adds a device-scope membar plus two dependent global
round-trips on a single contended cache line.** That is the finding that kills
the idea, and everything in §4 is bookkeeping on top of it.

### The launcher cannot issue a cooperative launch today

Production launches go
`KernelLaunch` (`spark-runtime/src/kernel_args.rs:51`) → `GpuBackend::launch`
(`gpu.rs:124`) → `gpu_impl.rs:205` → `AtlasRegistry::launch_on_stream`
(`atlas-core/src/registry.rs:392`) → `cuLaunchKernel` (`registry.rs:424`).

| symbol | status |
|---|---|
| `cuLaunchKernel` | declared `registry.rs:32`, the only launch path in use |
| `cuLaunchCooperativeKernel` | **not declared** in any of the three hand-rolled extern blocks (`atlas-core/registry.rs`, `spark-storage/cuda_module.rs`, `spark-comm/nccl_backend.rs`). Bound in vendored cudarc (`vendor/cudarc/src/driver/sys/mod.rs:11038`) with a safe wrapper at `result.rs:1203` and `LaunchArgs::launch_cooperative` at `safe/launch.rs:243` — **all unused** |
| `cuLaunchKernelEx` | bound in cudarc sys only (`sys/mod.rs:11098`), no safe wrapper, zero uses |

`grep` over `crates/` and `kernels/` confirms the tree has never used cooperative
groups: every hit for "cooperative" is the English word in a comment.

**A cooperative launch therefore needs a new entry point in three places**
(`registry.rs` extern + wrapper, a defaulted `GpuBackend` trait method, a terminal
method on `KernelLaunch`). That is mechanical, ~80 lines, and is *not* the
blocker — §2 and §4 are.

### One safety consequence worth recording

Because `this_grid()` reads `%envreg1/2` unconditionally, launching a grid-sync
kernel through the ordinary `cuLaunchKernel` at `registry.rs:424` is
**memory-unsafe, not merely wrong**: the envregs hold an undefined address and
the barrier's `ATOM.E.ADD.STRONG.GPU` writes to it. Any future cooperative kernel
must be structurally prevented from reaching the normal launch path — a
`KernelHandle` that only the cooperative terminal method accepts, not an env flag.

### Graphs are compatible

`CU_LAUNCH_ATTRIBUTE_COOPERATIVE = 2` is documented "Valid for graph nodes,
launches" (`cuda.h:2079`), with `CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE` at
`cuda.h:2364`. Cooperative kernels capture into graphs as long as MPS is off.
Graph capture is not a constraint here.

---

## 2. The resident-grid ceiling — arithmetic from measured ptxas output

A cooperative launch fails with `CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE` (=720)
unless the whole grid is simultaneously resident.

**Device limits** (`docs/kernels/00-index.md`, from `cudaGetDeviceProperties`
2026-08-03): 48 SMs, `maxThreadsPerMultiProcessor` 1536,
`sharedMemPerMultiprocessor` 102400 B, 65536 32-bit registers/SM (architectural,
CC 5.0+; warp-granular allocation in 256-register units).

```
regs_per_cta  = (threads/32) * ceil(regs*32/256) * 256
blocks_per_SM = min( 1536/threads, 65536/regs_per_cta, 102400/smem_per_cta )
resident_CTAs = 48 * blocks_per_SM
```

**Measured** — `bench/megakernel-feasibility/occupancy_census.sh`, `nvcc
-arch=sm_121a -O3 --fmad=false -DTQ_PLUS_SIGNS`, i.e. the tree's own flags:

| stage (M=1 decode) | kernel | thr | regs | smem/CTA | blk/SM | resident | actual grid | fits alone? |
|---|---|---:|---:|---:|---:|---:|---:|:--:|
| HC pre (mix) | `hc_pre_mix` | 512 | 40 | 2048 | 3 | 144 | 25 | yes |
| HC pre (finish) | `hc_pre_finish` | 256 | 40 | 16 | 6 | 288 | 16 | yes |
| norms | `rms_norm` | 1024 | 38 | 128 | 1 | 48 | 1 | yes |
| wq_a, wkv | `w8a16_gemv` | 256 | 38 | 1056 | 6 | 288 | 256 / 128 | yes |
| wq_b | `w4a16_gemv` | 256 | 40 | 96 | 6 | 288 | **8192** | **no, 28×** |
| wo_a | `w4a16_gemv_grouped` | 256 | 40 | 96 | 6 | 288 | **2048** | **no, 7.1×** |
| wo_b | `w4a16_gemv` | 256 | 40 | 96 | 6 | 288 | **1024** | **no, 3.6×** |
| paged attn | `mla_paged_decode_fp8` | 256 | **128** | **16448** | **2** | **96** | 64 | yes |
| MoE topk | `moe_topk_sqrtsoftplus` | 256 | 40 | 4416 | 6 | 288 | 1 | yes |
| MoE gate+up | `..._gate_up_shared_t_e8m0_v2s4` | 64 | 40 | 0 | 24 | 1152 | 896 | yes |
| MoE silu+down | `..._silu_down_shared_t_e8m0_v2s4` | 64 | 39 | 2048 dyn | 24 | 1152 | 896 | yes |
| MoE finalize | `moe_*_partial_finalize` | 64 | 40 | 0 | 24 | 1152 | 224 / 448 | yes |
| blend | `moe_weighted_sum_blend` | 256 | 42 | 36 | 6 | 288 | 16 | yes |
| HC post | `hc_post` | 256 | 40 | 0 | 6 | 288 | 1 / 16 | yes |
| lm_head (step) | `dense_gemv_fp8w` | 256 | 33 | 32 | 6 | 288 | **32320** | **no, 112×** |

Those are per-kernel ceilings. **A megakernel gets ONE `(blockDim, regs, smem)`
triple for every stage it swallows**, and ptxas allocates for the worst inlined
path. The union is driven by `mla_paged_decode_fp8` at 128 registers + 16448 B
static smem:

| fused blockDim | binding limit | blocks/SM | **resident grid ceiling** |
|---|---|---:|---:|
| 256 | registers: 65536 / (8 warps × 4096) | 2 | **96 CTAs** |
| 64 | smem: 102400 / 16448 | 6 | **288 CTAs** |

**Two independent consequences, either one fatal:**

**(a) Occupancy collapse on the stage that dominates the step.** The MoE expert
GEMV is 20.92 ms/token — 46% of the step — and is DRAM-latency-bound at 192 GB/s.
It runs today at **24 blocks/SM, 896 CTAs**. Inside a 256-thread megakernel it
would run at 2 blocks/SM. Memory-level parallelism falls ~9×. This tree has
already priced exactly this failure: `docs/kernels/00-index.md` lists
"`silu_down` dynamic smem … capping a 32-thread block at 8 blocks/SM = ~17%
occupancy" as a **4–8 ms** regression. A megakernel inflicts a worse version of
that bug by construction.

**(b) Block-size unification breaks bit-identity.** One layer's chain uses
blockDim **64, 128, 256, 512, 576, and 1024**. Every intra-CTA reduction is
defined at its own width: `rms_norm`'s 1024-lane block reduction over H=4096;
`hc_pre_mix`'s 512-lane `float4` dot over 16384 terms; the GEMV family's 64
threads/row × 4 rows; the MoE `_t` kernels' 64-thread VEC=2/SPLIT=4 layout;
`mla_cache_assemble_batched`'s 576. Forcing one blockDim re-maps at least four of
these onto a different summation tree — **different bits**. The tree already
documents this exact hazard at `hyper_connection.cu:440-445`, where switching
`hc_pre_mix` to `float4` lanes reassociated the dot and cost ~1e-7 drift, so that
split decode is no longer bit-identical to prefill.

The brief's own requirement — *"numerically identical to the existing chain —
same math, same order, only the synchronization mechanism changes"* — is
**unsatisfiable** for any fused chain spanning more than one block size. And
under the exact-GEMV law recorded in `DECODE-WATERFALL-2026-08-10.md:150`
("partial exactness is worse than none"), that is a hard stop, not a tunable.

**Which stages could fit:** exactly the ones that are already cheap. The 8 glue
stages (HC pre/post, norms, rope, cache) all fit under 96 CTAs. Every GEMV —
which is where 80%+ of the bytes and time are — needs a persistent grid-stride
rewrite. That rewrite is *legal* (each CTA owns whole output rows, so the K-order
inside a row is untouched and per-row bit-identity survives), but it is a rewrite
of the entire GEMV family for the residue quantified in §4.

---

## 3. Scope — what one layer's chain actually contains

One V4-Flash decode layer at M=1, default config (`ATLAS_HC_SPLIT` split,
`ATLAS_V4_DECODE_FUSED` **off**, FP8 KV, `ATLAS_UNIFIED_MOE_LAYOUT=1`, split-K on):
**34 kernel launches + 1 D2D copy = 35 graph nodes.**

```
hc_pre_mix → hc_pre_finish → rms_norm → wq_a → q_a_norm → wq_b → q_b_norm
  → wkv → kv_a_norm → [D2D K→V] → q_rope_extract → k_rope_extract → rope
  → q_rope_writeback → k_rope_writeback → cache_assemble → write_kv_cache
  → paged_attn → derot_extract → rope_inv → derot_writeback
  → wo_a_grouped → wo_b → hc_post
  → hc_pre_mix → hc_pre_finish → post_attn_norm
  → moe_gate → moe_topk → gate_up → gate_up_finalize → silu_down
  → down_finalize → wsum_blend → hc_post
```

(Note: HC-pre runs *before* the norm, not after — the brief had that pair
inverted. There is no separate residual add; `hc_post` *is* the residual.)

**Step totals:** 35 × 43 = **1462 nodes/token**, plus ~6 at step level
(embedding, final norm, `hc_head`, lm_head, sampling). With
`ATLAS_V4_DECODE_FUSED=1` the per-layer count drops to 27 → **~1118/token**.

This corrects the waterfall doc's "860 launches", which is a back-inference from
an 8 ms eager/graphed delta, not a count. **No dispatch counter exists in-tree**
(`gpu.launch_count()` is mock-backend only, `spark-runtime/src/gpu/mock.rs:47`).
The 1462 is counted from source.

### The collapse a megakernel would achieve

At M=1 essentially **every** boundary in that chain is a cross-CTA dependency —
each stage's output is a full 4096–32768-wide vector consumed by a reduction or a
GEMV over all of it. The only CTA-local boundaries are the ones
`ATLAS_V4_DECODE_FUSED` already collapses. So:

| | today | one-layer megakernel |
|---|---:|---:|
| per layer | 35 graph nodes | 1 launch + **33 grid.syncs** |
| per token | 1462 nodes | 1 launch + **~1420 grid.syncs** |

**The boundaries are not removed. They are re-implemented in software** — which,
given the SASS in §1, is the whole problem.

---

## 4. The arithmetic — what the residue is actually made of

### Where the "13 ms" goes

`45.3 ms` graphed step; `6.7 GB/token ÷ 229 GB/s = 29.3 ms` byte floor.

| ms/step | component | kernel-boundary residue? |
|---:|---|---|
| 3.3 | MoE `exp_unified_t` at 192 GB/s vs the 229 ceiling (486.5 vs 409 µs/layer) | **No** — dedup access pattern |
| 0.26 | small-N GEMVs (kv_proj 170 GB/s, bf16 n144 144 GB/s) | **No** |
| ~1.15 | the *entire* elementwise / norm / mHC / rope / cache category, realistic (`docs/kernels/04`) — of which 0.61 is irreducible bytes | Partly; 0.85 ms of it is two one-line **grid** fixes, not fusion |
| 0.7–1.2 | graph-node dispatch + CTA teardown/relaunch over 1462 boundaries | **Yes — the entire addressable market** |
| remainder | bytes moving at their real per-kernel rates | **No** |

The framing "45.3 ms of which ~32 ms is bytes, so 13 ms is residue" treats every
sub-ceiling byte as residue. It isn't. The waterfall closes to **<1%** using
kernel durations alone.

### Derivation of the removable part

**(1) Eager launch cost — already banked, not available again.**
`(50.4 − 45.3) ms / 1462 launches = 3.5 µs/launch`. Consistent with the "eager
kernel-launch floor ≈ 4–6 µs" in `docs/kernels/00-index.md`. CUDA graphs took
this in 2026-08-06 (`+18%`, the "plain decode graphs were the artifact" finding).

**(2) What remains at each boundary under graph replay:**

- *Graph-node dispatch.* Bounded above by the tree's own evidence: the waterfall
  reconciles to <1% of 45.3 ms from kernel durations alone ⇒ ≤0.45 ms
  unattributed over 1462 nodes ⇒ **≤0.31 µs/node**.
- *CTA teardown + relaunch* — register/smem dealloc-realloc, i-cache and
  constant-bank refetch for the new kernel: **~0.2–0.5 µs/boundary**. This is the
  **only** component `grid.sync` genuinely removes.
- *Ramp* (CWD filling 96–1152 CTAs): ~0.1 µs, partly removable by persistence.
- *Tail-wave quantisation:* **not removable**. `grid.sync` waits for every CTA
  exactly as a kernel boundary does, and in a persistent grid-stride loop the
  same imbalance reappears as a ragged final iteration.

**Removable ≈ 0.5–0.8 µs × 1462 = 0.7–1.2 ms/step.**

**(3) Cost of the replacement barrier.** From the §1 SASS: per `grid.sync`, per
CTA — `BAR.SYNC` (drain) → `MEMBAR.ALL.GPU` (device-scope fence) →
`ATOM.E.ADD.STRONG.GPU` on one global word → `LD.acquire.gpu` spin → `BAR.SYNC`.
Two dependent device-scope round trips plus N-way serialised atomics on a single
cache line. On GB10 (unified LPDDR5X, L2 round trip ~250–400 ns) at 96–1152 CTAs:
**~1.5–3 µs per grid.sync**.

**~1420 grid.syncs × 1.5–3 µs = 2.1–4.3 ms/step.**

### Bottom line

```
  removable boundary residue      +0.7 .. +1.2 ms
  grid.sync barrier cost          -2.1 .. -4.3 ms
  ------------------------------------------------
  NET                             -1.0 .. -3.6 ms      (a 2-8% LOSS)

  free-barrier upper bound        +1.3 ms = 2.9% = +0.6 tok/s
```

Both the realistic figure and the unattainable ceiling are **under the 3 ms bar**.
Stop.

---

## 5. The software-pipelined alternative — also NO-GO, for different reasons

Spin flags + `__threadfence()` instead of `grid.sync`, on the theory that it
dodges the residency cap. It does not, and it is worse:

1. **It does not escape the residency cap — it just removes the diagnostic.** A
   consumer CTA spinning on a producer's flag occupies an SM slot. If the
   producer CTA is not resident it can never be scheduled, and the spin is a
   deadlock. So the same resident-grid guarantee is required, but instead of
   `cuLaunchCooperativeKernel` returning
   `CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE` at launch you get a hung GPU in
   production. (The escape hatch — assuming CTAs are scheduled in increasing
   `blockIdx` order — is not architecturally guaranteed on any NVIDIA part.)

2. **The barrier is not cheaper.** A flag handshake *is* a device-scope membar
   plus a global atomic/acquire-load round trip. `cg::this_grid().sync()` already
   is exactly that, hand-written, with the barrier word supplied by the driver
   instead of by us. There is no cheaper primitive underneath.

3. **Its one real advantage does not apply here.** The published megakernel wins
   come from letting stage *k+1* begin before stage *k* drains. That needs
   independent work to overlap. At M=1 the V4 layer chain is strictly serial —
   every stage consumes the *whole* of the previous stage's output (H=4096
   vectors feeding 4096-wide reductions). There is nothing to overlap. This is
   why the technique pays on small dense models, where a stage's output *tile*
   feeds one consumer tile, and does not pay here.

Additionally, work-stealing (the usual way to fix load imbalance in a persistent
kernel) would let any CTA take any row, changing which lanes reduce which
partials — another bit-identity violation.

---

## 6. What to do instead

Every one of these is already identified in-tree and has a strictly better
ratio than the megakernel's **unattainable** 1.3 ms ceiling:

| gain | lever | cost |
|---:|---|---|
| **3.3 ms** | MoE expert GEMV 192 → ~220 GB/s (SASS/bank-conflict audit) | the real lever; 2.5× the megakernel's best case |
| **~0.5 ms** | `hc_post` at the post-attn site runs on **1 CTA** while the identical post-ffn site uses 16. `post_shards` is computed at `decode_inner.rs:521` and not passed at `:689`. 43 sites/token. | **one line**, zero numerical effect |
| **~0.5 ms** | `hc_pre_mix` grid is 25 blocks sized to a false 25-SM assumption (`hyper_connection.cu:220,253`); 23 of 48 SMs idle. Split-k to 50. | one launcher line |
| 0.1–0.3 ms + DRAM round trips | `ATLAS_V4_DECODE_FUSED=1` — **already built, default OFF**, already bit-identity-gated by `v4_decode_fused_microtest`. Removes 344 launches/step (1462 → 1118). | flip a flag after re-running the oracle |
| ~20 µs each | the 10 adjacent-kernel fusions tabulated in `docs/kernels/04-elementwise-norm-cache.md` | ordinary fusion |

Note the fourth row. **The megakernel's thesis — collapse the glue, keep the data
resident across stage boundaries — is already implemented in this tree as
ordinary kernel fusion, needs no cooperative machinery, has a byte-identity
oracle, and is switched off.** That is where the idea's value actually lives.

---

## 7. Reproduction (no GPU required)

```bash
# 1. Cooperative groups compile clean for sm_121 through the tree's own flags
nvcc --ptx -arch=sm_121f -O3 --fmad=false -DTQ_PLUS_SIGNS \
     bench/megakernel-feasibility/cg_grid_sync_probe.cu -o /tmp/probe.ptx
grep -n 'envreg\|atom.add.release\|ld.acquire' /tmp/probe.ptx

# 2. What a grid.sync costs, in instructions
nvcc -arch=sm_121a -cubin --resource-usage \
     bench/megakernel-feasibility/cg_grid_sync_probe.cu -o /tmp/probe.cubin
cuobjdump -sass /tmp/probe.cubin | grep -E 'BAR.SYNC|MEMBAR|ATOM|ERRBAR'

# 3. Resident-grid census — regs/smem for every M=1 decode kernel
bench/megakernel-feasibility/occupancy_census.sh

# 4. The cooperative-launch API is absent from the production launcher
grep -rn 'cuLaunchCooperativeKernel\|cuLaunchKernelEx' crates/    # -> no hits
grep -rn 'cooperative_groups\|this_grid\|grid\.sync' crates/ kernels/  # -> no hits
```

## 8. If this is ever revisited

The verdict is contingent on three measurements, any of which could change:

1. **`grid.sync` latency on GB10 is estimated, not measured** (1.5–3 µs, derived
   from the SASS shape and LPDDR5X L2 round-trip). A GPU-side microbenchmark
   would settle the sign of the net. If a `grid.sync` at 96–288 CTAs turned out
   to cost <0.4 µs, the net would go slightly positive — still ≤1 ms, still under
   the bar, but the argument would rest on §2 (occupancy + bit-identity) alone.
2. **The 128-register / 16448-byte `mla_paged_decode_fp8` sets the ceiling.** If
   MLA attention were ever restructured to ~64 registers, the 256-thread
   megakernel ceiling rises 96 → 192 CTAs. It would not change the barrier
   arithmetic.
3. **This is an M=1 argument.** The γ-verify path batches m=6 rows and has a
   genuinely different shape — larger grids, more per-stage work, better
   amortisation. The "no independent work to overlap" claim in §5.3 is specific
   to M=1 and would need re-deriving before being applied to the verify step.
