# Qwen3.8-27B on GB10 — production recipe and measurement record

Status: current as of 2026-08-23. Every number here is measured on this box;
estimates are labelled as such. Where a lever was tested and found null, it is
recorded as null — the negative results are the most expensive part of this
document and the most useful.

---

## 1. Headline numbers

Canonical probe: `qwen38/benchmark/weschera_minheap_repro.py` — the "Weschera
MinHeap" prompt, single stream, greedy, thinking off, median of 5.

| Configuration | 400-token probe | 1500-token probe |
|---|---:|---:|
| Campaign start (2026-08) | ~48 | — |
| Historical reference | 51.26 | 41.22 |
| **Current production** | **63.96 / 63.77** | 43.33 |
| Same, split-K off | 62.86 / 62.92 | — |

Two arms are quoted for the production rows because they were measured
interleaved in one session; the spread is the run-to-run drift.

AEON v4 text suite, best full board (greedy + all serving fixes, 2026-08-23):

| Category | Score | Notes |
|---|---:|---|
| Coding | 27/36 | easy 2/2, medium 3/3, hard 5/5, expert 7/8, frontier 9/12, god_mode 1/6 |
| Math | 27/34 | easy/medium 100%, hard 4/5 |
| Reasoning | 23/35 | easy/medium/hard 100% |
| Instruction | 19/34 | easy/medium 100%, hard 4/5 |
| Prose | 17/35 | |
| **Total** | **113/174 (65%)** | reference attested run: 147/174 (84.5%) on an RTX PRO 6000 |

The reference submission (`unsloth/Qwen3.8-27B-NVFP4`, stock vLLM) ran on an
RTX PRO 6000 Blackwell — ~1792 GB/s against GB10's ~273 GB/s. That 6.5x
bandwidth difference explains its 107-150 tok/s decode; the quality gap is a
separate question and is not fully explained by hardware.

---

## 2. The serve recipe

`qwen38/benchmark/serve-qwen38-quality.sh` carries the quality profile;
`arms/atlas-fork.sh` is the benchmark launcher. Speed profile:

```
MODEL_DIR=qwen38/optimized-qwen-unsloth-official     # see §3 for provenance
DRAFT_OVERRIDE=qwen38/drafter-qwen38-v2-epoch4-step24852
GAMMA=15  MTP_VOCAB=96000  ATLAS_DDTREE_MAX_NODES=16
KV_CACHE_DTYPE=bf16  KV_HIGH_PRECISION_LAYERS=0  SEQS=1  MAXLEN=8192
ATLAS_FFN_TC=1  ATLAS_SSM_PROJ_TC=1  ATLAS_LM_HEAD_TC=1  ATLAS_ACCEPT_FAST_ARGMAX=1
ATLAS_PREFILL_PROJ_FAST=0  ATLAS_PREFILL_FFN_FAST=0  ATLAS_DFLASH_FREE_SLOTS=0
ATLAS_SSM_GDN_SEQ_PERSISTENT=1  ATLAS_ATTN_QKV_FUSED=1  ATLAS_SSM_GDN_LAZY=1
ATLAS_DFLASH_DRAFT_SPLITK=8
ATLAS_WEIGHT_CACHE=1
CONFIDENCE_EARLY_STOP=off  SIMHASH_WATCHDOG=off  LOOP_WATCHDOG=off  THINK_LOOP_WATCHDOG=on
TOOL_CALL_PARSER=hermes
```

Quality profile differs in: `MAX_THINKING_BUDGET=57344`,
`ATLAS_MIN_THINKING_TOKENS=2048`, `MAXLEN=65536`,
`KV_HIGH_PRECISION_LAYERS=8`, greedy (no sampling override).

### Why each non-obvious flag is set

| Flag | Effect | Evidence |
|---|---|---|
| `GAMMA=15` | draft width | Optimal **and** maximal: the drafter's `block_size=16` gives `trained_drafts = block_size-1`. gamma 20 is refused by the loader. Measured 10/12/15 = 56.71/59.30/62.92 — monotonic into the ceiling. |
| `ATLAS_SSM_GDN_LAZY=1` | skips 15-of-16 discarded snapshot writes | +5.16 tok/s, bit-exact (replays the identical FP32 recurrence on commit) |
| `ATLAS_SSM_GDN_SEQ_PERSISTENT=1` | keeps H in registers across the block | +1.70 tok/s. Together with lazy commit this cut `ssm_gdn_fp32_seq` from 16.9 to 4.9 ms/step |
| `ATLAS_ATTN_QKV_FUSED=1` | fuses QKV projection | +0.79 tok/s |
| `ATLAS_DFLASH_DRAFT_SPLITK=8` | splits K on occupancy-starved drafter GEMMs | +1.13 tok/s (64.07 vs 62.94). Reassociates the K loop so bit-exactness is unproven; measured byte-identical on the probe |
| `ATLAS_WEIGHT_CACHE=1` | caches post-transform weights | Weight-load phase 17 s vs ~45-60 s. See `docs/weight-cache.md` |
| watchdogs mostly off | see §6 | each was measured killing healthy output |

---

## 3. Weights, drafter, and cache

**Target.** `unsloth/Qwen3.8-27B-NVFP4` @ `7d6f8d4d72f56b92b3cdbf22f156b90e1bab0108`.
Byte-verified against upstream: `model.safetensors` sha256 `c473512c70eace07…`,
`model_mtp.safetensors` `1d8268aa85ace093…`, 2 shards, 21.8 GB. Upstream
super-squashed the repo on 2026-08-15, which is why older revision hashes
(including the one the reference submission cites) now 404.

**Drafter.** `drafter-qwen38-v2-epoch4-step24852`, DFlash family, 69 tensors,
3.96 GiB BF16, `block_size: 16`. The engine quantises 6 layers x 7 dense + fc
to NVFP4 at load; the BF16 sources are retained (~3.3 GB held) because
`gpu.free()` on GB10 UVM posts in-band TLB invalidations that corrupt
neighbouring allocations (BUG #29). That is memory cost, not bandwidth.

Alternatives, both measured worse: the official `incoai` DFlash2 drafter runs
52% acceptance against v2's 66% and lands ~7 tok/s behind; DSpark checkpoints
measured 23.4 and 9.2 tok/s in separate audits.

**Weight cache.** ~13 GB per model variant, LRU-bounded to 32 GB, keyed on a
fingerprint including transform-affecting env vars and weight content samples.
704 slot-verifications with 0 failures. Full contract in `docs/weight-cache.md`.

---

## 4. Where the cycle goes

Cycle census (`ATLAS_DFLASH_SPEC_CYCLE_V2=1`, CUDA graphs on, 765 records,
gamma 15, 1500-token probe):

```
verify_complete   106.56 ms   80.5%
propose_complete   22.75 ms   17.2%
accept              1.84 ms    1.4%
TOTAL             132.42 ms
emitted/cycle       5.87        (accepted 4.87 of 15)
```

Verify cost is linear in draft width. Three-point fit (gamma 15/11/7):

```
verify(k) = 75.53 + 1.890k ms      residuals -0.20 / +0.41 / -0.20
```

The intercept is the weight sweep: 16.327 GB at 232 GB/s achievable = **70.37 ms**,
so verify runs at ~1.5x its floor. The slope was **2.996 ms/node** before
persistent-H and lazy commit landed; those changes cut it 1.59x.

**Acceptance falls with generation length** — the single most important
quality-of-drafter signal we have:

| probe length | accepted / 15 |
|---|---|
| 400 tokens | 7.23 (48.2%) |
| 800 tokens | 5.81 (38.7%) |
| 1500 tokens | 4.87 (32.5%) |

Per-position conditional match rate is **flat** at ~0.87 across 10,999 cycles
(position 1 = 0.88, position 12 = 0.88), and a constant-hazard model with a
single p=0.87 predicts E[L]=5.43 against a measured 5.42. The drafter's errors
are uniformly distributed, i.e. semantic difficulty — not local incoherence.

---

## 5. Numerics: what is bit-exact and what is not

The engine's stated contract was once "verify commits the scalar-oracle token".
That was deliberately retired on 2026-08-17 (REFREEZE block,
`benchmark/arms/atlas-fork.sh:54-62`). Current measurements are relative to
reference hash `12e0c0ad`, not to the scalar oracle.

| Change | Bit-exact? |
|---|---|
| GDN lazy commit, persistent-H, QKV fuse | **yes** |
| Weight cache | **yes** (704 verifications, 0 failures) |
| `ATLAS_FFN_TC`, `ATLAS_SSM_PROJ_TC`, `ATLAS_LM_HEAD_TC` | **no** — MMA reduction order + BF16 weight rounding |
| split-K (`ATLAS_FFN_DOWN_SPLITK`, `ATLAS_DFLASH_DRAFT_SPLITK`) | **no** — K-loop reassociation |

"Lossless" in older comments meant FP32 partials with no mid-accumulation BF16
rounding. That is true and is a different claim from token-exactness. Witness:
for `[2^24, 1, 1, -2^24]`, left-to-right FP32 gives 0 and a 2+2 split gives 1.

The bit-exact path remains available via `ATLAS_FFN_TC=0 ATLAS_SSM_PROJ_TC=0`
(hash `f376a16e`) and costs 27.2 vs 31.2 tok/s at gamma 6.

---

## 6. Gotchas that have cost real time

1. **`SEQS>8` hard-reboots the host.** Unified memory; the launcher refuses it.
   A `SEQS=16` corpus run caused a global OOM and took the machine down.
2. **`GPU_MEM_UTIL` up = KV pool down.** Measured: 0.55 → 16.6 GB allocatable
   (9338 blocks); 0.68 → 5.7 GB (3960 blocks). Do not "tune" it upward to get
   more batch — you get less.
3. **Kernel builds need the target env.** `ATLAS_TARGET_MODEL=qwen3.8-27b
   ATLAS_TARGET_QUANT=nvfp4 cargo build --release -p spark-server`. Without it
   the build silently targets `qwen3-next-80b-a3b` and the server will not serve.
4. **MODEL.toml is not tracked by the kernel build cache.** `touch
   crates/atlas-kernels/build.rs` after editing it or your change is a no-op
   with an unchanged binary hash.
5. **Model-level vs quant-level MODEL.toml.** `kernels/gb10/qwen3.8-27b/MODEL.toml`
   is read; `.../nvfp4/MODEL.toml` is not.
6. **`target/release/spark` is rebuilt with different kernel targets.** Pin the
   binary into `qwen38/benchmark/bin/` and grep it for the target string before
   trusting a measurement.
7. **`ATLAS_FULL_PROFILE` disables CUDA graphs.** It is still representative
   here (graphs are worth only ~2.5 ms of a 106 ms verify), but do not compare
   profiled absolute times against census times without knowing that.
8. **Dispatch shadowing.** `ATLAS_ATTN_QKV_BATCHED`, `ATLAS_ATTN_QKV_SPLITK`
   are inert on gated Qwen3.8 at the widths we serve — `exact_attention_qkv_route`
   returns early for n=4..17. `ATLAS_FFN_DOWN_SPLITK` is live only because
   `ATLAS_FFN_TC=1` forces the exact route off. Before trusting any flag, find
   the function that reads it and walk up to the first early return.
9. **Client-side kills do not cancel server-side generation.** Killing a corpus
   generator leaves its in-flight requests running; a subsequent 5-token probe
   measured 87 s while queued behind them.

---

## 7. Levers measured and found null

Recorded so they are not re-derived. Each cost real GPU time.

| Lever | Result |
|---|---|
| rt2 register-tiled GEMV (upstream PR 648 port) | +0.6%, inside drift |
| Exact-GEMV route instead of TC tiles | **2.5x slower** (24.98 vs 62.84) |
| `ATLAS_TC_NVFP4_M16` / `_MS_ATTN` on QKV | −30% / −27%; both together −36%. Independently reproduces a 2026-08-19 result (51.48 → 39.73) that was never written up |
| Kernel-launch batching in attention | zero — CUDA graphs already absorb it |
| k16 vs k8 load width | 0.61% of the instruction stream |
| Wave/tail utilisation | ~3% of the layer |
| CTA supply in attention | 3072/512 CTAs, 11x past saturation |
| Register pressure | attention 72 regs vs lm_head 76 — lm_head is worse and 3.6x faster |
| DDTree wide trees (old slope) | needed +45% acceptance at 31 nodes |
| PCTree (arXiv 2608.02123) | our DFlash checkpoint has no Markov head; and at b=1.890 the break-even needs +26.9% against a published +18.6% |
| Markov fixup on the chain | per-position hazard is flat; a perfect head caps at +9.2% |
| FFN dispatch variants | null, inside baseline drift |

**Unresolved:** `attn_qkv_proj` runs at ~55 GB/s against 313 GB/s that `lm_head`
achieves on the same machine with worse register pressure. Every mechanism we
can compute has been eliminated. The remaining hypothesis is that the true
floor is L2-bound on a 15.1x activation re-read (~4.0 ms, not the 2.85 ms
weight-bytes figure). `crates/spark-model/examples/w4a16_attention_qkv_throughput_probe.rs`
settles it in one GPU-minute and **has not been run**.

---

## 8. What remains for 70+ tok/s

Arithmetic, from §4: `tok/s = emitted_per_cycle / cycle_seconds`. Every kernel
term is at or near its floor. The open lever is acceptance:

- today: p ≈ 0.87 per token, 6.81 emitted/cycle, 62.9 tok/s
- for 70: p ≈ 0.92, ~9.2 emitted/cycle
- champion-class drafter, measured on its own target: p = 0.956

Acceptance is the open lever, and it is a property of the drafter rather than
the engine — every kernel term above is at or near its bandwidth floor, so no
serving-side knob moves it. Drafter work is out of tree.

---

## 9. Reproducing

```bash
cd /path/to/apathy-atlas

# build. Both target vars are mandatory, and nvcc must be on PATH: cudarc's
# build script shells out to a bare `nvcc` and does not consult CUDA_HOME.
export PATH=/usr/local/cuda/bin:$PATH
ATLAS_TARGET_MODEL=qwen3.8-27b ATLAS_TARGET_QUANT=nvfp4 \
  cargo build --release -p spark-server

# serve (speed profile)
MODEL_DIR=/path/to/Qwen3.8-27B-NVFP4 DRAFT=/path/to/dflash-drafter \
  ./bench/qwen38-gb10/serve.sh

# measure
python3 bench/qwen38-gb10/weschera_minheap_repro.py \
  --output /tmp/minheap.json --repetitions 5 --max-tokens 400
```

`arms/atlas-fork.sh`, referenced elsewhere in this document, is a working file
on the measurement box and does not ship. `bench/qwen38-gb10/serve.sh` is the
published equivalent.

Interleave A/B arms rather than running them sequentially — sequential arms on
this box manufacture phantom deltas. Baseline drift is ~±1 tok/s; treat
anything smaller as noise.

### Packaged reproduction

The same recipe, the same probe, and the same floor are packaged as a container
in `qwen38/container/production-v2/` — pinned binary, baked drafter, mounted
target, `make repro`. See
[`QWEN38_PRODUCTION_CONTAINER.md`](QWEN38_PRODUCTION_CONTAINER.md). Use it when
you want the number reproduced rather than the knobs varied; use the launcher
above when you are varying knobs.

---

## 12. What decode speed is actually attributable to

Measured 2026-08-24 on the published container, MinHeap probe, 400 tokens.
Decode rate is one ratio:

    tok/s = emitted_per_cycle / cycle_time

|                  | emitted/cycle | cycle time | tok/s |
|---|---:|---:|---:|
| no speculation   | 1.00 | 72.1 ms | 13.9 |
| γ=7              | 6.06 | 112.3 ms | 54.0 |
| γ=15             | 8.33 | 129.8 ms | 64.2 |

Speculation multiplies tokens-per-cycle by **8.33** while multiplying cycle cost
by only **1.80**. That ratio, 4.62x, is the entire speedup. The **numerator is
the drafter** (acceptance x depth); the **denominator is the engine**. Both are
required and neither alone gets there.

Isolated A/B contributions to the denominator, same drafter, same probe:

| Change | Δ tok/s |
|---|---:|
| GDN lazy commit + persistent-H | **+6.48** (63.80 vs 57.32) |
| split-K draft head | +1.13 (64.07 vs 62.94) |
| tensor-core verify flags | **+0.25** (64.10 vs 63.85) — null |

**The tensor-core result corrects an earlier claim in this document.** The TC
flags were described as load-bearing for speed. They are not: disabling all
three costs 0.25 tok/s. They remain a numerics re-reference (§5) — that part
stands — but they are not where the throughput comes from.

### Depth saturates, and more is not better

Solving the measured γ=15 point (8.33 emitted) under a constant-hazard model
gives per-token acceptance **p ≈ 0.904**. Expected accepted converges to
`p/(1-p)` = **9.4 tokens**, so draft width beyond ~γ12 buys almost nothing while
verify cost keeps growing at 1.890 ms/node:

| γ | emitted/cycle | cycle ms | tok/s |
|---:|---:|---:|---:|
| 15 | 8.33 | 129.8 | **64.2** |
| 23 | 9.47 | 151.3 | 62.6 |
| 31 | 9.98 | 172.8 | **57.7** |

A `block_size: 32` drafter at today's acceptance is a **loss**, not a win.
Depth only pays if acceptance rises with it:

| p | best γ | tok/s at best γ |
|---:|---:|---:|
| 0.904 (today) | 17 | 64.5 |
| 0.924 | 20 | 74.4 |
| 0.944 | 24 | 88.5 |

**70 tok/s needs p ≈ 0.924**, about +0.02 over today. That is a corpus and
training question, not a kernel or draft-width one — and it is why the remaining
work is drafter work.
