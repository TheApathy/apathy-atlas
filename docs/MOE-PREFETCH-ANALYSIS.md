# Predictive MoE Expert Prefetch — Locality Instrument + Tier Verdict

**Date**: 2026-08-12 · **Hardware**: one GB10 (DGX Spark, sm_121) · **Model**:
DeepSeek-V4-Flash-162B NVFP4 · **Branch**: `moe-expert-prefetch` off
`combined-residency`

**Status**: instrument LANDED (`ATLAS_MOE_ROUTE_LOG=1`, default off, zero cost).
Prefetch mechanism NOT implemented — **NO-GO on mechanism grounds**, see §3.
The go/no-go on locality is a measurement the reader must run (§5); it does not
change the §3 verdict, but it is worth having because it *does* decide the
NVMe/RDMA expert-tier prefetcher (§6), which is a real two-tier system.

---

## 1. The idea under test

MoE decode serializes: `[layer L compute] → [router sees L's output] → [pick 6 of
256 experts] → [stream ~93.6 MB] → [compute]`. Layer L+1's expert weights cannot
start moving until layer L finishes. If consecutive tokens reuse experts, a
predictor could start layer L+1's expert stream during layer L's compute and
break that dependency.

The MoE expert stream is the largest decode bucket by a factor of five:

| quantity | value | source |
|---|---|---|
| MoE `exp_unified_t` | 486.5 µs/layer × 43 = **20.92 ms/token** | `docs/DECODE-WATERFALL-2026-08-10.md` §3 |
| expert bytes | 93.6 MB/layer → **4.02 GB/token** at 192 GB/s | ibid. §3 reconciliation |
| full plain decode step, graphed | **45.3 ms = 21.9 tok/s**, 6.7 GB/token | ibid. §1 |

Model shape (`kernels/gb10/deepseek-v4-flash/MODEL.toml`): 43 layers, hidden
4096, `num_experts = 256`, `top_k = 6`, `moe_intermediate_size = 2048`, 1 shared
expert. One expert = 3 × 2048 × 4096 weights at NVFP4 (4 bits + one FP8 scale per
16) ≈ 14.2 MB nominal; the measured 93.6 MB/layer over 6 routed + 1 shared works
out to **≈ 13.4 MB per expert**. Both numbers give the same conclusion below.

---

## 2. The locality instrument (task 1) — `route_locality.rs`

### 2.1 Why the two existing instruments do not answer this

Both in-tree expert-set instruments measure the **wrong axis**:

| instrument | axis | file |
|---|---|---|
| `ATLAS_MOE_UNION_STATS=1` | union ACROSS THE ROWS of one verify batch (same layer, same step) | `crates/spark-model/src/layers/moe/union_stats.rs` |
| `ATLAS_MOE_OVERLAP=1` (commit `baa21190`) | same cross-row quantity, from the per-row decode path, split hash-vs-gate | `crates/spark-model/src/layers/moe/dump.rs:275+` |

Both bound the dedup'd `_m` verify kernel's speedup. A prefetcher needs the third
axis — for one **fixed layer**, how well does `S_t` predict `S_{t+1}` across
**steps**. Nothing measured that. `route_locality.rs` does.

### 2.2 What it computes

With `S_t` = the top-k expert-id set at token `t` for a fixed layer, and
`|S_t| = top_k` exactly (the top-k kernel always emits `top_k` distinct ids):

```
carry     = E[ |S_t ∩ S_{t-1}| ] / top_k        == P(e used at t | e used at t-1)
cov(W)    = E[ |S_t ∩ (S_{t-1} ∪ … ∪ S_{t-W})| ] / top_k     hit rate of a W-window predictor
cost(W)   = E[ |S_{t-1} ∪ … ∪ S_{t-W}| ] / top_k             bytes fetched, as a multiple of top_k
hotN      = share of routed slots landing in the layer's N most-used experts (N = k, 2k, 4k)
```

`carry` is exactly the number the task asked for. `cov/cost` is its
generalization: a predictor only helps if coverage climbs faster than cost,
because on GB10 the prefetch and the demand fetch contend for the **same DRAM**.
`hotN` evaluates the *static* predictor — if routing is skewed enough that a
fixed hot set covers most fires, no temporal prediction is needed at all.

Reported per layer and aggregated, split `hash` vs `gate` (DeepSeek-V4 hash-MoE
layers select via `tid2eid[token_id]` — a static function of the token id, so
their locality is luck of the draw; learned-gate layers route on the hidden state
and are where real temporal locality could live).

**Compare every number against the random-routing baseline `top_k/num_experts =
6/256 = 0.0234`.** A carry of 0.10 sounds small but is 4× random; a carry of 0.30
is 13× random and still (see §3.4) below the break-even for the only mechanism
that exists.

### 2.3 Layer identity without touching every weight loader

`MoeLayer` has no `layer_idx` field and threading one through would edit every
loader constructor (`deepseek_v4/assemble.rs`, `step3p7`, `minimax`, `qwen35`,
`mtp_head`, …). Instead the layer is keyed by the **address of the `MoeLayer`
instance**, unique and stable for the process lifetime; first-seen order of those
addresses is model order, because the layer loop visits layers in order. `L0…L42`
in the output are true MoE-layer ordinals. Zero intrusion, exact.

### 2.4 Hook site and cost

One call in `crates/spark-model/src/layers/moe/forward.rs`, immediately after the
top-k kernel, beside the existing `dump::route_group_row` — that is the M=1
decode arm, which is also the routine the per-row speculative verify calls
(`multi_seq/mod.rs`: "MLA models always take this path").

- **Off (default)**: one cached `bool` load per MoE fire.
- **On**: one `synchronize(stream)` + one `top_k*4`-byte D2H per fire. This
  perturbs **step timing** — do not read tok/s from a logging run — but not
  routing, which is what is being measured.

### 2.5 The two ways it silently sees nothing

1. **Inside CUDA-graph capture** a D2H invalidates the capture (CUDA 901). The
   call site is guarded by `!ctx.graph_capture` — same rule as `union_stats.rs`.
2. **CUDA-graph REPLAY runs no host code at all.** Once decode is graphed the tap
   goes blind and counters freeze. On DeepSeek-V4 graphs engage at `seq_len >
   266` (`fp8_kv_calibration_tokens = 256`). A long run therefore **must** also
   suppress graphs — `ATLAS_PROFILE=1` does, by design. **Check that `fires=` is
   rising before trusting any number.**

### 2.6 Env surface

| var | default | meaning |
|---|---|---|
| `ATLAS_MOE_ROUTE_LOG` | unset | `=1` enables |
| `ATLAS_MOE_ROUTE_LOG_EVERY` | 2048 | global fires between summary lines |
| `ATLAS_MOE_ROUTE_LOG_FILE` | unset | append a raw CSV trace: `fire,layer,tok_index,hash_routed,ids…` |

The CSV is the escape hatch: it lets any statistic be computed offline, not just
the four above.

---

## 3. PREFETCH-TIER VERDICT (task 3) — **there is no tier**

This is the crux, and it is decided by hardware, not by the locality number.

### 3.1 What the machine actually is

GB10 is a **single coherent LPDDR5X pool** — 119.7 GB unified, ~273 GB/s
theoretical, **229 GB/s measured achieved ceiling**
(`docs/MODEL-MIGRATION-PLAYBOOK.md` §5). There is **no discrete VRAM**. Expert
weights are allocated once at load with plain `cuMemAlloc_v2`
(`crates/spark-runtime/src/fast_weights/mod.rs:354-374`) into that one pool and
never move again. `cuMemAllocManaged` exists in the backend
(`crates/spark-runtime/src/gpu.rs:90`) **solely as an OOM escape hatch** that
lets Linux page to NVMe — its own doc comment says so.

The tree already records this conclusion, from a different investigation:
`crates/spark-model/src/weight_loader/laguna/load_layers.rs:307-317` rejects
mixed 4.5/3-bit experts partly because *"cudaHostRegister pins pages in the SAME
physical pool (unified memory)"* — i.e. there is no second place to put bytes.

### 3.2 Enumerate the candidate mechanisms and eliminate them

| mechanism | verdict |
|---|---|
| **`cuMemPrefetchAsync`** | **Inapplicable twice over.** (a) It requires *managed* memory; expert weights are `cuMemAlloc_v2` device memory, so the call returns `CUDA_ERROR_INVALID_VALUE`. (b) Even on managed pointers it is a *page-migration* primitive — it moves pages between physical memories. There is one physical memory. The pages are already there. It cannot bring bytes closer because there is nowhere closer. (Absent from first-party code; present only in unused vendored cudarc, `vendor/cudarc/src/driver/result.rs:771`.) |
| **`cuMemAdvise`** | Same. A page-*placement* hint on a machine with one place. |
| **L2 `accessPolicyWindow` / `cudaStreamSetAttribute`** | The **only real tier that exists** — and it is 4× too small (§3.3). It also **does not fetch**: an access-policy window changes the *eviction priority* of accesses made through it; it never moves a byte on its own. Realizing it requires actually executing loads (§ next row). Absent from the tree entirely. |
| **Warm-up "toucher" kernel on a side stream** | The only thing that can actually pull bytes into L2. It spends the **same DRAM bandwidth** the demand read would, and burns SM issue slots on 48 SMs concurrently with attention. Without a persisting window the touched bytes are evicted by the 93.6 MB stream long before use — a pure bandwidth tax. |
| **Side stream + event join** | Fully available (`create_stream`/`create_event`/`record_event`/`stream_wait_event`, `crates/spark-runtime/src/gpu.rs:285-324`) — but the *plumbing* was never the missing piece. |

### 3.3 The capacity arithmetic that kills it

`cudaGetDeviceProperties` on this box (`docs/kernels/00-index.md:89`):
`l2CacheSize = 25165824` (**24 MB**), `persistingL2CacheMaxSize = 18874368`
(**18 MB**).

```
one layer's expert working set   93.6 MB
one expert                       ~13.4 MB  (measured share) / 14.2 MB (nominal)
L2 total                          24 MB
L2 persisting carve-out (max)     18 MB    →  fits exactly ONE expert. Two (26.8 MB) do not.
```

**The tier's capacity is 1 of the 6 routed experts per layer — 14 % of the
stream.** That is the ceiling on everything below.

### 3.4 Best case, granting a perfect oracle

Assume a perfect predictor, perfect L2 persistence, and free prefetch bandwidth:

```
hit  → MoE reads 93.6 − 13.4 = 80.2 MB from DRAM
       saves 13.4 MB / 192 GB/s = 69.8 µs/layer × 43 = 3.00 ms/token
       step 45.3 → 42.3 ms  =  23.6 tok/s   (+7.7 %)
```

That is the **entire upside of the idea**, with an oracle. Now the deductions:

- **A miss costs real bandwidth.** 13.4 MB of DRAM read that would not otherwise
  happen; if it fully contends, 13.4 MB / 229 GB/s = 58.5 µs/layer × 43 =
  2.52 ms/token. Break-even hit rate:
  `p·3.00 = (1−p)·2.52 → p = 2.52/5.52 = **0.457**`.
  The predictor must be right about a *specific* expert **≥ 46 %** of the time
  just to be neutral.
- **The prior says it will not be.** The nearest existing measurement
  (`docs/2026-08-04-dspark-measured.md` §"γ sweep"): at γ=2 the union of two
  rows' expert sets is 10.3 of 12 slots → intersection 1.7 → **pairwise overlap
  ≈ 0.28**. That is 12× the random baseline (0.023) — the locality is *real* —
  but it is **below the 0.457 break-even**, and it is a hash+gate mixture whose
  hash half cannot beat random. §5 measures the honest per-layer number.
- **Carving 18 of 24 MB out of L2 hurts the incumbent.** The MoE split-K kernels
  reuse activations and scale tables through L2 (`docs/kernels/02-moe.md`), and
  `docs/kernels/00-index.md:89` records that the KV pools are L2-resident. A
  persisting window that takes 75 % of L2 pushes both back toward DRAM.
- **Concurrent side-stream traffic is measured net-negative on this box, twice.**
  `crates/spark-model/src/layers/moe/forward_prefill.rs:138` — `let use_overlap
  = false; // disabled: dual-stream contention worsens LPDDR5X bandwidth`.
  `crates/spark-model/src/layers/moe/helpers_b.rs:78-90` — the lazy-transpose
  side stream *"regressed cold TTFT by ~30 % on GB10"*. And
  `docs/atlas-on-the-table.md:98` — in-kernel cross-group double-buffer prefetch
  regressed the m6 leg 1.777 → 1.983 ms (+12 %), because ptxas already pipelines
  the loads and manual buffers only add register pressure.

**Verdict: NO-GO.** Ceiling +7.7 % with an oracle; break-even needs a hit rate
the prior says we do not have; and the only mechanism that could deliver it is
the exact pattern this hardware has rejected three times.

### 3.5 The one thing that is NOT the constraint

Worth stating plainly, because it inverts the premise. The step has **plenty of
idle bandwidth**:

```
step bytes 6.7 GB / 45.3 ms  = 148 GB/s average   (ceiling 229 GB/s)
unused bandwidth-time        = (229 − 148) × 45.3 ms ≈ 3.7 GB per step
```

MoE (192 GB/s) has ~0.8 GB of slack in its own window; the MLA GEMV chain
(232.6 GB/s) and lm_head (235 GB/s) are **at the ceiling** with none; and the
remaining ~13 ms of norms/rope/HC/paged-attn carries almost no bytes at all —
that is where ~3 GB of free bandwidth-time sits.

So the budget to move 91 % of the 4.02 GB expert stream into idle windows
**exists**. What does not exist is anywhere to *put* the bytes: 3.7 GB of free
bandwidth-time against 0.58 GB of total per-step staging capacity
(13.4 MB × 43 layer-windows). **The binding constraint is staging capacity, not
routing prediction and not bandwidth.** Breaking the routing dependency — the
thing the whole idea is about — buys nothing, because the tier it would feed is
7× too small.

---

## 4. Mechanism design review (task 2) — and why (c) dominates (a) and (b)

| option | predictor | hit rate | verdict |
|---|---|---|---|
| **(a) "same experts as last token"** | zero-cost, `S_{t-1}` | = `carry`, prior ≈ 0.28 | below the 0.457 break-even |
| **(b) learned/heuristic predictor on L's hidden state** | needs a model + a GEMV in the critical path | unknown, ≤ 1.0 | strictly worse than (c): more cost, less certainty, and no capacity left for it |
| **(c) shared expert + attention weights for L+1** | **none needed — static per layer** | **1.0** | **the only sane target, and it makes (a)/(b) pointless** |

Option (c) is not merely "the risk-free subset" — it **dominates**. The shared
expert is 2048-intermediate, i.e. **the same ~13.4 MB as a routed expert**, and
it is needed by every token with certainty. It therefore fills the entire 18 MB
persisting carve-out at a 100 % hit rate, delivering the *whole* +3.00 ms ceiling
of §3.4 with **zero** mispredict risk. The MLA weights for L+1 (49.4 MB/layer,
also static) are a second no-prediction queue behind it.

**Consequence: even with a perfect routing oracle, predicted experts are third in
line for a tier that holds one item.** The routing dependency is not worth
breaking. If anyone ever builds an L2-residency scheme here, it should prefetch
the shared expert, and the locality question never enters.

(And even option (c) still has to beat §3.4's other three deductions — the L2
carve-out cost, the toucher kernel's SM contention, and this box's three
measured overlap regressions. It is *plausible*, not *promising*; ~+3 ms/token
best case. It is a separate experiment, not part of this change.)

---

## 5. Measurement protocol (run this)

The verdict in §3 stands on hardware capacity and does not depend on the
locality number. Run this anyway: it is the input to §6, and it converts "the
prior says 0.28" into a measured per-layer, hash-vs-gate split.

```bash
# 1) plain decode (GAMMA="-"), graphs suppressed, raw trace on.
#    ATLAS_PROFILE=1 is REQUIRED — under graph replay the tap is blind (§2.5).
scripts/dsflash-serve-bench.sh routelog - \
  ATLAS_MOE_ROUTE_LOG=1 \
  ATLAS_MOE_ROUTE_LOG_FILE=/home/flocka/dsflash-combined/route-locality.csv \
  ATLAS_MOE_ROUTE_LOG_EVERY=2048 \
  ATLAS_PROFILE=1

# 2) drive ≥300 decode tokens on a ≥450-token prompt, four workloads
#    (prose / code / repeat / quote — repeat-only overstates locality badly,
#     since repeated text re-routes to the same experts by construction).
python3 scripts/decode_ab_probe.py routelog 8977 1

# 3) read the summary
grep 'moe-route-log' serve-routelog.log | tail -60
```

**Sanity gates before believing anything:**

- `fires=` must climb across successive summary lines. Frozen ⇒ graphs engaged
  ⇒ the tap is blind (§2.5) ⇒ the numbers describe only the first ~266 tokens.
- `layers=` should reach the MoE-layer count (43 on V4). Fewer ⇒ some layers
  never fired eagerly.
- Compare `carry` to **0.0234** (= 6/256, random). Compare it to **0.457**
  (break-even, §3.4).
- Read `gate` rows separately from `hash` rows. Hash layers route on
  `tid2eid[token_id]` and should sit near random except where tokens repeat;
  only the `gate` rows can carry real hidden-state locality.
- Read `hot6/hot12/hot24`. If a fixed hot set covers most slots, the static
  predictor beats every temporal one — and again needs no prediction.

**Interpretation table:**

| measured aggregate `carry` | reading |
|---|---|
| ≈ 0.023 | routing is memoryless; even the NVMe-tier prefetcher (§6) is dead |
| 0.05 – 0.45 | real locality, still below the L2 break-even — §3 verdict unchanged; §6 still viable |
| > 0.457 | locality clears the L2 break-even — but capacity (§3.3) and the three measured overlap regressions (§3.4) still bind, and option (c) still dominates at hit rate 1.0 |

---

## 6. Where this instrument's answer actually matters

A prefetcher needs a **latency gap between two tiers**. There is no such gap for
resident experts (§3). There *is* one, already built, for **non-resident**
experts:

- `crates/spark-storage/src/expert_tier.rs:12-21` defines the residency order
  **`device < UMA-over-NVMe < RDMA`**.
- `crates/spark-storage/src/expert_arena.rs:38-70` allocates a pinned host arena
  whose host VA == device VA on GB10, so records `pread(O_DIRECT)`'d into a slot
  are directly consumable by the fused MoE kernels with no H2D bounce.
- **`ExpertTier::fetch` (`expert_tier.rs:70`) is a *synchronous* pread on the
  calling thread, and the UMA tier ignores its `stream` argument.** It is
  demand-fetch, with no prefetch at all.

There the tier gap is NVMe (single-digit GB/s) vs LPDDR (229 GB/s) — two orders
of magnitude, not zero — and the staging capacity is the whole arena, not 18 MB.
A hit rate well under the 0.457 that L2 needs would pay for itself many times
over. **If the routing locality measured in §5 is meaningfully above random, the
correct place to spend it is `expert_tier.rs`, not L2** — and that is a
capacity-unbounded, genuinely two-tier system where "prefetch" means something.

---

## 7. What landed

| file | change |
|---|---|
| `crates/spark-model/src/layers/moe/route_locality.rs` | new — the instrument (§2), 4 unit tests |
| `crates/spark-model/src/layers/moe/forward.rs` | one call after top-k, inside the existing `!ctx.graph_capture` guard |
| `crates/spark-model/src/layers/moe/mod.rs` | `mod route_locality;` |
| `docs/MOE-PREFETCH-ANALYSIS.md` | this file |

**No prefetch mechanism was implemented**, because §3 finds none exists on this
hardware. `ATLAS_MOE_PREFETCH` is deliberately *not* a knob: shipping a disabled
mechanism that cannot work would be worse than the finding.

Validation: `cargo test -p spark-model --lib route_locality` (4/4 pass);
`cargo clippy -p spark-model --lib --no-deps` introduces zero new findings (the
crate has 54 pre-existing `manual_is_multiple_of` lints from a newer clippy,
none in the changed files).
