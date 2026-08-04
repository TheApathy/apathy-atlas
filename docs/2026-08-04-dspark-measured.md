# DSpark / DFlash on one GB10 — measured 2026-08-04

Binary: `combined-residency` @ `0c468dde`, DeepSeek-V4-Flash-162B, single GB10,
`--gpu-memory-utilization 0.91 --kv-cache-cap-tokens 1040 --max-num-seqs 1`.
Launcher: `scripts/cb-serve.sh`. All numbers are `response_token/s` from the
server's own usage block, steady state (first run of a fresh server discarded).

## Headline

Speculation finally beats plain decode — but only with **adaptive + low-gear**,
and only on structured content. Raw γ=6 DFlash loses to plain on everything.

| workload | plain | adaptive+low_gear | delta | reproducible? |
|----------|-------|-------------------|-------|---------------|
| repeat   | 21.48 | **26.2–26.5**     | **+23%** | yes, ±0.2 over 5 runs |
| quote    | 21.38 | **23.4–23.7**     | **+10%** | yes, ±0.2 over 5 runs |
| code     | 20.04 | 20.8–25.6         | 0 … +28% | **no — bimodal** |
| prose    | 21.33 | 19.2–19.4         | −10% | yes, ±0.2 over 5 runs |

**The code figure does not replicate.** Same server, same prompt, same config,
back-to-back: pass 1 = 20.76, pass 2 = 25.12. Whether the run lands fast depends
on where adaptive's suspend/re-probe cycle happens to fall relative to the
generation. Treat code as "sometimes 25, sometimes plain-equal", not as +28%.

`repeat` and `quote` are the solid wins: both stable to ±0.2 tok/s across five
runs and both well clear of plain.

Config that produced the middle column:

```
ATLAS_MTP_GATE_FORCE=1 ATLAS_DFLASH_ADAPTIVE=1 ATLAS_DFLASH_LOW_GEAR=1 \
ATLAS_DFLASH_UNIFIED_CTX=1
```
on top of the mandatory `ATLAS_UNIFIED_MOE_LAYOUT=1 ATLAS_DSPARK_CAPTURE=1
ATLAS_V4_ATTN_NVFP4=1 ATLAS_V4_ATTN_RELEASE_BF16=1 ATLAS_MOE_MROW_PARTITION=1`.

## Config ladder (same four prompts)

| config | code | repeat | quote | prose |
|--------|------|--------|-------|-------|
| plain decode | 20.04 | 21.48 | 21.38 | 21.33 |
| γ=6 forced, no adaptive | — | — | — | 14.30 |
| MTP gate left to arbitrate | — | — | — | 20.2 (gate picks serial) |
| adaptive only (no low-gear) | 20.46 | 26.41 | 23.69 | 19.38 |
| **adaptive + low-gear (REPROBE=256, default)** | 20.8–25.6 | **26.37** | **23.48** | 19.20 |
| adaptive + low-gear, REPROBE=128 | 22.26 | 26.39 | 23.57 | 18.28 |
| adaptive + low-gear, REPROBE=1024 | 20.68 | 26.36 | 23.53 | 19.78 |

`ATLAS_DFLASH_ADAPTIVE_REPROBE` was tested in both directions and the default
256 wins. 1024 is too slow to resume — a 400-token generation never re-probes at
all, so it degrades to the adaptive-only row. 128 re-probes so often that the
wasted probes cost more than the hot stretches recover (prose 19.20 → 18.28).

Two things this ladder settles:

* **Low-gear is what makes code ever reach 25.** With `LOW_GEAR` dropped, code
  never exceeds ~20.5 in any run. Code is dense in repeated tokens (`self.`,
  `def `, type hints, docstring scaffolding), which is exactly what the
  host-side longest-suffix n-gram in `scheduler/low_gear.rs` catches. It is
  *not* the neural drafter carrying that case. Low-gear costs nothing on the
  other three workloads, so it should stay on.
* **The prose regression is not low-gear.** Prose sits at 19.2 with low-gear
  and 19.4 without. It is the cost of adaptive re-probing: every 256 serial
  tokens it spends ~12 steps × ~190 ms re-testing the drafter before suspending
  again.

## γ sweep — γ=6 was the wrong operating point

Forced spec (no adaptive), prose, 300 tokens:

| γ | tok/s | expert union |
|---|-------|--------------|
| 2 | 16.91 | 10.3 |
| 3 | 16.96 | 12.9 |
| 4 | **17.32** | 15.4 |
| 6 | 14.30 | 18.1 |

The union grows sublinearly (+2.6, +2.5, +1.35 experts per extra row) but
acceptance does not keep up past 4, so γ=6 pays for bytes it cannot convert.
γ=4 is the raw-throughput optimum — a 21% gain over γ=6 for free.

Under adaptive + low-gear the picture is workload-dependent, because γ also sets
the cost of each failed adaptive re-probe:

| config | code | repeat | quote | prose |
|--------|------|--------|-------|-------|
| plain | 20.04 | 21.48 | 21.38 | 21.33 |
| γ=6 forced + adaptive + low-gear | 20.8–25.6 | **26.4** | 23.5 | **19.3** |
| γ=4 forced + adaptive + low-gear | **24.7–25.2** | 23.1 | 23.3 | 17.3 |
| γ=6 gate-arbitrated + adaptive + low-gear | 20.7–22.9 | 21.4–24.2 | **24.4–25.1** | 19.0 |

γ=4 is the one config where code is *stable* (24.7/25.2 across passes instead of
γ=6's 20.8–25.6 coin-flip), but it gives back repeat and prose. Letting the MTP
gate arbitrate instead of forcing it does **not** rescue prose — adaptive has
already suspended by the time the gate samples, so the gate sees acceptable
throughput and keeps MTP on.

**Recommended: γ=6 forced + adaptive + low-gear.** Best on 2 of 4 workloads,
competitive on the third, and the prose loss is bounded at −10%. Switch to γ=4 if
the workload is known to be code.

## Verify attribution, re-measured after the batching work

The attribution below the fold ("attention 46.7 ms = 33% of verify") is
**stale and was the basis for two wrong priorities.** Re-measured on
`389d8b2b` + `ATLAS_MOE_T_BLOCK=128` with `ATLAS_PROFILE=1`
`ATLAS_DFLASH_DEBUG_NO_GRAPH=1`, 774 layer-samples per phase:

| MLA phase | µs/layer | ×43 layers | what it is |
|-----------|----------|-----------|------------|
| A_proj  | 215.2 | 9.3 ms | batched Q + KV projections |
| B_attn  | **81.2** | **3.5 ms** | rope + cache write + paged attention |
| C_oproj | 303.3 | 13.0 ms | output projection (o_lora_rank=1024, o_groups=8) |
| **MLA total** | 599.7 | **25.8 ms** | |

Phase B was **467 µs/layer** before this session's two commits. It is now
81 µs — a **5.7× collapse** — because the whole rope/cache chain runs once for
all γ rows (`decode/rows_rope_cache.rs`) and the paged attention now runs as a
single `num_seqs=γ` launch instead of γ launches.

Three consequences, all of which redirect effort:

* **Attention is no longer a lever.** The actual attention kernel is 3.5 ms of
  a 140 ms verify — 2.5%. Task #29 (head-tile the paged decode so 64 Q heads
  stop re-reading KV) is now worth at most a couple of ms and should be
  deprioritised.
* **`C_oproj` is now 3.7× `B_attn`.** The output projection is the largest
  single piece of MLA and has never been looked at.
* **MoE is ~114 ms of the 140.5 ms verify (81%)**, not the 77.4 ms the stale
  table claimed. At ~10.4 GB of expert weights per step (18.1 unique experts ×
  3 matrices × 8.389 M params × ~0.53 B/param × 43 layers) that is **~99 GB/s**,
  against the m=1 plain-decode path's 154 GB/s on the same weights. Closing
  *that* ratio is the whole remaining story.

### Session deltas (all bit-exact, γ=6, `ATLAS_DFLASH_STEP_TIMING=1`)

| change | verify | Δ |
|--------|--------|---|
| baseline (per-row phase B) | 151.0 ms | — |
| + batched rope/cache write (`559bcb9d`) | 145.1 ms | −5.9 |
| + `num_seqs=γ` paged attention (`389d8b2b`) | 143.2 ms | −1.9 |
| + `ATLAS_MOE_T_BLOCK=128` | **140.5 ms** | −2.7 |
| (`ATLAS_MOE_T_BLOCK=256`) | 140.8 ms | saturated |

**−10.5 ms total (−7.0%).** Every leg was verified with the four-workload
sha256 probe: 8/8 exact text matches per leg against the baseline, two passes
each. Nothing here trades accuracy for speed.

`T_BLOCK` works because `fp8_moe.rs:565-568` sizes shared memory as
`arm_mrow * k * 4 / split` — **independent of `t_block`**. The MROW_PARTITION
duplicated arm (`arm_mrow=6, k=4096, split=4`) therefore burns 24 KB/block and
caps residency at ~9 blocks/SM; at the default `T_BLOCK=64` that is only 18
warps, far too few to hide the load latency the kernel is documented to stall
on. Doubling `t_block` doubles warps per block at identical shared memory. It
saturates at 128, which says occupancy was a real but secondary constraint —
the m=6 GEMV does not become bandwidth-bound just by adding warps.

## Why raw γ=6 loses — the step budget (stale attribution below)

`ATLAS_DFLASH_STEP_TIMING=1` / `ATLAS_STEP_TIMING2=1`, 64-step means:

```
verify=149.2ms  walk=0.6ms  propose=36.9ms  other=0.02ms   TOTAL=186.7ms/step
```

At 14.296 tok/s that is 2.68 tok/step — up from the 2.12 recorded before the
device-resident compressed-block fix (`8175867c`), so that fix did land.

The problem is bandwidth, not acceptance:

| path | bytes/step | time | achieved |
|------|-----------|------|----------|
| plain decode | 7.227 GB | 46.9 ms | **154 GB/s** |
| γ=6 verify | 14.408 GB | 149.2 ms | **96.7 GB/s** |

The verify moves 2.0× the bytes but costs 3.2× the time. It runs at 43% of the
227 GB/s usable figure while plain decode runs at 68%. That 96.7 → 154 GB/s gap
is the single largest remaining lever in the whole engine.

Per-token: verify is 69.6 ms/token against plain's 46.9 ms/token. Speculation is
*48% worse per token* at γ=6 before adaptive gets involved.

## Expert union — better than assumed

`ATLAS_MOE_UNION_STATS=1` at m=6, top_k=6:

```
mean_unique_experts=17.4–18.8   mean_routed_slots=30.6–34.5   overlap_saving=43–46%
```

The codex `three-x-frontier.md` budget assumed **20.5**. The real union is
~18.1. The ≤14.0 bridge gate is therefore closer than the report implies, and
the 14.408 GB/step verify budget is ~7% pessimistic.

## The 40 tok/s arithmetic

When the drafter is hot the accept trace is genuinely good — a sustained run
during the code prompt read `3,4,3,5,5,3,5,5,4,3,5,5,5,5,4` (γ=6 caps at 5
drafts). So ~4.5 tok/step is reachable; the ≥4.20 bridge gate is not the
blocker.

Holding 5 tok/step and varying only verify bandwidth:

| verify @ | verify ms | step ms | tok/s |
|----------|-----------|---------|-------|
| 96.7 GB/s (today) | 149 | 186 | 26.9 |
| 154 GB/s (plain's rate) | 93.6 | 130.5 | **38.3** |
| 227 GB/s (usable peak) | 63.5 | 100.4 | **49.8** |

**Getting the verify path to merely match plain decode's own achieved bandwidth
lands 38 tok/s.** That is the 40 tok/s path, and it needs no new acceptance work
— only the MoE verify kernel.

Note `propose=36.9 ms` is 20% of the step and is next after verify.

## What does not work

* **Plain decode cannot reach 40.** 7.227 GB/token × 40 = 289 GB/s, above the
  273 GB/s theoretical LPDDR5X peak. Plain's ceiling is 31.4 tok/s at 227 GB/s
  usable; we are at 21.3.
* **γ=6 without adaptive** is 14.3 tok/s — strictly worse than plain. The MTP
  gate correctly detects this on its own (`switching Mtp -> Serial (current 14.1
  vs other 19.0)`) and is right to.
* **T_BLOCK=64 and the MLA KV-alias** (commit `90c483fd`) are worth ~0.3 tok/s
  end-to-end on plain: 21.0 → 21.3. The occupancy argument is sound in isolation
  but does not show up at the token level.

## Versus the ds4-on-spark forum numbers

[NVIDIA devforum #378855](https://forums.developer.nvidia.com/t/1x-spark-deepseek-v4-flash-0731-1-000-tok-s-prefill-59-tok-s-multi-agent-serving/378855)
reports 1x Spark, DeepSeek-V4-Flash-0731, antirez's `ds4` fork, **2-bit quant**:

| their metric | value | comparable to ours? |
|--------------|-------|---------------------|
| 59 tok/s "multi-agent serving" | aggregate over **12 concurrent** streams = 4.9 tok/s/stream | **no** — we run `--max-num-seqs 1` |
| 28 tok/s chat decode @ 12k, speculation on | single stream | **yes** — this is the number to beat |
| 22 tok/s @ 240k | single stream | no — we cap at 1024 |
| ~1000 tok/s prefill | — | not measured here |

So the honest scoreboard is **28 (theirs, 2-bit, 12k ctx) vs 19.3–26.5 (ours, 1024 ctx)**,
and their 28 would be *higher* still at our 1k context, not lower.

The gap is dtype, not kernels. Decode is byte-bound; we measured plain decode at
7.227 GB/token and 154 GB/s achieved. To hit 28 tok/s at *our* achieved
bandwidth you need ≤5.5 GB/token — a 24% byte cut. A 2-bit recipe cuts far more
than 24% off an FP8/NVFP4 mix, so `ds4` is **not** achieving 154 GB/s: it is
buying tokens with dtype and giving most of it back in kernel efficiency. Our
kernels are ahead per byte moved; our bytes per token are behind.

Two consequences:

* The 59 tok/s headline is not a target for this work. Per stream it is 4.9
  tok/s, which single-stream plain decode beats 4× today.
* The cheapest remaining path to 40 tok/s single-stream may not be the MoE
  verify kernel at all — it may be a lower-bit expert recipe. That is untested
  here and would need an accuracy check the forum post does not provide.

## Reproduce

```bash
cd /home/flocka/dsflash-combined
./scripts/cb-serve.sh plain -                  # baseline leg
./scripts/cb-serve.sh spec 6 ATLAS_MTP_GATE_FORCE=1 \
    ATLAS_DFLASH_ADAPTIVE=1 ATLAS_DFLASH_LOW_GEAR=1 ATLAS_DFLASH_UNIFIED_CTX=1
python3 /tmp/probe.py                          # four-workload probe
```

The residency flags are not optional: without `--kv-cache-cap-tokens` the
budget arm turns all free memory into KV blocks (5.7 GB / 123k slots) when one
1024-token stream needs 51.5 MB, and the loader OOMs once the 10.86 GB drafter
and the graph pools land on top.
