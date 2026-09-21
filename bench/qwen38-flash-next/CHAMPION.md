# Qwen3.8-Flash-Next — champion recipe

The counterpart to `bench/qwen38-gb10/README.md` for the dense 27B. This is the
**qualified** launch configuration and the list of everything that has been
tried against it and rejected, with the number and the reason for each.

Independently re-measured 2026-08-30 on reaper (GB10). Every figure below is
from a run on this box, not carried over.

---

## 1. The champion recipe

```bash
ATLAS_PLE_CACHE_MB=512 \
spark serve \
  --model-from-path /path/to/Qwen3.8-Flash-Next-NVFP4-Offload \
  --model-name qwen3.8-flash-next \
  --kernel-target qwen3.8-flash-next \
  --port 8898 \
  --max-seq-len 2048 \
  --max-num-seqs 1 --max-batch-size 1 \
  --ssm-cache-slots 8 \
  --kv-cache-dtype bf16 \
  --no-tui
```

Wrapped by `serve.sh` in this directory. **Speculation is deliberately absent.**

**Measured: 42.22 tok/s decode, TTFT 0.95 s**, deterministic across trials.
Matches the recorded 42.1010 median. The binary must be built with
`ATLAS_TARGET_MODEL=qwen3.8-flash-next`; confirm before measuring, since
`target/release/spark` is rebuilt in place by any other build in the tree:

```bash
strings -a <elf> | grep -c qwen3.8-flash-next   # must be non-zero
```

### Why these specific flags

| flag | why |
|---|---|
| `--max-seq-len 2048` | keeps the request under the QSA boundary; `--qwen4-qsa` is then unnecessary |
| `--ssm-cache-slots 8` | GDN state slots. `0` costs ~1.5% |
| `ATLAS_PLE_CACHE_MB=512` | PLE sparse-row cache |
| `--kv-cache-dtype bf16` | `fp8` yields FEWER KV blocks here (21 vs 258) and makes the **baseline nondeterministic** |
| (default) `ATLAS_QWEN4_PLE_SEGMENTED_GRAPHS=1` | PLE-boundary segmented CUDA graphs: eager control 38.3594 -> 42.1010 (**+9.8%**). Set `=0` only as an eager diagnostic |

**42.22 tok/s is the ceiling, not a plateau to optimise.** Decode is
bandwidth-saturated: 5.59 GiB/token against a measured 238.7 GB/s bound implies
~39.8 tok/s, so the box is running at ~105% of the naive bound. There is no
non-speculative headroom.

---

## 2. Rejected: every speculation route

Speculation on Flash-Next is not merely unqualified — every route measured is a
**loss** against the 42 tok/s control.

| route | tok/s | vs control | note |
|---|---:|---:|---|
| **champion (no speculation)** | **42.22** | **1.00x** | — |
| native MTP K5, qualified batched route | 37.80 | 0.91x | best speculation available; output-identical |
| native MTP K2 | 35.73 | 0.86x | **NONDETERMINISTIC** across identical greedy requests — separate bug |
| native MTP K3 | 32.72 | 0.78x | output differs |
| DFlash2 gamma=15 (donor bridge) | 2.88 | 0.07x | 0.32/15 acceptance; output-identical |
| DFlash2 + `ATLAS_QWEN4_K16_BATCHED_VERIFY=1` | ~2.7 | 0.06x | removes the per-token MoE loop, still slower |

This reproduces the record's donor-bridge figure (3.32 tok/s, "essentially zero
draft acceptance") from an independent implementation path.

### Why no drafter can currently win

The drafter hard-requires gamma=15 (`native Qwen3.8-Flash-Next V3 uses
bidirectional B16 attention and requires gamma=15`; gamma 2 and 7 are refused).
That forces K=16 — the one width with **no qualified batched verify route**
(`verify_d.rs` qualifies K=2, K=3, and K=5-with-hybrid) — so verify runs the
row-serial oracle at ~418 ms.

At 23.8 ms/token control and a ceiling of `gamma+1 = 16` tokens per step:

| configuration | break-even needs | ceiling | verdict |
|---|---:|---:|---|
| propose 35 ms + verify 418 ms | 19.75 accepted+1 | 16 | impossible |
| **free propose + verify 418 ms** | 18.28 accepted+1 | 16 | **impossible** |

**Even a free propose at 100% acceptance yields 36.8 tok/s = 0.88x.** Verify must
fall below ~381 ms before any acceptance wins, and below ~155 ms for 2x. So this
is an infeasibility result for the current pairing, not a tuning gap.

**The champion dense recipe does not transfer.** Applying all ~30 env vars from
`~/atlas/clean/patches/serve-aeon-champion.sh` leaves verify **unchanged at
418 ms** (2.88 vs 2.87 tok/s). Its KGAMMA/FA2/splitk kernels target dense
qwen3.6-27b FFN/attention paths that Flash-Next never enters; its DDTree stack —
the real acceptance lever, 11.14/16 on the dense model — leaves **0 KV blocks**
on a 106 GB model; and `--dflash-quantization nvfp4` makes propose worse
(35 -> 48 ms) and nondeterministic.

---

## 3. Rejected: every prefill batching route

Prefill is serialized by default: `qwen3_ssm/trait_prefill.rs:32` runs
`decode_inner` **once per prompt token** unless opted out. Confirmed free of the
warmup confound — with `ATLAS_MOE_SITE_TRACE=1` the counters are empty after
warmup, then scale with prompt length (61 tokens -> 1,000 calls; 1,274 -> 20,000+).

Serialized prefill measures **40.4 tok/s marginal**, i.e. TTFT 26.9 s at 1,274
tokens and 62.5 s at 2,528 — essentially the decode rate, which is the tell.

| route | TTFT (1274 tok) | speedup | output vs serial oracle |
|---|---:|---:|---|
| **serial (default)** | 18.67 s | 1.00x | oracle |
| `ATLAS_QWEN4_ATTN_PREFILL_BATCH=1` | 14.85 s | 1.26x | **differs** (3.7% agreement) |
| `ATLAS_QWEN4_SSM_PREFILL_TILE=2` | 16.48 s | 1.13x | differs (3.7%) |
| `ATLAS_QWEN4_SSM_PREFILL_TILE=3` | 15.69 s | 1.19x | differs (34%) |
| `ATLAS_QWEN4_SSM_PREFILL_TILE=4` | 31.14 s | **0.60x** | differs |
| `ATLAS_QWEN4_SSM_PREFILL_BATCH=1` | 6.63 s | **2.79x** | **garbage — 0% agreement from char 0** |

Two things worth keeping:

* **Attention-only batching does not help correctness.** The 12 full-attention
  layers are not recurrent, so batching only those looked numerically safe. It
  is not — output still diverges at 3.7%. So the divergence is not solely the
  GDN state carry.
* **A kernel qualified for VERIFY is not qualified for PREFILL.**
  `trait_prefill.rs:39` claims K=3 uses families "already qualified by
  speculative verification"; measured, tile 2 and 3 both change prefill output.
  In verify, a batched row is checked against the target's own argmax and
  mismatches are rejected — self-correcting. In prefill the batched result
  silently **becomes the context**. Nothing checks it.

`ATLAS_QWEN4_SSM_PREFILL_BATCH=1` should be documented as **broken**, not
experimental: at a 12-token prompt, "What is 2+2?" answers `4` serialized and
`VML, a company that has been a leader in the 3D printing industry...` batched.

### The second prefill blocker

Even with a correct batched path, the fast grouped GEMM is unreachable:

    WARN load_layers: Skipping MoE weight transposition
         (63.3 GB needed, 17.2 GB available). Prefill will use fallback grouped GEMM.

`weight_loader/qwen35/load_layers.rs:106-113` sizes a **second full copy** of
every expert's weights. With a 106 GB model on 128 GiB it never fits, so prefill
always takes the fallback. Not tunable — the shortfall is 46 GB. The fix is a
JIT per-layer transpose (`63.3 / 48 = 1.32 GB` live at a time, ~265 ms of added
traffic per prefill) or a pre-transposed checkpoint.

---

## 4. Build inventory

| build | size | usable |
|---|---:|---|
| **NVFP4-Offload** | 106 GB | **yes — the champion target** |
| NVFP4-Radix | 126 GB | no — OOM (mixed bf16 + fp8-PLE, larger not smaller) |
| FP8 | 173 GB | no |
| Inferact-MTP | 1.6 GB | MTP sidecar only, not a target |

The served build is already the smallest usable one, consistent with decode
sitting at its bandwidth bound.

---

## 5. Open work, in value order

1. **A Flash-Next-native drafter that does not force gamma=15.** This is the only
   thing that unlocks decode speculation. With verify on a qualified K=2/3/5
   route, 75 tok/s needs per-token p ~= 0.84 and 150 tok/s needs p ~= 0.96 — both
   reachable in principle at gamma=15-equivalent widths, neither reachable with
   any current checkpoint. This is training, not configuration.
2. **The batched prefill state carry.** A measured 2.79x TTFT sits behind it, it
   is decode-neutral, and the serialized path is an exact oracle to debug
   against. Compare `h_state`/`conv_state` per layer after prefill to localize
   the divergence to a layer and a tensor.
3. **The MoE transposition memory shape** (item 2's multiplier).
4. **Native MTP K2 nondeterminism** — a correctness bug independent of speed.

## 6. Do not re-chase

* Stale drafter-ring rows — the arm with 600 stale rows had *higher* acceptance.
* F1 reflection suppression — gated it as `ATLAS_REFLECTION_SUPPRESS`; acceptance
  identical to two decimals, baseline output byte-identical with it off.
* The hyperconnection capture projection — `ATLAS_DFLASH_DEBUG_CTX_OFF=1`
  collapses acceptance 0.358 -> 0.062, so the captured states carry real signal.
* The accept fast path — `ATLAS_ACCEPT_FAST_ARGMAX=0` changes nothing; it logs
  zero engagements.
* A smaller build — see the inventory above.

---

## Prefill — `ATLAS_QWEN4_SSM_PREFILL_ROWWISE=1` (2.62x, measured 2026-08-31)

Add to the serve environment:

    ATLAS_QWEN4_SSM_PREFILL_ROWWISE=1

| prompt tokens | baseline TTFT | rowwise TTFT | prefill before -> after |
|---:|---:|---:|---|
| 253 | 5.81 s | 2.53 s | 43.5 -> 99.9 tok/s (2.29x) |
| 1811 | 42.50 s | 16.20 s | 42.6 -> **111.8 tok/s (2.62x)** |

Decode is unaffected (40.80 -> 41.07). The win grows with prompt length.

**Why it works.** The shipping per-token prefill route runs the ENTIRE layer —
norm, QKVZ GEMM, out-proj and the MoE FFN — once per prompt token, which is why
prefill measured ~40 tok/s, indistinguishable from decode. Only conv1d and the
GDN update are actually sequential. This route batches everything else and steps
only the recurrence row by row, using the exact single-token kernels already
qualified by speculative verification.

**Do NOT use** `ATLAS_QWEN4_SSM_PREFILL_BATCH=1` or `ATLAS_QWEN4_SSM_PREFILL_FP32=1`
(garbage output and empty output respectively, both at a 23-token prompt), or
`ATLAS_QWEN4_SSM_PREFILL_TILE>=2` (diverges), or `ATLAS_UNIFIED_MOE_LAYOUT=1`
(slower prefill AND slower decode, measured twice).

**Status: opt-in, not default.** Output differs from the per-token route — but
that route is itself NONDETERMINISTIC at temperature 0 (2 distinct outputs in 3
reps), while row-wise is deterministic (1 in 3). Before promoting to default it
needs a higher-rep determinism run and an accuracy check against a reference
implementation rather than against the unstable per-token path.

After this fix prefill is MoE-bound: `moe_ffn` 709 ms of ~1,075 ms (66%).
