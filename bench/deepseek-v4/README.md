# DeepSeek-V4-Flash on a single GB10 with Atlas (NVFP4/FP8)

This is the single-node bring-up and reproduction harness for
DeepSeek-V4-Flash on the `ds4-flash` Apathy Atlas branch.

## The single-node problem (why this harness exists)

DeepSeek-V4-Flash is a 284B-param MoE (43 layers, MLA, 256 experts / top-6 / 1
shared, mHC hyper-connections, a compressor + indexer sparse-attention stack).
The stock Atlas-loadable checkpoints are:

| checkpoint | format | size | fits one GB10 (119 GiB)? |
|---|---|---|---|
| `nvidia/DeepSeek-V4-Flash-NVFP4` | NVFP4 (+MTP) | 168 GB | no — needs EP=2 |
| `RedHatAI/DeepSeek-V4-Flash-NVFP4-FP8` | NVFP4+FP8 | 164 GB | no — needs EP=2 |
| `Intel/…-W4A16-AutoRound` | int4 | 156 GB | no |

Full-precision-expert quants are all 4-bit and land at 155–179 GB → two nodes.
The reference `ds4-on-spark` engine only fits one Spark because its GGUF uses
**2-bit** experts (~88 GB) — a format Atlas does not load.

To fit one GB10 with an **Atlas-loadable** checkpoint, the lever is fewer
experts, not fewer bits:

| checkpoint | experts | format | size | fits? |
|---|---|---|---|---|
| `0xSero/DeepSeek-V4-Flash-162B` | 144/256 | FP8 block (ue8m0) | **93.7 GB** | yes (comfortable) |
| `0xSero/DeepSeek-V4-Flash-0731-REAP` | 160/256 | FP8 block (ue8m0) | 107.8 GB | yes (tight) |

Both are `model_type: deepseek_v4`, 43 layers, `compress_ratios` present, FP8
block-scaled with UE8M0 scales — a format the Atlas `deepseek_v4` loader already
handles (FP8-block → NVFP4 transcode / native E8M0). They are **REAP** (Router-
weighted Expert Activation Pruning, arXiv:2510.13999) — redundant experts
dropped, top-6 routing and everything else preserved.

**Caveat, stated plainly:** REAP is a *smaller model*, so its quality is not
identical to stock DeepSeek-V4-Flash. `quality_probe.py` measures the actual gap
against the ds4 reference gates (GSM8K ~97.5%, coherence on Paris/counting/
haiku/math) so the tradeoff is a number, not a guess. For exact stock parity on
one node you would need a full-256-expert sub-4-bit Atlas quant (not yet built)
or EP=2 across two GB10s.

## Run it

```bash
# Pick public, machine-local locations once. No repository script requires
# these exact paths.
export CUTLASS_HOME=/path/to/cutlass
export MODEL_DIR=/path/to/DeepSeek-V4-Flash-162B

# 1. Build the deepseek-v4-flash NVFP4 engine (once):
ATLAS_TARGET_MODEL=deepseek-v4-flash ATLAS_TARGET_QUANT=nvfp4 \
ATLAS_TARGET_HW=gb10 ATLAS_CUDA_ARCH=sm_121f \
bash bench/laguna/build_cutlass.sh

# 2. Download a single-node checkpoint (~94 GB):
hf download 0xSero/DeepSeek-V4-Flash-162B \
    --local-dir "$MODEL_DIR"

# 3. Serve on one GB10 (EP=1, FP8 KV required):
bash bench/deepseek-v4/serve_single.sh

# 4. In another shell — prove coherence, then measure quality:
PORT=8899 bash bench/deepseek-v4/smoke.sh          # asserts "Paris"
python3 bench/deepseek-v4/quality_probe.py --port 8899
```

## Files

| file | purpose |
|---|---|
| `serve_single.sh` | single-node (EP=1) Atlas serve, FP8 KV, high gpu-mem-util |
| `smoke.sh` | first-token "Paris" sanity check (mirrors ds4-on-spark) |
| `quality_probe.py` | coherence gate + GSM8K-style accuracy vs the ds4 reference gate |

## mtp-bench suite (2026-08-02, wall tok/s, 512 tok, methodology of the
## Entrpi/ds4 published table — run `mtp_bench.py`)

| workload | Atlas plain | Atlas spec | ref plain (2-bit) | ref DSpark |
|---|---|---|---|---|
| stepwise_math | 18.6 | 18.6 | 20.1 | 34.5 |
| code_cpp | 19.5 | 19.1 | 20.4 | 30.0 |
| code_python | 19.7 | 19.4 | 20.5 | 26.4 |
| explain_concept | 19.8 | 19.4 | 20.5 | 25.2 |
| creative_short | 19.8 | 19.1 | 20.6 | 19.9 |
| **suite mean** | **17.7** | **17.4** | 20.1 | 27.7 |

Server-side decode (excl. prefill/HTTP): plain **21.0**, speculative **20.0**
tok/s. Wall numbers carry our ~680ms TTFT (reference: 340ms) — at 512 tokens
that costs ~2 wall tok/s; prefill/TTFT is an unmined lever. The reference's
DSpark column rides a separately-trained 3-layer drafter with 5-token blocks
(tok/step up to 4.0) — not reproducible with the checkpoint's 1-step MTP
module; our K=2 ceiling is 2.0 tok/step at 55-80% accept.

## Measured (2026-08-01, single GB10, graphs on)

| config | decode tok/s | TTFT | quality |
|---|---|---|---|
| bf16 lm-head | 17.2 | ~680ms | GSM8K 11/12, longgen 1/4 |
| **fp8 lm-head (default)** | **18.0** | ~680ms | GSM8K 12/12, longgen 2/4 (no regression) |

The longgen failures reproduce bit-identically at BF16 — they are REAP-model
instruction-following quirks, not precision damage. Gate any further precision
cut with `longgen_gate.py --baseline longgen_baseline_bf16.json` (regressions
vs the recorded full-precision run), not the absolute verdict.

### MTP speculative decode (`EXTRA_ARGS="--speculative"` + the two env gates)

| stage (2026-08-01) | accept | spec tok/s |
|---|---|---|
| as found (draft body on the generic BF16 MLA path) | 19–36% | ~7 |
| + draft body/cache on the FP8 MLA decode arms | 40–48% | 12.4 |
| + drafter prompt prefill & per-accept context feed | 49–67% | 13.3 |
| + draft argmaxes the target's FP8 head | **68–71%** | **17.3** (warm) |
| + MROW=2 dedup'd multi-row verify MoE (2026-08-02) | 63% | **21.0** |

Run with `ATLAS_MTP_DRAFTER_PREFILL=1 ATLAS_MTP_CATCHUP=1`; quality gates
PASS at every stage (GSM8K 11–12/12, longgen 0 regressions vs the BF16
baseline).

Open item (1) — amortize the verify weight reads — is **done**. The MROW=2
`_m2v2s4` kernels dedup the experts the two candidate rows share, and the
K=2 batched verify FFN is now default-on (gated on
`k2_verify_ffn_is_batched`; `ATLAS_MSHC_FFN_K2=0` opts out). 19.8 → 21.0
server tok/s. Note the earlier `batch2_t` batching was a *loss* (17.0):
its 1.33× amortization could not cover a ~2× per-byte deficit against the
split-K decode GEMV. Batching only pays on top of the fast kernel shape.

### `--num-drafts 2` (K=3) is a measured net loss — don't re-try it blind

Open item (2) is resolved as **won't-fix**. The K=3 verify no longer wedges
on graph capture, so it can be measured directly, and it loses:

| config | tok/s (server) | mean accepted | tok/step |
|---|---|---|---|
| `--num-drafts 1` (K=2, default) | **21.0** | 0.63 | 1.63 |
| `--num-drafts 2` (K=3) | 16.5 | 0.86 | 1.86 |

The reason is the *second* draft, not the machinery. Draft-1 is accepted
63% of the time, but draft-2 given draft-1 only 23/63 ≈ **37%** — the MTP
head is a 1-step predictor being applied recurrently to its own hidden
state, so error compounds immediately. K=3 buys +14% tok/step for a step
that costs ~45% more (one extra draft forward, plus a 3-row verify that
falls off the MROW=2 path onto the per-token loop).

An MROW=3 kernel does **not** close this. The 3-row MoE dedup would recover
roughly 8 ms of a ~35 ms/step penalty → ~18 tok/s, still under 21.0. The
ceiling here is draft *quality* at depth, not verify bandwidth. Spend
effort on accept rate (drafter FP8-KV scale calibration) or on the base
forward, not on deeper MTP chains.

## Notes that will save you time

- **`--kv-cache-dtype fp8` is mandatory.** BF16 KV on this checkpoint produces
  garbage (the MLA decode kernel's V=K rope reconstruction + attention sink are
  implemented on the FP8 path only).
- **EP=1 is the default** — there is no flag to set. The MoE all-reduce is a
  no-op with one rank. Do *not* borrow `scripts/start-deepseek-ep2.sh`; that is
  the two-node path.
- **gpu-memory-utilization** must clear weights + KV + scratch. 0.94 on a 119 GiB
  device ≈ 112 GB budget; the 162B checkpoint (93.7 GB) leaves comfortable KV
  headroom, the 0731-REAP (107.8 GB) is tight — drop `--max-seq-len` if it OOMs.
