# DeepSeek-V4-Flash on a single GB10 with Atlas (NVFP4/FP8)

This is the single-node bring-up + reproduction harness for DeepSeek-V4-Flash on
the champion `laguna` Atlas build. It is the DeepSeek analog of `bench/laguna/`.

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
# 1. Build the deepseek-v4-flash NVFP4 engine (once):
CUTLASS_HOME=/home/flocka/vllm-build/.deps/cutlass-src \
ATLAS_TARGET_MODEL=deepseek-v4-flash ATLAS_TARGET_QUANT=nvfp4 \
ATLAS_TARGET_HW=gb10 ATLAS_CUDA_ARCH=sm_121f \
bash bench/laguna/build_cutlass.sh

# 2. Download a single-node checkpoint (~94 GB):
hf download 0xSero/DeepSeek-V4-Flash-162B \
    --local-dir /home/flocka/models/DeepSeek-V4-Flash-162B

# 3. Serve on one GB10 (EP=1, FP8 KV required):
MODEL_DIR=/home/flocka/models/DeepSeek-V4-Flash-162B \
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

## Measured (2026-08-01, single GB10, graphs on)

| config | decode tok/s | TTFT | quality |
|---|---|---|---|
| bf16 lm-head | 17.2 | ~680ms | GSM8K 11/12, longgen 1/4 |
| **fp8 lm-head (default)** | **18.0** | ~680ms | GSM8K 12/12, longgen 2/4 (no regression) |

The longgen failures reproduce bit-identically at BF16 — they are REAP-model
instruction-following quirks, not precision damage. Gate any further precision
cut with `longgen_gate.py --baseline longgen_baseline_bf16.json` (regressions
vs the recorded full-precision run), not the absolute verdict. MTP speculative
decode currently LOSES (~7 tok/s): the draft head only reaches 19–36% accept
on this checkpoint — open item.

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
