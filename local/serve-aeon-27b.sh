#!/bin/bash
# Atlas Spark — AEON-7 Qwen3.6-27B with MTP K=3 speculative decode.
#
# WORKING CONFIG (2026-05-16): 17-20 tok/s avg with CORRECT output.
# Counting prompts hit 20.28 tok/s; narrative + code 15-17 tok/s.
#
# Why these flags:
#   --kv-cache-dtype fp8         FP8 KV cache. turbo4 (FP4) CORRUPTS the
#                                K-batched verify path: structured prompts
#                                ("1, 2, ..., 10," -> "11, 9, 9, 9...",
#                                "Capital of France" -> "1.") degrade silently
#                                while narrative prompts look fine. BF16 KV
#                                also works (~16.65 tok/s) but FP8 is faster.
#   --speculative --num-drafts 2 K=3 graphed verify (verify_k3_step). K=2
#                                works too (~16.15 tok/s) and K=4 works
#                                (~10 tok/s, slower). K=5+ routes through
#                                step_verify_dflash which is broken (see
#                                serve-aeon-27b-dflash.sh comments).
#   no --dflash                  The DFlash drafter integration produces
#                                ~0.6% accept rate AND the K=γ=17 verify
#                                path corrupts target output (separate bug
#                                from the FP4-KV one). MTP head gives ~55%
#                                accept rate on K=3 and works correctly.
exec /path/to/atlas-src/target/release/spark serve \
  --model-from-path /path/to/models/AEON-Q36-27B-Full \
  --model-name aeon-27b \
  --port 8889 \
  --kernel-target qwen3.6-27b \
  --gpu-memory-utilization 0.85 \
  --kv-cache-dtype fp8 \
  --max-seq-len 16384 \
  --enable-prefix-caching \
  --speculative \
  --num-drafts 2 \
  --mtp-quantization nvfp4 \
  --mtp-vocab 100000 \
  --ssm-cache-slots 16 \
  --max-thinking-budget 768 \
  --warmup-prompt /path/to/atlas-src/local/warmup.txt
