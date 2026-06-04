#!/bin/bash
# Atlas Spark — AEON-7 Qwen3.6-27B with MTP K=3 speculative decode.
#
# CRASH-SAFE WRAPPER (2026-05-16): preflight kills stale workers, waits for
# GPU memory to settle, caps utilization, and refuses to start if anything
# else is already binding the port. Repeated launches without these guards
# OOM'd the GB10 unified memory pool and locked the host (load avg >17,
# 5-day uptime preserved by recovery, not luck).
set -euo pipefail

PORT=${PORT:-8889}

# 1. Kill any prior spark serve. Tolerate no-match.
if pgrep -x spark >/dev/null 2>&1; then
  echo "[serve-aeon-27b] killing prior spark serve..."
  pkill -9 -x spark || true
  # Wait for GPU memory to actually release (not just process exit).
  sleep 4
fi

# 2. Refuse to start if port is bound by something else.
if ss -tnlp 2>/dev/null | grep -q ":${PORT} "; then
  echo "[serve-aeon-27b] ERROR: port ${PORT} still bound. lsof -i :${PORT}" >&2
  ss -tnlp 2>/dev/null | grep ":${PORT} " >&2
  exit 1
fi

# 3. Memory preflight: GB10 has 119 GB unified. Need ≥40 GB free to load a
# 27B BF16 + NVFP4 quantized weights + ~30 GB KV pool + scratch arenas.
FREE_GB=$(free -g | awk '/^Mem:/ {print $7}')
if [ "${FREE_GB:-0}" -lt 40 ]; then
  echo "[serve-aeon-27b] ERROR: only ${FREE_GB} GB free, need ≥40 GB" >&2
  free -h >&2
  exit 1
fi
echo "[serve-aeon-27b] preflight ok: ${FREE_GB} GB free, port ${PORT} clear"

# WORKING CONFIG (2026-06-04 re-bench): 30 tok/s avg at N=1, 28 tok/s at
# N=8 concurrent decode. Code generation 28 tok/s, counting 24-30 tok/s.
# +30% vs raw (no-flag) baseline of 22.94 tok/s, matching the published
# AEON container number (30.9 tok/s).
#
# Winning stack composition:
#   ATLAS_LM_HEAD_BATCH3=1        : +16% (K=3 LM-head triple GEMV — single
#                                   kernel replaces M=3 w4a16_gemm fallback
#                                   that wastes 95% of M_TILE=64)
#   ATLAS_SSM_OUT_BATCH3=1        : +12% on top (K=3 SSM out_proj GEMV
#                                   batched for K=3 verify, avoids same
#                                   M=3 waste)
#   ATLAS_SSM_MULTI_SEQ_KERNEL=1  : +0% at N=1, no regression at N≥2.
#                                   Added 2026-06-03 (commits b9e7c28 +
#                                   b57339d). FP32-input multi-seq
#                                   conv1d_update_l2norm + gdn_decode +
#                                   batched QKVZ/BA+gates/RMS-norm/out_proj
#                                   for multi-seq concurrent decode.
#                                   Critically fixes the silent garbage
#                                   that b9e7c28 alone produced at N≥2.
#
# Flags evaluated and DROPPED:
#   ATLAS_TC_NVFP4_M16=1          : -42% regression at N=8
#   ATLAS_TC_NVFP4_K3=1           : -18% regression (M=3 MMA-tile waste)
#   ATLAS_FUSE_SSM_QKVZ=1         : noise (+0.3%)
#   ATLAS_MOE_K3_FUSED_GATE_UP=1  : noise (within ±1%)
#   ATLAS_SSM_BA_BATCHED=1        : noise (BA is already 14μs/layer)
#   ATLAS_UNIFIED_MOE_LAYOUT=1    : no effect (needs t-layout weights
#                                   not built; would also regress at small N
#                                   per helpers_b.rs:181 — warp-reduction
#                                   wins at decode)
#   --ngram-speculative           : -30% (K=2 ceiling lower than K=3 MTP)
#   --num-drafts 4 (K=5)          : -84% (drafter not trained for K=5)
#
# Why these flags:
#   --kv-cache-dtype fp8         FP8 KV cache. turbo4 (FP4) CORRUPTS the
#                                K-batched verify path: structured prompts
#                                ("1, 2, ..., 10," -> "11, 9, 9, 9...",
#                                "Capital of France" -> "1.") degrade silently
#                                while narrative prompts look fine. BF16 KV
#                                also works (~16.65 tok/s) but FP8 is faster.
#   --speculative --num-drafts 2 K=3 graphed verify (verify_k3_step).
#   --gpu-memory-utilization 0.7 LOWERED from 0.85 — at 0.85 a repeated
#                                launch cycle with stale workers OOM'd the
#                                unified memory pool and crashed the host.
#                                0.70 leaves ~36 GB headroom for the OS,
#                                CUDA driver, and any background MCP/Python
#                                processes (mesh_mcp, reaper, claude).
# XS variant (20 GB) gives ~20 tok/s vs Full's 17 tok/s. Same 64-layer
# arch, smaller weights (different ModelOpt sweep), no quality loss
# observed on counting / fibonacci / narrative / factual prompts.
# Bumped gpu-memory-utilization 0.70 -> 0.85 because XS leaves more
# headroom (20 GB vs 27 GB weights); KV pool benefits from extra space.
#
# --enable-prefix-caching RE-ENABLED (2026-05-17): re-tested 3x with
# same prompt cold→warm→warm-warm; all three runs produced IDENTICAL
# correct output ("11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, ..."
# every time). The earlier "11, 12, 13, 14, *16, 18, 20*" corruption
# was a stale test-harness artifact (10-tok warmup hits + max_tokens=10
# bench truncation creating weird KV state), not prefix caching itself.
# On unique-prompt benches throughput is unchanged (~18 tok/s avg).
# On repeated prompts and shared prefixes (chat history, system
# prompts, RAG) we see Marconi SSM cache hits → ~22-25 tok/s with
# correct output, a clear win for any real workload that touches the
# same prompt prefix more than once.
export ATLAS_GDN_PREFILL_TUNED=1
export ATLAS_LM_HEAD_BATCH3=1
export ATLAS_SSM_OUT_BATCH3=1
export ATLAS_PREFILL_FFN_FAST=1
export ATLAS_FFN_M16_TRANSPOSED=1
# 2026-06-04: multi-seq batched conv1d + gdn + QKVZ + BA-split + RMS-norm
# + out_proj for concurrent decode (b9e7c28 + b57339d). FP32-input fix +
# 4 op-level launch-reductions per SSM layer per token. No regression at
# N=1, no regression at N=8 concurrent; correctness validated.
export ATLAS_SSM_MULTI_SEQ_KERNEL=1
exec /home/flocka/atlas-src/target/release/spark serve \
  --model-from-path /home/flocka/models/AEON-Q36-27B-XS \
  --model-name aeon-27b \
  --port "${PORT}" \
  --kernel-target qwen3.6-27b \
  --gpu-memory-utilization 0.85 \
  --kv-cache-dtype fp8 \
  --max-seq-len 16384 \
  --max-batch-size 8 \
  --max-num-seqs 8 \
  --enable-prefix-caching \
  --speculative \
  --num-drafts 2 \
  --mtp-quantization nvfp4 \
  --mtp-vocab 32000 \
  --ssm-cache-slots 16 \
  --ssm-checkpoint-interval 16 \
  --max-prefill-tokens 1024 \
  --max-thinking-budget 768 \
  --warmup-prompt /home/flocka/atlas-src/local/warmup.txt
