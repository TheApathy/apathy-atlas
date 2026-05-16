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
#   --speculative --num-drafts 2 K=3 graphed verify (verify_k3_step).
#   --gpu-memory-utilization 0.7 LOWERED from 0.85 — at 0.85 a repeated
#                                launch cycle with stale workers OOM'd the
#                                unified memory pool and crashed the host.
#                                0.70 leaves ~36 GB headroom for the OS,
#                                CUDA driver, and any background MCP/Python
#                                processes (mesh_mcp, reaper, claude).
exec /home/flocka/atlas-src/target/release/spark serve \
  --model-from-path /home/flocka/models/AEON-Q36-27B-Full \
  --model-name aeon-27b \
  --port "${PORT}" \
  --kernel-target qwen3.6-27b \
  --gpu-memory-utilization 0.70 \
  --kv-cache-dtype fp8 \
  --max-seq-len 8192 \
  --enable-prefix-caching \
  --speculative \
  --num-drafts 2 \
  --mtp-quantization nvfp4 \
  --mtp-vocab 100000 \
  --ssm-cache-slots 16 \
  --max-thinking-budget 768 \
  --warmup-prompt /home/flocka/atlas-src/local/warmup.txt
