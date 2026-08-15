#!/bin/bash
# Atlas Spark — SuperGemma4-26B-abl (Gemma-4-26B-A4B-it Uncensored, NVFP4 MoE).
#
# THE stories+coding daily driver (validated 2026-06-11, long-form greedy):
#   counting 45.3 / coding 48.1 / prose 44.6 tok/s — NO speculation.
# Uniform across workloads because the A4B MoE reads only the active
# expert set (~2-3 GB) per token: no draft-acceptance lottery. Beats the
# AEON vLLM container on prose 2x and every dense-27B Atlas stack on
# the real story/coding mix.
#
# Requires the universal binary (ATLAS_TARGET_MODEL='*' build) or one
# built with ATLAS_TARGET_MODEL=gemma-4-26b-a4b.
#
# Untested upside: stack DFlash via the local pilot drafter
# (AEON-7-supergemma4-26b-dflash-pilot) — but Atlas's sliding-window
# state rollback for Gemma-style targets is documented as deferred in
# verify_dflash_step.rs; validate with the greedy protocol before use.
set -euo pipefail

PORT=${PORT:-8891}

if pgrep -x spark >/dev/null 2>&1; then
  echo "[serve-supergemma4] killing prior spark serve..."
  pkill -9 -x spark || true
  sleep 4
fi
if ss -tnlp 2>/dev/null | grep -q ":${PORT} "; then
  echo "[serve-supergemma4] ERROR: port ${PORT} still bound" >&2
  exit 1
fi
FREE_GB=$(free -g | awk '/^Mem:/ {print $7}')
if [ "${FREE_GB:-0}" -lt 30 ]; then
  echo "[serve-supergemma4] ERROR: only ${FREE_GB} GB free, need ≥30 GB" >&2
  exit 1
fi
echo "[serve-supergemma4] preflight ok: ${FREE_GB} GB free"

exec /home/flocka/atlas/src/target/release/spark serve \
  --model-from-path /home/flocka/models/SuperGemma4-26B-abl-NVFP4 \
  --model-name supergemma4 \
  --port "${PORT}" \
  --kernel-target gemma-4-26b-a4b \
  --gpu-memory-utilization 0.65 \
  --kv-cache-dtype fp8 \
  --max-seq-len 8192 \
  --max-batch-size 1 \
  --max-num-seqs 1
