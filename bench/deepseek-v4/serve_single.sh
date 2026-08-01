#!/usr/bin/env bash
# Serve DeepSeek-V4-Flash on ONE GB10 with Atlas (single-node, EP=1).
#
# Why single-node works here: the stock NVFP4 / FP8 checkpoints are 155-179 GB
# and need EP=2 across two GB10s. A single GB10 has ~119 GiB usable, so the
# only way onto one node is a checkpoint that fits. This script defaults to a
# REAP expert-pruned FP8 checkpoint (model_type=deepseek_v4, 144/160 of 256
# experts, ue8m0 block-scaled FP8 — a format Atlas's deepseek_v4 loader reads).
#
# Atlas runs EP=1 by default (the MoE all-reduce is a no-op with one rank), so
# there is no --ep flag to set. What matters is kv-cache-dtype=fp8 (BF16 KV
# gives garbage on this checkpoint) and a gpu-memory-utilization high enough to
# hold weights + KV + scratch.
#
#   MODEL_DIR=/path/to/checkpoint bash bench/deepseek-v4/serve_single.sh
#
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"

export PATH="/usr/local/cuda/bin:$PATH"

# REQUIRED for single-node E8M0 (MXFP4) routed experts. DeepSeek-V4's native
# E8M0 experts only have a *transposed* prefill GEMM (moe_..._t_k64_e8m0); the
# non-transposed fallback misreads the [N,K/32] E8M0 scales as NVFP4 [N,K/16]
# and panics. Building the transposed tables the normal way needs cost_full GB
# of headroom, which a single-node 87 GB checkpoint (~22 GB free) does not have.
# Unified layout transposes phase-by-phase, freeing the untransposed copies as
# it goes, so it fits the tight budget. See factory/m2_setup.rs.
export ATLAS_UNIFIED_MOE_LAYOUT="${ATLAS_UNIFIED_MOE_LAYOUT:-1}"

# CUDA-graph capture is clean on the V4 decode path as of the diag-probe fix
# (the layer-0 diag_norm probes synchronized the stream mid-capture and raised
# CUDA 901; they are opt-in now). Capture is worth ~6% here — decode is
# bandwidth-bound, not launch-bound — but it is free, so leave it on. Set
# ATLAS_DEBUG_NO_GRAPH=1 to force eager for diagnostics.
export ATLAS_DEBUG_NO_GRAPH="${ATLAS_DEBUG_NO_GRAPH:-0}"

# FP8 lm-head: the 129280x4096 BF16 vocab projection is 1.06 GB/token — the
# single largest per-token weight read. FP8 halves it. This is the config the
# --lm-head-dtype help text screams about (it collapsed Qwen3.6-35B-A3B), so it
# was gated PER-MODEL on 2026-08-01: same-session A/B on DeepSeek-V4-Flash-162B
# REAP measured bf16 17.2 -> fp8 18.0 tok/s with NO regression (longgen_gate
# fp8 2/4 vs bf16 1/4 — the shared failures reproduce bit-identically at BF16,
# i.e. they are the model, not the precision; GSM8K 12/12; coherence 4/4).
# Set LM_HEAD_DTYPE=bf16 to fall back to the safe default.
LM_HEAD_DTYPE="${LM_HEAD_DTYPE:-fp8}"

MODEL_DIR="${MODEL_DIR:-/home/flocka/models/DeepSeek-V4-Flash-162B}"
BIN="${DS4_BIN:-$REPO/target/release/spark}"
PORT="${PORT:-8899}"
HOST="${HOST:-127.0.0.1}"
KV_DTYPE="${KV_DTYPE:-fp8}"                       # fp8 REQUIRED for coherence
GPU_MEM="${GPU_MEM:-0.94}"                        # 0.94*119 ~= 112 GB budget
MAX_SEQ="${MAX_SEQ:-16384}"
MAX_BATCH="${MAX_BATCH:-1}"
LOG="${LOG:-$REPO/serve-deepseek-single.log}"

[ -x "$BIN" ] || { echo "FATAL: spark binary not found/executable at $BIN (build first: bench/laguna/build_cutlass.sh with ATLAS_TARGET_MODEL=deepseek-v4-flash)"; exit 3; }
[ -f "$MODEL_DIR/config.json" ] || { echo "FATAL: no config.json under MODEL_DIR=$MODEL_DIR"; exit 3; }

echo "serve: $BIN"
echo "model: $MODEL_DIR"
echo "port : $HOST:$PORT   kv=$KV_DTYPE  lm_head=$LM_HEAD_DTYPE  gpu_mem=$GPU_MEM  max_seq=$MAX_SEQ  batch=$MAX_BATCH"
echo "log  : $LOG"

# EXTRA_ARGS is how perf variants are driven (--speculative, --lm-head-dtype
# fp8, --num-drafts N, ...) without forking this script per experiment.
read -r -a EXTRA <<<"${EXTRA_ARGS:-}"
[ ${#EXTRA[@]} -gt 0 ] && echo "extra: ${EXTRA[*]}"

exec "$BIN" serve "$MODEL_DIR" \
  --model-from-path "$MODEL_DIR" \
  --host "$HOST" \
  --port "$PORT" \
  --kv-cache-dtype "$KV_DTYPE" \
  --lm-head-dtype "$LM_HEAD_DTYPE" \
  --gpu-memory-utilization "$GPU_MEM" \
  --max-seq-len "$MAX_SEQ" \
  --max-batch-size "$MAX_BATCH" \
  "${EXTRA[@]}" \
  2>&1 | tee "$LOG"
