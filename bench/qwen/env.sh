#!/usr/bin/env bash
# Shared configuration + helpers for the Qwen3.6-27B champion decode harness.
#
#   source "$(dirname "${BASH_SOURCE[0]:-$0}")/env.sh"
#
# Companion to bench/laguna/env.sh and deliberately the same shape, but this is
# a DIFFERENT stack, not a re-parameterization of that one. The two disagree on
# gamma, on KV dtype, on which log lines carry the accept counter, and on which
# gates are safe. Copying a setting across without reading the comment on it is
# the specific mistake this file is written to prevent.
#
# Everything below is overridable from the environment, so the common case is
#
#   export QWEN_MODEL=/path/to/Qwen3.6-27B-target
#   export QWEN_DRAFT=/path/to/Qwen3.6-27B-DFlash-drafter
#   export QWEN_BIN=/path/to/spark
#
# and then any script here runs unmodified. See README.md.

set -uo pipefail

# --- repo layout --------------------------------------------------------------
# BENCH_DIR is this directory; REPO is the checkout above it. Derived, never
# hardcoded, so a clone anywhere works and no $HOME reaches a committed file.
# ${BASH_SOURCE[0]:-$0} rather than ${BASH_SOURCE[0]}: `set -u` is on, and under
# a shell that does not define BASH_SOURCE the bare form aborts on line 1 with an
# error about the wrong thing entirely.
BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO="${QWEN_REPO:-$(cd "$BENCH_DIR/../.." && pwd)}"

# Run artifacts (serve logs, eval logs, result JSON). Gitignored: they embed
# absolute paths and device topology and are regenerated per run.
OUT_ROOT="${QWEN_OUT:-$BENCH_DIR/ab}"

# --- what to serve ------------------------------------------------------------
BIN="${QWEN_BIN:-$REPO/target/release/spark}"

# Target and drafter checkpoints. There is no default that could be right on
# someone else's machine, so the scripts fail loudly rather than guess.
MODEL="${QWEN_MODEL:-}"
DRAFT="${QWEN_DRAFT:-}"

# The served model name. Requests must use this string; it is also what the
# harness sends in the JSON body.
MODEL_NAME="${QWEN_MODEL_NAME:-aeon-27b-dflash}"

# --- where to serve -----------------------------------------------------------
HOST="${QWEN_HOST:-127.0.0.1}"
PORT="${QWEN_PORT:-8890}"
BASE_URL="http://$HOST:$PORT"

# The GPU is a single exclusive resource and so is the port. Every script that
# launches a serve takes this lock first; see qwen_lock below for why fd 9.
LOCK="${QWEN_LOCK:-/tmp/qwen-port$PORT.lock}"

# --- serve geometry -----------------------------------------------------------
# gamma=16 is NOT a tuning knob here, it is a correctness constraint.
#
# gamma=16 plus the +1 prefix gives K=17 verify tokens, and K=17 is the only
# width that routes through the fused `gdn_wy17_k` SSM kernel, which saves all
# 17 intermediates. Every K in 5..16 falls through to the sequential per-token
# SSM path, which has a NaN bug at positions K-3..K-1 and corrupts the SSM
# rollback. The symptom is not a crash: the target emits a correct first token
# followed by `!!!!!`. It was localized against a 64-layer HF reference --
# layer-0 positions 13..15 are NaN at cap 15 and all of 0..16 are finite at 16.
#
# So GAMMA and DRAFT_CAP must move together and must both be 16. qwen_require_model
# asserts that rather than trusting whoever edits this next.
GAMMA="${QWEN_GAMMA:-16}"
DRAFT_CAP="${QWEN_DRAFT_CAP:-$GAMMA}"

MAX_SEQ_LEN="${QWEN_MAX_SEQ_LEN:-8192}"
# 0.65, not the 0.80 the Laguna harness uses. gamma=16 inflates the SSM-MTP pool
# plus KV by roughly 24 GB, and the higher utilization crashed the host on
# repeated launches. This is the measured safe ceiling for this stack.
GPU_UTIL="${QWEN_GPU_UTIL:-0.65}"
# Speculative decode is disabled whenever more than one sequence is active, so
# batch 1 is what the single-stream numbers in README.md describe.
MAX_BATCH_SIZE="${QWEN_MAX_BATCH_SIZE:-1}"

KV_DTYPE="${QWEN_KV_DTYPE:-fp8}"
# The first 5 layers keep a high-precision KV cache. fp8 everywhere measurably
# degrades this target; 5 is the shipped setting.
KV_HP_LAYERS="${QWEN_KV_HP_LAYERS:-5}"
MTP_VOCAB="${QWEN_MTP_VOCAB:-96000}"
KERNEL_TARGET="${QWEN_KERNEL_TARGET:-qwen3.6-27b}"
THINKING_BUDGET="${QWEN_THINKING_BUDGET:-768}"

# Phase-level timing breakdown. DEFAULT OFF, deliberately: the champion serve
# does not set it, and this harness reproduces the champion config rather than a
# near-miss of it. Turning it on adds a per-step log line -- see benchenv.py for
# why that line is excluded from the accept histogram by anchor rather than by
# hoping it is absent.
STEP_TIMING="${QWEN_STEP_TIMING:-0}"

# =============================================================================
# helpers
# =============================================================================

# Fail early and legibly when a checkpoint path is missing or wrong, rather than
# 90 seconds later inside a serve log nobody is tailing.
qwen_require_model() {
  local ok=1
  for var in MODEL DRAFT; do
    local val="${!var}"
    if [ -z "$val" ]; then
      echo "FATAL: \$QWEN_${var} is unset. Point it at the downloaded snapshot directory." >&2
      ok=0
    elif [ ! -f "$val/config.json" ]; then
      echo "FATAL: \$QWEN_${var}=$val has no config.json -- not a checkpoint snapshot." >&2
      ok=0
    fi
  done
  [ ! -x "$BIN" ] && { echo "FATAL: \$QWEN_BIN=$BIN is not executable. Run bench/laguna/build_cutlass.sh, or build with the kernel target for this model." >&2; ok=0; }

  # The K=17 constraint, asserted rather than commented. A mismatch here does
  # not fail loudly at runtime -- it produces `token!!!!!` output that reads as
  # a drafter-quality problem, so it must be caught before the serve starts.
  if [ "$DRAFT_CAP" != "$GAMMA" ]; then
    echo "FATAL: DRAFT_CAP=$DRAFT_CAP != GAMMA=$GAMMA." >&2
    echo "  K = gamma+1 must be 17 to hit the fused gdn_wy17_k SSM kernel. Any other" >&2
    echo "  width takes the sequential path, which NaNs at positions K-3..K-1 and" >&2
    echo "  corrupts SSM rollback (symptom: correct first token then '!!!!!')." >&2
    ok=0
  fi
  if [ "$GAMMA" != 16 ]; then
    echo "FATAL: GAMMA=$GAMMA. Only 16 is a supported width for this stack (see above)." >&2
    ok=0
  fi
  [ "$ok" = 1 ] || exit 2
}

# Serialize on the port. fd 9 is held for the life of the script and INHERITED
# by the serve we spawn, which is deliberate: an orphaned serve keeps the lock,
# so the next arm refuses to start instead of racing a process it cannot see.
# That inheritance is also why a stale lock means "kill the serve", not "delete
# the lock file".
qwen_lock() {
  exec 9>"$LOCK"
  flock -n 9 || {
    echo "FATAL: another arm holds $LOCK" >&2
    ps -eo pid,etime,cmd | grep -E "port $PORT" | grep -v grep | cut -c1-110 >&2
    echo "(children inherit fd 9 -- an orphaned serve keeps the lock; kill it first)" >&2
    exit 5
  }
}

# Kill serves by PID matched on the port argument. Never `pkill -f spark`: the
# pattern matches this script's own command line and kills the shell running it.
qwen_kill_serves() {
  for p in $(ps -eo pid,cmd | grep "port $PORT" | grep -v grep | awk '{print $1}'); do
    kill "$p" 2>/dev/null
  done
  sleep "${1:-4}"
}

# The champion decode stack. Every gate arm starts from exactly this set and then
# applies its own overrides, so "baseline" means one thing across scripts.
#
# These are transcribed from the launcher that produced the published numbers.
# The reasoning for each is in README.md; the two that are set to 0 are set to 0
# ON PURPOSE and are not dead entries to be tidied away.
qwen_champion_env() {
  # gamma=16 verify. DRAFT_CAP must equal gamma -- see qwen_require_model.
  export ATLAS_DFLASH_DRAFT_CAP="$DRAFT_CAP"
  export ATLAS_LM_HEAD_T=1
  # 4096 matches the drafter's trained sliding-window attention span. Capping the
  # drafter context at gamma instead cripples the accept rate.
  export ATLAS_DFLASH_CTX_WINDOW=4096
  # nvfp4 is required, not an optimization. With bf16 every drafter projection
  # runs a dense GEMM over the full context, and propose time scales to ~1.75s
  # per step at sequence length 800 -- coding prompts simply time out. nvfp4
  # routes those projections through the same w4a16 kernel the target uses.
  # Committed greedy tokens are unaffected either way: the target's logits are
  # the source of truth, so drafter precision moves acceptance, never output.
  export ATLAS_DFLASH_QUANT=nvfp4

  # Batched K=gamma FFN. Without these the K=17 verify spends the large majority
  # of the step in per-token FFN loops, re-reading the FFN weights 17x per step.
  export ATLAS_FFN_KGAMMA_M16=1
  export ATLAS_FFN_FUSED_GATEUP=1
  export ATLAS_FFN_M16_TRANSPOSED=1
  export ATLAS_FFN_KGAMMA_M128=1
  export ATLAS_DFLASH_FFN_KGAMMA=1
  export ATLAS_DFLASH_ATTN_KGAMMA=1
  export ATLAS_FFN_DOWN_SPLITK=4
  export ATLAS_ATTN_QKV_BATCHED=1
  export ATLAS_ATTN_QKV_SPLITK=4

  # Split-K over the two big SSM projections. out_proj [M=17..32, N=5120,
  # K=6144] ran at 28% of its DRAM roofline; split-K x4 takes it to 84%
  # (228 GB/s), a 2.6x kernel win worth ~6.7ms/step across 48 SSM layers.
  # qkvz [N=12288, K=5120] goes 220.5us -> 167.3us, ~2.2ms/step. Both keep
  # FP32 partials and reduce, so both are bit-exact by construction -- the
  # same pattern as the already-shipped ffn_down split-K.
  export ATLAS_SSM_OUT_SPLITK=4
  export ATLAS_SSM_QKVZ_SPLITK=4
  # Batches the 17 per-token BA GEMVs into one `dense_gemv_bf16_batchn` launch
  # per SSM layer: 144 launches -> 48. Worth ~1% on counting. The batchn kernel
  # is bit-exact; the `dense_gemm` variant of the same batching is NOT, so this
  # is a specific kernel choice and not an interchangeable one.
  export ATLAS_SSM_BA_BATCH=1

  # Attention and LM-head fast paths from the same kernel wave. Individually
  # small, collectively part of the configuration the published coding number
  # was measured on -- they are listed separately rather than folded into one
  # gate because each is independently revertible.
  export ATLAS_FA2_KGAMMA=1
  export ATLAS_LM_HEAD_BATCH3=1
  export ATLAS_NVFP4_GATE_UP_M128=1
  export ATLAS_PAGED_DECODE_SPLITK=1
  export ATLAS_FLASH_ATTN_KGAMMA_SPLITK=1

  # ---- MEASURED AND NOT SHIPPED ----------------------------------------------
  # Async propose overlaps the drafter forward with the step-tail CPU work. It
  # is lossless -- the drafter only proposes, verify still commits the target's
  # greedy token -- so it was rejected on speed alone: the drafter's own GPU
  # time (~35ms) is about equal to the sync propose it removes (29ms), so the
  # next step's collect blocks for nearly as long as was saved, and the step
  # interval grows (140ms -> 166ms) even though the step itself shrinks.
  # These are explicit zeros, not omissions: leaving them unset would make the
  # published config depend on whatever the binary's default happens to be.
  export ATLAS_DFLASH_ASYNC=0
  export ATLAS_DFLASH_SAM_ASYNC=0
  export ATLAS_DFLASH_FUSED=0

  # The gamma=16 correctness fix: the tree WY path is wrong at this width.
  export ATLAS_DISABLE_TREE_WY=1
  export ATLAS_WY17_LAZY=8
  export ATLAS_WY17_LAZY_COMMIT=1

  export ATLAS_DFLASH_NOISE_ONLY=1
  export ATLAS_DFLASH_SAM=1
  # Speculate through the thinking block instead of falling back to plain decode
  # for the majority of reasoning tokens. NOT byte-lossless -- batched verify and
  # sequential decode differ on the SSM layers, so this was gated on quality
  # rather than on hashes. See README.md for the pass@1 comparison.
  export ATLAS_THINK_SPEC=1

  # ---- DO NOT ENABLE ---------------------------------------------------------
  # The M_TILE=16 NVFP4 attention path corrupts at K=17. With it on, verify's
  # deep-slot argmax repeats earlier digits, greedy determinism breaks across
  # requests, and acceptance collapses from ~15.6/16 to ~1.5/16 -- which reads
  # exactly like a drafter-quality regression and sends you after the wrong
  # thing. Flag-isolated by controlled greedy A/B; every other combination of
  # the KGAMMA/TRANSPOSED gates above is token-exact. It bought ~20ms of verify.
  export ATLAS_TC_NVFP4_M16=0
  export ATLAS_TC_NVFP4_M16_MS_ATTN=0

  if [ "$STEP_TIMING" = 1 ]; then
    export ATLAS_DFLASH_STEP_TIMING=1
  else
    unset ATLAS_DFLASH_STEP_TIMING
  fi
}

# Launch the serve detached. $1 = log path, remaining args = extra serve flags.
qwen_serve() {
  local log="$1"; shift
  cd "$REPO" || exit 3
  setsid "$BIN" serve \
    --model-from-path "$MODEL" \
    --model-name "$MODEL_NAME" \
    --port "$PORT" \
    --kernel-target "$KERNEL_TARGET" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --kv-cache-dtype "$KV_DTYPE" \
    --kv-high-precision-layers "$KV_HP_LAYERS" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --max-batch-size "$MAX_BATCH_SIZE" \
    --max-num-seqs "$MAX_BATCH_SIZE" \
    --dflash \
    --draft-model "$DRAFT" \
    --dflash-gamma "$GAMMA" \
    --mtp-vocab "$MTP_VOCAB" \
    --dflash-quantization "$ATLAS_DFLASH_QUANT" \
    --max-thinking-budget "$THINKING_BUDGET" \
    "$@" \
    > "$log" 2>&1 < /dev/null &
  disown 2>/dev/null || true
}

# Poll /health until ready, but abort the moment the process dies -- otherwise a
# serve that OOMs at load costs the full timeout before saying so.
# NOTE: /health answers `ready` on ANY serve bound to this port, including one
# started by someone else. Readiness proves the port is answering, not that it is
# answering with YOUR binary. qwen_lock is what makes it yours.
qwen_wait_ready() {
  local log="$1" tries="${2:-120}"
  for _ in $(seq 1 "$tries"); do
    curl -s -m 3 "$BASE_URL/health" 2>/dev/null | grep -q ready && return 0
    ps -eo cmd | grep -q "port $PORT" || {
      echo "FATAL: serve died during load" >&2
      grep -Ei "out of memory|illegal|CUDA_ERROR|panicked|Error: " "$log" | head -5 >&2
      exit 4
    }
    sleep 5
  done
  echo "FATAL: serve never became ready after $((tries * 5))s" >&2
  exit 4
}

# Apply `KEY=VALUE` overrides from a script's positional args. An EMPTY value
# means unset, which is how an arm turns a baseline gate back OFF.
qwen_apply_overrides() {
  for kv in "$@"; do
    local k="${kv%%=*}" v="${kv#*=}"
    if [ -z "$v" ]; then unset "$k"; else export "$k=$v"; fi
  done
}
