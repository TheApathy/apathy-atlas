#!/usr/bin/env bash
# Shared configuration + helpers for the Laguna decode harness.
#
#   source "$(dirname "${BASH_SOURCE[0]:-$0}")/env.sh"
#
# Every path, port and binary name in this harness lives HERE and nowhere else.
# That is not tidiness for its own sake: the scripts in this directory grew as
# one-off sweeps, each with its own copy of the same absolute paths, and copies
# drift. Two arms that disagree about which binary they launch produce a clean
# A/B table describing two different builds.
#
# Everything below is overridable from the environment, so the common case is
#
#   export LAGUNA_MODEL=/path/to/Laguna-S-2.1-NVFP4
#   export LAGUNA_DRAFT=/path/to/Laguna-S-2.1-DFlash-NVFP4
#   export LAGUNA_BIN=/path/to/spark
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
REPO="${LAGUNA_REPO:-$(cd "$BENCH_DIR/../.." && pwd)}"

# Run artifacts (serve logs, eval logs, result JSON). Gitignored: they embed
# absolute paths and device topology and are regenerated per run.
OUT_ROOT="${LAGUNA_OUT:-$BENCH_DIR/ab}"

# --- what to serve ------------------------------------------------------------
# The release binary. `cargo build --release` alone is NOT enough: it produces a
# spark that defaults to a different kernel target and cannot load Laguna. Use
# build_cutlass.sh, or set LAGUNA_BIN to a binary built the same way.
BIN="${LAGUNA_BIN:-$REPO/target/release/spark}"

# Target and drafter checkpoints. Both are gated on Hugging Face; download them
# with `hf download` and point these at the resulting snapshot directories.
# There is no default that could be right on someone else's machine, so the
# scripts fail loudly rather than guess.
MODEL="${LAGUNA_MODEL:-}"
DRAFT="${LAGUNA_DRAFT:-}"

# --- where to serve -----------------------------------------------------------
HOST="${LAGUNA_HOST:-127.0.0.1}"
PORT="${LAGUNA_PORT:-8890}"
BASE_URL="http://$HOST:$PORT"

# The GPU is a single exclusive resource and so is the port. Every script that
# launches a serve takes this lock first; see laguna_lock below for why fd 9.
LOCK="${LAGUNA_LOCK:-/tmp/laguna-port$PORT.lock}"

# tool-eval-bench, used by the quality gates. Installed separately.
EVAL_BIN="${LAGUNA_EVAL_BIN:-tool-eval-bench}"

# --- serve geometry -----------------------------------------------------------
# gamma=6 is the swept optimum, and both directions were measured: gamma=8 costs
# -6.7% and gamma=10 costs -38.1% (verify time is superlinear in the verify
# width, with a knee between K=9 and K=11), while gamma<=4 does not degrade
# gracefully toward serial decode -- it breaks the block size the drafter was
# trained for and lands ~40% BELOW it.
GAMMA="${LAGUNA_GAMMA:-6}"
MAX_SEQ_LEN="${LAGUNA_MAX_SEQ_LEN:-8192}"
GPU_UTIL="${LAGUNA_GPU_UTIL:-0.80}"
# --max-batch-size 1 is a measured choice, not a default. Speculative decode is
# disabled whenever more than one sequence is active, so concurrency 2-3 falls
# to ~0.6x single-stream, and batched serial decode only overtakes single-stream
# DFlash somewhere between concurrency 5 and 6.
MAX_BATCH_SIZE="${LAGUNA_MAX_BATCH_SIZE:-1}"

# =============================================================================
# helpers
# =============================================================================

# Fail early and legibly when a checkpoint path is missing or wrong, rather than
# 90 seconds later inside a serve log nobody is tailing.
laguna_require_model() {
  local ok=1
  for var in MODEL DRAFT; do
    local val="${!var}"
    if [ -z "$val" ]; then
      echo "FATAL: \$LAGUNA_${var} is unset. Point it at the downloaded snapshot directory." >&2
      ok=0
    elif [ ! -f "$val/config.json" ]; then
      echo "FATAL: \$LAGUNA_${var}=$val has no config.json -- not a checkpoint snapshot." >&2
      ok=0
    fi
  done
  [ ! -x "$BIN" ] && { echo "FATAL: \$LAGUNA_BIN=$BIN is not executable. Run build_cutlass.sh first." >&2; ok=0; }
  [ "$ok" = 1 ] || exit 2
}

# Serialize on the port. fd 9 is held for the life of the script and INHERITED
# by the serve we spawn, which is deliberate: an orphaned serve keeps the lock,
# so the next arm refuses to start instead of racing a process it cannot see.
# That inheritance is also why a stale lock means "kill the serve", not "delete
# the lock file".
laguna_lock() {
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
laguna_kill_serves() {
  for p in $(ps -eo pid,cmd | grep "port $PORT" | grep -v grep | awk '{print $1}'); do
    kill "$p" 2>/dev/null
  done
  sleep "${1:-4}"
}

# Launch the serve detached, with fd 9 closed in the child ONLY when the caller
# wants the lock released on exit; by default the child inherits it (see above).
# $1 = log path, remaining args = extra serve flags.
laguna_serve() {
  local log="$1"; shift
  cd "$REPO" || exit 3   # CWD must be the repo root: the chat-template fallback
                         # is resolved relative to it.
  setsid "$BIN" serve "$MODEL" --draft-model "$DRAFT" --port "$PORT" \
    --dflash --dflash-gamma "$GAMMA" --kv-cache-dtype fp8 \
    --gpu-memory-utilization "$GPU_UTIL" --max-seq-len "$MAX_SEQ_LEN" \
    --max-batch-size "$MAX_BATCH_SIZE" "$@" \
    > "$log" 2>&1 < /dev/null &
  disown 2>/dev/null || true
}

# Poll /health until ready, but abort the moment the process dies -- otherwise a
# serve that OOMs at load costs the full 10-minute timeout before saying so.
# NOTE: /health answers `ready` on ANY serve bound to this port, including one
# started by someone else. Readiness proves the port is answering, not that it
# is answering with YOUR binary. laguna_lock is what makes it yours.
laguna_wait_ready() {
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

# The production decode stack. Every gate arm starts from exactly this set and
# then applies its own overrides, so "baseline" means one thing across scripts.
laguna_prod_env() {
  export ATLAS_CUBLAS_GEMM=1 ATLAS_DFLASH_ALL_MULTISEQ=1 ATLAS_DFLASH_BATCH_MOE=1
  export ATLAS_DFLASH_EAGLE_FIX=1 ATLAS_DFLASH_OPTION_B=1 ATLAS_DFLASH_SAM=1
  export ATLAS_DFLASH_SPEC_THINK=1 ATLAS_DFLASH_STEP_TIMING=1
  export ATLAS_KN_V2=1 ATLAS_KN_V4=1 ATLAS_KN_V5=1 ATLAS_MTP_GATE_FORCE=1
  export ATLAS_HOLO_MOE_GROUPED_CUTLASS=1 ATLAS_HOLO_MOE_GROUPED_DOWN=1
  # Retrieval/SAM pre-emption. HYBRID_MIN defaults to the l_max cap (16), which
  # needs an exact-16 match to ever fire; 8 is the measured setting.
  export ATLAS_RETRIEVAL_HYBRID_MIN=8 ATLAS_RETRIEVAL_LMIN=4
  # Suspend speculation on content the drafter cannot predict (rolling 12-step
  # mean accept below the threshold), serial-decode until re-probe. Worth +13%
  # weighted. Read the caveat in decode_bench.py: a suspended request is a
  # SERIAL request, and its reported accept describes only the small speculative
  # prefix, so it must not be averaged into a DFlash table.
  export ATLAS_DFLASH_ADAPTIVE=1 ATLAS_DFLASH_ADAPTIVE_MIN=1.2
  export ATLAS_DFLASH_DRAFTER_FASTGEMM=1 ATLAS_FUSED_ELEMWISE=1
  # The two precision mirrors that pay: attention FP8 is worth ~16% tok/s and an
  # FP8 lm_head ~6%, essentially additive, and neither costs measurable quality.
  export ATLAS_TARGET_ATTN_FP8_MIRROR=1 ATLAS_TARGET_LMHEAD_FP8=1
  # g_proj verify GEMV: the head gate ran the prefill tensor-core GEMM at M=6,
  # launching a single CTA. Part of the shipping baseline since it passed its
  # quality gate, not a lever.
  export ATLAS_VERIFY_GPROJ_GEMV=1
}

# Apply `KEY=VALUE` overrides from a script's positional args. An EMPTY value
# means unset, which is how an arm turns a baseline gate back OFF.
laguna_apply_overrides() {
  for kv in "$@"; do
    local k="${kv%%=*}" v="${kv#*=}"
    if [ -z "$v" ]; then unset "$k"; else export "$k=$v"; fi
  done
}
