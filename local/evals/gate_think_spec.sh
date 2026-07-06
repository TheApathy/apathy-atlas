#!/usr/bin/env bash
# Think-spec quality gate — FIRST CONSUMER of the evals harness.
#
# Objectively decides whether ATLAS_THINK_SPEC=1 (the ~1.7x thinking speedup)
# changes OUTPUT QUALITY, so it can ship as a quality-neutral speed win.
#
# It runs coding pass@1 (HumanEval + MBPP) under two boots — A (lever off) and
# B (lever on) — over the SAME problem set, then runs the ABBA paired-bootstrap
# CI on the pass@1 delta. Ship iff the CI lower bound stays above -epsilon.
#
# REQUIRES A GPU WINDOW: this boots the live Atlas server twice. Do NOT run it
# while v5 training owns the GPU. The stats step (abba.py) is CPU-only and can
# be re-run any time on the saved result files.
#
# temp=0 is used for reproducibility. NOTE ON THINKING: the coding pass@1 path
# uses /v1/completions (no thinking). ATLAS_THINK_SPEC only affects the thinking
# decode path, so to actually exercise it you must also run the thinking-mode
# quality check (chat with enable_thinking=true) — see STEP 4. The coding gate
# proves the lever does not regress non-thinking coding; the thinking check
# proves reasoning-mode answers are unchanged. Both must pass to ship.
#
# Usage:
#   bash gate_think_spec.sh
#
# Env overrides:
#   PORT=8890  LIMIT=  (blank = full 164 HumanEval + full MBPP sample/set)
#   SERVE=/home/flocka/atlas-src/local/serve-aeon-27b-dflash.sh
#   EPSILON=0.01  ITERS=10000
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-8890}"
LIMIT_ARG=""
[ -n "${LIMIT:-}" ] && LIMIT_ARG="--limit ${LIMIT}"
SERVE="${SERVE:-/home/flocka/atlas-src/local/serve-aeon-27b-dflash.sh}"
EPSILON="${EPSILON:-0.01}"
ITERS="${ITERS:-10000}"
OUTDIR="${OUTDIR:-/tmp/evals_think_spec}"
mkdir -p "$OUTDIR"

# For a hard network cutoff around untrusted model code, export EVALS_UNSHARE_NET=1
# (needs `unshare` privileges). Sandbox is process+rlimit isolated regardless.

wait_up() {
  echo "[gate] waiting for server on :$PORT ..."
  for _ in $(seq 1 180); do
    if curl -s "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
      echo "[gate] server up"; return 0
    fi
    sleep 2
  done
  echo "[gate] server did not come up" >&2; return 1
}

boot_and_run() {
  local label="$1"; shift          # A_baseline / B_thinkspec
  local out="$1"; shift            # results json path
  echo "=============================================================="
  echo "[gate] BOOT $label with env: $*"
  # Kill any prior server on the port. (serve script also guards the port.)
  pkill -f "spark-server" 2>/dev/null || true
  sleep 3
  # Boot server in background with the lever env for this arm.
  env "$@" bash "$SERVE" > "$OUTDIR/serve_${label}.log" 2>&1 &
  local serve_pid=$!
  wait_up
  # Run coding pass@1 (both datasets) at temp=0, merged into one results file.
  python3 "$HERE/runner.py" --dataset both --label "$label" \
    --out "$out" --base-url "http://127.0.0.1:${PORT}" \
    --temperature 0 --seed 0 $LIMIT_ARG
  echo "[gate] $label done -> $out"
  # Tear down this arm's server before the next boot.
  pkill -f "spark-server" 2>/dev/null || true
  wait "$serve_pid" 2>/dev/null || true
  sleep 3
}

RES_A="$OUTDIR/resultsA.json"
RES_B="$OUTDIR/resultsB.json"

# --- ARM A: lever OFF (baseline) ---
boot_and_run "A_baseline" "$RES_A"      # no ATLAS_THINK_SPEC

# --- ARM B: lever ON ---
boot_and_run "B_thinkspec" "$RES_B" ATLAS_THINK_SPEC=1

# --- CPU-side stats (re-runnable any time) ---
echo "=============================================================="
echo "[gate] ABBA paired-bootstrap CI on pass@1 delta (B - A):"
python3 "$HERE/abba.py" "$RES_A" "$RES_B" --epsilon "$EPSILON" --iters "$ITERS"
GATE_RC=$?

echo
echo "[gate] STEP 4 (thinking-mode quality, manual): with each arm booted, run"
echo "  python3 /tmp/vb35/think_bench.py <label> t2 ${PORT}   # and t3/t5/t6"
echo "  Compare md5(content) A-vs-B: reasoning tokens may differ (spec reorders"
echo "  decode) but the FINAL content should be semantically equivalent. For a"
echo "  numeric read, run mtbench.py against each arm and diff mean_score."
echo
echo "[gate] SHIP iff: (1) ABBA verdict = 'B not worse than A' (exit 0 above),"
echo "                 (2) thinking-mode content is equivalent (STEP 4)."
exit $GATE_RC
