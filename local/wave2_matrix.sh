#!/bin/bash
# Wave-2 kernel A/B matrix (v2). Waits for the GPU to be truly free
# (no spark, no vLLM, no retrain/torchrun), then for each config:
# serve → md5 gate + n=5 counting/coding → kill OWN spark only.
# Results accumulate in /tmp/wave2-matrix-results.txt.
# Usage: wave2_matrix.sh "LABEL:ENV=1 ENV2=4" "LABEL2:ENV=2" ...
set -uo pipefail

RES=/tmp/wave2-matrix-results.txt
log() { echo "[wave2 $(date +%T)] $*" | tee -a "$RES"; }

gpu_busy() {
  pgrep -x spark >/dev/null 2>&1 && return 0
  pgrep -f "EngineCore|vllm serve" >/dev/null 2>&1 && return 0
  pgrep -f "v4_pilot|torchrun|train_dflash|v4_capture" >/dev/null 2>&1 && return 0
  local heavy
  heavy=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null \
    | awk -F', ' '$2+0 > 2048 {print $1}' | head -1)
  [ -n "$heavy" ] && return 0
  return 1
}

wait_gpu_free() {
  local quiet=0
  while true; do
    if gpu_busy; then
      quiet=0
      sleep 60
    else
      quiet=$((quiet + 1))
      # Require 2 consecutive clear checks 30s apart (avoid racing a
      # foreign serve that is between restarts).
      [ "$quiet" -ge 2 ] && return 0
      sleep 30
    fi
  done
}

run_config() {
  local label="$1"; shift
  log "waiting for free GPU for $label..."
  wait_gpu_free
  log "=== CONFIG $label (env: $*) ==="
  env "$@" setsid nohup bash /home/flocka/atlas-src/local/serve-aeon-27b-dflash.sh \
    > "/tmp/wave2-serve-$label.log" 2>&1 < /dev/null &
  # Grace period: the serve script runs preflight (incl. up to 4s pkill
  # sleep) before exec'ing spark — don't death-check for the first 30s.
  sleep 30
  local t0=$(date +%s)
  while ! grep -q "Listening on" "/tmp/wave2-serve-$label.log" 2>/dev/null; do
    if ! pgrep -x spark >/dev/null; then
      log "$label: SPARK DIED during load"
      tail -3 "/tmp/wave2-serve-$label.log" >> "$RES"
      return 1
    fi
    if [ $(( $(date +%s) - t0 )) -gt 600 ]; then
      log "$label: LOAD TIMEOUT (wedged) — killing own spark"
      pkill -9 -x spark
      return 1
    fi
    sleep 10
  done
  log "$label: ready after $(( 30 + $(date +%s) - t0 ))s"
  python3 /home/flocka/atlas-src/local/bench_wave2.py 8890 5 "$label" 2>&1 | tee -a "$RES"
  pkill -9 -x spark || true
  sleep 4
}

for spec in "$@"; do
  label="${spec%%:*}"
  envs="${spec#*:}"
  # shellcheck disable=SC2086
  run_config "$label" $envs
done
log "=== MATRIX DONE ==="
