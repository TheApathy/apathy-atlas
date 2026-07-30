#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Concurrency CAPACITY sweep. ONE serve, several concurrency levels against it.
#
# WHAT THIS MEASURES -- read before quoting anything:
#   scheduler/mod.rs:530-556 gates ngram/self-spec/MTP on `active.len() == 1`,
#   and DFlash rides MTP. So C=1 is DFlash, and every C>=2 row is DFlash OFF
#   with `step_decode_only` batching the active sequences (decode_step.rs).
#   The C>=2 numbers are AGGREGATE CAPACITY UNDER LOAD. They are not a
#   DFlash-vs-anything comparison, and must never be set against a
#   single-stream speculative figure as though one engine produced both.
#
# WHY ONE SERVE: relaunching per level would put load-independent
# launch-to-launch variance (measured 1.1% on prefill) inside the scaling
# curve. Same serve = the only thing changing across rows is C.
#
# DELIBERATE DEVIATION FROM serve_prod.sh: --max-batch-size is MAX_BATCH (not
# 1), because C>=2 is unreachable at 1 -- the scheduler simply never admits a
# second sequence. That makes the C=1 row a control for the flag itself: it
# should reproduce the known single-stream figure. If C=1 here disagrees with
# the single-stream prefill arm, then --max-batch-size alone moved the baseline
# and the whole curve is suspect.
source "$(dirname "${BASH_SOURCE[0]:-$0}")/env.sh"
laguna_require_model

CONC="${CONC:-1,2,4,8}"
MAX_BATCH="${MAX_BATCH:-8}"
TOKENS="${TOKENS:-256}"
TAG="${TAG:-conc}"
OUT="$OUT_ROOT/conc"
mkdir -p "$OUT"
LOG="$OUT/serve-$TAG.log"

# Serialize on the port. laguna_lock takes fd 9, which the serve inherits: an
# orphaned serve keeps holding the lock, so the next arm refuses rather than
# racing a process it cannot see. See env.sh for the full rationale.
laguna_lock

# Stale-artifact discipline: a previous run's JSON scoring as this one is how a
# dead sweep printed a live-looking verdict before. Wipe what we are about to
# write, up front.
rm -f "$LOG" "$OUT/$TAG.json"

# Kill any serve on the port by PID. Never `pkill -f spark`: the pattern would
# match this script's own command line and kill the shell running it.
laguna_kill_serves 4

# Production stack + prefill cuBLASLt (part of the baseline, not a lever).
laguna_prod_env
export ATLAS_PREFILL_CUBLAS=1

# Gates must exist in THIS binary or the export is inert (the launcher-newer-
# than-bin trap). Never `grep -q` under pipefail -- SIGPIPE marks every gate
# absent and the check passes vacuously.
for g in ATLAS_PREFILL_CUBLAS ATLAS_VERIFY_GPROJ_GEMV; do
  n=$(grep -ac "$g" "$BIN" 2>/dev/null || echo 0)
  [ "$n" -gt 0 ] || { echo "FATAL: $BIN has no '$g' string -- export would be inert"; exit 3; }
  echo "gate present in BIN: $g ($n)"
done

echo "=== conc_capacity $TAG :: CONC=$CONC MAX_BATCH=$MAX_BATCH TOKENS=$TOKENS BIN=$BIN ==="
# Explicit launch (not laguna_serve): this arm overrides --max-batch-size to
# MAX_BATCH so C>=2 is admissible, and closes fd 9 in the child (9>&-) so an
# orphaned serve does not keep the lock. Follows laguna_serve's shape otherwise.
cd "$REPO" || exit 3
setsid "$BIN" serve "$MODEL" --draft-model "$DRAFT" --port "$PORT" \
  --dflash --dflash-gamma "$GAMMA" --kv-cache-dtype fp8 \
  --gpu-memory-utilization "$GPU_UTIL" --max-seq-len "$MAX_SEQ_LEN" --max-batch-size "$MAX_BATCH" \
  > "$LOG" 2>&1 < /dev/null 9>&- &
disown 2>/dev/null || true
for i in $(seq 1 120); do
  curl -s -m 3 "$BASE_URL/health" 2>/dev/null | grep -q ready && break
  ps -eo cmd | grep -q "port $PORT" || { echo "FATAL: serve died"; grep -Ei "out of memory|illegal|CUDA_ERROR|panicked|Error: " "$LOG" | head -5; exit 4; }
  sleep 5
done
curl -s -m 3 "$BASE_URL/health" 2>/dev/null | grep -q ready || { echo "FATAL: never ready"; exit 4; }

# NOTE: there is deliberately no `grep max_batch_size` check here. The serve
# never logs that string, so such a check would pass vacuously and report
# nothing forever. The real proof that C>=2 was admitted is the INVERSE of the
# DFlash gate: speculative decode requires active.len()==1, so a batched C>=2
# window contains ZERO `accepted=` steps. conc_capacity.py asserts that, and
# calls out the serialized case explicitly.

python3 "$BENCH_DIR/conc_capacity.py" \
  --log "$LOG" --conc "$CONC" --tokens "$TOKENS" --json-out "$OUT/$TAG.json"

echo "--- serve-side batching evidence (CONC_DIAG needs RUST_LOG=debug; absent is expected) ---"
grep -ac "CONC_DIAG" "$LOG" || true

laguna_kill_serves 0
echo "=== conc_capacity $TAG done ==="
