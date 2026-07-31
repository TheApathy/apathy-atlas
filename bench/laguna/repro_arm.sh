#!/usr/bin/env bash
# Run ONE decode arm of the published table.
#
#   repro_arm.sh <tag> [ENV=VAL ...]      # DFlash arm (laguna_serve adds --dflash)
#   ARM_SERIAL=1 repro_arm.sh serial      # serial arm (explicit launch, NO --dflash)
#
# This is gate_run.sh with the 69-case tool-eval removed: the decode table is
# what is being proven here, and the eval costs ~30 min/arm without touching the
# number under test. Everything else -- prod env, override semantics, launch
# shape, readiness and boot-failure detection -- comes from env.sh, so what gets
# proven is the published harness rather than a re-implementation of it.
#
# Normally you want repro_table.sh, which drives all four arms and grades them.
# This script exists separately because a single arm is the useful unit when a
# reproduction disagrees and you are bisecting which one.
set -uo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/env.sh"
laguna_require_model

TAG="${1:?usage: repro_arm.sh <tag> [ENV=VAL ...]}"; shift

# When run under repro_table.sh the driver already holds the real port lock for
# the whole campaign, and exports LAGUNA_LOCK to a private path for its
# children. laguna_lock re-execs fd 9 onto a fresh open-file-description, so
# without that split a child would deadlock against its own driver's lock.
laguna_lock

OUT="$OUT_ROOT/repro"
mkdir -p "$OUT"
LOG="$OUT/serve-$TAG.log"

# A stale artifact from an earlier sitting reads as a fresh result, and has
# produced a confident fake verdict on this project before -- a sweep whose arms
# had all died still printed a full table from 50-minute-old JSON. Wipe this
# arm's own outputs before it can be mistaken for this run's.
rm -rf "$OUT/$TAG.json" "$OUT/text-$TAG" "$LOG"

laguna_kill_serves 4

laguna_prod_env
laguna_apply_overrides "$@"

echo "=== ARM $TAG :: overrides: ${*:-<none>} ==="
echo "=== gproj gate: ${ATLAS_VERIFY_GPROJ_GEMV:-<unset>} ==="

if [ "${ARM_SERIAL:-0}" = 1 ]; then
  # laguna_serve hardcodes --dflash, so the serial arm launches explicitly.
  # Shape copied from capacity_table.sh's run_arm, including 9>&- so that an
  # orphaned serve cannot inherit and hold the port lock forever.
  cd "$REPO" || exit 3
  setsid "$BIN" serve "$MODEL" --port "$PORT" \
    --kv-cache-dtype "$KV_DTYPE" --gpu-memory-utilization "$GPU_UTIL" \
    --max-seq-len "$MAX_SEQ_LEN" --max-batch-size "$MAX_BATCH_SIZE" \
    > "$LOG" 2>&1 < /dev/null 9>&- &
  disown 2>/dev/null || true

  # Abort on the serve's OWN death, read from the log. `ps | grep -q "port
  # $PORT"` cannot be used here: it matches the grep's own command line and is
  # therefore true forever, which is how this loop silently span for the full
  # timeout on a serve that had already crashed.
  for _ in $(seq 1 120); do
    curl -s -m 3 "$BASE_URL/health" 2>/dev/null | grep -q ready && break
    laguna_serve_alive || { echo "FATAL: serve died during load ($TAG):"
                            tail -10 "$LOG"; exit 4; }
    sleep 5
  done
  curl -s -m 3 "$BASE_URL/health" 2>/dev/null | grep -q ready \
    || { echo "FATAL: never ready ($TAG)"; exit 4; }
else
  laguna_serve "$LOG"
  laguna_wait_ready "$LOG" || { echo "FATAL: never ready ($TAG)"; exit 4; }
fi

# The same three config invariants capacity_table.sh asserts per arm. A serve
# that came up on the wrong kernel target, the wrong KV dtype, or the ChatML
# fallback produces entirely plausible tok/s about the wrong thing -- the
# failure mode in this work is almost never a crash.
grep -qa "laguna-s-2.1, nvfp4" "$LOG" || { echo "FATAL: kernel target wrong ($TAG)"; exit 4; }
laguna_assert_kv_dtype "$LOG" || exit 4
grep -qa "using default ChatML" "$LOG" && { echo "FATAL: ChatML fallback ($TAG)"; exit 4; }

python3 "$BENCH_DIR/decode_bench.py" --tag "$TAG" --log "$LOG" \
  --tokens 256 --json-out "$OUT/$TAG.json" --dump-text "$OUT/text-$TAG"
rc=$?

laguna_kill_serves 0
exit $rc
