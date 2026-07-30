#!/usr/bin/env bash
# Combined speed+quality gate on ONE serve: production DFlash invariants,
# decode suite, then tool-eval safety categories. Used for changes that are
# NOT bit-exact and therefore need a quality gate rather than a parity gate.
#   gate_run.sh <tag> [ENV=VAL ...]
source "$(dirname "${BASH_SOURCE[0]:-$0}")/env.sh"
laguna_require_model
TAG="${1:?tag}"; shift

# Two arms sharing the port destroy each other silently. laguna_lock takes the
# flock on fd 9 and refuses if a peer already holds it; because children inherit
# fd 9, an orphaned serve keeps the lock, so the fix for a stale lock is to kill
# the serve, not delete the lock file.
laguna_lock

OUT="$OUT_ROOT/gate"
mkdir -p "$OUT"
LOG="$OUT/serve-$TAG.log"

# Kill by PID matched on the port, never `pkill -f spark`: that pattern also
# matches this script's own command line and would kill the shell running it.
laguna_kill_serves 4

# Full production stack (serve_prod.sh invariants).
laguna_prod_env
# g_proj VERIFY GEMV is part of laguna_prod_env's BASELINE stack, not a lever. A
# gate arm that wants the old behaviour must pass ATLAS_VERIFY_GPROJ_GEMV=
# (empty value) to unset it via the override loop below.
# Evidence: base 30.13 / base2 29.91 -> g_proj-on 32.30 weighted tok/s,
# step-weighted accept 2.876 -> 2.996, eval 62 -> 73.
# !! The binary MUST contain the ATLAS_VERIFY_GPROJ_GEMV string, not only
# ATLAS_DECODE_GPROJ_GEMV -- a binary built with just the decode gate sets a
# gate it cannot read and the result reads as "no effect". Prefer gate_ab.sh,
# which refuses an arm whose binary cannot read the gates named in its EXTRA.

laguna_apply_overrides "$@"
echo "=== GATE $TAG :: extra env: $* ==="

laguna_serve "$LOG"
laguna_wait_ready "$LOG"

python3 "$BENCH_DIR/decode_bench.py" --tag "$TAG" --log "$LOG" \
  --tokens 256 --json-out "$OUT/$TAG.json"

echo "--- tool-eval (categories K, error-rate 0, seed 1234) ---"
# --error-rate 0 --seed 1234 are MANDATORY: the bench's error injection uses an
# unseeded global RNG, so without them two runs are not comparable.
timeout 1800 "$EVAL_BIN" --base-url "$BASE_URL/v1" \
  --no-think --error-rate 0 --seed 1234 --categories K --no-live \
  > "$OUT/eval-$TAG.log" 2>&1
grep -aoE "TC-[0-9]+ [A-Za-z &-]+\.\.\. [^ ]*(PASS|FAIL|PARTIAL)[^ ]*" "$OUT/eval-$TAG.log" | sed 's/\x1b\[[0-9;]*m//g'
grep -aE "^Score:" "$OUT/eval-$TAG.log" | sed 's/\x1b\[[0-9;]*m//g'

laguna_kill_serves 0
