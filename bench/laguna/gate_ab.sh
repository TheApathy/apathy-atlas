#!/usr/bin/env bash
# Serialized quality+speed A/B on the PRODUCTION serve config (fp8 KV, gamma 6,
# 8K ctx) via gate_run.sh. For changes that are NOT bit-exact and therefore need
# a quality gate instead of a hash-parity gate.
#
#   ARMS_SPEC='tag|ENV=V ENV2=V2
#   tag2|'  bash gate_ab.sh
#
# Three things this adds over calling gate_run.sh by hand, each of which has
# silently destroyed a real run:
#   1. waits for the port lock instead of failing out (gate_run.sh uses
#      laguna_lock's flock -n, so a peer's serve makes every arm exit 5 in a burst)
#   2. gate-string check on $BIN -- a binary built without ATLAS_VERIFY_GPROJ_GEMV
#      sets a gate it cannot read, so an arm that toggles it reads as "no effect"
#   3. per-arm contamination assert over the DECODE WINDOW ONLY (exactly 7 Done
#      before the decode/eval boundary) -- foreign clients on the shared port fake
#      -60% regressions
source "$(dirname "${BASH_SOURCE[0]:-$0}")/env.sh"
laguna_require_model

OUT="$OUT_ROOT/gate"
RUNLOG="${RUNLOG:-$OUT_ROOT/gate_ab.log}"
mkdir -p "$OUT"

# A binary built for a different kernel target loads no model and is as dormant
# as a missing gate; laguna_require_model checks the binary is executable but not
# which target it was compiled for.
[ "$(strings -a "$BIN" | grep -cF -- 'laguna-s-2.1' || true)" -gt 0 ] \
  || { echo "FATAL: $BIN built without ATLAS_TARGET_MODEL=laguna-s-2.1"; exit 2; }

echo "=== gate_ab start $(date -u +%H:%M:%S) BIN=$BIN ==="

while IFS= read -r line; do
  [ -z "$line" ] && continue
  TAG="${line%%|*}"; EXTRA="${line#*|}"

  # -- 2. every gate named in EXTRA must exist as a string in the binary.
  # Must be `grep -c`, never `grep -q`: under pipefail, -q exits on first match,
  # strings takes SIGPIPE (141), and the PIPELINE reports failure -- so every
  # gate reads as absent and every real arm is refused. Substring, not -cx:
  # Rust concatenates literals into a blob with no newlines.
  missing=""
  for v in $(echo "$EXTRA" | tr ' ' '\n' | sed -n 's/^\([A-Z_][A-Z0-9_]*\)=.\+$/\1/p'); do
    n=$(strings -a "$BIN" | grep -cF -- "$v" || true)
    [ "${n:-0}" -gt 0 ] || missing="$missing $v"
  done
  [ -z "$missing" ] || { echo "$TAG: FATAL dormant/absent gate(s):$missing -- in $BIN"; continue; }

  # -- 1. wait for the box rather than burning the arm on flock -n.
  # MUST use flock's file-argument form. The fd form (`flock -n 8 ... 8>FILE`)
  # reports a HELD lock as free, because the probe inherits the holder's fd and
  # Linux flock does not reliably deny a same-descriptor-family retry -- measured,
  # not theorised. That false "free" would stomp a peer's live serve.
  # Also check for a live serve directly: a peer running without the lock at all
  # is invisible to any lock probe.
  for i in $(seq 1 240); do
    free=$(flock -n "$LOCK" -c true 2>/dev/null && echo 1 || echo 0)
    live=$(ps -eo cmd | grep "port $PORT" | grep -vc grep || true)
    [ "$free" = 1 ] && [ "${live:-0}" = 0 ] && break
    [ "$i" = 1 ] && echo "$TAG: box busy (lock_free=$free serves=$live) -- waiting ..."
    sleep 15
  done

  # Stale JSON from an earlier invocation reads as a fresh result and has already
  # produced a fake correctness escalation once -- clear this arm's outputs first.
  rm -f "$OUT/$TAG.json" "$OUT/eval-$TAG.log" "$OUT/serve-$TAG.log"

  echo "########## ARM=$TAG  EXTRA=${EXTRA:-<none>} ##########"
  BIN="$BIN" bash "$BENCH_DIR/gate_run.sh" "$TAG" $EXTRA 2>&1

  # -- 3. contamination assert, DECODE WINDOW ONLY.
  # This used to count Done/Thinking over the whole gate log and demand `>=7 Done,
  # 0 Thinking`. That is a decode-ONLY discriminator: gate_run.sh also runs
  # tool-eval, which contributes ~38 requests of its own and logs "Thinking
  # enabled" on ~10 of them despite --no-think. So the "intruder signature" WAS
  # the harness -- five separate CLEAN arms tripped it, and two were nearly
  # discarded. assert_decode_clean.sh cuts the log at the decode/eval boundary and
  # requires EXACTLY 7 there, which is what actually catches injected traffic in
  # either direction.
  bash "$BENCH_DIR/assert_decode_clean.sh" "$OUT/serve-$TAG.log" "$TAG" || true
  echo
done <<< "$ARMS_SPEC"

echo "########## SUMMARY ##########"
for f in "$OUT"/eval-*.log; do
  [ -f "$f" ] || continue
  tag=$(basename "$f" .log); tag="${tag#eval-}"
  printf '%-12s %s\n' "$tag" "$(grep -aE '^Score:' "$f" | sed 's/\x1b\[[0-9;]*m//g' | tr -d '\n')"
done
echo "=== gate_ab end $(date -u +%H:%M:%S) ==="
