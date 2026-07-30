#!/usr/bin/env bash
# Capacity + throughput table for Laguna-S-2.1-NVFP4 on this hardware class, in
# the shape of a single-node vs multi-node deployment table: weights resident,
# KV pool depth, prefill tok/s, decode tok/s.
#
# The reference table's second column is EP=2 (expert parallelism across two
# nodes). THAT COLUMN IS NOT REPRODUCIBLE HERE -- `nvidia-smi -L` reports one
# GPU, so there is no second rank to shard experts onto and no number to
# measure. This script does not estimate it. The second column here is the
# axis that IS real on one GPU: speculative decoding off vs on.
#
# Prefill has never been measured on this stack before; see prefill_bench.py
# for why it is a slope and not a ratio.
source "$(dirname "${BASH_SOURCE[0]:-$0}")/env.sh"
laguna_require_model

OUT="$OUT_ROOT/capacity"
TRIALS="${TRIALS:-3}"

ARMS="${ARMS:-serial dflash}"

mkdir -p "$OUT"
# Wipe the artifacts of the arms we are about to run -- a stale JSON from a
# previous sitting is how a dead arm gets scored as a live one. Arms NOT being
# re-run keep theirs on purpose (re-running one contaminated arm should not
# force a re-measure of a clean one), so the renderer's cross-arm comparison is
# only as same-sitting as the caller made it.
for arm in $ARMS; do rm -f "$OUT/prefill-$arm.json" "$OUT/decode-$arm.json" "$OUT/serve-$arm.log"; done

# Serialize on the port (fd 9, inherited by the serve -- see env.sh).
laguna_lock

[ "$(strings -a "$BIN" | grep -cF -- 'ATLAS_VERIFY_GPROJ_GEMV')" -gt 0 ] \
  || { echo "FATAL: $BIN predates g_proj -- wrong stack"; exit 2; }

ngpu=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
echo "=== CAPACITY TABLE  $(date -u +%FT%TZ) ==="
echo "=== GPUs visible: $ngpu (EP=2 needs 2; not reproducible at $ngpu) ==="

laguna_prod_env

run_arm() {
  local tag="$1"; shift          # remaining args are extra serve flags
  local log="$OUT/serve-$tag.log"

  # Kill any serve on the port by PID. Never `pkill -f spark` -- it matches this
  # script's own command line.
  laguna_kill_serves 4

  # Explicit launch (not laguna_serve): the serial arm must NOT pass --dflash,
  # and laguna_serve hardcodes it. fd 9 is closed in the child (9>&-) so an
  # orphaned serve does not keep the lock. Follows laguna_serve's shape.
  cd "$REPO" || exit 3
  setsid "$BIN" serve "$MODEL" --port "$PORT" \
    --kv-cache-dtype fp8 --gpu-memory-utilization "$GPU_UTIL" \
    --max-seq-len "$MAX_SEQ_LEN" --max-batch-size "$MAX_BATCH_SIZE" "$@" \
    > "$log" 2>&1 < /dev/null 9>&- &
  disown 2>/dev/null || true

  # Abort on the serve's OWN error line, not on a liveness probe.
  # `ps -eo cmd | grep -q "port $PORT"` cannot detect death here -- the pattern
  # matches the grep's own command line, so it is true forever and a serve that
  # exits at boot reads as a 10-minute "never ready" with the reason buried in
  # the log. Reading the reason is both faster and more informative.
  for i in $(seq 1 120); do
    curl -s -m 3 "$BASE_URL/health" 2>/dev/null | grep -q ready && break
    err=$(grep -aEi "out of memory|illegal memory|CUDA_ERROR|panicked|^Error: " "$log" | head -3)
    [ -n "$err" ] && { echo "FATAL: serve failed to boot ($tag):"; echo "$err"; exit 4; }
    sleep 5
  done
  curl -s -m 3 "$BASE_URL/health" 2>/dev/null | grep -q ready \
    || { echo "FATAL: never ready ($tag)"; exit 4; }
  grep -qa "laguna-s-2.1, nvfp4" "$log" || { echo "FATAL: kernel target wrong ($tag)"; exit 4; }
  grep -qa "ChatML" "$log" && { echo "FATAL: ChatML fallback ($tag)"; exit 4; }

  echo
  echo "--------- ARM: $tag  (serve flags: $*) ---------"
  python3 "$BENCH_DIR/prefill_bench.py" \
    --trials "$TRIALS" --json-out "$OUT/prefill-$tag.json"
  echo
  python3 "$BENCH_DIR/decode_bench.py" --tag "$tag" \
    --log "$log" --tokens 256 --json-out "$OUT/decode-$tag.json"
}

# --draft-model is NOT optional with --dflash: the serve exits at KV-cache setup
# with "no drafter HF id provided", which the readiness loop can only report as
# a 10-minute timeout. Both arms load the same target; only the drafter differs.
for arm in $ARMS; do
  case "$arm" in
    serial) run_arm serial ;;
    dflash) run_arm dflash --draft-model "$DRAFT" --dflash --dflash-gamma "$GAMMA" ;;
    *) echo "FATAL: unknown arm $arm"; exit 2 ;;
  esac
done

laguna_kill_serves 0

echo
python3 "$BENCH_DIR/capacity_table.py" --dir "$OUT" --arms $ARMS
echo "=== CAPACITY TABLE DONE $(date -u +%FT%TZ) ==="
