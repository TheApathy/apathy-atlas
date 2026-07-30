#!/usr/bin/env bash
# A/B the dense-FFN + head-gate prefill GEMMs on cuBLASLt (ATLAS_PREFILL_CUBLAS).
#
# WHAT IS BEING TESTED
# Laguna's layer 0 is a DENSE FFN (config `mlp_only_layers: [0]`) and its
# gate/up/down weights sit in the checkpoint's quantization ignore-list, so they
# are native BF16 and run through `dense_gemm_tc`. So does the per-layer head
# gate (g_proj) in both prefill routes. Two upstream commits move all of that
# onto cuBLASLt; one claims 33% of C=1 prefill.
#
# ONE BINARY, TWO ARMS. Upstream hard-wires the new route behind
# `ops::cublas_gemm_enabled()`, which production already exports -- that would
# make the control leg unreachable without a second build, and a two-binary A/B
# confounds the patch with the build. The tree here puts all three sites behind
# `ops::prefill_cublas_dense_enabled()` instead, so base and treatment are the
# same bytes and differ by one env var.
#
# THREE ARMS: base, cublas, base2. base/base2 is an A/A pair -- it measures this
# harness's floor today instead of inheriting one, and the contamination census
# needs >=3 arms to form a consensus (with 2 it can only say UNCHECKED).
#
# THREE THINGS THIS SCRIPT REFUSES TO CONFLATE
#   1. "the new route ran"     -- each patched site prints a one-shot
#      `route=cublaslt|dense_gemm_tc` line. dense_ffn.rs carries NO prof_step,
#      so this layer is invisible in the ATLAS_PREFILL_HOST_TIMING table; that
#      is exactly how its cost got read as "small" when it was never measured.
#      Without the route line a flat A/B cannot tell "cuBLASLt is no faster"
#      from "the cuBLASLt branch never ran". ARMED is not DISPATCHED: a gate
#      string present in the binary does not prove the code path executed, which
#      is why the route line is logged per site.
#   2. "the run was ours"      -- request census off the serve log.
#   3. "the answer is intact"  -- cuBLASLt reorders accumulation, so this is NOT
#      bit-exact and hash inequality is EXPECTED. The planted-needle oracle is
#      the arbiter; the base/base2 hash pair says whether hashes carry any
#      information here at all.
source "$(dirname "${BASH_SOURCE[0]:-$0}")/env.sh"
laguna_require_model

OUT="$OUT_ROOT/prefillcublas"
TRIALS="${TRIALS:-3}"
ARMS="${ARMS:-base cublas base2}"

mkdir -p "$OUT"
# Wipe only the arms being re-run. A stale JSON is how a dead arm gets scored as
# a live one -- that has already produced one fake correctness verdict here.
for a in $ARMS; do
  rm -f "$OUT/prefill-$a.json" "$OUT/decode-$a.json" "$OUT/hash-$a.json" "$OUT/serve-$a.log"
done

# Serialize on the port (fd 9, inherited by the serve -- see env.sh).
laguna_lock

# The gate AND its dispatch proof must live in THIS binary, not merely in the
# source tree -- a launcher shipping an env var its BIN predates is inert, and
# we have shipped that before. grep -c, never grep -q: under pipefail
# `strings | grep -q` takes SIGPIPE and reports every gate as absent.
for g in ATLAS_PREFILL_CUBLAS "DENSE_FFN prefill route=" "HEAD_GATE prefill"; do
  [ "$(strings -a "$BIN" | grep -cF -- "$g")" -gt 0 ] \
    || { echo "FATAL: $BIN does not carry '$g' -- wrong binary"; exit 2; }
done

echo "=== PREFILL cuBLASLt (dense-FFN + head-gate)  $(date -u +%FT%TZ) ==="
echo "=== BIN=$BIN  sha=$(sha256sum "$BIN" | cut -c1-8)  arms: $ARMS ==="

# Production env block, verbatim. ATLAS_CUBLAS_GEMM=1 is part of it and is a
# PRECONDITION of the new gate (prefill_cublas_dense_enabled() ANDs the two), so
# it must stay set in every arm including the controls.
laguna_prod_env

run_arm() {
  local tag="$1"
  local log="$OUT/serve-$tag.log"

  # The control legs must OMIT the var, not set it to 0. This particular
  # predicate is `== Some("1")`, but most ATLAS_* gates in this tree are
  # presence-based and would read =0 as ENABLED; unset is the only spelling
  # that is safe to copy.
  unset ATLAS_PREFILL_CUBLAS
  case "$tag" in
    base|base2) ;;
    cublas)     export ATLAS_PREFILL_CUBLAS=1 ;;
    *) echo "FATAL: unknown arm $tag"; exit 2 ;;
  esac

  # Kill any serve on the port by PID. Never `pkill -f spark` -- it matches this
  # script's own command line.
  laguna_kill_serves 4

  # Explicit launch (not laguna_serve): serial (no --dflash) -- all three
  # patched sites are on the prefill path and are identical either way, and
  # dropping the drafter removes its KV fill from the slope. fd 9 is closed in
  # the child (9>&-) so an orphaned serve does not keep the lock.
  cd "$REPO" || exit 3
  setsid "$BIN" serve "$MODEL" --port "$PORT" \
    --kv-cache-dtype fp8 --gpu-memory-utilization "$GPU_UTIL" \
    --max-seq-len "$MAX_SEQ_LEN" --max-batch-size "$MAX_BATCH_SIZE" \
    > "$log" 2>&1 < /dev/null 9>&- &
  disown 2>/dev/null || true

  # Abort on the serve's own error line. A `ps | grep -q "port $PORT"` liveness
  # probe cannot detect death here -- it matches the grep's own command line and
  # is true forever, turning a boot failure into a 10-minute timeout with the
  # reason buried in the log.
  for i in $(seq 1 120); do
    curl -s -m 3 "$BASE_URL/health" 2>/dev/null | grep -q ready && break
    err=$(grep -aEi "out of memory|illegal memory|CUDA_ERROR|panicked|^Error: " "$log" | head -3)
    [ -n "$err" ] && { echo "FATAL: serve failed to boot ($tag):"; echo "$err"; exit 4; }
    sleep 5
  done
  curl -s -m 3 "$BASE_URL/health" 2>/dev/null | grep -q ready \
    || { echo "FATAL: never ready ($tag)"; exit 4; }
  # /health says ready on a foreign serve too -- the port is a shared default
  # across serves on the same host. This line is what proves the answer came
  # from OUR target.
  grep -qa "laguna-s-2.1, nvfp4" "$log" || { echo "FATAL: kernel target wrong ($tag)"; exit 4; }

  echo
  echo "--------- ARM: $tag  (ATLAS_PREFILL_CUBLAS=${ATLAS_PREFILL_CUBLAS:-unset}) ---------"
  python3 "$BENCH_DIR/prefill_bench.py" \
    --trials "$TRIALS" --json-out "$OUT/prefill-$tag.json"
  echo
  # Correctness on prompts long enough to REACH the patched GEMMs at a real M.
  # decode_bench's one-line prompts prefill under 64 tokens; they are run below
  # as a decode sanity check only and the scorer does not read their hashes.
  python3 "$BENCH_DIR/prefill_hash_probe.py" --tag "$tag" \
    --json-out "$OUT/hash-$tag.json"
  echo
  python3 "$BENCH_DIR/decode_bench.py" --tag "$tag" \
    --log "$log" --tokens 256 --json-out "$OUT/decode-$tag.json"

  # Surface the dispatch proof immediately: if the gate did not take, there is
  # no reason to spend another ~8 minutes on the remaining arms.
  echo
  grep -aE "(DENSE_FFN|HEAD_GATE) prefill.* route=" "$log" | sed 's/^/    /' \
    || echo "    (no route lines -- the patched sites did not execute)"
}

for a in $ARMS; do run_arm "$a"; done

laguna_kill_serves 0

echo
python3 "$BENCH_DIR/prefill_cublas.py" --dir "$OUT" --arms $ARMS
rc=$?
echo "=== PREFILL cuBLASLt DONE $(date -u +%FT%TZ) ==="
exit $rc
