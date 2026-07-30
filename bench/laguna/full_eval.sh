#!/usr/bin/env bash
# FULL 69-scenario tool-eval-bench score on the shipping stack, plus the decode
# suite in the SAME sitting so speed and quality are quoted from one serve.
#
# Why this exists rather than gate_run.sh:
#   - gate_run.sh runs `--categories K` only (13 of 69 cases). That is a SAFETY
#     subset, not a score you can quote as "how the model performs".
#   - a binary that does NOT contain ATLAS_VERIFY_GPROJ_GEMV would run the
#     shipped lever inert, so the score would belong to the pre-g_proj stack.
#     This script refuses such a binary below.
#
# Context depth: 8192, NOT serve_prod.sh's 3072. Tool-eval scenarios are
# multi-turn (up to --max-turns 8); at 3072 a long scenario truncates and scores
# as a model failure when it is really a context failure. fp8 KV here.
source "$(dirname "${BASH_SOURCE[0]:-$0}")/env.sh"
laguna_require_model

TAG="${TAG:-full}"
OUT="$OUT_ROOT/fulleval"

mkdir -p "$OUT"
rm -f "$OUT/$TAG.json" "$OUT/serve-$TAG.log" "$OUT/eval-$TAG.log" "$OUT/eval-$TAG.json"

# Two arms sharing the port destroy each other silently. laguna_lock takes the
# flock on fd 9 and refuses if a peer holds it; an orphaned serve inherits fd 9
# and keeps the lock, so a stale lock means "kill the serve", not "delete it".
laguna_lock

[ -x "$EVAL_BIN" ] || { echo "FATAL: tool-eval-bench not found at $EVAL_BIN"; exit 2; }
# grep -c, never -q: under pipefail -q exits early and strings takes SIGPIPE(141),
# which makes the PIPELINE fail and every gate read as absent.
[ "$(strings -a "$BIN" | grep -cF -- 'laguna-s-2.1')" -gt 0 ] \
  || { echo "FATAL: $BIN built without ATLAS_TARGET_MODEL=laguna-s-2.1"; exit 2; }
# ARMED is not DISPATCHED, but a gate string absent from the binary cannot even be
# ARMED: without ATLAS_VERIFY_GPROJ_GEMV present the export is a silent no-op and
# the run scores the PRE-g_proj stack.
[ "$(strings -a "$BIN" | grep -cF -- 'ATLAS_VERIFY_GPROJ_GEMV')" -gt 0 ] \
  || { echo "FATAL: $BIN cannot read ATLAS_VERIFY_GPROJ_GEMV -- this would score the PRE-g_proj stack"; exit 2; }

# Kill by PID matched on the port, never `pkill -f spark`: that pattern also
# matches this script's own command line and would kill the shell running it.
laguna_kill_serves 4

# Production stack, identical gate set to gate_run.sh / serve_prod.sh.
laguna_prod_env
# Prefill-only cuBLASLt route for the dense FFN + head gate. Validated at
# 2182 -> 3345 tok/s prefill (+53.3%) against a 1.1% A/A floor, decode
# unaffected, needle 3/3. Exported here so the eval stack matches serve_prod.sh
# -- a score belongs to a harness+config, and these two must not silently diverge.
# PFCUBLAS=0 reproduces the stack from before this gate was exported. That is
# required to A/A against any arm recorded before it -- adding a gate and then
# calling the result a repeat would turn a noise-floor measurement into a
# one-variable A/B and quietly answer a different question.
# Control legs UNSET rather than =0: this gate's predicate is == Some("1"), but
# most ATLAS_* gates here are presence-based and would read =0 as ENABLED.
if [ "${PFCUBLAS:-1}" = "1" ]; then export ATLAS_PREFILL_CUBLAS=1; else unset ATLAS_PREFILL_CUBLAS; fi

laguna_serve "$OUT/serve-$TAG.log"
laguna_wait_ready "$OUT/serve-$TAG.log"
grep -qa "laguna-s-2.1, nvfp4" "$OUT/serve-$TAG.log" || { echo "FATAL: kernel target wrong"; exit 4; }
[ "$(grep -ac "FP8 KV cache using" "$OUT/serve-$TAG.log")" -gt 0 ] \
  || { echo "FATAL: fp8 KV did not take"; exit 4; }
# Assert the PROPERTY, not one template spelling: ChatML means tool-calling
# collapses (83.3% -> 0.0%) and every tool-eval score would be garbage.
grep -qa "ChatML" "$OUT/serve-$TAG.log" \
  && { echo "FATAL: serve fell back to a ChatML template -- tool-calling is dead, score would be meaningless"; exit 4; }

echo "=== FULL EVAL $TAG  $(date -u +%FT%TZ) ==="
echo "=== BIN=$BIN  stack: fp8 KV / 8192 ctx / gamma 6 / g_proj ON ==="

echo
echo "--- decode suite (same serve) ---"
python3 "$BENCH_DIR/decode_bench.py" --tag "$TAG" \
  --log "$OUT/serve-$TAG.log" --tokens 256 --json-out "$OUT/$TAG.json"

echo
# THINK=1 drops --no-think, so the model reasons. That is the config production
# actually serves (MODEL.toml thinking_default = true) and the ONLY one in which
# the thinking-phase fixes and enable_think_loop_watchdog can execute at all --
# under --no-think they are no-ops by construction, which is why a --no-think
# eval scored them 0 of 69 flipped. Default stays --no-think so every historical
# score remains comparable; a THINK=1 run is a DIFFERENT baseline and must never
# be diffed against a --no-think one.
if [ "${THINK:-0}" = "1" ]; then THINK_FLAG=""; MODE="THINKING ON (production config)"
else THINK_FLAG="--no-think";      MODE="--no-think"; fi
echo "--- tool-eval-bench: ALL 69 scenarios, $MODE, error-rate 0, seed 1234 ---"
# --error-rate 0 --seed 1234 are MANDATORY: the bench's error injection uses an
# unseeded global RNG, so without them two runs are not comparable.
timeout 7200 "$EVAL_BIN" --base-url "$BASE_URL/v1" \
  $THINK_FLAG --error-rate 0 --seed 1234 --no-live \
  > "$OUT/eval-$TAG.log" 2>&1
rc=$?
[ $rc -ne 0 ] && echo "!! tool-eval exited rc=$rc (124 = hit the 2h timeout; partial score below is NOT a full run)"

echo
sed 's/\x1b\[[0-9;]*m//g' "$OUT/eval-$TAG.log" \
  | grep -aoE "TC-[0-9]+ [A-Za-z0-9 &,'/-]+\.\.\..*(PASS|FAIL|PARTIAL)" | head -80
echo
sed 's/\x1b\[[0-9;]*m//g' "$OUT/eval-$TAG.log" | grep -aE "^(Score|Overall|Category|Total)" | head -40

echo
echo "--- counts (a score is only readable if the case count is right) ---"
clean=$(sed 's/\x1b\[[0-9;]*m//g' "$OUT/eval-$TAG.log")
np=$(printf '%s' "$clean" | grep -aoE "\bPASS\b" | wc -l)
nf=$(printf '%s' "$clean" | grep -aoE "\bFAIL\b" | wc -l)
nr=$(printf '%s' "$clean" | grep -aoE "\bPARTIAL\b" | wc -l)
echo "PASS=$np FAIL=$nf PARTIAL=$nr  total=$((np+nf+nr)) (want 69)"
[ $((np+nf+nr)) -eq 69 ] || echo "!! NOT 69 graded scenarios -- this is a PARTIAL run; do not quote it as the suite score"

# --- thinking census: prove the run measured what it claims to measure --------
# A THINK=1 run in which nothing actually reasoned is not a thinking-on
# baseline, it is a --no-think run with a misleading tag -- and it would score
# ~82 and look perfectly healthy. Assert the positive, the same way a gate needs
# a DISPATCH proof and not just an ARMED string.
nthink=$(grep -ac "Thinking enabled" "$OUT/serve-$TAG.log")
nwd=$(grep -ac "Thinking-loop watchdog fired" "$OUT/serve-$TAG.log")
nreq=$(grep -ac "Chunked prefill start:" "$OUT/serve-$TAG.log")
echo
echo "--- thinking census (arrivals=$nreq) ---"
echo "Thinking enabled        = $nthink"
echo "Think-loop watchdog fired = $nwd   (MODEL.toml sets enable_think_loop_watchdog=false;"
echo "                                    that key was UNPARSED for a while, so a binary predating"
echo "                                    the parse fix leaves the watchdog ON and it can fire here)"
if [ "${THINK:-0}" = "1" ]; then
  [ "$nthink" -gt 0 ] \
    || echo "!! THINK=1 but the model never entered thinking -- this is NOT a thinking-on baseline, do not quote it"
else
  [ "$nthink" -eq 0 ] \
    || echo "note: $nthink requests thought despite --no-think (decode_bench does not pass the flag; eval-phase hits are the interesting ones)"
fi

laguna_kill_serves 0
echo "=== FULL EVAL DONE $(date -u +%FT%TZ) ==="
