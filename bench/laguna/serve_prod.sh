#!/usr/bin/env bash
# Canonical Laguna single-stream serve -- self-verifying.
#
#   bash bench/laguna/serve_prod.sh
#
# This is the launcher to start from if you want to reproduce the decode
# numbers. It is not just a command line: it asserts every invariant the
# configuration depends on, and refuses to report a healthy serve unless all of
# them hold. Each assert exists because its absence once produced a
# clean-looking but wrong measurement.
#
# Prerequisites (see README.md):
#   LAGUNA_BIN    release binary built by build_cutlass.sh
#   LAGUNA_MODEL  Laguna-S-2.1-NVFP4 snapshot directory
#   LAGUNA_DRAFT  Laguna-S-2.1-DFlash-NVFP4 snapshot directory
source "$(dirname "${BASH_SOURCE[0]:-$0}")/env.sh"
laguna_require_model

LOG="${LAGUNA_SERVE_LOG:-$OUT_ROOT/serve-prod.log}"
mkdir -p "$(dirname "$LOG")"

# --- invariants ---------------------------------------------------------------
# 1. Native Laguna chat template -> NOT the ChatML fallback. The model card
#    measures tool-calling at 83.3% with the native template and 0.0% with
#    ChatML, so this is a correctness invariant, not a preference. Two spellings
#    are acceptable (the model's own bundled template, or the curated override in
#    ./jinja-templates/), which is why the assert below tests the PROPERTY
#    "never ChatML" rather than one spelling -- asserting one spelling produced a
#    false FAIL on a healthy serve when the resolution order changed upstream.
#    CWD must be the repo root either way: the override is resolved relative to
#    it. laguna_serve handles that.
# 2. CUTLASS grouped MoE on -> scale-factor blocks built at load, ~47 layers.
# 3. NO prefix caching     -> CUTLASS + prefix cache produces nondeterministic
#                             and sometimes corrupt output. Not a speed choice.
# 4. bf16 KV + temperature 0 -> greedy DFlash decode is byte-deterministic, which
#                             is what makes every hash comparison in this harness
#                             mean anything. This deliberately overrides the
#                             model config's fp8 KV recommendation; the gate
#                             scripts use fp8 and are a different stack, not a
#                             knob to copy across.
# 5. gamma=6, batch size 1 -> single-stream peak decode.
KV_DTYPE=bf16
SEQ_LEN=3072

laguna_kill_serves 3
laguna_prod_env

# ATLAS_PREFILL_CUBLAS routes the layer-0 dense FFN and the per-layer head gate
# through cuBLASLt instead of the tensor-core GEMM. PREFILL only; decode is
# untouched and measured unchanged. Worth ~+50% prefill throughput on a
# long-prompt sweep, replicated twice against a 1.1% A/A floor with R^2 >= 0.998
# on the latency fits. Both routes log which path they took once per site, so
# dispatch is proven per run rather than assumed.
#
# A third patched site (the paged head gate) is never executed by prompts that
# fit in --max-seq-len here, so it is UNTESTED. Untested is not broken, and it
# is not working either.
export ATLAS_PREFILL_CUBLAS=1

# ATLAS_CUBLAS_TUNED is DELIBERATELY NOT EXPORTED. Do not "complete the pair" --
# it lives in the same function as the gate above, so it looks like the natural
# follow-on, and it is a measured LOSS of ~2.1% against a 0.44% A/A floor.
# MECHANISM, so this is not re-litigated: the tuned plan cache is keyed on
# (m,n,k), and on the prefill route `m` is the chunk's token count, which varies
# per request. A run logged 373 tune events for 373 DISTINCT shapes -- zero
# reuse, still tuning when the log ended, several seconds of tuning GPU time and
# gigabytes of transient allocation churn to serve a decode phase measured in
# tens of seconds. The gate is sound where it was designed (decode/verify, where
# m = gamma+1 is constant); it is only wrong on variable-length prefill.

# --- PRE-LAUNCH GATE GUARD ----------------------------------------------------
# Every ATLAS_* gate this script exports must exist as a string in $BIN, or the
# export is a silent no-op and the serve runs a configuration nobody asked for.
# That has happened twice, and both times the knowledge existed in a header
# comment the next author did not read -- so this is a guard, not a third
# comment.
#
# The list is derived from this file's own exports plus laguna_prod_env, never
# from memory: checking a remembered gate NAME is its own failure mode, and a
# gate that does not exist fails into a false bug hunt rather than an error.
#
# `grep -cF ... || true`, never `grep -q`: under `set -o pipefail` a -q match
# makes `strings` take SIGPIPE and the pipeline report failure, which would mark
# EVERY gate absent -- the guard against silent no-ops getting its own silent
# failure. Scope it to real export lines too, because the comments above name
# gates on purpose (including one that deliberately must not be set) and a
# whole-file grep would harvest those and FATAL on a correct binary.
missing=""
for v in $( { grep -hE '^export ' "$0"; declare -f laguna_prod_env; } \
            | grep -oE 'ATLAS_[A-Z0-9_]+' | sort -u); do
  [ "$(strings -a "$BIN" | grep -cF -- "$v" || true)" -gt 0 ] || missing="$missing $v"
done
[ -z "$missing" ] || { echo "FATAL: $BIN cannot read gate(s):$missing"; \
  echo "(the export would be a silent no-op -- wrong binary for this launcher)"; exit 6; }
echo "OK gate guard: $BIN can read every ATLAS_* gate this script exports"

# A binary that cannot load the model is as dormant as a missing gate, and it
# fails only at serve time, long after the build looked clean.
[ "$(strings -a "$BIN" | grep -cF -- 'laguna-s-2.1' || true)" -gt 0 ] \
  || { echo "FATAL: $BIN was built without ATLAS_TARGET_MODEL=laguna-s-2.1"; exit 6; }

laguna_serve "$LOG" --kv-cache-dtype "$KV_DTYPE" --max-seq-len "$SEQ_LEN"
echo "launched (gamma=$GAMMA, KV=$KV_DTYPE, CUTLASS on, no prefix cache); waiting for ready..."
laguna_wait_ready "$LOG" 60

# --- POST-BOOT ASSERTS: fail loudly if any invariant is off -------------------
fail=0
grep -q "laguna-s-2.1, nvfp4" "$LOG" || { echo "ASSERT FAIL: kernel target is not laguna-s-2.1"; fail=1; }

# Invariant 1. Assert the property, not the spelling: either native template is
# fine, ChatML never is.
ntmpl=$(( $(grep -c "Using model's bundled chat template" "$LOG" || true) \
        + $(grep -c "override Jinja template from ./jinja-templates/laguna.jinja" "$LOG" || true) ))
[ "$ntmpl" -ge 1 ] || { echo "ASSERT FAIL: no native laguna chat template loaded"; fail=1; }
[ "$(grep -c "using default ChatML" "$LOG" || true)" -eq 0 ] \
  || { echo "ASSERT FAIL: fell back to ChatML -- tool-calling accuracy goes to ~0"; fail=1; }
echo "OK chat template: native laguna ($(grep -oE "Using model's bundled chat template \([0-9]+ chars\)|override Jinja template from [^ ]+ \([0-9]+ chars\)" "$LOG" | tail -1))"

# Invariant 2.
[ "$(grep -c 'CUTLASS grouped SFB' "$LOG")" -ge 40 ] || { echo "ASSERT FAIL: CUTLASS SFB not built"; fail=1; }

# Invariant 4: the same temperature-0 prompt twice must be byte-identical.
# Without this, every hash comparison elsewhere in the harness is measured
# against a moving target and a "divergence" means nothing.
req='{"model":"laguna","messages":[{"role":"user","content":"Write a Python function gcd(a,b) using Euclid. Output only code."}],"max_tokens":80,"temperature":0.0,"chat_template_kwargs":{"enable_thinking":false}}'
hash_once() {
  curl -s -m 60 "$BASE_URL/v1/chat/completions" -H 'Content-Type: application/json' -d "$req" \
    | python3 -c "import sys,json,hashlib;print(hashlib.sha256(json.load(sys.stdin)['choices'][0]['message']['content'].encode()).hexdigest()[:16])" 2>/dev/null
}
h1=$(hash_once); h2=$(hash_once)
if [ -n "$h1" ] && [ "$h1" = "$h2" ]; then
  echo "OK determinism: $h1 == $h2"
else
  echo "ASSERT FAIL: nondeterministic ($h1 != $h2)"; fail=1
fi

if [ "$fail" = 0 ]; then
  echo "PROD SERVE HEALTHY -- all invariants pass"
else
  echo "PROD SERVE INCONSISTENT -- see failures above"; exit 5
fi
