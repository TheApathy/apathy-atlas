#!/usr/bin/env bash
# Canonical Qwen3.6-27B champion single-stream serve -- self-verifying.
#
#   bash bench/qwen/serve_champion.sh
#
# This is the launcher to start from if you want to reproduce the champion decode
# numbers. It is not just a command line: it asserts every invariant the
# configuration depends on, and refuses to report a healthy serve unless all of
# them hold. Each assert exists because its absence once produced a
# clean-looking but wrong measurement.
#
# Prerequisites (see README.md):
#   QWEN_BIN    release binary built with the qwen3.6-27b kernel target
#   QWEN_MODEL  target checkpoint snapshot directory
#   QWEN_DRAFT  DFlash drafter checkpoint snapshot directory
source "$(dirname "${BASH_SOURCE[0]:-$0}")/env.sh"
qwen_require_model

LOG="${QWEN_SERVE_LOG:-$OUT_ROOT/serve-champion.log}"
mkdir -p "$(dirname "$LOG")"

qwen_kill_serves 3
qwen_champion_env

# --- PRE-LAUNCH GATE GUARD ----------------------------------------------------
# Every ATLAS_* gate this stack exports must exist as a string in $BIN, or the
# export is a silent no-op and the serve runs a configuration nobody asked for.
# That failure is invisible: the serve is healthy, the numbers are merely wrong.
#
# The list is derived from qwen_champion_env's actual body plus this file's own
# exports, never from memory -- checking a remembered gate NAME is its own
# failure mode, because a gate that does not exist fails into a false bug hunt
# rather than into an error. `declare -f` also strips comments, which is what
# keeps the "DO NOT ENABLE" names in env.sh from being harvested here.
#
# `grep -cF ... || true`, never `grep -q`: under `set -o pipefail` a -q match
# makes `strings` take SIGPIPE and the pipeline report failure, which would mark
# EVERY gate absent -- the guard against silent no-ops getting its own silent
# failure.
missing=""
for v in $( { grep -hE '^export ' "$0"; declare -f qwen_champion_env; } \
            | grep -oE 'ATLAS_[A-Z0-9_]+' | sort -u); do
  [ "$(strings -a "$BIN" | grep -cF -- "$v" || true)" -gt 0 ] || missing="$missing $v"
done
[ -z "$missing" ] || { echo "FATAL: $BIN cannot read gate(s):$missing"; \
  echo "(the export would be a silent no-op -- wrong binary for this launcher)"; exit 6; }
echo "OK gate guard: $BIN can read every ATLAS_* gate this stack exports"

# A binary built without this kernel target loads nothing, and it fails only at
# serve time -- long after the build looked clean.
[ "$(strings -a "$BIN" | grep -cF -- "$KERNEL_TARGET" || true)" -gt 0 ] \
  || { echo "FATAL: $BIN was built without the $KERNEL_TARGET kernel target"; exit 6; }

qwen_serve "$LOG"
echo "launched (gamma=$GAMMA, K=$((GAMMA + 1)), KV=$KV_DTYPE +${KV_HP_LAYERS}hp, nvfp4 drafter); waiting for ready..."
qwen_wait_ready "$LOG" 120

# --- POST-BOOT ASSERTS: fail loudly if any invariant is off -------------------
fail=0

# 1. Right kernel target. Matched as a PREFIX: the serve prints the target's
#    Display, which is "<model>, <quant>", and pinning the quant suffix here
#    would produce a false FAIL the day it changes for an unrelated reason.
[ "$(grep -cF "Selected kernel target: $KERNEL_TARGET" "$LOG" || true)" -ge 1 ] \
  || { echo "ASSERT FAIL: kernel target is not $KERNEL_TARGET"; fail=1; }

# 2. Determinism. The same temperature-0 prompt twice must be byte-identical.
#    Without this, every hash comparison in this harness is measured against a
#    moving target and a "divergence" means nothing.
#    enable_thinking=false on purpose: ATLAS_THINK_SPEC is deliberately NOT
#    byte-lossless in thinking mode (batched verify and sequential decode differ
#    on the SSM layers), so a thinking-mode probe would fail this assert by
#    design. It was gated on quality instead -- see README.md.
req="{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a Python function gcd(a,b) using Euclid. Output only code.\"}],\"max_tokens\":80,\"temperature\":0.0,\"chat_template_kwargs\":{\"enable_thinking\":false}}"
last_text=""
hash_once() {
  last_text=$(curl -s -m 120 "$BASE_URL/v1/chat/completions" -H 'Content-Type: application/json' -d "$req" \
    | python3 -c "import sys,json;print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null)
  printf '%s' "$last_text" | sha256sum | cut -c1-16
}
h1=$(hash_once); probe_text="$last_text"; h2=$(hash_once)
if [ -n "$h1" ] && [ "$h1" = "$h2" ]; then
  echo "OK determinism: $h1 == $h2"
else
  echo "ASSERT FAIL: nondeterministic ($h1 != $h2)"; fail=1
fi

# 3. The K=17 corruption symptom, asserted directly rather than trusted.
#    A DRAFT_CAP that is not equal to gamma routes the SSM through the
#    sequential path, which NaNs at positions K-3..K-1. It does NOT crash: the
#    target emits a correct first token followed by a run of '!'. env.sh refuses
#    to launch a mismatched pair, but that guard checks what we SET, and this one
#    checks what actually came out.
case "$probe_text" in
  *'!!!!'*) echo "ASSERT FAIL: output contains a '!!!!' run -- the K=$((GAMMA + 1)) SSM corruption signature."
            echo "  Check that ATLAS_DFLASH_DRAFT_CAP == dflash-gamma == $GAMMA."
            fail=1 ;;
  *)        echo "OK no SSM corruption signature in probe output" ;;
esac

# 4. Speculation is actually running at the documented width. A serve that
#    silently decodes serially is fast enough to look plausible and produces
#    numbers that are not about DFlash at all.
#    Anchored on the "N/M (P%)" form -- see benchenv.py for why a bare
#    `accepted=` would also match the step-timing line and double-count.
widths=$(grep -oE 'accepted=[0-9]+/[0-9]+ \([0-9]+%\)' "$LOG" | sed -E 's|.*/([0-9]+) .*|\1|' | sort -u)
nsteps=$(grep -cE 'accepted=[0-9]+/[0-9]+ \([0-9]+%\)' "$LOG" || true)
if [ "$nsteps" -eq 0 ]; then
  echo "ASSERT FAIL: no speculative verify steps logged -- DFlash is not running"; fail=1
elif [ "$widths" != "$GAMMA" ]; then
  echo "ASSERT FAIL: verify width(s) [$(echo "$widths" | tr '\n' ' ')] != $GAMMA"; fail=1
else
  echo "OK speculative decode: $nsteps verify steps, all at width $GAMMA"
fi

if [ "$fail" = 0 ]; then
  echo "CHAMPION SERVE HEALTHY -- all invariants pass"
else
  echo "CHAMPION SERVE INCONSISTENT -- see failures above"; exit 5
fi
