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

# --- GATE PARITY --------------------------------------------------------------
# qwen_champion_env() is a second copy of the configuration in
# local/serve-aeon-27b-dflash.sh, which is the script the published numbers were
# actually measured on. Two copies of one truth drift, and this pair already did
# -- silently, for eleven gates. Re-derive from the launcher and refuse to serve
# on any difference, rather than producing numbers for a near-miss of the
# configuration the README describes.
#
# Runs BEFORE qwen_kill_serves below: a config error should not cost you a
# running serve before it tells you what is wrong.
bash "$BENCH_DIR/verify_gate_parity.sh" \
  || { echo "FATAL: refusing to serve a configuration that is not the champion." >&2; exit 7; }

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
#
# NOT every ATLAS_* var is read from the environment by the binary. A few are
# consumed by the SHELL and forwarded on the command line -- ATLAS_DFLASH_QUANT
# becomes `--dflash-quantization "$ATLAS_DFLASH_QUANT"`. Nothing in the binary
# ever looks that name up, so its absence from `strings` is correct, and the
# first version of this guard failed a perfectly good binary on it. A guard that
# cries wolf gets switched off, so the classification matters.
#
# Waving those through by name would be worse than the false positive: it would
# turn a real "binary predates this flag" into a pass. So a forwarded var is not
# exempted, it is REDIRECTED -- we require the flag it feeds to be in the binary
# instead. Both the forwarding map and the gate list are re-derived from the
# authoritative scripts on every run; neither is a remembered list.
FWD_SRC="$REPO/local/serve-aeon-27b-dflash.sh $BENCH_DIR/env.sh"
for f in $FWD_SRC; do
  [ -f "$f" ] || { echo "FATAL: forwarding map source missing: $f"; \
    echo "(cannot classify shell-forwarded gates -- refusing to guess)"; exit 6; }
done
# shellcheck disable=SC2086
fwd_flag_for() {
  grep -hoE -- "--[a-z0-9-]+[= ]\"?\\\$\{?$1\b" $FWD_SRC 2>/dev/null \
    | grep -oE -- '--[a-z0-9-]+' | head -1
}

missing=""; forwarded=""; n_env=0
for v in $( { grep -hE '^export ' "$0"; declare -f qwen_champion_env; } \
            | grep -oE 'ATLAS_[A-Z0-9_]+' | sort -u); do
  if [ "$(strings -a "$BIN" | grep -cF -- "$v" || true)" -gt 0 ]; then
    n_env=$((n_env + 1)); continue
  fi
  flag="$(fwd_flag_for "$v")"
  if [ -n "$flag" ]; then
    # Forwarded on the command line. The value still has to reach the binary:
    # if the flag is missing too, this really is the wrong binary.
    if [ "$(strings -a "$BIN" | grep -cF -- "$flag" || true)" -gt 0 ]; then
      forwarded="$forwarded $v($flag)"
    else
      missing="$missing $v(forwarded as $flag, which $BIN does not accept)"
    fi
  else
    missing="$missing $v"
  fi
done
[ -z "$missing" ] || { echo "FATAL: $BIN cannot read gate(s):$missing"; \
  echo "(the export would be a silent no-op -- wrong binary for this launcher)"; exit 6; }
# Print the counts, not just "ok". A guard that reports success without saying
# how much it looked at reads identically whether it checked 32 gates or zero.
# "$n_env names the binary can read" -- deliberately not "gates this stack
# exports". The list is harvested from the function BODY, so it also picks up
# names guarded by a conditional (ATLAS_DFLASH_STEP_TIMING is not exported at
# the champion default). Checking those is right -- the stack can export them --
# but calling them exported would be a claim the number does not support.
echo "OK gate guard: $BIN can read $n_env of the ATLAS_* names this stack references"
[ -z "$forwarded" ] || echo "   + forwarded on the command line (flag verified present):$forwarded"

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
