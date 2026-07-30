#!/usr/bin/env bash
# Contamination assert for a serve log, scoped to the DECODE window only.
#
# Another serve bound to the same port answers /health identically, so readiness
# proves the port is answering, not WHOSE serve is answering. The only reliable
# discriminator is to census the REQUESTS that actually arrived inside the
# measurement window and reject foreign traffic there, rather than trust any
# content feature of the responses.
#
# WHY A WINDOW. The 7-Done/0-Thinking discriminator was derived on a decode-ONLY
# log, where our client issues exactly 7 requests (1 warmup + 6 prompts) and
# never thinks. gate_run.sh runs decode_bench AND THEN tool-eval, and tool-eval
# adds ~38 requests of its own of which ~10 log "Thinking enabled" despite
# --no-think. Applied to a gate log the "intruder signature" IS the harness:
# three separate CLEAN arms all land on exactly Done=45 / Thinking=10, which once
# produced a false CONTAMINATED and nearly discarded a good arm.
#
# Worse, the eval's request count is an OUTCOME, not a signal: failed scenarios burn
# extra turns, so MORE Done lines correlates with a LOWER score (a 34-Done arm ->
# score 69, the 45-Done arms -> 62). Reading it as contamination inverts the
# causality.
#
# So: cut the log at the first "Thinking enabled" line -- the decode/eval boundary --
# and assert on what precedes it.
#
# EXACTLY 7, never >=7. The equality is what makes this catch anything at all:
#   - a foreign THINKING request during decode moves the boundary earlier  -> <7 -> FAIL
#   - a foreign non-thinking request during decode adds a Done to the window -> >7 -> FAIL
# The old `-ge 7` could not detect injected traffic in either direction.
#
# NO TIMESTAMP-OVERLAP CHECK, DELIBERATELY. It is tempting to assert that no foreign
# request overlaps one of ours (start = finish - tokens/tok_s). Measured across every
# gate arm, including a corrupted one, that condition fires NEVER: the intruder burst
# always drains before our next prompt is issued (burst 28.5-34.6s, math issued 36.3s).
# A check that cannot fail is worse than no check -- it reads as coverage. The corrupted
# arm's bad prose hash had NO established cause; co-batching and adaptive-spec
# suspension were both proposed and both falsified.
#
# What DOES catch it is the third arg: compare hashes against a known-good arm. That is a
# result-level check, and the corruption was only ever visible at result level.
#
# Usage: assert_decode_clean.sh <serve-log> [tag] [arm.json [reference.json]]
#        -> exit 0 clean, 1 dirty
source "$(dirname "${BASH_SOURCE[0]:-$0}")/env.sh"

SRV="${1:?usage: assert_decode_clean.sh <serve-log> [tag] [arm.json [ref.json]]}"
TAG="${2:-$(basename "$SRV")}"
ARM_JSON="${3:-}"
REF_JSON="${4:-}"
WANT="${WANT_DONE:-7}"
WANT_RESULTS="${WANT_RESULTS:-6}"

if [ ! -s "$SRV" ]; then
  echo "!! $TAG: serve log missing or empty ($SRV) -- cannot assert, treat as DIRTY"
  exit 1
fi

# `|| true` absorbs grep's exit 1 on no-match; -a because the log carries ANSI/binary bytes.
ft=$(grep -an "Thinking enabled" "$SRV" 2>/dev/null | head -1 | cut -d: -f1 || true)
if [ -n "$ft" ]; then
  d=$(head -n "$ft" "$SRV" | grep -ac "Done: " || true)
else
  # no eval phase in this log (decode-only run) -- the whole log is the window
  d=$(grep -ac "Done: " "$SRV" || true)
fi
echo "-- decode-window[$TAG]: Done=$d (want == $WANT)  boundary=line ${ft:-EOF} --"

# ---- result-level check (independent of the count check above) ----------------
# Reports PER PROMPT, never a single arm-level verdict: gp_off had five perfectly
# good prompts and one bad one, and an arm-level FAIL would have thrown away the five.
RES_RC=0
if [ -n "$ARM_JSON" ]; then
  if [ ! -s "$ARM_JSON" ]; then
    echo "!! $TAG: results JSON missing/empty ($ARM_JSON)"
    RES_RC=1
  else
    out=$(ARM="$ARM_JSON" REF="$REF_JSON" WANTN="$WANT_RESULTS" python3 - <<'PY'
import json, os, sys
arm_p, ref_p, wantn = os.environ["ARM"], os.environ.get("REF",""), int(os.environ["WANTN"])
try:
    arm = json.load(open(arm_p)).get("results", [])
except Exception as e:
    print(f"!! results JSON unreadable: {e}"); sys.exit(1)
rc = 0
if len(arm) != wantn:
    print(f"!! INCOMPLETE: {len(arm)} results, expected {wantn} -- arm cannot be scored")
    rc = 1
ref = {}
if ref_p:
    try:
        ref = {x["name"]: x.get("hash") for x in json.load(open(ref_p)).get("results", [])}
    except Exception as e:
        print(f"!! reference JSON unreadable ({ref_p}): {e}"); rc = 1
for x in arm:
    n, h = x.get("name","?"), x.get("hash")
    if not h:
        print(f"   {n:<12} NO HASH -- cannot verify"); rc = 1
    elif ref:
        r = ref.get(n)
        if r is None:
            print(f"   {n:<12} {h[:8]} (absent from reference)")
        elif r != h:
            print(f"   {n:<12} {h[:8]} != ref {r[:8]}  <-- DIVERGED")
            rc = 1
        else:
            print(f"   {n:<12} {h[:8]} == ref")
if not ref and rc == 0:
    print("   results present; no reference given, hashes unverified")
sys.exit(rc)
PY
    ) || RES_RC=1
    printf '%s\n' "$out"
    [ "$RES_RC" = 0 ] || echo "   NOTE: a divergence here may be a real lever effect -- compare against the"
    [ "$RES_RC" = 0 ] || echo "   lever's OWN control, not just base, before calling it corruption."
  fi
fi

if [ "${d:-0}" = "$WANT" ] && [ "$RES_RC" = 0 ]; then
  echo "   $TAG decode window CLEAN"
  exit 0
fi
if [ "${d:-0}" = "$WANT" ]; then
  echo "!! $TAG: decode window clean, but RESULTS failed -- see above."
  exit 1
fi
if [ "${d:-0}" -lt "$WANT" ] 2>/dev/null; then
  echo "!! $TAG DIRTY/TRUNCATED: only $d of $WANT decode requests before the boundary."
  echo "   Either a foreign thinking-request landed mid-decode, or decode_bench lost prompts."
  echo "   DISPLACED != DROPPED: a corrupted arm once scored 5 here, but all 7 of its requests"
  echo "   ran -- the intruder merely arrived before math/prose were issued, pushing them past the"
  echo "   boundary. Check the JSON for missing results before concluding prompts were lost."
else
  echo "!! $TAG CONTAMINATED: $d decode requests, expected $WANT -- foreign traffic on the port."
  echo "   tok/s aggregate unusable; verify_med is still valid (median over our own steps)."
fi
exit 1
