#!/usr/bin/env bash
# Reproduce the published Laguna decode table, end to end, and grade it.
#
#   export LAGUNA_MODEL=/path/to/Laguna-S-2.1-NVFP4/snapshot
#   export LAGUNA_DRAFT=/path/to/Laguna-S-2.1-DFlash-NVFP4/snapshot
#   bash bench/laguna/repro_table.sh
#
# Runs four arms on one box, back to back, then checks every completion hash
# against reference_hashes.json. Budget ~25 minutes on GB10-class hardware.
#
#   serial     no speculation at all -- the denominator
#   nogproj    DFlash, g_proj VERIFY GEMV off
#   gproj      DFlash, full production stack (this is the headline arm)
#   gproj-p2   gproj again, unchanged -- measures this box's A/A noise floor
#
# The fourth arm is not redundant. A reproduction that lands 3% away from the
# published number means nothing until you know what 3% costs on your machine,
# and that is not a constant: it belongs to a harness plus a config plus a
# session. Measure it rather than inheriting ours.
#
# Why matching tok/s alone is not a reproduction: two stacks can agree on
# throughput to within a percent and still emit different tokens. The hash check
# is the part that actually constrains the computation; the speed table is the
# part people quote.
set -uo pipefail

BENCH="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "$BENCH/env.sh"
laguna_require_model

# This driver holds the real port lock for the entire campaign, so that a peer
# process cannot land a serve between two arms. Children get a private lock
# path: laguna_lock re-execs fd 9 onto a fresh open-file-description, so a child
# taking the same path would deadlock against this one.
laguna_lock

OUT="$OUT_ROOT/repro"

# Wipe the WHOLE output dir, not just the files about to be written. A run that
# writes 3 of 4 arms into a directory still holding a 4th from an earlier
# sitting produces a complete-looking table that is silently half-stale.
rm -rf "$OUT"
mkdir -p "$OUT"

export LAGUNA_LOCK="$OUT/child.lock"

ARMS="serial nogproj gproj gproj-p2"

echo "=== Laguna decode table reproduction ==="
echo "    repo   : $REPO"
echo "    binary : $BIN"
echo "    model  : $MODEL"
echo "    config : KV=$KV_DTYPE seq=$MAX_SEQ_LEN gamma=$GAMMA batch=$MAX_BATCH_SIZE util=$GPU_UTIL"
echo "    arms   : $ARMS"
echo

failed=""

# run_arm <tag> <expected gproj gate> [KEY=VALUE ...]
#
# Overrides are passed POSITIONALLY, because that is how repro_arm.sh consumes
# them (laguna_apply_overrides "$@"). An earlier version of this driver passed
# them through `env` instead, which looked equivalent and was not: repro_arm.sh
# calls laguna_prod_env FIRST, so the production value was exported over the top
# of the env one, laguna_apply_overrides then received no arguments, and the
# override silently evaporated. The nogproj arm ran WITH g_proj enabled and
# reported 33.2 tok/s -- a completely plausible number, for the wrong arm.
#
# Hence the second parameter. An arm whose gate did not take is not a failed
# arm; it is a DIFFERENT arm wearing this arm's name, and it scores clean.
run_arm() {
  local tag="$1" expect="$2"; shift 2
  local armlog="$OUT/arm-$tag.log"
  echo "--- [$(date +%H:%M:%S)] arm $tag  (expect gproj gate: $expect)"
  bash "$BENCH/repro_arm.sh" "$tag" "$@" 2>&1 | tee "$armlog" | sed 's/^/    /'
  local rc=${PIPESTATUS[0]}

  # repro_arm.sh echoes the gate value it is actually about to serve with.
  # Compare against what this arm MEANT, and treat absence as a mismatch --
  # "the line is missing" must not read the same as "the line agreed".
  local got
  got="$(grep -a -m1 '^=== gproj gate:' "$armlog" \
         | sed -E 's/^=== gproj gate: (.*) ===$/\1/')"
  if [ "$got" != "$expect" ]; then
    echo "    ARM MISCONFIGURED: $tag wanted gproj gate [$expect], serve ran with [${got:-<line absent>}]"
    echo "    This arm is not the arm it claims to be. Not scoring it."
    failed="$failed $tag"
    return 1
  fi

  if [ "$rc" != 0 ]; then
    echo "    ARM FAILED: $tag"
    failed="$failed $tag"
  fi
  return "$rc"
}

# The serial arm takes no speculative path at all, so its gate value is inert --
# it is asserted as 1 only because that is what laguna_prod_env leaves set, and
# an assert that matches reality is worth more than one that encodes intent.
run_arm serial   1         ARM_SERIAL=1
run_arm nogproj  '<unset>' ARM_SERIAL=0 ATLAS_VERIFY_GPROJ_GEMV=
run_arm gproj    1         ARM_SERIAL=0
run_arm gproj-p2 1         ARM_SERIAL=0

# Refuse to score an incomplete campaign. Scoring whatever survived is how a
# sweep whose arms had all died still printed a confident verdict: every
# per-arm number was absent, and absent rendered the same as clean.
missing=""
for a in $ARMS; do
  [ -s "$OUT/$a.json" ] || missing="$missing $a"
done
if [ -n "$missing" ]; then
  echo
  echo "REFUSING TO SCORE -- arms produced no output:$missing"
  echo "Serve logs are in $OUT/serve-<arm>.log; a partial table would read as a full one."
  exit 2
fi

echo
echo "=== hash check vs reference_hashes.json ==="
# gproj-p2 is deliberately graded against the gproj reference: it is the same
# configuration, so any row where it disagrees with gproj is this box's
# nondeterminism, not a reproduction failure.
hash_rc=0
for a in serial nogproj gproj; do
  python3 "$BENCH/check_repro.py" --arm "$a" "$OUT/$a.json" || hash_rc=$?
done
python3 "$BENCH/check_repro.py" --arm gproj "$OUT/gproj-p2.json" || hash_rc=$?

echo
echo "=== table ==="
python3 - "$OUT" $ARMS <<'PY'
import json, os, sys
out, arms = sys.argv[1], sys.argv[2:]
data = {}
for a in arms:
    with open(os.path.join(out, f"{a}.json")) as fh:
        data[a] = json.load(fh)

names = [r["name"] for r in data[arms[0]]["results"]]
w = max(len(n) for n in names) + 2
print("prompt".ljust(w) + "".join(a.rjust(11) for a in arms))
for n in names:
    row = n.ljust(w)
    for a in arms:
        hit = next((r for r in data[a]["results"] if r["name"] == n), None)
        row += (f"{hit['tok_s']:.1f}" if hit else "-").rjust(11)
    print(row)

# Token-weighted, not the mean of per-prompt rates: the arithmetic mean of six
# tok/s values silently weights a 130-token row the same as a 256-token one.
print("-" * (w + 11 * len(arms)))
tw = {}
for a in arms:
    rs = data[a]["results"]
    toks = sum(r["completion_tokens"] for r in rs)
    wall = sum(r["wall"] for r in rs)
    tw[a] = toks / wall if wall else 0.0
print("token-weighted".ljust(w) + "".join(f"{tw[a]:.1f}".rjust(11) for a in arms))

base = tw.get("serial") or 0.0
if base:
    print("vs serial".ljust(w) + "".join(
        (f"{tw[a]/base:.2f}x" if a != "serial" else "-").rjust(11) for a in arms))

# The A/A floor is the only honest yardstick for "did this reproduce".
if "gproj" in tw and "gproj-p2" in tw and tw["gproj"]:
    aa = abs(tw["gproj-p2"] - tw["gproj"]) / tw["gproj"] * 100
    print(f"\nA/A noise floor on this box: {aa:.1f}%  (gproj vs gproj-p2, identical config)")
    print("A difference smaller than this is not a difference.")
PY

echo
if [ -n "$failed" ]; then echo "RESULT: arms failed:$failed"; exit 2; fi
if [ "$hash_rc" -ne 0 ]; then
  echo "RESULT: speed table produced, but HASH CHECK FAILED (rc=$hash_rc)."
  echo "Matching tok/s with mismatched hashes is not a reproduction -- see check_repro.py."
  exit "$hash_rc"
fi
echo "RESULT: all arms ran and every graded row matched the published hashes."
