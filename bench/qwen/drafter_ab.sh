#!/usr/bin/env bash
# A/B two DFlash drafters against the same target, same binary, same gates.
#
#   export QWEN_MODEL=/path/to/Qwen3.6-27B-target
#   export QWEN_DRAFT_A=/path/to/drafter-under-test
#   export QWEN_DRAFT_B=/path/to/control-drafter
#   bash bench/qwen/drafter_ab.sh
#
# Three arms, ~8 min each:
#
#   a        QWEN_DRAFT_A
#   b        QWEN_DRAFT_B
#   a-p2     QWEN_DRAFT_A again, unchanged -- this box's A/A noise floor
#
# The A/A arm runs LAST, not adjacent to `a`, so the floor it measures spans the
# same elapsed time as the a-vs-b comparison it is used to judge. A repeat taken
# back-to-back understates drift and makes a null result look significant.
#
# WHAT THIS MEASURES, AND WHAT IT DOES NOT
#
# The headline for a drafter swap is MEAN ACCEPT per verify step, not tok/s.
# tok/s is downstream of acceptance and also of the target's cost, so two
# drafters can tie on throughput while differing on acceptance -- and acceptance
# is the only quantity the drafter alone controls. Both are reported; accept is
# the one to read first.
#
# Completion hashes are reported, NOT graded. On a greedy target the committed
# tokens are supposed to be drafter-independent, and this suite runs at
# temperature 0. If the two arms disagree on a hash, that is a FINDING about the
# verify path and not a fact about drafter quality -- do not average it away.
# The text sidecars are dumped precisely so a divergence can be diffed.
set -uo pipefail

BENCH="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
source "$BENCH/env.sh"

DRAFT_A="${QWEN_DRAFT_A:-}"
DRAFT_B="${QWEN_DRAFT_B:-}"
for v in DRAFT_A DRAFT_B; do
  p="${!v}"
  [ -n "$p" ] || { echo "FATAL: \$QWEN_${v} is unset." >&2; exit 2; }
  [ -f "$p/config.json" ] || { echo "FATAL: \$QWEN_${v}=$p has no config.json." >&2; exit 2; }
done

# A drafter swap is only an A/B if the two checkpoints hook into the target the
# same way. z-lab ships two families under near-identical names: a 5-layer one
# tapping [1,16,31,46,61] with mask_token_id 248070, and a 6-layer one tapping
# [1,10,18,27,35,44,52,61] with 248077. Serving one against a config expecting
# the other does not fail -- it produces a plausible, much lower accept rate,
# which reads exactly like "the other drafter is worse". Compare the wiring
# before spending 25 minutes of GPU on the weights.
python3 - "$DRAFT_A" "$DRAFT_B" <<'PY' || exit 2
import json, sys
def hookup(p):
    c = json.load(open(f"{p}/config.json"))
    d = c.get("dflash_config", {})
    return {
        "layers": c.get("num_hidden_layers"),
        "hidden": c.get("hidden_size"),
        "vocab": c.get("vocab_size"),
        "block_size": d.get("block_size", c.get("block_size")),
        "mask_token_id": d.get("mask_token_id"),
        "target_layer_ids": d.get("target_layer_ids"),
    }
a, b = (hookup(p) for p in sys.argv[1:3])
bad = [k for k in a if a[k] != b[k]]
for k in a:
    mark = "  <-- DIFFERS" if k in bad else ""
    print(f"  {k:<18} A={a[k]}  B={b[k]}{mark}")
if bad:
    print(f"\nFATAL: the two drafters are a different HOOKUP, not different weights: {bad}")
    print("An accept-rate gap here would be about the wiring. Refusing to call this an A/B.")
    sys.exit(1)
print("OK identical hookup -- the only difference between these arms is the weights")
PY

[ -x "$BIN" ] || { echo "FATAL: \$QWEN_BIN=$BIN is not executable." >&2; exit 2; }
DRAFT="$DRAFT_A"
qwen_require_model

# The comparison only needs the configuration held CONSTANT across arms, which
# this script guarantees by construction. Whether that constant is the champion
# is a separate question, and it decides whether the absolute tok/s here may be
# quoted alongside the published table. Check it when the launcher is present,
# and say plainly which of the two claims is supported when it is not.
LAUNCHER="${QWEN_LAUNCHER:-$REPO/local/serve-aeon-27b-dflash.sh}"
if [ -f "$LAUNCHER" ]; then
  bash "$BENCH/verify_gate_parity.sh" \
    || { echo "FATAL: gates differ from the champion launcher -- refusing to run." >&2; exit 7; }
else
  echo "NOTE: $LAUNCHER absent, so gate parity with the champion is UNVERIFIED."
  echo "      The A/B delta below is still valid (both arms share one config)."
  echo "      The absolute tok/s is NOT comparable to the published champion table."
fi

qwen_lock

OUT="$OUT_ROOT/drafter-ab"
# Wipe the whole directory. A run that writes 2 of 3 arms into a directory still
# holding a third from an earlier sitting produces a complete-looking table that
# is silently part-stale -- that has already happened on this project once.
rm -rf "$OUT"
mkdir -p "$OUT"

ARMS="a b a-p2"

echo "=== DFlash drafter A/B ==="
echo "    binary : $BIN"
echo "    target : $MODEL"
echo "    A      : $DRAFT_A"
echo "    B      : $DRAFT_B"
echo "    config : gamma=$GAMMA KV=$KV_DTYPE+${KV_HP_LAYERS}hp seq=$MAX_SEQ_LEN batch=$MAX_BATCH_SIZE util=$GPU_UTIL"
echo

failed=""
run_arm() {
  local tag="$1" draft="$2"
  local log="$OUT/serve-$tag.log"
  echo "--- [$(date +%H:%M:%S)] arm $tag  ($draft)"

  qwen_kill_serves 4
  qwen_champion_env
  DRAFT="$draft"
  qwen_serve "$log"
  qwen_wait_ready "$log" 120

  # The gate under test is the checkpoint PATH, and an env var that was set is
  # not a checkpoint that was loaded. Assert the serve's own resolution line, so
  # a typo'd path that silently fell back cannot be scored as an arm.
  if ! grep -qaF "DFlash: resolving drafter '$draft'" "$log"; then
    echo "    FATAL: serve did not resolve the drafter for arm $tag"
    grep -aim1 "resolving drafter" "$log" | sed 's/^/      got: /'
    failed="$failed $tag"; qwen_kill_serves 0; return 1
  fi
  # And assert the wiring the loader actually installed, not the one the config
  # claimed. Both arms must print the same hookup or the comparison is void.
  grep -aoE "BlockDiffusionDraftHead loaded:.*" "$log" | head -1 \
    | sed -E 's/.*(mask_token_id=[0-9]+, target_layers=\[[^]]*\]).*/      hookup: \1/'

  grep -qa "DFlash speculative decoding: ENABLED" "$log" \
    || { echo "    FATAL: DFlash not enabled ($tag)"; failed="$failed $tag"; qwen_kill_serves 0; return 1; }

  python3 "$BENCH/decode_bench.py" --tag "$tag" --log "$log" \
    --tokens 256 --json-out "$OUT/$tag.json" --dump-text "$OUT/text" 2>&1 | sed 's/^/    /'
  local rc=${PIPESTATUS[0]}
  qwen_kill_serves 0
  [ "$rc" = 0 ] || failed="$failed $tag"
  return "$rc"
}

run_arm a    "$DRAFT_A"
run_arm b    "$DRAFT_B"
run_arm a-p2 "$DRAFT_A"

# Refuse to score an incomplete campaign rather than tabling whatever survived.
missing=""
for arm in $ARMS; do
  [ -s "$OUT/$arm.json" ] || missing="$missing $arm"
done
if [ -n "$missing" ]; then
  echo
  echo "REFUSING TO SCORE -- arms produced no output:$missing"
  echo "Serve logs are in $OUT/serve-<arm>.log; a partial table would read as a full one."
  exit 2
fi

echo
echo "=== table ==="
QWEN_AB_A="$DRAFT_A" QWEN_AB_B="$DRAFT_B" python3 - "$OUT" $ARMS <<'PY'
import json, os, sys
out, arms = sys.argv[1], sys.argv[2:]
data = {}
for a in arms:
    with open(os.path.join(out, f"{a}.json")) as fh:
        data[a] = json.load(fh)

rows = {a: {r["name"]: r for r in data[a]["rows"]} for a in arms}
names = [r["name"] for r in data[arms[0]]["rows"]]
w = max(len(n) for n in names) + 2


def cell(r, key):
    if r is None:
        return "-"
    if key == "tok_s":
        return f"{r['tok_s']:.1f}"
    acc = r.get("accept") or {}
    m = acc.get("mean_accept")
    # "-" for an unmeasured accept, never 0.00: a printed zero here would read
    # as "the drafter proposed nothing" rather than "the scrape found nothing".
    return f"{m:.2f}" if m is not None else "-"


for key, title in (("tok_s", "tok/s"), ("accept", f"mean accept /{data[arms[0]]['gamma']}")):
    print(f"\n-- {title}")
    print("prompt".ljust(w) + "".join(a.rjust(10) for a in arms))
    for n in names:
        print(n.ljust(w) + "".join(cell(rows[a].get(n), key).rjust(10) for a in arms))

# Token-weighted, not the mean of per-prompt rates: an arithmetic mean over six
# tok/s values weights a 130-token row the same as a 256-token one.
tw = {}
for a in arms:
    rs = data[a]["rows"]
    tw[a] = sum(r["completion_tokens"] for r in rs) / sum(r["wall"] for r in rs)

# Accept pooled over verify STEPS, not averaged over prompts, for the same reason.
acc = {}
for a in arms:
    num = den = 0
    for r in data[a]["rows"]:
        s = r.get("accept") or {}
        if s.get("steps") and s.get("mean_accept") is not None:
            num += s["mean_accept"] * s["steps"]
            den += s["steps"]
    acc[a] = num / den if den else None

print("\n-- pooled")
print("token-weighted tok/s".ljust(24) + "".join(f"{tw[a]:.1f}".rjust(10) for a in arms))
print("mean accept".ljust(24) + "".join(
    (f"{acc[a]:.2f}" if acc[a] is not None else "-").rjust(10) for a in arms))

aa_s = abs(tw["a-p2"] - tw["a"]) / tw["a"] * 100
print(f"\nA/A noise floor (a vs a-p2, identical config): {aa_s:.2f}% on tok/s")
if acc["a"] and acc["a-p2"] is not None:
    aa_a = abs(acc["a-p2"] - acc["a"]) / acc["a"] * 100
    print(f"                                               {aa_a:.2f}% on accept")
else:
    aa_a = None
    print("                                               accept UNGRADED on one arm")
# One repeat is one sample. This is a LOWER BOUND on run-to-run spread, not a
# measured tolerance, and a delta that clears it by a little has not cleared
# much. Say so here rather than letting the number be read as a threshold.
print("  (n=1 repeat: a lower bound on spread, not a tolerance. A delta that")
print("   barely clears it is not established -- rerun before quoting one.)")

d_s = (tw["a"] - tw["b"]) / tw["b"] * 100
print(f"\nA vs B: {d_s:+.1f}% tok/s", end="")
if acc["a"] and acc["b"]:
    d_a = (acc["a"] - acc["b"]) / acc["b"] * 100
    print(f", {d_a:+.1f}% accept")
else:
    d_a = None
    print(" (accept UNGRADED)")

verdict = ["tok/s" + (" clears the floor" if abs(d_s) > aa_s
                      else " INSIDE THE FLOOR -- not a difference")]
if d_a is not None and aa_a is not None:
    verdict.append("accept" + (" clears the floor" if abs(d_a) > aa_a
                               else " INSIDE THE FLOOR -- not a difference"))
else:
    # Accept is the headline for a drafter swap, so its absence is a hole in the
    # result, not a footnote. An "UNGRADED" that prints as a missing line reads
    # as a clean verdict on the metric that actually matters.
    verdict.append("accept NOT MEASURED -- this is the headline metric; the "
                   "tok/s verdict alone does not settle the drafter question")
print("  verdict: " + "; ".join(verdict))
print(f"  A = {os.environ['QWEN_AB_A']}")
print(f"  B = {os.environ['QWEN_AB_B']}")

# Hashes are reported, never graded. At temperature 0 with a greedy target the
# committed tokens should not depend on which drafter proposed them, so a
# divergence is a claim about the verify path -- surface it, do not fold it in.
print("\n-- completion hashes (a vs b)")
diff = [n for n in names
        if rows["a"].get(n) and rows["b"].get(n)
        and rows["a"][n]["hash"] != rows["b"][n]["hash"]]
if diff:
    print(f"  {len(diff)}/{len(names)} rows DIFFER between drafters: {', '.join(diff)}")
    print("  Greedy decode should be drafter-independent. Diff the sidecars before")
    print(f"  quoting anything above:  diff -u {out}/text/a.<row>.txt {out}/text/b.<row>.txt")
else:
    print(f"  all {len(names)} rows byte-identical across drafters (expected)")

drift = [n for n in names
         if rows["a"].get(n) and rows["a-p2"].get(n)
         and rows["a"][n]["hash"] != rows["a-p2"][n]["hash"]]
if drift:
    print(f"  !! {len(drift)}/{len(names)} rows are not self-reproducible (a vs a-p2): {', '.join(drift)}")
    print("     Same drafter, same config. Any hash claim above is unsafe until this is explained.")
PY

echo
if [ -n "$failed" ]; then echo "RESULT: arms failed:$failed"; exit 2; fi
echo "RESULT: all three arms ran. Read the accept row before the tok/s row."
