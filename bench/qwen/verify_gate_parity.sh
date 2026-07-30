#!/usr/bin/env bash
# Assert that bench/qwen/env.sh reproduces the champion launcher's gate set
# EXACTLY -- same names, same values, nothing extra, nothing missing.
#
# Why this exists
# ---------------
# `local/serve-aeon-27b-dflash.sh` is the script that actually produced the
# published numbers. `qwen_champion_env()` in env.sh is a second, hand-written
# copy of the same configuration, kept separate so the bench harness can run
# without the launcher's box-specific preflight. Two copies of one truth drift.
#
# They already did. An earlier version of env.sh was checked "21 gates, 0
# missing" against the launcher's COMMITTED form while the configuration that
# produced the numbers lived in its UNCOMMITTED form -- so the check compared
# the copy against a stale reference, agreed with itself, and reported clean
# while eleven gates were missing. That is the failure this file is built to
# make impossible: a comparison that cannot see a difference reports the same
# "0" as a comparison that found none.
#
# So the launcher is the single source of truth and this script re-derives from
# it every run. Both sides are checked non-empty first (see NON-VACUITY below),
# because the cheapest way to pass a diff is to diff nothing against nothing.
#
#   bash bench/qwen/verify_gate_parity.sh          # compare, exit 1 on drift
#   bash bench/qwen/verify_gate_parity.sh --self-test  # prove the diff fires
#
set -uo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO="$(cd "$BENCH_DIR/../.." && pwd)"
LAUNCHER="${QWEN_LAUNCHER:-$REPO/local/serve-aeon-27b-dflash.sh}"
ENVSH="$BENCH_DIR/env.sh"

# A configuration this small cannot legitimately shrink to a handful of gates.
# If either extraction returns fewer, the extractor broke -- a regex that stops
# matching produces an empty set, and an empty set diffs clean against another
# empty set. Tuned well below the current 32 so it flags breakage, not edits.
MIN_GATES=25

[ -f "$LAUNCHER" ] || { echo "FATAL: launcher not found: $LAUNCHER" >&2; exit 2; }
[ -f "$ENVSH" ]    || { echo "FATAL: env.sh not found: $ENVSH" >&2; exit 2; }

# --- side A: the launcher -----------------------------------------------------
# Only column-1 `export ATLAS_*` lines. Gates nested inside an `if` block are
# indented and are deliberately excluded: they belong to opt-in modes
# (ATLAS_DFLASH_PORTFOLIO, ATLAS_DFLASH_FREE_SLOTS) that default off and are
# not part of the champion path.
#
# The file is PARSED, never sourced. Sourcing it would run a preflight that
# kills any running serve.
extract_launcher() {
  grep -E '^export ATLAS_[A-Z0-9_]+=' "$1" \
    | sed -E 's/^export ([A-Z0-9_]+)="?\$\{\1:-([^}]*)\}"?[[:space:]]*$/\1=\2/' \
    | sort
}

# --- side B: env.sh -----------------------------------------------------------
# Executed, not parsed: what matters is what qwen_champion_env() actually leaves
# in the environment, which is not always what its source lines appear to say.
#
# `env -i` because the ambient shell may already export ATLAS_* variables. Left
# in place they would be indistinguishable from gates the function set, and a
# missing gate would then be masked by whatever the operator happened to have
# exported -- passing on the one machine where it was written and nowhere else.
#
# QWEN_STEP_TIMING=0 is pinned to the champion default. At 1, env.sh adds
# ATLAS_DFLASH_STEP_TIMING, which the launcher does not set; that divergence is
# intentional and documented, so comparing against it would be a false failure.
extract_envsh() {
  env -i PATH="$PATH" QWEN_STEP_TIMING=0 bash -c '
    set -uo pipefail
    source "$1" || exit 9
    qwen_champion_env || exit 9
    env | grep -E "^ATLAS_[A-Z0-9_]+=" | sort
  ' _ "$1"
}

A="$(extract_launcher "$LAUNCHER")"
B="$(extract_envsh "$ENVSH")"

# --- NON-VACUITY --------------------------------------------------------------
na=$(printf '%s\n' "$A" | grep -c .)
nb=$(printf '%s\n' "$B" | grep -c .)

# A line the sed could not rewrite still contains `${`. Left alone it would be
# compared verbatim and reported as drift against a gate that is actually fine,
# which trains whoever sees it to ignore this script. Fail on the extractor
# instead of on its output.
if printf '%s\n' "$A" | grep -q '[$]{'; then
  echo "FATAL: launcher extractor failed to normalize these lines:" >&2
  printf '%s\n' "$A" | grep '[$]{' >&2
  exit 2
fi
if [ "$na" -lt "$MIN_GATES" ] || [ "$nb" -lt "$MIN_GATES" ]; then
  echo "FATAL: extraction looks broken (launcher=$na, env.sh=$nb, expected >=$MIN_GATES)." >&2
  echo "  Not reporting parity: an empty set matches an empty set." >&2
  exit 2
fi

# --- self-test ----------------------------------------------------------------
# Plants a known divergence and confirms the comparator reports it. Without
# this, a green run proves the two sides agree OR that the comparison silently
# did nothing, and those are not the same result.
if [ "${1:-}" = "--self-test" ]; then
  poisoned="$(printf '%s\n' "$A" | grep -v '^ATLAS_THINK_SPEC=')"
  if diff <(printf '%s\n' "$poisoned") <(printf '%s\n' "$B") >/dev/null 2>&1; then
    echo "SELF-TEST FAILED: comparator did not notice a deleted gate." >&2
    exit 3
  fi
  echo "self-test ok: comparator detects a planted single-gate divergence"
  echo "  (launcher=$na gates, env.sh=$nb gates, both above the $MIN_GATES floor)"
fi

# --- compare ------------------------------------------------------------------
if diff -u <(printf '%s\n' "$A") <(printf '%s\n' "$B") \
     --label "launcher: ${LAUNCHER#"$REPO"/}" --label "env.sh: qwen_champion_env()"; then
  echo "gate parity ok: $na gates, launcher == env.sh"
  exit 0
fi

cat >&2 <<EOF

FATAL: bench/qwen/env.sh has drifted from the launcher that produced the
published numbers. Lines prefixed '-' are in the launcher and missing from
env.sh; '+' are set by env.sh and absent from the launcher.

The launcher is authoritative. Reconcile env.sh to it -- and if a gate really
did change, change it there first.
EOF
exit 1
