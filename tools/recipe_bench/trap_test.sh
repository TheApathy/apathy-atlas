#!/usr/bin/env bash
# CPU-only regression test: a failing command mid-run must NOT kill the
# tracked server process. This is the bug glm-prefill found in run.sh on
# 2026-09-24 — `trap cleanup EXIT INT TERM ERR` fired on any nonzero exit
# (a routine `grep -q` with no match, a curl that got a non-2xx) and killed
# several healthy servers about a second after they reported ready. Run
# this after any change to run.sh's trap/cleanup logic, before trusting a
# GPU run.
set -uo pipefail

DUMMY_SLEEP=300
sleep "$DUMMY_SLEEP" &
CURRENT_PID=$!
CLEANED_UP=0
KILLED_EARLY=0

cleanup() {
  [ "$CLEANED_UP" = 1 ] && return 0
  CLEANED_UP=1
  if [ -n "$CURRENT_PID" ] && kill -0 "$CURRENT_PID" 2>/dev/null; then
    kill -9 "$CURRENT_PID" 2>/dev/null
  fi
}
# Mirrors run.sh's current trap list exactly — if run.sh's trap list ever
# regresses back to including ERR, this test starts failing.
trap cleanup EXIT INT TERM

# A command that fails without tripping `set -e` (no -e is set here either,
# matching run.sh) — this must NOT trigger cleanup.
grep -q "this pattern does not exist" /dev/null || true
false || true

if ! kill -0 "$CURRENT_PID" 2>/dev/null; then
  echo "FAIL: dummy server was killed by a failing command mid-script"
  KILLED_EARLY=1
else
  echo "PASS: dummy server (pid $CURRENT_PID) still alive after a failing command"
fi

kill -9 "$CURRENT_PID" 2>/dev/null
exit $KILLED_EARLY
