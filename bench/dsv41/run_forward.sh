#!/usr/bin/env bash
# Usage: run_forward.sh <label> <driver args...>
# GPU job: QUEUED line, then START only once the lock is HELD, then END with the rc, then a
# DONE line on stdout. The lock is taken on fd 9 in THIS shell, so START/END cannot drift
# from the lock's real lifetime.
set -uo pipefail
LABEL="$1"; shift
Q=/home/flocka/atlas/.gb10-queue; LOCK=/home/flocka/atlas/.gb10.lock
BIN=/home/flocka/atlas/dsv41-integration/target/release/examples/dsv41_forward
[ -x "$BIN" ] || { echo "FATAL: $BIN missing"; echo "DONE $LABEL rc=99"; exit 99; }
echo "$(date -u +%FT%TZ) dsv41-integrate forward driver $LABEL ($*) QUEUED pid=$$" >> "$Q"
exec 9>>"$LOCK"
if ! flock -w 7200 9; then
  echo "$(date -u +%FT%TZ) dsv41-integrate forward driver $LABEL GAVE UP waiting for the lock pid=$$" >> "$Q"
  echo "DONE $LABEL rc=98"; exit 98
fi
echo "$(date -u +%FT%TZ) dsv41-integrate forward driver $LABEL window START (lock held) pid=$$" >> "$Q"
"$BIN" "$@"; rc=$?
echo "$(date -u +%FT%TZ) dsv41-integrate forward driver $LABEL window END rc=$rc pid=$$" >> "$Q"
flock -u 9
echo "DONE $LABEL rc=$rc"
exit $rc
