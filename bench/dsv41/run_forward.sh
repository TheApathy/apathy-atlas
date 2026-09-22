#!/usr/bin/env bash
# Usage: run_forward.sh <label> <driver args...>   (GPU job; takes the lock; DONE line at the end)
set -uo pipefail
LABEL="$1"; shift
Q=/home/flocka/atlas/.gb10-queue; LOCK=/home/flocka/atlas/.gb10.lock
BIN=/home/flocka/atlas/dsv41-integration/target/release/examples/dsv41_forward
[ -x "$BIN" ] || { echo "FATAL: $BIN missing"; echo "DONE rc=99"; exit 99; }
echo "$(date -u +%FT%TZ) dsv41-integrate forward driver $LABEL ($*) QUEUED pid=$$" >> "$Q"
flock -w 7200 "$LOCK" bash -c '
  echo "$(date -u +%FT%TZ) dsv41-integrate forward driver '"$LABEL"' window START pid=$$" >> '"$Q"'
  '"$BIN"' "$@"; rc=$?
  echo "$(date -u +%FT%TZ) dsv41-integrate forward driver '"$LABEL"' window END rc=$rc" >> '"$Q"'
  exit $rc' _ "$@"
rc=$?
echo "DONE $LABEL rc=$rc"
exit $rc
