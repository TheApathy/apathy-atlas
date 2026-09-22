#!/usr/bin/env bash
# ONE GPU window at packed_keep=124 (71.7 GB arena). Lead-approved conditions (2026-09-22):
#  - nothing else resident; MemAvailable ~full before start
#  - arithmetic logged BEFORE the first alloc
#  - a watchdog samples MemAvailable every 100 ms and SIGKILLs the driver (by its own PID)
#    below 6 GB; the low-water mark is logged
#  - the lock is held for the whole window; START only once held; END always with rc
# Usage: keep124_window.sh <label> <driver args...> [ ::: <driver args...> ... ]
# Several driver runs separated by ':::' share ONE lock acquisition (one window).
set -uo pipefail
LABEL="$1"; shift
Q=/home/flocka/atlas/.gb10-queue; LOCK=/home/flocka/atlas/.gb10.lock
BIN=/home/flocka/atlas/dsv41-integration/target/release/examples/dsv41_forward
ABORT_KB=$(( ${ABORT_GB:-6} * 1024 * 1024 ))
NEED_KB=$((100 * 1024 * 1024))
memavail() { awk '/^MemAvailable:/ {print $2}' /proc/meminfo; }
[ -x "$BIN" ] || { echo "FATAL: $BIN missing"; echo "DONE $LABEL rc=99"; exit 99; }

echo "$(date -u +%FT%TZ) dsv41-integrate KEEP124 $LABEL ($*) QUEUED pid=$$" >> "$Q"
exec 9>>"$LOCK"
flock -w 14400 9 || { echo "$(date -u +%FT%TZ) dsv41-integrate KEEP124 $LABEL GAVE UP on the lock" >> "$Q"; echo "DONE $LABEL rc=98"; exit 98; }
echo "$(date -u +%FT%TZ) dsv41-integrate KEEP124 $LABEL window START (lock held) pid=$$" >> "$Q"
end() { echo "$(date -u +%FT%TZ) dsv41-integrate KEEP124 $LABEL window END rc=$1 lowwater_GB=$2 pid=$$" >> "$Q"; flock -u 9; echo "DONE $LABEL rc=$1 lowwater_GB=$2"; exit "$1"; }

# ---- preflight: nothing else resident
# Model-holding things only: a buildkit builder container or a cargo build is not resident on
# the GPU (a build is reported as a warning: it slows GPU work but cannot OOM the device).
others=$(docker ps --format '{{.Names}}' | grep -Ei 'dsv41|atlas|spark|vllm' | grep -v buildkit || true)
procs=$(pgrep -af 'spark serve|bin/spark |v41_engine|capture_ref|examples/dsv41_forward' | grep -v "$$" | grep -v keep124_window | grep -v cargo || true)
builds=$(pgrep -af 'cargo build|rustc|nvcc' | head -3 || true)
[ -n "$builds" ] && echo "WARNING: builds running (slow GPU work, not memory-resident): [${builds}]"
avail=$(memavail)
echo "preflight: MemAvailable $((avail/1024/1024)) GB; containers: [${others}]; procs: [${procs}]"
if [ -n "$others" ] || [ -n "$procs" ] || [ "$avail" -lt "$NEED_KB" ]; then
  echo "PREFLIGHT FAILED (need nothing resident and >= $((NEED_KB/1024/1024)) GB available)"; end 97 "$((avail/1024/1024))"
fi

# ---- arithmetic, before the first alloc
cat <<ARITH
memory plan (GB):
  CB3 arena, 40 layers x 124 experts x 14.45 MB   71.67
  dense store (all non-expert, non-engram, non-mtp) 9.99   (safetensors headers, measured)
  router gate_w widened to fp32, 40 x 384 x 5120    0.31
  fp8 dequant scratch (largest: engram wkv) bf16    0.31
  pass + attention + MoE scratch at T=512           ~0.40
  window rings 40 x 4096 x 512 bf16                 0.17
  CUDA context + cuBLASLt workspace                 ~1.0
  TOTAL                                             ~83.9  vs MemAvailable $((avail/1024/1024))
  headroom after load                               ~$(( avail/1024/1024 - 84 ))  (floor 16; auto-abort below 6)
ARITH

low=$(memavail)
rc=0
run_one() {
  echo "=== run: $*"
  "$BIN" "$@" &
  child=$!
  while kill -0 "$child" 2>/dev/null; do
    m=$(memavail)
    [ "$m" -lt "$low" ] && low=$m
    if [ "$m" -lt "$ABORT_KB" ]; then
      echo "WATCHDOG: MemAvailable $((m/1024)) MB < 6 GB -> SIGKILL pid $child"
      kill -9 "$child"
    fi
    sleep 0.1
  done
  wait "$child"; local r=$?
  echo "=== run rc=$r, low-water so far $((low/1024/1024)) GB"
  [ "$r" -ne 0 ] && rc=$r
  return 0
}
cur=()
for a in "$@"; do
  if [ "$a" = ":::" ]; then run_one "${cur[@]}"; cur=(); else cur+=("$a"); fi
done
[ ${#cur[@]} -gt 0 ] && run_one "${cur[@]}"
echo "low-water MemAvailable: $((low/1024/1024)) GB ($((low/1024)) MB)"
end "$rc" "$((low/1024/1024))"
