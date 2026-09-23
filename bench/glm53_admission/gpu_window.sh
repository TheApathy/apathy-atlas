#!/usr/bin/env bash
# One exclusive GB10 GPU window for the GLM admission lane.
# Usage: gpu_window.sh <label> <max-seconds> <command...>
# Queues, takes the lock EXCLUSIVELY, preflights AFTER the lock (no model process,
# MemAvailable >= 100 GB), runs the command in its own process group without the
# lock fd, SIGKILLs only that group if MemAvailable drops below 6 GB, logs the
# low-water mark, and waits for the group to exit before logging END.
set -uo pipefail
LABEL=${1:?label}; MAX_SECS=${2:?max seconds}; shift 2
Q=/home/flocka/atlas/.gb10-queue; LOCK=/home/flocka/atlas/.gb10.lock
LANE=glm-admission
log() { echo "$(date -u +%FT%TZ) $LANE $LABEL $* pid=$$" >> "$Q"; echo "[$(date +%T)] $*" >&2; }
mem_kb() { awk '/MemAvailable/{print $2}' /proc/meminfo; }

log "QUEUED ($*)"
exec 9>>"$LOCK"
flock -w 14400 9 || { log "gave up on lock"; exit 98; }
log "START (exclusive lock held)"

for pid in $(ls /proc | grep -E '^[0-9]+$'); do
  exe=$(readlink -f /proc/$pid/exe 2>/dev/null) || continue
  case "$(basename "$exe")" in
    spark-dashboard) ;;
    spark|spark-*|llama-server|llama-cli|vllm) log "END PREFLIGHT-FAIL model process $pid $exe"; exit 2;;
    python*)
      # Match on the python process itself, never on this script's own argv.
      if tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | grep -q 'exl3_reference.py'; then
        log "END PREFLIGHT-FAIL reference process alive $pid"; exit 2
      fi;;
  esac
done
[ "$(mem_kb)" -ge $((100*1024*1024)) ] || { log "END PREFLIGHT-FAIL MemAvailable $(mem_kb) kB < 100 GB"; exit 2; }

CHILD=""
LOW=$(mem_kb)
finish() {
  local rc=$?
  if [ -n "$CHILD" ] && kill -0 "$CHILD" 2>/dev/null; then
    kill -INT -- "-$CHILD" 2>/dev/null
    for _ in $(seq 1 60); do kill -0 "$CHILD" 2>/dev/null || break; sleep 1; done
    kill -9 -- "-$CHILD" 2>/dev/null
    while kill -0 "$CHILD" 2>/dev/null; do sleep 1; done
  fi
  log "END rc=$rc mem_available_low_water_kb=$LOW"
}
trap finish EXIT
trap 'exit 130' INT TERM

setsid "$@" 9>&- &
CHILD=$!
DEADLINE=$(( $(date +%s) + MAX_SECS ))
while kill -0 "$CHILD" 2>/dev/null; do
  m=$(mem_kb); [ "$m" -lt "$LOW" ] && LOW=$m
  if [ "$m" -lt $((6*1024*1024)) ]; then
    log "WATCHDOG MemAvailable ${m} kB < 6 GB: SIGKILL group $CHILD"
    kill -9 -- "-$CHILD" 2>/dev/null
  fi
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then
    log "TIMEOUT after ${MAX_SECS}s: stopping group $CHILD"
    kill -INT -- "-$CHILD" 2>/dev/null; sleep 20; kill -9 -- "-$CHILD" 2>/dev/null
  fi
  sleep 1
done
wait "$CHILD"; RC=$?
CHILD=""
exit $RC
