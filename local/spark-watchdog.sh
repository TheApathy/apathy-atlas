#!/bin/bash
# Crash-safe watchdog for Atlas Spark.
# Runs alongside a serve script and kills it if:
#   - load average rises above LOAD_MAX (default 12.0), OR
#   - free RAM drops below MEM_MIN_GB (default 10), OR
#   - the spark process holds RSS > RSS_MAX_GB (default 90).
#
# These thresholds caught the 2026-05-16 host lockup in postmortem:
# load avg reached 17.69 before the machine became unresponsive. Killing
# spark when load > 12 would have prevented the lockup.
#
# Usage:
#   bash spark-watchdog.sh &   # run in background alongside the server
#   bash spark-watchdog.sh -1  # one-shot status check, no kill
set -euo pipefail

LOAD_MAX=${LOAD_MAX:-12.0}
MEM_MIN_GB=${MEM_MIN_GB:-10}
RSS_MAX_GB=${RSS_MAX_GB:-90}
INTERVAL=${INTERVAL:-5}
ONESHOT=${1:-}

check_once() {
  local load free_gb spark_pid spark_rss_gb
  load=$(awk '{print $1}' /proc/loadavg)
  free_gb=$(free -g | awk '/^Mem:/ {print $7}')
  spark_pid=$(pgrep -x spark | head -1 || true)
  spark_rss_gb=0
  if [ -n "${spark_pid}" ]; then
    spark_rss_gb=$(awk '/VmRSS/ {printf "%.0f", $2/1024/1024}' /proc/${spark_pid}/status 2>/dev/null || echo 0)
  fi

  printf "[watchdog] load=%s free=%sG spark_pid=%s spark_rss=%sG\n" \
    "$load" "$free_gb" "${spark_pid:-none}" "$spark_rss_gb"

  # Trigger kill if any threshold breached AND spark is the likely culprit.
  local should_kill=0
  if awk -v l="$load" -v m="$LOAD_MAX" 'BEGIN {exit !(l > m)}'; then
    echo "[watchdog] TRIP: load $load > $LOAD_MAX" >&2
    should_kill=1
  fi
  if [ "${free_gb:-99}" -lt "$MEM_MIN_GB" ]; then
    echo "[watchdog] TRIP: free $free_gb GB < $MEM_MIN_GB GB" >&2
    should_kill=1
  fi
  if [ "${spark_rss_gb:-0}" -gt "$RSS_MAX_GB" ]; then
    echo "[watchdog] TRIP: spark RSS $spark_rss_gb GB > $RSS_MAX_GB GB" >&2
    should_kill=1
  fi

  if [ "$should_kill" = "1" ] && [ -n "${spark_pid}" ] && [ "$ONESHOT" != "-1" ]; then
    echo "[watchdog] KILLING spark (pid $spark_pid) to prevent host lockup" >&2
    kill -9 "$spark_pid" || true
    return 2
  fi
  return 0
}

if [ "$ONESHOT" = "-1" ]; then
  check_once
  exit $?
fi

echo "[watchdog] started (load_max=$LOAD_MAX mem_min=${MEM_MIN_GB}G rss_max=${RSS_MAX_GB}G interval=${INTERVAL}s)"
while true; do
  check_once || true
  sleep "$INTERVAL"
done
