#!/usr/bin/env bash
# py_t1_story: Python T=1 sampling reference (lighthouse chat prompt, seeds 1-3, spec off/on), PRODUCTION numerics.
# Python's own sampler (T + top_p only). Engine floor 12 GB; host watchdog kills the container below 12 GB.
set -uo pipefail
Q=/home/flocka/atlas/.gb10-queue
echo "$(date -u +%FT%TZ) dsv41-integrate py_t1_story reference (python engine, K124 arena ~72GB, ~15min) QUEUED pid=$$" >> "$Q"
export EXTRA_DOCKER_ARGS="-e DSV41_DEV_SKIP_VERIFY=1 -e DSV41_MEM_FLOOR_GB=12.0 --mount type=bind,src=/home/flocka/atlas/DSV41_PORT/oracle,dst=/oracle"
export CONTAINER_NAME="dsv41-integrate-pyt1-$$"
LOW=/tmp/claude-1000/-home-flocka-atlas/153d8f07-762d-4884-a8f5-8f7a1a13f1ce/scratchpad/pyt1_low.txt
( until docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; do sleep 2; kill -0 $$ 2>/dev/null || exit 0; done
  echo "$(date -u +%FT%TZ) dsv41-integrate pyt1 window START (lock held, container running) pid=$$" >> "$Q"
  low=999999999
  while docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; do
    m=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo); [ "$m" -lt "$low" ] && low=$m && echo "$((low/1024)) MB" > "$LOW"
    if [ "$m" -lt $((12 * 1024 * 1024)) ]; then echo "WATCHDOG: MemAvailable $((m/1024)) MB < 12 GB -> docker kill"; docker kill "$CONTAINER_NAME"; fi
    sleep 0.2
  done ) &
/home/flocka/atlas/dsv41-prefill-work/bench/in_image.sh /oracle/py_t1_story.py
rc=$?
echo "$(date -u +%FT%TZ) dsv41-integrate pyt1 window END rc=$rc lowwater=$(cat $LOW 2>/dev/null)" >> "$Q"
echo "DONE pyt1 rc=$rc lowwater=$(cat $LOW 2>/dev/null)"
