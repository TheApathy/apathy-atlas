#!/usr/bin/env bash
# py_dist: Python reference samples for the distribution gate (2 prompts x 500 seeds x 2 tokens, T=1).
# Python's own sampler (T + top_p only). Engine floor 12 GB; host watchdog kills the container below 12 GB.
set -uo pipefail
Q=/home/flocka/atlas/.gb10-queue
echo "$(date -u +%FT%TZ) dsv41-integrate py_dist reference (python engine, K124 arena ~72GB, ~15min) QUEUED pid=$$" >> "$Q"
export EXTRA_DOCKER_ARGS="-e DSV41_DEV_SKIP_VERIFY=1 -e DSV41_MEM_FLOOR_GB=12.0 --mount type=bind,src=/home/flocka/atlas/DSV41_PORT/oracle,dst=/oracle"
export CONTAINER_NAME="dsv41-integrate-pydist-$$"
LOW=/tmp/claude-1000/-home-flocka-atlas/153d8f07-762d-4884-a8f5-8f7a1a13f1ce/scratchpad/pydist_low.txt
( until docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; do sleep 2; kill -0 $$ 2>/dev/null || exit 0; done
  echo "$(date -u +%FT%TZ) dsv41-integrate pydist window START (lock held, container running) pid=$$" >> "$Q"
  low=999999999
  while docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; do
    m=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo); [ "$m" -lt "$low" ] && low=$m && echo "$((low/1024)) MB" > "$LOW"
    if [ "$m" -lt $((12 * 1024 * 1024)) ]; then echo "WATCHDOG: MemAvailable $((m/1024)) MB < 12 GB -> docker kill"; docker kill "$CONTAINER_NAME"; fi
    sleep 0.2
  done ) &
/home/flocka/atlas/dsv41-prefill-work/bench/in_image.sh /oracle/py_dist.py
rc=$?
echo "$(date -u +%FT%TZ) dsv41-integrate pydist window END rc=$rc lowwater=$(cat $LOW 2>/dev/null)" >> "$Q"
echo "DONE pydist rc=$rc lowwater=$(cat $LOW 2>/dev/null)"
