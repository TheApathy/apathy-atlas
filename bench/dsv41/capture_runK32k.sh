#!/usr/bin/env bash
# runK_32k: the first oracle where the sparse index truly PRUNES (32,768 real code tokens: top-512 of
# ~16K compressed positions at ratio-2 layers). Checkpoint numerics like runG (DENSE_FP4 off, bf16
# head, SWA replay on, torch attention), chunk 2048 (= Atlas default). Taps at L2/L20/L24: topk,
# attn_out, h, n_c + last-token logits + 16 greedy tokens. Engine floor 12 GB (lead: watchdog >= 12 GB);
# a host-side watchdog also kills the container below 12 GB MemAvailable.
set -uo pipefail
Q=/home/flocka/atlas/.gb10-queue
echo "$(date -u +%FT%TZ) dsv41-integrate runK_32k oracle capture (python engine, K124 arena ~72GB, 32K prompt, ~20min) QUEUED pid=$$" >> "$Q"
export EXTRA_DOCKER_ARGS="-e DSV41_DEV_SKIP_VERIFY=1 -e DSV41_MEM_FLOOR_GB=12.0 -e DSV41_DENSE_FP4=off -e DSV41_HEAD_FMT=bf16 -e DSV41_SWA_REPLAY=1 -e DSV41_PREFILL_CHUNK=2048 --mount type=bind,src=/home/flocka/atlas/DSV41_PORT/oracle,dst=/oracle"
export CONTAINER_NAME="dsv41-integrate-runK-$$"
LOW=/tmp/claude-1000/-home-flocka-atlas/153d8f07-762d-4884-a8f5-8f7a1a13f1ce/scratchpad/runK_low.txt
( until docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; do sleep 2; kill -0 $$ 2>/dev/null || exit 0; done
  echo "$(date -u +%FT%TZ) dsv41-integrate runK window START (lock held, container running) pid=$$" >> "$Q"
  low=999999999
  while docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; do
    m=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo); [ "$m" -lt "$low" ] && low=$m && echo "$((low/1024)) MB" > "$LOW"
    if [ "$m" -lt $((12 * 1024 * 1024)) ]; then echo "WATCHDOG: MemAvailable $((m/1024)) MB < 12 GB -> docker kill"; docker kill "$CONTAINER_NAME"; fi
    sleep 0.2
  done ) &
/home/flocka/atlas/dsv41-prefill-work/bench/in_image.sh /oracle/capture_ref.py \
  --out /oracle/ref/runK_32k --attn torch --tap-logits --gen-tokens 16 \
  --prompt-ids-json /oracle/prompts_32k.json --prompt-name code_32k \
  --max-prompt-tokens 32768 --max-seq 34816 --layers 2,20,24 --taps topk,attn_out,h,n_c,logits_last
rc=$?
echo "$(date -u +%FT%TZ) dsv41-integrate runK window END rc=$rc lowwater=$(cat $LOW 2>/dev/null)" >> "$Q"
echo "DONE runK rc=$rc lowwater=$(cat $LOW 2>/dev/null)"
