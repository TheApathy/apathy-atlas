#!/usr/bin/env bash
# runH: runG settings, ALL 40 layers, only the taps needed to teacher-force the whole model forward.
# Differs from runA-E ON PURPOSE: dense attn/wo_a NOT re-quantized to fp4 (DSV41_DENSE_FP4=off),
# bf16 LM head (not fp8), SWA replay ON (layers 21-39 over the last 128 prompt tokens), fp32-P torch
# attention. Everything else = production. 1024 tokens in 2 chunks of 512.
set -uo pipefail
Q=/home/flocka/atlas/.gb10-queue
echo "$(date -u +%FT%TZ) dsv41-integrate runH all-40-layer teacher-forcing capture (runG settings) (python engine, K124 arena ~72GB, ~15min) QUEUED pid=$$" >> "$Q"
export EXTRA_DOCKER_ARGS="-e DSV41_DEV_SKIP_VERIFY=1 -e DSV41_MEM_FLOOR_GB=5.0 -e DSV41_DENSE_FP4=off -e DSV41_HEAD_FMT=bf16 -e DSV41_SWA_REPLAY=1 -e DSV41_PREFILL_CHUNK=512 --mount type=bind,src=/home/flocka/atlas/DSV41_PORT/oracle,dst=/oracle"
# in_image.sh takes the flock itself and holds it for exactly the container's lifetime, so START
# is written by a watcher when the named container is RUNNING (= lock held), not before.
export CONTAINER_NAME="dsv41-integrate-runH-$$"
( until docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; do sleep 2; kill -0 $$ 2>/dev/null || exit 0; done
  echo "$(date -u +%FT%TZ) dsv41-integrate runH window START (lock held, container running) pid=$$" >> "$Q" ) &
/home/flocka/atlas/dsv41-prefill-work/bench/in_image.sh /oracle/capture_ref.py \
  --out /oracle/ref/runH_feed40 --attn torch --tap-logits --gen-tokens 48 \
  --prompt-ids-json /work/bench/prompts_2048.json --prompt-name code_python \
  --max-prompt-tokens 1024 --max-seq 8192 --layers $(seq -s, 0 39) --taps attn_o_pre_inverse_rope,moe_routed,h,pre_mix,attn_out,engram_out
rc=$?
echo "$(date -u +%FT%TZ) dsv41-integrate runH window END rc=$rc" >> "$Q"
echo "DONE runH rc=$rc"
