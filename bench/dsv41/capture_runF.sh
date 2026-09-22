#!/usr/bin/env bash
# runF: CHECKPOINT-FAITHFUL oracle for the Rust port (dsv41-integrate).
# Differs from runA-E ON PURPOSE: dense attn/wo_a NOT re-quantized to fp4 (DSV41_DENSE_FP4=off),
# bf16 LM head (not fp8), NO SWA replay (all 40 layers over every prompt token), fp32-P torch
# attention. Everything else = production. 1024 tokens in 2 chunks of 512.
set -uo pipefail
Q=/home/flocka/atlas/.gb10-queue
echo "$(date -u +%FT%TZ) dsv41-integrate runF faithful oracle capture (python engine, K124 arena ~72GB, ~15min) QUEUED pid=$$" >> "$Q"
export EXTRA_DOCKER_ARGS="-e DSV41_DEV_SKIP_VERIFY=1 -e DSV41_MEM_FLOOR_GB=5.0 -e DSV41_DENSE_FP4=off -e DSV41_HEAD_FMT=bf16 -e DSV41_SWA_REPLAY=0 -e DSV41_PREFILL_CHUNK=512 --mount type=bind,src=/home/flocka/atlas/DSV41_PORT/oracle,dst=/oracle"
# in_image.sh takes the flock itself; the START line is written from inside by a wrapper-free echo
# just before, so START means "about to contend for the lock".
echo "$(date -u +%FT%TZ) dsv41-integrate runF window START (waiting on flock inside in_image.sh) pid=$$" >> "$Q"
/home/flocka/atlas/dsv41-prefill-work/bench/in_image.sh /oracle/capture_ref.py \
  --out /oracle/ref/runF_faithful --attn torch --tap-logits --gen-tokens 48 \
  --prompt-ids-json /work/bench/prompts_2048.json --prompt-name code_python \
  --max-prompt-tokens 1024 --max-seq 8192 --layers 0,1,2,8,14,20,21,30,39
rc=$?
echo "$(date -u +%FT%TZ) dsv41-integrate runF window END rc=$rc" >> "$Q"
echo "DONE runF rc=$rc"
