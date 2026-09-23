#!/bin/bash
# GPU job: attention-lane seam gate vs runF_faithful (4 layers of attention weights, <2 min).
Q=/home/flocka/atlas/.gb10-queue
BIN=/home/flocka/atlas/dsv41-attention/target/release/examples/dsv41_attn_seam
LOG=${SEAM_LOG:-/home/flocka/atlas/dsv41-attention/kernels/gb10/deepseek-v4.1/attn/seam_gate.log}
[ -x "$BIN" ] || { echo "DONE rc=127 (binary missing)"; exit 127; }
echo "$(date -u +%FT%TZ) dsv41-attention seam gate vs runF (compress/index/attention, <2 min) QUEUED pid=$$" >> $Q
exec 9>/home/flocka/atlas/.gb10.lock
flock 9
echo "$(date -u +%FT%TZ) dsv41-attention seam gate START pid=$$" >> $Q
"$BIN" $RUN > "$LOG" 2>&1; rc=$?
echo "$(date -u +%FT%TZ) dsv41-attention seam gate END rc=$rc pid=$$" >> $Q
flock -u 9
echo "DONE rc=$rc"
