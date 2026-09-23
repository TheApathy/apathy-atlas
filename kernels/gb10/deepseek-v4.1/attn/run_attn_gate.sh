#!/bin/bash
# GPU job: sparse attention gate on the REAL (runC_2048 L2) and SYNTHETIC (all-masked row) fixtures.
Q=/home/flocka/atlas/.gb10-queue
DIR=/home/flocka/atlas/dsv41-attention/kernels/gb10/deepseek-v4.1/attn
BIN=$DIR/sparse_attn
[ -x "$BIN" ] || { echo "DONE rc=127 (binary missing)"; exit 127; }
echo "$(date -u +%FT%TZ) dsv41-attention sparse_attn gate + production entry (2 fixtures, <1 min) QUEUED pid=$$" >> $Q
exec 9>/home/flocka/atlas/.gb10.lock
flock 9
echo "$(date -u +%FT%TZ) dsv41-attention sparse_attn gate START pid=$$" >> $Q
cd $DIR/real && $BIN > $DIR/attn_gate_real.log 2>&1; rc1=$?
cd $DIR/sk1 && $BIN > $DIR/attn_gate_synth.log 2>&1; rc2=$?
echo "$(date -u +%FT%TZ) dsv41-attention sparse_attn gate END rc=$rc1,$rc2 pid=$$" >> $Q
flock -u 9
echo "DONE rc_real=$rc1 rc_synth=$rc2"
