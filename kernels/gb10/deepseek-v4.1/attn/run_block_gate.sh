#!/bin/bash
# GPU job: whole attention block (integrator's projections + this lane's core) through the
# driver on runH_feed40, layers 0..20, MoE fed. Arm A: core fed from the capture (baseline);
# arm B: the real core. Positive DONE line with both exit codes.
Q=/home/flocka/atlas/.gb10-queue
BIN=/home/flocka/atlas/dsv41-attention/target/release/examples/dsv41_forward
OUT=/tmp/claude-1000/-home-flocka-atlas/153d8f07-762d-4884-a8f5-8f7a1a13f1ce/scratchpad/block
[ -x "$BIN" ] || { echo "DONE rc=127 (binary missing)"; exit 127; }
rm -rf $OUT; mkdir -p $OUT/fed $OUT/real
echo "$(date -u +%FT%TZ) dsv41-attention whole-block gate (21 layers dense, MoE fed, 2 arms, ~3 min) QUEUED pid=$$" >> $Q
exec 9>/home/flocka/atlas/.gb10.lock
flock 9
echo "$(date -u +%FT%TZ) dsv41-attention whole-block gate START pid=$$" >> $Q
$BIN --run runH_feed40 --layers 21 --feed attn-core,moe --engram live --core fed  --tap-dir $OUT/fed  > $OUT/fed.log 2>&1; rc1=$?
$BIN --run runH_feed40 --layers 21 --feed attn-core,moe --engram live --core real --tap-dir $OUT/real > $OUT/real.log 2>&1; rc2=$?
echo "$(date -u +%FT%TZ) dsv41-attention whole-block gate END rc=$rc1,$rc2 pid=$$" >> $Q
flock -u 9
echo "DONE rc_fed=$rc1 rc_real=$rc2"
