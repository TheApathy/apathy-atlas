#!/bin/bash
# GPU-lock job: v3 fused FP8 GEMM sweep + gate (7 shapes x M=512/2048, <2 GB, ~2 min).
Q=/home/flocka/atlas/.gb10-queue
D=/home/flocka/atlas/dsv41-decode/kernels/gb10/deepseek-v4.1/fp8gemm3
FX=/home/flocka/atlas/dsv41-attention/kernels/gb10/deepseek-v4.1/fp8gemm/fx
echo "$(date -u +%FT%TZ) dsv41-decode fp8-gemm v3 sweep (7 shapes x 2 M, <2 GB, ~2 min, timing) QUEUED pid=$$" >> $Q
exec 9>>/home/flocka/atlas/.gb10.lock
flock -w 14400 9 || { echo "$(date -u +%FT%TZ) dsv41-decode fp8-gemm v3 GAVE UP pid=$$" >> $Q; exit 98; }
echo "$(date -u +%FT%TZ) dsv41-decode fp8-gemm v3 sweep START (lock held) pid=$$" >> $Q
cd $D && ./gate $FX/wq_b $FX/wo_b $FX/w1 $FX/w2 $FX/wq_a $FX/wkv $FX/wo_a_g0 > gate.log 2>&1; rc=$?
echo "$(date -u +%FT%TZ) dsv41-decode fp8-gemm v3 sweep END rc=$rc pid=$$" >> $Q
flock -u 9
echo "DONE rc=$rc"
