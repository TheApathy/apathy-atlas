#!/bin/bash
# GPU-lock job: which cuBLASLt kernel each dense shape uses (algo attrs + nsys kernel names), <1 GB, <1 min.
Q=/home/flocka/atlas/.gb10-queue
D=/home/flocka/atlas/dsv41-decode/kernels/gb10/deepseek-v4.1/fp8gemm3
FX=/home/flocka/atlas/dsv41-attention/kernels/gb10/deepseek-v4.1/fp8gemm/fx
echo "$(date -u +%FT%TZ) dsv41-decode cuBLAS algo probe (<1 GB, <1 min) QUEUED pid=$$" >> $Q
exec 9>>/home/flocka/atlas/.gb10.lock
flock -w 14400 9 || exit 98
echo "$(date -u +%FT%TZ) dsv41-decode cuBLAS algo probe START (lock held) pid=$$" >> $Q
cd $D && GATE_ALGO=1 nsys profile -t cuda -s none --cpuctxsw=none -f true -o algo ./gate $FX/wq_b $FX/wo_b $FX/w1 $FX/w2 $FX/wq_a $FX/wkv $FX/wo_a_g0 > algo.log 2>&1; rc=$?
echo "$(date -u +%FT%TZ) dsv41-decode cuBLAS algo probe END rc=$rc pid=$$" >> $Q
flock -u 9
nsys stats -q -r cuda_gpu_kern_sum --format csv algo.nsys-rep 2>/dev/null | cut -d, -f2,3,9- | head -20 >> algo.log
echo "DONE rc=$rc"
