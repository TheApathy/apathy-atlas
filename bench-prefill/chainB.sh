#!/usr/bin/env bash
W=/home/flocka/atlas/qwen27b-prefill-work/bench
cd $W
until grep -q CHAINA-DONE chainA.log; do sleep 15; done
step() { echo "[$(date -u +%T)] START $1"; }
EX="ATLAS_PREFILL_PROJ_PIPE_M128=1"
step gemm_test2; flock -w 14400 /home/flocka/atlas/.gb10.lock timeout 600 ./gemm_test > runs/gemm_test2.txt 2>&1
step m128b-exact-ssmout; ./measure.sh m128b-exact-ssmout $W/bin/spark-m128b $W/env.p35 $EX ATLAS_SSM_OUT_PREFILL_M128=1 > runs/m128b-exact-ssmout.log 2>&1
step m128b-exact-ctl;    ./measure.sh m128b-exact-ctl    $W/bin/spark-m128b $W/env.p35 $EX > runs/m128b-exact-ctl.log 2>&1
for t in 2,4 0,0 1,1 3,3 5,5 4,2; do
  step fast-tactic-$t; ./measure.sh fast-tactic-${t/,/-} $W/bin/spark-m128b $W/env.p35 ATLAS_PREFILL_PROJ_FAST=1 ATLAS_SSM_OUT_PREFILL_M128=1 ATLAS_FLASHINFER_FFN_TACTIC=$t > runs/fast-tactic-${t/,/-}.log 2>&1
done
step fast-p36-ssmout; ./measure.sh fast-p36-ssmout $W/bin/spark-m128b $W/env.p35 ATLAS_PREFILL_PROJ_FAST=1 ATLAS_SSM_OUT_PREFILL_M128=1 > runs/fast-p36-ssmout.log 2>&1
echo "[$(date -u +%T)] CHAINB-DONE"
