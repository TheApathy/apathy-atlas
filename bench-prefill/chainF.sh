#!/usr/bin/env bash
W=/home/flocka/atlas/qwen27b-prefill-work/bench
cd $W
step() { echo "[$(date -u +%T)] START $1"; }
EX="ATLAS_PREFILL_PROJ_PIPE_M128=1 ATLAS_SSM_OUT_PREFILL_M128=1 ATLAS_SSM_RESET_ASYNC=1 ATLAS_DFLASH_CAPTURE_STRIDED=1"
step final-repeat; ./measure.sh final-repeat $W/bin/spark-m128d $W/env.p35 $EX > runs/final-repeat.log 2>&1
step final-nsys; mkdir -p runs/final-nsys; NSYS_OUT=$W/runs/final-nsys/profile NTRIALS=1 ./measure.sh final-nsys $W/bin/spark-m128d $W/env.p35 $EX > runs/final-nsys.log 2>&1
step p35-repeat; ./measure.sh p35-repeat $W/bin/spark-p35-sealed $W/env.p35 > runs/p35-repeat.log 2>&1
echo "[$(date -u +%T)] CHAINF-DONE"
