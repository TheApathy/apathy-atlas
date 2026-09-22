#!/usr/bin/env bash
W=/home/flocka/atlas/qwen27b-prefill-work/bench
cd $W
until grep -q CHAIN1-DONE chain1.log; do sleep 10; done
./measure.sh m128-exact-base $W/bin/spark-m128 $W/env.p35 > runs/m128-exact-base.log 2>&1
./measure.sh m128-exact-on $W/bin/spark-m128 $W/env.p35 ATLAS_PREFILL_PROJ_PIPE_M128=1 > runs/m128-exact-on.log 2>&1
./gate/gate_run.sh p35 $W/bin/spark-m128 $W/env.p35 > runs/gate-p35.log 2>&1
./gate/gate_run.sh m128 $W/bin/spark-m128 $W/env.p35 ATLAS_PREFILL_PROJ_PIPE_M128=1 > runs/gate-m128.log 2>&1
./gate/gate_run.sh p36 $W/bin/spark-m128 $W/env.p35 ATLAS_PREFILL_PROJ_FAST=1 > runs/gate-p36.log 2>&1
echo CHAIN2-DONE
