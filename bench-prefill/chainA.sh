#!/usr/bin/env bash
# Sequential GPU queue. Each step holds the lock only for its own start->measure->stop cycle.
W=/home/flocka/atlas/qwen27b-prefill-work/bench
cd $W
step() { echo "[$(date -u +%T)] START $1"; }
step m128-exact-base; ./measure.sh m128-exact-base $W/bin/spark-m128 $W/env.p35 > runs/m128-exact-base.log 2>&1
step m128-exact-on;   ./measure.sh m128-exact-on   $W/bin/spark-m128 $W/env.p35 ATLAS_PREFILL_PROJ_PIPE_M128=1 > runs/m128-exact-on.log 2>&1
step nsys; NSYS_OUT=$W/runs/p35-nsys/profile NTRIALS=1 ./measure.sh p35-nsys $W/bin/spark-m128 $W/env.p35 > runs/p35-nsys.log 2>&1
step gate-p35;  ./gate/gate_run.sh p35  $W/bin/spark-m128 $W/env.p35 > runs/gate-p35.log 2>&1
step gate-m128; ./gate/gate_run.sh m128 $W/bin/spark-m128 $W/env.p35 ATLAS_PREFILL_PROJ_PIPE_M128=1 > runs/gate-m128.log 2>&1
step gate-p36;  ./gate/gate_run.sh p36  $W/bin/spark-m128 $W/env.p35 ATLAS_PREFILL_PROJ_FAST=1 > runs/gate-p36.log 2>&1
step peak; flock -w 14400 /home/flocka/atlas/.gb10.lock timeout 300 ./peak > runs/peak.txt 2>&1
step gdn; flock -w 14400 /home/flocka/atlas/.gb10.lock timeout 300 ./gdn/gdn_test 2048 132352 > gdn/gdn_test1.txt 2>&1
echo "[$(date -u +%T)] CHAINA-DONE"
