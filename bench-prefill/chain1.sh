#!/usr/bin/env bash
W=/home/flocka/atlas/qwen27b-prefill-work/bench
cd $W
./measure.sh p35-sealed-control $W/bin/spark-p35-sealed $W/env.p35 > runs/p35-sealed-control.log 2>&1
./measure.sh p36-sealed-control $W/bin/spark-p35-sealed $W/env.p35 ATLAS_PREFILL_PROJ_FAST=1 > runs/p36-sealed-control.log 2>&1
NSYS_OUT=$W/runs/p35-nsys/profile NTRIALS=1 ./measure.sh p35-nsys $W/bin/spark-p35-sealed $W/env.p35 > runs/p35-nsys.log 2>&1
flock -w 14400 /home/flocka/atlas/.gb10.lock ./peak > runs/peak.txt 2>&1
echo CHAIN1-DONE
