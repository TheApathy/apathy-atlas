#!/usr/bin/env bash
W=/home/flocka/atlas/qwen27b-prefill-work/bench
cd $W
until grep -q CHAIN2-DONE chain2.log; do sleep 10; done
./measure.sh fast-p36-projfi $W/bin/spark-m128 $W/env.p35 ATLAS_PREFILL_PROJ_FAST=1 ATLAS_PREFILL_PROJ_FLASHINFER=1 > runs/fast-p36-projfi.log 2>&1
./measure.sh fast-p36-ssmfi $W/bin/spark-m128 $W/env.p35 ATLAS_PREFILL_PROJ_FAST=1 ATLAS_PREFILL_SSM_FLASHINFER=1 > runs/fast-p36-ssmfi.log 2>&1
echo CHAIN3-DONE
