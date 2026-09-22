#!/usr/bin/env bash
W=/home/flocka/atlas/qwen27b-prefill-work/bench
cd $W
until grep -q CHAINC-DONE chainC.log; do sleep 15; done
until grep -q "target grep" build-m128d.log; do sleep 15; done
step() { echo "[$(date -u +%T)] START $1"; }
EX="ATLAS_PREFILL_PROJ_PIPE_M128=1 ATLAS_SSM_OUT_PREFILL_M128=1"
GDN=""; grep -q BIT-EXACT gdn/gdn_test1.txt && GDN="ATLAS_GDN_PREFILL_GATECACHE_V2=1"
step m128d-exact-glue; ./measure.sh m128d-exact-glue $W/bin/spark-m128d $W/env.p35 $EX $GDN ATLAS_SSM_RESET_ASYNC=1 ATLAS_DFLASH_CAPTURE_STRIDED=1 > runs/m128d-exact-glue.log 2>&1
step m128d-exact-ctl;  ./measure.sh m128d-exact-ctl  $W/bin/spark-m128d $W/env.p35 $EX $GDN > runs/m128d-exact-ctl.log 2>&1
step gate-glue; ./gate/gate_run.sh glue $W/bin/spark-m128d $W/env.p35 $EX $GDN ATLAS_SSM_RESET_ASYNC=1 ATLAS_DFLASH_CAPTURE_STRIDED=1 > runs/gate-glue.log 2>&1
step fast-final; ./measure.sh fast-final $W/bin/spark-m128d $W/env.p35 ATLAS_PREFILL_PROJ_FAST=1 ATLAS_SSM_OUT_PREFILL_M128=1 $GDN ATLAS_SSM_RESET_ASYNC=1 ATLAS_DFLASH_CAPTURE_STRIDED=1 > runs/fast-final.log 2>&1
echo "[$(date -u +%T)] CHAIND-DONE"
