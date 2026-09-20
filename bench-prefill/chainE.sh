#!/usr/bin/env bash
W=/home/flocka/atlas/qwen27b-prefill-work/bench
cd $W
until grep -q CHAIND-DONE chainD.log; do sleep 15; done
until grep -q "target grep\|^error" build-m128e.log; do sleep 15; done
grep -q "target grep" build-m128e.log || { echo "m128e build failed"; exit 1; }
step() { echo "[$(date -u +%T)] START $1"; }
step gemm_test3; flock -w 14400 /home/flocka/atlas/.gb10.lock timeout 600 ./gemm_test > runs/gemm_test3.txt 2>&1
grep -q ALL-EXACT runs/gemm_test3.txt || { echo "gemm_test3 not exact"; exit 1; }
GDN=""; grep -q BIT-EXACT gdn/gdn_test1.txt && GDN="ATLAS_GDN_PREFILL_GATECACHE_V2=1"
EX="ATLAS_PREFILL_PROJ_PIPE_M128=1 $GDN ATLAS_SSM_RESET_ASYNC=1 ATLAS_DFLASH_CAPTURE_STRIDED=1"
step m128e-exact-w8;  ./measure.sh m128e-exact-w8  $W/bin/spark-m128e $W/env.p35 $EX ATLAS_PREFILL_FP8_W8=1 > runs/m128e-exact-w8.log 2>&1
step m128e-exact-ctl; ./measure.sh m128e-exact-ctl $W/bin/spark-m128e $W/env.p35 $EX > runs/m128e-exact-ctl.log 2>&1
step gate-w8; ./gate/gate_run.sh w8 $W/bin/spark-m128e $W/env.p35 $EX ATLAS_PREFILL_FP8_W8=1 > runs/gate-w8.log 2>&1
step fast-w8; ./measure.sh fast-w8 $W/bin/spark-m128e $W/env.p35 ATLAS_PREFILL_PROJ_FAST=1 $GDN ATLAS_SSM_RESET_ASYNC=1 ATLAS_DFLASH_CAPTURE_STRIDED=1 ATLAS_PREFILL_FP8_W8=1 > runs/fast-w8.log 2>&1
step gate-fast-w8; ./gate/gate_run.sh fast-w8 $W/bin/spark-m128e $W/env.p35 ATLAS_PREFILL_PROJ_FAST=1 $GDN ATLAS_SSM_RESET_ASYNC=1 ATLAS_DFLASH_CAPTURE_STRIDED=1 ATLAS_PREFILL_FP8_W8=1 > runs/gate-fast-w8.log 2>&1
echo "[$(date -u +%T)] CHAINE-DONE"
