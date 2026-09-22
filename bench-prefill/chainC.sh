#!/usr/bin/env bash
W=/home/flocka/atlas/qwen27b-prefill-work/bench
cd $W
until grep -q CHAINB-DONE chainB.log; do sleep 15; done
step() { echo "[$(date -u +%T)] START $1"; }
EX="ATLAS_PREFILL_PROJ_PIPE_M128=1 ATLAS_SSM_OUT_PREFILL_M128=1"
# standalone GDN v2 exactness gate must pass before the server A/B
if ! grep -q BIT-EXACT gdn/gdn_test1.txt; then echo "GDN standalone not exact; skipping v2 server runs"; else
step m128c-exact-gdnv2; ./measure.sh m128c-exact-gdnv2 $W/bin/spark-m128c $W/env.p35 $EX ATLAS_GDN_PREFILL_GATECACHE_V2=1 > runs/m128c-exact-gdnv2.log 2>&1
step m128c-exact-ctl;   ./measure.sh m128c-exact-ctl   $W/bin/spark-m128c $W/env.p35 $EX > runs/m128c-exact-ctl.log 2>&1
step gate-final; ./gate/gate_run.sh final $W/bin/spark-m128c $W/env.p35 $EX ATLAS_GDN_PREFILL_GATECACHE_V2=1 > runs/gate-final.log 2>&1
fi
step profile-final; NTRIALS=1 ./measure.sh m128c-profile $W/bin/spark-m128c $W/env.p35 $EX ATLAS_GDN_PREFILL_GATECACHE_V2=1 ATLAS_PROFILE_FIRST=1 > runs/m128c-profile.log 2>&1
echo "[$(date -u +%T)] CHAINC-DONE"
