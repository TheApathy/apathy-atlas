#!/usr/bin/env bash
# (1) keep=124: driver runG prefill (h taps) + 48-token decode [post-merge text regression], then
#     [512,512,20] vs [1024,20] byte test (routed-MoE invariance after engine2's 6699b381c);
# (2) serve at the same build with model h taps + a 1-token runG request (prefill only), the
#     48-token one, and parity's tool+thinking request;
# (3) keep=6 drop gate + its leak control, one lock.
# ATLAS_LOG_PINNED_ALGO=1 everywhere: pinned cuBLASLt algos + stream + ATLAS_* env per process.
set -uo pipefail
S=/tmp/claude-1000/-home-flocka-atlas/153d8f07-762d-4884-a8f5-8f7a1a13f1ce/scratchpad
B=/home/flocka/atlas/dsv41-integration/bench/dsv41
I=/home/flocka/atlas/DSV41_PORT/integrate
export ABORT_GB=20 ATLAS_LOG_PINNED_ALGO=1
rm -rf "$S/drvG" "$S/srvG" "$S/inv3"
$B/keep124_window.sh k124_drv_postmerge \
  --run runG_replay --layers 40 --path model --moe-real --attn-real --tap-names h --tap-dir $S/drvG --decode 48 ::: \
  --run runI_plus20 --layers 40 --path model --moe-real --attn-real --tap-names h,logits_last,moe_routed --split "512,512,20;1024,20" --tap-dir $S/inv3 \
  > $I/k124_drv_postmerge.log 2>&1
python3 $B/compare_splits.py $S/inv3 > $I/inv3.txt 2>&1
ATLAS_DSV41_TAP_DIR=$S/srvG ATLAS_DSV41_TAP_NAMES=h $B/serve_window.sh serve3 8900 $I/serve3 > $I/serve3.log 2>&1
# drop gate (keep=6: ~17 GB, not a keep=124 window) under the lock
Q=/home/flocka/atlas/.gb10-queue; exec 9>>/home/flocka/atlas/.gb10.lock
echo "$(date -u +%FT%TZ) dsv41-integrate drop gate keep=6 (+leak control) QUEUED pid=$$" >> $Q
flock -w 7200 9 && {
  echo "$(date -u +%FT%TZ) dsv41-integrate drop gate window START (lock held) pid=$$" >> $Q
  G=/home/flocka/atlas/dsv41-integration/target/release/examples/dsv41_drop_gate
  ATLAS_DSV41_PACKED_KEEP=6 $G > $I/drop_gate.log 2>&1; r1=$?
  ATLAS_DSV41_PACKED_KEEP=6 $G --control leak > $I/drop_gate_control.log 2>&1; r2=$?
  echo "$(date -u +%FT%TZ) dsv41-integrate drop gate window END rc=$r1,$r2 pid=$$" >> $Q
  flock -u 9; }
echo "DONE serve-bisect chain"
