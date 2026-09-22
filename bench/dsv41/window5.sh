#!/usr/bin/env bash
# Window5 (fd8667599: decode f9229ba49/a8e2fca59, parity 68482c1e1, attention 4efeacf20/fc551ccc1,
# DSpark loader+seed 5b2e74bff). (a) runG KL + 48-token decode + warm prefill on the merged build;
# (b) invariance incl. 1-token tails; (c) serve (runG twice, tool, images); (c2) serve with
# ATLAS_DSV41_DSPARK=1 (drafter load + memory, output unchanged); (d) drop gate with owned allocations.
set -uo pipefail
S=/tmp/claude-1000/-home-flocka-atlas/153d8f07-762d-4884-a8f5-8f7a1a13f1ce/scratchpad
B=/home/flocka/atlas/dsv41-integration/bench/dsv41
I=/home/flocka/atlas/DSV41_PORT/integrate
export ABORT_GB=20
rm -rf "$S/w5a" "$S/inv5"
$B/keep124_window.sh k124_w5 \
  --run runG_replay --layers 40 --path model --moe-real --attn-real --tap-names logits_last --tap-dir $S/w5a --decode 48 --warm-prefill ::: \
  --run runI_plus20 --layers 40 --path model --moe-real --attn-real --tap-names h,logits_last,moe_routed,attn_out --split "512,512,20;1024,20;1043,1;512,512,19,1;1040,4" --tap-dir $S/inv5 \
  > $I/k124_w5.log 2>&1
R=/home/flocka/atlas/DSV41_PORT/oracle/ref/runG_replay/L40.logits_last.000.bin
echo "default: $(python3 $B/logits_vs_oracle.py $R $S/w5a/L40.logits_last.000.bin)" > $I/w5_kl.txt
python3 $B/compare_splits.py $S/inv5 > $I/inv5.txt 2>&1
$B/serve_window.sh serve5 8900 $I/serve5 > $I/serve5.log 2>&1
ATLAS_DSV41_DSPARK=1 $B/serve_window.sh serve5d 8900 $I/serve5d > $I/serve5d.log 2>&1
Q=/home/flocka/atlas/.gb10-queue; exec 9>>/home/flocka/atlas/.gb10.lock
echo "$(date -u +%FT%TZ) dsv41-integrate drop gate keep=6 (+leak control) QUEUED pid=$$" >> $Q
flock -w 7200 9 && {
  echo "$(date -u +%FT%TZ) dsv41-integrate drop gate window START (lock held) pid=$$" >> $Q
  G=/home/flocka/atlas/dsv41-integration/target/release/examples/dsv41_drop_gate
  ATLAS_DSV41_PACKED_KEEP=6 $G > $I/drop_gate5.log 2>&1; r1=$?
  ATLAS_DSV41_PACKED_KEEP=6 $G --control leak > $I/drop_gate5_control.log 2>&1; r2=$?
  echo "$(date -u +%FT%TZ) dsv41-integrate drop gate window END rc=$r1,$r2 pid=$$" >> $Q
  flock -u 9; }
echo "DONE window5 chain"
