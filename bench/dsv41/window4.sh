#!/usr/bin/env bash
# Next window (lead, 2026-09-22): attention2's a88c4c72b merged. (a) COMP_CUBLAS 0 (SIMT default)
# vs 1 on runG: step-0 logits vs oracle + 48-token greedy; default also warm prefill;
# (a2) ATLAS_DSV41_ATTN_SPLIT=0 vs default decode arm (split-KV e2e gate); (b) invariance byte test;
# (c) serve (runG 48 twice: text == driver default arm, tok/s >= 12; images; int effort); (d) drop gate.
set -uo pipefail
S=/tmp/claude-1000/-home-flocka-atlas/153d8f07-762d-4884-a8f5-8f7a1a13f1ce/scratchpad
B=/home/flocka/atlas/dsv41-integration/bench/dsv41
I=/home/flocka/atlas/DSV41_PORT/integrate
export ABORT_GB=20
rm -rf "$S/cc0" "$S/cc1" "$S/inv4"
$B/keep124_window.sh k124_w4 \
  --run runG_replay --layers 40 --path model --moe-real --attn-real --env ATLAS_DSV41_COMP_CUBLAS=0 --tap-names logits_last --tap-dir $S/cc0 --decode 48 --warm-prefill ::: \
  --run runG_replay --layers 40 --path model --moe-real --attn-real --env ATLAS_DSV41_COMP_CUBLAS=1 --tap-names logits_last --tap-dir $S/cc1 --decode 48 ::: \
  --run runG_replay --layers 40 --path model --moe-real --attn-real --env ATLAS_DSV41_ATTN_SPLIT=0 --decode 48 ::: \
  --run runI_plus20 --layers 40 --path model --moe-real --attn-real --tap-names h,logits_last,moe_routed,attn_out --split "512,512,20;1024,20;500,544" --tap-dir $S/inv4 \
  > $I/k124_w4.log 2>&1
R=/home/flocka/atlas/DSV41_PORT/oracle/ref/runG_replay/L40.logits_last.000.bin
{ echo "COMP_CUBLAS=0: $(python3 $B/logits_vs_oracle.py $R $S/cc0/L40.logits_last.000.bin)"
  echo "COMP_CUBLAS=1: $(python3 $B/logits_vs_oracle.py $R $S/cc1/L40.logits_last.000.bin)"; } > $I/w4_comp_cublas.txt
python3 $B/compare_splits.py $S/inv4 > $I/inv4.txt 2>&1
$B/serve_window.sh serve4 8900 $I/serve4 > $I/serve4.log 2>&1
Q=/home/flocka/atlas/.gb10-queue; exec 9>>/home/flocka/atlas/.gb10.lock
echo "$(date -u +%FT%TZ) dsv41-integrate drop gate keep=6 (+leak control, vision loaded) QUEUED pid=$$" >> $Q
flock -w 7200 9 && {
  echo "$(date -u +%FT%TZ) dsv41-integrate drop gate window START (lock held) pid=$$" >> $Q
  G=/home/flocka/atlas/dsv41-integration/target/release/examples/dsv41_drop_gate
  ATLAS_DSV41_PACKED_KEEP=6 $G > $I/drop_gate4.log 2>&1; r1=$?
  ATLAS_DSV41_PACKED_KEEP=6 $G --control leak > $I/drop_gate4_control.log 2>&1; r2=$?
  echo "$(date -u +%FT%TZ) dsv41-integrate drop gate window END rc=$r1,$r2 pid=$$" >> $Q
  flock -u 9; }
echo "DONE window4 chain"
