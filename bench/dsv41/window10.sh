#!/usr/bin/env bash
# Window10 (integration with engine2 6c7bf1c8d L1 MoE, decode e3d5cb6aa graph drop, sampled spec +
# accept-all control): (a) score + PROF (CORE_PROF) at chunk 2048 on the 2048-token prompt, HC_FUSED arm;
# (b) 8192-token prompt at chunk 2048 vs 3968 (lead: length-aware default if 3968 wins > 3%);
# (c) drop gate with decode graphs on (+ leak control); (d) sampled-spec distribution gate: serves
# plain / DSPARK / DSPARK + accept-all control, 2 prompts x 500 seeds x 2 tokens, dist_gate.py.
set -uo pipefail
S=/tmp/claude-1000/-home-flocka-atlas/153d8f07-762d-4884-a8f5-8f7a1a13f1ce/scratchpad
B=/home/flocka/atlas/dsv41-integration/bench/dsv41
I=/home/flocka/atlas/DSV41_PORT/integrate
export ABORT_GB=20
rm -rf "$S"/w10_*
C="--run runG_replay --layers 40 --path model --moe-real --attn-real --tap-names logits_last --warm-prefill"
$B/keep124_window.sh k124_w10 \
  $C --chunk 2048 --tile-prompt 2 --tap-dir $S/w10_def ::: \
  $C --chunk 2048 --tile-prompt 2 --env ATLAS_DSV41_CORE_PROF=1 --prof ::: \
  $C --chunk 2048 --tile-prompt 2 --env ATLAS_DSV41_HC_FUSED=1 --tap-dir $S/w10_fu ::: \
  $C --chunk 2048 --tile-prompt 8 --max-seq 10240 --tap-dir $S/w10_8k2048 ::: \
  $C --chunk 3968 --tile-prompt 8 --max-seq 10240 --tap-dir $S/w10_8k3968 \
  > $I/k124_w10.log 2>&1
same() { if cmp -s "$1" "$2"; then echo IDENTICAL; else echo DIFFER; fi; }
L=L40.logits_last.000.bin
{ echo "HC_FUSED vs default: $(same $S/w10_def/$L $S/w10_fu/$L)"
  echo "8K chunk 3968 vs 2048: $(same $S/w10_8k2048/$L $S/w10_8k3968/$L)"
  echo "control 8K vs 2K prompt (must DIFFER): $(same $S/w10_8k2048/$L $S/w10_def/$L)"
  grep -E 'WARM prefill|this run' $I/k124_w10.log; } > $I/w10_summary.txt
Q=/home/flocka/atlas/.gb10-queue; exec 9>>/home/flocka/atlas/.gb10.lock
echo "$(date -u +%FT%TZ) dsv41-integrate drop gate keep=6 graphs ON (+leak control) QUEUED pid=$$" >> $Q
flock -w 7200 9 && {
  echo "$(date -u +%FT%TZ) dsv41-integrate drop gate window START (lock held) pid=$$" >> $Q
  G=/home/flocka/atlas/dsv41-integration/target/release/examples/dsv41_drop_gate
  ATLAS_DSV41_GRAPH=1 ATLAS_DSV41_PACKED_KEEP=6 $G > $I/drop_gate10.log 2>&1; r1=$?
  ATLAS_DSV41_GRAPH=1 ATLAS_DSV41_PACKED_KEEP=6 $G --control leak > $I/drop_gate10_control.log 2>&1; r2=$?
  echo "$(date -u +%FT%TZ) dsv41-integrate drop gate window END rc=$r1,$r2 pid=$$" >> $Q
  flock -u 9; }
exec 9>&-
$B/serve_window.sh dist_plain 8900 $I/dist_plain > $I/dist_plain.log 2>&1
ATLAS_DSV41_DSPARK=1 $B/serve_window.sh dist_dspark 8900 $I/dist_dspark > $I/dist_dspark.log 2>&1
ATLAS_DSV41_DSPARK=1 ATLAS_DSV41_CONTROL_SPEC_ACCEPT_ALL=1 $B/serve_window.sh dist_control 8900 $I/dist_control > $I/dist_control.log 2>&1
grep -h "DSpark:" $I/dist_dspark/server.log | tail -3 > $I/w10_dspark_stats.txt 2>&1
python3 $B/dist_gate.py $I > $I/w10_dist_gate.txt 2>&1
echo "DONE window10 chain"
