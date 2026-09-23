#!/usr/bin/env bash
# Window9: (a) mHC v2 (one reduction tree + warp Sinkhorn): MIX_V2 and HC_FUSED vs default on the
# 2048-token prompt (byte identity + warm speed), PROF with HC_FUSED + ATLAS_DSV41_CORE_PROF=1 (attention2's
# per-phase split), split tests; (b) runK_32k: Atlas at 32,768 real tokens vs the Python oracle
# (step-0 KL/top-10, per-layer h/attn_out at L2/L20/L24, 16 greedy), prefill tok/s, low-water;
# chunk 3968 vs 2048 identity at 32K.
set -uo pipefail
S=/tmp/claude-1000/-home-flocka-atlas/153d8f07-762d-4884-a8f5-8f7a1a13f1ce/scratchpad
B=/home/flocka/atlas/dsv41-integration/bench/dsv41
I=/home/flocka/atlas/DSV41_PORT/integrate
O=/home/flocka/atlas/DSV41_PORT/oracle/ref
export ABORT_GB=20
rm -rf "$S"/w9_*
C="--run runG_replay --layers 40 --path model --moe-real --attn-real --tap-names logits_last --warm-prefill --chunk 2048 --tile-prompt 2"
K="--run runK_32k --layers 40 --path model --moe-real --attn-real --max-seq 34816 --warm-prefill"
$B/keep124_window.sh k124_w9 \
  $C --tap-dir $S/w9_def ::: \
  $C --env ATLAS_DSV41_HC_MIX_V2=1 --tap-dir $S/w9_mv2 ::: \
  $C --env ATLAS_DSV41_HC_FUSED=1 --tap-dir $S/w9_fu ::: \
  $C --env ATLAS_DSV41_HC_FUSED=1 --env ATLAS_DSV41_CORE_PROF=1 --prof ::: \
  --run runI_plus20 --layers 40 --path model --moe-real --attn-real --env ATLAS_DSV41_HC_MIX_V2=1 --tap-names h,logits_last --split "512,512,20;1024,20;1043,1;512,512,19,1;1040,4" --tap-dir $S/w9_mv2inv ::: \
  --run runI_plus20 --layers 40 --path model --moe-real --attn-real --env ATLAS_DSV41_HC_FUSED=1 --tap-names h,logits_last --split "512,512,20;1024,20;1043,1;512,512,19,1;1040,4" --tap-dir $S/w9_fuinv ::: \
  $K --chunk 2048 --tap-names logits_last,h,attn_out,topk,n_c --tap-layers 2,20,24,40 --tap-dir $S/w9_k2048 --decode 16 ::: \
  $K --chunk 3968 --tap-names logits_last --tap-dir $S/w9_k3968 \
  > $I/k124_w9.log 2>&1
same() { if cmp -s "$1" "$2"; then echo IDENTICAL; else echo DIFFER; fi; }
L=L40.logits_last.000.bin
{ echo "MIX_V2 vs default: $(same $S/w9_def/$L $S/w9_mv2/$L)"
  echo "HC_FUSED vs default: $(same $S/w9_def/$L $S/w9_fu/$L)"
  echo "MIX_V2 split invariance: $(python3 $B/compare_splits.py $S/w9_mv2inv 2>&1 | tail -1)"
  echo "HC_FUSED split invariance: $(python3 $B/compare_splits.py $S/w9_fuinv 2>&1 | tail -1)"
  echo "runK_32k step-0 vs Python oracle: $(python3 $B/logits_vs_oracle.py $O/runK_32k/$L $S/w9_k2048/$L)"
  echo "runK_32k chunk 3968 vs 2048: $(same $S/w9_k2048/$L $S/w9_k3968/$L)"
  echo "control runK vs 2048-tok prompt (must DIFFER): $(same $S/w9_k2048/$L $S/w9_def/$L)"
  grep -E 'WARM prefill|^prefill|this run|^decode:|tok/s\)' $I/k124_w9.log; } > $I/w9_summary.txt
python3 $B/compare_all.py $O/runK_32k $S/w9_k2048 > $I/w9_runK_layers.txt 2>&1
echo "DONE window9 chain"
