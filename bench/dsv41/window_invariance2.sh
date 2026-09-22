#!/usr/bin/env bash
# Window chain (approved 2026-09-22): (a) FP8 policy byte test pinned vs fixedm on 3 chunkings with
# L-wide sub-layer taps; decide the policy; (b)+(c) driver decode 48 + unprofiled warm prefill at
# that policy; (d) spark serve at that policy, runG twice. Each step its own lock acquisition.
set -uo pipefail
S=/tmp/claude-1000/-home-flocka-atlas/153d8f07-762d-4884-a8f5-8f7a1a13f1ce/scratchpad
B=/home/flocka/atlas/dsv41-integration/bench/dsv41
I=/home/flocka/atlas/DSV41_PORT/integrate
export ABORT_GB=20
rm -rf "$S/inv_pinned" "$S/inv_fixedm"
SPL="512,512,20;1024,20;500,544"
NAMES="h,logits_last,attn_x,attn_out,moe_in,moe_routed,moe_shared"
$B/keep124_window.sh k124_inv2 \
  --run runI_plus20 --layers 40 --path model --moe-real --attn-real --fp8-policy pinned --tap-names $NAMES --split "$SPL" --tap-dir $S/inv_pinned ::: \
  --run runI_plus20 --layers 40 --path model --moe-real --attn-real --fp8-policy fixedm --tap-names $NAMES --split "$SPL" --tap-dir $S/inv_fixedm \
  > $I/k124_inv2.log 2>&1
python3 $B/compare_splits.py $S/inv_pinned > $I/inv_pinned.txt 2>&1
python3 $B/compare_splits.py $S/inv_fixedm > $I/inv_fixedm.txt 2>&1
# Decide: pinned wins if [500,544] (split_2) is identical to split_0 at every tensor.
if ! grep 'split_2:' $I/inv_pinned.txt | grep -v 'split_2: identical' | grep -q .; then POL=pinned
elif ! grep 'split_2:' $I/inv_fixedm.txt | grep -v 'split_2: identical' | grep -q .; then POL=fixedm
else POL=rowtile; fi
echo "POLICY CHOSEN: $POL" | tee $I/policy_chosen.txt
$B/keep124_window.sh k124_decode_same_commit \
  --run runG_replay --layers 40 --path model --moe-real --attn-real --fp8-policy $POL --decode 48 --warm-prefill \
  > $I/k124_decode2.log 2>&1
ATLAS_DSV41_FP8_POLICY=$POL $B/serve_window.sh serve2 8900 $I/serve2 > $I/serve2.log 2>&1
echo "DONE chain policy=$POL"
