#!/usr/bin/env bash
# Window6 (lead, 2026-09-22): prefill chunk 512 vs 1024 vs 2048 on the SAME 2048-token prompt
# (runG ids x2, no oracle): per-run MemAvailable low-water (ABORT_GB=20), warm prefill tok/s, and
# byte identity of the last-token logits across chunk sizes. Then runG (1024 tok) at chunk 1024 vs
# the oracle (KL/top-10) so the default switch has an accuracy receipt. Ascending chunk order.
# Then attention2's fused FP8 GEMM (ATLAS_DSV41_FP8_FUSED=1): logits byte-identical to the default
# path at chunk 512 and 1024, warm prefill speed, and the 1-token-tail split test under it.
set -uo pipefail
S=/tmp/claude-1000/-home-flocka-atlas/153d8f07-762d-4884-a8f5-8f7a1a13f1ce/scratchpad
B=/home/flocka/atlas/dsv41-integration/bench/dsv41
I=/home/flocka/atlas/DSV41_PORT/integrate
export ABORT_GB=20
rm -rf "$S"/w6_*
C="--run runG_replay --layers 40 --path model --moe-real --attn-real --tap-names logits_last --warm-prefill"
$B/keep124_window.sh k124_w6 \
  $C --tile-prompt 2 --chunk 512 --tap-dir $S/w6_c512 ::: \
  $C --tile-prompt 2 --chunk 1024 --tap-dir $S/w6_c1024 ::: \
  $C --tile-prompt 2 --chunk 2048 --tap-dir $S/w6_c2048 ::: \
  $C --chunk 1024 --tap-dir $S/w6_g1024 --decode 48 ::: \
  $C --tile-prompt 2 --chunk 512 --env ATLAS_DSV41_FP8_FUSED=1 --tap-dir $S/w6_f512 ::: \
  $C --tile-prompt 2 --chunk 1024 --env ATLAS_DSV41_FP8_FUSED=1 --tap-dir $S/w6_f1024 ::: \
  --run runI_plus20 --layers 40 --path model --moe-real --attn-real --env ATLAS_DSV41_FP8_FUSED=1 --tap-names h,logits_last --split "512,512,20;1024,20;1043,1;512,512,19,1;1040,4" --tap-dir $S/w6_finv \
  > $I/k124_w6.log 2>&1
R=/home/flocka/atlas/DSV41_PORT/oracle/ref/runG_replay/L40.logits_last.000.bin
{ for c in 1024 2048; do
    if cmp -s $S/w6_c512/L40.logits_last.000.bin $S/w6_c$c/L40.logits_last.000.bin; then echo "chunk $c vs 512 logits: IDENTICAL"; else echo "chunk $c vs 512 logits: DIFFER"; fi
  done
  for a in f512 f1024; do
    if cmp -s $S/w6_c512/L40.logits_last.000.bin $S/w6_$a/L40.logits_last.000.bin; then echo "FUSED $a vs default c512 logits: IDENTICAL"; else echo "FUSED $a vs default c512 logits: DIFFER"; fi
  done
  echo "FUSED split invariance: $(python3 $B/compare_splits.py $S/w6_finv 2>&1 | tail -1)"
  # negative control: the 2048-token prompt's logits must differ from the 1024-token prompt's
  if cmp -s $S/w6_c512/L40.logits_last.000.bin $S/w6_g1024/L40.logits_last.000.bin; then echo "control (2048-tok vs 1024-tok prompt): IDENTICAL -> comparison is blind"; else echo "control (2048-tok vs 1024-tok prompt): DIFFER (comparison can fail)"; fi
  echo "runG chunk 1024: $(python3 $B/logits_vs_oracle.py $R $S/w6_g1024/L40.logits_last.000.bin)"
  grep -E 'WARM prefill|this run|^decode:|tok/s\)' $I/k124_w6.log; } > $I/w6_summary.txt
echo "DONE window6 chain"
