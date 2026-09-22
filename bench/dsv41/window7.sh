#!/usr/bin/env bash
# Window7 (integration 78bdf641b+: decode f186feee8 small-M GEMV default, SSM hooks refuse, fused FP8 rule
# on (N,K) only, serve chunk default 2048). (a) runG at chunk 2048: KL vs oracle + 48-token decode + warm;
# (b) 2048-token prompt at chunk 2048, default vs FUSED=1: logits byte identity + warm speed;
# (c) FUSED=1 5-way split test on the new rule; (d) serve at the new default (runG x2 == driver (a), tool);
# (f) PROF=1 per-op table at chunk 2048 (2048-tok prompt); (g) 4096-tok prompt at chunk 2048 vs 4096:
# low-water + identity. (e) LAST: DSPARK=1 serve at chunk 2048, low-water vs ABORT_GB=20 (projected ~21.9 GB).
set -uo pipefail
S=/tmp/claude-1000/-home-flocka-atlas/153d8f07-762d-4884-a8f5-8f7a1a13f1ce/scratchpad
B=/home/flocka/atlas/dsv41-integration/bench/dsv41
I=/home/flocka/atlas/DSV41_PORT/integrate
export ABORT_GB=20
rm -rf "$S"/w7_*
C="--run runG_replay --layers 40 --path model --moe-real --attn-real --tap-names logits_last --warm-prefill --chunk 2048"
$B/keep124_window.sh k124_w7 \
  $C --tap-dir $S/w7_g --decode 48 ::: \
  $C --tile-prompt 2 --tap-dir $S/w7_c2048 ::: \
  $C --tile-prompt 2 --env ATLAS_DSV41_FP8_FUSED=1 --tap-dir $S/w7_f2048 ::: \
  --run runI_plus20 --layers 40 --path model --moe-real --attn-real --env ATLAS_DSV41_FP8_FUSED=1 --tap-names h,logits_last,attn_out,moe_routed --split "512,512,20;1024,20;1043,1;512,512,19,1;1040,4" --tap-dir $S/w7_finv ::: \
  $C --tile-prompt 2 --prof ::: \
  $C --tile-prompt 4 --tap-dir $S/w7_t4c2048 ::: \
  --run runG_replay --layers 40 --path model --moe-real --attn-real --tap-names logits_last --warm-prefill --chunk 4096 --tile-prompt 4 --tap-dir $S/w7_t4c4096 \
  > $I/k124_w7.log 2>&1
R=/home/flocka/atlas/DSV41_PORT/oracle/ref/runG_replay/L40.logits_last.000.bin
{ echo "runG chunk 2048: $(python3 $B/logits_vs_oracle.py $R $S/w7_g/L40.logits_last.000.bin)"
  if cmp -s $S/w7_c2048/L40.logits_last.000.bin $S/w7_f2048/L40.logits_last.000.bin; then echo "FUSED vs default at chunk 2048: IDENTICAL"; else echo "FUSED vs default at chunk 2048: DIFFER"; fi
  if cmp -s $S/w7_c2048/L40.logits_last.000.bin $S/w7_g/L40.logits_last.000.bin; then echo "control: IDENTICAL -> blind"; else echo "control (2048 vs 1024-tok prompt): DIFFER (comparison can fail)"; fi
  if cmp -s $S/w7_t4c2048/L40.logits_last.000.bin $S/w7_t4c4096/L40.logits_last.000.bin; then echo "4096-tok prompt, chunk 4096 vs 2048: IDENTICAL"; else echo "4096-tok prompt, chunk 4096 vs 2048: DIFFER"; fi
  if cmp -s $S/w7_t4c2048/L40.logits_last.000.bin $S/w7_c2048/L40.logits_last.000.bin; then echo "control 4096 vs 2048-tok prompt: IDENTICAL -> blind"; else echo "control 4096 vs 2048-tok prompt: DIFFER"; fi
  echo "FUSED split invariance (new rule): $(python3 $B/compare_splits.py $S/w7_finv 2>&1 | tail -1)"
  grep -E 'WARM prefill|this run|^decode:|tok/s\)' $I/k124_w7.log; } > $I/w7_summary.txt
$B/serve_window.sh serve7 8900 $I/serve7 > $I/serve7.log 2>&1
# (e) LAST: DSpark at chunk 2048.
ATLAS_DSV41_DSPARK=1 $B/serve_window.sh serve7d 8900 $I/serve7d > $I/serve7d.log 2>&1
echo "DONE window7 chain"
