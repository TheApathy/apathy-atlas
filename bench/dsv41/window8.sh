#!/usr/bin/env bash
# Window8 (+HC_FUSED arms, a default decode arm for plain decode tok/s, image + thinking requests):
# (a) HC_MIX_TB=1 (hc_mixes 4 tokens/block) vs default: runG logits byte identity, warm speed
# on the 2048-token prompt, PROF table, 5-way split test; (b) chunk 3968 (the core's RING-WINDOW cap)
# vs 2048 on the 4096-token prompt: identity, low-water, speed; (c) serve plain at the default
# (runG x2 + tool); (d) LAST: serve ATLAS_DSV41_DSPARK=1 with decode_multi speculation: text must
# EQUAL (c)'s for every request, plus tok/s and low-water (ABORT_GB=20).
set -uo pipefail
S=/tmp/claude-1000/-home-flocka-atlas/153d8f07-762d-4884-a8f5-8f7a1a13f1ce/scratchpad
B=/home/flocka/atlas/dsv41-integration/bench/dsv41
I=/home/flocka/atlas/DSV41_PORT/integrate
export ABORT_GB=20
rm -rf "$S"/w8_*
C="--run runG_replay --layers 40 --path model --moe-real --attn-real --tap-names logits_last --warm-prefill --chunk 2048"
TB="--env ATLAS_DSV41_HC_MIX_TB=1"
$B/keep124_window.sh k124_w8 \
  $C --tile-prompt 2 --tap-dir $S/w8_c2048 ::: \
  $C --tile-prompt 2 $TB --tap-dir $S/w8_tb2048 ::: \
  $C --tile-prompt 2 $TB --prof ::: \
  $C $TB --tap-dir $S/w8_gtb --decode 48 ::: \
  --run runI_plus20 --layers 40 --path model --moe-real --attn-real $TB --tap-names h,logits_last --split "512,512,20;1024,20;1043,1;512,512,19,1;1040,4" --tap-dir $S/w8_tbinv ::: \
  $C --tile-prompt 2 --env ATLAS_DSV41_HC_FUSED=1 --tap-dir $S/w8_fu2048 ::: \
  $C --tile-prompt 2 --env ATLAS_DSV41_HC_FUSED=1 --prof ::: \
  --run runI_plus20 --layers 40 --path model --moe-real --attn-real --env ATLAS_DSV41_HC_FUSED=1 --tap-names h,logits_last --split "512,512,20;1024,20;1043,1;512,512,19,1;1040,4" --tap-dir $S/w8_fuinv ::: \
  $C --tile-prompt 2 --env ATLAS_DSV41_SHARED_OVERLAP=1 --tap-dir $S/w8_ov2048 ::: \
  $C --tile-prompt 2 --env ATLAS_DSV41_SHARED_OVERLAP=1 --env ATLAS_DSV41_HC_FUSED=1 --tap-dir $S/w8_ovfu2048 ::: \
  $C --tap-dir $S/w8_gdef --decode 48 ::: \
  $C --tile-prompt 4 --tap-dir $S/w8_t4c2048 ::: \
  --run runG_replay --layers 40 --path model --moe-real --attn-real --tap-names logits_last --warm-prefill --chunk 3968 --tile-prompt 4 --tap-dir $S/w8_t4c3968 \
  > $I/k124_w8.log 2>&1
R=/home/flocka/atlas/DSV41_PORT/oracle/ref/runG_replay/L40.logits_last.000.bin
same() { if cmp -s "$1" "$2"; then echo IDENTICAL; else echo DIFFER; fi; }
L=L40.logits_last.000.bin
{ echo "HC_MIX_TB vs default (2048-tok, chunk 2048): $(same $S/w8_c2048/$L $S/w8_tb2048/$L)"
  echo "runG with HC_MIX_TB: $(python3 $B/logits_vs_oracle.py $R $S/w8_gtb/$L)"
  echo "HC_MIX_TB split invariance: $(python3 $B/compare_splits.py $S/w8_tbinv 2>&1 | tail -1)"
  echo "HC_FUSED vs default (2048-tok, chunk 2048): $(same $S/w8_c2048/$L $S/w8_fu2048/$L)"
  echo "HC_FUSED split invariance: $(python3 $B/compare_splits.py $S/w8_fuinv 2>&1 | tail -1)"
  echo "SHARED_OVERLAP vs default: $(same $S/w8_c2048/$L $S/w8_ov2048/$L)"
  echo "SHARED_OVERLAP+HC_FUSED vs default: $(same $S/w8_c2048/$L $S/w8_ovfu2048/$L)"
  echo "runG default (plain decode arm): $(python3 $B/logits_vs_oracle.py $R $S/w8_gdef/$L)"
  echo "4096-tok chunk 3968 vs 2048: $(same $S/w8_t4c2048/$L $S/w8_t4c3968/$L)"
  echo "control 4096-tok vs 2048-tok prompt (must DIFFER): $(same $S/w8_t4c2048/$L $S/w8_c2048/$L)"
  grep -E 'WARM prefill|this run|^decode:|tok/s\)' $I/k124_w8.log; } > $I/w8_summary.txt
$B/serve_window.sh serve8 8900 $I/serve8 > $I/serve8.log 2>&1
ATLAS_DSV41_DSPARK=1 $B/serve_window.sh serve8d 8900 $I/serve8d > $I/serve8d.log 2>&1
grep -h "DSpark:" $I/serve8d/server.log > $I/w8_dspark_stats.txt 2>&1
python3 - "$I" > $I/w8_serve_cmp.txt 2>&1 <<'PY'
import json, sys, glob, os
I = sys.argv[1]
for f in sorted(glob.glob(f"{I}/serve8/*.response.json")):
    g = f.replace("/serve8/", "/serve8d/")
    try:
        a, b = json.load(open(f)), json.load(open(g))
    except Exception as e:
        print(os.path.basename(f), "MISSING", e); continue
    ca, cb = a["choices"][0], b["choices"][0]
    ta = json.dumps(ca.get("text", ca.get("message")), sort_keys=True)
    tb = json.dumps(cb.get("text", cb.get("message")), sort_keys=True)
    # tool-call ids are random per request: compare with ids stripped
    import re
    strip = lambda s: re.sub(r'"id": "call_[0-9a-f]+"', '"id": "X"', s)
    print(os.path.basename(f), "EQUAL" if strip(ta) == strip(tb) else "DIFFER",
          "tok/s plain %.2f dspark %.2f" % (a["usage"].get("response_token/s", 0), b["usage"].get("response_token/s", 0)))
PY
echo "DONE window8 chain"
