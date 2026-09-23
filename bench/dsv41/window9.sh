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
# (c) serve with decode CUDA graphs (now default in Dsv41Model): plain and DSPARK=1 (chunk auto-capped
# to 1024). Text must EQUAL serve8 (eager, same requests) and plain == DSpark; tok/s; low-water.
$B/serve_window.sh serve9 8900 $I/serve9 > $I/serve9.log 2>&1
ATLAS_DSV41_DSPARK=1 $B/serve_window.sh serve9d 8900 $I/serve9d > $I/serve9d.log 2>&1
grep -h "DSpark:\|capping the prefill chunk" $I/serve9d/server.log > $I/w9_dspark_stats.txt 2>&1
python3 - "$I" > $I/w9_serve_cmp.txt 2>&1 <<'PY'
import json, sys, glob, os, re
I = sys.argv[1]
strip = lambda s: re.sub(r'"id": "call_[0-9a-f]+"', '"id": "X"', s)
def body(f):
    r = json.load(open(f)); c = r["choices"][0]
    return strip(json.dumps(c.get("text", c.get("message")), sort_keys=True)), r["usage"].get("response_token/s", 0)
for f in sorted(glob.glob(f"{I}/serve9/*.response.json")):
    n = os.path.basename(f)
    try:
        a, ta = body(f); e, te = body(f"{I}/serve8/{n}"); d, td = body(f"{I}/serve9d/{n}")
    except Exception as ex:
        print(n, "MISSING", ex); continue
    print(n, "graph==eager:", a == e, "dspark==plain:", d == a, "tok/s eager %.2f graph %.2f graph+dspark %.2f" % (te, ta, td))
PY
echo "DONE window9 chain"
