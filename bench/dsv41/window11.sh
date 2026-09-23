#!/usr/bin/env bash
# Window11: (a) mHC v3 (h row in smem) and REPLAY_FUSED (replay FP8 via the fused GEMM, all aligned
# shapes at M=128) vs default: logits byte identity + speed + 5-way splits; PROF with both +
# CORE_PROF (replay:<scope> rows show the replay share). (b) engram prefetch off/on, interleaved,
# 2048- and 8192-token prompts, chunk 2048, logits identical. (c) runK_32k with attention2's core
# taps (topk, n_c, index_score_rows) for their overlap gate. (d) serve DSPARK=1 at an explicit
# chunk 2048: low-water re-measure (cap lifts if >= 24 GB). (e) LAST, only if (d) >= 24 GB: a
# load-only serve at max_seq 1,048,576 (one short request), low-water.
set -uo pipefail
S=/tmp/claude-1000/-home-flocka-atlas/153d8f07-762d-4884-a8f5-8f7a1a13f1ce/scratchpad
B=/home/flocka/atlas/dsv41-integration/bench/dsv41
I=/home/flocka/atlas/DSV41_PORT/integrate
O=/home/flocka/atlas/DSV41_PORT/oracle/ref
export ABORT_GB=20
rm -rf "$S"/w11_*
C="--run runG_replay --layers 40 --path model --moe-real --attn-real --tap-names logits_last --warm-prefill --chunk 2048"
SPL='--split 512,512,20;1024,20;1043,1;512,512,19,1;1040,4'
$B/keep124_window.sh k124_w11 \
  $C --tile-prompt 2 --tap-dir $S/w11_def ::: \
  $C --tile-prompt 2 --env ATLAS_DSV41_HC_FUSED_V3=1 --tap-dir $S/w11_v3 ::: \
  $C --tile-prompt 2 --env ATLAS_DSV41_REPLAY_FUSED=1 --tap-dir $S/w11_rf ::: \
  $C --tile-prompt 2 --env ATLAS_DSV41_HC_FUSED_V3=1 --env ATLAS_DSV41_REPLAY_FUSED=1 --tap-dir $S/w11_both ::: \
  $C --tile-prompt 2 --env ATLAS_DSV41_HC_FUSED_V3=1 --env ATLAS_DSV41_REPLAY_FUSED=1 --env ATLAS_DSV41_CORE_PROF=1 --prof ::: \
  --run runI_plus20 --layers 40 --path model --moe-real --attn-real --env ATLAS_DSV41_HC_FUSED_V3=1 --env ATLAS_DSV41_REPLAY_FUSED=1 --tap-names h,logits_last $SPL --tap-dir $S/w11_inv ::: \
  $C --tile-prompt 2 --env ATLAS_DSV41_ENGRAM_PREFETCH=0 --tap-dir $S/w11_pf0a ::: \
  $C --tile-prompt 2 --tap-dir $S/w11_pf1a ::: \
  $C --tile-prompt 8 --max-seq 10240 --env ATLAS_DSV41_ENGRAM_PREFETCH=0 --tap-dir $S/w11_pf0_8k ::: \
  $C --tile-prompt 8 --max-seq 10240 --tap-dir $S/w11_pf1_8k ::: \
  $C --tile-prompt 2 --env ATLAS_DSV41_ENGRAM_PREFETCH=0 --tap-dir $S/w11_pf0b ::: \
  $C --tile-prompt 2 --tap-dir $S/w11_pf1b ::: \
  --run runK_32k --layers 40 --path model --moe-real --attn-real --max-seq 34816 --chunk 2048 --env ATLAS_DSV41_TAP_INDEX_SCORE=1 --tap-names logits_last,topk,n_c,index_score_rows --tap-layers 2,20,24,40 --tap-dir $S/w11_ktaps \
  > $I/k124_w11.log 2>&1
same() { if cmp -s "$1" "$2"; then echo IDENTICAL; else echo DIFFER; fi; }
L=L40.logits_last.000.bin
{ echo "HC_FUSED_V3 vs default: $(same $S/w11_def/$L $S/w11_v3/$L)"
  echo "REPLAY_FUSED vs default: $(same $S/w11_def/$L $S/w11_rf/$L)"
  echo "V3+REPLAY_FUSED vs default: $(same $S/w11_def/$L $S/w11_both/$L)"
  echo "V3+REPLAY_FUSED split invariance: $(python3 $B/compare_splits.py $S/w11_inv 2>&1 | tail -1)"
  echo "prefetch 2K on vs off: $(same $S/w11_pf0a/$L $S/w11_pf1a/$L) / $(same $S/w11_pf0b/$L $S/w11_pf1b/$L)"
  echo "prefetch 8K on vs off: $(same $S/w11_pf0_8k/$L $S/w11_pf1_8k/$L)"
  echo "runK taps written: $(ls $S/w11_ktaps 2>/dev/null | grep -c topk) topk files, $(ls $S/w11_ktaps 2>/dev/null | grep -c index_score_rows) index_score_rows files"
  echo "runK (this build) vs oracle: $(python3 $B/logits_vs_oracle.py $O/runK_32k/$L $S/w11_ktaps/$L)"
  grep -E 'WARM prefill|this run' $I/k124_w11.log; } > $I/w11_summary.txt
# serve TTFT on the 2048-token prompt (4 identical requests; 1 is cold) vs the driver's warm prefill
$B/serve_window.sh serve11t 8900 $I/serve11t > $I/serve11t.log 2>&1
python3 -c "
import json,glob
for f in sorted(glob.glob('$I/serve11t/*.response.json')):
    u=json.load(open(f)).get('usage',{}); print('serve TTFT', f.split('/')[-1], u.get('prompt_tokens'), u.get('time_to_first_token_ms'), 'ms')
" >> $I/w11_summary.txt 2>&1
# graphs (decode 1c2175d1c, re-capture per sequence): 7 requests back to back, plain and DSpark;
# text must equal serve8 (eager, same requests) for every request.
ATLAS_DSV41_GRAPH=1 $B/serve_window.sh serve11g 8900 $I/serve11g > $I/serve11g.log 2>&1
ATLAS_DSV41_GRAPH=1 ATLAS_DSV41_DSPARK=1 $B/serve_window.sh serve11gd 8900 $I/serve11gd > $I/serve11gd.log 2>&1
python3 - "$I" >> $I/w11_summary.txt 2>&1 <<'PY'
import json, sys, glob, os, re
I = sys.argv[1]
strip = lambda s: re.sub(r'"id": "call_[0-9a-f]+"', '"id": "X"', s)
def body(f):
    r = json.load(open(f))
    if "choices" not in r:
        return "ERROR " + json.dumps(r)[:120], 0
    c = r["choices"][0]
    return strip(json.dumps(c.get("text", c.get("message")), sort_keys=True)), r["usage"].get("response_token/s", 0)
for f in sorted(glob.glob(f"{I}/serve11g/*.response.json")):
    n = os.path.basename(f)
    ref = f"{I}/serve8/{n}" if os.path.exists(f"{I}/serve8/{n}") else f"{I}/serve8/01_completion_raw_runG_48.response.json"
    e, te = body(ref); g, tg = body(f); d, td = body(f"{I}/serve11gd/{n}")
    print("graph", n, "graph==eager:", g == e, "graph+dspark==eager:", d == e, "tok/s eager %.2f graph %.2f graph+dspark %.2f" % (te, tg, td))
PY
ATLAS_DSV41_DSPARK=1 ATLAS_DSV41_CHUNK=2048 $B/serve_window.sh serve11d 8900 $I/serve11d > $I/serve11d.log 2>&1
low=$(awk '/^low-water MemAvailable:/ {print $3}' $I/serve11d.log)
echo "DSpark chunk-2048 low-water: ${low:-?} GB" >> $I/w11_summary.txt
if [ -n "$low" ] && [ "$low" -ge 24 ]; then
  SERVE_MAX_SEQ=1048576 $B/serve_window.sh serve11m 8900 $I/serve11m > $I/serve11m.log 2>&1
  echo "1M load-only serve: $(tail -1 $I/serve11m.log)" >> $I/w11_summary.txt
else
  echo "1M load-only serve: SKIPPED (DSpark low-water ${low:-?} < 24 GB)" >> $I/w11_summary.txt
fi
# sampled-spec distribution gate (rerun: window10's requests hit the chat endpoint and returned
# errors; the gate now FAILS on < 400 samples/arm), then adaptive k (DSpark, ADAPTIVE=1) on the
# serve11gd request set: greedy text == eager, tok/s, tokens/step.
for d in dist_plain dist_dspark dist_control; do rm -f $I/$d/*.response.json; done
$B/serve_window.sh dist_plain 8900 $I/dist_plain > $I/dist_plain.log 2>&1
ATLAS_DSV41_DSPARK=1 $B/serve_window.sh dist_dspark 8900 $I/dist_dspark > $I/dist_dspark.log 2>&1
ATLAS_DSV41_DSPARK=1 ATLAS_DSV41_CONTROL_SPEC_ACCEPT_ALL=1 $B/serve_window.sh dist_control 8900 $I/dist_control > $I/dist_control.log 2>&1
python3 $B/dist_gate.py $I > $I/w11_dist_gate.txt 2>&1
ATLAS_DSV41_DSPARK=1 ATLAS_DSV41_DSPARK_ADAPTIVE=1 $B/serve_window.sh serve11a 8900 $I/serve11a > $I/serve11a.log 2>&1
grep -h "DSpark:" $I/serve11a/server.log > $I/w11_adaptive_stats.txt 2>&1
echo "DONE window11 chain"
