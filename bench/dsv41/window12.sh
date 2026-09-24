#!/usr/bin/env bash
# Window12 (DSpark default-on decision; lead top priority; ORDER per lead: T=1 story, dist gate, Python
# reference, chunk-2048 low-water, TTFT, then the driver arms last -- prefill speed here is NOT to be
# judged: this build has the fused-FP8-over-resident -1.2% regression decode's a90fcd45f fixes). Build: integration afd92d54a (decode 98e969e62
# fused FP8 v7 + slice32, shared/attention residency, balanced chunks, parity sampler fix, attention
# 889adb9fd, TRIM_LAST opt-in). Driver: default 2048-token score, TRIM_LAST identity + speed + 5-way
# splits, PROF (CORE_PROF). Serves (graphs + residency default): TTFT 2048; T=1 story seeds 1-3 plain vs
# DSpark (tokens/step, distinct-4gram vs Python py_t1_story); distribution gate plain/DSpark/control
# (+ Python py_dist); DSpark low-water at the default chunk (cap) and at explicit chunk 2048.
set -uo pipefail
S=/tmp/claude-1000/-home-flocka-atlas/153d8f07-762d-4884-a8f5-8f7a1a13f1ce/scratchpad
B=/home/flocka/atlas/dsv41-integration/bench/dsv41
I=/home/flocka/atlas/DSV41_PORT/integrate
export ABORT_GB=12
rm -rf "$S"/w12_*
C="--run runG_replay --layers 40 --path model --moe-real --attn-real --tap-names logits_last --warm-prefill --chunk 2048"
SPL='--split 512,512,20;1024,20;1043,1;512,512,19,1;1040,4'
$B/serve_window.sh serve12p 8900 $I/serve12p > $I/serve12p.log 2>&1
ATLAS_DSV41_DSPARK=1 $B/serve_window.sh serve12d 8900 $I/serve12d > $I/serve12d.log 2>&1
for d in dist_plain dist_dspark dist_control; do rm -f $I/$d/*.response.json; done
$B/serve_window.sh dist_plain 8900 $I/dist_plain > $I/dist_plain.log 2>&1
ATLAS_DSV41_DSPARK=1 $B/serve_window.sh dist_dspark 8900 $I/dist_dspark > $I/dist_dspark.log 2>&1
ATLAS_DSV41_DSPARK=1 ATLAS_DSV41_CONTROL_SPEC_ACCEPT_ALL=1 $B/serve_window.sh dist_control 8900 $I/dist_control > $I/dist_control.log 2>&1
$B/capture_py_dist.sh > $I/py_dist.log 2>&1
python3 $B/dist_gate.py $I > $I/w12_dist_gate.txt 2>&1
ATLAS_DSV41_DSPARK=1 ATLAS_DSV41_CHUNK=2048 $B/serve_window.sh serve12d2k 8900 $I/serve12d2k > $I/serve12d2k.log 2>&1
$B/serve_window.sh serve12t 8900 $I/serve12t > $I/serve12t.log 2>&1
$B/keep124_window.sh k124_w12 \
  $C --tile-prompt 2 --tap-dir $S/w12_def ::: \
  $C --tile-prompt 2 --env ATLAS_DSV41_TRIM_LAST=1 --tap-dir $S/w12_trim ::: \
  --run runI_plus20 --layers 40 --path model --moe-real --attn-real --env ATLAS_DSV41_TRIM_LAST=1 --tap-names h,logits_last $SPL --tap-dir $S/w12_triminv ::: \
  $C --tile-prompt 2 --env ATLAS_DSV41_CORE_PROF=1 --prof ::: \
  $C --tile-prompt 8 --max-seq 10240 --chunk 3968 --tap-dir $S/w12_8k \
  > $I/k124_w12.log 2>&1
same() { if cmp -s "$1" "$2"; then echo IDENTICAL; else echo DIFFER; fi; }
L=L40.logits_last.000.bin
{ echo "TRIM_LAST vs default: $(same $S/w12_def/$L $S/w12_trim/$L)"
  echo "TRIM_LAST split invariance: $(python3 $B/compare_splits.py $S/w12_triminv 2>&1 | tail -1)"
  grep -E 'WARM prefill|this run|^decode' $I/k124_w12.log; } >> $I/w12_summary.txt
python3 - "$I" >> $I/w12_summary.txt 2>&1 <<'PY'
import json, glob, os, sys
I = sys.argv[1]
def d4(t):
    w = t.split(); g = [tuple(w[i:i+4]) for i in range(len(w) - 3)]
    return len(set(g)) / max(1, len(g))
for f in sorted(glob.glob(f"{I}/serve12t/*.response.json")):
    u = json.load(open(f)).get("usage", {}); print("TTFT", os.path.basename(f), u.get("prompt_tokens"), u.get("time_to_first_token_ms"), "ms")
for d in ("serve12p", "serve12d"):
    for f in sorted(glob.glob(f"{I}/{d}/*.response.json")):
        r = json.load(open(f)); c = r["choices"][0]["message"]; t = c.get("content") or ""
        print(d, os.path.basename(f), "tok/s %.2f" % r["usage"].get("response_token/s", 0), "tokens", r["usage"]["completion_tokens"], "distinct4 %.2f" % d4(t))
py = json.load(open("/home/flocka/atlas/DSV41_PORT/oracle/ref/py_t1_story.json"))
for r in py["runs"]:
    print("python spec=%s seed=%d tokens %d distinct4 %.2f" % (r["spec"], r["seed"], len(r["tokens"]), d4(r["text"])))
for d in ("serve12p", "serve12d", "serve12d2k", "dist_plain", "dist_dspark"):
    try:
        print(d, [l.strip() for l in open(f"{I}/{d}.log") if "low-water" in l][-1])
    except Exception as e:
        print(d, "no log", e)
PY
grep -h "DSpark:\|capping" $I/serve12d/server.log $I/serve12d2k/server.log >> $I/w12_summary.txt 2>&1
echo "DONE window12 chain"
