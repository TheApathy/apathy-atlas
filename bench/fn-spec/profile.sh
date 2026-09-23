#!/usr/bin/env bash
# usage: profile.sh <name> <binary> <prompt-id> "<extra serve args>" "<extra env>"
# One exclusive GB10 window: launch the server under an nsys session (no collection
# during load), start collection for ONE greedy request of <prompt-id>, stop, export
# a cuda_gpu_kern_sum + cuda_gpu_trace. Timing from this window is NOT scoreable.
set -uo pipefail
NAME=$1; BIN=$2; PID_=$3; XARGS=$4; XENV=$5
SRC=$(cd "$(dirname "$0")" && pwd)
OUT=/home/flocka/atlas/fn-spec-bench/prof/$NAME; rm -rf "$OUT"; mkdir -p "$OUT"
Q=/home/flocka/atlas/.gb10-queue; LOCK=/home/flocka/atlas/.gb10.lock
MODEL_DIR=/home/flocka/models/Qwen3.8-Flash-Next-NVFP4-Offload; PORT=8897
FLOOR_GB=${FLOOR_GB:-5}; MEM_NEED_GB=${MEM_NEED_GB:-100}; SESS=fnspec$$
python3 - "$SRC/prompts.json" "$PID_" > "$OUT/prompt.json" <<'PY'
import json,sys
p=[x for x in json.load(open(sys.argv[1])) if x["id"]==sys.argv[2]][0]
json.dump([p],sys.stdout)
PY
echo "$(date -u +%FT%TZ) fn-spec prof/$NAME (FN nsys, ~8min) QUEUED pid=$$" >> $Q
exec 9>>"$LOCK"; flock 9
echo "$(date -u +%FT%TZ) fn-spec prof/$NAME window START pid=$$" >> $Q
for i in $(seq 1 60); do
  APPS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
  MA=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  [ "$APPS" -eq 0 ] && [ "$MA" -ge "$MEM_NEED_GB" ] && break; sleep 5
done
if [ "$APPS" -ne 0 ] || [ "$MA" -lt "$MEM_NEED_GB" ]; then
  echo "$(date -u +%FT%TZ) fn-spec prof/$NAME window END rc=2 (preflight) pid=$$" >> $Q; exit 2; fi
setsid env -i HOME=$HOME LANG=C.UTF-8 PATH=/usr/local/cuda-13.0/bin:/usr/local/bin:/usr/bin:/bin \
  LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64 RUST_LOG=info $XENV \
  nsys launch --session-new=$SESS -t cuda,nvtx --cuda-graph-trace=node \
  timeout -s INT -k 60 2400 "$BIN" serve --model-from-path $MODEL_DIR --model-name m --port $PORT \
    --kernel-target qwen3.8-flash-next --max-seq-len 4096 --max-prefill-tokens 2048 \
    --ssm-cache-slots 16 --kv-cache-dtype bf16 --qwen4-qsa --gpu-memory-utilization 0.85 \
    --oom-guard-mb 4096 --request-timeout 600 --bind 127.0.0.1 --no-tui --max-batch-size 1 \
    --max-num-seqs 1 --enable-prefix-caching false $XARGS < /dev/null > "$OUT/server.log" 2>&1 &
PG=$!; RC=0
( while kill -0 $PG 2>/dev/null; do m=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    echo "$m" >> "$OUT/memavail.trace"
    [ "$m" -lt "$FLOOR_GB" ] && { echo "WATCHDOG kill pg=$PG memavail=$m" >> "$OUT/watchdog.txt"; kill -9 -- -$PG; }
    sleep 1; done ) < /dev/null &
WD=$!
for i in $(seq 1 900); do
  curl -sf -m 2 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q ready && break
  kill -0 $PG 2>/dev/null || { RC=3; break; }; sleep 1
done
if [ $RC -eq 0 ]; then
  # warm the request once without collection, then collect exactly one
  python3 "$SRC/client.py" $PORT "$OUT/prompt.json" 0 "$OUT/warm.jsonl" < /dev/null > "$OUT/warm.txt" 2>&1
  nsys start --session=$SESS -o "$OUT/trace" > "$OUT/nsys-start.txt" 2>&1
  python3 "$SRC/client.py" $PORT "$OUT/prompt.json" 0 "$OUT/prof.jsonl" < /dev/null > "$OUT/prof.txt" 2>&1 || RC=6
  nsys stop --session=$SESS > "$OUT/nsys-stop.txt" 2>&1
fi
kill -INT -- -$PG 2>/dev/null
for i in $(seq 1 120); do kill -0 -- -$PG 2>/dev/null || break; sleep 1; done
kill -9 -- -$PG 2>/dev/null; wait $WD 2>/dev/null
for i in $(seq 1 60); do nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . || break; sleep 1; done
LOW=$(sort -n "$OUT/memavail.trace" 2>/dev/null | head -1)
echo "$(date -u +%FT%TZ) fn-spec prof/$NAME window END rc=$RC lowwater=${LOW}GB pid=$$" >> $Q
flock -u 9
[ -f "$OUT/trace.nsys-rep" ] && nsys stats -r cuda_gpu_kern_sum,cuda_gpu_trace -f csv -o "$OUT/stats" "$OUT/trace.nsys-rep" > /dev/null 2>&1
echo "PROF $NAME rc=$RC lowwater=${LOW}GB"
