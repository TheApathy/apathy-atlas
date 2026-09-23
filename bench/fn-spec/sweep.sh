#!/usr/bin/env bash
# usage: sweep.sh <sweep-name> <binary> <arms-file> [reps]
# arms-file lines: <label>|<extra serve args>|<extra env, space separated>
# One server lifetime per arm, arms run in file order (write C T1 C T2 C for
# paired local controls). The GB10 lock is taken EXCLUSIVELY per arm and the
# runner yields between arms when a dsv41-* window is QUEUED without START.
# A watchdog kills ONLY this sweep's server process group below FLOOR_GB.
set -uo pipefail
NAME=$1; BIN=$2; ARMS=$3; REPS=${4:-5}
SRC=$(cd "$(dirname "$0")" && pwd)
OUT=/home/flocka/atlas/fn-spec-bench/runs/$NAME; mkdir -p "$OUT"
cp "$ARMS" "$OUT/arms.txt"; cp "$SRC/client.py" "$SRC/prompts.json" "$OUT/"
Q=/home/flocka/atlas/.gb10-queue; LOCK=/home/flocka/atlas/.gb10.lock
MODEL_DIR=/home/flocka/models/Qwen3.8-Flash-Next-NVFP4-Offload
ENVFILE=${ENVFILE:-/home/flocka/atlas/allmodels-bench/env-empty.txt}
PORT=8897; MEM_NEED_GB=${MEM_NEED_GB:-100}; FLOOR_GB=${FLOOR_GB:-5}
sha256sum "$BIN" | tee "$OUT/binary.sha256"
grep -ac "qwen3.8-flash-next" "$BIN" > "$OUT/binary.target-grep"

dsv_pending() { tail -120 $Q | python3 -c '
import sys
st={}
for l in sys.stdin:
    f=l.split()
    if len(f)<3 or not f[1].startswith("dsv41") or "BUILD" in l: continue
    if "QUEUED" in l: st[f[1]]="Q"
    elif "START" in l or "END" in l: st[f[1]]="S"
print(sum(v=="Q" for v in st.values()))'; }

run_arm() {
  local LABEL=$1 XARGS=$2 XENV=$3 RUN=$OUT/$1
  rm -rf "$RUN"; mkdir -p "$RUN"
  while [ "$(dsv_pending)" != 0 ]; do sleep 20; done
  echo "$(date -u +%FT%TZ) fn-spec $NAME/$LABEL (FN serve+decode ~6min) QUEUED pid=$$" >> $Q
  exec 9>>"$LOCK"; flock 9
  echo "$(date -u +%FT%TZ) fn-spec $NAME/$LABEL window START pid=$$" >> $Q
  local APPS MA
  for i in $(seq 1 60); do
    APPS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
    MA=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    [ "$APPS" -eq 0 ] && [ "$MA" -ge "$MEM_NEED_GB" ] && break; sleep 5
  done
  echo "preflight apps=$APPS memavail=${MA}GB" > "$RUN/preflight.txt"
  if [ "$APPS" -ne 0 ] || [ "$MA" -lt "$MEM_NEED_GB" ]; then
    echo "$(date -u +%FT%TZ) fn-spec $NAME/$LABEL window END rc=2 (preflight) pid=$$" >> $Q
    flock -u 9; exec 9>&-; echo "ARM $LABEL PREFLIGHT FAIL apps=$APPS ma=$MA"; return 2
  fi
  { grep -v '^#' "$ENVFILE" | grep -v '^$'; for e in $XENV; do echo "$e"; done; } > "$RUN/env.txt"
  echo "$XARGS" > "$RUN/xargs.txt"
  setsid env -i HOME=$HOME LANG=C.UTF-8 PATH=/usr/local/cuda-13.0/bin:/usr/bin:/bin \
    LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64 RUST_LOG=info \
    $(xargs -d '\n' < "$RUN/env.txt") \
    timeout -s INT -k 60 2400 "$BIN" serve --model-from-path $MODEL_DIR --model-name m --port $PORT \
      --kernel-target qwen3.8-flash-next --max-seq-len 4096 --max-prefill-tokens 2048 \
      --ssm-cache-slots 16 --kv-cache-dtype bf16 --qwen4-qsa --gpu-memory-utilization 0.85 \
      --oom-guard-mb 4096 --request-timeout 600 --bind 127.0.0.1 --no-tui --max-batch-size 1 \
      --max-num-seqs 1 --enable-prefix-caching false $XARGS < /dev/null > "$RUN/server.log" 2>&1 &
  local PG=$! LOW=999 RC=0
  ( while kill -0 $PG 2>/dev/null; do
      m=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
      echo "$m" >> "$RUN/memavail.trace"
      if [ "$m" -lt "$FLOOR_GB" ]; then echo "WATCHDOG kill pg=$PG memavail=$m" >> "$RUN/watchdog.txt"; kill -9 -- -$PG; fi
      sleep 1; done ) < /dev/null &
  local WD=$!
  for i in $(seq 1 900); do
    curl -sf -m 2 http://127.0.0.1:$PORT/health 2>/dev/null | grep -q ready && break
    kill -0 $PG 2>/dev/null || { RC=3; break; }
    sleep 1
  done
  if [ $RC -eq 0 ] && curl -sf -m 2 http://127.0.0.1:$PORT/health | grep -q ready; then
    python3 "$OUT/client.py" $PORT "$OUT/prompts.json" $REPS "$RUN/trials.jsonl" < /dev/null > "$RUN/client.txt" 2>&1 || RC=6
  else
    [ $RC -eq 0 ] && RC=4
  fi
  kill -INT -- -$PG 2>/dev/null
  for i in $(seq 1 90); do kill -0 -- -$PG 2>/dev/null || break; sleep 1; done
  kill -9 -- -$PG 2>/dev/null; wait $WD 2>/dev/null
  for i in $(seq 1 60); do nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . || break; sleep 1; done
  LOW=$(sort -n "$RUN/memavail.trace" 2>/dev/null | head -1)
  echo "$(date -u +%FT%TZ) fn-spec $NAME/$LABEL window END rc=$RC lowwater=${LOW}GB pid=$$" >> $Q
  flock -u 9; exec 9>&-
  echo "ARM $LABEL rc=$RC lowwater=${LOW}GB"
}

while IFS='|' read -r LABEL XARGS XENV; do
  [ -z "$LABEL" ] || [ "${LABEL:0:1}" = "#" ] && continue
  run_arm "$LABEL" "$XARGS" "$XENV"
done < "$OUT/arms.txt"
echo "SWEEP $NAME DONE"
