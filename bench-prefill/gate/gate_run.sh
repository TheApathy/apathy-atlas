#!/usr/bin/env bash
# usage: gate_run.sh <label> <binary> <envfile> [EXTRA_ENV=..]
# Starts the server with ATLAS_PREFILL_HIDDEN_DUMP, sends the three 2048-token gate texts, collects dumps.
set -uo pipefail
W=/home/flocka/atlas/qwen27b-prefill-work
LABEL=${1:?}; BIN=${2:?}; ENVF=${3:?}; shift 3
RUN=$W/bench/runs/gate-$LABEL; mkdir -p $RUN; DUMP=$RUN/dump; rm -rf $DUMP; mkdir -p $DUMP
PORT=8896
exec 9>/home/flocka/atlas/.gb10.lock
echo "[$(date -u +%T)] waiting for GPU lock"; flock -w 14400 9 || exit 2
echo "[$(date -u +%T)] lock acquired"
for i in $(seq 1 240); do
  FOREIGN=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -c . || true)
  MA=$(awk '/MemAvailable/{print int($2/1024/1024)}' /proc/meminfo)
  [ "$FOREIGN" -eq 0 ] && [ "$MA" -ge 100 ] && break; sleep 5
done
[ "$FOREIGN" -eq 0 ] && [ "$MA" -ge 100 ] || { echo "preflight failed foreign=$FOREIGN MA=$MA"; exit 3; }
sha256sum $BIN | tee $RUN/binary.sha256
cp $ENVF $RUN/env.txt; printf '%s\n' "$@" >> $RUN/env.txt
PREFIX=""
source $W/bench/common_start.sh
start_server $RUN/server.log "$@" ATLAS_PREFILL_HIDDEN_DUMP=$DUMP || exit $?
for t in code prose legal; do
  curl -s -m 300 -o $RUN/$t.response.json -H 'Content-Type: application/json' -d @$W/bench/gate/req_$t.json http://127.0.0.1:$PORT/v1/completions
  sleep 1
  f=$(ls $DUMP/hidden_*.bin 2>/dev/null | head -1); [ -n "$f" ] && mv "$f" "$RUN/hidden_$t.bin"
  echo "$t: $(python3 -c "import json;r=json.load(open('$RUN/$t.response.json'));print(r['choices'][0]['text'].encode(), r['usage']['prompt_tokens'], r['usage']['time_to_first_token_ms'])")"
done
grep -c "ATLAS_PREFILL_HIDDEN_DUMP: wrote" $RUN/server.log
stop_server
echo "[$(date -u +%T)] done"
