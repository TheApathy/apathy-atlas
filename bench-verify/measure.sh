#!/usr/bin/env bash
# usage: measure.sh <run-label> <binary> <envfile> [EXTRA_ENV=.. ...]
# Holds the box-wide GPU lock for the whole start->measure->stop cycle.
# Emits bench/runs/<label>/{server.log,warmup.*,trial-N.*,summary.txt}
set -uo pipefail
W=/home/flocka/atlas/integrate/qwen27b
LABEL=${1:?label}; BIN=${2:?binary}; ENVF=${3:?envfile}; shift 3
RUN=$W/bench-verify/runs/$LABEL; mkdir -p $RUN
PORT=8896
PREFIX=""
exec 9>/home/flocka/atlas/.gb10.lock
echo "[$(date -u +%T)] waiting for GPU lock" ; flock -w 14400 9 || { echo lock-timeout; exit 2; }
echo "[$(date -u +%T)] lock acquired"
# preflight
for i in $(seq 1 240); do
  FOREIGN=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -c . || true)
  MA=$(awk '/MemAvailable/{print int($2/1024/1024)}' /proc/meminfo)
  [ "$FOREIGN" -eq 0 ] && [ "$MA" -ge 100 ] && break
  [ $((i % 12)) -eq 1 ] && echo "[$(date -u +%T)] waiting: foreign=$FOREIGN MemAvailable=${MA}GB"
  sleep 5
done
echo "MemAvailable ${MA} GB foreign=$FOREIGN"
[ "$FOREIGN" -eq 0 ] || { echo "foreign GPU process still present"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv; exit 3; }
[ "$MA" -ge 100 ] || { echo "MemAvailable < 100 GB"; exit 4; }
if ss -ltn | grep -q ":$PORT "; then echo "port $PORT busy"; exit 5; fi
sha256sum $BIN | tee $RUN/binary.sha256
cp $ENVF $RUN/env.txt; printf '%s\n' "$@" >> $RUN/env.txt
cp $W/bench-verify/request.json $RUN/request.json
# launch
PREFIX=""
if [ -n "${NSYS_OUT:-}" ]; then PREFIX="/usr/local/bin/nsys profile --trace=cuda,nvtx --cuda-graph-trace=node -o $NSYS_OUT -f true"; fi
source $W/bench-verify/common_start.sh
start_server $RUN/server.log "$@" || exit $?
curl -s http://127.0.0.1:$PORT/health > $RUN/health.json; echo; cat $RUN/health.json; echo
req() { # name
  local t0=$(date +%s.%N)
  curl -s -m 300 -o $RUN/$1.response.json -w '%{http_code} %{time_starttransfer} %{time_total}\n' \
    -H 'Content-Type: application/json' -d @$RUN/request.json http://127.0.0.1:$PORT/v1/completions > $RUN/$1.transport.txt
  cat $RUN/$1.transport.txt
}
req warmup
NTRIALS=${NTRIALS:-5}
for i in $(seq 1 $NTRIALS); do req trial-$i; done
python3 - $RUN <<'PY'
import json,sys,statistics,os
run=sys.argv[1]; rows=[]
for n in ['warmup']+[f'trial-{i}' for i in range(1,int(os.environ.get('NTRIALS','5'))+1)]:
    r=json.load(open(f'{run}/{n}.response.json'))
    u=r['usage']; t=u['time_to_first_token_ms']
    rows.append((n,t,u['prompt_tokens'],u['completion_tokens'],u.get('prompt_tokens_details',{}).get('cached_tokens'),r['choices'][0]['text'],r['choices'][0]['finish_reason']))
for row in rows: print(row)
ts=[r[1] for r in rows[1:]]; med=statistics.median(ts); pt=rows[1][2]
s=f"median TTFT {med:.3f} ms  -> {pt/med*1000:.1f} tok/s (prompt_tokens={pt}) texts={[r[5] for r in rows[1:]]}"
print(s); open(f'{run}/summary.txt','w').write('\n'.join(map(str,rows))+'\n'+s+'\n')
PY
grep -i -E 'error|nonfinite|CUDA_ERROR|panic' $RUN/server.log | head -5
stop_server
