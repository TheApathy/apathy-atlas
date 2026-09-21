#!/bin/bash
# run_arm.sh ARM_NAME BINARY ENV_FILE [EXTRA_SERVER_ARGS...]
# Starts the server under the box-wide GPU lock, sends warmup + 5 measured
# 2048-token requests, stops the server, releases the lock.
set -u
ARM="$1"; BIN="$2"; ENVFILE="$3"; shift 3
EXTRA_ARGS="$*"
BENCH=/home/flocka/atlas/flashnext-prefill-work/bench
OUT=$BENCH/results/$ARM; mkdir -p "$OUT"
PORT=${PORT:-8898}
NREQ=${NREQ:-5}
PRE="${PRE:-}"   # optional wrapper (e.g. nsys profile ...)
MODEL=/home/flocka/models/Qwen3.8-Flash-Next-NVFP4-Offload
export ARM BIN ENVFILE EXTRA_ARGS OUT PORT NREQ PRE MODEL BENCH
flock -w 14400 /home/flocka/atlas/.gb10.lock bash -c '
  set -u
  cd "$OUT"
  # preflight
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
  memavail=$(awk "/MemAvailable/{print int(\$2/1024/1024)}" /proc/meminfo)
  for w in $(seq 1 60); do
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    memavail=$(awk "/MemAvailable/{print int(\$2/1024/1024)}" /proc/meminfo)
    [ "$apps" = "0" ] && [ "$memavail" -ge 100 ] && break
    echo "preflight wait $w: apps=$apps memavail=${memavail}G" >> preflight.txt; sleep 30
  done
  if [ "$apps" != "0" ] || [ "$memavail" -lt 100 ]; then echo "PREFLIGHT FAIL apps=$apps memavail=${memavail}G" | tee -a preflight.txt; exit 2; fi
  if pgrep -f "[s]park-v[0-9].* serve\|release/spark serve" >/dev/null; then echo "PREFLIGHT FAIL spark serve already running" | tee preflight.txt; exit 2; fi
  echo "preflight ok apps=$apps memavail=${memavail}G $(date -u +%FT%TZ)" > preflight.txt
  sha256sum "$BIN" > binary.sha256
  cp "$ENVFILE" env.txt
  # launch
  ENVS=$(grep -v "^#" "$ENVFILE" | grep -v "^$" | tr "\n" " ")
  CMD="env -i PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin HOME=$HOME LANG=C.UTF-8 RUST_LOG=info ATLAS_NEMO_DUMP=$OUT/dump $ENVS $PRE $BIN serve --model-from-path $MODEL --model-name qwen3.8-flash-next --kernel-target qwen3.8-flash-next --port $PORT --max-seq-len 4096 --max-prefill-tokens 2048 --max-num-seqs 1 --max-batch-size 1 --ssm-cache-slots 16 --kv-cache-dtype bf16 --qwen4-qsa --gpu-memory-utilization 0.90 --oom-guard-mb 4096 --request-timeout 300 --no-tui $EXTRA_ARGS"
  echo "$CMD" > command.txt
  setsid timeout 7200 bash -c "exec $CMD" > server.log 2>&1 &
  SPID=$!
  trap "kill -INT $SPID 2>/dev/null; sleep 5; kill -9 $SPID 2>/dev/null" EXIT
  echo $SPID > server.pid
  # wait ready
  for i in $(seq 1 2400); do
    if curl -s -m 2 http://localhost:$PORT/health 2>/dev/null | grep -q "\"ready\""; then break; fi
    if ! kill -0 $SPID 2>/dev/null; then echo "SERVER DIED" | tee died.txt; tail -20 server.log; exit 3; fi
    sleep 1
  done
  echo "ready after ${i}s" > ready.txt
  sleep 2
  for f in warmup $(seq 1 $NREQ); do
    curl -s -m 600 -o response-$f.json -w "HTTP=%{http_code} TTFT=%{time_starttransfer} TOTAL=%{time_total}\n" -H "Content-Type: application/json" --data-binary @$BENCH/request-$f.json http://localhost:$PORT/v1/completions > timing-$f.txt
    cat timing-$f.txt
    if [ -d "$OUT/dump" ]; then mv "$OUT/dump" "$OUT/dump-$f"; fi
  done
  for f in $(seq 1 $NREQ); do
    curl -s -m 900 -o response-gen-$f.json -H "Content-Type: application/json" --data-binary @$BENCH/request-gen-$f.json http://localhost:$PORT/v1/completions
    rm -rf "$OUT/dump"
  done
  # stop
  kill -INT $SPID 2>/dev/null
  for i in $(seq 1 90); do kill -0 $SPID 2>/dev/null || break; sleep 1; done
  if kill -0 $SPID 2>/dev/null; then echo "FORCE KILL" >> stop.txt; pkill -9 -f "spark serve"; sleep 3; fi
  # ensure GPU clear
  for i in $(seq 1 30); do [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ] && break; sleep 1; done
  echo "stopped $(date -u +%FT%TZ) gpu_apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" >> stop.txt
  # summary
  python3 - <<PY
import json,glob,statistics,os
rows=[]
for f in ["warmup"]+[str(i) for i in range(1,int(os.environ["NREQ"])+1)]:
    try:
        t=open(f"timing-{f}.txt").read().split(); ttft=float([x for x in t if x.startswith("TTFT=")][0][5:])
        r=json.load(open(f"response-{f}.json")); txt=r["choices"][0]["text"]; pt=r["usage"]["prompt_tokens"]; st=r["usage"].get("time_to_first_token_ms")
    except Exception as e:
        rows.append((f,"ERR",str(e))); continue
    rows.append((f,ttft,pt,st,txt))
meas=[r[1] for r in rows if r[0]!="warmup" and r[1]!="ERR"]
print("ARM",os.environ["ARM"])
for r in rows: print(r)
if meas:
    med=statistics.median(meas); print(f"median TTFT {med:.3f}s = {2048/med:.1f} tok/s (n={len(meas)})")
PY
' 2>&1 | tee "$OUT/run.log"
