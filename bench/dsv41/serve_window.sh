#!/usr/bin/env bash
# ONE GPU window: `spark serve` DeepSeek-V4.1 (keep=124 arena, ~84 GB) and exercise it over HTTP.
# Same safety rules as keep124_window.sh: preflight (nothing resident, >= 100 GB available),
# START only once the lock is held, a 100 ms MemAvailable watchdog that SIGKILLs the server
# (by its own PID) below 6 GB, END always with rc and the low-water mark.
# Usage: serve_window.sh <label> <port> <requests-dir>
#   <requests-dir>/NN_name.json : POST bodies for /v1/chat/completions (or /v1/completions when the
#                                 file name contains "completion_raw"); responses land next to them.
set -uo pipefail
LABEL="$1"; PORT="$2"; REQ="$3"
Q=/home/flocka/atlas/.gb10-queue; LOCK=/home/flocka/atlas/.gb10.lock
BIN=/home/flocka/atlas/dsv41-integration/target/release/spark
MODEL=/home/flocka/models/DeepSeek-V4.1-Flash-Next-DGX-Spark-512K
ABORT_KB=$(( ${ABORT_GB:-6} * 1024 * 1024 )); NEED_KB=$((100 * 1024 * 1024))
memavail() { awk '/^MemAvailable:/ {print $2}' /proc/meminfo; }
[ -x "$BIN" ] || { echo "FATAL: $BIN missing"; echo "DONE $LABEL rc=99"; exit 99; }

echo "$(date -u +%FT%TZ) dsv41-integrate SERVE $LABEL (port $PORT) QUEUED pid=$$" >> "$Q"
exec 9>>"$LOCK"
flock -w 14400 9 || { echo "DONE $LABEL rc=98"; exit 98; }
echo "$(date -u +%FT%TZ) dsv41-integrate SERVE $LABEL window START (lock held) pid=$$" >> "$Q"
low=$(memavail); rc=0; srv=""
end() {
  [ -n "$srv" ] && kill -0 "$srv" 2>/dev/null && { kill -TERM "$srv"; sleep 5; kill -0 "$srv" 2>/dev/null && kill -9 "$srv"; }
  echo "low-water MemAvailable: $((low/1024/1024)) GB"
  echo "$(date -u +%FT%TZ) dsv41-integrate SERVE $LABEL window END rc=$1 lowwater_GB=$((low/1024/1024)) pid=$$" >> "$Q"
  flock -u 9; echo "DONE $LABEL rc=$1 lowwater_GB=$((low/1024/1024))"; exit "$1"
}
others=$(docker ps --format '{{.Names}}' | grep -Ei 'dsv41|atlas|spark|vllm' | grep -v buildkit || true)
# GPU-resident candidates by EXECUTABLE name (argv[0]), not by any command line that merely
# mentions one: a queued job's wrapper shell contains "examples/dsv41_forward" in its script text.
procs=$(ps -eo pid=,args= | awk '{split($2,a,"/"); e=a[length(a)]} e=="spark" || e=="dsv41_forward" || e=="dsv41_drop_gate" || (e ~ /^python/ && ($0 ~ /v41_engine|capture_ref/))' | grep -v "^ *$$ " || true)
avail=$(memavail)
echo "preflight: MemAvailable $((avail/1024/1024)) GB; containers: [${others}]; procs: [${procs}]"
if [ -n "$others" ] || [ -n "$procs" ] || [ "$avail" -lt "$NEED_KB" ]; then echo "PREFLIGHT FAILED"; end 97; fi
if ss -ltn | grep -q ":$PORT "; then echo "PORT $PORT IN USE (serve would leak its scheduler on bind failure)"; end 96; fi
echo "memory plan: arena 71.67 + dense store 9.99 + fp32 routers 0.31 + scratch/rings/score ~1 + context ~1 = ~84 GB vs $((avail/1024/1024)) GB available"

export ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=deepseek-v4.1 ATLAS_TARGET_QUANT=cb3
export ATLAS_DSV41_MODEL_DIR=$MODEL ATLAS_DSV41_CHUNK=512
"$BIN" serve --model-from-path "$MODEL" --port "$PORT" --max-seq-len "${SERVE_MAX_SEQ:-8192}" --max-batch-size 1 \
  --kv-cache-dtype bf16 > "$REQ/server.log" 2>&1 &
srv=$!
echo "server pid $srv"
ready=0
for i in $(seq 1 12000); do  # up to 20 min of 100 ms ticks for the load
  m=$(memavail); [ "$m" -lt "$low" ] && low=$m
  if [ "$m" -lt "$ABORT_KB" ]; then echo "WATCHDOG: MemAvailable $((m/1024)) MB -> SIGKILL $srv"; kill -9 "$srv"; end 95; fi
  kill -0 "$srv" 2>/dev/null || { echo "server exited during load:"; tail -30 "$REQ/server.log"; end 94; }
  if (( i % 20 == 0 )) && curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ready=1; break; fi
  sleep 0.1
done
[ "$ready" = 1 ] || { echo "server never became healthy"; tail -30 "$REQ/server.log"; end 93; }
echo "server healthy after ~$((i/10)) s"

# watchdog continues in the background during requests
( while kill -0 "$srv" 2>/dev/null; do m=$(memavail); [ "$m" -lt "$ABORT_KB" ] && { echo "WATCHDOG (requests): SIGKILL $srv"; kill -9 "$srv"; }; echo "$m" >> "$REQ/.mem"; sleep 0.1; done ) &
for f in "$REQ"/*.json; do
  case "$f" in *.response.json) continue ;; esac
  ep=/v1/chat/completions; case "$f" in *completion_raw*) ep=/v1/completions ;; esac
  t0=$(date +%s.%N)
  curl -s -m 900 -H 'Content-Type: application/json' -d @"$f" "http://127.0.0.1:$PORT$ep" > "${f%.json}.response.json"; crc=$?
  t1=$(date +%s.%N)
  echo "request $(basename "$f") -> $ep curl rc=$crc in $(echo "$t1 - $t0" | bc) s"
  [ "$crc" -ne 0 ] && rc=$crc
done
if [ -f "$REQ/.mem" ]; then m=$(sort -n "$REQ/.mem" | head -1); [ "$m" -lt "$low" ] && low=$m; rm -f "$REQ/.mem"; fi
end "$rc"
