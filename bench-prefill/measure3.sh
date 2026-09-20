#!/usr/bin/env bash
# One server-start -> warmup + N measured -> server-stop cycle under the box lock.
# Usage: measure.sh <label> <elf> <envfile> [extra KEY=VAL ...]
#   env: TRIALS (default 5), NSYS=1 to wrap the server in nsys, KEEP_LOG=1
set -euo pipefail
LABEL=$1; ELF=$2; ENVFILE=$3; shift 3
BENCH=/home/flocka/atlas/glm53-prefill-work/bench
OUT=$BENCH/runs/$LABEL; mkdir -p "$OUT"
TRIALS=${TRIALS:-5}
PORT=8893
LOCK=/home/flocka/atlas/.gb10.lock

exec 9>"$LOCK"
echo "[$(date +%T)] waiting for GPU lock" >&2
flock -w 14400 9
echo "[$(date +%T)] lock acquired" >&2

cleanup() {
  if [ -n "${SRV_PID:-}" ] && kill -0 "$SRV_PID" 2>/dev/null; then
    kill -INT "$SRV_PID" 2>/dev/null || true
    for _ in $(seq 1 60); do kill -0 "$SRV_PID" 2>/dev/null || break; sleep 1; done
    kill -9 "$SRV_PID" 2>/dev/null || true
  fi
  # nsys leaves a child; make sure no spark serve survives
  pkill -f "serve --model-from-path /home/flocka/models/GLM-5.3-Flash" 2>/dev/null || true
  sleep 2
  flock -u 9 || true
  echo "[$(date +%T)] server stopped, lock released" >&2
}
trap cleanup EXIT

# preflight
for pid in $(ls /proc | grep -E '^[0-9]+$'); do
  exe=$(readlink -f /proc/$pid/exe 2>/dev/null) || continue
  case "$(basename "$exe")" in spark-dashboard) ;; spark|spark-*|llama-server|llama-cli|vllm) echo "PREFLIGHT: model process alive: $pid $exe" >&2; exit 2;; esac
done
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then
  echo "PREFLIGHT: GPU compute apps present" >&2; exit 2
fi
AVAIL_KB=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
if [ "$AVAIL_KB" -lt $((100*1024*1024)) ]; then
  echo "PREFLIGHT: MemAvailable $((AVAIL_KB/1024/1024)) GB < 100 GB" >&2; exit 2
fi

sha256sum "$ELF" | tee "$OUT/elf.sha256"
readelf -n "$ELF" | grep "Build ID" | tee "$OUT/elf.buildid"
echo "glm5.3-flash strings: $(grep -ac 'glm5.3-flash' "$ELF")" | tee "$OUT/elf.grep"

# environment: file + extra overrides
ENV_ARGS=()
while IFS= read -r line; do [ -n "$line" ] && ENV_ARGS+=("$line"); done < "$ENVFILE"
for kv in "$@"; do ENV_ARGS+=("$kv"); done
printf '%s\n' "${ENV_ARGS[@]}" > "$OUT/env.effective"
mapfile -t ARGV < "$BENCH/argv.p57"

RUNNER=()
if [ "${NSYS:-0}" = "1" ]; then
  RUNNER=(/usr/local/bin/nsys profile --trace=cuda,nvtx --cuda-graph-trace=node -o "$OUT/profile" --force-overwrite true)
fi

env -i "${ENV_ARGS[@]}" "${RUNNER[@]}" timeout -s INT 3600 "$ELF" "${ARGV[@]}" > "$OUT/server.log" 2>&1 &
SRV_PID=$!
echo "[$(date +%T)] server pid $SRV_PID" >&2

# wait ready
for i in $(seq 1 600); do
  if curl -sf "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ready"'; then break; fi
  if ! kill -0 "$SRV_PID" 2>/dev/null; then echo "server died; see $OUT/server.log" >&2; tail -30 "$OUT/server.log" >&2; exit 3; fi
  sleep 1
done
curl -s "http://127.0.0.1:$PORT/health" > "$OUT/health.json"; echo >> "$OUT/health.json"
echo "[$(date +%T)] ready: $(cat "$OUT/health.json")" >&2

req() {
  local name=$1
  local t0 t1
  t0=$(date +%s.%N)
  curl -s -X POST "http://127.0.0.1:$PORT/v1/completions" -H 'content-type: application/json' \
    --data-binary @"$BENCH/request.json" -o "$OUT/$name.response.json"
  t1=$(date +%s.%N)
  python3 - "$OUT/$name.response.json" "$t0" "$t1" "$name" <<'EOF'
import json,sys
r=json.load(open(sys.argv[1])); wall=(float(sys.argv[3])-float(sys.argv[2]))*1000
u=r.get('usage',{}); ttft=u.get('time_to_first_token_ms')
print(f"{sys.argv[4]}: ttft_ms={ttft:.1f} wall_ms={wall:.1f} prompt_tokens={u.get('prompt_tokens')} text={r['choices'][0]['text']!r} tok_s={u.get('prompt_tokens')/ttft*1000:.1f}")
EOF
}
req warmup | tee "$OUT/timing.txt"
for i in $(seq 1 "$TRIALS"); do req "trial-$i" | tee -a "$OUT/timing.txt"; done
python3 - "$OUT/timing.txt" <<'EOF'
import re,sys,statistics
v=[float(m.group(1)) for l in open(sys.argv[1]) if l.startswith('trial') for m in [re.search(r'ttft_ms=([\d.]+)',l)] if m]
print(f"MEDIAN ttft_ms={statistics.median(v):.1f}  tok/s={2047/statistics.median(v)*1000:.1f}  min={min(v):.1f} max={max(v):.1f}")
EOF
sha256sum "$OUT"/trial-*.response.json | sed 's/.*runs\///' > "$OUT/responses.sha256"
python3 -c "
import json,glob,sys
t=[json.load(open(f))['choices'][0]['text'] for f in sorted(glob.glob('$OUT/trial-*.response.json'))]
print('TEXTS', t)
"
# numeric oracle: top-20 logprobs of the first generated token
curl -s -X POST "http://127.0.0.1:$PORT/v1/completions" -H 'content-type: application/json' \
  --data-binary @"$BENCH/request-lp.json" -o "$OUT/probe-logprobs.json"
python3 - "$OUT/probe-logprobs.json" <<'EOF2'
import json,sys
r=json.load(open(sys.argv[1])); lp=r['choices'][0].get('logprobs') or {}
tl=(lp.get('top_logprobs') or [{}])[0]
items=sorted(tl.items(), key=lambda kv:-kv[1])[:20]
print('TOPLOGPROBS', r['usage'].get('time_to_first_token_ms'), json.dumps(items))
EOF2
sha256sum "$OUT/probe-logprobs.json" >> "$OUT/responses.sha256"
