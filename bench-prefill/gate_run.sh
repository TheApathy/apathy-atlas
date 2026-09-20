#!/usr/bin/env bash
# Teacher-forced gate: one server, all prompts, full-logits dumps. Usage: gate_run.sh <label> <elf> [KEY=VAL...]
set -euo pipefail
LABEL=$1; ELF=$2; shift 2
BENCH=/home/flocka/atlas/glm53-prefill-work/bench; OUT=$BENCH/runs/gate-$LABEL; mkdir -p "$OUT/logits"
PORT=8893; LOCK=/home/flocka/atlas/.gb10.lock
exec 9>"$LOCK"; echo "[$(date +%T)] waiting for GPU lock" >&2; flock -w 14400 9; echo "[$(date +%T)] lock acquired" >&2
cleanup() { [ -n "${SRV_PID:-}" ] && kill -INT "$SRV_PID" 2>/dev/null; for _ in $(seq 1 60); do kill -0 "${SRV_PID:-0}" 2>/dev/null || break; sleep 1; done; kill -9 "${SRV_PID:-0}" 2>/dev/null || true; sleep 2; flock -u 9 || true; echo "[$(date +%T)] server stopped, lock released" >&2; }
trap cleanup EXIT
for pid in $(ls /proc | grep -E '^[0-9]+$'); do exe=$(readlink -f /proc/$pid/exe 2>/dev/null) || continue; case "$(basename "$exe")" in spark-dashboard) ;; spark|spark-*|llama-server|llama-cli|vllm) echo "PREFLIGHT: model process alive: $pid $exe" >&2; exit 2;; esac; done
[ "$(awk '/MemAvailable/{print $2}' /proc/meminfo)" -ge $((100*1024*1024)) ] || { echo "PREFLIGHT: low memory" >&2; exit 2; }
sha256sum "$ELF" | tee "$OUT/elf.sha256"
ENV_ARGS=(); while IFS= read -r l; do [ -n "$l" ] && ENV_ARGS+=("$l"); done < "$BENCH/env.p57"; for kv in "$@"; do ENV_ARGS+=("$kv"); done
ENV_ARGS+=("ATLAS_GLM53_LOGITS_DUMP=$OUT/logits" "ATLAS_GLM53_LOGITS_DUMP_ALL=1" "ATLAS_GLM53_EXL3_LAST_ROW_HEAD=0")
printf '%s\n' "${ENV_ARGS[@]}" > "$OUT/env.effective"; mapfile -t ARGV < "$BENCH/argv.p57"
env -i "${ENV_ARGS[@]}" timeout -s INT 3600 "$ELF" "${ARGV[@]}" > "$OUT/server.log" 2>&1 & SRV_PID=$!
for i in $(seq 1 600); do curl -sf "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ready"' && break; kill -0 "$SRV_PID" 2>/dev/null || { echo "server died" >&2; tail -20 "$OUT/server.log" >&2; exit 3; }; sleep 1; done
python3 - "$OUT" "$BENCH/prompts_1024.json" <<'PY'
import json, sys, urllib.request, time
out, pf = sys.argv[1], sys.argv[2]
for i, p in enumerate(json.load(open(pf))):
    body = json.dumps({"model": "glm53-flash", "prompt": p["ids"], "max_tokens": 1, "temperature": 0, "stream": False}).encode()
    t0 = time.time(); r = json.load(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:8893/v1/completions", body, {"content-type": "application/json"}), timeout=600))
    print(f"{i} {p['name']}: ttft={r['usage']['time_to_first_token_ms']:.0f} prompt_tokens={r['usage']['prompt_tokens']} text={r['choices'][0]['text']!r} wall={time.time()-t0:.1f}s", flush=True)
PY
ls -la "$OUT/logits"
