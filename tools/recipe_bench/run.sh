#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Launch a built-in recipe end to end and measure it: prefill at 2048/8192
# (warm median, rep 1 dropped) and decode at 256 tokens on a coding and a
# chat prompt. Writes one JSON result file per model. GB10-lock-safe: queues
# itself, takes the lock exclusively, preflights, and is trap-safe against
# leaving an orphaned server holding the lock or GPU memory (the failure
# mode that cost this tool's predecessor 17 minutes of shared GPU time on
# 2026-09-23 — see the trap below for what actually went wrong).
#
# usage:
#   run.sh <lane-name> <label> <recipe.yaml> <model_dir> <model_name> <bin> \
#          <prefill_req_2048.json> <prefill_req_8192.json> [port]
#
# The prefill request files are pre-tokenized (`prompt_token_ids`, one token
# repeated/tiled to the target length is fine for a pure throughput probe)
# `/v1/completions` bodies with `max_tokens: 1` — model/tokenizer-specific,
# so this tool does not generate them; point it at existing ones (e.g.
# allmodels-bench/req-*.json) or build fresh ones with the model's tokenizer.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATLAS_ROOT=/home/flocka/atlas
Q="$ATLAS_ROOT/.gb10-queue"
LOCK="$ATLAS_ROOT/.gb10.lock"

LANE=${1:?lane name}
LABEL=${2:?label}
YAML=${3:?recipe yaml path}
MODEL_DIR=${4:?model dir}
MODEL_NAME=${5:?model name}
BIN=${6:?spark binary path}
REQ_2048=${7:?prefill request file, 2048 tokens}
REQ_8192=${8:?prefill request file, 8192 tokens}
PORT=${9:-8897}

OUT="$HERE/results/$LABEL"
mkdir -p "$OUT"

echo "$(date -u +%FT%TZ) $LANE recipe_bench:$LABEL (${YAML##*/}) ~10min QUEUED pid=$$" >> "$Q"
exec 9>"$LOCK"
flock -w 14400 9 || { echo "LOCK TIMEOUT"; exit 1; }
echo "$(date -u +%FT%TZ) $LANE recipe_bench:$LABEL window START pid=$$" >> "$Q"

# Guaranteed cleanup on every exit path (EXIT/INT/TERM/ERR): kill the whole
# process group the server runs in — not a bare PID, which is what let an
# earlier version of this harness orphan a NEVER-READY server (it was
# actually alive on a different port than the health-check was polling; the
# port mismatch is fixed below by generating --port explicitly, but the
# process-group kill and this trap are the defense-in-depth that makes a
# similar future bug fail safely instead of costing the shared queue 17
# minutes) — and wait for it to actually be gone before logging END.
CURRENT_PID=""
cleanup() {
  if [ -n "$CURRENT_PID" ] && kill -0 "$CURRENT_PID" 2>/dev/null; then
    kill -INT -- -"$CURRENT_PID" 2>/dev/null
    for w in $(seq 1 30); do kill -0 "$CURRENT_PID" 2>/dev/null || break; sleep 1; done
    kill -9 -- -"$CURRENT_PID" 2>/dev/null
    for w in $(seq 1 15); do kill -0 "$CURRENT_PID" 2>/dev/null || break; sleep 1; done
  fi
  for w in $(seq 1 30); do nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . || break; sleep 2; done
  echo "$(date -u +%FT%TZ) $LANE recipe_bench:$LABEL window END" >> "$Q"
}
trap cleanup EXIT INT TERM ERR

for i in $(seq 1 60); do
  APPS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
  MA=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  [ "$APPS" -eq 0 ] && [ "$MA" -ge 100 ] && break
  sleep 5
done
echo "preflight apps=$APPS memavail=${MA}GB"
if [ "$APPS" -ne 0 ] || [ "$MA" -lt 100 ]; then echo "PREFLIGHT FAILED"; exit 2; fi

# Generate argv/env from the yaml, never by hand. port is forced to $PORT —
# a mismatch between the yaml's own default port and the port this script
# polls is exactly what caused the orphan referenced above.
python3 "$HERE/generate.py" "$YAML" "$MODEL_DIR" "$MODEL_NAME" env > "$OUT/gen_env.txt"
python3 "$HERE/generate.py" "$YAML" "$MODEL_DIR" "$MODEL_NAME" argv "port=$PORT" > "$OUT/gen_argv.txt"
if ! grep -A1 -- "--port" "$OUT/gen_argv.txt" | tail -1 | grep -qx "$PORT"; then
  echo "ABORT: generated argv does not carry --port $PORT"; cat "$OUT/gen_argv.txt"; exit 3
fi

setsid env -i HOME=$HOME LANG=C.UTF-8 PATH=/usr/local/cuda-13.0/bin:/usr/local/cuda/bin:/usr/bin:/bin \
  LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:/usr/local/cuda/lib64 RUST_LOG=info \
  $(cat "$OUT/gen_env.txt" | xargs -d '\n') \
  timeout -s INT -k 60 1200 "$BIN" $(cat "$OUT/gen_argv.txt") 9>&- > "$OUT/server.log" 2>&1 &
CURRENT_PID=$!
echo $CURRENT_PID > "$OUT/pid"

READY=0
for i in $(seq 1 400); do
  curl -sf -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ready && { READY=1; break; }
  kill -0 "$CURRENT_PID" 2>/dev/null || { echo "SERVER DIED during load"; tail -25 "$OUT/server.log"; break; }
  sleep 2
done
if [ "$READY" != 1 ]; then
  echo '{"label":"'"$LABEL"'","status":"load_failed"}' > "$OUT/result.json"
  echo "LOAD FAILED"
  exit 1
fi
echo "READY"

# Proof the server actually saw the generated env, not just that we intended
# to set it.
REAL_PID=$(pgrep -f "bin/spark serve|target/release/spark serve" | tail -1)
if [ -n "$REAL_PID" ] && [ -r "/proc/$REAL_PID/environ" ]; then
  tr '\0' '\n' < "/proc/$REAL_PID/environ" | grep '^ATLAS_' | sort > "$OUT/actual_env.txt"
  ENV_MATCH=$(diff <(sort "$OUT/gen_env.txt") "$OUT/actual_env.txt" > /dev/null 2>&1 && echo true || echo false)
else
  ENV_MATCH=unknown
fi
echo "environ proof: match=$ENV_MATCH"

prefill_warm_median() {
  local req=$1 n=$2
  for i in 1 2 3 4; do
    curl -s -m 300 -o "$OUT/prefill-$n-$i.json" -H 'Content-Type: application/json' --data-binary @"$req" "http://127.0.0.1:$PORT/v1/completions"
  done
}

decode_reps() {
  local label=$1 req=$2
  for i in warmup 1; do
    S=$(date +%s.%N)
    curl -s -m 180 -o "$OUT/decode-$label-$i.json" -H 'Content-Type: application/json' --data-binary @"$req" "http://127.0.0.1:$PORT/v1/chat/completions"
    E=$(date +%s.%N); echo "$S $E" > "$OUT/decode-$label-$i.wall"
  done
}

prefill_warm_median "$REQ_2048" 2048
prefill_warm_median "$REQ_8192" 8192
decode_reps coding "$HERE/prompts/coding.json"
decode_reps chat "$HERE/prompts/chat.json"

grep -a -iE 'panic|CUDA_ERROR|error' "$OUT/server.log" | grep -v -i 'error_rate\|0 error' > "$OUT/errors.txt" || true

python3 - "$OUT" "$LABEL" "$ENV_MATCH" <<'PY'
import json, statistics, sys, hashlib

out, label, env_match = sys.argv[1], sys.argv[2], sys.argv[3]

def prefill(n):
    rows = []
    for i in [1, 2, 3, 4]:
        try:
            r = json.load(open(f"{out}/prefill-{n}-{i}.json"))
            if "error" in r:
                rows.append({"rep": i, "error": r["error"]})
                continue
            u = r["usage"]
            rows.append({"rep": i, "prompt_tokens": u["prompt_tokens"], "ttft_ms": u["time_to_first_token_ms"]})
        except Exception as e:
            rows.append({"rep": i, "parse_error": str(e)})
    warm = [row["ttft_ms"] for row in rows if row.get("rep") != 1 and "ttft_ms" in row]
    result = {"reps": rows}
    if warm:
        med = statistics.median(warm)
        pt = next((row["prompt_tokens"] for row in rows if "prompt_tokens" in row), None)
        result["warm_median_ttft_ms"] = med
        if pt:
            result["tok_s"] = pt / med * 1000
    return result

def decode(label_):
    try:
        r = json.load(open(f"{out}/decode-{label_}-1.json"))
        if "error" in r:
            return {"error": r["error"]}
        u = r["usage"]; msg = r["choices"][0]["message"]; text = msg.get("content") or ""
        s, e = map(float, open(f"{out}/decode-{label_}-1.wall").read().split())
        wall = (e - s) * 1000
        ttft = u.get("time_to_first_token_ms", 0.0)
        ct = u["completion_tokens"]
        dec = (ct - 1) / ((wall - ttft) / 1000) if wall > ttft and ct > 1 else None
        return {
            "completion_tokens": ct,
            "finish_reason": r["choices"][0]["finish_reason"],
            "decode_tok_s": dec,
            "text_sha256_16": hashlib.sha256(text.encode()).hexdigest()[:16],
        }
    except Exception as e:
        return {"parse_error": str(e)}

result = {
    "label": label,
    "status": "ok",
    "env_proof_match": env_match,
    "prefill_2048": prefill(2048),
    "prefill_8192": prefill(8192),
    "decode_coding": decode("coding"),
    "decode_chat": decode("chat"),
}
json.dump(result, open(f"{out}/result.json", "w"), indent=2)
print(json.dumps(result, indent=2))
PY

echo "DONE"
