#!/usr/bin/env bash
# Measure decode throughput + speculative acceptance against a running server.
#
# Usage: scripts/dspark-bench.sh <log-file> [max_tokens] [prompt]
#
# Prints: tok/s (server-reported), tok/step acceptance, and the generated text
# (so greedy outputs can be diffed across configs — a lossless speculative
# decoder MUST produce byte-identical text to plain greedy decode).
set -uo pipefail

LOG="${1:?usage: dspark-bench.sh <log-file> [max_tokens] [prompt]}"
MAXTOK="${2:-200}"
PROMPT="${3:-Explain how paged attention works in a modern LLM inference server.}"
PORT="${PORT:-8977}"

# Wait for the model to finish loading (warmup is 30s+ on this checkpoint).
until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
  if ! pgrep -f "target/release/spark serve" >/dev/null; then
    echo "server died during warmup — see $LOG"; tail -20 "$LOG"; exit 1
  fi
  sleep 3
done

MARK=$(wc -l <"$LOG")

RESP=$(curl -s "http://127.0.0.1:$PORT/v1/completions" \
  -H 'Content-Type: application/json' \
  -d "$(python3 -c '
import json,sys
print(json.dumps({"model":"deepseek","prompt":sys.argv[1],
                  "max_tokens":int(sys.argv[2]),"temperature":0.0}))' "$PROMPT" "$MAXTOK")")

echo "=== generated text ==="
python3 -c '
import json,sys
try:
    d = json.loads(sys.stdin.read())
    print(d["choices"][0]["text"])
except Exception as e:
    print("PARSE FAIL:", e)
' <<<"$RESP"

echo "=== throughput / acceptance (log lines since request) ==="
tail -n "+$MARK" "$LOG" | grep -oE "tok/s[^,]*|accepted=[0-9]+|tok/step[^,]*" | tail -40

echo "=== summary ==="
tail -n "+$MARK" "$LOG" | awk '
  match($0, /([0-9.]+) tok\/s/, m) { s+=m[1]; n++ }
  match($0, /accepted=([0-9]+)/, a) { acc+=a[1]+1; steps++ }
  END {
    if (n)     printf "mean tok/s      : %.2f  (%d samples)\n", s/n, n
    if (steps) printf "mean tok/step   : %.3f  (%d verify steps)\n", acc/steps, steps
    if (!n && !steps) print "no throughput lines found — check the log format"
  }'
