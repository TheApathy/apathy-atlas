#!/usr/bin/env bash
# First-token sanity check for a running DeepSeek-V4-Flash Atlas serve.
# Mirrors ds4-on-spark/scripts/smoke-test.sh: ask the capital of France,
# assert "Paris" appears. A pass proves the loader + kernels + sampler produce
# coherent tokens end to end (not just that the server booted).
#
#   PORT=8899 bash bench/deepseek-v4/smoke.sh
#
set -uo pipefail
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8899}"
MODEL="${MODEL_NAME:-deepseek-v4-flash}"
PROMPT="${PROMPT:-What is the capital of France? Answer in one sentence.}"

url="http://$HOST:$PORT/v1/chat/completions"
echo "POST $url  (model=$MODEL)"
resp="$(curl -s "$url" -H 'Content-Type: application/json' -d "$(python3 - "$MODEL" "$PROMPT" <<'PY'
import json,sys
print(json.dumps({
  "model": sys.argv[1],
  "messages": [{"role":"user","content": sys.argv[2]}],
  "max_tokens": 64,
  "temperature": 0.0,
  "stream": False
}))
PY
)")"
echo "--- raw response (truncated) ---"
echo "$resp" | head -c 1200; echo
text="$(echo "$resp" | python3 -c 'import sys,json;
try:
  d=json.load(sys.stdin); print(d["choices"][0]["message"]["content"])
except Exception as e:
  print("PARSE_ERROR:",e)' 2>/dev/null)"
echo "--- completion ---"; echo "$text"
if echo "$text" | grep -qi "paris"; then
  echo "SMOKE: PASS (found Paris)"; exit 0
else
  echo "SMOKE: FAIL (no Paris in output)"; exit 1
fi
