#!/bin/bash
# Throughput bench for the DFlash γ=16 work: streams a cat-story prompt
# (the actual target workload — creative prose, NOT counting) and reports
# decode tok/s measured client-side from streaming chunk arrival times.
#
# Usage: bash bench-catstory.sh [PORT] [MAX_TOKENS] [MODEL_NAME]
set -euo pipefail

PORT=${1:-8890}
MAX_TOKENS=${2:-400}
MODEL=${3:-aeon-27b-dflash}
PROMPT=${PROMPT:-"Write a short, warm bedtime story about a curious orange cat named Miso who discovers a hidden garden behind the bakery. Keep it about 300 words."}

payload=$(jq -n --arg m "$MODEL" --arg p "$PROMPT" --argjson n "$MAX_TOKENS" \
  '{model:$m, max_tokens:$n, stream:true, messages:[{role:"user",content:$p}]}')

curl -sN "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "$payload" \
| awk '
  /^data: \[DONE\]/ { done=1; next }
  /^data: / {
    now = systime_ms()
    if (first == 0) { first = now }
    last = now
    chunks++
  }
  function systime_ms(   cmd, t) {
    cmd = "date +%s%3N"; cmd | getline t; close(cmd); return t + 0
  }
  END {
    if (chunks > 1) {
      dur = (last - first) / 1000.0
      # chunks-1: rate over the decode interval, excluding TTFT
      printf "chunks=%d decode_window=%.2fs decode_tok/s=%.2f (chunk≈token)\n", chunks, dur, (chunks - 1) / dur
    } else {
      print "insufficient chunks for measurement"
    }
  }'
