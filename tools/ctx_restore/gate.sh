#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Byte-exact gate for on-disk context checkpoints (ATLAS_CTX_CACHE=1).
#
# One GB10 window, one server process per arm, all launched from the
# built-in recipe via tools/recipe_bench/generate.py:
#   write      turn 1 (C tokens) for every case -> checkpoints on disk
#   restore    turn 2 (N tokens): restores C from disk, prefills N-C   [new process]
#   reference  turn 2 with the same index but no restore: full prefill with
#              a forced chunk boundary at C (same grid + C)
#   ctl_conv   restore with the GDN conv state deliberately NOT restored:
#              MUST differ from reference (proves the gate can fail)
#   ctl_bad    one checkpoint byte flipped + one forged stale-token file:
#              both MUST be rejected (logged, deleted), then full prefill
#   ctl_key    one extra ATLAS_* env var: MUST see zero checkpoints
# PASS iff restore == reference byte for byte (logits and output tokens) on
# every case, ctl_conv differs on every case, and every control rejects.
#
# usage: gate.sh <lane> <label> <bin> <prompts_dir> <case C:N>...
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATLAS_ROOT=/home/flocka/atlas
Q="$ATLAS_ROOT/.gb10-queue"
LOCK="$ATLAS_ROOT/.gb10.lock"
YAML="$HERE/../../crates/spark-server/src/recipe/builtin/qwen3.8-27b-optimized-local.yaml"
MODEL_DIR=/home/flocka/atlas/qwen38/optimized-qwen
PORT=8897

LANE=${1:?lane}; LABEL=${2:?label}; BIN=${3:?spark binary}; PROMPTS=${4:?prompts dir}; shift 4
CASES=("$@")
[ ${#CASES[@]} -gt 0 ] || { echo "no cases"; exit 64; }
MAXLEN=0
for cn in "${CASES[@]}"; do n=${cn#*:}; [ "$n" -gt "$MAXLEN" ] && MAXLEN=$n; done
MAXLEN=$(( (MAXLEN / 1024 + 2) * 1024 ))

OUT="$HERE/results/$LABEL"
CKPT="$OUT/ckpt"
rm -rf "$OUT"; mkdir -p "$OUT" "$CKPT"

echo "$(date -u +%FT%TZ) $LANE ctx-gate:$LABEL (q38 27B, 6 server arms, <=30min) QUEUED pid=$$" >> "$Q"
exec 9>>"$LOCK"
flock -w 14400 9 || { echo "LOCK TIMEOUT"; exit 1; }
echo "$(date -u +%FT%TZ) $LANE ctx-gate:$LABEL window START pid=$$" >> "$Q"

CURRENT_PID=""; WATCH_PID=""; CLEANED_UP=0
stop_server() {
  if [ -n "$CURRENT_PID" ] && kill -0 "$CURRENT_PID" 2>/dev/null; then
    kill -INT -- -"$CURRENT_PID" 2>/dev/null
    for w in $(seq 1 30); do kill -0 "$CURRENT_PID" 2>/dev/null || break; sleep 1; done
    kill -9 -- -"$CURRENT_PID" 2>/dev/null
    for w in $(seq 1 15); do kill -0 "$CURRENT_PID" 2>/dev/null || break; sleep 1; done
  fi
  CURRENT_PID=""
  for w in $(seq 1 30); do nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q . || break; sleep 2; done
}
cleanup() {
  [ "$CLEANED_UP" = 1 ] && return 0
  CLEANED_UP=1
  [ -n "$WATCH_PID" ] && kill "$WATCH_PID" 2>/dev/null
  stop_server
  # Checkpoints are large; the gate keeps only their listing.
  ls -la "$CKPT"/*/ > "$OUT/ckpt_listing.txt" 2>/dev/null
  rm -rf "$CKPT"
  echo "$(date -u +%FT%TZ) $LANE ctx-gate:$LABEL window END" >> "$Q"
}
trap cleanup EXIT INT TERM

preflight() {  # $1 = minimum MemAvailable GB
  local apps ma
  for i in $(seq 1 60); do
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
    ma=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    [ "$apps" -eq 0 ] && [ "$ma" -ge "$1" ] && { echo "preflight ok apps=$apps memavail=${ma}GB"; return 0; }
    sleep 5
  done
  echo "PREFLIGHT FAILED apps=$apps memavail=${ma}GB (need $1)"; return 1
}
preflight 100 || exit 2

# Watchdog: SIGKILL only our own server group if the host drops below 6 GB.
( while sleep 1; do
    ma=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    if [ "$ma" -lt 6 ] && [ -s "$OUT/pid" ]; then
      p=$(cat "$OUT/pid"); echo "WATCHDOG memavail=${ma}GB kill $p" >> "$OUT/watchdog.log"
      kill -9 -- -"$p" 2>/dev/null
    fi
  done ) 9>&- &
WATCH_PID=$!

python3 "$HERE/../recipe_bench/generate.py" "$YAML" "$MODEL_DIR" m env > "$OUT/gen_env.txt"
python3 "$HERE/../recipe_bench/generate.py" "$YAML" "$MODEL_DIR" m argv "port=$PORT" "host=127.0.0.1" \
  "max_model_len=$MAXLEN" > "$OUT/gen_argv.txt"

launch() {  # $1 = arm, rest = extra KEY=VAL env
  local arm=$1; shift
  preflight 16 || return 1
  mkdir -p "$OUT/$arm"
  setsid env -i HOME=$HOME LANG=C.UTF-8 PATH=/usr/local/cuda-13.0/bin:/usr/local/cuda/bin:/usr/bin:/bin \
    LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:/usr/local/cuda/lib64 RUST_LOG=info \
    $(xargs -d '\n' < "$OUT/gen_env.txt") \
    ATLAS_CTX_CACHE=1 ATLAS_CTX_CACHE_DIR="$CKPT" ATLAS_CTX_GATE_DUMP="$OUT/$arm" "$@" \
    timeout -s INT -k 60 1500 "$BIN" $(cat "$OUT/gen_argv.txt") 9>&- > "$OUT/$arm/server.log" 2>&1 &
  CURRENT_PID=$!
  echo $CURRENT_PID > "$OUT/pid"
  for i in $(seq 1 400); do
    curl -sf -m 2 "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q ready && { echo "[$arm] READY"; return 0; }
    kill -0 "$CURRENT_PID" 2>/dev/null || { echo "[$arm] SERVER DIED"; tail -25 "$OUT/$arm/server.log"; return 1; }
    sleep 2
  done
  echo "[$arm] LOAD TIMEOUT"; return 1
}

send() {  # $1 = arm, $2 = request name
  curl -s -m 900 -o "$OUT/$1/$2.resp.json" -H 'Content-Type: application/json' \
    --data-binary @"$PROMPTS/$2.json" "http://127.0.0.1:$PORT/v1/completions"
  python3 -c "import json,sys; r=json.load(open(sys.argv[1])); u=r.get('usage',{}); print('[%s] %s ttft_ms=%s prompt=%s' % (sys.argv[2], sys.argv[3], u.get('time_to_first_token_ms'), u.get('prompt_tokens')))" \
    "$OUT/$1/$2.resp.json" "$1" "$2" 2>/dev/null || echo "[$1] $2 BAD RESPONSE: $(head -c 300 "$OUT/$1/$2.resp.json")"
}

wait_written() {  # $1 = arm, $2 = expected checkpoint count
  for i in $(seq 1 300); do
    [ "$(grep -c 'checkpoint written' "$OUT/$1/server.log")" -ge "$2" ] && return 0
    grep -q 'checkpoint write failed' "$OUT/$1/server.log" && break
    sleep 1
  done
  echo "[$1] checkpoints not all written"; return 1
}

run_turn2() {  # $1 = arm, extra env...
  local arm=$1; shift
  launch "$arm" ATLAS_CTX_MIN_TOKENS=100000000 "$@" || return 1
  for cn in "${CASES[@]}"; do send "$arm" "p2-${cn%:*}-${cn#*:}"; done
  stop_server
}

# ── write ──
launch write || exit 3
for cn in "${CASES[@]}"; do send write "p1-${cn%:*}"; done
wait_written write "${#CASES[@]}" || exit 4
stop_server
ls -la "$CKPT"/*/ | tee "$OUT/ckpt_after_write.txt"

run_turn2 restore || exit 5
run_turn2 reference ATLAS_CTX_CACHE_MODE=reference || exit 5
run_turn2 ctl_conv ATLAS_CTX_GATE_SKIP=conv || exit 5

# ── ctl_bad: flip one byte in the first case's state; forge a stale-token
# file for the second case (a copy of the first case's file renamed to the
# forged prompt's prefix hash).
KEYDIR=$(ls -d "$CKPT"/*/ | head -1)
first=${CASES[0]}; C1=${first%:*}
F1=$(ls "$KEYDIR" | grep "^$(printf %010d "$C1")-")
python3 - "$KEYDIR/$F1" <<'PY'
import os, sys
p = sys.argv[1]; size = os.path.getsize(p)
with open(p, "r+b") as f:
    f.seek(size // 2); b = f.read(1); f.seek(size // 2); f.write(bytes([b[0] ^ 0x40]))
print("flipped byte", size // 2, "of", p)
PY
FORGE_CASE=${CASES[1]:-${CASES[0]}}
FC=${FORGE_CASE%:*}; FN=${FORGE_CASE#*:}
FH=$(python3 -c "import json,sys; m=[x for x in json.load(open(sys.argv[1])) if x['c']==int(sys.argv[2])][0]; print(m['forged_hash'])" "$PROMPTS/manifest.json" "$FC")
SRC=$(ls "$KEYDIR" | grep "^$(printf %010d "$FC")-")
cp "$KEYDIR/$SRC" "$KEYDIR/$(printf %010d "$FC")-$FH.ckpt"
launch ctl_bad ATLAS_CTX_MIN_TOKENS=100000000 || exit 6
send ctl_bad "p2-${C1}-${first#*:}"
send ctl_bad "forged-${FC}-${FN}"
stop_server

launch ctl_key ATLAS_GATE_WRONG_RECIPE=1 ATLAS_CTX_MIN_TOKENS=100000000 || exit 7
send ctl_key "p2-${FC}-${FN}"
stop_server

python3 "$HERE/compare.py" "$OUT" "${CASES[@]}" | tee "$OUT/verdict.txt"
