#!/usr/bin/env bash
# One exclusive GB10 window running a list of GLM-5.3 arms, one server per arm.
# Usage: window.sh <window-label> <elf> <arm> [<arm> ...]
#   arm = NAME:MODE[:KEY=VAL,KEY=VAL...]   MODE = target | dflash | dflashN (N = max drafts)
# Each arm: fresh server -> warm-up -> TRIALS x PROMPTS greedy 160-token decodes -> stop.
# Output: runs/<window>/<arm>/ ; summary via score.py afterwards.
set -uo pipefail
WIN=$1; ELF=$2; shift 2
B=/home/flocka/atlas/glm-spec-bench
OUT=$B/runs/$WIN; mkdir -p "$OUT"
Q=/home/flocka/atlas/.gb10-queue; LOCK=/home/flocka/atlas/.gb10.lock
TRIALS=${TRIALS:-2}
PROMPTS=${PROMPTS:-short long prose}
PORT=8893
MODEL=/home/flocka/models/GLM-5.3-Flash-exl3-2.05bpw
DRAFT=/home/flocka/models/GLM-5.3-Flash-DFlash2-bf582e4
log() { echo "[$(date -u +%T)] $*" | tee -a "$OUT/window.log" >&2; }

echo "$(date -u +%FT%TZ) glm-dflash2 GLM spec A/B window $WIN ($# arms, ~$(( $# * 6 )) min, timing: no builds please) QUEUED pid=$$" >> "$Q"
exec 9>>"$LOCK"
flock -w 14400 9 || { echo "$(date -u +%FT%TZ) glm-dflash2 window $WIN gave up on lock pid=$$" >> "$Q"; exit 98; }
echo "$(date -u +%FT%TZ) glm-dflash2 window $WIN START (lock held) pid=$$" >> "$Q"
LOW=999
SRV=""; WD=""
finish() {
  [ -n "$WD" ] && kill "$WD" 2>/dev/null
  if [ -n "$SRV" ] && kill -0 "$SRV" 2>/dev/null; then
    kill -INT "$SRV"; for _ in $(seq 1 60); do kill -0 "$SRV" 2>/dev/null || break; sleep 1; done
    kill -9 "$SRV" 2>/dev/null
  fi
  echo "$(date -u +%FT%TZ) glm-dflash2 window $WIN END rc=${1:-0} lowwater_GB=$LOW pid=$$" >> "$Q"
}
trap 'finish 130; exit 130' INT TERM

preflight() {
  for pid in $(ls /proc | grep -E '^[0-9]+$'); do
    exe=$(readlink -f /proc/$pid/exe 2>/dev/null) || continue
    case "$(basename "$exe")" in spark-dashboard) ;; spark|spark-*|llama-server|llama-cli|vllm)
      log "PREFLIGHT FAIL: model process alive: $pid $exe"; return 1;; esac
  done
  if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then log "PREFLIGHT FAIL: GPU compute apps present"; return 1; fi
  local avail=$(( $(awk '/MemAvailable/{print $2}' /proc/meminfo) / 1024 / 1024 ))
  [ "$avail" -ge 100 ] || { log "PREFLIGHT FAIL: MemAvailable ${avail} GB < 100"; return 1; }
  log "preflight ok: MemAvailable ${avail} GB"
}

BASE_ENV=(
  ATLAS_GLM53_CUBLASLT_PREWARM=1 ATLAS_GLM53_EXACT_VERIFY=1 ATLAS_GLM53_EXACT_WIDE_DSA_PRECOMPUTE=1
  ATLAS_GLM53_EXACT_WIDE_KDA_BATCH_COPY=1 ATLAS_GLM53_EXACT_WIDE_ROWEXACT=1 ATLAS_GLM53_EXL3_KDA_QKV_GRID_ORDER=row-x
  ATLAS_GLM53_EXL3_MOE_PREFILL_DIRECT_COMBINE=1 ATLAS_GLM53_EXL3_ROUTE_PRIVATE=1 ATLAS_GLM53_EXL3_ROUTE_PRIVATE_PREFILL=1
  ATLAS_GLM53_FFN_GRAPHS=1 ATLAS_GLM53_LAYER_MAJOR_PREFILL=1 ATLAS_GLM53_LAYER_MAJOR_PREFILL_ROWS=2048
  ATLAS_GLM53_LAYER_MAJOR_VISION_PREFILL=0 ATLAS_GLM53_PARTIAL_WIDE_REPLAY=1 ATLAS_GLM53_UNVALIDATED_BRINGUP=1
  ATLAS_GLM53_VISION_FLASH_ATTN=1 ATLAS_GLM53_WIDE_PREFILL=1 ATLAS_GLM53_WIDE_PREFILL_ROWS=8
  ATLAS_GLM53_EXL3_MOE_CHUNK_ROWS=64 ATLAS_GLM53_EXL3_PREFILL_RECONSTRUCT=1 ATLAS_GLM53_EXL3_RECONSTRUCT_PRECISION=f16
  ATLAS_GLM53_EXL3_RECONSTRUCT_DTYPE=bf16 ATLAS_GLM53_ROUTER_GEMM=1 ATLAS_GLM53_HC_PRE_GEMM=1 ATLAS_GLM53_DSA_ABSORB_GEMM=1
  ATLAS_GLM53_KDA_CONV_FUSED=1 ATLAS_GLM53_DSA_DENSE_FAST=1 ATLAS_GLM53_HC_POST_PROMPT=1
  CUDA_VISIBLE_DEVICES=0 HOME=$HOME
  LD_LIBRARY_PATH=/usr/local/cuda/targets/sbsa-linux/lib:/usr/local/cuda-13.0/targets/sbsa-linux/lib:/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu
  PATH=/usr/local/cuda-13.0/bin:/usr/local/cuda/bin:/usr/bin:/bin
)
DFLASH_ENV=(
  ATLAS_GLM53_DFLASH2_CAPTURE_TILE128=1 ATLAS_GLM53_DFLASH2_COMMITTED_PROJECTION=original
  ATLAS_GLM53_DFLASH2_DEVICE_ARGMAX=1 ATLAS_GLM53_DFLASH2_KV_PREFIX=0 ATLAS_MTP_GATE_FORCE=1 ATLAS_GLM53_PHASE_TIMING=1
)
ARGV_BASE=(serve --model-from-path "$MODEL" --model-name glm53-flash --bind 127.0.0.1 --port $PORT
  --max-seq-len 2048 --max-prefill-tokens 2048 --max-num-seqs 1 --max-batch-size 1
  --enable-prefix-caching false --oom-guard-mb 4096 --no-tui --disable-thinking)

req() {  # req <dir> <name> <prompt>
  local d=$1 name=$2 which=$3 t0 t1
  t0=$(date +%s.%N)
  curl -s --max-time 600 -X POST "http://127.0.0.1:$PORT/v1/completions" -H 'content-type: application/json' \
    --data-binary @"$B/req/request-$which.json" -o "$d/$name.response.json"
  t1=$(date +%s.%N)
  python3 - "$d/$name.response.json" "$t0" "$t1" "$name" "$which" <<'PY' | tee -a "$d/timing.txt"
import json,sys,hashlib
try:
    r=json.load(open(sys.argv[1])); text=r['choices'][0]['text']
except Exception as e:
    print(f"{sys.argv[4]} {sys.argv[5]}: ERROR {e}"); sys.exit(0)
wall=(float(sys.argv[3])-float(sys.argv[2]))*1000
u=r.get('usage',{}); ttft=u.get('time_to_first_token_ms') or 0.0
ct=u.get('completion_tokens') or 0
dec_ms=wall-ttft; toks=(ct-1)/dec_ms*1000 if dec_ms>0 and ct>1 else 0.0
h=hashlib.sha256(text.encode()).hexdigest()[:16]
print(f"{sys.argv[4]} {sys.argv[5]}: prompt={u.get('prompt_tokens')} completion={ct} ttft_ms={ttft:.1f} wall_ms={wall:.1f} decode_tok_s={toks:.3f} server_tok_s={u.get('response_token/s')} finish={r['choices'][0].get('finish_reason')} sha={h}")
PY
}

run_arm() {
  local spec=$1 name mode extra d
  name=${spec%%:*}; mode=$(echo "$spec" | cut -d: -f2); extra=$(echo "$spec" | cut -d: -f3- | sed "s|%RUN%|$OUT|g")
  local ntr=$TRIALS; case $name in *@*) ntr=${name#*@}; name=${name%@*};; esac
  d=$OUT/$name; mkdir -p "$d"
  preflight || return 2
  local envs=("${BASE_ENV[@]}") argv=("${ARGV_BASE[@]}")
  case $mode in
    target) ;;
    dflash*) local nd=${mode#dflash}; nd=${nd:-3}
      envs+=("${DFLASH_ENV[@]}")
      argv+=(--dflash --dflash-gamma 8 --dflash-window-size 2048 --draft-model "$DRAFT" --glm-dflash-max-drafts "$nd");;
    *) log "bad mode $mode"; return 2;;
  esac
  if [ -n "$extra" ]; then IFS=',' read -ra kv <<< "$extra"; envs+=("${kv[@]}"); fi
  printf '%s\n' "${envs[@]}" > "$d/env.effective"; printf '%s\n' "${argv[@]}" > "$d/argv.effective"
  sha256sum "$ELF" > "$d/elf.sha256"
  log "arm $name ($mode $extra) starting"
  env -i "${envs[@]}" timeout -s INT -k 30 1800 "$ELF" "${argv[@]}" > "$d/server.log" 2>&1 &
  SRV=$!
  ( while kill -0 "$SRV" 2>/dev/null; do
      a=$(( $(awk '/MemAvailable/{print $2}' /proc/meminfo) / 1024 / 1024 ))
      echo "$a" >> "$d/memavail.txt"
      if [ "$a" -lt 6 ]; then echo "WATCHDOG kill $SRV at ${a}GB" >> "$d/server.log"; pkill -9 -P "$SRV"; kill -9 "$SRV"; fi
      sleep 2; done ) &
  WD=$!
  local t0=$(date +%s) ok=0
  for _ in $(seq 1 600); do
    if curl -sf "http://127.0.0.1:$PORT/health" 2>/dev/null | grep -q '"ready"'; then ok=1; break; fi
    kill -0 "$SRV" 2>/dev/null || break
    sleep 1
  done
  if [ $ok = 1 ]; then
    log "arm $name ready after $(( $(date +%s) - t0 )) s"
    req "$d" warmup short > /dev/null
    for i in $(seq 1 "$ntr"); do for p in $PROMPTS; do req "$d" "trial-$i-$p" "$p" >/dev/null; done; done
    [ "${CONTROL_PERTURB:-}" = "$name" ] && req "$d" "trial-1-ctlperturb" ctlperturb >/dev/null
  else
    log "arm $name FAILED to start"; tail -20 "$d/server.log" >> "$OUT/window.log"
  fi
  kill -INT "$SRV" 2>/dev/null; for _ in $(seq 1 60); do kill -0 "$SRV" 2>/dev/null || break; sleep 1; done
  pkill -9 -P "$SRV" 2>/dev/null; kill -9 "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; SRV=""
  kill "$WD" 2>/dev/null; wait "$WD" 2>/dev/null; WD=""
  local m=$(sort -n "$d/memavail.txt" 2>/dev/null | head -1); [ -n "$m" ] && [ "$m" -lt "$LOW" ] && LOW=$m
  grep -h "decode_tok_s" "$d/timing.txt" | sed "s/^/$name /" >> "$OUT/window.log"
  sleep 3
}

if [ -n "${PRE_CMD:-}" ]; then preflight && { log "pre-command: $PRE_CMD"; bash -c "$PRE_CMD" > "$OUT/pre_cmd.log" 2>&1; log "pre-command rc=$? (see pre_cmd.log)"; }; fi
for arm in "$@"; do run_arm "$arm"; done
finish 0
log "window $WIN done"
