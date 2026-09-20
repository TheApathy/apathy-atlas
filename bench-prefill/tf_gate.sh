#!/bin/bash
# tf_gate.sh ARM BINARY ENV_FILE : teacher-forced logits capture over the 6 prompts x 100 truncations
set -u
ARM="$1"; BIN="$2"; ENVFILE="$3"
BENCH=/home/flocka/atlas/flashnext-prefill-work/bench
OUT=$BENCH/results/tf-$ARM; mkdir -p "$OUT"
PORT=8898
MODEL=/home/flocka/models/Qwen3.8-Flash-Next-NVFP4-Offload
export ARM BIN ENVFILE OUT PORT MODEL BENCH
flock -w 14400 /home/flocka/atlas/.gb10.lock bash -c '
  set -u
  cd "$OUT"
  apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
  memavail=$(awk "/MemAvailable/{print int(\$2/1024/1024)}" /proc/meminfo)
  for w in $(seq 1 60); do
    apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    memavail=$(awk "/MemAvailable/{print int(\$2/1024/1024)}" /proc/meminfo)
    [ "$apps" = "0" ] && [ "$memavail" -ge 100 ] && break
    echo "preflight wait $w: apps=$apps memavail=${memavail}G" >> preflight.txt; sleep 30
  done
  if [ "$apps" != "0" ] || [ "$memavail" -lt 100 ]; then echo "PREFLIGHT FAIL apps=$apps memavail=${memavail}G" | tee -a preflight.txt; exit 2; fi
  sha256sum "$BIN" > binary.sha256; cp "$ENVFILE" env.txt
  ENVS=$(grep -v "^#" "$ENVFILE" | grep -v "^$" | tr "\n" " ")
  CMD="env -i PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin HOME=$HOME LANG=C.UTF-8 RUST_LOG=info ATLAS_NEMO_DUMP=$OUT/dump $ENVS $BIN serve --model-from-path $MODEL --model-name qwen3.8-flash-next --kernel-target qwen3.8-flash-next --port $PORT --max-seq-len 4096 --max-prefill-tokens 2048 --max-num-seqs 1 --max-batch-size 1 --ssm-cache-slots 16 --kv-cache-dtype bf16 --qwen4-qsa --gpu-memory-utilization 0.90 --oom-guard-mb 4096 --request-timeout 300 --no-tui"
  echo "$CMD" > command.txt
  setsid timeout 5400 bash -c "exec $CMD" > server.log 2>&1 &
  SPID=$!
  trap "kill -INT $SPID 2>/dev/null; sleep 5; kill -9 $SPID 2>/dev/null" EXIT
  for i in $(seq 1 900); do
    curl -s -m 2 http://localhost:$PORT/health 2>/dev/null | grep -q "\"ready\"" && break
    kill -0 $SPID 2>/dev/null || { echo "SERVER DIED" | tee died.txt; exit 3; }
    sleep 1
  done
  python3 $BENCH/tf_capture.py || echo "CAPTURE FAILED" | tee -a tf.log
  kill -INT $SPID 2>/dev/null
  for i in $(seq 1 90); do kill -0 $SPID 2>/dev/null || break; sleep 1; done
  kill -0 $SPID 2>/dev/null && { pkill -9 -f "spark-v[0-9].* serve"; sleep 3; }
  for i in $(seq 1 30); do [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ] && break; sleep 1; done
  echo "stopped $(date -u +%FT%TZ)" > stop.txt
' 2>&1 | tee "$OUT/run.log"
