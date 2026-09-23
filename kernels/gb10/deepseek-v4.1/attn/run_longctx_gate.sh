#!/bin/bash
# GPU-lock job (EXCLUSIVE): build (lib tests + gate examples + index gate) under the lock, then
#   static-core gate at 8K and 1M, runJ digest A/B (default | 16-row score blocks | 1M layout | CTRL
#   split off), seam runF/runG, replay runI, C++ index gate. Preflight + watchdog on OUR child only.
Q=/home/flocka/atlas/.gb10-queue
W=/home/flocka/atlas/dsv41-attention
L=$W/kernels/gb10/deepseek-v4.1/attn
B=$W/target/release/examples
NEED_KB=$((30 * 1024 * 1024)); ABORT_KB=$((12 * 1024 * 1024))
memavail() { awk '/^MemAvailable:/ {print $2}' /proc/meminfo; }
echo "$(date -u +%FT%TZ) dsv41-attention build + long-context gates (cand bits, score row blocks, 1M layout; ~12 min, <=12 GB) QUEUED pid=$$" >> $Q
exec 9>/home/flocka/atlas/.gb10.lock
flock 9
echo "$(date -u +%FT%TZ) dsv41-attention long-context gates START pid=$$" >> $Q
end() { echo "$(date -u +%FT%TZ) dsv41-attention long-context gates END rc=$1 pid=$$" >> $Q; flock -u 9; echo "DONE $1"; exit 0; }
avail=$(memavail); echo "preflight MemAvailable $((avail/1024/1024)) GB"
[ "$avail" -lt "$NEED_KB" ] && end "preflight:$((avail/1024/1024))GB"
cd $W
export PATH=/usr/local/cuda/bin:$PATH ATLAS_TARGET_MODEL=deepseek-v4.1 ATLAS_TARGET_QUANT=cb3
cargo test -p spark-model --lib deepseek_v41 > $L/lc_libtest.log 2>&1; rt=$?
cargo build --release -j 16 -p spark-model --example dsv41_core_static_gate --example dsv41_attn_seam --example dsv41_attn_runj_gate --example dsv41_attn_replay_gate --features cuda,gpu-examples > $L/build.log 2>&1; rb=$?
(cd $L && nvcc -O3 -std=c++17 -arch=sm_121a --fmad=false -DDSV41_INDEX_GATE sparse_index_gate.cu -o sparse_index_gate > nvcc_index.log 2>&1); rn=$?
[ $rb -ne 0 ] && end "libtest:$rt,build:$rb,nvcc:$rn"
cd $L
arm() {   # arm <log> <env...> -- <cmd...>
  local log=$1; shift; local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  env "${envs[@]}" "$@" > "$log" 2>&1 &
  local c=$!
  while kill -0 $c 2>/dev/null; do
    [ "$(memavail)" -lt "$ABORT_KB" ] && { echo "WATCHDOG: MemAvailable < 12 GB, killing $c" >> "$log"; kill -9 $c; }
    sleep 0.2
  done
  wait $c; echo $?
}
s8=$(arm lc_static_8k.log X=1 -- $B/dsv41_core_static_gate runJ_decode)
s1m=$(arm lc_static_1m.log ATLAS_GATE_MAX_SEQ=1048576 -- $B/dsv41_core_static_gate runJ_decode)
ja=$(arm lc_runj_default.log X=1 -- $B/dsv41_attn_runj_gate runJ_decode)
jb=$(arm lc_runj_rows16.log ATLAS_DSV41_SCORE_ROWS=16 -- $B/dsv41_attn_runj_gate runJ_decode)
jc=$(arm lc_runj_1m.log ATLAS_GATE_MAX_SEQ=1048576 -- $B/dsv41_attn_runj_gate runJ_decode)
jd=$(arm lc_runj_ctrl_nosplit.log ATLAS_DSV41_ATTN_SPLIT=0 -- $B/dsv41_attn_runj_gate runJ_decode)
sf=$(arm core_gate_runF_faithful.log X=1 -- $B/dsv41_attn_seam runF_faithful)
sg=$(arm core_gate_runG_replay.log X=1 -- $B/dsv41_attn_seam runG_replay)
ri=$(arm replay_gate_runI.log X=1 -- $B/dsv41_attn_replay_gate runI_plus20)
ig=$(arm index_gate.log X=1 -- bash run_index_gate_nolock.sh)
d() { grep -o 'DIGEST [0-9a-f]*' "$1" | cut -d' ' -f2; }
A=$(d lc_runj_default.log); Bd=$(d lc_runj_rows16.log); C=$(d lc_runj_1m.log); D=$(d lc_runj_ctrl_nosplit.log)
dig="rows16:$([ -n "$A" ] && [ "$A" = "$Bd" ] && echo same || echo DIFF),1m:$([ -n "$A" ] && [ "$A" = "$C" ] && echo same || echo DIFF),ctrl_nosplit:$([ -n "$D" ] && [ "$A" != "$D" ] && echo differs-ok || echo SAME-BAD)"
echo "digests A=$A rows16=$Bd 1m=$C ctrl=$D -> $dig" > lc_digests.log
end "libtest:$rt,build:$rb,nvcc:$rn,static8k:$s8,static1m:$s1m,runJ:$ja/$jb/$jc/ctrl$jd,runF:$sf,runG:$sg,runI:$ri,index:$ig,$dig"
