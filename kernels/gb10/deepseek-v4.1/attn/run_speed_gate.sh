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
echo "$(date -u +%FT%TZ) dsv41-attention build + speed gates (single-launch projections keyed on pass kind, last-block combine, tail splits 16,4 / 19,1; digest vs pinned A; ~12 min, <=12 GB) QUEUED pid=$$" >> $Q
exec 9>/home/flocka/atlas/.gb10.lock
flock 9
echo "$(date -u +%FT%TZ) dsv41-attention speed gates START pid=$$" >> $Q
end() { echo "$(date -u +%FT%TZ) dsv41-attention speed gates END rc=$1 pid=$$" >> $Q; flock -u 9; echo "DONE $1"; exit 0; }
avail=$(memavail); echo "preflight MemAvailable $((avail/1024/1024)) GB"
[ "$avail" -lt "$NEED_KB" ] && end "preflight:$((avail/1024/1024))GB"
cd $W
export PATH=/usr/local/cuda/bin:$PATH ATLAS_TARGET_MODEL=deepseek-v4.1 ATLAS_TARGET_QUANT=cb3
cargo test -p spark-model --lib deepseek_v41 > $L/sp_libtest.log 2>&1; rt=$?
cargo build --release -j 16 -p spark-model --example dsv41_core_static_gate --example dsv41_attn_seam --example dsv41_attn_runj_gate --example dsv41_attn_replay_gate --features cuda,gpu-examples > $L/build.log 2>&1; rb=$?
[ $rb -ne 0 ] && end "libtest:$rt,build:$rb"
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
A_PIN=083eae0cc608ae71   # runJ DIGEST of the tiled core, d9960e30b (lc_digests.log)
s8=$(arm sp_static.log X=1 -- $B/dsv41_core_static_gate runJ_decode)
slb=$(arm sp_static_lb.log ATLAS_DSV41_ATTN_LB=1 -- $B/dsv41_core_static_gate runJ_decode)
ja=$(arm sp_runj_default.log X=1 -- $B/dsv41_attn_runj_gate runJ_decode)
jb=$(arm sp_runj_lb.log ATLAS_DSV41_ATTN_LB=1 -- $B/dsv41_attn_runj_gate runJ_decode)
jc=$(arm sp_runj_tiled.log ATLAS_DSV41_CORE_TILED=1 -- $B/dsv41_attn_runj_gate runJ_decode)
je=$(arm sp_runj_tail16_4.log ATLAS_GATE_TAIL_SPLIT=16,4 -- $B/dsv41_attn_runj_gate runJ_decode)
jf=$(arm sp_runj_tail19_1.log ATLAS_GATE_TAIL_SPLIT=19,1 -- $B/dsv41_attn_runj_gate runJ_decode)
jd=$(arm sp_runj_ctrl_nosplit.log ATLAS_DSV41_ATTN_SPLIT=0 -- $B/dsv41_attn_runj_gate runJ_decode)
sf=$(arm core_gate_runF_faithful.log X=1 -- $B/dsv41_attn_seam runF_faithful)
sg=$(arm core_gate_runG_replay.log X=1 -- $B/dsv41_attn_seam runG_replay)
ri=$(arm replay_gate_runI.log X=1 -- $B/dsv41_attn_replay_gate runI_plus20)
d() { grep -o 'DIGEST [0-9a-f]*' "$1" | cut -d' ' -f2; }
eq() { [ "$(d $1)" = "$A_PIN" ] && echo same || echo "DIFF($(d $1))"; }
dig="default:$(eq sp_runj_default.log),lb:$(eq sp_runj_lb.log),tiled:$(eq sp_runj_tiled.log),tail16_4:$(eq sp_runj_tail16_4.log),tail19_1:$(eq sp_runj_tail19_1.log),ctrl_nosplit:$([ "$(d sp_runj_ctrl_nosplit.log)" != "$A_PIN" ] && echo differs-ok || echo SAME-BAD)"
echo "pinned A=$A_PIN -> $dig" > sp_digests.log
end "libtest:$rt,build:$rb,static:$s8,static_lb:$slb,runJ:$ja/$jb/$jc/$je/$jf/ctrl$jd,runF:$sf,runG:$sg,runI:$ri,$dig"
