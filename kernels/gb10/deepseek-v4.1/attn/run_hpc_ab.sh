#!/bin/bash
# GPU-lock job: build, then the full-model prefill profile with 16 / 32 / 64 heads per CTA
# (interleaved twice), plus all correctness gates at the default.
Q=/home/flocka/atlas/.gb10-queue
W=/home/flocka/atlas/dsv41-attention
D=$W/kernels/gb10/deepseek-v4.1/attn
echo "$(date -u +%FT%TZ) dsv41-attention build (under lock) + heads-per-CTA A/B profile + gates (~6 min) QUEUED pid=$$" >> $Q
exec 9>/home/flocka/atlas/.gb10.lock
flock 9
echo "$(date -u +%FT%TZ) dsv41-attention hpc-ab START pid=$$" >> $Q
cd $W
export PATH=/usr/local/cuda/bin:$PATH ATLAS_TARGET_MODEL=deepseek-v4.1 ATLAS_TARGET_QUANT=cb3
cargo build --release -j 16 -p spark-model --example dsv41_attn_seam --example dsv41_attn_replay_gate --example dsv41_attn_runj_gate --example dsv41_forward --features cuda,gpu-examples > $D/build.log 2>&1; rb=$?
: > $D/hpc_ab.log
if [ $rb -eq 0 ]; then
  for rep in 1 2; do for h in 16 32 64; do
    echo "=== HPC=$h rep $rep" >> $D/hpc_ab.log
    ATLAS_DSV41_ATTN_HPC=$h ATLAS_DSV41_CORE_PROF=1 $W/target/release/examples/dsv41_forward --run runH_feed40 --layers 40 --path model --attn-real 2>&1 | grep -E "attn\.sparse|argmax|^prefill" >> $D/hpc_ab.log
  done; done
  cd $D
  $W/target/release/examples/dsv41_attn_seam runF_faithful > core_gate_runF_faithful.log 2>&1; r1=$?
  $W/target/release/examples/dsv41_attn_seam runG_replay > core_gate_runG_replay.log 2>&1; r2=$?
  $W/target/release/examples/dsv41_attn_replay_gate runI_plus20 > replay_gate_runI.log 2>&1; r3=$?
  $W/target/release/examples/dsv41_attn_runj_gate runJ_decode > runj_gate.log 2>&1; r4=$?
fi
echo "$(date -u +%FT%TZ) dsv41-attention hpc-ab END rc=build:$rb,runF:$r1,runG:$r2,runI:$r3,runJ:$r4 pid=$$" >> $Q
flock -u 9
echo "DONE build=$rb runF=$r1 runG=$r2 runI=$r3 runJ=$r4"
