#!/bin/bash
# GPU-lock job: build (lib tests + gate examples) under the lock, then the shape-static core gate
# plus the regression gates on the refactored topk/candidates/combine kernels (default path).
Q=/home/flocka/atlas/.gb10-queue
W=/home/flocka/atlas/dsv41-attention
echo "$(date -u +%FT%TZ) dsv41-attention build + static-core gate + regression gates (~6 min, partial-layer loads ~6 GB) QUEUED pid=$$" >> $Q
exec 9>/home/flocka/atlas/.gb10.lock
flock 9
echo "$(date -u +%FT%TZ) dsv41-attention build+static-gate START pid=$$" >> $Q
cd $W
export PATH=/usr/local/cuda/bin:$PATH ATLAS_TARGET_MODEL=deepseek-v4.1 ATLAS_TARGET_QUANT=cb3
L=$W/kernels/gb10/deepseek-v4.1/attn
cargo test -p spark-model --lib deepseek_v41 > $L/static_libtest.log 2>&1; rt=$?
cargo build --release -j 16 -p spark-model --example dsv41_core_static_gate --example dsv41_attn_seam --example dsv41_attn_runj_gate --features cuda,gpu-examples > $L/build.log 2>&1; rb=$?
r0=-1; r1=-1; r2=-1
if [ $rb -eq 0 ]; then
  cd $L
  $W/target/release/examples/dsv41_core_static_gate runJ_decode > static_gate.log 2>&1; r0=$?
  $W/target/release/examples/dsv41_attn_runj_gate runJ_decode > runj_gate.log 2>&1; r1=$?
  $W/target/release/examples/dsv41_attn_seam runG_replay > core_gate_runG_replay.log 2>&1; r2=$?
fi
echo "$(date -u +%FT%TZ) dsv41-attention build+static-gate END rc=libtest:$rt,build:$rb,static:$r0,runJ:$r1,runG:$r2 pid=$$" >> $Q
flock -u 9
echo "DONE libtest=$rt build=$rb static=$r0 runJ=$r1 runG=$r2"
