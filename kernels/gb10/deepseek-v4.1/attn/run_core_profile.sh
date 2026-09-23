#!/bin/bash
# GPU-lock job: build the driver, then the full-model prefill (runH_feed40, MoE fed from the
# capture, no arena, ~10 GB) with the REAL attention core, cold + warm, core phases profiled
# (ATLAS_DSV41_CORE_PROF=1, synchronized per phase) and the driver's scope profile (--prof).
Q=/home/flocka/atlas/.gb10-queue
W=/home/flocka/atlas/dsv41-attention
OUT=$W/kernels/gb10/deepseek-v4.1/attn/core_profile.log
echo "$(date -u +%FT%TZ) dsv41-attention build (under lock) + core prefill profile (runH_feed40, fed MoE, ~10 GB, ~3 min) QUEUED pid=$$" >> $Q
exec 9>/home/flocka/atlas/.gb10.lock
flock 9
echo "$(date -u +%FT%TZ) dsv41-attention core-profile START pid=$$" >> $Q
cd $W
export PATH=/usr/local/cuda/bin:$PATH ATLAS_TARGET_MODEL=deepseek-v4.1 ATLAS_TARGET_QUANT=cb3
cargo build --release -j 16 -p spark-model --example dsv41_forward --features cuda,gpu-examples > $W/kernels/gb10/deepseek-v4.1/attn/build.log 2>&1; rb=$?
r=-1
if [ $rb -eq 0 ]; then
  ATLAS_DSV41_CORE_PROF=1 $W/target/release/examples/dsv41_forward --run runH_feed40 --layers 40 --path model --attn-real --warm-prefill --prof > $OUT 2>&1; r=$?
fi
echo "$(date -u +%FT%TZ) dsv41-attention core-profile END rc=build:$rb,run:$r pid=$$" >> $Q
flock -u 9
echo "DONE build=$rb run=$r"
