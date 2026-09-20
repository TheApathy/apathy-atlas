#!/usr/bin/env bash
# usage: build.sh <label>   -> bench/bin/spark-<label>, sha and kernel-target check
set -euo pipefail
W=/home/flocka/atlas/qwen27b-prefill-work
LABEL=${1:?label}
export PATH=/usr/local/cuda-13.0/bin:$PATH
export CUDA_HOME=/usr/local/cuda-13.0 CUDARC_CUDA_VERSION=13000
export ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=qwen3.8-27b ATLAS_TARGET_QUANT=nvfp4
export CARGO_TARGET_DIR=$W/target
unset RUSTFLAGS ATLAS_SKIP_BUILD
cd $W/src
touch crates/atlas-kernels/build.rs
cargo +1.93.1 build --locked --offline --release -p spark-server --bin spark 2>&1 | tail -15
cp $CARGO_TARGET_DIR/release/spark $W/bench/bin/spark-$LABEL
chmod 555 $W/bench/bin/spark-$LABEL
sha256sum $W/bench/bin/spark-$LABEL
echo "target grep: $(grep -ac 'qwen3\.8-27b' $W/bench/bin/spark-$LABEL)"
# PTX freshness: newest .cu vs newest ptx in the build out dir
NEWCU=$(find kernels/gb10/qwen3.8-27b kernels/gb10/common -name '*.cu' -newer $W/bench/bin/spark-$LABEL | wc -l)
echo "cu newer than binary: $NEWCU"
