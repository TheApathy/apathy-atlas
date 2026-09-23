#!/usr/bin/env bash
# usage: build.sh <label>   builds glm-spec-fix for GLM EXL3 -> bin/spark-<label>
set -uo pipefail
LABEL=$1; SRC=${SRC:-/home/flocka/atlas/glm-spec-fix}; OUT=/home/flocka/atlas/glm-spec-bench
LOG=$OUT/build-$LABEL.log
export PATH=/usr/local/cuda-13.0/bin:/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
export CUDA_HOME=/usr/local/cuda-13.0 CUDARC_CUDA_VERSION=13000
export ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=glm5.3-flash ATLAS_TARGET_QUANT=exl3
export ATLAS_EXLLAMAV3_SOURCE=/var/tmp/exllamav3-glm53-r28
export CARGO_TARGET_DIR=${TDIR:-$OUT/target}
unset RUSTFLAGS ATLAS_SKIP_BUILD
cd "$SRC"
echo "BUILD $LABEL head=$(git rev-parse HEAD) dirty=$(git status --short | wc -l) start=$(date -u +%FT%TZ)" > "$LOG"
find kernels/gb10/glm5.3-flash \( -name '*.cu' -o -name '*.cuh' \) -exec touch {} +
touch crates/atlas-kernels/build.rs
nice -n 15 cargo build -j 12 --release -p spark-server --bin spark >> "$LOG" 2>&1
RC=$?
if [ $RC -eq 0 ]; then
  rm -f "$OUT/bin/spark-$LABEL"; cp "${TDIR:-$OUT/target}/release/spark" "$OUT/bin/spark-$LABEL" && chmod 555 "$OUT/bin/spark-$LABEL"
  sha256sum "$OUT/bin/spark-$LABEL" >> "$LOG"
fi
grep -a 'compiled .* kernels for target' "$LOG" | sed 's/.*atlas-kernels: //' | sort -u | tee -a "$LOG"
echo "DONE $LABEL rc=$RC end=$(date -u +%FT%TZ)" | tee -a "$LOG"
