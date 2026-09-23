#!/usr/bin/env bash
# usage: build.sh <label>  -> runs/bin/spark-<label> (GLM-5.3 EXL3 target, GB10)
# Run through tools/locked_build.sh. Mirrors allmodels-bench/build.sh; the
# binary is copied out so later edits cannot change what a gate run measured.
set -uo pipefail
LABEL=${1:?label}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "$HERE/../.." && pwd)"
OUT=$HERE/runs/bin; mkdir -p "$OUT"
LOG=$OUT/build-$LABEL.log
export PATH=/usr/local/cuda-13.0/bin:/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
export CUDA_HOME=/usr/local/cuda-13.0 CUDARC_CUDA_VERSION=13000
export ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=glm5.3-flash ATLAS_TARGET_QUANT=exl3
export ATLAS_EXLLAMAV3_SOURCE=/var/tmp/exllamav3-glm53-r28
export CARGO_TARGET_DIR=$SRC/target-glm
unset RUSTFLAGS ATLAS_SKIP_BUILD
cd "$SRC"
echo "BUILD $LABEL head=$(git rev-parse HEAD) dirty=$(git status --porcelain | wc -l) start=$(date -u +%FT%TZ)" > "$LOG"
git diff HEAD > "$OUT/build-$LABEL.diff"
touch crates/atlas-kernels/build.rs
nice -n 15 cargo build -j 14 --release -p spark-server --bin spark >> "$LOG" 2>&1
RC=$?
if [ $RC -eq 0 ]; then
  rm -f "$OUT/spark-$LABEL"; cp "$CARGO_TARGET_DIR/release/spark" "$OUT/spark-$LABEL" && chmod 555 "$OUT/spark-$LABEL"
  sha256sum "$OUT/spark-$LABEL" >> "$LOG"
fi
grep -a 'compiled .* kernels for target' "$LOG" | sed 's/.*atlas-kernels: //' | sort -u
echo "DONE $LABEL rc=$RC end=$(date -u +%FT%TZ)" | tee -a "$LOG"
exit $RC
