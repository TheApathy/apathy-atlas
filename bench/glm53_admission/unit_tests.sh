#!/usr/bin/env bash
# CPU unit tests for the GLM admission lane. Run through tools/locked_build.sh.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH=/usr/local/cuda-13.0/bin:/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
export CUDA_HOME=/usr/local/cuda-13.0 CUDARC_CUDA_VERSION=13000
export ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=glm5.3-flash ATLAS_TARGET_QUANT=exl3
export ATLAS_EXLLAMAV3_SOURCE=/var/tmp/exllamav3-glm53-r28
export CARGO_TARGET_DIR=$HERE/../../target-glm
unset ATLAS_GLM53_UNVALIDATED_BRINGUP ATLAS_GLM53_NEGATIVE_CONTROL
cd "$HERE/../.."
nice -n 10 cargo test -j 14 --release -p spark-model --lib "$@"
