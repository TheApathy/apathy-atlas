#!/usr/bin/env bash
# Build the Qwen3.6-27B champion release binary.
#
#   CUTLASS_HOME=/path/to/cutlass bash bench/qwen/build_cutlass.sh
#
# A bare `cargo build --release` is NOT equivalent and is the single most common
# way to spend an afternoon debugging a healthy tree. It produces a `spark` that
# defaults to a different kernel target, cannot load this model, and fails only
# at serve time -- long after the build looked clean. The four ATLAS_TARGET_*
# variables below are what select the qwen3.6-27b NVFP4 kernel set, and
# CUTLASS_HOME is what lets the grouped MoE GEMMs compile at all.
#
# Note also that a clean `cargo build --release` never compiles examples or
# benches; a stale kernel caller in examples/ can survive a green build. Use
# `--features="cuda gpu-examples"` when you need those checked.
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]:-$0}")/env.sh"

# CUTLASS checkout. No default is right on someone else's machine; if you built
# vLLM from source you already have one under its .deps directory.
: "${CUTLASS_HOME:?set CUTLASS_HOME to a CUTLASS source checkout}"
export CUTLASS_HOME

# sm_121f is Grace Blackwell. Change ATLAS_CUDA_ARCH for a different GPU; the
# other three select the model/quantization kernel set and correspond directly
# to the on-disk layout kernels/$ATLAS_TARGET_HW/$ATLAS_TARGET_MODEL/$ATLAS_TARGET_QUANT.
export ATLAS_CUDA_ARCH="${ATLAS_CUDA_ARCH:-sm_121f}"
export ATLAS_TARGET_HW="${ATLAS_TARGET_HW:-gb10}"
export ATLAS_TARGET_MODEL="${ATLAS_TARGET_MODEL:-qwen3.6-27b}"
export ATLAS_TARGET_QUANT="${ATLAS_TARGET_QUANT:-nvfp4}"

cd "$REPO" || { echo "FATAL: repo root $REPO missing"; exit 3; }

# Assert the kernel set this build selects actually exists in the tree, rather
# than discovering it as a link error twenty minutes in.
KDIR="kernels/$ATLAS_TARGET_HW/$ATLAS_TARGET_MODEL/$ATLAS_TARGET_QUANT"
[ -d "$KDIR" ] || { echo "FATAL: no kernel set at $KDIR"; \
  echo "available: $(ls kernels/$ATLAS_TARGET_HW 2>/dev/null | tr '\n' ' ')"; exit 3; }

# Editing a .cu does not reliably trigger a kernel rebuild, and the "compiled N
# kernels" line is itself cached, so it will happily report success over stale
# PTX. Clearing the fingerprint is the only reliable way to force one.
if [ "${QWEN_FORCE_KERNELS:-0}" = 1 ]; then
  echo "forcing kernel rebuild (clearing atlas-kernels fingerprint)"
  rm -rf target/release/.fingerprint/atlas-kernels-*
fi

echo "BUILD START $(date -u +%H:%M:%S) CUTLASS=$CUTLASS_HOME MODEL=$ATLAS_TARGET_MODEL ARCH=$ATLAS_CUDA_ARCH"
cargo build --release --bin spark 2>&1
rc=$?
echo "BUILD END rc=$rc $(date -u +%H:%M:%S)"
[ "$rc" = 0 ] && echo "binary: $REPO/target/release/spark  (export QWEN_BIN to point the harness at it)"
exit $rc
