#!/usr/bin/env bash
# Resident-grid census for a hypothetical cooperative decode megakernel.
#
# Compiles the real V4-Flash M=1 decode kernels for sm_121a with the tree's own
# nvcc flags and reports ptxas register/smem usage, which is the input to the
# cooperative resident-grid arithmetic in
# docs/MEGAKERNEL-FEASIBILITY-2026-08-12.md §2.
#
# Needs no GPU. `-cubin` instead of the tree's `--ptx` only so that ptxas runs
# here rather than at cuModuleLoadData time; register allocation is the same.
#
# Usage:  bench/megakernel-feasibility/occupancy_census.sh
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

# Mirrors crates/atlas-kernels/build_target.rs:60 + kernels/gb10/common/KERNEL.toml
FLAGS=(-arch=sm_121a -O3 --fmad=false -DTQ_PLUS_SIGNS -cubin --resource-usage)

COMMON="$ROOT/kernels/gb10/common"
V4="$ROOT/kernels/gb10/deepseek-v4-flash/nvfp4"

# The kernels one V4 decode layer actually dispatches at M=1
# (see docs/MEGAKERNEL-FEASIBILITY-2026-08-12.md §3 for the ordered chain).
SOURCES=(
  "$V4/hyper_connection.cu"
  "$V4/mla_paged_decode_fp8.cu"
  "$V4/mla_absorbed.cu"
  "$COMMON/rms_norm.cu"
  "$COMMON/w8a16_gemv.cu"
  "$COMMON/w4a16_gemv.cu"
  "$COMMON/w4a16_gemm.cu"
  "$COMMON/rope.cu"
  "$COMMON/reshape_and_cache.cu"
  "$COMMON/dense_gemv_bf16.cu"
  "$COMMON/dense_gemv_fp8w.cu"
  "$COMMON/moe_topk_sqrtsoftplus.cu"
  "$COMMON/moe_shared_expert_fused_t.cu"
  "$COMMON/moe_expert_gemv.cu"
)

echo "# ptxas resource usage, sm_121a — inputs to the resident-grid arithmetic"
echo "# device: 48 SMs, 1536 threads/SM, 102400 B smem/SM, 65536 regs/SM"
echo
for src in "${SOURCES[@]}"; do
  [ -f "$src" ] || { echo "### MISSING ${src#$ROOT/}"; continue; }
  echo "################ ${src#$ROOT/}"
  "$NVCC" "${FLAGS[@]}" "$src" -o "$OUT/$(basename "$src" .cu).cubin" 2>&1 \
    | sed 's/^ptxas info    : //' \
    | grep -E "Compiling entry|Used .* registers|error"
done

echo
echo "# blocks/SM = min( 1536/threads,"
echo "#                  65536 / ((threads/32) * ceil(regs*32/256) * 256),"
echo "#                  102400 / smem_per_cta )"
echo "# resident CTAs = 48 * blocks/SM  <-- the cooperative-launch grid ceiling"
