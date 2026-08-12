// Cooperative-groups / grid.sync feasibility probe for GB10 (sm_121).
//
// This file is a STUDY ARTIFACT, not a production kernel. It lives under
// bench/ deliberately: `crates/atlas-kernels/build.rs` only walks
// kernels/<hw>/common and kernels/<hw>/<model>/<quant>, so nothing here is
// compiled into the binary or JIT'd at AtlasRegistry::init.
//
// Purpose: establish, without touching the GPU, whether
// `cooperative_groups::this_grid().sync()` is usable on sm_121 through the
// exact toolchain the tree uses (`nvcc --ptx -arch=sm_121f -O3 --fmad=false`,
// then cuModuleLoadData JIT at startup).
//
// The three things it proves, all verifiable with the commands in
// docs/MEGAKERNEL-FEASIBILITY-2026-08-12.md §1:
//
//   1. It compiles for sm_121f/sm_121a with no -rdc=true and no device link
//      step. `this_grid()` resolves the grid barrier by reading %envreg1/%envreg2
//      (cooperative_groups/details/driver_abi.h::get_grid_workspace) — an address
//      the DRIVER writes only on a cooperative launch. So a stock
//      `nvcc --ptx` module carrying grid.sync is loadable through the tree's
//      existing cuModuleLoadData path with zero build-system change.
//
//   2. Consequence of (1): launching this kernel with the ordinary
//      `cuLaunchKernel` in atlas-core/src/registry.rs:424 is MEMORY-UNSAFE, not
//      merely wrong — envreg1/2 hold an undefined address and the barrier
//      atomic scribbles on it. A cooperative kernel MUST go through
//      cuLaunchCooperativeKernel (or cuLaunchKernelEx with
//      CU_LAUNCH_ATTRIBUTE_COOPERATIVE), neither of which is declared in any of
//      the tree's three hand-rolled extern blocks.
//
//   3. The SASS shows what a grid.sync actually costs, which is the whole
//      economic argument: per barrier, per CTA, ptxas emits
//        BAR.SYNC.DEFER_BLOCKING          (full CTA drain)
//        MEMBAR.ALL.GPU / ERRBAR / CGAERRBAR   (device-scope fence)
//        ATOM.E.ADD.STRONG.GPU            (one contended global cache line)
//        LD.acquire.gpu spin
//        BAR.SYNC.DEFER_BLOCKING
//      i.e. the same drain a kernel boundary has, PLUS two dependent
//      device-scope memory round trips. grid.sync does not remove the
//      kernel-boundary drain; it re-implements it in software.
//
// __launch_bounds__(256, 4) mirrors the widest occupancy any V4 decode stage
// asks for, so the reported register count is comparable with the per-kernel
// census in the study doc.

#include <cooperative_groups.h>
namespace cg = cooperative_groups;

extern "C" __global__ void __launch_bounds__(256, 4)
cg_probe(float* a, float* b, int n) {
    cg::grid_group g = cg::this_grid();
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) a[i] = b[i] * 2.0f;
    g.sync();
    // Cross-CTA read: forces the barrier to be load-bearing so ptxas cannot
    // elide it. a[n-1-i] is written by a different CTA than the one reading it.
    if (i < n) b[i] = a[n - 1 - i] + 1.0f;
    g.sync();
    if (i < n) a[i] += b[i];
}
