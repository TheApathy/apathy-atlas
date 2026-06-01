// SPDX-License-Identifier: AGPL-3.0-only
//
// Atlas SM120/SM121 NVFP4×NVFP4 GEMM — closes the FlashInfer prefill MFU gap.
//
// =========================================================================
// PURPOSE
// =========================================================================
// AEON-7 container achieves ~30% MFU on prefill GEMM via FlashInfer's
// CUTLASS NVFP4 kernel. Atlas's w4a16_gemm (BF16×NVFP4) reaches ~12% MFU.
// This file ports the underlying hardware path (native NVFP4 tensor cores)
// into Atlas's PTX-only kernel idiom.
//
// FlashInfer's path is built on top of CUTLASS 3.x CollectiveBuilder
// (~50K lines of templates: TMA, warp-specialized mainloop, swizzled
// Sm1xxBlkScaledConfig scale layout, GemmUniversalAdapter, workspace
// management). Atlas compiles `.cu` → PTX via `nvcc --ptx -arch=sm_121f`
// and loads via cuModuleLoad — it cannot link the CUTLASS host runtime.
//
// PIVOT: Use the same hardware MMA instruction directly:
//   mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m1.e2m1.f32
//
// This is the same E2M1×E2M1→F32 native NVFP4 tensor core op CUTLASS
// dispatches under the hood (cute/arch/mma_sm120.hpp line 70). It delivers
// the same FLOPs without the CUTLASS host stack.
//
// =========================================================================
// WEIGHT / ACTIVATION LAYOUT
// =========================================================================
// Atlas's existing NVFP4 weight format (compressed-tensors compatible):
//   B_packed: [N, K/2] uint8     — two E2M1 nibbles per byte
//   B_scale:  [N, K/16] uint8    — one FP8 E4M3 scale per group of 16
//   B_scale2: scalar fp32        — per-tensor second-level scale
//
// For W4A4 NVFP4×NVFP4 MMA we ALSO need activations in NVFP4 format:
//   A_packed: [M, K/2] uint8     — same packed E2M1 layout as B
//   A_scale:  [M, K/16] uint8    — per-token, per-group FP8 E4M3 scale
//   A_scale2: scalar fp32        — per-tensor second-level scale
//
// The activation quantization is done out-of-band by `quantize_bf16_to_nvfp4`
// (kernels/gb10/nvfp4/quantize_bf16_to_nvfp4.cu) before this kernel is
// invoked. The Rust dispatch layer (dense_ffn.rs) handles this prequant
// step gated by `ATLAS_E2M1_GEMM=1`.
//
// =========================================================================
// MMA FRAGMENT LAYOUT (m16n8k32, sm120 e2m1×e2m1)
// =========================================================================
// Per the PTX ISA:
//   - A: 4 × uint32 per thread = 16 bytes = 32 E2M1 values (k-dim)
//        Layout: each thread holds 32 packed E2M1 values across k
//   - B: 2 × uint32 per thread = 8 bytes = 16 E2M1 values
//        Layout: each thread holds 16 packed E2M1 values across k
//   - D: 4 × float per thread (matches existing m16n8k32 FP8 path)
//
// Compared to the existing FP8 MMA (m16n8k32 e4m3.e4m3):
//   - Register layout for D matches (4×float per thread)
//   - A needs 4 uint32 (same) but holds 32 e2m1 instead of 16 e4m3
//   - B needs 2 uint32 (same) but holds 16 e2m1 instead of 8 e4m3
//   - Throughput: 2x FLOPs/byte vs FP8 → potential 2x speedup
//
// Scale factors are applied in the EPILOGUE: dequant the FP32 accumulator
// by A_scale[m,g] * B_scale[n,g] * A_scale2 * B_scale2 per K-group.
// Because the MMA accumulates across k=32 in a single instruction, we
// pick K_STEP=32 (= 2 groups of 16) and apply per-group rescaling between
// MMA calls (similar to how some MX-FP4 kernels work).
//
// =========================================================================
// TILING
// =========================================================================
//   M_TILE = 64 (4 warps × m16) — same as existing FP8 path
//   N_TILE = 128 (16 n8 fragments per warp)
//   K_STEP = 64  (2× K=32 MMA per inner iter; K=64 = 4 NVFP4 groups)
//   Outer K iterations: K / K_STEP
//
// SMEM (double-buffered):
//   A_packed: 2 × 64 × 32 = 4 KB  (K_STEP/2 bytes per row)
//   A_scale:  2 × 64 × 4  = 0.5 KB (K_STEP/16 bytes per row)
//   B_packed: 2 × 128 × 32 = 8 KB
//   B_scale:  2 × 128 × 4 = 1 KB
//   Total ~13.5 KB/CTA → ~7 CTAs/SM (well below SMEM ceiling)
//
// =========================================================================
// FIRST-PASS STATUS
// =========================================================================
// This is a first-pass implementation. Optimizations deferred:
//   - cp.async wait-then-sync pipeline (mirror w4a16_gemm_t pattern)
//   - swizzled smem for bank-conflict-free MMA loads
//   - per-warp register reuse for B fragments across N-tiles
//   - dispatch heuristic vs w4a16_gemm_t_m128 at small M
//
// At parity with FlashInfer's MFU (~30%) we get ~2.5× prefill speedup:
// TTFT 4K: 5.76s → 2.3s expected.
//
// =========================================================================
// FIX PATH 1 LANDED 2026-05-22 (gated behind ATLAS_E2M1_GEMM=1, default off)
// =========================================================================
// Original first wire-up crashed the inference server (CUDA illegal access)
// on the very first prefill chunk because COMPUTE_KSTEP assumed
// `mma.kind::f8f6f4.m16n8k32` (e2m1×e2m1) covers k=32 e2m1 per MMA. It does
// not: per the SM120 cute layout (cute/arch/mma_sm120.hpp:55, A=4×.b32 per
// thread, total per-fragment bytes = 32×16 = 512 = 16 rows × 32 packed bytes
// = 16 rows × 64 e2m1 dense nibbles), each MMA actually covers K_LOGICAL=64
// e2m1 values. So the prior `K_STEP/32 = 2 sub-MMA` walk indexed past
// row size 32 at ksub=1 + tid=3 (offset 36 vs 32).
//
// Fix Path 1 applied: drop ksub loop to 1 iteration, use byte offsets
// a_col_byte ∈ {4*tid, 16+4*tid} (mirroring the proven FP8 m16n8k32 layout
// in w4a16_gemm.cu:303-306). K_STEP remains 64 e2m1 elements; one m16n8k32
// MMA per outer K-iter covers all 64. Per-group scales (4 groups per MMA,
// since GROUP_SIZE=16) are MEAN-approximated — exact per-group scaling
// requires Fix Path 2 (split into 4× m16n8k16-kind::f8f6f4 or switch to
// kind::mxf4nvf4.block_scale.m16n8k64 with hardware SF registers).
//
// =========================================================================
// FIX PATH 2 LANDED 2026-05-22 (gated behind ATLAS_E2M1_GEMM=1, default off)
// =========================================================================
// Hardware block-scaled NVFP4 MMA. PATH-1 used mma.kind::f8f6f4.m16n8k32
// with MEAN-approximated per-MMA scale (4 groups averaged). PATH-2 uses
// the native block-scaled instruction:
//
//   mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64
//                    .row.col.f32.e2m1.e2m1.f32.ue4m3
//
// The tensor core fetches A/B nibbles AND the 4 per-group e4m3 scales
// (packed in sfa/sfb 32-bit registers) and does dequant + matmul atomically.
// This eliminates the mean-approximation and the FP8→FP32 cvt + 4× FMA
// overhead in the inner loop.
//
// SASS confirmed on sm_121f: `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` —
// hardware support is present (test: /tmp/test_mxf4_real.cu, 2026-05-22).
//
// Scale-register layout (decoded from CUTE Sm120 SFALayout/SFBLayout):
//   SFA (T32,V64) -> (M16,K64): thread t packs 4× e4m3 bytes for
//     m_row = ((t >> 2) & 7) + 8 * (t & 1), kg_base + 0..3.
//     Threads with bit 1 set (t & 2) are broadcast duplicates — they
//     still load and pack but the tensor core uses one copy per row pair.
//   SFB (T32,V64) -> (N8,K64): thread t packs 4× e4m3 bytes for
//     n_col = (t >> 2) & 7, kg_base + 0..3. The (4:0) stride means
//     all 4 lanes within a column-group hold the same sfb value.
//
// The build gate define `CUTE_ARCH_MXF4NVF4_4X_UE4M3_MMA_ENABLED` is
// NOT needed here — we inline the PTX directly; nvcc emits the SASS
// regardless on sm_120/sm_121.
//
// =========================================================================
// FIX PATH 2.1 LANDED 2026-05-23 (gated behind ATLAS_E2M1_GEMM=1, default off)
// =========================================================================
// Two surgical perf improvements on the inner COMPUTE_KSTEP loop:
//   (a) Collapsed the 4-byte SFA/SFB scale gather (4 byte-loads + 3 shifts
//       + 3 ORs each) into a single u32 SMEM load. The smem layouts are
//       declared as [.][.][K_STEP/GROUP_SIZE] = [.][.][4], so the per-row
//       scale stride is exactly 4 bytes — naturally u32-aligned. Inside the
//       16-iter N-fragment loop that's a ~6x reduction in scale-gather
//       instruction count.
//   (b) Added `__launch_bounds__(128, 4)` to lock in 4 CTAs/SM. Without it
//       ptxas computed 116 regs but didn't pin occupancy; with it the
//       compiler kept the inner loop at 127 regs (no spills) and 4 CTAs.
//
// Microbench results (criterion, GB10 sm_121f, 2026-05-23):
//
//   Shape (M=128 K=N=2048):
//       Path A (w4a16_t_m128):    96 µs   (BF16 A × NVFP4 B)
//       Path B (this kernel mma): 39 µs   — 2.5x faster
//       Path B full:              56 µs   — 1.7x faster (incl. D2H absmax sync)
//       cos(A, B vs CPU ref):     0.9996 / 0.9949 — both pass
//
//   Shape (gate/up_proj prod, M=128 K=5120 N=17408):
//       Path A:                   383 µs
//       Path B mma:               854 µs  — 2.2x SLOWER
//       Path B full:              846 µs
//       cos(A, B):                0.9948
//
//   Shape (down_proj prod, M=128 K=17408 N=5120):
//       Path A:                   909 µs
//       Path B mma:               663 µs  — 1.37x faster
//       Path B full:              704 µs  — 1.3x faster
//
// Per FFN at production shape:
//   Path A: 383 + 383 + 909 = 1675 µs
//   Path B: 854 + 854 + 704 = 2412 µs  → +44% per FFN
//
// Conclusion: the hardware MMA is shape-dependent on GB10. It wins at
// small K (where the kernel's 16-N-fragment unroll has high ILP and few
// CTAs) and on shapes where N is moderate. It LOSES on the wide gate/up
// projection because at N=17408 the OMMA.SF instruction's latency chain
// inside the 16-iter unroll dominates, and we run out of CTAs to hide it
// (272 CTAs vs 192-max-in-flight on 48 SMs × 4 blocks/SM).
//
// Production decision (2026-05-23): keep the path gated. Enabling
// ATLAS_E2M1_GEMM=1 globally would REGRESS TTFT at production shape —
// the per-FFN gain on down_proj does not offset the loss on gate/up.
//
// A future Fix Path 3 should explore: (i) M_TILE=128 with halved N
// partition, (ii) split-K to give more CTAs at large N, (iii) per-shape
// dispatch (route only down_proj through Path B). Until that ships, the
// production prefill stays on `w4a16_gemm_t_m128`.
//
// =========================================================================
// FIX PATH 3 EXPERIMENTS 2026-05-23 (negative results — kernel unchanged)
// =========================================================================
// Three experiments were run against the K=5120 N=17408 wide-N regression:
//
//   (3a) SPLIT-K. Added `nvfp4_nvfp4_gemm_t_m64_splitk` (split-K variant,
//        F32 partials written to scratch [K_SPLITS, M, N]) + reduce
//        kernel `nvfp4_splitk_reduce` (sum + scale2_ab + BF16 cast).
//        Correctness: cos(B, B-splitK) = 1.0000 (perfect numerical
//        match — float addition associativity is harmless here since
//        partials sum to the same accumulator slots in different order).
//        Perf: K_SPLITS=2 → 897 µs (worse than B 791 µs); K_SPLITS=4
//        → 936 µs; K_SPLITS=8 → 1370 µs. Increasing splits monotonically
//        hurts. Diagnosis: scratch writes (4B F32 per output vs 2B BF16)
//        are 4x the output-write bandwidth, and reduce overhead at
//        wave-balanced grids (548 CTAs at K_SPLITS=2) saturates GMEM
//        bw before it can recover from the deeper wave pool. At the
//        K=2048 N=2048 small shape split-K WINS marginally (36 vs 41 µs)
//        — but small shapes are not the production bottleneck.
//
//   (3b) SMEM BANK-CONFLICT PADDING. Identified 2-way bank conflict
//        in the inner MMA loop: row stride K_STEP/2 = 32 B = 8 banks,
//        groups 0..7 hit banks {0,8,16,24,0,8,16,24} → 2-way. Tried
//        PAD_AP=16 (must be 16-aligned for cp.async dest alignment),
//        making row stride 48 B = 12 banks → groups hit
//        {0,12,24,4,16,28,8,20} all distinct. Measured: 791 vs 794 µs —
//        no perf change. SMEM bank conflicts are NOT the bottleneck.
//
//   (3c) ROOT CAUSE CONFIRMED: GMEM B-load bandwidth dominates. Each
//        N-CTA reads a unique slice of the HF row-major `B_packed[N][K/2]`
//        weight — no inter-CTA cache-line sharing. Path A's
//        `w4a16_gemm_t_m128` uses K-major `B_packed[K/2][N]` layout where
//        every N-CTA at the same K range reads the SAME cache lines —
//        L2 amplifies its effective bandwidth ~4x. Roofline for the
//        weight-load alone: 50 MB / 200 GB/s = 250 µs (matches A's
//        376 µs at 67% efficiency; B is 794 µs at 31% efficiency).
//
// The split-K kernel is SHIPPED for future use at small-shape contexts
// (e.g. SSM/attention projections at K=K_h × n_heads) where it wins.
// At the production gate/up shape it is opt-in via the bench harness
// only; not wired into the dense_ffn dispatch path.
//
// The real fix requires changing the on-device B weight layout to
// K-major at quantization time — a model-loader-level change out of
// scope for this kernel session.
//
// Bench harness lives at:
//   crates/atlas-spark-bench/benches/nvfp4_gemm.rs
//   ATLAS_TARGET_MODEL=qwen3.6-27b ATLAS_BENCH_M=128 ATLAS_BENCH_K=5120 \
//     ATLAS_BENCH_N=17408 ATLAS_BENCH_K_SPLITS=2 \
//     cargo bench --bench nvfp4_gemm -p atlas-spark-bench
// =========================================================================

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define M_TILE      64
#define N_TILE_LG   128
#define K_STEP      64    // 2× MMA per inner iter (K=32 per MMA)
#define GROUP_SIZE  16    // NVFP4 group size — fixed by format

// E2M1 LUT — kept for any fallback / debugging paths.
// The native NVFP4 MMA does NOT need this LUT; the tensor core decodes
// E2M1 nibbles in hardware. Included for reference / non-MMA paths.
__device__ __constant__ float E2M1_LUT_NV[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// cp.async helpers (mirroring w4a16_gemm.cu definitions to keep this file
// self-contained).
__device__ __forceinline__ void cp_async_pred_16(void* dst_smem, const void* src_gmem, bool pred) {
    unsigned int dst = __cvta_generic_to_shared(dst_smem);
    unsigned int src_bytes = pred ? 16 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
                 :: "r"(dst), "l"(src_gmem), "r"(src_bytes));
}

__device__ __forceinline__ void cp_async_commit_nv() {
    asm volatile("cp.async.commit_group;");
}

__device__ __forceinline__ void cp_async_wait_all_nv() {
    asm volatile("cp.async.wait_group 0;");
}

// =========================================================================
// nvfp4_nvfp4_gemm — full W4A4 NVFP4 GEMM using native SM120 tensor cores.
//
// Inputs:
//   A_packed:  [M, K/2] uint8        — E2M1 nibbles (A activation, row-major)
//   A_scale:   [M, K/GROUP_SIZE] uint8  — FP8 E4M3 per-token per-group scales
//   B_packed:  [N, K/2] uint8        — E2M1 nibbles (B weights, col-major in K)
//   B_scale:   [N, K/GROUP_SIZE] uint8  — FP8 E4M3 per-group weight scales
//   scale2_ab: scalar fp32           — A_scale2 * B_scale2 product
//   C:         [M, N] bf16           — output
//
// Grid:  (ceil(N/128), ceil(M/64))
// Block: (128, 1, 1)  — 4 warps
// =========================================================================
extern "C" __global__
__launch_bounds__(128, 4)  // force 4 CTAs/SM — 116 regs/thread gives 4
void nvfp4_nvfp4_gemm_t_m64(
    const unsigned char* __restrict__ A_packed,  // [M, K/2]
    const unsigned char* __restrict__ A_scale,   // [M, K/16]
    const unsigned char* __restrict__ B_packed,  // [N, K/2]
    const unsigned char* __restrict__ B_scale,   // [N, K/16]
    const float scale2_ab,
    __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K
) {
    const unsigned int cta_n = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m = blockIdx.y * M_TILE;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;  // each warp owns 16 M rows
    const unsigned int group_id = lane_id >> 2;        // 8 groups per warp
    const unsigned int tid = lane_id & 3;              // 4 lanes per group

    // Double-buffered smem.
    // A_packed is [M_TILE][K_STEP/2] uint8 — 64 × 32 = 2048 B per buffer.
    // B_packed is [N_TILE][K_STEP/2] uint8 — 128 × 32 = 4096 B per buffer.
    // A_scale is [M_TILE][K_STEP/GROUP] uint8 — 64 × 4 = 256 B per buffer.
    // B_scale is [N_TILE][K_STEP/GROUP] uint8 — 128 × 4 = 512 B per buffer.
    //
    // FIX-PATH 3 NOTE (2026-05-23): tried PAD_AP=16 to break the
    // 2-way bank-conflict pattern (groups 0..7 hit banks {0,8,16,24,
    // 0,8,16,24} with stride-32 row). Measured: no perf change at
    // K=5120 N=17408 (791 vs 794 µs). Conclusion: SMEM bank conflicts
    // are NOT the bottleneck at the wide-N shape; GMEM B-load
    // bandwidth dominates because the HF row-major B layout prevents
    // cross-CTA cache-line sharing (Path A uses K-major B which
    // serves N-CTAs from the same K cache lines).
    __shared__ unsigned char smem_Ap[2][M_TILE][K_STEP / 2];
    __shared__ unsigned char smem_Bp[2][N_TILE_LG][K_STEP / 2];
    __shared__ unsigned char smem_As[2][M_TILE][K_STEP / GROUP_SIZE];
    __shared__ unsigned char smem_Bs[2][N_TILE_LG][K_STEP / GROUP_SIZE];

    // Accumulators: 16 N-fragments × 4 floats per thread.
    // n_tile_per_warp = N_TILE_LG / 8 = 16 (each warp processes all 16
    // n8 fragments since warps split on M, not N).
    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    // -------------------------------------------------------------------
    // Load helpers — cp.async issued by the whole CTA.
    //
    // A_packed: 64 rows × 32 bytes/row = 2048 bytes. With 128 threads
    // and 16-byte cp.async, each thread loads exactly 1 row pair:
    // thread t loads row (t>>1) bytes [(t&1)*16 .. (t&1)*16+15].
    //
    // B_packed: 128 rows × 32 bytes/row = 4096 bytes. Each thread loads
    // 2 × 16 bytes: row=t, cols [0..15] and [16..31].
    //
    // A_scale: 64 × 4 = 256 bytes. 128 threads × 2 bytes? Use the first
    // 64 threads to load 4 bytes each.
    //
    // B_scale: 128 × 4 = 512 bytes. Each thread loads 4 bytes.
    // -------------------------------------------------------------------
    #define LOAD_TILE(buf, kb) do { \
        /* A_packed: 64 rows, 32 bytes/row. 128 threads × 16 B = 2048 B */ \
        { \
            unsigned int row = threadIdx.x >> 1; \
            unsigned int col = (threadIdx.x & 1) << 4; \
            unsigned int gr = cta_m + row; \
            unsigned int gc_byte = (kb) / 2 + col; \
            bool valid = (gr < M) && (gc_byte + 15 < K / 2); \
            cp_async_pred_16(&smem_Ap[(buf)][row][col], \
                &A_packed[(unsigned long long)gr * (K / 2) + gc_byte], valid); \
        } \
        /* B_packed: 128 rows, 32 bytes/row. 128 threads × (2×16 B) = 4096 B */ \
        { \
            unsigned int row = threadIdx.x; \
            unsigned int gn = cta_n + row; \
            unsigned int gc_byte = (kb) / 2; \
            bool valid = (gn < N) && (gc_byte + 31 < K / 2); \
            cp_async_pred_16(&smem_Bp[(buf)][row][0], \
                &B_packed[(unsigned long long)gn * (K / 2) + gc_byte], valid); \
            cp_async_pred_16(&smem_Bp[(buf)][row][16], \
                &B_packed[(unsigned long long)gn * (K / 2) + gc_byte + 16], valid); \
        } \
        /* A_scale: 64 × 4 = 256 B. First 64 threads load 4 bytes each. */ \
        if (threadIdx.x < M_TILE) { \
            unsigned int row = threadIdx.x; \
            unsigned int gr = cta_m + row; \
            unsigned int gg_base = (kb) / GROUP_SIZE; \
            _Pragma("unroll") \
            for (int g = 0; g < K_STEP / GROUP_SIZE; g++) { \
                bool valid = (gr < M) && (gg_base + g < K / GROUP_SIZE); \
                smem_As[(buf)][row][g] = valid \
                    ? A_scale[(unsigned long long)gr * (K / GROUP_SIZE) + gg_base + g] \
                    : (unsigned char)0; \
            } \
        } \
        /* B_scale: 128 × 4 = 512 B. Each thread loads 4 bytes. */ \
        { \
            unsigned int row = threadIdx.x; \
            unsigned int gn = cta_n + row; \
            unsigned int gg_base = (kb) / GROUP_SIZE; \
            _Pragma("unroll") \
            for (int g = 0; g < K_STEP / GROUP_SIZE; g++) { \
                bool valid = (gn < N) && (gg_base + g < K / GROUP_SIZE); \
                smem_Bs[(buf)][row][g] = valid \
                    ? B_scale[(unsigned long long)gn * (K / GROUP_SIZE) + gg_base + g] \
                    : (unsigned char)0; \
            } \
        } \
    } while(0)

    // -------------------------------------------------------------------
    // FIX PATH 2 COMPUTE — hardware block-scaled NVFP4 MMA (m16n8k64).
    //
    // Per the SM120 CUTE traits (mma_traits_sm120.hpp:136):
    //   - A fragment: 4× uint32 per thread (16 B) covers k=64 e2m1
    //   - B fragment: 2× uint32 per thread (8 B) covers k=64 e2m1
    //   - SFA scale register: 32-bit packing 4× e4m3 group scales for
    //       m_row = ((lane >> 2) & 7) + 8 * (lane & 1)
    //     (the 16 'real' contributors; the other 16 lanes mirror via
    //     the 2:0 broadcast in SFALayout)
    //   - SFB scale register: 32-bit packing 4× e4m3 group scales for
    //       n_col = (lane >> 2) & 7
    //     (only 8 unique values needed; the 4:0 broadcast duplicates
    //     across the 4 lanes within a column-group)
    //
    // A/B fragment byte layout per-lane (TN, K-major in SMEM, matches
    // the m16n8k32 layout doubled in K dimension):
    //   a0 = smem_Ap[fr0][4*tid       .. +3]
    //   a1 = smem_Ap[fr1][4*tid       .. +3]
    //   a2 = smem_Ap[fr0][16 + 4*tid  .. +3]
    //   a3 = smem_Ap[fr1][16 + 4*tid  .. +3]
    //   b0 = smem_Bp[nc][4*tid       .. +3]
    //   b1 = smem_Bp[nc][16 + 4*tid  .. +3]
    // where fr0 = warp_m_offset + group_id, fr1 = fr0 + 8, tid = lane&3
    // and group_id = lane >> 2.
    //
    // ONE hardware MMA covers k=64 = 4 NVFP4 groups (GROUP_SIZE=16). The
    // tensor core internally multiplies each group's product by its
    // e4m3 scale before accumulating into f32 — no manual rescaling.
    // The per-tensor scale2_ab is still applied in the epilogue.
    // -------------------------------------------------------------------
    /* OPT (PATH 2.1, 2026-05-23): collapse 4-byte scale gather to single \
     * u32 SMEM load. Per-row scale stride = K_STEP/GROUP_SIZE = 4 bytes = \
     * naturally u32-aligned (the smem arrays are declared as [.][.][4]). \
     * Previously: 4 byte-loads + 3 shifts + 3 ORs per sfa, AND repeated \
     * inside the 16-iter sfb gather → 4×16 byte loads + 48 shift/ORs per \
     * COMPUTE_KSTEP. Now: 1 u32 load per sfa + 16 u32 loads per sfb. \
     * Cuts the scale-gather instruction count by ~6x. */ \
    #define COMPUTE_KSTEP(buf) do { \
        unsigned int fr0 = warp_m_offset + group_id; \
        unsigned int fr1 = fr0 + 8; \
        unsigned int a_col_byte = 4 * tid; \
        unsigned int a0 = *(const unsigned int*)&smem_Ap[(buf)][fr0][a_col_byte]; \
        unsigned int a1 = *(const unsigned int*)&smem_Ap[(buf)][fr1][a_col_byte]; \
        unsigned int a2 = *(const unsigned int*)&smem_Ap[(buf)][fr0][a_col_byte + 16]; \
        unsigned int a3 = *(const unsigned int*)&smem_Ap[(buf)][fr1][a_col_byte + 16]; \
        \
        /* SFA: per-thread M row per CUTE SFALayout = ((lane>>2)&7)+8*(lane&1). \
         * smem_As[buf][m][0..3] is 4 byte-aligned → single u32 load. */ \
        unsigned int m_sfa = ((threadIdx.x & 31) >> 2) + 8u * (threadIdx.x & 1u); \
        m_sfa += warp_m_offset; \
        unsigned int sfa = *(const unsigned int*)&smem_As[(buf)][m_sfa][0]; \
        \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_Bp[(buf)][nc][a_col_byte]; \
            unsigned int b1 = *(const unsigned int*)&smem_Bp[(buf)][nc][a_col_byte + 16]; \
            unsigned int sfb = *(const unsigned int*)&smem_Bs[(buf)][nc][0]; \
            asm volatile( \
                "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X." \
                "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13}," \
                "{%14},{%15,%16},{%17},{%18,%19};" \
                : "=f"(acc[nt][0]), "=f"(acc[nt][1]), "=f"(acc[nt][2]), "=f"(acc[nt][3]) \
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), \
                  "r"(b0), "r"(b1), \
                  "f"(acc[nt][0]), "f"(acc[nt][1]), "f"(acc[nt][2]), "f"(acc[nt][3]), \
                  "r"(sfa), "h"((unsigned short)0), "h"((unsigned short)0), \
                  "r"(sfb), "h"((unsigned short)0), "h"((unsigned short)0)); \
        } \
    } while(0)

    // -------------------------------------------------------------------
    // Prolog: load first tile, wait.
    // -------------------------------------------------------------------
    LOAD_TILE(0, 0);
    cp_async_commit_nv();
    cp_async_wait_all_nv();
    __syncthreads();

    // -------------------------------------------------------------------
    // Main loop: LOAD(nxt) || COMPUTE(cur) → wait → sync
    // -------------------------------------------------------------------
    int cur = 0;
    for (unsigned int k_base = K_STEP; k_base < K; k_base += K_STEP) {
        int nxt = 1 - cur;
        LOAD_TILE(nxt, k_base);
        cp_async_commit_nv();
        COMPUTE_KSTEP(cur);
        cp_async_wait_all_nv();
        __syncthreads();
        cur = nxt;
    }
    COMPUTE_KSTEP(cur);

    #undef LOAD_TILE
    #undef COMPUTE_KSTEP

    // -------------------------------------------------------------------
    // Apply per-tensor scale2_ab and write BF16 output.
    // -------------------------------------------------------------------
    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt * 8 + tid * 2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C[(unsigned long long)r0 * N + c0] = __float2bfloat16(acc[nt][0] * scale2_ab);
        if (r0 < M && c1 < N) C[(unsigned long long)r0 * N + c1] = __float2bfloat16(acc[nt][1] * scale2_ab);
        if (r1 < M && c0 < N) C[(unsigned long long)r1 * N + c0] = __float2bfloat16(acc[nt][2] * scale2_ab);
        if (r1 < M && c1 < N) C[(unsigned long long)r1 * N + c1] = __float2bfloat16(acc[nt][3] * scale2_ab);
    }
}

// =========================================================================
// FIX PATH 3 LANDED 2026-05-23 (gated behind ATLAS_E2M1_GEMM=1)
// =========================================================================
// Split-K variant. Addresses the wide-N=17408 wave-parallelism deficit:
//
//   At M=128 K=5120 N=17408 the non-split kernel needs 137×2 = 274 CTAs.
//   GB10 has 48 SMs × ~4 CTAs/SM = ~192 in-flight max → 1.4 waves.
//   The 1.4-wave second pass leaves ~82 SMs underutilized and the
//   OMMA.SF latency chain inside the 16-N-fragment loop dominates.
//
// Strategy: split the K axis into NUM_K_SPLITS independent shards. Each
// shard computes a partial product over K/SPLITS and writes to an
// FP32 scratch buffer of shape [SPLITS, M, N]. A separate reduce
// kernel sums + applies scale2_ab + writes BF16. This:
//   (a) Multiplies CTA count by SPLITS — at SPLITS=2 we have 548 CTAs,
//       2.85 waves. The longer tail keeps SMs busy hiding OMMA latency.
//   (b) Halves per-CTA inner-K work → each CTA finishes faster, more
//       independent CTAs become available to the scheduler.
//   (c) Reduce kernel is bandwidth-bound but tiny (M=128 × N=17408 ×
//       SPLITS=2 × 4B = 17.4 MB) — negligible vs the GEMM.
//
// Correctness: the existing per-CTA accumulator already covers the full
// K range; splitting K cleanly partitions the sum (linear). Per-group
// scales are applied in-MMA by the tensor core, so no inter-shard
// rescaling needed. Per-tensor scale2_ab is held until the reduce
// kernel so partials stay at native MMA magnitudes.
//
// Constraint: K must be divisible by (K_STEP × SPLITS) for clean tiling.
// For K=5120 K_STEP=64: SPLITS ∈ {1,2,4,5,8,10,16,20,40,80} (all factors
// of K/K_STEP=80). Production picks 2 — minimal scratch, max wave gain.
//
// Grid: (ceil(N/128), ceil(M/64), K_SPLITS)
// Block: (128, 1, 1)
// =========================================================================
extern "C" __global__
__launch_bounds__(128, 4)
void nvfp4_nvfp4_gemm_t_m64_splitk(
    const unsigned char* __restrict__ A_packed,
    const unsigned char* __restrict__ A_scale,
    const unsigned char* __restrict__ B_packed,
    const unsigned char* __restrict__ B_scale,
    float* __restrict__ C_partial,           // [K_SPLITS, M, N] FP32 scratch
    unsigned int M, unsigned int N, unsigned int K,
    unsigned int K_SPLITS
) {
    const unsigned int cta_n  = blockIdx.x * N_TILE_LG;
    const unsigned int cta_m  = blockIdx.y * M_TILE;
    const unsigned int split  = blockIdx.z;
    const unsigned int warp_id = threadIdx.x / 32;
    const unsigned int lane_id = threadIdx.x % 32;
    const unsigned int warp_m_offset = warp_id * 16;
    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid = lane_id & 3;

    // K range for this split. K is required to be divisible by
    // (K_STEP * K_SPLITS) — the host wrapper enforces this; if not we
    // gracefully clamp.
    unsigned int k_chunk = K / K_SPLITS;
    // Round chunk down to K_STEP multiple to keep tiling clean.
    k_chunk = (k_chunk / K_STEP) * K_STEP;
    if (k_chunk == 0) k_chunk = K_STEP;
    unsigned int k_start = split * k_chunk;
    unsigned int k_end = (split == K_SPLITS - 1) ? K : (k_start + k_chunk);
    if (k_start >= K) return;

    __shared__ unsigned char smem_Ap[2][M_TILE][K_STEP / 2];
    __shared__ unsigned char smem_Bp[2][N_TILE_LG][K_STEP / 2];
    __shared__ unsigned char smem_As[2][M_TILE][K_STEP / GROUP_SIZE];
    __shared__ unsigned char smem_Bs[2][N_TILE_LG][K_STEP / GROUP_SIZE];

    float acc[16][4];
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        acc[i][0] = 0.0f; acc[i][1] = 0.0f;
        acc[i][2] = 0.0f; acc[i][3] = 0.0f;
    }

    #define LOAD_TILE_SK(buf, kb) do { \
        { \
            unsigned int row = threadIdx.x >> 1; \
            unsigned int col = (threadIdx.x & 1) << 4; \
            unsigned int gr = cta_m + row; \
            unsigned int gc_byte = (kb) / 2 + col; \
            bool valid = (gr < M) && (gc_byte + 15 < K / 2); \
            cp_async_pred_16(&smem_Ap[(buf)][row][col], \
                &A_packed[(unsigned long long)gr * (K / 2) + gc_byte], valid); \
        } \
        { \
            unsigned int row = threadIdx.x; \
            unsigned int gn = cta_n + row; \
            unsigned int gc_byte = (kb) / 2; \
            bool valid = (gn < N) && (gc_byte + 31 < K / 2); \
            cp_async_pred_16(&smem_Bp[(buf)][row][0], \
                &B_packed[(unsigned long long)gn * (K / 2) + gc_byte], valid); \
            cp_async_pred_16(&smem_Bp[(buf)][row][16], \
                &B_packed[(unsigned long long)gn * (K / 2) + gc_byte + 16], valid); \
        } \
        if (threadIdx.x < M_TILE) { \
            unsigned int row = threadIdx.x; \
            unsigned int gr = cta_m + row; \
            unsigned int gg_base = (kb) / GROUP_SIZE; \
            _Pragma("unroll") \
            for (int g = 0; g < K_STEP / GROUP_SIZE; g++) { \
                bool valid = (gr < M) && (gg_base + g < K / GROUP_SIZE); \
                smem_As[(buf)][row][g] = valid \
                    ? A_scale[(unsigned long long)gr * (K / GROUP_SIZE) + gg_base + g] \
                    : (unsigned char)0; \
            } \
        } \
        { \
            unsigned int row = threadIdx.x; \
            unsigned int gn = cta_n + row; \
            unsigned int gg_base = (kb) / GROUP_SIZE; \
            _Pragma("unroll") \
            for (int g = 0; g < K_STEP / GROUP_SIZE; g++) { \
                bool valid = (gn < N) && (gg_base + g < K / GROUP_SIZE); \
                smem_Bs[(buf)][row][g] = valid \
                    ? B_scale[(unsigned long long)gn * (K / GROUP_SIZE) + gg_base + g] \
                    : (unsigned char)0; \
            } \
        } \
    } while(0)

    #define COMPUTE_KSTEP_SK(buf) do { \
        unsigned int fr0 = warp_m_offset + group_id; \
        unsigned int fr1 = fr0 + 8; \
        unsigned int a_col_byte = 4 * tid; \
        unsigned int a0 = *(const unsigned int*)&smem_Ap[(buf)][fr0][a_col_byte]; \
        unsigned int a1 = *(const unsigned int*)&smem_Ap[(buf)][fr1][a_col_byte]; \
        unsigned int a2 = *(const unsigned int*)&smem_Ap[(buf)][fr0][a_col_byte + 16]; \
        unsigned int a3 = *(const unsigned int*)&smem_Ap[(buf)][fr1][a_col_byte + 16]; \
        unsigned int m_sfa = ((threadIdx.x & 31) >> 2) + 8u * (threadIdx.x & 1u); \
        m_sfa += warp_m_offset; \
        unsigned int sfa = *(const unsigned int*)&smem_As[(buf)][m_sfa][0]; \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            unsigned int nc = nt * 8 + group_id; \
            unsigned int b0 = *(const unsigned int*)&smem_Bp[(buf)][nc][a_col_byte]; \
            unsigned int b1 = *(const unsigned int*)&smem_Bp[(buf)][nc][a_col_byte + 16]; \
            unsigned int sfb = *(const unsigned int*)&smem_Bs[(buf)][nc][0]; \
            asm volatile( \
                "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X." \
                "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 " \
                "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13}," \
                "{%14},{%15,%16},{%17},{%18,%19};" \
                : "=f"(acc[nt][0]), "=f"(acc[nt][1]), "=f"(acc[nt][2]), "=f"(acc[nt][3]) \
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), \
                  "r"(b0), "r"(b1), \
                  "f"(acc[nt][0]), "f"(acc[nt][1]), "f"(acc[nt][2]), "f"(acc[nt][3]), \
                  "r"(sfa), "h"((unsigned short)0), "h"((unsigned short)0), \
                  "r"(sfb), "h"((unsigned short)0), "h"((unsigned short)0)); \
        } \
    } while(0)

    LOAD_TILE_SK(0, k_start);
    cp_async_commit_nv();
    cp_async_wait_all_nv();
    __syncthreads();

    int cur = 0;
    for (unsigned int k_base = k_start + K_STEP; k_base < k_end; k_base += K_STEP) {
        int nxt = 1 - cur;
        LOAD_TILE_SK(nxt, k_base);
        cp_async_commit_nv();
        COMPUTE_KSTEP_SK(cur);
        cp_async_wait_all_nv();
        __syncthreads();
        cur = nxt;
    }
    COMPUTE_KSTEP_SK(cur);

    #undef LOAD_TILE_SK
    #undef COMPUTE_KSTEP_SK

    // Write FP32 partial. Reduce kernel will sum across split-axis and
    // apply scale2_ab + BF16 cast.
    const unsigned long long mn_stride = (unsigned long long)M * (unsigned long long)N;
    const unsigned long long base_off = (unsigned long long)split * mn_stride;
    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        unsigned int c0 = cta_n + nt * 8 + tid * 2;
        unsigned int c1 = c0 + 1;
        unsigned int r0 = cta_m + warp_m_offset + group_id;
        unsigned int r1 = r0 + 8;
        if (r0 < M && c0 < N) C_partial[base_off + (unsigned long long)r0 * N + c0] = acc[nt][0];
        if (r0 < M && c1 < N) C_partial[base_off + (unsigned long long)r0 * N + c1] = acc[nt][1];
        if (r1 < M && c0 < N) C_partial[base_off + (unsigned long long)r1 * N + c0] = acc[nt][2];
        if (r1 < M && c1 < N) C_partial[base_off + (unsigned long long)r1 * N + c1] = acc[nt][3];
    }
}

// Reduce kernel: sum K_SPLITS partials and scale into BF16 output.
// Grid: (ceil(N/256), M, 1)  Block: (256, 1, 1)
extern "C" __global__ void nvfp4_splitk_reduce(
    const float* __restrict__ C_partial,
    __nv_bfloat16* __restrict__ C,
    const float scale2_ab,
    unsigned int M, unsigned int N, unsigned int K_SPLITS
) {
    unsigned int row = blockIdx.y;
    unsigned int col = blockIdx.x * 256 + threadIdx.x;
    if (row >= M || col >= N) return;

    const unsigned long long mn_stride = (unsigned long long)M * (unsigned long long)N;
    float sum = 0.0f;
    #pragma unroll 4
    for (unsigned int s = 0; s < K_SPLITS; s++) {
        sum += C_partial[(unsigned long long)s * mn_stride + (unsigned long long)row * N + col];
    }
    C[(unsigned long long)row * N + col] = __float2bfloat16(sum * scale2_ab);
}
