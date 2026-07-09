// SPDX-License-Identifier: AGPL-3.0-only
//
// w4a16_dequant_prmt_proto — COMPILE-ONLY prototype from the 2026-07-08
// dequant instruction audit. NOT wired into the engine.
//
// Context: SASS audit of w4a16_gemm_t_m32_n64* showed the per-K-step
// B-dequant stage costs ~94 instr/thread (9 LDS.U8 + 16 SMEM-LUT LDS +
// 17 FMUL + 8 F2FP cvt + 8 STS.U16 + bit-manip/addressing) — 38% of the
// 248-instr main loop. Hidden behind the bandwidth wall at M<=32, but it
// scales with ceil(M/32) y-tiles at batched-verify M (concurrency), where
// it becomes part of the compute ceiling.
//
// This file prototypes the Marlin-style replacement: instead of per-ELEMENT
//     f32 = smem_LUT[nibble] * sv;  cvt.rn.satfinite.e4m3(f32)
// build a per-(group,column) 8-entry E4M3 TABLE once (8 FMUL + 4 F2FP),
// then convert nibbles by REGISTER PRMT byte-select + sign XOR — no SMEM
// LUT loads, no per-element FMUL, no per-element F2FP.
//
// BIT-EXACTNESS ARGUMENT (md5 constitution):
//   Baseline byte for nibble n:  b(n) = cvt_rn_satfinite_e4m3( LUT[n] * sv )
//   LUT[n] = sgn(n) * LUT[n & 7] with sgn = -1 iff (n & 8)  (E2M1_LUT layout,
//   incl. LUT[8] = -0.0). IEEE-754 f32 multiply is sign-symmetric:
//   (-x)*sv = -(x*sv) BIT-exactly, and cvt.rn.satfinite is sign-symmetric
//   (round-to-nearest-even and the +/-448 satfinite clamp are both symmetric,
//   +/-0 preserved). Hence for finite sv:
//       b(n) = b(n & 7) XOR (n & 8) << 4          -- the sign bit (0x80)
//   The table entries ARE b(0..7) computed by the identical FMUL + cvt
//   sequence, so the selected byte is bit-identical to the baseline byte.
//   Sole caveat: sv = NaN (scale byte 0x7F/0xFF in the checkpoint) yields
//   0x7F from cvt for EVERY element in the baseline, while the XOR path
//   yields 0xFF (also NaN) for negative nibbles. Real checkpoints have no
//   NaN group scales; a debug assert can guard this.
//
// MEASURED (static SASS, sm_121a, this file): see audit report —
// proto_dequant_lut loop vs proto_dequant_prmt loop, same bytes in/out.
//
// Kernels:
//   proto_dequant_lut       — verbatim replica of the DEQUANT_T_M32_N64 stage
//   proto_dequant_prmt      — PRMT-table variant, same input/output layout
//   proto_dequant_validate  — exhaustive 256-nibble-pair x scale grid
//                             equivalence check (run later on the GPU box:
//                             expects mismatches == 0 for finite scales)

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define GROUP_SIZE 16
#define K_STEP_T 32
#define N_TILE 64
#define BP_PAD 16

__device__ __constant__ float PROTO_E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

// ─────────────────────────────────────────────────────────────────────
// Shared helpers
// ─────────────────────────────────────────────────────────────────────

// Build the 8-entry E4M3 magnitude table for one (group, column) scale.
// Byte j of {t0,t1} = cvt.rn.satfinite.e4m3( LUT[j] * sv ), j = 0..7.
// Cost: 8 FMUL + 4 F2FP(pack2) + 2 merges.
__device__ __forceinline__ void build_e4m3_table(float sv, unsigned& t0, unsigned& t1) {
    unsigned short p01, p23, p45, p67;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(p01) : "f"(0.5f * sv), "f"(0.0f * sv));
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(p23) : "f"(1.5f * sv), "f"(1.0f * sv));
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(p45) : "f"(3.0f * sv), "f"(2.0f * sv));
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(p67) : "f"(6.0f * sv), "f"(4.0f * sv));
    t0 = ((unsigned)p23 << 16) | p01;
    t1 = ((unsigned)p67 << 16) | p45;
}

// Dequant 4 packed W4 bytes (8 weights, K-ascending: byte j holds k=2j lo
// nibble, k=2j+1 hi nibble) into 8 E4M3 bytes in K order, via 2 table PRMTs.
// Cost: ~20 reg ops, zero SMEM traffic, zero FMUL/F2FP.
__device__ __forceinline__ uint2 prmt_dequant8(unsigned w, unsigned t0, unsigned t1) {
    // Compact the 4 LOW nibble magnitudes into PRMT selector nibbles 0..3.
    unsigned xlo = w & 0x07070707u;
    unsigned tlo = (xlo | (xlo >> 4)) & 0x00FF00FFu;
    unsigned sel_lo = (tlo | (tlo >> 8)) & 0x0000FFFFu;
    // Same for the 4 HIGH nibbles.
    unsigned xhi = (w >> 4) & 0x07070707u;
    unsigned thi = (xhi | (xhi >> 4)) & 0x00FF00FFu;
    unsigned sel_hi = (thi | (thi >> 8)) & 0x0000FFFFu;

    unsigned mag_lo, mag_hi;
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(mag_lo) : "r"(t0), "r"(t1), "r"(sel_lo));
    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(mag_hi) : "r"(t0), "r"(t1), "r"(sel_hi));

    // Sign injection: E2M1 nibble bit3 → E4M3 byte bit7 (XOR handles sv<0:
    // table magnitudes already carry sv's sign; nibble sign FLIPS it).
    unsigned out_lo = mag_lo ^ ((w << 4) & 0x80808080u);
    unsigned out_hi = mag_hi ^ (w & 0x80808080u);

    // Interleave back to K order: k = lo0,hi0,lo1,hi1 | lo2,hi2,lo3,hi3.
    uint2 r;
    asm("prmt.b32 %0, %1, %2, 0x5140;" : "=r"(r.x) : "r"(out_lo), "r"(out_hi));
    asm("prmt.b32 %0, %1, %2, 0x7362;" : "=r"(r.y) : "r"(out_lo), "r"(out_hi));
    return r;
}

// ─────────────────────────────────────────────────────────────────────
// proto_dequant_lut — verbatim replica of the DEQUANT_T_M32_N64 stage of
// w4a16_gemm_t_m32_n64 (SMEM LUT + per-element FMUL + F2FP). The loop over
// `iters` stands in for the K-loop; buffers alternate to defeat hoisting.
// ─────────────────────────────────────────────────────────────────────
extern "C" __global__ void proto_dequant_lut(
    const unsigned char* __restrict__ Bp_g,   // [2][K_STEP_T/2][N_TILE+BP_PAD]
    const unsigned char* __restrict__ Bs_g,   // [2][2][N_TILE+BP_PAD]
    const float scale2,
    unsigned int iters,
    unsigned int* __restrict__ sink
) {
    __shared__ unsigned char smem_Bp[2][K_STEP_T / 2][N_TILE + BP_PAD];
    __shared__ unsigned char smem_Bs[2][2][N_TILE + BP_PAD];
    __shared__ unsigned char smem_B_fp8[N_TILE][K_STEP_T];
    __shared__ float smem_LUT[16];

    if (threadIdx.x < 16) smem_LUT[threadIdx.x] = PROTO_E2M1_LUT[threadIdx.x];
    for (unsigned i = threadIdx.x; i < sizeof(smem_Bp); i += blockDim.x)
        ((unsigned char*)smem_Bp)[i] = Bp_g[i];
    for (unsigned i = threadIdx.x; i < sizeof(smem_Bs); i += blockDim.x)
        ((unsigned char*)smem_Bs)[i] = Bs_g[i];
    __syncthreads();

    const unsigned my_n = threadIdx.x >> 1;   // 0..63
    const unsigned half = threadIdx.x & 1;    // 0..1

    for (unsigned it = 0; it < iters; it++) {
        const unsigned buf = it & 1;
        unsigned char sb = smem_Bs[buf][half][my_n];
        __nv_fp8_e4m3 f;
        *(unsigned char*)&f = sb;
        float sv = (float)f * scale2;
        unsigned kp0 = half << 3;
        #pragma unroll
        for (unsigned kp = kp0; kp < kp0 + 8; kp++) {
            unsigned char packed = smem_Bp[buf][kp][my_n];
            float lo = smem_LUT[packed & 0xF] * sv;
            float hi = smem_LUT[packed >> 4] * sv;
            unsigned short fp8_pair;
            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                         : "=h"(fp8_pair) : "f"(hi), "f"(lo));
            *(unsigned short*)&smem_B_fp8[my_n][kp * 2] = fp8_pair;
        }
        __syncthreads();
        // Consume so the store is live (stands in for the MMA read).
        if ((it & 15) == 15) {
            unsigned v = *(unsigned*)&smem_B_fp8[my_n][half * 16];
            if (v == 0xDEADBEEFu) atomicAdd(sink, 1u);
        }
        __syncthreads();
    }
}

// ─────────────────────────────────────────────────────────────────────
// proto_dequant_prmt — PRMT-table variant. Identical input/output layout
// and BIT-IDENTICAL output bytes (see header argument).
// ─────────────────────────────────────────────────────────────────────
extern "C" __global__ void proto_dequant_prmt(
    const unsigned char* __restrict__ Bp_g,
    const unsigned char* __restrict__ Bs_g,
    const float scale2,
    unsigned int iters,
    unsigned int* __restrict__ sink
) {
    __shared__ unsigned char smem_Bp[2][K_STEP_T / 2][N_TILE + BP_PAD];
    __shared__ unsigned char smem_Bs[2][2][N_TILE + BP_PAD];
    __shared__ unsigned char smem_B_fp8[N_TILE][K_STEP_T];

    for (unsigned i = threadIdx.x; i < sizeof(smem_Bp); i += blockDim.x)
        ((unsigned char*)smem_Bp)[i] = Bp_g[i];
    for (unsigned i = threadIdx.x; i < sizeof(smem_Bs); i += blockDim.x)
        ((unsigned char*)smem_Bs)[i] = Bs_g[i];
    __syncthreads();

    const unsigned my_n = threadIdx.x >> 1;
    const unsigned half = threadIdx.x & 1;

    for (unsigned it = 0; it < iters; it++) {
        const unsigned buf = it & 1;
        unsigned char sb = smem_Bs[buf][half][my_n];
        __nv_fp8_e4m3 f;
        *(unsigned char*)&f = sb;
        float sv = (float)f * scale2;

        unsigned t0, t1;
        build_e4m3_table(sv, t0, t1);

        // 8 packed bytes for this thread's K-half, K-contiguous in the
        // [kp][my_n] tile → strided by row: gather as 2 u32 pairs.
        // (In the real kernel smem_Bp is [kp][n]; per-thread bytes sit in
        // 8 different rows, so gather 4+4 via 2 loops of byte loads OR
        // restage; here we index rows directly to keep layout identical.)
        unsigned kp0 = half << 3;
        #pragma unroll
        for (unsigned q = 0; q < 2; q++) {
            unsigned w = (unsigned)smem_Bp[buf][kp0 + q * 4 + 0][my_n]
                       | ((unsigned)smem_Bp[buf][kp0 + q * 4 + 1][my_n] << 8)
                       | ((unsigned)smem_Bp[buf][kp0 + q * 4 + 2][my_n] << 16)
                       | ((unsigned)smem_Bp[buf][kp0 + q * 4 + 3][my_n] << 24);
            uint2 r = prmt_dequant8(w, t0, t1);
            *(unsigned*)&smem_B_fp8[my_n][(kp0 + q * 4) * 2]     = r.x;
            *(unsigned*)&smem_B_fp8[my_n][(kp0 + q * 4) * 2 + 4] = r.y;
        }
        __syncthreads();
        if ((it & 15) == 15) {
            unsigned v = *(unsigned*)&smem_B_fp8[my_n][half * 16];
            if (v == 0xDEADBEEFu) atomicAdd(sink, 1u);
        }
        __syncthreads();
    }
}

// ─────────────────────────────────────────────────────────────────────
// proto_dequant_validate — exhaustive equivalence check.
// Grid-stride over all (packed byte 0..255) x (scale byte 0..255) pairs
// for a given scale2; counts BYTE mismatches between the baseline LUT
// path and the PRMT path. Expected: *mismatches == 0 (finite scales;
// NaN scale bytes 0x7F/0xFF both produce NaN encodings — see header).
// Run later on the GPU box; compile-only for this audit.
// ─────────────────────────────────────────────────────────────────────
extern "C" __global__ void proto_dequant_validate(
    const float scale2,
    unsigned int* __restrict__ mismatches
) {
    unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= 256 * 256) return;
    unsigned packed = idx & 0xFF;
    unsigned scale_byte = idx >> 8;

    __nv_fp8_e4m3 f;
    *(unsigned char*)&f = (unsigned char)scale_byte;
    float sv = (float)f * scale2;

    // Baseline path (per-element LUT * sv → cvt), both nibbles.
    float lo = PROTO_E2M1_LUT[packed & 0xF] * sv;
    float hi = PROTO_E2M1_LUT[packed >> 4] * sv;
    unsigned short ref_pair;
    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;"
                 : "=h"(ref_pair) : "f"(hi), "f"(lo));

    // PRMT path on a 4-byte replication of `packed` (checks all lanes).
    unsigned t0, t1;
    build_e4m3_table(sv, t0, t1);
    uint2 r = prmt_dequant8(packed * 0x01010101u, t0, t1);
    // K order: byte0 = lo nibble of byte0, byte1 = hi nibble of byte0.
    unsigned short got_pair = (unsigned short)(r.x & 0xFFFFu);

    if (got_pair != ref_pair) atomicAdd(mismatches, 1u);
    // Also check lanes 2/3 and the second PRMT output for consistency.
    if ((r.x >> 16) != (r.x & 0xFFFFu)) atomicAdd(mismatches, 1u);
    if (r.y != r.x) atomicAdd(mismatches, 1u);
}
