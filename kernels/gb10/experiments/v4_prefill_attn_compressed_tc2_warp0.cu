// SPDX-License-Identifier: AGPL-3.0-only

// Isolated compile-only DeepSeek-V4 TC2 experiment. Warp 0 owns each tile's
// softmax and publishes lane-indexed BF16 P fragments plus online state to the
// three output-dimension warps. No production registry includes this file.

#include <cuda_bf16.h>
#include <stdint.h>

#define BR 16
#define TC2_KT 16
#define TC2_HD 512
// bf16 row stride of the K/V tile. 520*2 = 1040 B: ≡ 0 mod 16 (ldmatrix needs
// 16-B-aligned row addresses) and 260 words ≡ 4 (mod 32) (conflict-free).
#define TC2_KVPAD 520
// float4 row stride of the S-partial tile. 17*16 B = 272 B = 68 words ≡ 4
// (mod 32) — the same 4-bank-stride property, applied to the softmax read.
#define TC2_SPPAD 17

// ldmatrix: each lane supplies the address of ONE 8-bf16 row; the warp's 32
// addresses name four 8x8 matrices (lanes 8j..8j+7 → matrix j), returned in
// d0..d3. For a 16x16 tile the canonical address is
// &tile[(lane & 15) * stride + (lane >> 4) * 8].
__device__ __forceinline__ void tc2_ldm_x4(
    unsigned int& d0, unsigned int& d1, unsigned int& d2, unsigned int& d3,
    const __nv_bfloat16* p
) {
    const unsigned int a = (unsigned int)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3) : "r"(a));
}
__device__ __forceinline__ void tc2_ldm_x4_trans(
    unsigned int& d0, unsigned int& d1, unsigned int& d2, unsigned int& d3,
    const __nv_bfloat16* p
) {
    const unsigned int a = (unsigned int)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 "
                 "{%0,%1,%2,%3}, [%4];"
                 : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3) : "r"(a));
}

// Pack two f32 as bf16 into one u32 A-operand register: low half = lo (the
// lower column index), high half = hi — the same order tc's sP round trip
// produced, so the rounding is identical.
__device__ __forceinline__ unsigned int tc2_pack_bf16(float lo, float hi) {
    // Per-element __float2bfloat16 (NOT __floats2bfloat162_rn) so the rounding
    // is the same call tc's sP round trip made, element for element.
    const unsigned int a = __bfloat16_as_ushort(__float2bfloat16(lo));
    const unsigned int b = __bfloat16_as_ushort(__float2bfloat16(hi));
    return (b << 16) | a;
}

struct __align__(16) TC2Warp0Broadcast {
    unsigned int pa0, pa1, pa2, pa3;
    float eo0, eo1;
    float m0, l0, m1, l1;
};
static_assert(sizeof(TC2Warp0Broadcast) == 48, "broadcast ABI must remain 48 bytes");

__device__ __forceinline__ bool v4_tc2_aligned(const void* pointer,
                                                size_t alignment) {
    return (reinterpret_cast<uintptr_t>(pointer) & (alignment - 1u)) == 0u;
}

__device__ __forceinline__ bool v4_tc2_ranges_overlap(
    const void* lhs, size_t lhs_bytes, const void* rhs, size_t rhs_bytes
) {
    if (lhs_bytes == 0u || rhs_bytes == 0u) return false;
    const uintptr_t lhs_address = reinterpret_cast<uintptr_t>(lhs);
    const uintptr_t rhs_address = reinterpret_cast<uintptr_t>(rhs);
    return lhs_address <= rhs_address
        ? lhs_bytes > rhs_address - lhs_address
        : rhs_bytes > lhs_address - rhs_address;
}

extern "C" __global__ void v4_prefill_attn_compressed_tc2_warp0(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ V,
    const __nv_bfloat16* __restrict__ Kc,
    const __nv_bfloat16* __restrict__ Vc,
    const float* __restrict__ sinks,
    __nv_bfloat16* __restrict__ O,
    const unsigned int seq_len,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int n_comp,
    const unsigned int ratio,
    const unsigned int sliding_window,
    const float inv_sqrt_d
) {
    __shared__ __align__(16) __nv_bfloat16 sKV[TC2_KT * TC2_KVPAD];
    __shared__ __align__(16) float4 sSp[BR * TC2_SPPAD];
    __shared__ TC2Warp0Broadcast broadcast[32];

    const unsigned int required_q_blocks =
        seq_len / BR + ((seq_len % BR) != 0u);
    if (blockDim.x != 128u || blockDim.y != 1u || blockDim.z != 1u ||
        gridDim.x != num_q_heads || gridDim.y != required_q_blocks ||
        gridDim.z != 1u || num_kv_heads == 0u || head_dim != TC2_HD ||
        num_q_heads % num_kv_heads != 0u || ratio == 0u ||
        num_q_heads > UINT32_MAX / head_dim ||
        num_kv_heads > UINT32_MAX / head_dim ||
        !isfinite(inv_sqrt_d) || inv_sqrt_d <= 0.0f ||
        Q == nullptr || K == nullptr || V == nullptr || Kc == nullptr ||
        Vc == nullptr || O == nullptr)
        return;

    if (!v4_tc2_aligned(Q, 4u) || !v4_tc2_aligned(K, 16u) ||
        !v4_tc2_aligned(V, 16u) || !v4_tc2_aligned(Kc, 16u) ||
        !v4_tc2_aligned(Vc, 16u) || !v4_tc2_aligned(O, 4u) ||
        (sinks != nullptr && !v4_tc2_aligned(sinks, 4u)))
        return;

    const size_t output_bytes = static_cast<size_t>(seq_len) * num_q_heads *
                                head_dim * sizeof(__nv_bfloat16);
    const size_t kv_bytes = static_cast<size_t>(seq_len) * num_kv_heads *
                            head_dim * sizeof(__nv_bfloat16);
    const size_t compressed_bytes = static_cast<size_t>(n_comp) * head_dim *
                                    sizeof(__nv_bfloat16);
    const size_t sink_bytes = static_cast<size_t>(num_q_heads) * sizeof(float);
    // K, V, Kc, and Vc may alias read-only, including K=V=Kc=Vc for the
    // production dense route. O is writable and must be disjoint from every
    // input extent before any restrict-qualified access occurs.
    if (v4_tc2_ranges_overlap(O, output_bytes, Q, output_bytes) ||
        v4_tc2_ranges_overlap(O, output_bytes, K, kv_bytes) ||
        v4_tc2_ranges_overlap(O, output_bytes, V, kv_bytes) ||
        v4_tc2_ranges_overlap(O, output_bytes, Kc, compressed_bytes) ||
        v4_tc2_ranges_overlap(O, output_bytes, Vc, compressed_bytes) ||
        (sinks != nullptr &&
         v4_tc2_ranges_overlap(O, output_bytes, sinks, sink_bytes)))
        return;

    const unsigned int q_head = blockIdx.x;
    const unsigned int q_block = blockIdx.y;
    if (q_head >= num_q_heads) return;

    const unsigned int tid_x = threadIdx.x;
    const unsigned int warp = tid_x >> 5;          // 0..3: owns dims [warp*128, +128)
    const unsigned int laneid = tid_x & 31u;
    const unsigned int g = laneid >> 2;            // fragment group_id (row g / g+8)
    const unsigned int t = laneid & 3u;            // fragment tid (col pair t*2)
    const unsigned int ldm_row = laneid & 15u;     // ldmatrix row selector
    const unsigned int ldm_col = (laneid >> 4) * 8u;

    const unsigned int gqa = num_q_heads / num_kv_heads;
    const unsigned int kv_head = q_head / gqa;
    const unsigned int q_stride = num_q_heads * head_dim;
    const unsigned int kv_stride = num_kv_heads * head_dim;

    const unsigned int q_first = q_block * BR;
    const unsigned int q_last_excl =
        (q_first + BR < seq_len) ? (q_first + BR) : seq_len;
    if (q_first >= seq_len) return;

    // This thread's two fragment rows and their per-row softmax state.
    const unsigned int r0 = g, r1 = g + 8;
    const unsigned int qrow0 = q_first + r0, qrow1 = q_first + r1;
    const bool v0 = qrow0 < seq_len, v1 = qrow1 < seq_len;
    const unsigned int kvs0 =
        (sliding_window > 0u && qrow0 + 1u > sliding_window) ? (qrow0 + 1u - sliding_window) : 0u;
    const unsigned int kvl0 = qrow0 + 1u;
    const unsigned int kvs1 =
        (sliding_window > 0u && qrow1 + 1u > sliding_window) ? (qrow1 + 1u - sliding_window) : 0u;
    const unsigned int kvl1 = qrow1 + 1u;
    unsigned int cvis0 = (qrow0 + 1u) / ratio; if (cvis0 > n_comp) cvis0 = n_comp;
    unsigned int cvis1 = (qrow1 + 1u) / ratio; if (cvis1 > n_comp) cvis1 = n_comp;

    float m0 = -1e30f, l0 = 0.0f, m1 = -1e30f, l1 = 0.0f;

    // o_acc: 16 n-tiles (dims warp*128 + nt*8 + {t*2, t*2+1}) x 4 f32.
    float o_acc[16][4];
    #pragma unroll
    for (int nt = 0; nt < 16; nt++) {
        o_acc[nt][0] = 0.0f; o_acc[nt][1] = 0.0f;
        o_acc[nt][2] = 0.0f; o_acc[nt][3] = 0.0f;
    }

    // Q A-fragments for this warp's 128-dim k-slice, from global ONCE.
    // Invalid rows load zeros (their scores are masked to -INFINITY anyway).
    unsigned int qa[8][4];
    {
        const unsigned int col_base = warp * 128u + t * 2u;
        const __nv_bfloat16* Qr0 = v0
            ? Q + (size_t)qrow0 * q_stride + (size_t)q_head * head_dim
            : nullptr;
        const __nv_bfloat16* Qr1 = v1
            ? Q + (size_t)qrow1 * q_stride + (size_t)q_head * head_dim
            : nullptr;
        #pragma unroll
        for (int s = 0; s < 8; s++) {
            const unsigned int c0 = col_base + (unsigned int)s * 16u;
            const unsigned int c1 = c0 + 8u;
            qa[s][0] = v0 ? *reinterpret_cast<const unsigned int*>(Qr0 + c0) : 0u;
            qa[s][1] = v1 ? *reinterpret_cast<const unsigned int*>(Qr1 + c0) : 0u;
            qa[s][2] = v0 ? *reinterpret_cast<const unsigned int*>(Qr0 + c1) : 0u;
            qa[s][3] = v1 ? *reinterpret_cast<const unsigned int*>(Qr1 + c1) : 0u;
        }
    }

    const bool kv_same = (K == V);
    const bool comp_same = (Kc == Vc);

    // Stage COUNT keys of SRC into sKV key-major, zeroing rows [COUNT, 16).
    // Per thread d0 is FIXED (= (tid & 63) * 8) and kk walks {tid>>6, +2, ...},
    // so each of the 8 iterations is 64 lanes moving 1024 contiguous global
    // bytes into 1024 contiguous smem bytes — both sides conflict-free.
    // The zero fill is the 0*NaN guard: on a partial tile P is exactly 0 for
    // keys >= COUNT, but 0 * NaN (uninitialised smem) would poison the PV MMA.
    // (Garbage in the QK^T for those columns is harmless — the score for
    // column c >= COUNT is replaced by -INFINITY before the softmax, and D[m][n]
    // of an mma depends only on B[:, n], so a NaN column cannot leak sideways.)
    #define TC2_STAGE(SRC, STRIDE, HEAD_OFF, BASE, COUNT)                      \
    do {                                                                        \
        const unsigned int kk0 = tid_x >> 6;                                   \
        const unsigned int d0  = (tid_x & 63u) << 3;                           \
        _Pragma("unroll")                                                      \
        for (unsigned int i = 0; i < 8u; ++i) {                                \
            const unsigned int kk = kk0 + i * 2u;                              \
            uint4 vv = make_uint4(0u, 0u, 0u, 0u);                             \
            if (kk < (COUNT)) {                                                \
                vv = *reinterpret_cast<const uint4*>(                          \
                    (SRC) + (size_t)((BASE) + kk) * (STRIDE) + (HEAD_OFF) + d0);\
            }                                                                  \
            *reinterpret_cast<uint4*>(&sKV[kk * TC2_KVPAD + d0]) = vv;         \
        }                                                                      \
    } while (0)

    // One KT-key tile. Barriers: (1) the previous tile's PV has finished
    // reading sKV, (2) the tile is staged, (3) all four S partials are visible
    // AND every warp's QK^T reads of sKV have retired — which is exactly what
    // makes the non-aliased V re-stage legal in place.
    #define TC2_TILE(KSRC, VSRC, STRIDE, HEAD_OFF, BASE, COUNT, SAME,          \
                     LO0, HI0, LO1, HI1)                                       \
    do {                                                                        \
        __syncthreads();                                                       \
        TC2_STAGE(KSRC, STRIDE, HEAD_OFF, BASE, COUNT);                        \
        __syncthreads();                                                       \
        /* QK^T over this warp's 128 dims: 8 k-steps x 2 n-tiles. One          \
           ldmatrix.x4 per k-step yields BOTH n-tiles' B fragments:            \
             f0 = keys 0-7  dims kd..kd+7   -> b0 of n-tile 0                  \
             f1 = keys 8-15 dims kd..kd+7   -> b0 of n-tile 1                  \
             f2 = keys 0-7  dims kd+8..+15  -> b1 of n-tile 0                  \
             f3 = keys 8-15 dims kd+8..+15  -> b1 of n-tile 1               */ \
        {                                                                      \
            float sp[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};                     \
            _Pragma("unroll")                                                  \
            for (int s = 0; s < 8; s++) {                                      \
                const unsigned int kd = warp * 128u + (unsigned int)s * 16u;   \
                unsigned int f0, f1, f2, f3;                                   \
                tc2_ldm_x4(f0, f1, f2, f3,                                     \
                           &sKV[ldm_row * TC2_KVPAD + kd + ldm_col]);          \
                asm volatile(                                                  \
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "     \
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"   \
                    : "=f"(sp[0][0]), "=f"(sp[0][1]),                          \
                      "=f"(sp[0][2]), "=f"(sp[0][3])                           \
                    : "r"(qa[s][0]), "r"(qa[s][1]),                            \
                      "r"(qa[s][2]), "r"(qa[s][3]),                            \
                      "r"(f0), "r"(f2),                                        \
                      "f"(sp[0][0]), "f"(sp[0][1]),                            \
                      "f"(sp[0][2]), "f"(sp[0][3]));                           \
                asm volatile(                                                  \
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "     \
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"   \
                    : "=f"(sp[1][0]), "=f"(sp[1][1]),                          \
                      "=f"(sp[1][2]), "=f"(sp[1][3])                           \
                    : "r"(qa[s][0]), "r"(qa[s][1]),                            \
                      "r"(qa[s][2]), "r"(qa[s][3]),                            \
                      "r"(f1), "r"(f3),                                        \
                      "f"(sp[1][0]), "f"(sp[1][1]),                            \
                      "f"(sp[1][2]), "f"(sp[1][3]));                           \
            }                                                                  \
            /* warp w -> lane w of the float4 at [row][key]. */                \
            _Pragma("unroll")                                                  \
            for (int nt = 0; nt < 2; nt++) {                                   \
                const unsigned int c0 = (unsigned int)nt * 8u + t * 2u;        \
                ((float*)&sSp[r0 * TC2_SPPAD + c0])[warp]     = sp[nt][0];     \
                ((float*)&sSp[r0 * TC2_SPPAD + c0 + 1])[warp] = sp[nt][1];     \
                ((float*)&sSp[r1 * TC2_SPPAD + c0])[warp]     = sp[nt][2];     \
                ((float*)&sSp[r1 * TC2_SPPAD + c0 + 1])[warp] = sp[nt][3];     \
            }                                                                  \
        }                                                                      \
        __syncthreads();                                                       \
        /* V does not alias K: re-stage the same tile with V. Legal in place   \
           because barrier (3) retired every warp's QK^T reads of sKV, and the \
           softmax below touches only sSp, so it overlaps this staging.     */ \
        if (!(SAME)) { TC2_STAGE(VSRC, STRIDE, HEAD_OFF, BASE, COUNT); }       \
        /* Warp 0 alone reconstructs sSp and executes the exact incumbent \
           softmax sequence. Lane L publishes the A fragment and online state \
           consumed by lane L in every output-dimension warp. */ \
        unsigned int pa0, pa1, pa2, pa3; \
        float eo0, eo1; \
        if (warp == 0) { \
            float s0[4], s1[4]; \
            _Pragma("unroll") \
            for (int cix = 0; cix < 4; cix++) { \
                const unsigned int c = \
                    (cix / 2) * 8u + t * 2u + (unsigned int)(cix % 2); \
                const float4 z0 = sSp[r0 * TC2_SPPAD + c]; \
                const float4 z1 = sSp[r1 * TC2_SPPAD + c]; \
                const float a = z0.x + z0.y + z0.z + z0.w; \
                const float b = z1.x + z1.y + z1.z + z1.w; \
                const unsigned int kp = (BASE) + c; \
                const bool in0 = v0 && c < (COUNT) && kp >= (LO0) && kp < (HI0); \
                const bool in1 = v1 && c < (COUNT) && kp >= (LO1) && kp < (HI1); \
                s0[cix] = in0 ? (a * inv_sqrt_d) : -INFINITY; \
                s1[cix] = in1 ? (b * inv_sqrt_d) : -INFINITY; \
            } \
            float mx0 = fmaxf(fmaxf(s0[0], s0[1]), fmaxf(s0[2], s0[3])); \
            float mx1 = fmaxf(fmaxf(s1[0], s1[1]), fmaxf(s1[2], s1[3])); \
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xFFFFFFFFu, mx0, 1)); \
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xFFFFFFFFu, mx0, 2)); \
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xFFFFFFFFu, mx1, 1)); \
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xFFFFFFFFu, mx1, 2)); \
            const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1); \
            eo0 = __expf(m0 - mn0); eo1 = __expf(m1 - mn1); \
            float en0[4], en1[4], se0 = 0.0f, se1 = 0.0f; \
            _Pragma("unroll") \
            for (int cix = 0; cix < 4; cix++) { \
                en0[cix] = __expf(s0[cix] - mn0); \
                en1[cix] = __expf(s1[cix] - mn1); \
                se0 += en0[cix]; se1 += en1[cix]; \
            } \
            se0 += __shfl_xor_sync(0xFFFFFFFFu, se0, 1); \
            se0 += __shfl_xor_sync(0xFFFFFFFFu, se0, 2); \
            se1 += __shfl_xor_sync(0xFFFFFFFFu, se1, 1); \
            se1 += __shfl_xor_sync(0xFFFFFFFFu, se1, 2); \
            l0 = l0 * eo0 + se0; m0 = mn0; \
            l1 = l1 * eo1 + se1; m1 = mn1; \
            TC2Warp0Broadcast& slot = broadcast[laneid]; \
            slot.pa0 = tc2_pack_bf16(en0[0], en0[1]); \
            slot.pa1 = tc2_pack_bf16(en1[0], en1[1]); \
            slot.pa2 = tc2_pack_bf16(en0[2], en0[3]); \
            slot.pa3 = tc2_pack_bf16(en1[2], en1[3]); \
            slot.eo0 = eo0; slot.eo1 = eo1; \
            slot.m0 = m0; slot.l0 = l0; slot.m1 = m1; slot.l1 = l1; \
        } \
        /* This is one additional block barrier for the alias path. On the \
           non-alias path it also replaces the incumbent V-stage barrier. */ \
        __syncthreads(); \
        const TC2Warp0Broadcast slot = broadcast[laneid]; \
        pa0 = slot.pa0; pa1 = slot.pa1; pa2 = slot.pa2; pa3 = slot.pa3; \
        eo0 = slot.eo0; eo1 = slot.eo1; \
        m0 = slot.m0; l0 = slot.l0; m1 = slot.m1; l1 = slot.l1; \
        _Pragma("unroll") \
        for (int nt = 0; nt < 16; nt++) { \
            o_acc[nt][0] *= eo0; o_acc[nt][1] *= eo0; \
            o_acc[nt][2] *= eo1; o_acc[nt][3] *= eo1; \
        } \
                                      \
        /* PV. One ldmatrix.x4.TRANS per dim-pair yields both n-tiles: the     \
           .trans of the key-major tile is exactly B[key][dim] with the two    \
           contracted keys packed per register.                                \
             f0, f1 = b0, b1 of the n-tile at dims D                           \
             f2, f3 = b0, b1 of the n-tile at dims D+8                      */ \
        {                                                                      \
            _Pragma("unroll")                                                  \
            for (int p = 0; p < 8; p++) {                                      \
                const unsigned int D = warp * 128u + (unsigned int)p * 16u;    \
                unsigned int f0, f1, f2, f3;                                   \
                tc2_ldm_x4_trans(f0, f1, f2, f3,                               \
                                 &sKV[ldm_row * TC2_KVPAD + D + ldm_col]);     \
                asm volatile(                                                  \
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "     \
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"   \
                    : "=f"(o_acc[2 * p][0]), "=f"(o_acc[2 * p][1]),            \
                      "=f"(o_acc[2 * p][2]), "=f"(o_acc[2 * p][3])             \
                    : "r"(pa0), "r"(pa1), "r"(pa2), "r"(pa3),                  \
                      "r"(f0), "r"(f1),                                        \
                      "f"(o_acc[2 * p][0]), "f"(o_acc[2 * p][1]),              \
                      "f"(o_acc[2 * p][2]), "f"(o_acc[2 * p][3]));             \
                asm volatile(                                                  \
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "     \
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"   \
                    : "=f"(o_acc[2 * p + 1][0]), "=f"(o_acc[2 * p + 1][1]),    \
                      "=f"(o_acc[2 * p + 1][2]), "=f"(o_acc[2 * p + 1][3])     \
                    : "r"(pa0), "r"(pa1), "r"(pa2), "r"(pa3),                  \
                      "r"(f2), "r"(f3),                                        \
                      "f"(o_acc[2 * p + 1][0]), "f"(o_acc[2 * p + 1][1]),      \
                      "f"(o_acc[2 * p + 1][2]), "f"(o_acc[2 * p + 1][3]));     \
            }                                                                  \
        }                                                                      \
    } while (0)

    // ── raw arm: block-union sliding-window causal tiles ──
    {
        unsigned int union_lo = 0u;
        if (sliding_window > 0u && q_first + 1u > sliding_window)
            union_lo = q_first + 1u - sliding_window;
        const unsigned int union_hi = q_last_excl;
        for (unsigned int base = union_lo; base < union_hi; base += TC2_KT) {
            unsigned int count = union_hi - base;
            if (count > TC2_KT) count = TC2_KT;
            TC2_TILE(K, V, kv_stride, (size_t)kv_head * head_dim, base, count,
                     kv_same, kvs0, kvl0, kvs1, kvl1);
        }
    }

    // ── compressed arm: windowed-causal tiles ──
    {
        unsigned int comp_union = (q_last_excl > 0u) ? (q_last_excl / ratio) : 0u;
        if (comp_union > n_comp) comp_union = n_comp;
        for (unsigned int base = 0; base < comp_union; base += TC2_KT) {
            unsigned int count = comp_union - base;
            if (count > TC2_KT) count = TC2_KT;
            TC2_TILE(Kc, Vc, head_dim, (size_t)0u, base, count,
                     comp_same, 0u, cvis0, 0u, cvis1);
        }
    }
    #undef TC2_TILE
    #undef TC2_STAGE

    // ── sink: per-head logit in the denominator only ──
    if (sinks != nullptr) {
        const float sg = sinks[q_head];
        const float mn0 = fmaxf(m0, sg), mn1 = fmaxf(m1, sg);
        const float eo0 = __expf(m0 - mn0), eo1 = __expf(m1 - mn1);
        #pragma unroll
        for (int nt = 0; nt < 16; nt++) {
            o_acc[nt][0] *= eo0; o_acc[nt][1] *= eo0;
            o_acc[nt][2] *= eo1; o_acc[nt][3] *= eo1;
        }
        l0 = l0 * eo0 + __expf(sg - mn0);
        l1 = l1 * eo1 + __expf(sg - mn1);
        m0 = mn0; m1 = mn1;
    }

    // ── epilogue: normalize and store this warp's 128-dim slice ──
    {
        const float il0 = (l0 > 0.0f) ? (1.0f / l0) : 0.0f;
        const float il1 = (l1 > 0.0f) ? (1.0f / l1) : 0.0f;
        __nv_bfloat16* O0 = v0
            ? O + (size_t)qrow0 * q_stride + (size_t)q_head * head_dim
            : nullptr;
        __nv_bfloat16* O1 = v1
            ? O + (size_t)qrow1 * q_stride + (size_t)q_head * head_dim
            : nullptr;
        #pragma unroll
        for (int nt = 0; nt < 16; nt++) {
            const unsigned int c = warp * 128u + (unsigned int)nt * 8u + t * 2u;
            if (v0) {
                __nv_bfloat162 p;
                p.x = __float2bfloat16(o_acc[nt][0] * il0);
                p.y = __float2bfloat16(o_acc[nt][1] * il0);
                *reinterpret_cast<__nv_bfloat162*>(O0 + c) = p;
            }
            if (v1) {
                __nv_bfloat162 p;
                p.x = __float2bfloat16(o_acc[nt][2] * il1);
                p.y = __float2bfloat16(o_acc[nt][3] * il1);
                *reinterpret_cast<__nv_bfloat162*>(O1 + c) = p;
            }
        }
    }
}
