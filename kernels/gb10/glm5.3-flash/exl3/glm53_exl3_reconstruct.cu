// SPDX-License-Identifier: AGPL-3.0-only
// Device implementation core: ExLlamaV3 MIT (`quant/reconstruct.cu`,
// `reconstruct_had_kernel`), pinned and verified by build.rs.
//
// Fused reconstruction of one EXL3 trellis matrix into an ORIGINAL-basis F16
// weight `W[K, N]` (row-major, K rows of N): both Hadamard transforms and the
// `suh`/`svh` sign vectors are folded in, so a plain GEMM on the raw
// activation reproduces the trellis GEMM's `(x H) W_hat H` product without the
// per-16-row-tile trellis decode the cooperative kernel repeats at large M.
//
// Grid: [N/128, K/128], block 256. `packed_blocks_n = N/16`.

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <util.h>
#include <util.cuh>
#include <ptx.cuh>
#include <quant/exl3_dq.cuh>
#include <quant/hadamard_inner.cuh>

#define RH_THREADS 256

template <int K, int cb, bool OUT_BF16>
__device__ __forceinline__ void atlas_glm53_exl3_reconstruct_had_body
(
    half* __restrict__ g_unpacked,
    const uint16_t* __restrict__ g_packed,
    const half* __restrict__ suh,
    const half* __restrict__ svh,
    int packed_blocks_n,
    int packed_n_offset
)
{
    constexpr int packed_size = 256 * K / 16;
    constexpr float r_scale = 0.08838834764831845f;

    int t = threadIdx.x;
    int lane_id = t % 32;
    int warp_id = t / 32;
    int kb = blockIdx.y;
    int nb = blockIdx.x;
    int n = nb * 8;
    int row_len = gridDim.x * 128;

    __shared__ uint32_t s_packed[8][8][packed_size / 2];
    __shared__ half2 stile[128 * 64];

    auto tix = [&] (int R, int q, int p)
    {
        return R * 64 + (q ^ ((R >> 2) & 31)) * 2 + p;
    };

    constexpr int j_int4 = packed_size / 8;
    for (int u = t; u < 8 * 8 * j_int4; u += RH_THREADS)
    {
        int j = u / (8 * j_int4);
        int r = u % (8 * j_int4);
        const uint16_t* gp = g_packed +
            ((size_t) ((kb * 8 + j) * packed_blocks_n + packed_n_offset + n)) * packed_size;
        ((int4*) s_packed[j])[r] = ((const int4*) gp)[r];
    }
    __syncthreads();

    for (int jj = 0; jj < 8 * 8 / (RH_THREADS / 32); ++jj)
    {
        int j = (warp_id / 8) * (8 / (RH_THREADS / 256)) + jj;
        int wn = warp_id % 8;
        FragB frag[2];
        dq_dispatch<K, cb>(s_packed[j][wn], lane_id * 8, frag[0], frag[1]);

        half2 n0 = __shfl_down_sync(0xFFFFFFFF, frag[0][0], 4, 32);
        half2 n1 = __shfl_down_sync(0xFFFFFFFF, frag[0][1], 4, 32);
        half2 n2 = __shfl_down_sync(0xFFFFFFFF, frag[1][0], 4, 32);
        half2 n3 = __shfl_down_sync(0xFFFFFFFF, frag[1][1], 4, 32);

        if (!(lane_id & 4))
        {
            half2 m0 = __halves2half2(__low2half(frag[0][0]), __low2half(n0));
            half2 m1 = __halves2half2(__high2half(frag[0][0]), __high2half(n0));
            half2 m2 = __halves2half2(__low2half(frag[0][1]), __low2half(n1));
            half2 m3 = __halves2half2(__high2half(frag[0][1]), __high2half(n1));
            half2 m4 = __halves2half2(__low2half(frag[1][0]), __low2half(n2));
            half2 m5 = __halves2half2(__high2half(frag[1][0]), __high2half(n2));
            half2 m6 = __halves2half2(__low2half(frag[1][1]), __low2half(n3));
            half2 m7 = __halves2half2(__high2half(frag[1][1]), __high2half(n3));
            int r0 = j * 16 + (lane_id % 4) * 2;
            int r1 = r0 + 1;
            int r2 = r0 + 8;
            int r3 = r0 + 9;
            int c0 = lane_id / 8;
            int q0 = (wn * 8 + c0) >> 1, p0 = c0 & 1;
            int q1 = (wn * 8 + c0 + 4) >> 1, p1 = c0 & 1;
            stile[tix(r0, q0, p0)] = m0;
            stile[tix(r1, q0, p0)] = m1;
            stile[tix(r2, q0, p0)] = m2;
            stile[tix(r3, q0, p0)] = m3;
            stile[tix(r0, q1, p1)] = m4;
            stile[tix(r1, q1, p1)] = m5;
            stile[tix(r2, q1, p1)] = m6;
            stile[tix(r3, q1, p1)] = m7;
        }
    }
    __syncthreads();

    const half2 rs2 = __float2half2_rn(r_scale);
    constexpr int CHUNKS_PW = 32 / (RH_THREADS / 32);
    #pragma unroll
    for (int qq = 0; qq < CHUNKS_PW; ++qq)
    {
        int q = warp_id * CHUNKS_PW + qq;
        int qs = q ^ lane_id;
        half2 a[4], b[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            half4 v = *((const half4*) (stile + (lane_id * 4 + i) * 64 + qs * 2));
            a[i] = v.x;
            b[i] = v.y;
        }
        #pragma unroll
        for (int x = 0; x < 2; ++x)
        {
            half2* v = x == 0 ? a : b;
            half2 s0 = __hadd2(v[0], v[1]), d0 = __hsub2(v[0], v[1]);
            half2 s1 = __hadd2(v[2], v[3]), d1 = __hsub2(v[2], v[3]);
            v[0] = __hmul2(__hadd2(s0, s1), rs2);
            v[1] = __hmul2(__hadd2(d0, d1), rs2);
            v[2] = __hmul2(__hsub2(s0, s1), rs2);
            v[3] = __hmul2(__hsub2(d0, d1), rs2);
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                v[i] = shuffle_had_h2x32(v[i], lane_id);
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            half4 v;
            v.x = a[i];
            v.y = b[i];
            *((half4*) (stile + (lane_id * 4 + i) * 64 + qs * 2)) = v;
        }
    }
    __syncthreads();

    constexpr int ROWS_PW = 128 / (RH_THREADS / 32);
    const half4 sv4 = ((const half4*) svh)[nb * 32 + lane_id];
    #pragma unroll
    for (int rr = 0; rr < ROWS_PW; ++rr)
    {
        int R = warp_id * ROWS_PW + rr;
        int base = R * 64 + (lane_id ^ ((R >> 2) & 31)) * 2;
        half2 v01 = stile[base];
        half2 v23 = stile[base + 1];
        float v0 = __low2float(v01), v1 = __high2float(v01);
        float v2 = __low2float(v23), v3 = __high2float(v23);
        float s0 = v0 + v1, d0 = v0 - v1;
        float s1 = v2 + v3, d1 = v2 - v3;
        half2 h01 = __hmul2(__floats2half2_rn(s0 + s1, d0 + d1), rs2);
        half2 h23 = __hmul2(__floats2half2_rn(s0 - s1, d0 - d1), rs2);
        h01 = shuffle_had_h2x32(h01, lane_id);
        h23 = shuffle_had_h2x32(h23, lane_id);
        half2 su2 = __half2half2(suh[kb * 128 + R]);
        half4 v;
        v.x = __hmul2(__hmul2(h01, su2), sv4.x);
        v.y = __hmul2(__hmul2(h23, su2), sv4.y);
        const size_t out_index = (size_t) (kb * 128 + R) * row_len + nb * 128 + lane_id * 4;
        if constexpr (OUT_BF16)
        {
            // Same F16 value path; only the final store rounds to BF16.
            __nv_bfloat162 b0 = __float22bfloat162_rn(__half22float2(v.x));
            __nv_bfloat162 b1 = __float22bfloat162_rn(__half22float2(v.y));
            __nv_bfloat162* out = (__nv_bfloat162*) (((__nv_bfloat16*) g_unpacked) + out_index);
            out[0] = b0;
            out[1] = b1;
        }
        else
        {
            *((half4*) (g_unpacked + out_index)) = v;
        }
    }
}


// f32-butterfly variant: identical dequant and data movement, but both Hadamard
// transforms and the sign/scale products run in f32 and the ONLY rounding is
// the final store (F16 or BF16). Tile is float2 [128 x 64] = 64 KB, so shared
// memory is dynamic: bytes = 8*8*packed_size*2 (packed) + 128*128*4 (tile).
template <int K, int cb, bool OUT_BF16>
__device__ __forceinline__ void atlas_glm53_exl3_reconstruct_had_body_f32
(
    half* __restrict__ g_unpacked,
    const uint16_t* __restrict__ g_packed,
    const half* __restrict__ suh,
    const half* __restrict__ svh,
    int packed_blocks_n,
    int packed_n_offset
)
{
    constexpr int packed_size = 256 * K / 16;
    constexpr float r_scale = 0.08838834764831845f;
    constexpr int packed_bytes = 8 * 8 * packed_size * 2;

    int t = threadIdx.x;
    int lane_id = t % 32;
    int warp_id = t / 32;
    int kb = blockIdx.y;
    int nb = blockIdx.x;
    int n = nb * 8;
    int row_len = gridDim.x * 128;

    extern __shared__ __align__(16) unsigned char dyn_smem[];
    uint32_t (*s_packed)[8][packed_size / 2] = (uint32_t (*)[8][packed_size / 2]) dyn_smem;
    float2* stile = (float2*) (dyn_smem + packed_bytes);

    auto tix = [&] (int R, int q, int p)
    {
        return R * 64 + (q ^ ((R >> 2) & 31)) * 2 + p;
    };

    constexpr int j_int4 = packed_size / 8;
    for (int u = t; u < 8 * 8 * j_int4; u += RH_THREADS)
    {
        int j = u / (8 * j_int4);
        int r = u % (8 * j_int4);
        const uint16_t* gp = g_packed +
            ((size_t) ((kb * 8 + j) * packed_blocks_n + packed_n_offset + n)) * packed_size;
        ((int4*) s_packed[j])[r] = ((const int4*) gp)[r];
    }
    __syncthreads();

    for (int jj = 0; jj < 8 * 8 / (RH_THREADS / 32); ++jj)
    {
        int j = (warp_id / 8) * (8 / (RH_THREADS / 256)) + jj;
        int wn = warp_id % 8;
        FragB frag[2];
        dq_dispatch<K, cb>(s_packed[j][wn], lane_id * 8, frag[0], frag[1]);

        half2 n0 = __shfl_down_sync(0xFFFFFFFF, frag[0][0], 4, 32);
        half2 n1 = __shfl_down_sync(0xFFFFFFFF, frag[0][1], 4, 32);
        half2 n2 = __shfl_down_sync(0xFFFFFFFF, frag[1][0], 4, 32);
        half2 n3 = __shfl_down_sync(0xFFFFFFFF, frag[1][1], 4, 32);

        if (!(lane_id & 4))
        {
            half2 m0 = __halves2half2(__low2half(frag[0][0]), __low2half(n0));
            half2 m1 = __halves2half2(__high2half(frag[0][0]), __high2half(n0));
            half2 m2 = __halves2half2(__low2half(frag[0][1]), __low2half(n1));
            half2 m3 = __halves2half2(__high2half(frag[0][1]), __high2half(n1));
            half2 m4 = __halves2half2(__low2half(frag[1][0]), __low2half(n2));
            half2 m5 = __halves2half2(__high2half(frag[1][0]), __high2half(n2));
            half2 m6 = __halves2half2(__low2half(frag[1][1]), __low2half(n3));
            half2 m7 = __halves2half2(__high2half(frag[1][1]), __high2half(n3));
            int r0 = j * 16 + (lane_id % 4) * 2;
            int r1 = r0 + 1;
            int r2 = r0 + 8;
            int r3 = r0 + 9;
            int c0 = lane_id / 8;
            int q0 = (wn * 8 + c0) >> 1, p0 = c0 & 1;
            int q1 = (wn * 8 + c0 + 4) >> 1, p1 = c0 & 1;
            stile[tix(r0, q0, p0)] = __half22float2(m0);
            stile[tix(r1, q0, p0)] = __half22float2(m1);
            stile[tix(r2, q0, p0)] = __half22float2(m2);
            stile[tix(r3, q0, p0)] = __half22float2(m3);
            stile[tix(r0, q1, p1)] = __half22float2(m4);
            stile[tix(r1, q1, p1)] = __half22float2(m5);
            stile[tix(r2, q1, p1)] = __half22float2(m6);
            stile[tix(r3, q1, p1)] = __half22float2(m7);
        }
    }
    __syncthreads();

    constexpr int CHUNKS_PW = 32 / (RH_THREADS / 32);
    #pragma unroll
    for (int qq = 0; qq < CHUNKS_PW; ++qq)
    {
        int q = warp_id * CHUNKS_PW + qq;
        int qs = q ^ lane_id;
        float2 a[4], b[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            float4 v = *((const float4*) (stile + (lane_id * 4 + i) * 64 + qs * 2));
            a[i] = make_float2(v.x, v.y);
            b[i] = make_float2(v.z, v.w);
        }
        #pragma unroll
        for (int x = 0; x < 2; ++x)
        {
            float2* v = x == 0 ? a : b;
            float2 s0 = make_float2(v[0].x + v[1].x, v[0].y + v[1].y);
            float2 d0 = make_float2(v[0].x - v[1].x, v[0].y - v[1].y);
            float2 s1 = make_float2(v[2].x + v[3].x, v[2].y + v[3].y);
            float2 d1 = make_float2(v[2].x - v[3].x, v[2].y - v[3].y);
            v[0] = make_float2((s0.x + s1.x) * r_scale, (s0.y + s1.y) * r_scale);
            v[1] = make_float2((d0.x + d1.x) * r_scale, (d0.y + d1.y) * r_scale);
            v[2] = make_float2((s0.x - s1.x) * r_scale, (s0.y - s1.y) * r_scale);
            v[3] = make_float2((d0.x - d1.x) * r_scale, (d0.y - d1.y) * r_scale);
            #pragma unroll
            for (int i = 0; i < 4; ++i)
                shuffle_had_f2x32(v[i].x, v[i].y, lane_id);
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i)
        {
            float4 v = make_float4(a[i].x, a[i].y, b[i].x, b[i].y);
            *((float4*) (stile + (lane_id * 4 + i) * 64 + qs * 2)) = v;
        }
    }
    __syncthreads();

    constexpr int ROWS_PW = 128 / (RH_THREADS / 32);
    const half4 sv4h = ((const half4*) svh)[nb * 32 + lane_id];
    const float2 sv01 = __half22float2(sv4h.x);
    const float2 sv23 = __half22float2(sv4h.y);
    #pragma unroll
    for (int rr = 0; rr < ROWS_PW; ++rr)
    {
        int R = warp_id * ROWS_PW + rr;
        int base = R * 64 + (lane_id ^ ((R >> 2) & 31)) * 2;
        float2 v01 = stile[base];
        float2 v23 = stile[base + 1];
        float s0 = v01.x + v01.y, d0 = v01.x - v01.y;
        float s1 = v23.x + v23.y, d1 = v23.x - v23.y;
        float h0 = (s0 + s1) * r_scale, h1 = (d0 + d1) * r_scale;
        float h2 = (s0 - s1) * r_scale, h3 = (d0 - d1) * r_scale;
        shuffle_had_f2x32(h0, h1, lane_id);
        shuffle_had_f2x32(h2, h3, lane_id);
        const float su = __half2float(suh[kb * 128 + R]);
        const float o0 = h0 * su * sv01.x;
        const float o1 = h1 * su * sv01.y;
        const float o2 = h2 * su * sv23.x;
        const float o3 = h3 * su * sv23.y;
        const size_t out_index = (size_t) (kb * 128 + R) * row_len + nb * 128 + lane_id * 4;
        if constexpr (OUT_BF16)
        {
            __nv_bfloat162* out = (__nv_bfloat162*) (((__nv_bfloat16*) g_unpacked) + out_index);
            out[0] = __floats2bfloat162_rn(o0, o1);
            out[1] = __floats2bfloat162_rn(o2, o3);
        }
        else
        {
            half4 v;
            v.x = __floats2half2_rn(o0, o1);
            v.y = __floats2half2_rn(o2, o3);
            *((half4*) (g_unpacked + out_index)) = v;
        }
    }
}

#define ATLAS_GLM53_RECONSTRUCT(K) \
    extern "C" __global__ __launch_bounds__(RH_THREADS) \
    void atlas_glm53_exl3_reconstruct_had_k##K##_cb2 \
    ( \
        half* __restrict__ g_unpacked, \
        const uint16_t* __restrict__ g_packed, \
        const half* __restrict__ suh, \
        const half* __restrict__ svh, \
        int packed_blocks_n, \
        int packed_n_offset \
    ) \
    { \
        atlas_glm53_exl3_reconstruct_had_body<K, 2, false> \
            (g_unpacked, g_packed, suh, svh, packed_blocks_n, packed_n_offset); \
    } \
    extern "C" __global__ __launch_bounds__(RH_THREADS) \
    void atlas_glm53_exl3_reconstruct_had_k##K##_cb2_bf16 \
    ( \
        half* __restrict__ g_unpacked, \
        const uint16_t* __restrict__ g_packed, \
        const half* __restrict__ suh, \
        const half* __restrict__ svh, \
        int packed_blocks_n, \
        int packed_n_offset \
    ) \
    { \
        atlas_glm53_exl3_reconstruct_had_body<K, 2, true> \
            (g_unpacked, g_packed, suh, svh, packed_blocks_n, packed_n_offset); \
    } \
    extern "C" __global__ __launch_bounds__(RH_THREADS) \
    void atlas_glm53_exl3_reconstruct_had_k##K##_cb2_f32 \
    ( \
        half* __restrict__ g_unpacked, \
        const uint16_t* __restrict__ g_packed, \
        const half* __restrict__ suh, \
        const half* __restrict__ svh, \
        int packed_blocks_n, \
        int packed_n_offset \
    ) \
    { \
        atlas_glm53_exl3_reconstruct_had_body_f32<K, 2, false> \
            (g_unpacked, g_packed, suh, svh, packed_blocks_n, packed_n_offset); \
    } \
    extern "C" __global__ __launch_bounds__(RH_THREADS) \
    void atlas_glm53_exl3_reconstruct_had_k##K##_cb2_f32_bf16 \
    ( \
        half* __restrict__ g_unpacked, \
        const uint16_t* __restrict__ g_packed, \
        const half* __restrict__ suh, \
        const half* __restrict__ svh, \
        int packed_blocks_n, \
        int packed_n_offset \
    ) \
    { \
        atlas_glm53_exl3_reconstruct_had_body_f32<K, 2, true> \
            (g_unpacked, g_packed, suh, svh, packed_blocks_n, packed_n_offset); \
    }

ATLAS_GLM53_RECONSTRUCT(2)
ATLAS_GLM53_RECONSTRUCT(3)
ATLAS_GLM53_RECONSTRUCT(4)
ATLAS_GLM53_RECONSTRUCT(5)
