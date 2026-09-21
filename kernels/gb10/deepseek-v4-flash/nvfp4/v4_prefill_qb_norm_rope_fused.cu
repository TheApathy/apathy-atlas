// SPDX-License-Identifier: AGPL-3.0-only

// DeepSeek-V4 prefill: fuse the pure per-Q-head q_b_norm with direct in-place
// interleaved YaRN RoPE, while one legal Q-head-0 warp rotates K.
//
// q_b_norm is unweighted in the V4 graph. Its incumbent call uses rms_norm's
// offset convention with norm_unit_w, a buffer allocated and permanently
// zero-filled by spark-runtime: (1 + weight) is therefore exactly one. Keeping
// the pointer and operand order here preserves that incumbent boundary.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define V4_QB_NORM_ROPE_NQ 64
#define V4_QB_NORM_ROPE_NKV 1
#define V4_QB_NORM_ROPE_HEAD_DIM 512
#define V4_QB_NORM_ROPE_NOPE_DIM 448
#define V4_QB_NORM_ROPE_DIM 64
#define V4_QB_NORM_ROPE_PAIRS (V4_QB_NORM_ROPE_DIM / 2)
#define V4_QB_NORM_ROPE_THREADS 512

static_assert(V4_QB_NORM_ROPE_NOPE_DIM + V4_QB_NORM_ROPE_DIM ==
                  V4_QB_NORM_ROPE_HEAD_DIM,
              "RoPE must occupy the exact trailing DeepSeek-V4 channels");
static_assert(V4_QB_NORM_ROPE_HEAD_DIM / 2 == 256,
              "one packed BF16 word per active normalization thread");
static_assert(V4_QB_NORM_ROPE_NOPE_DIM / 2 == 224,
              "the final active warp must own all Q RoPE pairs");
static_assert(V4_QB_NORM_ROPE_PAIRS == 32,
              "one Q-head-0 warp must own all K RoPE pairs");

__device__ __forceinline__ void v4_qb_unpack_bf16x2(
    const unsigned int packed, float& v0, float& v1) {
    v0 = __bfloat162float(__ushort_as_bfloat16(
        static_cast<unsigned short>(packed & 0xFFFFU)));
    v1 = __bfloat162float(__ushort_as_bfloat16(
        static_cast<unsigned short>(packed >> 16)));
}

__device__ __forceinline__ unsigned int v4_qb_pack_bf16x2(
    const __nv_bfloat16 v0, const __nv_bfloat16 v1) {
    const unsigned int lo = __bfloat16_as_ushort(v0);
    const unsigned int hi = __bfloat16_as_ushort(v1);
    return lo | (hi << 16);
}

__device__ __forceinline__ float v4_qb_warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ __forceinline__ unsigned int v4_qb_rotate_rounded_pair(
    const __nv_bfloat16 rounded0,
    const __nv_bfloat16 rounded1,
    const unsigned int abs_pos,
    const float freq,
    const float mscale) {
    // The widening happens only after the exact incumbent BF16 norm boundary.
    const float q0 = __bfloat162float(rounded0);
    const float q1 = __bfloat162float(rounded1);
    const float angle = static_cast<float>(abs_pos) * freq;
    const float cos_val = cosf(angle) * mscale;
    const float sin_val = sinf(angle) * mscale;
    const float y0 = q0 * cos_val - q1 * sin_val;
    const float y1 = q1 * cos_val + q0 * sin_val;
    return v4_qb_pack_bf16x2(__float2bfloat16(y0), __float2bfloat16(y1));
}

// Grid (num_tokens, 64, 1), block (512, 1, 1).
//
// Each CTA exactly replaces the incumbent 512-thread RMSNorm CTA for one Q
// head. Threads 0..255 preserve its packed input/reduction/output ownership and
// retain x0/x1 through the reduction, removing rms_norm's apply-pass Q reread.
// Threads 224..255 keep the BF16-rounded tail in registers and rotate it before
// its single final store, avoiding the direct-RoPE kernel's intermediate tail
// read+write. In Q head 0 only, threads 0..31 independently rotate K's tail;
// Q and K are disjoint production tensor regions, so there is one K owner and
// no cross-CTA ordering dependency.
extern "C" __global__ __launch_bounds__(512) void v4_prefill_qb_norm_rope_fused(
    __nv_bfloat16* __restrict__ Q,
    __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ weight,
    const unsigned int* __restrict__ positions,
    const float* __restrict__ inv_freq,
    const unsigned int num_tokens,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int nope_dim,
    const unsigned int rotary_dim,
    const float eps,
    const float mscale) {
    if (num_tokens == 0 || Q == nullptr || K == nullptr || weight == nullptr ||
        positions == nullptr || inv_freq == nullptr || Q == K ||
        num_q_heads != V4_QB_NORM_ROPE_NQ ||
        num_kv_heads != V4_QB_NORM_ROPE_NKV ||
        head_dim != V4_QB_NORM_ROPE_HEAD_DIM ||
        nope_dim != V4_QB_NORM_ROPE_NOPE_DIM ||
        rotary_dim != V4_QB_NORM_ROPE_DIM || !(eps > 0.0f) || !isfinite(eps) ||
        !isfinite(mscale) || blockDim.x != 512 || blockDim.y != 1 ||
        blockDim.z != 1 || gridDim.x != num_tokens ||
        gridDim.y != V4_QB_NORM_ROPE_NQ || gridDim.z != 1) {
        return;
    }

    const unsigned int token = blockIdx.x;
    const unsigned int head = blockIdx.y;
    const unsigned int tid = threadIdx.x;
    const unsigned long long q_offset =
        (static_cast<unsigned long long>(token) * V4_QB_NORM_ROPE_NQ + head) *
        V4_QB_NORM_ROPE_HEAD_DIM;
    __nv_bfloat16* const q = Q + q_offset;
    const unsigned int* const q32 = reinterpret_cast<const unsigned int*>(q);

    // Exact rms_norm H=512 lane ownership: one packed word in threads 0..255,
    // zero in 256..511, then the same 16-warp XOR reduction tree.
    float sum_sq = 0.0f;
    unsigned int q_packed = 0;
    float x0 = 0.0f;
    float x1 = 0.0f;
    if (tid < V4_QB_NORM_ROPE_HEAD_DIM / 2) {
        q_packed = q32[tid];
        v4_qb_unpack_bf16x2(q_packed, x0, x1);
        sum_sq += x0 * x0 + x1 * x1;
    }
    sum_sq = v4_qb_warp_reduce_sum(sum_sq);

    __shared__ float warp_sums[32];
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;
    if (lane_id == 0) {
        warp_sums[warp_id] = sum_sq;
    }
    __syncthreads();
    if (warp_id == 0) {
        float val = lane_id < (V4_QB_NORM_ROPE_THREADS + 31) / 32
                        ? warp_sums[lane_id]
                        : 0.0f;
        val = v4_qb_warp_reduce_sum(val);
        if (lane_id == 0) {
            warp_sums[0] = val;
        }
    }
    __syncthreads();

    float rms = rsqrtf(warp_sums[0] / (float)V4_QB_NORM_ROPE_HEAD_DIM + eps);
    if (tid < V4_QB_NORM_ROPE_HEAD_DIM / 2) {
        float w0;
        float w1;
        const unsigned int* const weight32 =
            reinterpret_cast<const unsigned int*>(weight);
        v4_qb_unpack_bf16x2(weight32[tid], w0, w1);
        const float normalized0 = x0 * rms * (1.0f + w0);
        const float normalized1 = x1 * rms * (1.0f + w1);
        const __nv_bfloat16 rounded0 = __float2bfloat16(normalized0);
        const __nv_bfloat16 rounded1 = __float2bfloat16(normalized1);
        unsigned int output = v4_qb_pack_bf16x2(rounded0, rounded1);
        if (tid >= V4_QB_NORM_ROPE_NOPE_DIM / 2) {
            const unsigned int pair_idx = tid - V4_QB_NORM_ROPE_NOPE_DIM / 2;
            output = v4_qb_rotate_rounded_pair(
                rounded0, rounded1, positions[token], inv_freq[pair_idx], mscale);
        }
        reinterpret_cast<unsigned int*>(q)[tid] = output;
    }

    // K is not normalized here. Exactly one CTA (Q head 0) and one warp own
    // its 32 trailing pairs for this token; the first 448 K channels are inert.
    if (head == 0 && tid < V4_QB_NORM_ROPE_PAIRS) {
        __nv_bfloat16* const k =
            K + static_cast<unsigned long long>(token) * V4_QB_NORM_ROPE_HEAD_DIM;
        const unsigned int k_packed =
            reinterpret_cast<unsigned int*>(k + V4_QB_NORM_ROPE_NOPE_DIM)[tid];
        const __nv_bfloat16 k0 = __ushort_as_bfloat16(
            static_cast<unsigned short>(k_packed & 0xFFFFU));
        const __nv_bfloat16 k1 = __ushort_as_bfloat16(
            static_cast<unsigned short>(k_packed >> 16));
        const unsigned int output = v4_qb_rotate_rounded_pair(
            k0, k1, positions[token], inv_freq[tid], mscale);
        reinterpret_cast<unsigned int*>(k + V4_QB_NORM_ROPE_NOPE_DIM)[tid] = output;
    }
}
