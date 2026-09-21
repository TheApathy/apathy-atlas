// SPDX-License-Identifier: AGPL-3.0-only

// Exact Qwen3.8 C=1 prefill fusion:
//   deinterleave Q/Gate + per-head Q RMSNorm
//   per-head K RMSNorm
//   interleaved multi-modal RoPE for Q and K
//
// Grid: (num_tokens, 1, 1), block: (256, 1, 1)
// Dynamic shared: num_q_heads * head_dim * 2 * sizeof(BF16) = 24,576 B
// for Q24/HD256. After Q is fully consumed, its storage is reused for one
// 256-element K head plus eight FP32 block-reduction partials.

#include <cuda_bf16.h>

__device__ __forceinline__ float q38_q_warp_sum(float value) {
    value = __shfl_xor_sync(0xFFFFFFFF, value, 16) + value;
    value = __shfl_xor_sync(0xFFFFFFFF, value, 8) + value;
    value = __shfl_xor_sync(0xFFFFFFFF, value, 4) + value;
    value = __shfl_xor_sync(0xFFFFFFFF, value, 2) + value;
    value = __shfl_xor_sync(0xFFFFFFFF, value, 1) + value;
    return value;
}

__device__ __forceinline__ float q38_k_warp_sum(float value) {
    value += __shfl_xor_sync(0xFFFFFFFF, value, 16);
    value += __shfl_xor_sync(0xFFFFFFFF, value, 8);
    value += __shfl_xor_sync(0xFFFFFFFF, value, 4);
    value += __shfl_xor_sync(0xFFFFFFFF, value, 2);
    value += __shfl_xor_sync(0xFFFFFFFF, value, 1);
    return value;
}

__device__ __forceinline__ void q38_qknorm_unpack_bf16x2(
    unsigned int packed, float& v0, float& v1
) {
    v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xFFFF)));
    v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed >> 16)));
}

__device__ __forceinline__ unsigned int q38_qknorm_pack_bf16x2(
    float v0, float v1
) {
    unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0));
    unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1));
    return lo | (hi << 16);
}

__device__ __forceinline__ unsigned int q38_mrope_position(
    unsigned int pair_idx,
    unsigned int token,
    const unsigned int* __restrict__ pos_t,
    const unsigned int* __restrict__ pos_h,
    const unsigned int* __restrict__ pos_w
) {
    const unsigned int section = pair_idx % 3;
    if (section == 0) return pos_t[token];
    if (section == 1) return pos_h[token];
    return pos_w[token];
}

__device__ __forceinline__ void q38_mrope_rotate_pair(
    const __nv_bfloat16* __restrict__ source,
    __nv_bfloat16* __restrict__ output,
    unsigned int pair_idx,
    unsigned int rotary_dim,
    unsigned int abs_pos,
    float theta
) {
    const double freq_exp_d = (double)(2 * pair_idx) / (double)rotary_dim;
    const float freq = (float)(1.0 / pow((double)theta, freq_exp_d));
    const float angle = (float)abs_pos * freq;
    const float cos_val = cosf(angle);
    const float sin_val = sinf(angle);
    const unsigned int half_rot = rotary_dim / 2;
    const unsigned int d0 = pair_idx;
    const unsigned int d1 = pair_idx + half_rot;
    const float x0 = (float)source[d0];
    const float x1 = (float)source[d1];
    const float y0 = x0 * cos_val - x1 * sin_val;
    const float y1 = x1 * cos_val + x0 * sin_val;
    output[d0] = __float2bfloat16(y0);
    output[d1] = __float2bfloat16(y1);
}

extern "C" __global__ __launch_bounds__(256, 2) void qwen38_prefill_qknorm_rope(
    __nv_bfloat16* __restrict__ qg_data,
    __nv_bfloat16* __restrict__ q_out,
    __nv_bfloat16* __restrict__ k_data,
    const __nv_bfloat16* __restrict__ q_norm_weight,
    const __nv_bfloat16* __restrict__ k_norm_weight,
    const unsigned int* __restrict__ pos_t,
    const unsigned int* __restrict__ pos_h,
    const unsigned int* __restrict__ pos_w,
    const unsigned int num_tokens,
    const unsigned int num_q_heads,
    const unsigned int num_kv_heads,
    const unsigned int head_dim,
    const unsigned int qg_stride,
    const unsigned int rotary_dim,
    const float eps,
    const float theta
) {
    const unsigned int token = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    if (token >= num_tokens) return;

    extern __shared__ __align__(16) unsigned char smem_raw[];
    __nv_bfloat16* smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    const unsigned int q_total = num_q_heads * head_dim;
    const unsigned int qg_total = q_total * 2;
    const unsigned int group_dim = head_dim * 2;
    __nv_bfloat16* token_qg = qg_data + (unsigned long long)token * qg_stride;
    __nv_bfloat16* token_q = q_out + (unsigned long long)token * q_total;

    // Identical load and Gate destination to deinterleave_qg_split_qnorm.
    for (unsigned int i = tid; i < qg_total; i += blockDim.x) {
        smem[i] = token_qg[i];
    }
    __syncthreads();
    for (unsigned int i = tid; i < q_total; i += blockDim.x) {
        const unsigned int head = i / head_dim;
        const unsigned int dim = i % head_dim;
        token_qg[q_total + i] = smem[head * group_dim + head_dim + dim];
    }

    // Identical per-head/lane accumulation and shuffle order to the parent Q
    // fusion. Store the normalized BF16 boundary back to shared for RoPE.
    const unsigned int warp_id = tid / 32;
    const unsigned int lane = tid % 32;
    const unsigned int num_warps = blockDim.x / 32;
    const unsigned int elems_per_thread = head_dim / 32;
    for (unsigned int head = warp_id; head < num_q_heads; head += num_warps) {
        float sum_sq = 0.0f;
        for (unsigned int e = 0; e < elems_per_thread; e++) {
            const unsigned int dim = lane + e * 32;
            const unsigned int source = head * group_dim + dim;
            const float value = __bfloat162float(smem[source]);
            sum_sq += value * value;
        }
        sum_sq = q38_q_warp_sum(sum_sq);
        const float rms = rsqrtf(sum_sq / (float)head_dim + eps);
        for (unsigned int e = 0; e < elems_per_thread; e++) {
            const unsigned int dim = lane + e * 32;
            const unsigned int source = head * group_dim + dim;
            const float value = __bfloat162float(smem[source]);
            const float weight = __bfloat162float(q_norm_weight[dim]);
            smem[source] = __float2bfloat16(value * rms * (1.0f + weight));
        }
    }
    __syncthreads();

    // Preserve the normalized BF16 values outside the rotary prefix.
    const unsigned int q_tail = num_q_heads * (head_dim - rotary_dim);
    for (unsigned int i = tid; i < q_tail; i += blockDim.x) {
        const unsigned int head = i / (head_dim - rotary_dim);
        const unsigned int dim = rotary_dim + i % (head_dim - rotary_dim);
        token_q[head * head_dim + dim] = smem[head * group_dim + dim];
    }
    const unsigned int pairs = rotary_dim / 2;
    const unsigned int q_pairs = num_q_heads * pairs;
    for (unsigned int i = tid; i < q_pairs; i += blockDim.x) {
        const unsigned int head = i / pairs;
        const unsigned int pair = i % pairs;
        const unsigned int abs_pos = q38_mrope_position(pair, token, pos_t, pos_h, pos_w);
        q38_mrope_rotate_pair(
            smem + head * group_dim,
            token_q + head * head_dim,
            pair,
            rotary_dim,
            abs_pos,
            theta
        );
    }
    __syncthreads();

    // The Q tile is dead. Reuse its first HD BF16 values for one K head and
    // the next 8 floats for the exact rms_norm block-reduction partials.
    float* k_warp_sums = reinterpret_cast<float*>(smem + head_dim);
    for (unsigned int head = 0; head < num_kv_heads; head++) {
        __nv_bfloat16* token_k = k_data
            + (unsigned long long)token * num_kv_heads * head_dim
            + head * head_dim;
        for (unsigned int dim = tid; dim < head_dim; dim += blockDim.x) {
            smem[dim] = token_k[dim];
        }
        __syncthreads();

        float sum_sq = 0.0f;
        const unsigned int half_size = head_dim / 2;
        if (tid < half_size) {
            float v0, v1;
            q38_qknorm_unpack_bf16x2(
                reinterpret_cast<const unsigned int*>(smem)[tid], v0, v1
            );
            sum_sq += v0 * v0 + v1 * v1;
        }
        sum_sq = q38_k_warp_sum(sum_sq);
        if (lane == 0) k_warp_sums[warp_id] = sum_sq;
        __syncthreads();
        if (warp_id == 0) {
            float value = lane < 8 ? k_warp_sums[lane] : 0.0f;
            value = q38_k_warp_sum(value);
            if (lane == 0) k_warp_sums[0] = value;
        }
        __syncthreads();
        const float rms = rsqrtf(k_warp_sums[0] / (float)head_dim + eps);
        if (tid < half_size) {
            float x0, x1, w0, w1;
            q38_qknorm_unpack_bf16x2(
                reinterpret_cast<const unsigned int*>(smem)[tid], x0, x1
            );
            q38_qknorm_unpack_bf16x2(
                reinterpret_cast<const unsigned int*>(k_norm_weight)[tid], w0, w1
            );
            reinterpret_cast<unsigned int*>(smem)[tid] = q38_qknorm_pack_bf16x2(
                x0 * rms * (1.0f + w0), x1 * rms * (1.0f + w1)
            );
        }
        __syncthreads();

        for (unsigned int dim = rotary_dim + tid; dim < head_dim; dim += blockDim.x) {
            token_k[dim] = smem[dim];
        }
        if (tid < pairs) {
            const unsigned int abs_pos = q38_mrope_position(tid, token, pos_t, pos_h, pos_w);
            q38_mrope_rotate_pair(
                smem, token_k, tid, rotary_dim, abs_pos, theta
            );
        }
        __syncthreads();
    }
}
