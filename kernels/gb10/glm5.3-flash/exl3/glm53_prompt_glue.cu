// SPDX-License-Identifier: AGPL-3.0-only
// Prompt-scope glue for the GLM-5.3 layer-major prefill (private tree):
//  * bf16 -> f32 cast (router logits GEMM input)
//  * `atlas_glm53_hc_pre_mixed`: `atlas_glm53_hc_pre` with the 24 per-token
//    mixing dots hoisted into a cuBLASLt f32 GEMM. Everything after the mix
//    (inverse RMS, sigmoid pre/post, Sinkhorn-20, collapse) is the same code.
#include <cuda_bf16.h>
#include <float.h>
#include <math.h>
#define GLM53_HIDDEN 4096U
#define GLM53_HC 4U
#define GLM53_MIX 24U
#define GLM53_THREADS 256U

__device__ __forceinline__ float glm53_block_sum(float *values, unsigned int tid) {
    for (unsigned int stride = GLM53_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            values[tid] += values[tid + stride];
        }
        __syncthreads();
    }
    return values[0];
}

extern "C" __global__ void __launch_bounds__(256)
atlas_glm53_prompt_bf16_to_f32(const __nv_bfloat16 * __restrict__ input,
                               float * __restrict__ output,
                               unsigned int elements) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < elements) {
        output[index] = __bfloat162float(input[index]);
    }
}

extern "C" __global__ void __launch_bounds__(GLM53_THREADS, 4)
atlas_glm53_hc_pre_mixed(
        const float * __restrict__ streams,
        const float * __restrict__ mixed_raw,
        const float * __restrict__ scale,
        const float * __restrict__ base,
        __nv_bfloat16 * __restrict__ collapsed,
        __nv_bfloat16 * __restrict__ post,
        __nv_bfloat16 * __restrict__ combination,
        unsigned int hidden_size, unsigned int hc,
        unsigned int sinkhorn_iters, float norm_eps, float hc_eps) {
    if (hidden_size != GLM53_HIDDEN || hc != GLM53_HC ||
        sinkhorn_iters != 20U || norm_eps != 1.0e-5f || hc_eps != 1.0e-6f) {
        return;
    }
    const unsigned int token = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int flat_size = hc * hidden_size;
    const unsigned long long stream_base = (unsigned long long) token * flat_size;

    __shared__ float reduction[GLM53_THREADS];
    __shared__ float inverse_rms;
    __shared__ float mixed[GLM53_MIX];
    __shared__ float pre[GLM53_HC];
    __shared__ float sinkhorn[GLM53_HC * GLM53_HC];

    float square_sum = 0.0f;
    for (unsigned int index = tid; index < flat_size; index += GLM53_THREADS) {
        const float value = streams[stream_base + index];
        square_sum += value * value;
    }
    reduction[tid] = square_sum;
    __syncthreads();
    const float total = glm53_block_sum(reduction, tid);
    if (tid == 0) {
        inverse_rms = rsqrtf(total / (float) flat_size + norm_eps);
    }
    __syncthreads();

    // The 24 mixing dots were hoisted into one f32 GEMM over all tokens
    // (`mixed_raw[token, 24]`); only the per-token inverse-RMS scale remains.
    if (tid < GLM53_MIX) {
        mixed[tid] = mixed_raw[(unsigned long long) token * GLM53_MIX + tid] * inverse_rms;
    }
    __syncthreads();

    if (tid == 0) {
        #pragma unroll
        for (unsigned int index = 0; index < GLM53_HC; ++index) {
            const float pre_logit = mixed[index] * scale[0] + base[index];
            pre[index] = 1.0f / (1.0f + expf(-pre_logit)) + hc_eps;
            const float post_logit =
                mixed[GLM53_HC + index] * scale[1] + base[GLM53_HC + index];
            post[(unsigned long long) token * hc + index] =
                __float2bfloat16(2.0f / (1.0f + expf(-post_logit)));
        }
        #pragma unroll
        for (unsigned int row = 0; row < GLM53_HC; ++row) {
            float maximum = -FLT_MAX;
            #pragma unroll
            for (unsigned int column = 0; column < GLM53_HC; ++column) {
                const unsigned int index = row * GLM53_HC + column;
                sinkhorn[index] = mixed[2 * GLM53_HC + index] * scale[2] +
                    base[2 * GLM53_HC + index];
                maximum = fmaxf(maximum, sinkhorn[index]);
            }
            float sum = 0.0f;
            #pragma unroll
            for (unsigned int column = 0; column < GLM53_HC; ++column) {
                const unsigned int index = row * GLM53_HC + column;
                sinkhorn[index] = expf(sinkhorn[index] - maximum);
                sum += sinkhorn[index];
            }
            #pragma unroll
            for (unsigned int column = 0; column < GLM53_HC; ++column) {
                const unsigned int index = row * GLM53_HC + column;
                sinkhorn[index] = sinkhorn[index] / sum + hc_eps;
            }
        }
        #pragma unroll
        for (unsigned int column = 0; column < GLM53_HC; ++column) {
            float sum = hc_eps;
            #pragma unroll
            for (unsigned int row = 0; row < GLM53_HC; ++row) {
                sum += sinkhorn[row * GLM53_HC + column];
            }
            #pragma unroll
            for (unsigned int row = 0; row < GLM53_HC; ++row) {
                sinkhorn[row * GLM53_HC + column] /= sum;
            }
        }
        for (unsigned int iteration = 1; iteration < sinkhorn_iters; ++iteration) {
            #pragma unroll
            for (unsigned int row = 0; row < GLM53_HC; ++row) {
                float sum = hc_eps;
                #pragma unroll
                for (unsigned int column = 0; column < GLM53_HC; ++column) {
                    sum += sinkhorn[row * GLM53_HC + column];
                }
                #pragma unroll
                for (unsigned int column = 0; column < GLM53_HC; ++column) {
                    sinkhorn[row * GLM53_HC + column] /= sum;
                }
            }
            #pragma unroll
            for (unsigned int column = 0; column < GLM53_HC; ++column) {
                float sum = hc_eps;
                #pragma unroll
                for (unsigned int row = 0; row < GLM53_HC; ++row) {
                    sum += sinkhorn[row * GLM53_HC + column];
                }
                #pragma unroll
                for (unsigned int row = 0; row < GLM53_HC; ++row) {
                    sinkhorn[row * GLM53_HC + column] /= sum;
                }
            }
        }
        #pragma unroll
        for (unsigned int index = 0; index < GLM53_HC * GLM53_HC; ++index) {
            combination[(unsigned long long) token * GLM53_HC * GLM53_HC + index] =
                __float2bfloat16(sinkhorn[index]);
        }
    }
    __syncthreads();

    for (unsigned int column = tid; column < hidden_size; column += GLM53_THREADS) {
        float sum = 0.0f;
        #pragma unroll
        for (unsigned int stream = 0; stream < GLM53_HC; ++stream) {
            sum = fmaf(
                pre[stream],
                streams[stream_base + (unsigned long long) stream * hidden_size + column],
                sum);
        }
        collapsed[(unsigned long long) token * hidden_size + column] =
            __float2bfloat16(sum);
    }
}


// Prompt-scope mHC mixing: mixed_raw[token][24] = <function[j], streams[token]> and
// inv_rms[token] = rsqrt(mean(streams^2) + eps), four tokens per block so the
// 1.5 MB `function` matrix is streamed from L2 once per four tokens.
#define GLM53_MIX_TOKENS 4U
extern "C" __global__ void __launch_bounds__(GLM53_THREADS, 2)
atlas_glm53_hc_mix_rms(
        const float * __restrict__ streams,
        const float * __restrict__ function,
        float * __restrict__ mixed_raw,
        float * __restrict__ inv_rms,
        unsigned int tokens, unsigned int hidden_size, unsigned int hc, float norm_eps) {
    if (hidden_size != GLM53_HIDDEN || hc != GLM53_HC) return;
    const unsigned int flat_size = hc * hidden_size;
    const unsigned int token0 = blockIdx.x * GLM53_MIX_TOKENS;
    if (token0 >= tokens) return;
    const unsigned int tid = threadIdx.x;
    const unsigned int lane = tid & 31U;
    const unsigned int warp = tid >> 5U;
    const unsigned int live = min(GLM53_MIX_TOKENS, tokens - token0);
    float acc[GLM53_MIX_TOKENS][GLM53_MIX];
    float sq[GLM53_MIX_TOKENS];
    #pragma unroll
    for (unsigned int t = 0; t < GLM53_MIX_TOKENS; ++t) {
        sq[t] = 0.0f;
        #pragma unroll
        for (unsigned int j = 0; j < GLM53_MIX; ++j) acc[t][j] = 0.0f;
    }
    const float * s_base = streams + (unsigned long long) token0 * flat_size;
    for (unsigned int index = tid; index < flat_size; index += GLM53_THREADS) {
        float sv[GLM53_MIX_TOKENS];
        #pragma unroll
        for (unsigned int t = 0; t < GLM53_MIX_TOKENS; ++t) {
            sv[t] = t < live ? s_base[(unsigned long long) t * flat_size + index] : 0.0f;
            sq[t] = fmaf(sv[t], sv[t], sq[t]);
        }
        #pragma unroll
        for (unsigned int j = 0; j < GLM53_MIX; ++j) {
            const float f = function[(unsigned long long) j * flat_size + index];
            #pragma unroll
            for (unsigned int t = 0; t < GLM53_MIX_TOKENS; ++t) acc[t][j] = fmaf(f, sv[t], acc[t][j]);
        }
    }
    __shared__ float partial[8][GLM53_MIX_TOKENS * (GLM53_MIX + 1)];
    #pragma unroll
    for (unsigned int t = 0; t < GLM53_MIX_TOKENS; ++t) {
        #pragma unroll
        for (unsigned int j = 0; j < GLM53_MIX; ++j) {
            float v = acc[t][j];
            #pragma unroll
            for (unsigned int o = 16U; o > 0U; o >>= 1U) v += __shfl_down_sync(0xffffffffU, v, o);
            if (lane == 0U) partial[warp][t * (GLM53_MIX + 1) + j] = v;
        }
        float v = sq[t];
        #pragma unroll
        for (unsigned int o = 16U; o > 0U; o >>= 1U) v += __shfl_down_sync(0xffffffffU, v, o);
        if (lane == 0U) partial[warp][t * (GLM53_MIX + 1) + GLM53_MIX] = v;
    }
    __syncthreads();
    if (tid < GLM53_MIX_TOKENS * (GLM53_MIX + 1)) {
        const unsigned int t = tid / (GLM53_MIX + 1);
        const unsigned int j = tid % (GLM53_MIX + 1);
        if (t < live) {
            float sum = 0.0f;
            #pragma unroll
            for (unsigned int w = 0; w < 8U; ++w) sum += partial[w][tid];
            if (j < GLM53_MIX) {
                mixed_raw[(unsigned long long) (token0 + t) * GLM53_MIX + j] = sum;
            } else {
                inv_rms[token0 + t] = rsqrtf(sum / (float) flat_size + norm_eps);
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(GLM53_THREADS, 4)
atlas_glm53_hc_pre_mixed2(
        const float * __restrict__ streams,
        const float * __restrict__ mixed_raw,
        const float * __restrict__ inv_rms_in,
        const float * __restrict__ scale,
        const float * __restrict__ base,
        __nv_bfloat16 * __restrict__ collapsed,
        __nv_bfloat16 * __restrict__ post,
        __nv_bfloat16 * __restrict__ combination,
        unsigned int hidden_size, unsigned int hc,
        unsigned int sinkhorn_iters, float norm_eps, float hc_eps) {
    if (hidden_size != GLM53_HIDDEN || hc != GLM53_HC ||
        sinkhorn_iters != 20U || norm_eps != 1.0e-5f || hc_eps != 1.0e-6f) {
        return;
    }
    const unsigned int token = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int flat_size = hc * hidden_size;
    const unsigned long long stream_base = (unsigned long long) token * flat_size;

    __shared__ float inverse_rms;
    __shared__ float mixed[GLM53_MIX];
    __shared__ float pre[GLM53_HC];
    __shared__ float sinkhorn[GLM53_HC * GLM53_HC];

    if (tid == 0) {
        inverse_rms = inv_rms_in[token];
    }
    __syncthreads();

    // The 24 mixing dots were hoisted into one f32 GEMM over all tokens
    // (`mixed_raw[token, 24]`); only the per-token inverse-RMS scale remains.
    if (tid < GLM53_MIX) {
        mixed[tid] = mixed_raw[(unsigned long long) token * GLM53_MIX + tid] * inverse_rms;
    }
    __syncthreads();

    if (tid == 0) {
        #pragma unroll
        for (unsigned int index = 0; index < GLM53_HC; ++index) {
            const float pre_logit = mixed[index] * scale[0] + base[index];
            pre[index] = 1.0f / (1.0f + expf(-pre_logit)) + hc_eps;
            const float post_logit =
                mixed[GLM53_HC + index] * scale[1] + base[GLM53_HC + index];
            post[(unsigned long long) token * hc + index] =
                __float2bfloat16(2.0f / (1.0f + expf(-post_logit)));
        }
        #pragma unroll
        for (unsigned int row = 0; row < GLM53_HC; ++row) {
            float maximum = -FLT_MAX;
            #pragma unroll
            for (unsigned int column = 0; column < GLM53_HC; ++column) {
                const unsigned int index = row * GLM53_HC + column;
                sinkhorn[index] = mixed[2 * GLM53_HC + index] * scale[2] +
                    base[2 * GLM53_HC + index];
                maximum = fmaxf(maximum, sinkhorn[index]);
            }
            float sum = 0.0f;
            #pragma unroll
            for (unsigned int column = 0; column < GLM53_HC; ++column) {
                const unsigned int index = row * GLM53_HC + column;
                sinkhorn[index] = expf(sinkhorn[index] - maximum);
                sum += sinkhorn[index];
            }
            #pragma unroll
            for (unsigned int column = 0; column < GLM53_HC; ++column) {
                const unsigned int index = row * GLM53_HC + column;
                sinkhorn[index] = sinkhorn[index] / sum + hc_eps;
            }
        }
        #pragma unroll
        for (unsigned int column = 0; column < GLM53_HC; ++column) {
            float sum = hc_eps;
            #pragma unroll
            for (unsigned int row = 0; row < GLM53_HC; ++row) {
                sum += sinkhorn[row * GLM53_HC + column];
            }
            #pragma unroll
            for (unsigned int row = 0; row < GLM53_HC; ++row) {
                sinkhorn[row * GLM53_HC + column] /= sum;
            }
        }
        for (unsigned int iteration = 1; iteration < sinkhorn_iters; ++iteration) {
            #pragma unroll
            for (unsigned int row = 0; row < GLM53_HC; ++row) {
                float sum = hc_eps;
                #pragma unroll
                for (unsigned int column = 0; column < GLM53_HC; ++column) {
                    sum += sinkhorn[row * GLM53_HC + column];
                }
                #pragma unroll
                for (unsigned int column = 0; column < GLM53_HC; ++column) {
                    sinkhorn[row * GLM53_HC + column] /= sum;
                }
            }
            #pragma unroll
            for (unsigned int column = 0; column < GLM53_HC; ++column) {
                float sum = hc_eps;
                #pragma unroll
                for (unsigned int row = 0; row < GLM53_HC; ++row) {
                    sum += sinkhorn[row * GLM53_HC + column];
                }
                #pragma unroll
                for (unsigned int row = 0; row < GLM53_HC; ++row) {
                    sinkhorn[row * GLM53_HC + column] /= sum;
                }
            }
        }
        #pragma unroll
        for (unsigned int index = 0; index < GLM53_HC * GLM53_HC; ++index) {
            combination[(unsigned long long) token * GLM53_HC * GLM53_HC + index] =
                __float2bfloat16(sinkhorn[index]);
        }
    }
    __syncthreads();

    for (unsigned int column = tid; column < hidden_size; column += GLM53_THREADS) {
        float sum = 0.0f;
        #pragma unroll
        for (unsigned int stream = 0; stream < GLM53_HC; ++stream) {
            sum = fmaf(
                pre[stream],
                streams[stream_base + (unsigned long long) stream * hidden_size + column],
                sum);
        }
        collapsed[(unsigned long long) token * hidden_size + column] =
            __float2bfloat16(sum);
    }
}


// ---------------------------------------------------------------------------
// `atlas_glm53_hc_post_prompt`: the mHC output fold for the prompt scope.
//
// `atlas_glm53_hc_post` (iq3/glm53_hyper.cu) is declared
// `__launch_bounds__(256, 1)` and launched one block per token, so a purely
// memory-bound fold runs at 256 resident threads per SM (~12% occupancy).
// At 2047 tokens it moves ~278 MB and takes ~2.7 ms per layer.
//
// This variant keeps the arithmetic EXACTLY as the original -- same f32
// `placement`, same fmaf accumulation order over the four sources, same single
// `glm53_stream_store` rounding -- and changes only the decomposition:
//   * 2D grid (column-chunk, token) so a token's 4096 columns span several
//     blocks, and `__launch_bounds__(256, 6)` lets six blocks share an SM;
//   * each thread owns 4 CONSECUTIVE columns and moves them as float4, so the
//     residual/output traffic issues as 16-byte transactions.
//
// In-place safety is unchanged and is the reason `residual` is still not
// `__restrict__`: the walk passes `widened_hc` as both `residual` and
// `output`. One thread owns a column for the whole grid and reads all four
// sources of its columns before writing any target, so the aliasing is safe --
// but only while the compiler is not told the pointers cannot alias.
#define GLM53_POST_VEC 4U
extern "C" __global__ void __launch_bounds__(GLM53_THREADS, 6)
atlas_glm53_hc_post_prompt(
        const __nv_bfloat16 * __restrict__ block_output,
        const float * residual,
        const __nv_bfloat16 * __restrict__ post,
        const __nv_bfloat16 * __restrict__ combination,
        float * output,
        unsigned int hidden_size, unsigned int hc,
        unsigned int tokens, unsigned int round_to_bf16) {
    if (hidden_size != GLM53_HIDDEN || hc != GLM53_HC) {
        return;
    }
    const unsigned int token = blockIdx.y;
    if (token >= tokens) {
        return;
    }
    const unsigned int column =
        (blockIdx.x * GLM53_THREADS + threadIdx.x) * GLM53_POST_VEC;
    if (column >= hidden_size) {
        return;
    }
    const unsigned long long stream_base =
        (unsigned long long) token * hc * hidden_size;
    const unsigned long long post_base = (unsigned long long) token * hc;
    const unsigned long long comb_base = (unsigned long long) token * hc * hc;

    // Read all four sources for the owned columns BEFORE any store.
    float4 residual_values[GLM53_HC];
    #pragma unroll
    for (unsigned int source = 0; source < GLM53_HC; ++source) {
        residual_values[source] = *reinterpret_cast<const float4 *>(
            residual + stream_base + (unsigned long long) source * hidden_size + column);
    }
    float4 block_values;
    {
        const __nv_bfloat16 *bo =
            block_output + (unsigned long long) token * hidden_size + column;
        block_values.x = __bfloat162float(bo[0]);
        block_values.y = __bfloat162float(bo[1]);
        block_values.z = __bfloat162float(bo[2]);
        block_values.w = __bfloat162float(bo[3]);
    }
    float post_values[GLM53_HC];
    float comb_values[GLM53_HC * GLM53_HC];
    #pragma unroll
    for (unsigned int target = 0; target < GLM53_HC; ++target) {
        post_values[target] = __bfloat162float(post[post_base + target]);
        #pragma unroll
        for (unsigned int source = 0; source < GLM53_HC; ++source) {
            comb_values[source * GLM53_HC + target] = __bfloat162float(
                combination[comb_base + (unsigned long long) source * GLM53_HC + target]);
        }
    }

    #pragma unroll
    for (unsigned int target = 0; target < GLM53_HC; ++target) {
        float4 out;
        float *o = reinterpret_cast<float *>(&out);
        const float *b = reinterpret_cast<const float *>(&block_values);
        #pragma unroll
        for (unsigned int lane = 0; lane < GLM53_POST_VEC; ++lane) {
            const float placement = post_values[target] * b[lane];
            float mixed = 0.0f;
            #pragma unroll
            for (unsigned int source = 0; source < GLM53_HC; ++source) {
                mixed = fmaf(
                    comb_values[source * GLM53_HC + target],
                    reinterpret_cast<const float *>(&residual_values[source])[lane],
                    mixed);
            }
            o[lane] = round_to_bf16
                ? __bfloat162float(__float2bfloat16(placement + mixed))
                : (placement + mixed);
        }
        *reinterpret_cast<float4 *>(
            output + stream_base + (unsigned long long) target * hidden_size + column) = out;
    }
}

// ---------------------------------------------------------------------------
// `atlas_glm53_dsa_index_projection_tiled`: the DSA indexer head projection
// `[rows,4096] x [32,4096]^T -> [rows,32]` for the prompt scope.
//
// `atlas_glm53_dsa_index_projection_f32_bf16` (iq3/glm53_dsa_norm_projection.cu)
// runs one block of 32 threads per token, thread `h` walking
// `weight[h*4096 + column]` for column 0..4095. The 32 threads of the warp are
// therefore 4096 floats (16 KB) apart on every single load: each iteration
// issues 32 separate sectors and uses 4 of the 128 bytes each one fetches. The
// weight is only 512 KB and stays L2-resident, so this costs L2 bandwidth
// rather than DRAM, but it is still ~32x more traffic than the data needs.
//
// This variant stages the weight through shared memory in column tiles. The
// cooperative load is coalesced (consecutive threads read consecutive columns
// of the same head), and each thread then reads its own head's row out of
// shared. The per-thread accumulation is the SAME serial f32 `fmaf` over
// columns 0..4095 in the same order, so the result is bit-identical.
#define GLM53_IDX_HEADS 32U
#define GLM53_IDX_TILE 128U
extern "C" __global__ void __launch_bounds__(GLM53_IDX_HEADS, 8)
atlas_glm53_dsa_index_projection_tiled(
        const __nv_bfloat16 * __restrict__ input,
        const float * __restrict__ weight,
        __nv_bfloat16 * __restrict__ output,
        unsigned int rows, unsigned int inner, unsigned int heads) {
    if (blockDim.x != GLM53_IDX_HEADS || blockDim.y != 1U || blockDim.z != 1U ||
        gridDim.x != rows || gridDim.y != 1U || gridDim.z != 1U ||
        rows == 0U || inner != GLM53_HIDDEN || heads != GLM53_IDX_HEADS) {
        return;
    }
    const unsigned int row = blockIdx.x;
    const unsigned int head = threadIdx.x;
    if (row >= rows) {
        return;
    }
    __shared__ float w_tile[GLM53_IDX_HEADS][GLM53_IDX_TILE];
    __shared__ float x_tile[GLM53_IDX_TILE];
    const unsigned long long input_base = (unsigned long long) row * GLM53_HIDDEN;

    float sum = 0.0f;
    for (unsigned int base = 0U; base < GLM53_HIDDEN; base += GLM53_IDX_TILE) {
        __syncthreads();
        // Coalesced: for a fixed `h` the 32 lanes cover 32 consecutive columns.
        for (unsigned int idx = head; idx < GLM53_IDX_HEADS * GLM53_IDX_TILE;
             idx += GLM53_IDX_HEADS) {
            const unsigned int h = idx / GLM53_IDX_TILE;
            const unsigned int c = idx % GLM53_IDX_TILE;
            w_tile[h][c] =
                weight[(unsigned long long) h * GLM53_HIDDEN + base + c];
        }
        for (unsigned int c = head; c < GLM53_IDX_TILE; c += GLM53_IDX_HEADS) {
            x_tile[c] = __bfloat162float(input[input_base + base + c]);
        }
        __syncthreads();
        // Same order, same fmaf chain as the donor kernel.
        #pragma unroll 8
        for (unsigned int c = 0U; c < GLM53_IDX_TILE; ++c) {
            sum = fmaf(w_tile[head][c], x_tile[c], sum);
        }
    }
    output[(unsigned long long) row * GLM53_IDX_HEADS + head] =
        __float2bfloat16_rn(sum);
}
