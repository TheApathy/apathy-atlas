// SPDX-License-Identifier: AGPL-3.0-only
// Exact GLM-5.3 mHC and standard RMSNorm boundaries.
//
// The four mHC residual STREAMS are F32. Everything else on this path stays
// BF16 (the hidden vector, the collapsed block input/output, and the per-token
// `post`/`comb` mixing coefficients), matching the checkpoint's own dtypes.
// The streams are the one buffer that is written and re-read once per layer for
// 45 layers, so a bf16 store there is a ~0.4% seed that the mHC fold then
// amplifies at ~1.3x/layer. These kernels already accumulate in f32 internally;
// holding the stream buffer in f32 simply stops throwing that away twice per
// layer.

#include <cuda_bf16.h>
#include <float.h>
#include <math.h>

#define GLM53_HIDDEN 4096U
#define GLM53_HC 4U
#define GLM53_MIX 24U
#define GLM53_THREADS 256U

// SINGLE-VARIABLE A/B (see GLM53_F32_MHC_STREAMS in
// crates/spark-model/src/layers/glm53_target_schedule.rs).
//
// `round_to_bf16` is 0 for the shipping build and 1 only for the deliberate
// regression. It gates the PRECISION carried across a layer boundary and
// nothing else: the buffer stays f32, so widened_hc, the arena, the capture
// slots and every extent test are untouched, and the ONLY thing that changes
// is whether a stream value survives the layer at f32 or bf16.
//
// It is applied at the two sites that WRITE a stream and nowhere else. In
// particular the fold below still spends exactly ONE rounding on
// `placement + mixed`; restoring the old three-rounding form here would fold a
// separate landed fix back in and reconfound the very measurement this exists
// to make.
__device__ __forceinline__ float glm53_stream_store(float value,
                                                    unsigned int round_to_bf16) {
    return round_to_bf16 ? __bfloat162float(__float2bfloat16(value)) : value;
}

__device__ __forceinline__ float glm53_block_sum(float *values, unsigned int tid) {
    for (unsigned int stride = GLM53_THREADS / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            values[tid] += values[tid + stride];
        }
        __syncthreads();
    }
    return values[0];
}

extern "C" __global__ void __launch_bounds__(GLM53_THREADS, 1)
atlas_glm53_hc_expand(
        const __nv_bfloat16 * __restrict__ hidden,
        float * __restrict__ streams,
        unsigned int hidden_size, unsigned int hc,
        unsigned int round_to_bf16) {
    if (hidden_size != GLM53_HIDDEN || hc != GLM53_HC) {
        return;
    }
    const unsigned int token = blockIdx.x;
    const unsigned long long hidden_base = (unsigned long long) token * hidden_size;
    const unsigned long long stream_base = hidden_base * hc;
    for (unsigned int column = threadIdx.x; column < hidden_size;
         column += GLM53_THREADS) {
        const float value = __bfloat162float(hidden[hidden_base + column]);
        #pragma unroll
        for (unsigned int stream = 0; stream < GLM53_HC; ++stream) {
            streams[stream_base + (unsigned long long) stream * hidden_size + column] =
                glm53_stream_store(value, round_to_bf16);
        }
    }
}

extern "C" __global__ void __launch_bounds__(GLM53_THREADS, 1)
atlas_glm53_hc_pre(
        const float * __restrict__ streams,
        const float * __restrict__ function,
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

    #pragma unroll
    for (unsigned int output = 0; output < GLM53_MIX; ++output) {
        const unsigned long long function_base =
            (unsigned long long) output * flat_size;
        float sum = 0.0f;
        for (unsigned int index = tid; index < flat_size; index += GLM53_THREADS) {
            sum = fmaf(
                function[function_base + index],
                streams[stream_base + index],
                sum);
        }
        reduction[tid] = sum;
        __syncthreads();
        const float dot = glm53_block_sum(reduction, tid);
        if (tid == 0) {
            mixed[output] = dot * inverse_rms;
        }
        __syncthreads();
    }

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

extern "C" __global__ void __launch_bounds__(GLM53_THREADS, 1)
atlas_glm53_hc_post(
        const __nv_bfloat16 * __restrict__ block_output,
        // NOT __restrict__: the walk runs this fold IN PLACE, passing
        // `widened_hc` as both `residual` and `output` (dispatch.rs, the
        // Glm53HyperPostBuffers literal). `__restrict__` would promise the
        // compiler they cannot alias, licensing it to sink these loads past
        // the stores below and silently read back half-written streams. The
        // per-column ownership here makes the aliasing itself safe -- one
        // thread owns a column, and it reads all four sources before writing
        // any target -- but only if the compiler is not told otherwise.
        const float * residual,
        const __nv_bfloat16 * __restrict__ post,
        const __nv_bfloat16 * __restrict__ combination,
        float * output,
        unsigned int hidden_size, unsigned int hc,
        unsigned int round_to_bf16) {
    if (hidden_size != GLM53_HIDDEN || hc != GLM53_HC) {
        return;
    }
    const unsigned int token = blockIdx.x;
    const unsigned long long stream_base =
        (unsigned long long) token * hc * hidden_size;
    const unsigned long long post_base = (unsigned long long) token * hc;
    const unsigned long long comb_base = (unsigned long long) token * hc * hc;
    for (unsigned int column = threadIdx.x; column < hidden_size;
         column += GLM53_THREADS) {
        float residual_values[GLM53_HC];
        #pragma unroll
        for (unsigned int source = 0; source < GLM53_HC; ++source) {
            residual_values[source] =
                residual[stream_base + (unsigned long long) source * hidden_size + column];
        }
        const float block_value = __bfloat162float(
            block_output[(unsigned long long) token * hidden_size + column]);
        #pragma unroll
        for (unsigned int target = 0; target < GLM53_HC; ++target) {
            // Keep f32: `placement` and `mixed` are both f32 quantities that
            // are summed below. Rounding EACH to bf16 first (and the sum again)
            // spent three roundings where one suffices -- 2 folds/layer x 45
            // layers = 180 needless roundings feeding the seed error that this
            // very fold then amplifies at ~1.3x/layer.
            const float placement =
                __bfloat162float(post[post_base + target]) * block_value;
            float mixed = 0.0f;
            #pragma unroll
            for (unsigned int source = 0; source < GLM53_HC; ++source) {
                mixed = fmaf(
                    __bfloat162float(combination[
                        comb_base + (unsigned long long) source * GLM53_HC + target]),
                    residual_values[source],
                    mixed);
            }
            output[stream_base + (unsigned long long) target * hidden_size + column] =
                glm53_stream_store(placement + mixed, round_to_bf16);
        }
    }
}

extern "C" __global__ void __launch_bounds__(GLM53_THREADS, 1)
atlas_glm53_hc_mean(
        const float * __restrict__ streams,
        __nv_bfloat16 * __restrict__ hidden,
        unsigned int hidden_size, unsigned int hc) {
    if (hidden_size != GLM53_HIDDEN || hc != GLM53_HC) {
        return;
    }
    const unsigned int token = blockIdx.x;
    const unsigned long long stream_base =
        (unsigned long long) token * hc * hidden_size;
    for (unsigned int column = threadIdx.x; column < hidden_size;
         column += GLM53_THREADS) {
        float sum = 0.0f;
        #pragma unroll
        for (unsigned int stream = 0; stream < GLM53_HC; ++stream) {
            sum += streams[stream_base + (unsigned long long) stream * hidden_size + column];
        }
        hidden[(unsigned long long) token * hidden_size + column] =
            __float2bfloat16(sum * 0.25f);
    }
}

__device__ __forceinline__ float glm53_warp_sum(float value) {
    #pragma unroll
    for (unsigned int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffU, value, offset);
    }
    return value;
}

extern "C" __global__ void __launch_bounds__(1024, 1)
atlas_glm53_rms_norm(
        // NOT __restrict__: `Glm53HyperKernels::norm` explicitly permits
        // output == input and the walk uses that (collapsed -> collapsed).
        const __nv_bfloat16 * input,
        const float * __restrict__ weight,
        __nv_bfloat16 * output,
        unsigned int hidden_size, float eps) {
    if (hidden_size != GLM53_HIDDEN || eps != 1.0e-5f) {
        return;
    }
    const unsigned int token = blockIdx.x;
    const unsigned long long base = (unsigned long long) token * hidden_size;
    float sum = 0.0f;
    for (unsigned int column = threadIdx.x; column < hidden_size; column += blockDim.x) {
        const float value = __bfloat162float(input[base + column]);
        sum += value * value;
    }
    sum = glm53_warp_sum(sum);
    __shared__ float warp_sums[32];
    const unsigned int lane = threadIdx.x & 31U;
    const unsigned int warp = threadIdx.x >> 5;
    if (lane == 0) {
        warp_sums[warp] = sum;
    }
    __syncthreads();
    if (warp == 0) {
        float warp_value = warp_sums[lane];
        warp_value = glm53_warp_sum(warp_value);
        if (lane == 0) {
            warp_sums[0] = warp_value;
        }
    }
    __syncthreads();
    const float inverse_rms = rsqrtf(warp_sums[0] / (float) hidden_size + eps);
    for (unsigned int column = threadIdx.x; column < hidden_size; column += blockDim.x) {
        const __nv_bfloat16 normalized = __float2bfloat16(
            __bfloat162float(input[base + column]) * inverse_rms);
        const __nv_bfloat16 model_weight = __float2bfloat16(weight[column]);
        output[base + column] = __float2bfloat16(
            __bfloat162float(normalized) * __bfloat162float(model_weight));
    }
}
