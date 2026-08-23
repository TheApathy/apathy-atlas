// SPDX-License-Identifier: AGPL-3.0-only

// Fused M17 gate/up projection and FP32 activation materialization. Each
// projection retains the exact K1 K8 lane ownership, ordered FP32 updates,
// shuffle tree, cross-warp add, and BF16 rounding boundary.

template <int MAX_M>
__device__ __forceinline__ void w4a16_dual_exact_materialize_f32_body(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ gate_packed,
    const unsigned char* __restrict__ gate_scale,
    const float gate_scale2,
    const unsigned char* __restrict__ up_packed,
    const unsigned char* __restrict__ up_scale,
    const float up_scale2,
    float* __restrict__ activation,
    unsigned int M, unsigned int N, unsigned int K,
    float* s_lut, float* smem, __nv_bfloat16* rounded_gate)
{
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int warp_lane = lane % WARP_SIZE;
    const unsigned int warp_idx = lane / WARP_SIZE;
    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    const bool valid = M > 0u && M <= (unsigned int)MAX_M && n < N;

    if (threadIdx.x < 16) s_lut[threadIdx.x] = E2M1_LUT_FUSED_W4[threadIdx.x];
    __syncthreads();

    // Pass 0 is gate and pass 1 is up. The barrier at the end of each pass
    // makes the BF16 gate rounding boundary visible before up is consumed.
    #pragma unroll 1
    for (int proj = 0; proj < 2; ++proj) {
        const unsigned char* B_packed = proj == 0 ? gate_packed : up_packed;
        const unsigned char* B_scale = proj == 0 ? gate_scale : up_scale;
        const float scale2 = proj == 0 ? gate_scale2 : up_scale2;

        float acc[MAX_M];
        #pragma unroll
        for (int row = 0; row < MAX_M; ++row) acc[row] = 0.0f;

        if (valid) {
            const unsigned int half_K = K / 2;
            const unsigned int num_groups = K / GROUP_SIZE;
            const unsigned int K8 = K / 8;
            const unsigned long long weight_row =
                (unsigned long long)n * half_K;
            const unsigned long long scale_row =
                (unsigned long long)n * num_groups;

            for (unsigned int k8 = lane; k8 < K8; k8 += 64u) {
                const unsigned int base_k = k8 * 8;
                const unsigned int packed4 = *(const unsigned int*)(
                    B_packed + weight_row + k8 * 4);
                const unsigned char scale_byte =
                    B_scale[scale_row + base_k / GROUP_SIZE];
                __nv_fp8_e4m3 fp8;
                *(unsigned char*)&fp8 = scale_byte;
                const float scale = (float)fp8 * scale2;

                float w_lo[4];
                float w_hi[4];
                #pragma unroll
                for (int b = 0; b < 4; ++b) {
                    const unsigned char byte_val =
                        (unsigned char)(packed4 >> (b * 8));
                    w_lo[b] = s_lut[byte_val & 0xF] * scale;
                    w_hi[b] = s_lut[byte_val >> 4] * scale;
                }

                #pragma unroll
                for (int row = 0; row < MAX_M; ++row) {
                    if ((unsigned int)row < M) {
                        const __nv_bfloat16* A_row =
                            A + (unsigned long long)row * K;
                        const uint4 a_data = ((const uint4*)A_row)[k8];
                        const unsigned int a_raw[4] = {
                            a_data.x, a_data.y, a_data.z, a_data.w
                        };
                        #pragma unroll
                        for (int b = 0; b < 4; ++b) {
                            __nv_bfloat16 a_lo, a_hi;
                            *(unsigned short*)&a_lo =
                                (unsigned short)(a_raw[b] & 0xFFFF);
                            *(unsigned short*)&a_hi =
                                (unsigned short)(a_raw[b] >> 16);
                            acc[row] += __bfloat162float(a_lo) * w_lo[b];
                            acc[row] += __bfloat162float(a_hi) * w_hi[b];
                        }
                    }
                }
            }
        }

        #pragma unroll
        for (int row = 0; row < MAX_M; ++row) {
            if ((unsigned int)row < M) {
                #pragma unroll
                for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                    acc[row] += __shfl_down_sync(0xFFFFFFFF, acc[row], offset);
                }
                if (warp_lane == 0) {
                    smem[(row * N_PER_BLOCK + local_out) * 2 + warp_idx] =
                        acc[row];
                }
            }
        }
        __syncthreads();

        if (valid && lane == 0) {
            #pragma unroll
            for (int row = 0; row < MAX_M; ++row) {
                if ((unsigned int)row < M) {
                    const unsigned int slot = row * N_PER_BLOCK + local_out;
                    const __nv_bfloat16 rounded = __float2bfloat16(
                        smem[slot * 2] + smem[slot * 2 + 1]);
                    if (proj == 0) {
                        rounded_gate[slot] = rounded;
                    } else {
                        const float gate =
                            __bfloat162float(rounded_gate[slot]);
                        const float up = __bfloat162float(rounded);
                        activation[(unsigned long long)row * N + n] =
                            (gate / (1.0f + __expf(-gate))) * up;
                    }
                }
            }
        }
        __syncthreads();
    }
}

extern "C" __global__ void w4a16_gemv_dual_exact_materialize_f32_m17(
    const __nv_bfloat16* A,
    const unsigned char* gate_packed, const unsigned char* gate_scale,
    const float gate_scale2,
    const unsigned char* up_packed, const unsigned char* up_scale,
    const float up_scale2, float* activation,
    unsigned int M, unsigned int N, unsigned int K)
{
    __shared__ float s_lut[16];
    __shared__ float smem[17 * N_PER_BLOCK * 2];
    __shared__ __nv_bfloat16 rounded_gate[17 * N_PER_BLOCK];
    w4a16_dual_exact_materialize_f32_body<17>(
        A, gate_packed, gate_scale, gate_scale2,
        up_packed, up_scale, up_scale2, activation,
        M, N, K, s_lut, smem, rounded_gate);
}
