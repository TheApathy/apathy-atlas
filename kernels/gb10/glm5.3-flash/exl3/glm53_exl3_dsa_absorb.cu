// SPDX-License-Identifier: AGPL-3.0-only

// GLM-5.3 sparse-attention absorption banks at prompt M <= 2048.
//
// The checkpoint packs each head as K_b[256,512], V_b[256,512].  Atlas
// transposes activations to head-major before these calls.  A single launch
// covers all 64 independent heads, replacing 64 tiny cuBLASLt launches.

#include <cuda_bf16.h>

namespace {

constexpr unsigned HEAD_DIM = 256;
constexpr unsigned LATENT = 512;
constexpr unsigned HEAD_WEIGHT_STRIDE = 2 * HEAD_DIM * LATENT;
constexpr unsigned K_STEP = 16;
constexpr unsigned N_TILE = 32;
constexpr unsigned THREADS = 128;
constexpr unsigned ROW_TILE = 16;

template <bool WEIGHT_NK>
__device__ __forceinline__ void absorb_bank(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output,
    unsigned rows,
    unsigned k_dim,
    unsigned n_dim) {
    const unsigned head = blockIdx.x;
    const unsigned n_base = blockIdx.y * N_TILE;
    const unsigned row_base = blockIdx.z * ROW_TILE;
    if (row_base >= rows) return;
    const unsigned warp = threadIdx.x >> 5;
    const unsigned lane = threadIdx.x & 31;
    const unsigned group = lane >> 2;
    const unsigned pair = lane & 3;

    __shared__ __nv_bfloat16 tile_a[16][K_STEP + 2];
    __shared__ __nv_bfloat16 tile_b[K_STEP][N_TILE + 2];

    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const __nv_bfloat16* head_input = input + (unsigned long long)head * rows * k_dim;
    const __nv_bfloat16* head_weight = weight + (unsigned long long)head * HEAD_WEIGHT_STRIDE;
    __nv_bfloat16* head_output = output + (unsigned long long)head * rows * n_dim;

    for (unsigned k_base = 0; k_base < k_dim; k_base += K_STEP) {
        for (unsigned index = threadIdx.x; index < ROW_TILE * K_STEP; index += THREADS) {
            const unsigned row = index / K_STEP;
            const unsigned col = index % K_STEP;
            const unsigned global_row = row_base + row;
            tile_a[row][col] = global_row < rows
                ? head_input[(unsigned long long)global_row * k_dim + k_base + col]
                : __float2bfloat16(0.0f);
        }
        for (unsigned index = threadIdx.x; index < K_STEP * N_TILE; index += THREADS) {
            // Follow the contiguous checkpoint dimension in both layouts:
            // key weights are [K,N], value weights are [N,K].
            const unsigned k = WEIGHT_NK ? index % K_STEP : index / N_TILE;
            const unsigned n = WEIGHT_NK ? index / K_STEP : index % N_TILE;
            const unsigned global_n = n_base + n;
            __nv_bfloat16 value = __float2bfloat16(0.0f);
            if (global_n < n_dim) {
                if constexpr (WEIGHT_NK) {
                    value = head_weight[(unsigned long long)global_n * k_dim + k_base + k];
                } else {
                    value = head_weight[(unsigned long long)(k_base + k) * n_dim + global_n];
                }
            }
            tile_b[k][n] = value;
        }
        __syncthreads();

        const unsigned short* a = reinterpret_cast<const unsigned short*>(tile_a);
        const unsigned short* b = reinterpret_cast<const unsigned short*>(tile_b);
        constexpr unsigned a_stride = K_STEP + 2;
        constexpr unsigned b_stride = N_TILE + 2;
        const unsigned row0 = group;
        const unsigned row1 = row0 + 8;
        const unsigned col0 = pair * 2;
        const unsigned col1 = col0 + 8;
        const unsigned a0 = *reinterpret_cast<const unsigned int*>(&a[row0 * a_stride + col0]);
        const unsigned a1 = *reinterpret_cast<const unsigned int*>(&a[row1 * a_stride + col0]);
        const unsigned a2 = *reinterpret_cast<const unsigned int*>(&a[row0 * a_stride + col1]);
        const unsigned a3 = *reinterpret_cast<const unsigned int*>(&a[row1 * a_stride + col1]);
        const unsigned n = warp * 8 + group;
        const unsigned bk0 = pair * 2;
        const unsigned bk1 = bk0 + 8;
        const unsigned b0 = (static_cast<unsigned>(b[(bk0 + 1) * b_stride + n]) << 16) |
                            static_cast<unsigned>(b[bk0 * b_stride + n]);
        const unsigned b1 = (static_cast<unsigned>(b[(bk1 + 1) * b_stride + n]) << 16) |
                            static_cast<unsigned>(b[bk1 * b_stride + n]);
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
            : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1),
              "f"(acc[0]), "f"(acc[1]), "f"(acc[2]), "f"(acc[3]));
        __syncthreads();
    }

    const unsigned out_col0 = n_base + warp * 8 + pair * 2;
    const unsigned out_col1 = out_col0 + 1;
    const unsigned out_row0 = row_base + group;
    const unsigned out_row1 = out_row0 + 8;
    if (out_row0 < rows && out_col0 < n_dim)
        head_output[(unsigned long long)out_row0 * n_dim + out_col0] = __float2bfloat16(acc[0]);
    if (out_row0 < rows && out_col1 < n_dim)
        head_output[(unsigned long long)out_row0 * n_dim + out_col1] = __float2bfloat16(acc[1]);
    if (out_row1 < rows && out_col0 < n_dim)
        head_output[(unsigned long long)out_row1 * n_dim + out_col0] = __float2bfloat16(acc[2]);
    if (out_row1 < rows && out_col1 < n_dim)
        head_output[(unsigned long long)out_row1 * n_dim + out_col1] = __float2bfloat16(acc[3]);
}

}  // namespace

extern "C" __global__ __launch_bounds__(THREADS) void atlas_glm53_dsa_absorb_key_bank(
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    __nv_bfloat16* output,
    unsigned rows) {
    absorb_bank<false>(input, weight, output, rows, HEAD_DIM, LATENT);
}

extern "C" __global__ __launch_bounds__(THREADS) void atlas_glm53_dsa_absorb_value_bank(
    const __nv_bfloat16* input,
    const __nv_bfloat16* weight,
    __nv_bfloat16* output,
    unsigned rows) {
    absorb_bank<true>(input, weight, output, rows, LATENT, HEAD_DIM);
}
