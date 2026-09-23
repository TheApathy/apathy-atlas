// SPDX-License-Identifier: AGPL-3.0-only
// Device implementation core: ExLlamaV3 MIT, pinned and verified by build.rs.

#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#include <util.h>
#include <util.cuh>
#include <ptx.cuh>
#include <quant/exl3_kernel_map.cuh>
#include <quant/hadamard_inner.cuh>

#define barrier_acquire(lock, stage) \
    do \
    { \
        (void) (lock); \
        (void) (stage); \
        __syncthreads(); \
    } while (0)
#define barrier_release(lock, value, reset) \
    do \
    { \
        (void) (lock); \
        (void) (value); \
        (void) (reset); \
        __syncthreads(); \
    } while (0)

struct AtlasIndex3
{
    unsigned int x;
    unsigned int y;
    unsigned int z;
};

__device__ __forceinline__ AtlasIndex3 atlas_original_block_idx()
{
    return {blockIdx.x, blockIdx.y, blockIdx.z};
}

__device__ __forceinline__ AtlasIndex3 atlas_original_grid_dim()
{
    return {gridDim.x, gridDim.y, gridDim.z};
}

__device__ __forceinline__ AtlasIndex3 atlas_swapped_block_idx()
{
    const AtlasIndex3 value = atlas_original_block_idx();
    return {value.z, value.y, value.x};
}

__device__ __forceinline__ AtlasIndex3 atlas_swapped_grid_dim()
{
    const AtlasIndex3 value = atlas_original_grid_dim();
    return {value.z, value.y, value.x};
}

#define blockIdx atlas_swapped_block_idx()
#define gridDim atlas_swapped_grid_dim()
#include <quant/exl3_gemm_inner.cuh>
#undef gridDim
#undef blockIdx
#undef barrier_release
#undef barrier_acquire

constexpr int QKV_ROW_X_THREADS = 256;

extern "C" __global__ __launch_bounds__(QKV_ROW_X_THREADS, 3)
void atlas_glm53_exl3_kda_qkv_n256_row_x
(
    const half* __restrict__ input_hadamard,
    const uint16_t* __restrict__ trellis,
    half* __restrict__ output,
    const int rows,
    const int size_k,
    const int size_n,
    int* __restrict__ locks,
    const half* __restrict__ output_scale
)
{
    // X is the row slab so linear CTA order presents adjacent rows with the
    // same Z-owned weight slice. The donor inner sees swapped X/Z above.
    const int row = 16 * blockIdx.x;
    if (row >= rows) return;
    const int local_rows = MIN(rows - row, 16);
    const int lock_stride = size_n / 16;
    exl3_gemm_kernel_inner
    <4, false, 2, 16, 16, 256, 3, 1, true>
    (
        input_hadamard + row * size_k,
        trellis,
        output + row * size_n,
        local_rows,
        size_k,
        size_n,
        locks + blockIdx.x * lock_stride,
        output_scale
    );
}
