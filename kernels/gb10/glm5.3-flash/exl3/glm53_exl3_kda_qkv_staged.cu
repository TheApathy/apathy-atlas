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

// Every staged core block owns all 256 K16 tiles for one output-N tile.
// There is no split-K producer to order through the donor's global locks.
// Keep its two block-local synchronization points without sharing state across
// independent M16 rows.
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
#include <quant/exl3_gemm_inner.cuh>
#undef barrier_acquire
#undef barrier_release

constexpr int QKV_THREADS = 256;
constexpr int QKV_WARPS = QKV_THREADS / 32;

extern "C" __global__ __launch_bounds__(QKV_THREADS, 4)
void atlas_glm53_exl3_kda_qkv_hadamard
(
    const half* __restrict__ input,
    half* __restrict__ output,
    const half* __restrict__ scale,
    const int rows,
    const int size_k
)
{
    const int warp = blockIdx.x * QKV_WARPS + threadIdx.x / 32;
    const int warp_stride = gridDim.x * QKV_WARPS;
    const int total_warps = rows * size_k / 128;
    for (int index = warp; index < total_warps; index += warp_stride)
    {
        had_hf_r_128_inner<true, false>
        (
            input + index * 128,
            output + index * 128,
            scale + (index * 128) % size_k,
            0.088388347648f
        );
    }
}

template<int TILE_N, int SH_STAGES, int FRAG_STAGES>
inline __device__ void qkv_row_tile
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
    // The pinned output Hadamard helper uses blockIdx.y for scale indexing,
    // so keep Y exactly zero and carry independent M16 rows in grid Z.
    const int row = 16 * blockIdx.z;
    if (row >= rows) return;
    const int local_rows = MIN(rows - row, 16);
    const int lock_stride = size_n / 16;
    exl3_gemm_kernel_inner
    <4, false, 2, 16, 16, TILE_N, SH_STAGES, FRAG_STAGES, true>
    (
        input_hadamard + row * size_k,
        trellis,
        output + row * size_n,
        local_rows,
        size_k,
        size_n,
        locks + blockIdx.z * lock_stride,
        output_scale
    );
}

#define QKV_KERNEL(NAME, TILE_N, SH_STAGES, FRAG_STAGES, MIN_BLOCKS) \
    extern "C" __global__ __launch_bounds__(QKV_THREADS, MIN_BLOCKS) \
    void NAME \
    ( \
        const half* __restrict__ input_hadamard, \
        const uint16_t* __restrict__ trellis, \
        half* __restrict__ output, \
        const int rows, \
        const int size_k, \
        const int size_n, \
        int* __restrict__ locks, \
        const half* __restrict__ output_scale \
    ) \
    { \
        qkv_row_tile<TILE_N, SH_STAGES, FRAG_STAGES> \
        ( \
            input_hadamard, trellis, output, rows, size_k, size_n, \
            locks, output_scale \
        ); \
    }

QKV_KERNEL(atlas_glm53_exl3_kda_qkv_n256, 256, 3, 1, 3)

#undef QKV_KERNEL

template<int TILE_N, int SH_STAGES, int FRAG_STAGES>
inline __device__ void qkv_row_tile_at
(
    const half* __restrict__ input_hadamard,
    const uint16_t* __restrict__ trellis,
    half* __restrict__ output,
    const int rows,
    const int size_k,
    const int size_n,
    int* __restrict__ locks,
    const half* __restrict__ output_scale,
    const int row
)
{
    const int local_rows = MIN(rows - row, 16);
    const int lock_stride = size_n / 16;
    exl3_gemm_kernel_inner
    <4, false, 2, 16, 16, TILE_N, SH_STAGES, FRAG_STAGES, true>
    (
        input_hadamard + row * size_k,
        trellis,
        output + row * size_n,
        local_rows,
        size_k,
        size_n,
        locks + (row / 16) * lock_stride,
        output_scale
    );
}

// Process two adjacent M16 slabs while the same blockIdx.x weight slice is
// still hot. Each donor invocation retains its independent ascending-K
// arithmetic and writes a disjoint output/lock slab.
extern "C" __global__ __launch_bounds__(QKV_THREADS, 3)
void atlas_glm53_exl3_kda_qkv_n256_pair2
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
    const int row = 32 * blockIdx.z;
    if (row >= rows) return;
    qkv_row_tile_at<256, 3, 1>
    (
        input_hadamard, trellis, output, rows, size_k, size_n,
        locks, output_scale, row
    );
    if (row + 16 < rows)
    {
        __syncthreads();
        qkv_row_tile_at<256, 3, 1>
        (
            input_hadamard, trellis, output, rows, size_k, size_n,
            locks, output_scale, row + 16
        );
    }
}
