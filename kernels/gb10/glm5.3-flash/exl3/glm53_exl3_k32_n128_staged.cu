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

// Keep the donor's exact 48-way split-K partial-sum and lock ordering inside
// every independent M16 row tile. Each grid-Z row receives a disjoint lock
// slab, so rows execute concurrently without changing reduction order.
#include <quant/exl3_gemm_inner.cuh>

constexpr int K32_N128_THREADS = 512;

template<int SH_STAGES, int FRAG_STAGES>
inline __device__ void k32_n128_row_tile
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
    // The donor output Hadamard helper indexes scale through blockIdx.y.
    // Keep Y zero and carry independent M16 rows in grid Z.
    const int row = 16 * blockIdx.z;
    if (row >= rows) return;
    const int local_rows = MIN(rows - row, 16);
    const int lock_stride = size_n / 16;
    exl3_gemm_kernel_inner
    <4, false, 2, 16, 32, 128, SH_STAGES, FRAG_STAGES, true>
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

#define K32_N128_KERNEL(NAME, SH_STAGES, FRAG_STAGES) \
    extern "C" __global__ __launch_bounds__(K32_N128_THREADS, 2) \
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
        k32_n128_row_tile<SH_STAGES, FRAG_STAGES> \
        ( \
            input_hadamard, trellis, output, rows, size_k, size_n, \
            locks, output_scale \
        ); \
    }

K32_N128_KERNEL(atlas_glm53_exl3_k32_n128_sh4_f1, 4, 1)

#undef K32_N128_KERNEL
