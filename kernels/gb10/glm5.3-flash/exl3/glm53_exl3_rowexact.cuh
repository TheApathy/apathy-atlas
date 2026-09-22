// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

// Execute a small-M projection as an ordered sequence of the pinned upstream
// M=1 kernels inside one cooperative launch.  The per-row Hadamard transform,
// GEMM dimensions, reduction order, and F16 output path are identical to T=1;
// grid barriers replace the kernel-launch boundaries between rows.
template<
    const int bits,
    const int TILESIZE_M,
    const int TILESIZE_K,
    const int TILESIZE_N,
    const int SH_STAGES,
    const int FRAG_STAGES>
inline __device__ void glm53_exl3_rowexact_body(EXL3_GEMM_ARGS)
{
    auto grid = cg::this_grid();
    const int warps_grid = gridDim.x * blockDim.x / 32;
    const int first_warp = threadIdx.x / 32 + blockDim.x / 32 * blockIdx.x;

    for (int row = 0; row < size_m; ++row)
    {
        for (int warp = first_warp; warp < size_k / 128; warp += warps_grid)
            had_hf_r_128_inner<true, false>
            (
                A + row * size_k + warp * 128,
                A_had + row * size_k + warp * 128,
                suh + warp * 128,
                0.088388347648f
            );

        grid.sync();
        exl3_gemm_kernel_inner
        <bits, false, 2, TILESIZE_M, TILESIZE_K, TILESIZE_N, SH_STAGES, FRAG_STAGES, true>
        (
            A_had + row * size_k,
            B,
            ((half*) C) + row * size_n,
            1,
            size_k,
            size_n,
            locks,
            svh
        );
        grid.sync();
    }
}

#define GLM53_EXL3_ROWEXACT_INSTANCE(NAME, BITS, TILE_M, TILE_K, TILE_N, SH, FRAG) \
    extern "C" __global__ __launch_bounds__(EXL3_GEMM_BASE_THREADS * TILE_K / 16) \
    void NAME(EXL3_GEMM_ARGS) \
    { \
        glm53_exl3_rowexact_body<BITS, TILE_M, TILE_K, TILE_N, SH, FRAG> \
        (A, B, C, size_m, size_k, size_n, locks, suh, A_had, svh); \
    }

#define GLM53_EXL3_ROWEXACT_INSTANCES(BITS) \
    GLM53_EXL3_ROWEXACT_INSTANCE(glm53_exl3_rowexact_k##BITS##_s1, BITS, 16, 16, 128, 6, 5) \
    GLM53_EXL3_ROWEXACT_INSTANCE(glm53_exl3_rowexact_k##BITS##_s2, BITS, 16, 32, 128, 4, 3) \
    GLM53_EXL3_ROWEXACT_INSTANCE(glm53_exl3_rowexact_k##BITS##_s3, BITS, 16, 32, 256, 4, 3) \
    GLM53_EXL3_ROWEXACT_INSTANCE(glm53_exl3_rowexact_k##BITS##_s4, BITS, 16, 16, 512, 4, 3)
