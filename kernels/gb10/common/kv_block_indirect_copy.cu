// SPDX-License-Identifier: AGPL-3.0-only

// DDTree M5 — indirect KV block copy for the CUDA-graphed tree verify.
//
// Replaces the per-layer host-driven canonical→scratch re-seed d2d memcpys
// (which take per-step canonical block POINTERS and therefore cannot be
// captured in a CUDA graph) with a copy whose source/destination block IDs
// are read from a small device buffer uploaded pre-replay:
//
//   meta[0]            = n_pairs (may vary across graph replays)
//   meta[1 + 2*i]      = src block id (canonical physical block)
//   meta[2 + 2*i]      = dst block id (persistent scratch block)
//
// Launch shape (all baked at graph capture):
//   grid  = (chunks, MAX_PAIRS, 2)   — z: 0 = K pool, 1 = V pool
//   block = (256, 1, 1)
//
// Pairs at blockIdx.y >= meta[0] exit immediately, so a graph captured for
// MAX_PAIRS replays correctly for any n_pairs <= MAX_PAIRS. Copies are
// 16-byte vectorized; the host asserts k_bytes/v_bytes % 16 == 0 (KV blocks
// are head_dim-strided, always 16B-aligned).

#include <cstdint>

extern "C" __global__ void kv_block_indirect_copy(
    unsigned char* __restrict__ k_pool,
    unsigned char* __restrict__ v_pool,
    unsigned long long k_stride,   // pool block stride in bytes (K)
    unsigned long long v_stride,   // pool block stride in bytes (V)
    unsigned int k_bytes,          // bytes to copy per K block
    unsigned int v_bytes,          // bytes to copy per V block
    const unsigned int* __restrict__ meta
) {
    const unsigned int n_pairs = meta[0];
    const unsigned int pair = blockIdx.y;
    if (pair >= n_pairs) {
        return;
    }
    const unsigned int src = meta[1 + 2 * pair];
    const unsigned int dst = meta[2 + 2 * pair];

    const bool is_v = (blockIdx.z != 0);
    unsigned char* pool = is_v ? v_pool : k_pool;
    const unsigned long long stride = is_v ? v_stride : k_stride;
    const unsigned int bytes = is_v ? v_bytes : k_bytes;

    const uint4* s = reinterpret_cast<const uint4*>(
        pool + (unsigned long long)src * stride);
    uint4* d = reinterpret_cast<uint4*>(
        pool + (unsigned long long)dst * stride);

    const unsigned int n16 = bytes >> 4;  // 16B vectors
    const unsigned int step = gridDim.x * blockDim.x;
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < n16;
         i += step) {
        d[i] = s[i];
    }
}
