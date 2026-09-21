// SPDX-License-Identifier: AGPL-3.0-only

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

namespace {

constexpr uint32_t kQkvVectors = 1280;
constexpr uint32_t kQVectors = 256;
constexpr uint32_t kKVectors = 256;
constexpr uint32_t kVVectors = 768;
constexpr uint32_t kHeads = 48;
constexpr uint32_t kDim = 128;
constexpr uint32_t kStateTiles = 16;
constexpr uint32_t kStateBlocks = kHeads * kStateTiles;

struct Region {
    const void* pointer;
    uint64_t bytes;
};

__device__ __forceinline__ float clamp_log(float value) {
    const float clamped = value < 1.0e-30f ? 1.0e-30f : value;
    return logf(clamped);
}

__global__ __launch_bounds__(256) void atlas_gdn_c143_pack_inputs(
    const uint4* __restrict__ atlas_qkv,
    const float* __restrict__ atlas_gate_beta,
    const float* __restrict__ atlas_state_hkv,
    uint4* __restrict__ q,
    uint4* __restrict__ k,
    uint4* __restrict__ v,
    float* __restrict__ log_gate,
    float* __restrict__ beta,
    float* __restrict__ state_hvk,
    uint32_t m) {
    const uint32_t task = blockIdx.x;
    const uint32_t lane = threadIdx.x;
    if (task < m) {
        const uint64_t source_row = static_cast<uint64_t>(task) * kQkvVectors;
        for (uint32_t vector = lane; vector < kQkvVectors; vector += blockDim.x) {
            const uint4 value = atlas_qkv[source_row + vector];
            if (vector < kQVectors) {
                q[static_cast<uint64_t>(task) * kQVectors + vector] = value;
            } else if (vector < kQVectors + kKVectors) {
                k[static_cast<uint64_t>(task) * kKVectors + vector - kQVectors] = value;
            } else {
                v[static_cast<uint64_t>(task) * kVVectors + vector - kQVectors - kKVectors] = value;
            }
        }
        if (lane < kHeads) {
            const uint64_t gate_row = static_cast<uint64_t>(task) * 96U;
            const uint64_t output_row = static_cast<uint64_t>(task) * kHeads;
            log_gate[output_row + lane] = clamp_log(atlas_gate_beta[gate_row + lane]);
            beta[output_row + lane] = atlas_gate_beta[gate_row + kHeads + lane];
        }
        return;
    }

    __shared__ float tile[32][33];
    const uint32_t state_task = task - m;
    const uint32_t head = state_task / kStateTiles;
    const uint32_t tile_index = state_task % kStateTiles;
    const uint32_t tile_k = (tile_index / 4U) * 32U;
    const uint32_t tile_v = (tile_index % 4U) * 32U;
    const uint32_t x = lane & 31U;
    const uint32_t y = lane >> 5U;
    for (uint32_t offset = 0; offset < 32U; offset += 8U) {
        const uint32_t key = tile_k + y + offset;
        const uint32_t value = tile_v + x;
        tile[y + offset][x] = atlas_state_hkv[(head * kDim + key) * kDim + value];
    }
    __syncthreads();
    for (uint32_t offset = 0; offset < 32U; offset += 8U) {
        const uint32_t value = tile_v + y + offset;
        const uint32_t key = tile_k + x;
        state_hvk[(head * kDim + value) * kDim + key] = tile[x][y + offset];
    }
}

bool valid_region(const Region& region) {
    if (region.pointer == nullptr || region.bytes == 0U) return false;
    const uintptr_t begin = reinterpret_cast<uintptr_t>(region.pointer);
    return (begin & 15U) == 0U && begin <= UINTPTR_MAX - region.bytes;
}

bool overlaps(const Region& left, const Region& right) {
    const uintptr_t l = reinterpret_cast<uintptr_t>(left.pointer);
    const uintptr_t r = reinterpret_cast<uintptr_t>(right.pointer);
    return l < r + right.bytes && r < l + left.bytes;
}

}  // namespace

extern "C" int atlas_gdn_c143_pack_inputs_launch(
    const void* atlas_qkv, uint64_t atlas_qkv_bytes,
    const void* atlas_gate_beta, uint64_t atlas_gate_beta_bytes,
    const void* atlas_state_hkv, uint64_t atlas_state_hkv_bytes,
    void* q, uint64_t q_bytes,
    void* k, uint64_t k_bytes,
    void* v, uint64_t v_bytes,
    void* log_gate, uint64_t log_gate_bytes,
    void* beta, uint64_t beta_bytes,
    void* state_hvk, uint64_t state_hvk_bytes,
    uint32_t m, cudaStream_t stream) {
    if (m != 2079U && m != 8192U) return 1;
    if (stream == nullptr) return 2;
    const uint64_t state_bytes = 48ULL * 128ULL * 128ULL * 4ULL;
    const Region regions[] = {
        {atlas_qkv, static_cast<uint64_t>(m) * 10240ULL * 2ULL},
        {atlas_gate_beta, static_cast<uint64_t>(m) * 96ULL * 4ULL},
        {atlas_state_hkv, state_bytes},
        {q, static_cast<uint64_t>(m) * 2048ULL * 2ULL},
        {k, static_cast<uint64_t>(m) * 2048ULL * 2ULL},
        {v, static_cast<uint64_t>(m) * 6144ULL * 2ULL},
        {log_gate, static_cast<uint64_t>(m) * 48ULL * 4ULL},
        {beta, static_cast<uint64_t>(m) * 48ULL * 4ULL},
        {state_hvk, state_bytes},
    };
    const uint64_t supplied[] = {
        atlas_qkv_bytes, atlas_gate_beta_bytes, atlas_state_hkv_bytes,
        q_bytes, k_bytes, v_bytes, log_gate_bytes, beta_bytes, state_hvk_bytes,
    };
    for (uint32_t i = 0; i < 9U; ++i) {
        if (!valid_region(regions[i]) || supplied[i] != regions[i].bytes) return 3;
        for (uint32_t j = 0; j < i; ++j) {
            if (overlaps(regions[i], regions[j])) return 4;
        }
    }
    (void)cudaGetLastError();
    atlas_gdn_c143_pack_inputs<<<m + kStateBlocks, 256, 0, stream>>>(
        static_cast<const uint4*>(atlas_qkv),
        static_cast<const float*>(atlas_gate_beta),
        static_cast<const float*>(atlas_state_hkv),
        static_cast<uint4*>(q), static_cast<uint4*>(k), static_cast<uint4*>(v),
        static_cast<float*>(log_gate), static_cast<float*>(beta),
        static_cast<float*>(state_hvk), m);
    return cudaPeekAtLastError() == cudaSuccess ? 0 : 5;
}
