// SPDX-License-Identifier: AGPL-3.0-only
//
// Focused DSpark attention parity fixture.
// Build/run:
//   nvcc -O3 -std=c++17 -arch=sm_121a \
//     tests/cuda/dspark_asymmetric_attention_parity.cu -o /tmp/dspark-attn-parity
//   /tmp/dspark-attn-parity

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../../kernels/gb10/common/inferspark_prefill_h128.cu"

#define CUDA_OK(expr)                                                          \
    do {                                                                       \
        cudaError_t err_ = (expr);                                             \
        if (err_ != cudaSuccess) {                                             \
            std::fprintf(stderr, "%s:%d CUDA error: %s\n", __FILE__, __LINE__, \
                         cudaGetErrorString(err_));                             \
            std::exit(2);                                                      \
        }                                                                      \
    } while (0)

static uint16_t bf16_rne(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>(bits >> 16);
}

static void fill_bf16(std::vector<uint16_t>& out, uint32_t seed) {
    uint32_t state = seed;
    for (uint16_t& value : out) {
        state = state * 1664525u + 1013904223u;
        const int32_t centered = static_cast<int32_t>((state >> 8) & 0xffffu) - 32768;
        value = bf16_rne(static_cast<float>(centered) / 32768.0f);
    }
}

static bool run_case(uint32_t eff_ctx) {
    constexpr uint32_t drafts = 14;
    constexpr uint32_t q_heads = 40;
    constexpr uint32_t kv_heads = 8;
    constexpr uint32_t head_dim = 128;
    constexpr uint32_t br = 32;
    const uint32_t seq_len = eff_ctx + drafts;
    const uint32_t query_start = eff_ctx - eff_ctx % br;
    const uint32_t query_len = seq_len - query_start;
    const size_t q_elems = static_cast<size_t>(seq_len) * q_heads * head_dim;
    const size_t kv_elems = static_cast<size_t>(seq_len) * kv_heads * head_dim;

    std::vector<uint16_t> h_q(q_elems), h_k(kv_elems), h_v(kv_elems);
    std::vector<uint16_t> h_full(q_elems, 0xa5a5u), h_tail(q_elems, 0x5a5au);
    fill_bf16(h_q, 1u + eff_ctx);
    fill_bf16(h_k, 2u + eff_ctx);
    fill_bf16(h_v, 3u + eff_ctx);

    __nv_bfloat16 *d_q, *d_k, *d_v, *d_full, *d_tail;
    CUDA_OK(cudaMalloc(&d_q, q_elems * sizeof(uint16_t)));
    CUDA_OK(cudaMalloc(&d_k, kv_elems * sizeof(uint16_t)));
    CUDA_OK(cudaMalloc(&d_v, kv_elems * sizeof(uint16_t)));
    CUDA_OK(cudaMalloc(&d_full, q_elems * sizeof(uint16_t)));
    CUDA_OK(cudaMalloc(&d_tail, q_elems * sizeof(uint16_t)));
    CUDA_OK(cudaMemcpy(d_q, h_q.data(), q_elems * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_k, h_k.data(), kv_elems * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_v, h_v.data(), kv_elems * sizeof(uint16_t), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemset(d_full, 0xa5, q_elems * sizeof(uint16_t)));
    CUDA_OK(cudaMemset(d_tail, 0x5a, q_elems * sizeof(uint16_t)));

    dim3 full_grid(q_heads, (seq_len + br - 1) / br, 1);
    dim3 tail_grid(q_heads, (query_len + br - 1) / br, 1);
    dim3 block(128, 1, 1);
    inferspark_prefill_h128<<<full_grid, block>>>(
        d_q, d_k, d_v, d_full, seq_len, 0, seq_len, q_heads, kv_heads,
        head_dim, 1.0f / 11.313708498984761f, 0, 0);
    CUDA_OK(cudaGetLastError());
    inferspark_prefill_h128<<<tail_grid, block>>>(
        d_q, d_k, d_v, d_tail, seq_len, query_start, query_len, q_heads,
        kv_heads, head_dim, 1.0f / 11.313708498984761f, 0, 0);
    CUDA_OK(cudaGetLastError());
    CUDA_OK(cudaDeviceSynchronize());
    CUDA_OK(cudaMemcpy(h_full.data(), d_full, q_elems * sizeof(uint16_t), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(h_tail.data(), d_tail, q_elems * sizeof(uint16_t), cudaMemcpyDeviceToHost));

    size_t mismatches = 0;
    size_t first = 0;
    const size_t row_elems = static_cast<size_t>(q_heads) * head_dim;
    for (size_t i = static_cast<size_t>(eff_ctx) * row_elems; i < q_elems; ++i) {
        if (h_full[i] != h_tail[i]) {
            if (mismatches == 0) first = i;
            ++mismatches;
        }
    }
    std::printf("eff_ctx=%u seq=%u query=[%u..%u) full_ctas=%u tail_ctas=%u mismatches=%zu",
                eff_ctx, seq_len, query_start, query_start + query_len,
                full_grid.y * q_heads, tail_grid.y * q_heads, mismatches);
    if (mismatches) {
        std::printf(" first_element=%zu full=0x%04x tail=0x%04x\n", first,
                    h_full[first], h_tail[first]);
    } else {
        std::printf(" PASS\n");
    }

    CUDA_OK(cudaFree(d_q));
    CUDA_OK(cudaFree(d_k));
    CUDA_OK(cudaFree(d_v));
    CUDA_OK(cudaFree(d_full));
    CUDA_OK(cudaFree(d_tail));
    return mismatches == 0;
}

int main() {
    const bool aligned = run_case(512);
    const bool unaligned = run_case(513);
    return aligned && unaligned ? 0 : 1;
}
