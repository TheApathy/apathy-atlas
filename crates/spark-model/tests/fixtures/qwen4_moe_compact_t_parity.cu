// SPDX-License-Identifier: AGPL-3.0-only
// Standalone synthetic raw-parity gate; never registered in KERNEL.toml.
#include <cuda_runtime.h>

#define ATLAS_QWEN4_MOE_COMPACT_CONTRACT 1
#define OI640_STEP_K 32
#include "../../../../kernels/gb10/qwen3.8-flash-next/nvfp4/moe_w4a16_orig_i640_compact_prefill.cuh"
namespace oi640 = flash_next_orig_i640_prefill;
#define moe_w4a16_orig_i640_compact_prefill_plan parity_plan
#include "../../../../kernels/gb10/qwen3.8-flash-next/nvfp4/moe_w4a16_orig_i640_compact_prefill_plan.cuh"
#undef moe_w4a16_orig_i640_compact_prefill_plan
#define OI640_E2M1_LUT parity_original_lut
#define moe_w4a16_orig_i640_compact_prefill_gemm parity_original
#include "../../../../kernels/gb10/qwen3.8-flash-next/nvfp4/moe_w4a16_orig_i640_compact_prefill_gemm.cuh"
#undef moe_w4a16_orig_i640_compact_prefill_gemm
#undef OI640_E2M1_LUT

#include "../../../../kernels/gb10/common/moe_transpose_batched.cu"
#undef OI640_TRANSPOSED
#define OI640_TRANSPOSED 1
#define OI640_E2M1_LUT parity_transposed_lut
#define moe_w4a16_orig_i640_compact_prefill_gemm parity_transposed
#include "../../../../kernels/gb10/qwen3.8-flash-next/nvfp4/moe_w4a16_orig_i640_compact_prefill_gemm.cuh"
#undef moe_w4a16_orig_i640_compact_prefill_gemm
#undef OI640_E2M1_LUT

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_OK(expr) do { const cudaError_t e_ = (expr); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
    std::exit(2); } } while (0)

template <typename T> struct Dev {
    T* p = nullptr;
    explicit Dev(size_t n) { CUDA_OK(cudaMalloc(&p, n * sizeof(T))); }
    ~Dev() { if (p) cudaFree(p); }
};

std::vector<unsigned char> bytes(size_t n, unsigned salt) {
    std::vector<unsigned char> out(n);
    unsigned x = 0x9e3779b9U ^ salt;
    for (auto& value : out) {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        value = static_cast<unsigned char>(x);
    }
    return out;
}

bool shape(unsigned rows, unsigned n, unsigned k) {
    const size_t packed_cols = k / 2, scale_cols = k / oi640::GROUP_K;
    auto packed = bytes(static_cast<size_t>(n) * packed_cols, n + k);
    auto scales = bytes(static_cast<size_t>(n) * scale_cols, n ^ k);
    for (auto& value : scales) value = static_cast<unsigned char>(1 + value % 125);
    const unsigned expanded = rows * oi640::TOP_K;
    const unsigned input_rows = k == oi640::HIDDEN ? rows : expanded;
    std::vector<__nv_bfloat16> a(static_cast<size_t>(input_rows) * k);
    for (size_t i = 0; i < a.size(); ++i)
        a[i] = __float2bfloat16(float(int((i * 29 + 7) % 257) - 128) / 256.0f);
    std::vector<int> offsets(oi640::EXPERTS + 1, expanded), tokens(expanded);
    offsets[0] = 0;
    for (unsigned i = 0; i < expanded; ++i) tokens[i] = i % rows;
    std::vector<unsigned long long> packed_ptrs(oi640::EXPERTS);
    std::vector<unsigned long long> scales_ptrs(oi640::EXPERTS);
    std::vector<float> scale2(oi640::EXPERTS, 0.0078125f);
    constexpr size_t guard = 32;
    Dev<unsigned char> d_p(packed.size()), d_s(scales.size());
    Dev<unsigned char> d_pt(packed.size() * oi640::EXPERTS + 2 * guard);
    Dev<unsigned char> d_st(scales.size() * oi640::EXPERTS + 2 * guard);
    Dev<__nv_bfloat16> d_a(a.size()), d_o(static_cast<size_t>(expanded) * n), d_t(static_cast<size_t>(expanded) * n);
    Dev<int> d_offsets(offsets.size()), d_tokens(tokens.size()), d_status(1);
    Dev<unsigned long long> d_pp(oi640::EXPERTS), d_sp(oi640::EXPERTS), d_ppt(oi640::EXPERTS), d_spt(oi640::EXPERTS);
    Dev<float> d_s2(oi640::EXPERTS);
    Dev<oi640::Contract> d_contract(1);
    Dev<oi640::Workspace> d_workspace(1);
    CUDA_OK(cudaMemcpy(d_p.p, packed.data(), packed.size(), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_s.p, scales.data(), scales.size(), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemset(d_pt.p, 0xa5, packed.size() * oi640::EXPERTS + 2 * guard));
    CUDA_OK(cudaMemset(d_st.p, 0xa5, scales.size() * oi640::EXPERTS + 2 * guard));
    std::fill(packed_ptrs.begin(), packed_ptrs.end(), reinterpret_cast<unsigned long long>(d_p.p));
    std::fill(scales_ptrs.begin(), scales_ptrs.end(), reinterpret_cast<unsigned long long>(d_s.p));
    CUDA_OK(cudaMemcpy(d_pp.p, packed_ptrs.data(), packed_ptrs.size() * 8, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_sp.p, scales_ptrs.data(), scales_ptrs.size() * 8, cudaMemcpyHostToDevice));
    for (unsigned expert = 0; expert < oi640::EXPERTS; ++expert) {
        packed_ptrs[expert] = reinterpret_cast<unsigned long long>(d_pt.p + guard + expert * packed.size());
        scales_ptrs[expert] = reinterpret_cast<unsigned long long>(d_st.p + guard + expert * scales.size());
    }
    CUDA_OK(cudaMemcpy(d_ppt.p, packed_ptrs.data(), packed_ptrs.size() * 8, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_spt.p, scales_ptrs.data(), scales_ptrs.size() * 8, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_s2.p, scale2.data(), scale2.size() * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_a.p, a.data(), a.size() * 2, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_offsets.p, offsets.data(), offsets.size() * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(d_tokens.p, tokens.data(), tokens.size() * 4, cudaMemcpyHostToDevice));
    const oi640::Contract contract{oi640::MAGIC, oi640::VERSION, rows, 2560, 640, 512, 10,
        expanded, 64, 64, oi640::FIXED_GRID, oi640::MAX_WORK_ITEMS, sizeof(oi640::Workspace), 0, 0};
    CUDA_OK(cudaMemcpy(d_contract.p, &contract, sizeof(contract), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemset(d_status.p, 0xff, sizeof(int)));
    const dim3 transpose_block(32, 8, 1);
    const dim3 packed_grid((packed_cols + 31) / 32, (n + 31) / 32, oi640::EXPERTS);
    const dim3 scale_grid((scale_cols + 31) / 32, (n + 31) / 32, oi640::EXPERTS);
    moe_transpose_u8_batched<<<packed_grid, transpose_block>>>(d_pp.p, d_ppt.p, n, packed_cols);
    moe_transpose_u8_batched<<<scale_grid, transpose_block>>>(d_sp.p, d_spt.p, n, scale_cols);
    parity_plan<<<1, 512>>>(d_offsets.p, d_tokens.p, d_contract.p, d_workspace.p, d_status.p);
    parity_original<<<oi640::FIXED_GRID, 128>>>(d_a.p, d_pp.p, d_sp.p, d_s2.p, d_o.p,
        d_offsets.p, d_tokens.p, d_contract.p, d_workspace.p, d_status.p, n, k);
    parity_transposed<<<oi640::FIXED_GRID, 128>>>(d_a.p, d_ppt.p, d_spt.p, d_s2.p, d_t.p,
        d_offsets.p, d_tokens.p, d_contract.p, d_workspace.p, d_status.p, n, k);
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<unsigned char> original(static_cast<size_t>(expanded) * n * 2), transposed(original.size());
    CUDA_OK(cudaMemcpy(original.data(), d_o.p, original.size(), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(transposed.data(), d_t.p, transposed.size(), cudaMemcpyDeviceToHost));
    std::vector<unsigned char> source_p(packed.size()), source_s(scales.size());
    std::vector<unsigned char> guard_p(2 * guard), guard_s(2 * guard);
    CUDA_OK(cudaMemcpy(source_p.data(), d_p.p, source_p.size(), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(source_s.data(), d_s.p, source_s.size(), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(guard_p.data(), d_pt.p, guard, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(guard_p.data() + guard,
        d_pt.p + guard + packed.size() * oi640::EXPERTS, guard, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(guard_s.data(), d_st.p, guard, cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(guard_s.data() + guard,
        d_st.p + guard + scales.size() * oi640::EXPERTS, guard, cudaMemcpyDeviceToHost));
    const bool guards = std::all_of(guard_p.begin(), guard_p.end(), [](unsigned char v) { return v == 0xa5; })
        && std::all_of(guard_s.begin(), guard_s.end(), [](unsigned char v) { return v == 0xa5; });
    const bool equal = original == transposed && source_p == packed && source_s == scales && guards;
    std::printf("rows=%u n=%u k=%u exact=%s source=%s guards=%s\n", rows, n, k,
        original == transposed ? "true" : "false",
        source_p == packed && source_s == scales ? "true" : "false", guards ? "true" : "false");
    return equal;
}

int main() {
    bool ok = true;
    for (unsigned rows : {2U, 17U, 64U, 65U}) {
        ok &= shape(rows, 640, 2560);
        ok &= shape(rows, 2560, 640);
    }
    std::printf("FLASHNEXT_F41_STREAM_RAW_PARITY cases=8 pass=%s\n", ok ? "true" : "false");
    return ok ? 0 : 1;
}
