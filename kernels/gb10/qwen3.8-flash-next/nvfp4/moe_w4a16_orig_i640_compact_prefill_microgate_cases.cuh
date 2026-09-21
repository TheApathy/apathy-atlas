// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

namespace oi640_gate {

__global__ void oi640_silu_mul(const __nv_bfloat16* gate, const __nv_bfloat16* up,
                               __nv_bfloat16* output, size_t count) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const float g = __bfloat162float(gate[i]);
        output[i] = __float2bfloat16((g / (1.0f + expf(-g))) * __bfloat162float(up[i]));
    }
}

struct ShapeCase {
    uint32_t rows;
    uint32_t expanded;
    uint32_t max_rows = 0;
    std::vector<int> host_offsets;
    Guarded<__nv_bfloat16> input, parent_gate, parent_up, parent_act, parent_down;
    Guarded<__nv_bfloat16> candidate_gate, candidate_up, candidate_act, candidate_down;
    Guarded<int> offsets, tokens, status;
    Guarded<oi640::Contract> contract;
    Guarded<oi640::Workspace> workspace;
    uint32_t work_count = 0, active_experts = 0;
    uint64_t census_hash = 0, output_hash = 0;

    ShapeCase(uint32_t m, const std::vector<int>& captured) : rows(m), expanded(m * oi640::TOP_K),
        host_offsets(captured) {
        for (size_t i = 1; i < captured.size(); ++i)
            max_rows = std::max(max_rows, static_cast<uint32_t>(captured[i] - captured[i - 1]));
        offsets.allocate(captured.size(), true); offsets.upload(captured);
        std::vector<int> host_tokens(expanded);
        for (uint32_t expert = 0; expert < oi640::EXPERTS; ++expert)
            for (int i = captured[expert]; i < captured[expert + 1]; ++i)
                host_tokens[i] = static_cast<int>((uint64_t(i) * 131 + expert * 17) % rows);
        tokens.allocate(expanded, true); tokens.upload(host_tokens);
        std::vector<__nv_bfloat16> host_input(static_cast<size_t>(rows) * oi640::HIDDEN);
        for (size_t i = 0; i < host_input.size(); ++i)
            host_input[i] = __float2bfloat16(float(int((i * 29 + 7) % 257) - 128) / 256.0f);
        input.allocate(host_input.size(), true); input.upload(host_input);
        const oi640::Contract c{oi640::MAGIC, oi640::VERSION, rows, oi640::HIDDEN,
            oi640::INTER, oi640::EXPERTS, oi640::TOP_K, expanded, oi640::TILE_M,
            oi640::TILE_N, oi640::FIXED_GRID, oi640::MAX_WORK_ITEMS,
            sizeof(oi640::Workspace), 0, 0};
        contract.allocate(1, true); contract.upload({c});
        workspace.allocate(1); status.allocate(1);
        const size_t gi = static_cast<size_t>(expanded) * oi640::INTER;
        const size_t dh = static_cast<size_t>(expanded) * oi640::HIDDEN;
        parent_gate.allocate(gi); parent_up.allocate(gi); parent_act.allocate(gi);
        candidate_gate.allocate(gi); candidate_up.allocate(gi); candidate_act.allocate(gi);
        parent_down.allocate(dh); candidate_down.allocate(dh);
    }

    void reset(Guarded<__nv_bfloat16>& a, Guarded<__nv_bfloat16>& b,
               Guarded<__nv_bfloat16>& c, Guarded<__nv_bfloat16>& d, cudaStream_t stream) {
        a.fill_async(0xcd, stream); b.fill_async(0xcd, stream);
        c.fill_async(0xcd, stream); d.fill_async(0xcd, stream);
    }

    void launch_parent(Weights& w, cudaStream_t stream) {
        reset(parent_gate, parent_up, parent_act, parent_down, stream);
        const dim3 gu((oi640::INTER + 63) / 64, (max_rows + 63) / 64, oi640::EXPERTS);
        const dim3 dn((oi640::HIDDEN + 63) / 64, (max_rows + 63) / 64, oi640::EXPERTS);
        moe_w4a16_grouped_gemm_ptrtable<<<gu, 128, 0, stream>>>(input.data, w.gate_pt.data,
            w.gate_st.data, w.gate_s2.data, parent_gate.data, offsets.data, tokens.data,
            oi640::EXPERTS, oi640::INTER, oi640::HIDDEN);
        moe_w4a16_grouped_gemm_ptrtable<<<gu, 128, 0, stream>>>(input.data, w.up_pt.data,
            w.up_st.data, w.up_s2.data, parent_up.data, offsets.data, tokens.data,
            oi640::EXPERTS, oi640::INTER, oi640::HIDDEN);
        oi640_silu_mul<<<oi640::FIXED_GRID, 256, 0, stream>>>(parent_gate.data, parent_up.data,
            parent_act.data, static_cast<size_t>(expanded) * oi640::INTER);
        moe_w4a16_grouped_gemm_ptrtable<<<dn, 128, 0, stream>>>(parent_act.data, w.down_pt.data,
            w.down_st.data, w.down_s2.data, parent_down.data, offsets.data, nullptr,
            oi640::EXPERTS, oi640::HIDDEN, oi640::INTER);
    }

    void launch_candidate(Weights& w, cudaStream_t stream) {
        reset(candidate_gate, candidate_up, candidate_act, candidate_down, stream);
        CUDA_OK(cudaMemsetAsync(status.data, 0xff, sizeof(int), stream));
        moe_w4a16_orig_i640_compact_prefill_plan<<<1, oi640::EXPERTS, 0, stream>>>(
            offsets.data, tokens.data, contract.data, workspace.data, status.data);
        moe_w4a16_orig_i640_compact_prefill_gemm<<<oi640::FIXED_GRID, 128, 0, stream>>>(
            input.data, w.gate_pt.data, w.gate_st.data, w.gate_s2.data, candidate_gate.data,
            offsets.data, tokens.data, contract.data, workspace.data, status.data,
            oi640::INTER, oi640::HIDDEN);
        moe_w4a16_orig_i640_compact_prefill_gemm<<<oi640::FIXED_GRID, 128, 0, stream>>>(
            input.data, w.up_pt.data, w.up_st.data, w.up_s2.data, candidate_up.data,
            offsets.data, tokens.data, contract.data, workspace.data, status.data,
            oi640::INTER, oi640::HIDDEN);
        oi640_silu_mul<<<oi640::FIXED_GRID, 256, 0, stream>>>(candidate_gate.data,
            candidate_up.data, candidate_act.data, static_cast<size_t>(expanded) * oi640::INTER);
        moe_w4a16_orig_i640_compact_prefill_gemm<<<oi640::FIXED_GRID, 128, 0, stream>>>(
            candidate_act.data, w.down_pt.data, w.down_st.data, w.down_s2.data,
            candidate_down.data, offsets.data, tokens.data, contract.data, workspace.data,
            status.data, oi640::HIDDEN, oi640::INTER);
    }

    template <typename T> bool exact(const Guarded<T>& a, const Guarded<T>& b) const {
        const auto x = a.download(), y = b.download();
        return x.size() == y.size() && std::memcmp(x.data(), y.data(), x.size() * sizeof(T)) == 0;
    }
    template <typename T> uint64_t digest(const Guarded<T>& buffer) const {
        const auto host = buffer.download();
        uint64_t h = 1469598103934665603ULL;
        const auto* bytes = reinterpret_cast<const unsigned char*>(host.data());
        for (size_t i = 0; i < host.size() * sizeof(T); ++i)
            h = (h ^ bytes[i]) * 1099511628211ULL;
        return h;
    }
    bool guards() const {
        return input.safe() && offsets.safe() && tokens.safe() && contract.safe() && workspace.safe() &&
            status.safe() && parent_gate.safe() && parent_up.safe() && parent_act.safe() &&
            parent_down.safe() && candidate_gate.safe() && candidate_up.safe() &&
            candidate_act.safe() && candidate_down.safe();
    }
    bool correctness(Weights& w, cudaStream_t stream) {
        launch_parent(w, stream); launch_candidate(w, stream);
        CUDA_OK(cudaStreamSynchronize(stream)); CUDA_OK(cudaGetLastError());
        const auto plan = workspace.download()[0];
        bool ok = status.download()[0] == oi640::PLANNED && plan.ready_magic == oi640::MAGIC &&
            plan.version == oi640::VERSION && plan.rows == rows && plan.expanded == expanded &&
            plan.work_count > 0 && plan.work_count <= oi640::MAX_WORK_ITEMS &&
            exact(parent_gate, candidate_gate) && exact(parent_up, candidate_up) &&
            exact(parent_act, candidate_act) && exact(parent_down, candidate_down);
        work_count = plan.work_count; active_experts = plan.active_experts;
        census_hash = plan.census_hash; output_hash = digest(candidate_down);
        const auto first_gate = candidate_gate.download();
        const auto first_up = candidate_up.download();
        const auto first_down = candidate_down.download();
        launch_candidate(w, stream); CUDA_OK(cudaStreamSynchronize(stream));
        ok &= status.download()[0] == oi640::PLANNED &&
            std::memcmp(first_gate.data(), candidate_gate.download().data(), first_gate.size()*sizeof(__nv_bfloat16)) == 0 &&
            std::memcmp(first_up.data(), candidate_up.download().data(), first_up.size()*sizeof(__nv_bfloat16)) == 0 &&
            std::memcmp(first_down.data(), candidate_down.download().data(), first_down.size()*sizeof(__nv_bfloat16)) == 0 &&
            guards() && w.safe();
        return ok;
    }
    bool hostile(cudaStream_t stream) {
        auto c = contract.download()[0];
        c.magic ^= 1; contract.upload({c}, false);
        CUDA_OK(cudaMemsetAsync(status.data, 0xff, sizeof(int), stream));
        moe_w4a16_orig_i640_compact_prefill_plan<<<1, oi640::EXPERTS, 0, stream>>>(
            offsets.data, tokens.data, contract.data, workspace.data, status.data);
        CUDA_OK(cudaStreamSynchronize(stream));
        bool ok = status.download()[0] == oi640::ERR_ABI;
        c.magic ^= 1; contract.upload({c}, false);
        const auto good_tokens = tokens.download(); auto bad_tokens = good_tokens;
        bad_tokens[17] = rows; tokens.upload(bad_tokens, false);
        CUDA_OK(cudaMemsetAsync(status.data, 0xff, sizeof(int), stream));
        moe_w4a16_orig_i640_compact_prefill_plan<<<1, oi640::EXPERTS, 0, stream>>>(
            offsets.data, tokens.data, contract.data, workspace.data, status.data);
        CUDA_OK(cudaStreamSynchronize(stream)); ok &= status.download()[0] == oi640::ERR_TOKEN;
        tokens.upload(good_tokens, false);
        return ok && guards();
    }
};

} // namespace oi640_gate
