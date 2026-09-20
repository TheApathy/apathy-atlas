// SPDX-License-Identifier: AGPL-3.0-only

// GPU-ready raw gate. Compile/static checks never execute CUDA. Runtime use
// requires a separately reserved exact sm_121 GB10.

#define main flash_next_exact_k16_unused_main
#include "moe_w4a16_exact_k16_microgate.cu"
#undef main
#define FLASH_NEXT_EXACT_K16_PARENT_INCLUDED 1
#include "moe_w4a16_routed_k16_persistent.cu"

namespace routed_gate {

using namespace flash_next_exact_k16;
using namespace flash_next_routed_k16;
constexpr unsigned int BLOCKS = 96; // two guaranteed-resident CTAs x 48 SMs

struct RoutedState {
    Guarded<RoutedContract> contract;
    Guarded<RoutedWorkspace> workspace, repeat_workspace;
    RoutedState() {
        RoutedContract c{MAGIC, VERSION, ROWS, HIDDEN, INTER, EXPERTS, TOP_K, 0,
            static_cast<unsigned long long>(ROWS) * HIDDEN,
            static_cast<unsigned long long>(ROUTES) * INTER,
            static_cast<unsigned long long>(ROUTES) * HIDDEN,
            ROUTES, EXPERTS, EXPERTS, sizeof(RoutedWorkspace)};
        contract.allocate(1, true); contract.upload({c});
        workspace.allocate(1); repeat_workspace.allocate(1);
        workspace.fill(OUTPUT_BYTE); repeat_workspace.fill(OUTPUT_BYTE);
    }
};

std::vector<unsigned int> route_case(unsigned int divisor) {
    std::vector<unsigned int> ids(ROUTES);
    for (unsigned int i = 0; i < ROUTES; ++i) ids[i] = divisor == 0 ? 7 : i % divisor;
    return ids;
}

void preflight(Fixture& f, Outputs& out, RoutedState& state, RoutedWorkspace* workspace,
               const RoutedContract* contract = nullptr, int* status = nullptr) {
    moe_w4a16_routed_k16_preflight<<<1, 1, 0, f.stream>>>(
        f.input.data, f.weights.gate_packed_table.data, f.weights.gate_scale_table.data,
        f.weights.gate_s2.data, out.gate.data, f.weights.up_packed_table.data,
        f.weights.up_scale_table.data, f.weights.up_s2.data, out.up.data,
        f.weights.down_packed_table.data, f.weights.down_scale_table.data,
        f.weights.down_s2.data, out.down.data, f.indices.data,
        contract == nullptr ? state.contract.data : contract, workspace,
        status == nullptr ? f.status.data : status);
}

void launch_compute(Fixture& f, Outputs& out, RoutedWorkspace* workspace, bool finish) {
    moe_w4a16_routed_k16_gate_up<<<dim3(BLOCKS, 2, 1), 128, 0, f.stream>>>(
        f.input.data, f.weights.gate_packed_table.data, f.weights.gate_scale_table.data,
        f.weights.gate_s2.data, out.gate.data, f.weights.up_packed_table.data,
        f.weights.up_scale_table.data, f.weights.up_s2.data, out.up.data, workspace, f.status.data);
    moe_w4a16_routed_k16_silu_down<<<BLOCKS, 128, 0, f.stream>>>(
        out.gate.data, out.up.data, f.weights.down_packed_table.data,
        f.weights.down_scale_table.data, f.weights.down_s2.data, out.down.data,
        workspace, f.status.data);
    if (finish) {
        moe_w4a16_routed_k16_finalize<<<1, 1, 0, f.stream>>>(workspace, f.status.data);
        moe_w4a16_exact_k16_weighted_sum_blend<<<dim3(HIDDEN / 256, ROWS, 1), 256, 0, f.stream>>>(
            out.final.data, out.down.data, f.expert_weights.data, out.shared_down.data,
            f.input.data, f.shared_gate_weight.data, f.contract.data, f.status.data);
    }
}

void launch_routed(Fixture& f, Outputs& out, RoutedState& state, RoutedWorkspace* workspace,
                   bool finish) {
    f.arm_status(); preflight(f, out, state, workspace);
    moe_w4a16_routed_k16_validate<<<1, 1, 0, f.stream>>>(workspace, f.indices.data, f.status.data);
    launch_compute(f, out, workspace, finish);
}

void launch_baseline(Fixture& f, Outputs& out) {
    moe_w4a16_routed_k16_baseline_gate_up<<<dim3(ROUTES * (INTER / 8), 2, 1), 128, 0, f.stream>>>(
        f.input.data, f.weights.gate_packed_table.data, f.weights.gate_scale_table.data,
        f.weights.gate_s2.data, out.gate.data, f.weights.up_packed_table.data,
        f.weights.up_scale_table.data, f.weights.up_s2.data, out.up.data, f.indices.data);
    moe_w4a16_routed_k16_baseline_down<<<ROUTES * (HIDDEN / 8), 128,
        INTER * sizeof(float), f.stream>>>(out.gate.data, out.up.data,
        f.weights.down_packed_table.data, f.weights.down_scale_table.data,
        f.weights.down_s2.data, out.down.data, f.indices.data);
}

bool fixture_case(Fixture& f, RoutedState& state, unsigned int divisor,
                  unsigned int expected_groups, const char* name) {
    const auto ids = route_case(divisor); f.indices.upload(ids); f.parent.reset();
    f.serial.reset(); f.candidate.reset(); f.candidate_repeat.reset();
    f.launch_parent(f.parent); f.launch_serial(f.serial);
    CUDA_OK(cudaMemcpyAsync(f.candidate.shared_gate.data, f.parent.shared_gate.data,
        f.parent.shared_gate.count * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, f.stream));
    CUDA_OK(cudaMemcpyAsync(f.candidate.shared_up.data, f.parent.shared_up.data,
        f.parent.shared_up.count * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, f.stream));
    CUDA_OK(cudaMemcpyAsync(f.candidate.shared_down.data, f.parent.shared_down.data,
        f.parent.shared_down.count * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, f.stream));
    CUDA_OK(cudaMemcpyAsync(f.candidate_repeat.shared_gate.data, f.parent.shared_gate.data,
        f.parent.shared_gate.count * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, f.stream));
    CUDA_OK(cudaMemcpyAsync(f.candidate_repeat.shared_up.data, f.parent.shared_up.data,
        f.parent.shared_up.count * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, f.stream));
    CUDA_OK(cudaMemcpyAsync(f.candidate_repeat.shared_down.data, f.parent.shared_down.data,
        f.parent.shared_down.count * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, f.stream));
    launch_routed(f, f.candidate, state, state.workspace.data, true);
    launch_routed(f, f.candidate_repeat, state, state.repeat_workspace.data, true);
    CUDA_OK(cudaStreamSynchronize(f.stream)); CUDA_OK(cudaGetLastError());
    const auto ws = state.workspace.download()[0];
    bool ok = read_status(f) == READY && ws.group_count == expected_groups &&
        ws.ideal_weight_ceiling_us == MEASURED_FFN_US * (ROUTES - expected_groups) / ROUTES &&
        ws.production_eligible == (expected_groups <= MAX_PLAUSIBLE_GROUPS);
    ok &= outputs_exact(f.parent, f.serial, (std::string(name) + ".parent_serial").c_str());
    ok &= outputs_exact(f.candidate, f.parent, (std::string(name) + ".candidate_parent").c_str());
    ok &= outputs_exact(f.candidate_repeat, f.candidate,
                        (std::string(name) + ".determinism").c_str());
    ok &= exact(state.workspace, state.repeat_workspace, "worklist_determinism");
    ok &= f.readonly_and_canaries() && state.contract.canary_ok("routed_contract") &&
        state.contract.immutable_ok("routed_contract") && state.workspace.canary_ok("workspace") &&
        state.repeat_workspace.canary_ok("repeat_workspace");
    std::printf("ROUTE_CASE name=%s groups=%u ideal_weight_ceiling_us=%u eligible=%s pass=%s\n", name,
        ws.group_count, ws.ideal_weight_ceiling_us, ws.production_eligible ? "true" : "false",
        ok ? "true" : "false");
    return ok;
}

bool hostile_gates(Fixture& f, RoutedState& state) {
    bool ok = true; const RoutedContract good = state.contract.download()[0];
    auto check = [&](const char* name, RoutedContract bad, RoutedWorkspace* ws, int* status,
                     int wanted) {
        f.candidate.reset(); f.arm_status(); state.contract.upload({bad}, false);
        preflight(f, f.candidate, state, ws, state.contract.data, status);
        CUDA_OK(cudaStreamSynchronize(f.stream));
        const int got = read_status(f); const bool pass = got == wanted && outputs_untouched(f.candidate);
        if (!pass) std::fprintf(stderr, "hostile gate %s status=%d wanted=%d\n", name, got, wanted);
        ok &= pass; state.contract.upload({good}, false); return pass;
    };
    RoutedContract bad = good; bad.rows = 15;
    check("abi", bad, state.workspace.data, nullptr, ERR_ROUTED_ABI);
    bad = good; bad.workspace_bytes--;
    check("short_workspace", bad, state.workspace.data, nullptr, ERR_ROUTED_WORKSPACE);
    check("workspace_alias", good, reinterpret_cast<RoutedWorkspace*>(f.candidate.gate.data),
          nullptr, ERR_ROUTED_ALIAS);
    check("workspace_alias_weight", good,
          reinterpret_cast<RoutedWorkspace*>(f.weights.gate_packed[4].data),
          nullptr, ERR_ROUTED_ALIAS);
    f.candidate.reset(); f.arm_status();
    preflight(f, f.candidate, state, state.workspace.data, nullptr,
              reinterpret_cast<int*>(f.candidate.gate.data));
    CUDA_OK(cudaStreamSynchronize(f.stream)); ok &= outputs_untouched(f.candidate) && read_status(f) == PENDING;
    f.candidate.reset(); f.arm_status();
    preflight(f, f.candidate, state, state.workspace.data, nullptr,
              reinterpret_cast<int*>(f.weights.gate_packed[4].data));
    CUDA_OK(cudaStreamSynchronize(f.stream));
    ok &= outputs_untouched(f.candidate) && read_status(f) == PENDING && f.weights.validate();
    f.candidate.reset(); f.arm_status();
    preflight(f, f.candidate, state, reinterpret_cast<RoutedWorkspace*>(UINTPTR_MAX - 15));
    CUDA_OK(cudaStreamSynchronize(f.stream)); ok &= outputs_untouched(f.candidate) && read_status(f) == PENDING;
    f.candidate.reset(); f.arm_status();
    preflight(f, f.candidate, state, state.workspace.data, nullptr,
              reinterpret_cast<int*>(UINTPTR_MAX - 3));
    CUDA_OK(cudaStreamSynchronize(f.stream)); ok &= outputs_untouched(f.candidate) && read_status(f) == PENDING;
    auto ids = f.indices.download(); ids[0] = EXPERTS; f.indices.upload(ids, false);
    check("invalid_route", good, state.workspace.data, nullptr, ERR_ROUTED_ROUTE);
    ids[0] = 0; f.indices.upload(ids);
    auto corrupt = [&](const char* name, auto mutate) {
        f.candidate.reset(); f.arm_status(); preflight(f, f.candidate, state, state.workspace.data);
        CUDA_OK(cudaStreamSynchronize(f.stream)); auto ws = state.workspace.download()[0]; mutate(ws);
        state.workspace.upload({ws}, false);
        moe_w4a16_routed_k16_validate<<<1, 1, 0, f.stream>>>(
            state.workspace.data, f.indices.data, f.status.data);
        launch_compute(f, f.candidate, state.workspace.data, false);
        CUDA_OK(cudaStreamSynchronize(f.stream));
        const bool pass = read_status(f) == ERR_ROUTED_WORKSPACE && outputs_untouched(f.candidate);
        if (!pass) std::fprintf(stderr, "corrupt worklist gate failed: %s\n", name); ok &= pass;
    };
    corrupt("expert_oob", [](RoutedWorkspace& ws) { ws.groups[0].expert = EXPERTS; });
    corrupt("count_oob", [](RoutedWorkspace& ws) { ws.groups[0].count = GROUP_WIDTH + 1; });
    corrupt("slot_oob", [](RoutedWorkspace& ws) { ws.groups[0].slots[0] = ROUTES; });
    corrupt("duplicate_slot", [](RoutedWorkspace& ws) { ws.groups[1].slots[1] = ws.groups[1].slots[0]; });
    corrupt("same_expert_group_swap", [](RoutedWorkspace& ws) { std::swap(ws.groups[1], ws.groups[2]); });
    corrupt("underfilled_split", [](RoutedWorkspace& ws) { ws.groups[1].count = 1; ws.groups[1].slots[1] = ws.groups[1].slots[2] = ws.groups[1].slots[3] = 0; });
    corrupt("route_mismatch", [](RoutedWorkspace& ws) { ws.groups[0].slots[0] = 1; });
    corrupt("group_reserved", [](RoutedWorkspace& ws) { ws.groups[0].reserved = 1; });
    corrupt("group_padding", [](RoutedWorkspace& ws) { ws.groups[0].padding[0] = 1; });
    corrupt("nonzero_tail", [](RoutedWorkspace& ws) { ws.groups[ws.group_count].expert = 1; });
    corrupt("expert_regression", [](RoutedWorkspace& ws) { std::swap(ws.groups[0], ws.groups[1]); });
    corrupt("expert_count", [](RoutedWorkspace& ws) { ++ws.expert_counts[0]; });
    corrupt("expert_cursor", [](RoutedWorkspace& ws) { ++ws.expert_cursors[7]; });
    corrupt("group_base", [](RoutedWorkspace& ws) { ++ws.expert_group_base[7]; });
    corrupt("header_group_count", [](RoutedWorkspace& ws) { --ws.group_count; ws.saved_weight_loads = ROUTES - ws.group_count; ws.ideal_weight_ceiling_us = MEASURED_FFN_US * ws.saved_weight_loads / ROUTES; ws.production_eligible = ws.group_count <= MAX_PLAUSIBLE_GROUPS && ws.ideal_weight_ceiling_us >= REQUIRED_SAVING_US; });
    return ok && f.readonly_and_canaries() && state.contract.canary_ok("routed_contract") &&
        state.contract.immutable_ok("routed_contract") && state.workspace.canary_ok("workspace");
}

float pct(std::vector<float> v, double p) {
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, static_cast<size_t>(std::ceil(p * v.size())) - 1)];
}

bool timing_gate(Fixture& f, RoutedState& state) {
    f.indices.upload(route_case(40));
    for (int i = 0; i < 3; ++i) { launch_baseline(f, f.parent); launch_routed(f, f.candidate, state, state.workspace.data, false); }
    CUDA_OK(cudaStreamSynchronize(f.stream)); std::vector<float> base, candidate, delta;
    cudaEvent_t begin, end; CUDA_OK(cudaEventCreate(&begin)); CUDA_OK(cudaEventCreate(&end));
    auto measure = [&](auto&& launch) { CUDA_OK(cudaEventRecord(begin, f.stream)); launch();
        CUDA_OK(cudaEventRecord(end, f.stream)); CUDA_OK(cudaEventSynchronize(end));
        float ms = 0; CUDA_OK(cudaEventElapsedTime(&ms, begin, end)); return ms; };
    for (int r = 0; r < 21; ++r) { float b, c; if ((r & 1) == 0) {
        b = measure([&] { launch_baseline(f, f.parent); });
        c = measure([&] { launch_routed(f, f.candidate, state, state.workspace.data, false); });
    } else { c = measure([&] { launch_routed(f, f.candidate, state, state.workspace.data, false); });
        b = measure([&] { launch_baseline(f, f.parent); }); }
        base.push_back(b); candidate.push_back(c); delta.push_back(b - c); }
    CUDA_OK(cudaEventDestroy(begin)); CUDA_OK(cudaEventDestroy(end));
    const float bm = pct(base, .5), cm = pct(candidate, .5), bp = pct(base, .9), cp = pct(candidate, .9);
    const float dm = pct(delta, .5); std::vector<float> dev;
    for (float d : delta) dev.push_back(std::fabs(d - dm)); const float mad = pct(dev, .5);
    const float frame = dm * 48.0f; const auto ws = state.workspace.download()[0];
    bool ok = cm < bm && cp < bp && dm - 3 * mad > 0 && frame >= 7.455f &&
        ws.production_eligible && read_status(f) == ROUTED_VALIDATED;
    ok &= exact(f.candidate.gate, f.parent.gate, "post_timing.gate");
    ok &= exact(f.candidate.up, f.parent.up, "post_timing.up");
    ok &= exact(f.candidate.down, f.parent.down, "post_timing.down");
    ok &= f.readonly_and_canaries() && state.workspace.canary_ok("post_timing.workspace") &&
        state.contract.immutable_ok("post_timing.contract");
    std::printf("TIMING baseline_median_ms=%.6f candidate_median_ms=%.6f baseline_p90_ms=%.6f "
        "candidate_p90_ms=%.6f paired_median_ms=%.6f mad_ms=%.6f x48_ms=%.6f pass=%s\n",
        bm, cm, bp, cp, dm, mad, frame, ok ? "true" : "false"); return ok;
}

} // namespace routed_gate

int main(int argc, char** argv) {
    const bool timing = argc == 2 && std::string(argv[1]) == "--timing";
    if (argc > 2 || (argc == 2 && !timing)) return 2;
    int device = 0; CUDA_OK(cudaGetDevice(&device)); cudaDeviceProp prop{};
    CUDA_OK(cudaGetDeviceProperties(&prop, device));
    if (prop.major != 12 || prop.minor != 1 || prop.multiProcessorCount != 48) return 2;
    Fixture f; routed_gate::RoutedState state; bool ok = true;
    ok &= routed_gate::fixture_case(f, state, 160, 160, "no_collision");
    ok &= routed_gate::fixture_case(f, state, 116, 116, "collision_threshold");
    ok &= routed_gate::fixture_case(f, state, 40, 40, "collision_quads");
    ok &= routed_gate::fixture_case(f, state, 0, 40, "degenerate_one_expert");
    ok &= routed_gate::hostile_gates(f, state);
    if (timing && ok) ok &= routed_gate::timing_gate(f, state);
    std::printf("FLASH_NEXT_ROUTED_K16_PERSISTENT parent=exact serial_k1=exact fixtures=4 "
        "hostile=24 required_x48_ms=7.455 timing=%s pass=%s\n",
        timing ? "measured" : "unrun", ok ? "true" : "false"); return ok ? 0 : 1;
}
