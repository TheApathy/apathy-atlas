// SPDX-License-Identifier: AGPL-3.0-only
namespace v2_gate {

struct State {
    Guarded<v2::Contract> contract;
    Guarded<v2::Plan> plan, repeat_plan;
    Guarded<float> activation, repeat_activation;
    Guarded<int> status, repeat_status;

    State() {
        v2::Contract c{
            v2::MAGIC, v2::VERSION, ROWS, HIDDEN, INTER, EXPERTS, TOP_K,
            v2::ROUTES, v2::DESCRIPTORS, sizeof(v2::Plan), v2::ACTIVATION_ELEMS,
            v2::INPUT_ELEMS, v2::ROUTED_INTER_ELEMS, v2::ROUTED_HIDDEN_ELEMS,
            v2::SELECTED_WEIGHT_BYTES, v2::SELECTED_SCALE_BYTES
        };
        contract.allocate(1, true);
        contract.upload({c});
        plan.allocate(1);
        repeat_plan.allocate(1);
        activation.allocate(v2::ACTIVATION_ELEMS);
        repeat_activation.allocate(v2::ACTIVATION_ELEMS);
        status.allocate(1);
        repeat_status.allocate(1);
        status.upload({PENDING}, false);
        repeat_status.upload({PENDING}, false);
    }

    void arm(int* destination, cudaStream_t stream) {
        CUDA_OK(cudaMemsetAsync(destination, 0xff, sizeof(int), stream));
    }

    bool canaries() const {
        return contract.canary_ok("v2.contract") && contract.immutable_ok("v2.contract") &&
            plan.canary_ok("v2.plan") && repeat_plan.canary_ok("v2.repeat_plan") &&
            activation.canary_ok("v2.activation") &&
            repeat_activation.canary_ok("v2.repeat_activation") &&
            status.canary_ok("v2.status") && repeat_status.canary_ok("v2.repeat_status");
    }
};

struct Census { unsigned int homogeneous = 0, heterogeneous = 0; };

Census expected_census(const std::vector<unsigned int>& ids) {
    unsigned int histogram[EXPERTS]{};
    for (unsigned int expert : ids) ++histogram[expert];
    Census result;
    unsigned int tails = 0;
    for (unsigned int count : histogram) {
        result.homogeneous += count / v2::DESCRIPTOR_WIDTH;
        tails += count % v2::DESCRIPTOR_WIDTH;
    }
    result.heterogeneous = tails / v2::DESCRIPTOR_WIDTH;
    return result;
}

std::vector<unsigned int> route_case(unsigned int divisor) {
    std::vector<unsigned int> ids(v2::ROUTES);
    for (unsigned int i = 0; i < v2::ROUTES; ++i) ids[i] = divisor ? i % divisor : 7;
    return ids;
}

std::vector<unsigned int> mixed_tails() {
    std::vector<unsigned int> ids;
    ids.reserve(v2::ROUTES);
    for (unsigned int expert = 0; ids.size() < v2::ROUTES; ++expert)
        for (unsigned int n = 0; n < expert % 4 + 1 && ids.size() < v2::ROUTES; ++n)
            ids.push_back(expert);
    return ids;
}

constexpr unsigned int DISTINCT_EXPERTS = 40;
struct Distinct40 {
    std::vector<Guarded<unsigned char>> gp, gs, up, us, dp, ds;

    explicit Distinct40(Fixture& f) {
        constexpr size_t PACKED = v2::SELECTED_WEIGHT_BYTES;
        constexpr size_t SCALES = v2::SELECTED_SCALE_BYTES;
        auto gpt = f.weights.gate_packed_table.download();
        auto gst = f.weights.gate_scale_table.download();
        auto upt = f.weights.up_packed_table.download();
        auto ust = f.weights.up_scale_table.download();
        auto dpt = f.weights.down_packed_table.download();
        auto dst = f.weights.down_scale_table.download();
        for (unsigned int expert = 0; expert < DISTINCT_EXPERTS; ++expert) {
            gp.emplace_back(); gp.back().allocate(PACKED, true);
            gp.back().upload(packed_values(PACKED, 101 + expert));
            gs.emplace_back(); gs.back().allocate(SCALES, true);
            gs.back().upload(scale_values(SCALES, 151 + expert));
            up.emplace_back(); up.back().allocate(PACKED, true);
            up.back().upload(packed_values(PACKED, 201 + expert));
            us.emplace_back(); us.back().allocate(SCALES, true);
            us.back().upload(scale_values(SCALES, 251 + expert));
            dp.emplace_back(); dp.back().allocate(PACKED, true);
            dp.back().upload(packed_values(PACKED, 301 + expert));
            ds.emplace_back(); ds.back().allocate(SCALES, true);
            ds.back().upload(scale_values(SCALES, 351 + expert));
            gpt[expert] = reinterpret_cast<unsigned long long>(gp.back().data);
            gst[expert] = reinterpret_cast<unsigned long long>(gs.back().data);
            upt[expert] = reinterpret_cast<unsigned long long>(up.back().data);
            ust[expert] = reinterpret_cast<unsigned long long>(us.back().data);
            dpt[expert] = reinterpret_cast<unsigned long long>(dp.back().data);
            dst[expert] = reinterpret_cast<unsigned long long>(ds.back().data);
        }
        f.weights.gate_packed_table.upload(gpt);
        f.weights.gate_scale_table.upload(gst);
        f.weights.up_packed_table.upload(upt);
        f.weights.up_scale_table.upload(ust);
        f.weights.down_packed_table.upload(dpt);
        f.weights.down_scale_table.upload(dst);
    }

    bool valid() const {
        bool ok = true;
        for (const auto* bank : {&gp, &gs, &up, &us, &dp, &ds})
            for (const auto& buffer : *bank)
                ok &= buffer.canary_ok("distinct_weight") &&
                    buffer.immutable_ok("distinct_weight");
        return ok;
    }
};

void prepare_parent(Fixture& f, Outputs& output) {
    f.arm_status();
    f.preflight(output, f.contract.data, nullptr);
}

// Launch geometry depends only on compile-time official constants.  Expected
// descriptor census is intentionally absent from this interface and is used
// only after synchronization when validating the downloaded receipt.
void launch_v2(
    Fixture& f,
    Outputs& output,
    State& state,
    v2::Plan* plan,
    float* activation,
    int* status
) {
    constexpr unsigned int GATE_GRID = v2::DESCRIPTORS * (INTER / OUTPUTS_PER_CTA);
    constexpr unsigned int DOWN_GRID = v2::DESCRIPTORS * (HIDDEN / OUTPUTS_PER_CTA);
    state.arm(status, f.stream);
    moe_w4a16_routed_k16_v2_plan<<<1, v2::ROUTES, 0, f.stream>>>(
        f.indices.data, state.contract.data, plan, activation, f.status.data, status);
    moe_w4a16_routed_k16_v2_fixed_gate_up<<<GATE_GRID, 512, 0, f.stream>>>(
        f.input.data, f.weights.gate_packed_table.data, f.weights.gate_scale_table.data,
        f.weights.gate_s2.data, output.gate.data, plan, status);
    moe_w4a16_routed_k16_v2_fixed_gate_up<<<GATE_GRID, 512, 0, f.stream>>>(
        f.input.data, f.weights.up_packed_table.data, f.weights.up_scale_table.data,
        f.weights.up_s2.data, output.up.data, plan, status);
    moe_w4a16_routed_k16_v2_silu_stage<<<v2::ROUTES, 128, 0, f.stream>>>(
        output.gate.data, output.up.data, activation, plan, status);
    moe_w4a16_routed_k16_v2_fixed_down<<<DOWN_GRID, 512, 0, f.stream>>>(
        activation, f.weights.down_packed_table.data, f.weights.down_scale_table.data,
        f.weights.down_s2.data, output.down.data, plan, status);
}

void launch_routed_parent(Fixture& f, Outputs& output) {
    moe_w4a16_routed_k16_v2_parent_gate<<<
        dim3(v2::ROUTES * (INTER / OUTPUTS_PER_CTA), 2), 128, 0, f.stream>>>(
        f.input.data, f.weights.gate_packed_table.data, f.weights.gate_scale_table.data,
        f.weights.gate_s2.data, output.gate.data, f.weights.up_packed_table.data,
        f.weights.up_scale_table.data, f.weights.up_s2.data, output.up.data, f.indices.data);
    moe_w4a16_routed_k16_v2_parent_down<<<
        v2::ROUTES * (HIDDEN / OUTPUTS_PER_CTA), 128, INTER * sizeof(float), f.stream>>>(
        output.gate.data, output.up.data, f.weights.down_packed_table.data,
        f.weights.down_scale_table.data, f.weights.down_s2.data, output.down.data, f.indices.data);
}

