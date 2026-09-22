// SPDX-License-Identifier: AGPL-3.0-only
void launch_candidate_blend(Fixture& f, Outputs& output, const Outputs& shared_source) {
    moe_weighted_sum_blend_batch3<<<dim3(HIDDEN / 256, ROWS, 1), 256, 0, f.stream>>>(
        output.final.data, output.down.data, f.expert_weights.data,
        shared_source.shared_down.data, f.input.data, f.shared_gate_weight.data,
        HIDDEN, TOP_K, HIDDEN);
}

bool exact_routed(const Outputs& a, const Outputs& b, const char* label, bool final = false) {
    bool ok = exact(a.gate, b.gate, (std::string(label) + ".gate").c_str()) &&
        exact(a.up, b.up, (std::string(label) + ".up").c_str()) &&
        exact(a.down, b.down, (std::string(label) + ".down").c_str());
    if (final) ok &= exact(a.final, b.final, (std::string(label) + ".final").c_str());
    return ok;
}

bool host_plan_ok(
    const v2::Plan& plan,
    const std::vector<unsigned int>& ids,
    Census expected
) {
    bool ok = plan.ready_magic == v2::MAGIC && plan.version == v2::VERSION &&
        plan.descriptor_count == v2::DESCRIPTORS && plan.route_count == v2::ROUTES &&
        plan.homogeneous_count == expected.homogeneous &&
        plan.heterogeneous_count == expected.heterogeneous &&
        plan.homogeneous_count + plan.heterogeneous_count == v2::DESCRIPTORS &&
        plan.reserved0 == 0 && plan.reserved1 == 0 && plan.reserved2 == 0;
    std::vector<unsigned int> coverage(v2::ROUTES);
    unsigned int homogeneous = 0, heterogeneous = 0;
    for (unsigned int slot = 0; slot < v2::ROUTES; ++slot)
        ok &= plan.route_snapshot[slot] == ids[slot];
    for (const auto& descriptor : plan.descriptors) {
        bool all_same = true;
        unsigned int first = EXPERTS;
        for (unsigned int i = 0; i < v2::DESCRIPTOR_WIDTH; ++i) {
            const unsigned int slot = descriptor.slots[i];
            if (slot >= v2::ROUTES) { ok = false; continue; }
            ++coverage[slot];
            const unsigned int expert = ids[slot];
            if (i == 0) first = expert; else all_same &= expert == first;
            for (unsigned int j = i + 1; j < v2::DESCRIPTOR_WIDTH; ++j)
                ok &= descriptor.slots[i] != descriptor.slots[j];
        }
        for (unsigned char byte : descriptor.reserved) ok &= byte == 0;
        if (descriptor.mode == v2::HOMOGENEOUS) {
            ++homogeneous;
            ok &= all_same && descriptor.shared_expert == first;
        } else if (descriptor.mode == v2::HETEROGENEOUS) {
            ++heterogeneous;
            ok &= !all_same && descriptor.shared_expert == v2::HETEROGENEOUS_EXPERT;
        } else {
            ok = false;
        }
    }
    for (unsigned int count : coverage) ok &= count == 1;
    return ok && homogeneous == expected.homogeneous &&
        heterogeneous == expected.heterogeneous;
}

bool fixture_case(
    Fixture& f,
    State& state,
    const std::vector<unsigned int>& ids,
    const char* name
) {
    const Census expected = expected_census(ids);
    f.indices.upload(ids);
    f.parent.reset();
    f.serial.reset();
    f.candidate.reset();
    f.candidate_repeat.reset();
    f.launch_parent(f.parent);
    f.launch_serial(f.serial);
    prepare_parent(f, f.candidate);
    launch_v2(f, f.candidate, state, state.plan.data, state.activation.data, state.status.data);
    launch_candidate_blend(f, f.candidate, f.parent);
    prepare_parent(f, f.candidate_repeat);
    launch_v2(f, f.candidate_repeat, state, state.repeat_plan.data,
              state.repeat_activation.data, state.repeat_status.data);
    launch_candidate_blend(f, f.candidate_repeat, f.parent);
    CUDA_OK(cudaStreamSynchronize(f.stream));
    CUDA_OK(cudaGetLastError());
    const v2::Plan plan = state.plan.download()[0];
    const v2::Plan repeat_plan = state.repeat_plan.download()[0];
    const bool ok = state.status.download()[0] == v2::PLANNED &&
        state.repeat_status.download()[0] == v2::PLANNED && read_status(f) == READY &&
        host_plan_ok(plan, ids, expected) && host_plan_ok(repeat_plan, ids, expected) &&
        exact(state.plan, state.repeat_plan, "plan_determinism") &&
        exact(state.activation, state.repeat_activation, "activation_determinism") &&
        exact_routed(f.candidate, f.parent,
                     (std::string(name) + ".shipping_batch3").c_str(), true) &&
        exact_routed(f.candidate, f.serial,
                     (std::string(name) + ".shipping_serial_k1").c_str(), true) &&
        exact_routed(f.candidate_repeat, f.candidate,
                     (std::string(name) + ".candidate_determinism").c_str(), true) &&
        f.readonly_and_canaries() && state.canaries();
    std::printf("V2_CASE name=%s descriptors=40 homogeneous=%u heterogeneous=%u "
                "shipping_batch3=exact serial_k1=exact pass=%s\n",
                name, expected.homogeneous, expected.heterogeneous, ok ? "true" : "false");
    return ok;
}

bool hostile(Fixture& f, State& state) {
    bool ok = true;
    const auto good = state.contract.download()[0];
    const auto good_ids = route_case(40);
    f.indices.upload(good_ids);
    f.candidate.reset();
    prepare_parent(f, f.candidate);
    auto planner = [&](const v2::Contract* contract, v2::Plan* plan, float* activation,
                       const int* parent_status, int* status) {
        state.arm(status, f.stream);
        moe_w4a16_routed_k16_v2_plan<<<1, v2::ROUTES, 0, f.stream>>>(
            f.indices.data, contract, plan, activation, parent_status, status);
        CUDA_OK(cudaStreamSynchronize(f.stream));
        return state.status.download()[0];
    };

    auto bad = good;
    --bad.selected_scale_bytes;
    state.contract.upload({bad}, false);
    ok &= planner(state.contract.data, state.plan.data, state.activation.data,
                  f.status.data, state.status.data) == v2::ERR_ABI;
    state.contract.upload({good}, false);
    ok &= planner(state.contract.data, reinterpret_cast<v2::Plan*>(state.activation.data),
                  state.activation.data, f.status.data, state.status.data) == v2::ERR_ALIAS;
    ok &= planner(state.contract.data, reinterpret_cast<v2::Plan*>(UINTPTR_MAX - 15),
                  state.activation.data, f.status.data, state.status.data) == v2::ERR_OVERFLOW;

    auto bad_ids = good_ids;
    bad_ids[37] = EXPERTS;
    f.indices.upload(bad_ids, false);
    ok &= planner(state.contract.data, state.plan.data, state.activation.data,
                  f.status.data, state.status.data) == v2::ERR_ROUTE;
    f.indices.upload(good_ids);

    f.arm_status();
    ok &= planner(state.contract.data, state.plan.data, state.activation.data,
                  f.status.data, state.status.data) == v2::ERR_PARENT;
    prepare_parent(f, f.candidate);
    CUDA_OK(cudaStreamSynchronize(f.stream));

    CUDA_OK(cudaMemsetAsync(state.plan.data, 0xff, sizeof(int), f.stream));
    moe_w4a16_routed_k16_v2_plan<<<1, v2::ROUTES, 0, f.stream>>>(
        f.indices.data, state.contract.data, state.plan.data, state.activation.data,
        f.status.data, reinterpret_cast<int*>(state.plan.data));
    CUDA_OK(cudaStreamSynchronize(f.stream));
    ok &= (state.plan.download()[0].ready_magic & 0xffffffffULL) == 0xffffffffULL;

    ok &= planner(state.contract.data, state.plan.data, state.activation.data,
                  f.status.data, state.status.data) == v2::PLANNED;
    CUDA_OK(cudaMemsetAsync(&state.plan.data->descriptors[0].slots[1], 0, 1, f.stream));
    state.arm(state.status.data, f.stream);
    CUDA_OK(cudaMemsetAsync(state.status.data, 0, sizeof(int), f.stream));
    moe_w4a16_routed_k16_v2_fixed_gate_up<<<
        v2::DESCRIPTORS * (INTER / OUTPUTS_PER_CTA), 512, 0, f.stream>>>(
        f.input.data, f.weights.gate_packed_table.data, f.weights.gate_scale_table.data,
        f.weights.gate_s2.data, f.candidate.gate.data, state.plan.data, state.status.data);
    CUDA_OK(cudaStreamSynchronize(f.stream));
    ok &= state.status.download()[0] == v2::ERR_PLAN;
    ok &= f.readonly_and_canaries() && state.canaries();
    state.contract.upload({good});
    f.indices.upload(good_ids);
    std::printf("V2_HOSTILE cases=7 pass=%s\n", ok ? "true" : "false");
    return ok;
}

