// SPDX-License-Identifier: AGPL-3.0-only
float percentile(std::vector<float> values, double p) {
    std::sort(values.begin(), values.end());
    const size_t index = std::min(values.size() - 1,
        static_cast<size_t>(std::ceil(p * values.size())) - 1);
    return values[index];
}

bool timing(Fixture& f, State& state) {
    Distinct40 distinct(f);
    const auto ids = route_case(DISTINCT_EXPERTS);
    f.indices.upload(ids);
    f.parent.reset();
    f.candidate.reset();
    prepare_parent(f, f.candidate);
    CUDA_OK(cudaStreamSynchronize(f.stream));
    for (int i = 0; i < 3; ++i) {
        launch_routed_parent(f, f.parent);
        launch_v2(f, f.candidate, state, state.plan.data,
                  state.activation.data, state.status.data);
    }
    CUDA_OK(cudaStreamSynchronize(f.stream));

    std::vector<float> parent, candidate, delta;
    cudaEvent_t begin, end;
    CUDA_OK(cudaEventCreate(&begin));
    CUDA_OK(cudaEventCreate(&end));
    auto milliseconds = [&](auto launch) {
        CUDA_OK(cudaEventRecord(begin, f.stream));
        launch();
        CUDA_OK(cudaEventRecord(end, f.stream));
        CUDA_OK(cudaEventSynchronize(end));
        float value = 0.0f;
        CUDA_OK(cudaEventElapsedTime(&value, begin, end));
        return value;
    };
    for (int repetition = 0; repetition < 21; ++repetition) {
        float p = 0.0f, c = 0.0f;
        if ((repetition & 1) == 0) {
            p = milliseconds([&] { launch_routed_parent(f, f.parent); });
            c = milliseconds([&] { launch_v2(f, f.candidate, state, state.plan.data,
                                              state.activation.data, state.status.data); });
        } else {
            c = milliseconds([&] { launch_v2(f, f.candidate, state, state.plan.data,
                                              state.activation.data, state.status.data); });
            p = milliseconds([&] { launch_routed_parent(f, f.parent); });
        }
        parent.push_back(p);
        candidate.push_back(c);
        delta.push_back(p - c);
    }
    CUDA_OK(cudaEventDestroy(begin));
    CUDA_OK(cudaEventDestroy(end));
    const float parent_median = percentile(parent, 0.5);
    const float candidate_median = percentile(candidate, 0.5);
    const float delta_median = percentile(delta, 0.5);
    std::vector<float> deviations;
    for (float value : delta) deviations.push_back(std::fabs(value - delta_median));
    const float mad = percentile(deviations, 0.5);
    const float robust_x48 = 48.0f * (delta_median - 3.0f * mad);
    const bool ok = candidate_median < parent_median &&
        percentile(candidate, 0.9) < percentile(parent, 0.9) &&
        robust_x48 >= 7.455f && state.status.download()[0] == v2::PLANNED &&
        exact_routed(f.candidate, f.parent, "post_timing_routed_parent") &&
        distinct.valid() && f.readonly_and_canaries() && state.canaries();
    std::printf("V2_TIMING_SCOPE baseline=exact_parent_routed_only "
                "candidate=planner_plus_fixed_launches preflight=excluded shared_blend=excluded "
                "conditional=true\n");
    std::printf("V2_TIMING parent=%.6f candidate=%.6f delta=%.6f mad=%.6f "
                "robust_x48=%.6f threshold=7.455000 pass=%s\n",
                parent_median, candidate_median, delta_median, mad, robust_x48,
                ok ? "true" : "false");
    return ok;
}

} // namespace v2_gate

