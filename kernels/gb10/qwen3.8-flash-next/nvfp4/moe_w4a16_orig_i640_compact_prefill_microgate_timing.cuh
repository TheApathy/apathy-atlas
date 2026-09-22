// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

namespace oi640_gate {

struct TimingResult {
    float parent_median = 0, parent_p90 = 0;
    float candidate_median = 0, candidate_p90 = 0, paired_median = 0;
    uint32_t samples = 0;
    bool pass = false;
};

float nearest_rank(std::vector<float> values, float fraction) {
    std::sort(values.begin(), values.end());
    const size_t rank = std::max<size_t>(1, static_cast<size_t>(std::ceil(values.size() * fraction)));
    return values[std::min(rank, values.size()) - 1];
}

TimingResult time_real_skew(ShapeCase& shape, Weights& weights, cudaStream_t stream) {
    const uint32_t sample_count = shape.rows == oi640::M_SMALL ? 22 : 32;
    for (int i = 0; i < 3; ++i) {
        shape.launch_parent(weights, stream);
        shape.launch_candidate(weights, stream);
    }
    CUDA_OK(cudaStreamSynchronize(stream));
    cudaEvent_t begin{}, end{};
    CUDA_OK(cudaEventCreate(&begin)); CUDA_OK(cudaEventCreate(&end));
    auto measure = [&](auto launch) {
        CUDA_OK(cudaEventRecord(begin, stream)); launch(); CUDA_OK(cudaEventRecord(end, stream));
        CUDA_OK(cudaEventSynchronize(end));
        float ms = 0; CUDA_OK(cudaEventElapsedTime(&ms, begin, end)); return ms;
    };
    std::vector<float> parent, candidate, paired;
    for (uint32_t i = 0; i < sample_count; ++i) {
        float p = 0, c = 0;
        if ((i & 1) == 0) {
            p = measure([&] { shape.launch_parent(weights, stream); });
            c = measure([&] { shape.launch_candidate(weights, stream); });
        } else {
            c = measure([&] { shape.launch_candidate(weights, stream); });
            p = measure([&] { shape.launch_parent(weights, stream); });
        }
        parent.push_back(p); candidate.push_back(c); paired.push_back(p - c);
    }
    CUDA_OK(cudaEventDestroy(begin)); CUDA_OK(cudaEventDestroy(end));
    TimingResult result;
    result.parent_median = nearest_rank(parent, 0.5f);
    result.parent_p90 = nearest_rank(parent, 0.9f);
    result.candidate_median = nearest_rank(candidate, 0.5f);
    result.candidate_p90 = nearest_rank(candidate, 0.9f);
    result.paired_median = nearest_rank(paired, 0.5f);
    result.samples = sample_count;
    shape.launch_parent(weights, stream); shape.launch_candidate(weights, stream);
    CUDA_OK(cudaStreamSynchronize(stream));
    result.pass = result.candidate_median < result.parent_median &&
        result.candidate_p90 < result.parent_p90 && result.paired_median > 0 &&
        shape.status.download()[0] == oi640::PLANNED &&
        shape.exact(shape.parent_gate, shape.candidate_gate) &&
        shape.exact(shape.parent_up, shape.candidate_up) &&
        shape.exact(shape.parent_down, shape.candidate_down) && shape.guards() && weights.safe();
    return result;
}

} // namespace oi640_gate
