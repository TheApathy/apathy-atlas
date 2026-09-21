// SPDX-License-Identifier: AGPL-3.0-only

// Inverse-only GB10 promotion probe. Reuse the already source-locked incumbent,
// candidate, guarded buffers, deterministic inputs, and ABBA machinery from the
// full RoPE probe while replacing its entry point with an inverse-only gate.
#define main v4_combined_rope_probe_main_unreachable
#include "v4_prefill_rope_fused_probe.cu"
#undef main

namespace inverse_probe {

size_t mismatch_bytes(const std::vector<__nv_bfloat16>& lhs,
                      const std::vector<__nv_bfloat16>& rhs) {
    if (lhs.size() != rhs.size()) probe::fail("mismatch-size");
    const auto* left = reinterpret_cast<const unsigned char*>(lhs.data());
    const auto* right = reinterpret_cast<const unsigned char*>(rhs.data());
    size_t mismatches = 0;
    for (size_t index = 0; index < lhs.size() * sizeof(__nv_bfloat16); ++index) {
        mismatches += static_cast<size_t>(left[index] != right[index]);
    }
    return mismatches;
}

bool non_rope_input_unchanged(const std::vector<__nv_bfloat16>& input,
                              const std::vector<__nv_bfloat16>& output) {
    for (unsigned int token = 0; token < probe::kTokens; ++token) {
        for (unsigned int head = 0; head < probe::kNq; ++head) {
            const size_t base =
                (static_cast<size_t>(token) * probe::kNq + head) * probe::kHeadDim;
            for (unsigned int dim = 0; dim < probe::kNopeDim; ++dim) {
                if (__bfloat16_as_ushort(input[base + dim]) !=
                    __bfloat16_as_ushort(output[base + dim])) {
                    return false;
                }
            }
        }
    }
    return true;
}

}  // namespace inverse_probe

int main(int argc, char** argv) {
    using namespace probe;
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <min_speedup>\n", argv[0]);
        return 2;
    }
    const double min_speedup = parse_speedup_threshold(argv[1]);
    if (!valid_build_id(V4_ROPE_PROBE_BUILD_ID)) fail("invalid-build-id");

    int device = 0;
    check(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, device), "cudaGetDeviceProperties");
    int driver = 0;
    int runtime = 0;
    check(cudaDriverGetVersion(&driver), "cudaDriverGetVersion");
    check(cudaRuntimeGetVersion(&runtime), "cudaRuntimeGetVersion");

    GuardedBf16Buffer baseline(kQElements, 0x11, 0x12);
    GuardedBf16Buffer candidate(kQElements, 0x21, 0x22);
    DeviceBuffer<__nv_bfloat16> scratch(kQTempElements);
    DeviceBuffer<unsigned int> device_positions(kTokens);
    DeviceBuffer<float> device_frequencies(kRopeDim / 2);
    std::vector<__nv_bfloat16> input(kQElements);
    std::vector<__nv_bfloat16> expected(kQElements);
    std::vector<__nv_bfloat16> actual(kQElements);
    std::vector<unsigned int> positions(kTokens);
    std::vector<float> frequencies(kRopeDim / 2);
    std::vector<unsigned int> positions_after(kTokens);
    std::vector<float> frequencies_after(kRopeDim / 2);
    const float mscales[kParityCases] = {1.0f, 0.70710677f, 1.125f};
    uint64_t input_hash = 1469598103934665603ULL;
    uint64_t position_hash = 1469598103934665603ULL;
    uint64_t frequency_hash = 1469598103934665603ULL;
    size_t inverse_mismatch_bytes = 0;
    bool input_unchanged = true;
    bool table_unchanged = true;

    // Device-side positivity is part of the production ABI, not just a host
    // eligibility convention. Poison the complete payload and require exact
    // no-write behavior for both boundary classes before any parity launch.
    fill_tables(positions, frequencies, 1, 4);
    check(cudaMemcpy(device_positions.get(), positions.data(), device_positions.bytes(),
                     cudaMemcpyHostToDevice),
          "guard positions upload");
    check(cudaMemcpy(device_frequencies.get(), frequencies.data(),
                     device_frequencies.bytes(), cudaMemcpyHostToDevice),
          "guard frequencies upload");
    auto require_nonpositive_mscale_rejected = [&](float guard_mscale,
                                                    const char* label) {
        candidate.poison_payload(0x3C);
        candidate_inverse(candidate.get(), device_positions.get(),
                          device_frequencies.get(), guard_mscale);
        check(cudaGetLastError(), label);
        check(cudaDeviceSynchronize(), label);
        require_redzones(candidate, label);
        check(cudaMemcpy(actual.data(), candidate.get(), kQBytes,
                         cudaMemcpyDeviceToHost),
              "read inverse mscale poison");
        if (!all_poison(actual, 0x3C)) fail(label);
    };
    require_nonpositive_mscale_rejected(0.0f, "inverse-mscale-zero");
    require_nonpositive_mscale_rejected(-1.0f, "inverse-mscale-negative");

    for (int test_case = 0; test_case < kParityCases; ++test_case) {
        const unsigned int salt = static_cast<unsigned int>(test_case + 1);
        fill_values(input, 19 + salt);
        fill_tables(positions, frequencies, salt, salt + 3);
        input_hash = fnv1a(input_hash, input.data(), kQBytes);
        position_hash = fnv1a(
            position_hash, positions.data(), positions.size() * sizeof(unsigned int));
        frequency_hash =
            fnv1a(frequency_hash, frequencies.data(), frequencies.size() * sizeof(float));
        frequency_hash = fnv1a(frequency_hash, &mscales[test_case], sizeof(float));

        check(cudaMemcpy(baseline.get(), input.data(), kQBytes, cudaMemcpyHostToDevice),
              "parity baseline upload");
        check(cudaMemcpy(candidate.get(), input.data(), kQBytes, cudaMemcpyHostToDevice),
              "parity candidate upload");
        check(cudaMemcpy(device_positions.get(), positions.data(), device_positions.bytes(),
                         cudaMemcpyHostToDevice),
              "parity positions upload");
        check(cudaMemcpy(device_frequencies.get(), frequencies.data(),
                         device_frequencies.bytes(), cudaMemcpyHostToDevice),
              "parity frequencies upload");
        incumbent_inverse(baseline.get(), scratch.get(), device_positions.get(),
                          device_frequencies.get(), mscales[test_case]);
        candidate_inverse(candidate.get(), device_positions.get(), device_frequencies.get(),
                          mscales[test_case]);
        check(cudaDeviceSynchronize(), "inverse parity synchronize");
        require_redzones(baseline, "baseline-redzone");
        require_redzones(candidate, "candidate-redzone");
        check(cudaMemcpy(expected.data(), baseline.get(), kQBytes, cudaMemcpyDeviceToHost),
              "parity baseline download");
        check(cudaMemcpy(actual.data(), candidate.get(), kQBytes, cudaMemcpyDeviceToHost),
              "parity candidate download");
        check(cudaMemcpy(positions_after.data(), device_positions.get(), device_positions.bytes(),
                         cudaMemcpyDeviceToHost),
              "positions audit");
        check(cudaMemcpy(frequencies_after.data(), device_frequencies.get(),
                         device_frequencies.bytes(), cudaMemcpyDeviceToHost),
              "frequencies audit");
        inverse_mismatch_bytes += inverse_probe::mismatch_bytes(expected, actual);
        input_unchanged &= inverse_probe::non_rope_input_unchanged(input, actual);
        table_unchanged &= positions_after == positions && frequencies_after == frequencies;
    }
    if (inverse_mismatch_bytes != 0) fail("inverse-byte-parity");
    if (!input_unchanged) fail("non-rope-input-mutated");
    if (!table_unchanged) fail("input-table-mutated");

    const float timing_mscale = mscales[kParityCases - 1];
    auto baseline_reset = [&]() {
        check(cudaMemcpy(baseline.get(), input.data(), kQBytes, cudaMemcpyHostToDevice),
              "timing reset inverse baseline");
    };
    auto candidate_reset = [&]() {
        check(cudaMemcpy(candidate.get(), input.data(), kQBytes, cudaMemcpyHostToDevice),
              "timing reset inverse candidate");
    };
    auto baseline_launch = [&]() {
        incumbent_inverse(baseline.get(), scratch.get(), device_positions.get(),
                          device_frequencies.get(), timing_mscale);
    };
    auto candidate_launch = [&]() {
        candidate_inverse(candidate.get(), device_positions.get(), device_frequencies.get(),
                          timing_mscale);
    };
    const Timing timing = time_abba(
        baseline_reset, baseline_launch, candidate_reset, candidate_launch);
    // Recompute the inverse-only ratio explicitly: baseline_ms / candidate_ms.
    const float inverse_speedup = timing.baseline_ms / timing.candidate_ms;
    if (!std::isfinite(inverse_speedup) || timing.speedup < min_speedup) {
        fail("inverse-speedup-threshold");
    }
    require_redzones(baseline, "timing-baseline-redzone");
    require_redzones(candidate, "timing-candidate-redzone");

    const auto* uuid = reinterpret_cast<const unsigned char*>(&properties.uuid);
    std::printf("build_id=%s\n", V4_ROPE_PROBE_BUILD_ID);
    std::printf("device_uuid=");
    for (int index = 0; index < 16; ++index) std::printf("%02x", uuid[index]);
    std::printf(" driver=%d runtime=%d\n", driver, runtime);
    std::printf("input_hash=%016llx position_hash=%016llx frequency_hash=%016llx\n",
                static_cast<unsigned long long>(input_hash),
                static_cast<unsigned long long>(position_hash),
                static_cast<unsigned long long>(frequency_hash));
    std::printf("parity cases=%d inverse_mismatch_bytes=%zu\n", kParityCases,
                inverse_mismatch_bytes);
    std::printf("guards redzone_buffers=2 prefix=clean suffix=clean input_unchanged=%d table_unchanged=%d\n",
                input_unchanged ? 1 : 0, table_unchanged ? 1 : 0);
    std::printf("timing baseline_ms=%.9g candidate_ms=%.9g speedup=%.9g abba_rounds=%d\n",
                timing.baseline_ms, timing.candidate_ms, inverse_speedup, kAbbaRounds);
    std::printf("threshold min_speedup=%s\n", argv[1]);
    std::printf("result=PASS\n");
    return 0;
}
