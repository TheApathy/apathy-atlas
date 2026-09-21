// SPDX-License-Identifier: AGPL-3.0-only
//
// Immutable GB10 admission probe for the exact M32xN256 fused W2A8 arm.
// The only timed operators are the production N128 fused incumbent and the
// production N256 fused candidate. This TU is outside every serving registry.

#include <algorithm>
#include <array>

#ifndef W2A8_FUSED_GU_N256_PROBE_BUILD_ID
#error "W2A8_FUSED_GU_N256_PROBE_BUILD_ID must be receipt-bound"
#endif
#ifndef W2A8_FUSED_GU_N256_PROBE_MIN_SPEEDUP
#error "W2A8_FUSED_GU_N256_PROBE_MIN_SPEEDUP must be receipt-bound"
#endif
#ifndef W2A8_FUSED_GU_N256_PROBE_MIN_SPEEDUP_TEXT
#error "W2A8_FUSED_GU_N256_PROBE_MIN_SPEEDUP_TEXT must be receipt-bound"
#endif

// Reuse the mature fixture/input generators without changing the incumbent
// probe. Rename its internal N128 helpers so the N256 component can coexist in
// this translation unit, and rename its main so it cannot be the entry point.
#define W2A8_FUSED_GU_PROBE_BUILD_ID W2A8_FUSED_GU_N256_PROBE_BUILD_ID
#define W2A8_FUSED_GU_PROBE_MIN_SPEEDUP W2A8_FUSED_GU_N256_PROBE_MIN_SPEEDUP
#define W2A8_FUSED_GU_PROBE_MIN_SPEEDUP_TEXT \
    W2A8_FUSED_GU_N256_PROBE_MIN_SPEEDUP_TEXT
#define W2FHalf2Bits W2F128Half2Bits
#define W2FLaneGeom W2F128LaneGeom
#define W2FGemmScratch W2F128GemmScratch
#define W2FWork W2F128Work
#define W2FShared W2F128Shared
#define w2f_had128 w2f128_had128
#define w2f_lane_geom w2f128_lane_geom
#define w2f_decode2 w2f128_decode2
#define w2f_decode8 w2f128_decode8
#define w2f_encode_pair w2f128_encode_pair
#define w2f_repack_b w2f128_repack_b
#define w2f_mma w2f128_mma
#define w2f_compute_leg w2f128_compute_leg
#define w2f_emit_group w2f128_emit_group
#define main embedded_n128_probe_main
#include "exl3_w2a8_fused_gu_down_emit_n128_probe.cu"
#undef main
#undef w2f_emit_group
#undef w2f_compute_leg
#undef w2f_mma
#undef w2f_repack_b
#undef w2f_encode_pair
#undef w2f_decode8
#undef w2f_decode2
#undef w2f_lane_geom
#undef w2f_had128
#undef W2FShared
#undef W2FWork
#undef W2FGemmScratch
#undef W2FLaneGeom
#undef W2FHalf2Bits
#undef W2A8_FUSED_GU_PROBE_MIN_SPEEDUP_TEXT
#undef W2A8_FUSED_GU_PROBE_MIN_SPEEDUP
#undef W2A8_FUSED_GU_PROBE_BUILD_ID

// The N128 component leaves its private constants defined after inclusion.
#undef W2F_M_TILE
#undef W2F_N_TILE
#undef W2F_GATE_UP_N
#undef W2F_GATE_UP_K
#undef W2F_K_STAGE
#undef W2F_K_GROUP
#undef W2F_WEIGHT_SCALE
#undef W2F_INV_WEIGHT_SCALE
#undef W2F_E4M3_MAX
#undef W2F_MIN_SCALE
#undef W2F_MCG_MULT
#undef W2F_HROW_WARPS
#undef W2F_RSQRT128
#undef W2F_SWIGLU_LIMIT
#undef W2F_EXPERTS

#define W2A8_KERNEL_NAME exl3_w2a8_fused_gu_down_emit_n256
#include "exl3_w2a8_fused_gu_down_emit_n256.cu"
#undef W2A8_KERNEL_NAME

namespace n256_probe {

constexpr unsigned kN = 2048;
constexpr unsigned kK = 4096;
constexpr unsigned kMaxExperts = 256;
constexpr unsigned kProductionRows = 14460;
constexpr unsigned kThreads = 256;
constexpr unsigned kN256Tile = 256;
constexpr std::size_t kDownFp8Bytes = 29614080;
constexpr std::size_t kDownScaleBytes = 925440;
constexpr int kWarmups = 2;
constexpr int kTimingIterations = 8;
constexpr double kMinSpeedup = W2A8_FUSED_GU_N256_PROBE_MIN_SPEEDUP;
constexpr std::array<unsigned, 10> kBoundaryCounts =
    {1, 31, 32, 33, 63, 64, 65, 127, 128, 129};
static_assert(kDownFp8Bytes ==
              static_cast<std::size_t>(kProductionRows) * kN);
static_assert(kDownScaleBytes ==
              static_cast<std::size_t>(kProductionRows) * (kN / 128) * 4);
static_assert(kMinSpeedup > 1.0 && kMinSpeedup <= 100.0);

using probe::Buffer;
using probe::Fixture;

struct RouteCase {
    const char* label;
    std::vector<int> offsets;
};

std::vector<int> offsets_from_counts(const std::vector<unsigned>& counts) {
    if (counts.size() != kMaxExperts) probe::fail("route-count-shape");
    std::vector<int> route(kMaxExperts + 1, 0);
    for (unsigned expert = 0; expert < kMaxExperts; ++expert) {
        route[expert + 1] = route[expert] + static_cast<int>(counts[expert]);
    }
    if (route.back() != static_cast<int>(kProductionRows)) {
        probe::fail("production-route-total");
    }
    return route;
}

std::vector<unsigned> even_counts(unsigned total, unsigned experts) {
    std::vector<unsigned> counts(experts, total / experts);
    for (unsigned expert = 0; expert < total % experts; ++expert) {
        ++counts[expert];
    }
    return counts;
}

std::vector<RouteCase> production_routes() {
    std::vector<unsigned> balanced = even_counts(kProductionRows, kMaxExperts);

    std::vector<unsigned> empty = balanced;
    empty[17] = 0;
    empty[18] += balanced[17];

    std::vector<unsigned> skewed(kMaxExperts, 0);
    skewed[0] = 2410;
    const std::vector<unsigned> skew_tail =
        even_counts(kProductionRows - skewed[0], kMaxExperts - 1);
    std::copy(skew_tail.begin(), skew_tail.end(), skewed.begin() + 1);

    std::vector<unsigned> boundaries(kMaxExperts, 0);
    std::copy(
        kBoundaryCounts.begin(), kBoundaryCounts.end(), boundaries.begin());
    unsigned assigned = 0;
    for (unsigned value : kBoundaryCounts) assigned += value;
    const std::vector<unsigned> boundary_tail =
        even_counts(
            kProductionRows - assigned,
            kMaxExperts - kBoundaryCounts.size());
    std::copy(
        boundary_tail.begin(), boundary_tail.end(),
        boundaries.begin() + kBoundaryCounts.size());

    return {
        {"production-balanced", offsets_from_counts(balanced)},
        {"production-empty", offsets_from_counts(empty)},
        {"production-skewed", offsets_from_counts(skewed)},
        {"production-boundaries", offsets_from_counts(boundaries)},
    };
}

template <typename T>
void verify_buffer(
    const Buffer& buffer, const std::vector<T>& expected, const char* name) {
    const auto actual = buffer.download();
    const std::size_t expected_bytes = expected.size() * sizeof(T);
    if (!buffer.guards_clean() || actual.size() != expected_bytes ||
        std::memcmp(actual.data(), expected.data(), expected_bytes) != 0) {
        probe::fail(name);
    }
}

struct InputSnapshot {
    std::vector<unsigned char> offsets;
    std::vector<unsigned char> experts;
    std::vector<unsigned char> gate_trellis_tab;
    std::vector<unsigned char> up_trellis_tab;
    std::vector<unsigned char> gate_svh_tab;
    std::vector<unsigned char> up_svh_tab;
    std::vector<unsigned char> down_suh_tab;
};

InputSnapshot snapshot_inputs(const Fixture& fixture) {
    return {
        fixture.offsets.download(),
        fixture.experts.download(),
        fixture.gate_trellis_tab.download(),
        fixture.up_trellis_tab.download(),
        fixture.gate_svh_tab.download(),
        fixture.up_svh_tab.download(),
        fixture.down_suh_tab.download(),
    };
}

void verify_snapshot(
    const Buffer& buffer, const std::vector<unsigned char>& expected,
    const char* name) {
    const auto actual = buffer.download();
    if (!buffer.guards_clean() || actual != expected) probe::fail(name);
}

// Exact byte checks use the fixture's authoritative host copies for all large
// tensor inputs, and snapshots for route/pointer tables whose values are
// device-address-dependent.
void verify_inputs_immutable(
    const Fixture& fixture, const InputSnapshot& snapshot) {
    verify_buffer(
        fixture.gate_activation, fixture.gate_activation_h,
        "gate-activation-mutated");
    verify_buffer(
        fixture.up_activation, fixture.up_activation_h,
        "up-activation-mutated");
    verify_buffer(fixture.gate_scale, fixture.gate_scale_h, "gate-scale-mutated");
    verify_buffer(fixture.up_scale, fixture.up_scale_h, "up-scale-mutated");
    verify_buffer(
        fixture.gate_trellis, fixture.gate_trellis_h,
        "gate-trellis-mutated");
    verify_buffer(
        fixture.up_trellis, fixture.up_trellis_h, "up-trellis-mutated");
    verify_buffer(fixture.gate_svh, fixture.gate_svh_h, "gate-svh-mutated");
    verify_buffer(fixture.up_svh, fixture.up_svh_h, "up-svh-mutated");
    verify_buffer(fixture.down_suh, fixture.down_suh_h, "down-suh-mutated");
    verify_snapshot(fixture.offsets, snapshot.offsets, "offsets-mutated");
    verify_snapshot(fixture.experts, snapshot.experts, "experts-mutated");
    verify_snapshot(
        fixture.gate_trellis_tab, snapshot.gate_trellis_tab,
        "gate-trellis-table-mutated");
    verify_snapshot(
        fixture.up_trellis_tab, snapshot.up_trellis_tab,
        "up-trellis-table-mutated");
    verify_snapshot(
        fixture.gate_svh_tab, snapshot.gate_svh_tab,
        "gate-svh-table-mutated");
    verify_snapshot(
        fixture.up_svh_tab, snapshot.up_svh_tab, "up-svh-table-mutated");
    verify_snapshot(
        fixture.down_suh_tab, snapshot.down_suh_tab,
        "down-suh-table-mutated");
}

void launch_n128(Fixture& fixture) {
    probe::launch_candidate(fixture, kProductionRows);
}

void launch_n256_raw(
    Fixture& fixture, const int* offsets, unsigned char* down_fp8,
    float* down_scale) {
    exl3_w2a8_fused_gu_down_emit_n256<<<
        dim3(kMaxExperts * (kN / kN256Tile), 1, 1),
        dim3(kThreads, 1, 1)>>>(
        fixture.gate_activation.data,
        reinterpret_cast<float*>(fixture.gate_scale.data),
        fixture.up_activation.data,
        reinterpret_cast<float*>(fixture.up_scale.data),
        reinterpret_cast<unsigned long long*>(fixture.gate_trellis_tab.data),
        reinterpret_cast<unsigned long long*>(fixture.up_trellis_tab.data),
        reinterpret_cast<unsigned long long*>(fixture.gate_svh_tab.data),
        reinterpret_cast<unsigned long long*>(fixture.up_svh_tab.data),
        reinterpret_cast<unsigned long long*>(fixture.down_suh_tab.data),
        down_fp8, down_scale, offsets, kMaxExperts, kProductionRows, kN, kK, 2,
        1);
    probe::check(cudaGetLastError(), "N256 fused launch");
}

void launch_n256(Fixture& fixture) {
    launch_n256_raw(
        fixture, reinterpret_cast<int*>(fixture.offsets.data),
        fixture.reference_fp8.data,
        reinterpret_cast<float*>(fixture.reference_scale.data));
}

struct PoisonPair {
    unsigned char n128;
    unsigned char n256;
};

constexpr PoisonPair kPoisonPairs[] = {
    {probe::kPoisonA, probe::kPoisonB},
    {probe::kPoisonB, probe::kPoisonA},
};

void verify_guard(const Buffer& buffer, const char* name) {
    if (!buffer.guards_clean()) probe::fail(name);
}

std::uint64_t hash_buffer(
    std::uint64_t hash, const Buffer& buffer, std::size_t bytes) {
    auto values = buffer.download();
    if (values.size() < bytes) probe::fail("hash-extent");
    values.resize(bytes);
    return probe::hash_vector(hash, values);
}

struct ParityResult {
    std::size_t fp8_mismatches = 0;
    std::size_t scale_mismatches = 0;
    unsigned cases = 0;
    std::uint64_t routing_hash = 0xcbf29ce484222325ull;
    std::uint64_t n128_output_hash = 0xcbf29ce484222325ull;
    std::uint64_t n256_output_hash = 0xcbf29ce484222325ull;
};

void run_parity_case(
    Fixture& fixture, ParityResult& result, const RouteCase& route_case) {
    const auto& route = route_case.offsets;
    if (route.size() != kMaxExperts + 1 || route.front() != 0 ||
        route.back() != static_cast<int>(kProductionRows)) {
        probe::fail("production-route-shape");
    }
    fixture.upload_routing(route);
    const InputSnapshot input_snapshot = snapshot_inputs(fixture);
    result.routing_hash = probe::hash_vector(result.routing_hash, route);
    const std::vector<int> ids = probe::expert_ids(route, kProductionRows);
    result.routing_hash = probe::hash_vector(result.routing_hash, ids);

    std::vector<unsigned char> n128_fp8_first;
    std::vector<unsigned char> n128_scale_first;
    std::vector<unsigned char> n256_fp8_first;
    std::vector<unsigned char> n256_scale_first;
    for (unsigned pass = 0; pass < 2; ++pass) {
        const PoisonPair poison = kPoisonPairs[pass];
        fixture.candidate_fp8.fill(poison.n128);
        fixture.candidate_scale.fill(poison.n128);
        fixture.reference_fp8.fill(poison.n256);
        fixture.reference_scale.fill(poison.n256);
        if (pass == 0) {
            launch_n128(fixture);
            launch_n256(fixture);
        } else {
            launch_n256(fixture);
            launch_n128(fixture);
        }
        probe::check(cudaDeviceSynchronize(), "N128/N256 parity synchronize");

        result.fp8_mismatches += probe::byte_mismatches(
            fixture.candidate_fp8, fixture.reference_fp8, kDownFp8Bytes);
        result.scale_mismatches += probe::byte_mismatches(
            fixture.candidate_scale, fixture.reference_scale,
            kDownScaleBytes);
        ++result.cases;

        const auto n128_fp8 = fixture.candidate_fp8.download();
        const auto n128_scale = fixture.candidate_scale.download();
        const auto n256_fp8 = fixture.reference_fp8.download();
        const auto n256_scale = fixture.reference_scale.download();
        if (pass == 0) {
            n128_fp8_first = n128_fp8;
            n128_scale_first = n128_scale;
            n256_fp8_first = n256_fp8;
            n256_scale_first = n256_scale;
        } else if (n128_fp8 != n128_fp8_first ||
                   n128_scale != n128_scale_first ||
                   n256_fp8 != n256_fp8_first ||
                   n256_scale != n256_scale_first) {
            probe::fail("poison-dependent-output");
        }
        verify_guard(fixture.candidate_fp8, "N128-fp8-redzone");
        verify_guard(fixture.candidate_scale, "N128-scale-redzone");
        verify_guard(fixture.reference_fp8, "N256-fp8-redzone");
        verify_guard(fixture.reference_scale, "N256-scale-redzone");
        probe::verify_scale_values(fixture.candidate_scale, kProductionRows);
        probe::verify_scale_values(fixture.reference_scale, kProductionRows);

        result.n128_output_hash =
            hash_buffer(result.n128_output_hash, fixture.candidate_fp8,
                        kDownFp8Bytes);
        result.n128_output_hash =
            hash_buffer(result.n128_output_hash, fixture.candidate_scale,
                        kDownScaleBytes);
        result.n256_output_hash =
            hash_buffer(result.n256_output_hash, fixture.reference_fp8,
                        kDownFp8Bytes);
        result.n256_output_hash =
            hash_buffer(result.n256_output_hash, fixture.reference_scale,
                        kDownScaleBytes);
    }
    verify_inputs_immutable(fixture, input_snapshot);
}

void verify_fully_poisoned(
    const Buffer& buffer, unsigned char poison, const char* name) {
    if (!buffer.guards_clean() ||
        !probe::all_value(buffer.download(), 0, poison)) {
        probe::fail(name);
    }
}

void malformed_no_write(
    Fixture& fixture, const char* name, const std::vector<int>& route) {
    fixture.offsets.upload(route);
    const InputSnapshot input_snapshot = snapshot_inputs(fixture);
    for (unsigned char poison : {probe::kPoisonA, probe::kPoisonB}) {
        fixture.reference_fp8.fill(poison);
        fixture.reference_scale.fill(poison);
        launch_n256(fixture);
        probe::check(cudaDeviceSynchronize(), name);
        n256_probe::verify_fully_poisoned(
            fixture.reference_fp8, poison, name);
        n256_probe::verify_fully_poisoned(
            fixture.reference_scale, poison, name);
    }
    verify_inputs_immutable(fixture, input_snapshot);
}

void run_malformed_routes(Fixture& fixture) {
    std::vector<int> route = probe::balanced_offsets();
    route[0] = 1;
    malformed_no_write(fixture, "first-malformed-route", route);

    route = probe::balanced_offsets();
    route[kMaxExperts - 8] = route[kMaxExperts - 9] - 1;
    malformed_no_write(fixture, "late-malformed-route", route);
}

template <typename Launch>
float time_launch(Launch launch) {
    cudaEvent_t start = nullptr;
    cudaEvent_t finish = nullptr;
    probe::check(cudaEventCreate(&start), "event-create-start");
    probe::check(cudaEventCreate(&finish), "event-create-finish");
    probe::check(cudaEventRecord(start), "event-record-start");
    for (int iteration = 0; iteration < kTimingIterations; ++iteration) launch();
    probe::check(cudaEventRecord(finish), "event-record-finish");
    probe::check(cudaEventSynchronize(finish), "event-synchronize");
    float milliseconds = 0.0f;
    probe::check(
        cudaEventElapsedTime(&milliseconds, start, finish), "event-elapsed");
    probe::check(cudaEventDestroy(start), "event-destroy-start");
    probe::check(cudaEventDestroy(finish), "event-destroy-finish");
    return milliseconds / kTimingIterations;
}

struct TimingResult {
    float n128_ms;
    float n256_ms;
    float speedup;
};

TimingResult run_abba_timing(Fixture& fixture) {
    fixture.upload_routing(probe::balanced_offsets());
    const InputSnapshot input_snapshot = snapshot_inputs(fixture);
    auto n128 = [&] { launch_n128(fixture); };
    auto n256 = [&] { launch_n256(fixture); };
    for (int warmup = 0; warmup < kWarmups; ++warmup) {
        n128();
        n256();
    }
    probe::check(cudaDeviceSynchronize(), "ABBA warmup synchronize");
    const float a0 = time_launch(n128);
    const float b0 = time_launch(n256);
    const float b1 = time_launch(n256);
    const float a1 = time_launch(n128);
    const float n128_ms = 0.5f * (a0 + a1);
    const float n256_ms = 0.5f * (b0 + b1);
    if (!(n128_ms > 0.0f) || !(n256_ms > 0.0f) ||
        !std::isfinite(n128_ms) || !std::isfinite(n256_ms)) {
        probe::fail("invalid-ABBA-timing");
    }
    verify_inputs_immutable(fixture, input_snapshot);
    return {n128_ms, n256_ms, n128_ms / n256_ms};
}

}  // namespace n256_probe

int main(int argc, char** argv) {
    if (argc != 1) {
        std::fprintf(stderr, "usage: %s\n", argv[0]);
        return 2;
    }
    const char* build_id = W2A8_FUSED_GU_N256_PROBE_BUILD_ID;
    if (!probe::valid_build_id(build_id)) {
        std::fprintf(stderr, "invalid N256 probe build id\n");
        return 2;
    }

    probe::Fixture fixture(n256_probe::kProductionRows);
    n256_probe::ParityResult parity;
    const std::vector<n256_probe::RouteCase> routes =
        n256_probe::production_routes();
    for (const auto& route : routes) {
        n256_probe::run_parity_case(fixture, parity, route);
    }
    n256_probe::run_malformed_routes(fixture);
    if (parity.fp8_mismatches != 0 || parity.scale_mismatches != 0 ||
        parity.n128_output_hash != parity.n256_output_hash) {
        probe::fail("N128-N256-byte-parity");
    }
    const n256_probe::TimingResult timing =
        n256_probe::run_abba_timing(fixture);
    using n256_probe::kMinSpeedup;
    if (timing.speedup < kMinSpeedup) {
        probe::fail("speedup-threshold");
    }

    int device = 0;
    int driver = 0;
    int runtime = 0;
    cudaDeviceProp properties{};
    probe::check(cudaGetDevice(&device), "cudaGetDevice");
    probe::check(
        cudaGetDeviceProperties(&properties, device),
        "cudaGetDeviceProperties");
    probe::check(cudaDriverGetVersion(&driver), "cudaDriverGetVersion");
    probe::check(cudaRuntimeGetVersion(&runtime), "cudaRuntimeGetVersion");

    std::printf("build_id=%s\n", build_id);
    std::printf("device_uuid=");
    for (unsigned char byte : properties.uuid.bytes) {
        std::printf("%02x", static_cast<unsigned>(byte));
    }
    std::printf(" driver=%d runtime=%d\n", driver, runtime);
    std::printf(
        "input_hash=%016llx routing_hash=%016llx tables_hash=%016llx\n",
        static_cast<unsigned long long>(probe::input_hash(fixture)),
        static_cast<unsigned long long>(parity.routing_hash),
        static_cast<unsigned long long>(probe::tables_hash(fixture)));
    std::printf(
        "n128_output_hash=%016llx n256_output_hash=%016llx\n",
        static_cast<unsigned long long>(parity.n128_output_hash),
        static_cast<unsigned long long>(parity.n256_output_hash));
    std::printf(
        "rows=14460 fp8_bytes=29614080 scale_bytes=925440 experts=256 "
        "routes=4 boundary_counts=10 malformed_routes=2 parity_cases=8\n");
    std::printf("routing=synthetic_production_total representative=0\n");
    std::printf(
        "threshold min_speedup=%s\n",
        W2A8_FUSED_GU_N256_PROBE_MIN_SPEEDUP_TEXT);
    std::printf(
        "n128_ms=%.9g n256_ms=%.9g speedup=%.9g abba_samples=4\n",
        timing.n128_ms, timing.n256_ms, timing.speedup);
    std::printf(
        "fp8_mismatches=0 scale_mismatches=0 guards=clean "
        "poison_a=clean poison_b=clean input_immutable=clean\n");
    std::printf("result=PASS\n");
    return 0;
}
