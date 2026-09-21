// SPDX-License-Identifier: AGPL-3.0-only

// Standalone composed promotion probe for the registered V4 Q-B/RoPE and
// strided-K-full FP8-cache kernels. It is not loaded by Atlas production code.

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#ifndef V4_QB_ROPE_CACHE_JOINT_PROBE_BUILD_ID
#error "V4_QB_ROPE_CACHE_JOINT_PROBE_BUILD_ID must bind a build receipt"
#endif

// BEGIN joint kernel contract
#include "../common/rms_norm.cu"
#include "../deepseek-v4-flash/nvfp4/mla_absorbed.cu"
#include "../common/rope.cu"
#include "../common/reshape_and_cache.cu"
#include "../deepseek-v4-flash/nvfp4/v4_prefill_qb_norm_rope_fused.cu"
#include "../deepseek-v4-flash/nvfp4/v4_prefill_cache_kfull_fp8_fused.cu"

namespace joint_probe {
constexpr unsigned kTokens = 2410;
constexpr unsigned kNq = 64;
constexpr unsigned kNkv = 1;
constexpr unsigned kHeadDim = 512;
constexpr unsigned kNope = 448;
constexpr unsigned kRope = 64;
constexpr unsigned kKvLora = 512;
constexpr unsigned kCacheDim = 576;
constexpr unsigned kBlockSize = 16;
constexpr unsigned kNumBlocks = 151;
constexpr unsigned kNormThreads = 512;
constexpr unsigned kCopyThreads = 256;
constexpr unsigned kRopeThreads = 128;
constexpr unsigned kQExtractCtas = 38560;
constexpr unsigned kKExtractCtas = 603;
constexpr unsigned kRopeSeqCtas = 603;
constexpr unsigned long long kCacheStride = kBlockSize * kCacheDim;
constexpr int kParityCases = 2;
constexpr int kPoisonCases = 2;
constexpr int kWarmupRounds = 2;
constexpr int kAbbaRounds = 6;
static_assert((kTokens * kNq * kRope + kCopyThreads - 1) / kCopyThreads ==
                  kQExtractCtas,
              "Q extraction launch changed");
static_assert((kTokens * kRope + kCopyThreads - 1) / kCopyThreads ==
                  kKExtractCtas,
              "K extraction launch changed");
static_assert((kTokens + 3) / 4 == kRopeSeqCtas,
              "interleaved RoPE launch changed");

// BEGIN joint launch ABIs
static void launch_baseline_chain(
    __nv_bfloat16* q, __nv_bfloat16* k, const __nv_bfloat16* zero_weight,
    const unsigned* positions, const float* inv_freq, const __nv_bfloat16* latent,
    const long long* slots, __nv_bfloat16* q_rope, __nv_bfloat16* k_rope,
    __nv_bfloat16* k_assembled, __nv_bfloat16* v_assembled,
    unsigned char* k_cache, unsigned char* v_cache, float eps, float mscale,
    float k_scale, float v_scale) {
    rms_norm<<<dim3(kTokens * kNq, 1, 1), dim3(kNormThreads, 1, 1)>>>(
        q, zero_weight, q, kHeadDim, eps);
    mla_q_rope_extract_batched<<<dim3(kQExtractCtas, 1, 1),
                                 dim3(kCopyThreads, 1, 1)>>>(
        q, q_rope, kTokens, kNq, kHeadDim, kNope, kRope, kNq * kHeadDim);
    mla_q_rope_extract_batched<<<dim3(kKExtractCtas, 1, 1),
                                 dim3(kCopyThreads, 1, 1)>>>(
        k, k_rope, kTokens, kNkv, kHeadDim, kNope, kRope, kHeadDim);
    rope_forward_yarn_interleaved<<<dim3(kNq + kNkv, kRopeSeqCtas, 1),
                                    dim3(kRopeThreads, 1, 1)>>>(
        q_rope, k_rope, positions, kTokens, kNq, kNkv, kRope, kRope,
        inv_freq, mscale);
    mla_q_rope_writeback_batched<<<dim3(kQExtractCtas, 1, 1),
                                   dim3(kCopyThreads, 1, 1)>>>(
        q_rope, q, kTokens, kNq, kHeadDim, kNope, kRope, kNq * kHeadDim);
    mla_q_rope_writeback_batched<<<dim3(kKExtractCtas, 1, 1),
                                   dim3(kCopyThreads, 1, 1)>>>(
        k_rope, k, kTokens, kNkv, kHeadDim, kNope, kRope, kHeadDim);
    mla_cache_assemble_batched<<<dim3(kTokens, 1, 1),
                                 dim3(kCacheDim, 1, 1)>>>(
        latent, k_rope, k_assembled, v_assembled, kKvLora, kRope, kCacheDim);
    reshape_and_cache_flash_fp8<<<dim3(kTokens, 1, 1),
                                  dim3(kCopyThreads, 1, 1)>>>(
        k_assembled, v_assembled, k_cache, v_cache, slots, kNkv, kCacheDim,
        kBlockSize, k_scale, v_scale, kCacheDim, kCacheDim, kCacheStride);
}

static void launch_candidate_chain(
    __nv_bfloat16* q, __nv_bfloat16* k, const __nv_bfloat16* zero_weight,
    const unsigned* positions, const float* inv_freq, const __nv_bfloat16* latent,
    const long long* slots, unsigned char* k_cache, unsigned char* v_cache,
    float eps, float mscale, float k_scale, float v_scale) {
    v4_prefill_qb_norm_rope_fused<<<dim3(kTokens, kNq, 1),
                                    dim3(kNormThreads, 1, 1)>>>(
        q, k, zero_weight, positions, inv_freq, kTokens, kNq, kNkv,
        kHeadDim, kNope, kRope, eps, mscale);
    v4_prefill_cache_kfull_fp8_fused<<<dim3(kTokens, 1, 1),
                                       dim3(kCopyThreads, 1, 1)>>>(
        latent, k, k_cache, v_cache, slots, kTokens, kNumBlocks, kBlockSize,
        k_scale, v_scale, kCacheStride);
}
// launch_abi_count=2: the candidate has no extraction or assembly scratch.
// END joint launch ABIs
// END joint kernel contract

[[noreturn]] static void die(const char* message) {
    std::fprintf(stderr, "%s\n", message);
    std::exit(1);
}

static void check(cudaError_t error, const char* where) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", where, cudaGetErrorString(error));
        std::exit(1);
    }
}

static uint64_t fnv(uint64_t hash, const void* data, size_t bytes) {
    const auto* input = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < bytes; ++i) {
        hash = (hash ^ input[i]) * 1099511628211ull;
    }
    return hash;
}

template <class T>
struct Guarded {
    static constexpr size_t kGuardBytes = 256;
    unsigned char* allocation = nullptr;
    T* data = nullptr;
    size_t count = 0;

    explicit Guarded(size_t elements) : count(elements) {
        check(cudaMalloc(&allocation, bytes() + 2 * kGuardBytes), "cudaMalloc");
        data = reinterpret_cast<T*>(allocation + kGuardBytes);
        check(cudaMemset(allocation, 0xa7, kGuardBytes), "prefix guard");
        check(cudaMemset(allocation + kGuardBytes + bytes(), 0x7a, kGuardBytes),
              "suffix guard");
    }
    Guarded(const Guarded&) = delete;
    Guarded& operator=(const Guarded&) = delete;
    ~Guarded() { cudaFree(allocation); }
    size_t bytes() const { return count * sizeof(T); }
    uintptr_t begin() const { return reinterpret_cast<uintptr_t>(allocation); }
    uintptr_t end() const { return begin() + bytes() + 2 * kGuardBytes; }
    void fill(unsigned char poison) {
        check(cudaMemset(data, poison, bytes()), "poison fill");
    }
    void upload(const std::vector<T>& host) {
        if (host.size() != count) die("host/device extent mismatch");
        check(cudaMemcpy(data, host.data(), bytes(), cudaMemcpyHostToDevice), "upload");
    }
    std::vector<unsigned char> download() const {
        std::vector<unsigned char> host(bytes());
        check(cudaMemcpy(host.data(), data, bytes(), cudaMemcpyDeviceToHost),
              "download");
        return host;
    }
    bool guards_clean() const {
        std::vector<unsigned char> prefix(kGuardBytes), suffix(kGuardBytes);
        check(cudaMemcpy(prefix.data(), allocation, kGuardBytes,
                         cudaMemcpyDeviceToHost),
              "prefix guard download");
        check(cudaMemcpy(suffix.data(), allocation + kGuardBytes + bytes(),
                         kGuardBytes, cudaMemcpyDeviceToHost),
              "suffix guard download");
        return std::all_of(prefix.begin(), prefix.end(),
                           [](unsigned char byte) { return byte == 0xa7; }) &&
               std::all_of(suffix.begin(), suffix.end(),
                           [](unsigned char byte) { return byte == 0x7a; });
    }
};

struct CachePair {
    Guarded<unsigned char> k{kNumBlocks * kCacheStride};
    Guarded<unsigned char> v{kNumBlocks * kCacheStride};
    void fill(unsigned char poison) { k.fill(poison); v.fill(poison); }
    bool guards_clean() const { return k.guards_clean() && v.guards_clean(); }
};

struct ProbeState {
    Guarded<__nv_bfloat16> baseline_q{static_cast<size_t>(kTokens) * kNq * kHeadDim};
    Guarded<__nv_bfloat16> candidate_q{static_cast<size_t>(kTokens) * kNq * kHeadDim};
    Guarded<__nv_bfloat16> baseline_k{static_cast<size_t>(kTokens) * kHeadDim};
    Guarded<__nv_bfloat16> candidate_k{static_cast<size_t>(kTokens) * kHeadDim};
    Guarded<__nv_bfloat16> zero_weight{kHeadDim};
    Guarded<unsigned> positions{kTokens};
    Guarded<float> inv_freq{kRope / 2};
    Guarded<__nv_bfloat16> latent{static_cast<size_t>(kTokens) * kKvLora};
    Guarded<long long> slots{kTokens};
    Guarded<__nv_bfloat16> q_rope{static_cast<size_t>(kTokens) * kNq * kRope};
    Guarded<__nv_bfloat16> k_rope{static_cast<size_t>(kTokens) * kRope};
    Guarded<__nv_bfloat16> k_assembled{static_cast<size_t>(kTokens) * kCacheDim};
    Guarded<__nv_bfloat16> v_assembled{static_cast<size_t>(kTokens) * kCacheDim};
    Guarded<unsigned char> candidate_no_scratch{4096};
    CachePair baseline_cache;
    CachePair candidate_cache;

    bool allocations_disjoint() const {
        std::vector<std::pair<uintptr_t, uintptr_t>> ranges = {
            {baseline_q.begin(), baseline_q.end()}, {candidate_q.begin(), candidate_q.end()},
            {baseline_k.begin(), baseline_k.end()}, {candidate_k.begin(), candidate_k.end()},
            {zero_weight.begin(), zero_weight.end()}, {positions.begin(), positions.end()},
            {inv_freq.begin(), inv_freq.end()}, {latent.begin(), latent.end()},
            {slots.begin(), slots.end()}, {q_rope.begin(), q_rope.end()},
            {k_rope.begin(), k_rope.end()}, {k_assembled.begin(), k_assembled.end()},
            {v_assembled.begin(), v_assembled.end()},
            {candidate_no_scratch.begin(), candidate_no_scratch.end()},
            {baseline_cache.k.begin(), baseline_cache.k.end()},
            {baseline_cache.v.begin(), baseline_cache.v.end()},
            {candidate_cache.k.begin(), candidate_cache.k.end()},
            {candidate_cache.v.begin(), candidate_cache.v.end()},
        };
        std::sort(ranges.begin(), ranges.end());
        for (size_t i = 1; i < ranges.size(); ++i) {
            if (ranges[i - 1].second > ranges[i].first) return false;
        }
        return true;
    }

    bool all_guards_clean() const {
        return baseline_q.guards_clean() && candidate_q.guards_clean() &&
               baseline_k.guards_clean() && candidate_k.guards_clean() &&
               zero_weight.guards_clean() && positions.guards_clean() &&
               inv_freq.guards_clean() && latent.guards_clean() &&
               slots.guards_clean() && q_rope.guards_clean() &&
               k_rope.guards_clean() && k_assembled.guards_clean() &&
               v_assembled.guards_clean() && candidate_no_scratch.guards_clean() &&
               baseline_cache.guards_clean() && candidate_cache.guards_clean();
    }
};

static uint16_t bf16_bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    if ((bits & 0x7f800000u) == 0x7f800000u) {
        return static_cast<uint16_t>(bits >> 16);
    }
    const uint32_t bias = 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>((bits + bias) >> 16);
}

// BEGIN joint deterministic parity
// kParityCases = 2: "main" and "compressor-yarn" position/frequency/mscale.
// kPoisonCases = 2 with full Q, full K, valid cache bytes, holes and tails.
constexpr unsigned char kPoisonA = 0x5a;
constexpr unsigned char kPoisonB = 0xc3;

struct CaseSpec {
    const char* name;
    float eps;
    float mscale;
    float kScale;
    float vScale;
    double theta;
    unsigned salt;
};

constexpr CaseSpec kCases[kParityCases] = {
    {"main", 1.0e-6f, 1.0f, 0.5f, 2.0f, 10000.0, 17},
    {"compressor-yarn", 1.0e-5f, 1.125f, 0.0625f, 3.25f, 160000.0, 73},
};

struct HostCase {
    std::vector<__nv_bfloat16> q{static_cast<size_t>(kTokens) * kNq * kHeadDim};
    std::vector<__nv_bfloat16> k{static_cast<size_t>(kTokens) * kHeadDim};
    std::vector<__nv_bfloat16> weight{kHeadDim};
    std::vector<unsigned> positions{kTokens};
    std::vector<float> inv_freq{kRope / 2};
    std::vector<__nv_bfloat16> latent{static_cast<size_t>(kTokens) * kKvLora};
    std::vector<long long> slots{kTokens};
};

static void make_case(const CaseSpec& spec, HostCase& host) {
    auto* q = reinterpret_cast<uint16_t*>(host.q.data());
    auto* k = reinterpret_cast<uint16_t*>(host.k.data());
    auto* latent = reinterpret_cast<uint16_t*>(host.latent.data());
    std::fill(host.weight.begin(), host.weight.end(), __float2bfloat16(0.0f));
    for (size_t i = 0; i < host.q.size(); ++i) {
        const int value = static_cast<int>((i * 131u + spec.salt * 19u) % 1009u) - 504;
        q[i] = bf16_bits(static_cast<float>(value) * (1.0f / 128.0f));
    }
    for (size_t i = 0; i < host.k.size(); ++i) {
        const int value = static_cast<int>((i * 193u + spec.salt * 23u) % 509u) - 254;
        k[i] = bf16_bits(static_cast<float>(value) * (1.0f / 64.0f));
    }
    for (size_t i = 0; i < host.latent.size(); ++i) {
        const int value = static_cast<int>((i * 97u + spec.salt * 29u) % 997u) - 498;
        latent[i] = bf16_bits(static_cast<float>(value) * (1.0f / 64.0f));
    }
    for (unsigned token = 0; token < kTokens; ++token) {
        host.positions[token] = spec.salt == 17
                                    ? (token * 29u + 3u) % 32768u
                                    : (token * 509u + 100003u) % 262139u;
        long long slot = static_cast<long long>((token * 37u) % kTokens);
        if ((token + spec.salt) % 257u == 0) slot = -1;
        host.slots[token] = slot;
    }
    for (unsigned pair = 0; pair < kRope / 2; ++pair) {
        host.inv_freq[pair] = static_cast<float>(
            1.0 / std::pow(spec.theta, (2.0 * pair) / static_cast<double>(kRope)));
    }
}

static std::vector<unsigned char> validate_slots(const std::vector<long long>& slots) {
    std::vector<unsigned char> written(kNumBlocks * kBlockSize, 0);
    size_t valid = 0;
    size_t negative = 0;
    for (long long slot : slots) {
        if (slot < 0) { ++negative; continue; }
        if (static_cast<unsigned long long>(slot) >= written.size()) {
            die("parity slot exceeds cache pool");
        }
        if (written[static_cast<size_t>(slot)] != 0) die("parity slot collision");
        written[static_cast<size_t>(slot)] = 1;
        ++valid;
    }
    for (size_t slot = kTokens; slot < written.size(); ++slot) {
        if (slot >= kTokens && written[slot] != 0) die("cache tail was mapped");
    }
    if (valid == 0 || negative == 0 || valid + negative != slots.size()) {
        die("slot coverage incomplete");
    }
    return written;
}

struct InputSnapshot {
    std::vector<unsigned char> weight;
    std::vector<unsigned char> positions;
    std::vector<unsigned char> frequencies;
    std::vector<unsigned char> latent;
    std::vector<unsigned char> slots;
};

static InputSnapshot snapshot_inputs(const ProbeState& state) {
    return {state.zero_weight.download(), state.positions.download(),
            state.inv_freq.download(), state.latent.download(), state.slots.download()};
}

static bool immutable_inputs_clean(const ProbeState& state,
                                   const InputSnapshot& before) {
    const InputSnapshot after = snapshot_inputs(state);
    return before.weight == after.weight && before.positions == after.positions &&
           before.frequencies == after.frequencies && before.latent == after.latent &&
           before.slots == after.slots;
}

static size_t mismatch_bytes(const std::vector<unsigned char>& baseline,
                             const std::vector<unsigned char>& candidate) {
    if (baseline.size() != candidate.size()) die("comparison extent mismatch");
    size_t mismatches = 0;
    for (size_t i = 0; i < baseline.size(); ++i) {
        mismatches += baseline[i] != candidate[i];
    }
    return mismatches;
}

struct CacheAudit {
    size_t mismatches = 0;
    size_t valid_bytes = 0;
    size_t hole_bytes = 0;
};

static CacheAudit audit_cache(const std::vector<unsigned char>& baseline,
                              const std::vector<unsigned char>& candidate,
                              unsigned char baseline_poison,
                              unsigned char candidate_poison,
                              const std::vector<unsigned char>& written) {
    if (baseline.size() != kNumBlocks * kCacheStride ||
        candidate.size() != baseline.size() ||
        written.size() != kNumBlocks * kBlockSize ||
        baseline_poison == candidate_poison) {
        die("invalid cache audit");
    }
    CacheAudit audit;
    for (size_t slot = 0; slot < written.size(); ++slot) {
        const size_t row = (slot / kBlockSize) * kCacheStride +
                           (slot % kBlockSize) * kCacheDim;
        for (size_t dim = 0; dim < kCacheDim; ++dim) {
            const size_t index = row + dim;
            if (written[slot] != 0) {
                audit.mismatches += baseline[index] != candidate[index];
                ++audit.valid_bytes;
            } else {
                if (baseline[index] != baseline_poison) {
                    die("baseline hole or tail changed");
                }
                if (candidate[index] != candidate_poison) {
                    die("candidate hole or tail changed");
                }
                ++audit.hole_bytes;
            }
        }
    }
    return audit;
}

struct ParityResult {
    size_t q_mismatches = 0;
    size_t k_mismatches = 0;
    size_t cache_k_mismatches = 0;
    size_t cache_v_mismatches = 0;
    size_t valid_bytes = 0;
    size_t hole_bytes = 0;
    bool immutable = true;
    bool guards = true;
    bool candidate_scratch = true;
    bool disjoint = true;
    uint64_t input_hash = 1469598103934665603ull;
    uint64_t position_hash = 1469598103934665603ull;
    uint64_t frequency_hash = 1469598103934665603ull;
    uint64_t slot_hash = 1469598103934665603ull;
    uint64_t scale_hash = 1469598103934665603ull;
};

static void upload_read_inputs(ProbeState& state, const HostCase& host) {
    state.zero_weight.upload(host.weight);
    state.positions.upload(host.positions);
    state.inv_freq.upload(host.inv_freq);
    state.latent.upload(host.latent);
    state.slots.upload(host.slots);
}

static ParityResult run_parity(ProbeState& state, HostCase& host) {
    ParityResult result;
    result.disjoint = state.allocations_disjoint();
    for (const CaseSpec& spec : kCases) {
        make_case(spec, host);
        const auto written = validate_slots(host.slots);
        upload_read_inputs(state, host);
        const InputSnapshot before = snapshot_inputs(state);
        result.input_hash = fnv(result.input_hash, host.q.data(), host.q.size() * sizeof(host.q[0]));
        result.input_hash = fnv(result.input_hash, host.k.data(), host.k.size() * sizeof(host.k[0]));
        result.input_hash = fnv(result.input_hash, host.latent.data(), host.latent.size() * sizeof(host.latent[0]));
        result.position_hash = fnv(result.position_hash, host.positions.data(), host.positions.size() * sizeof(host.positions[0]));
        result.frequency_hash = fnv(result.frequency_hash, host.inv_freq.data(), host.inv_freq.size() * sizeof(host.inv_freq[0]));
        result.slot_hash = fnv(result.slot_hash, host.slots.data(), host.slots.size() * sizeof(host.slots[0]));
        result.scale_hash = fnv(result.scale_hash, &spec.eps, sizeof(spec.eps));
        result.scale_hash = fnv(result.scale_hash, &spec.mscale, sizeof(spec.mscale));
        result.scale_hash = fnv(result.scale_hash, &spec.kScale, sizeof(spec.kScale));
        result.scale_hash = fnv(result.scale_hash, &spec.vScale, sizeof(spec.vScale));
        for (int poison_pass = 0; poison_pass < kPoisonCases; ++poison_pass) {
            const unsigned char baseline_poison = poison_pass == 0 ? kPoisonA : kPoisonB;
            const unsigned char candidate_poison = poison_pass == 0 ? kPoisonB : kPoisonA;
            state.baseline_q.upload(host.q); state.candidate_q.upload(host.q);
            state.baseline_k.upload(host.k); state.candidate_k.upload(host.k);
            state.q_rope.fill(baseline_poison); state.k_rope.fill(baseline_poison);
            state.k_assembled.fill(baseline_poison); state.v_assembled.fill(baseline_poison);
            state.baseline_cache.fill(baseline_poison);
            state.candidate_cache.fill(candidate_poison);
            state.candidate_no_scratch.fill(candidate_poison);
            const auto no_scratch_before = state.candidate_no_scratch.download();
            launch_baseline_chain(
                state.baseline_q.data, state.baseline_k.data, state.zero_weight.data,
                state.positions.data, state.inv_freq.data, state.latent.data,
                state.slots.data, state.q_rope.data, state.k_rope.data,
                state.k_assembled.data, state.v_assembled.data,
                state.baseline_cache.k.data, state.baseline_cache.v.data,
                spec.eps, spec.mscale, spec.kScale, spec.vScale);
            launch_candidate_chain(
                state.candidate_q.data, state.candidate_k.data, state.zero_weight.data,
                state.positions.data, state.inv_freq.data, state.latent.data,
                state.slots.data, state.candidate_cache.k.data,
                state.candidate_cache.v.data, spec.eps, spec.mscale,
                spec.kScale, spec.vScale);
            check(cudaGetLastError(), "joint parity launch");
            check(cudaDeviceSynchronize(), "joint parity sync");
            result.q_mismatches += mismatch_bytes(state.baseline_q.download(), state.candidate_q.download());
            result.k_mismatches += mismatch_bytes(state.baseline_k.download(), state.candidate_k.download());
            const CacheAudit k_audit = audit_cache(
                state.baseline_cache.k.download(), state.candidate_cache.k.download(),
                baseline_poison, candidate_poison, written);
            const CacheAudit v_audit = audit_cache(
                state.baseline_cache.v.download(), state.candidate_cache.v.download(),
                baseline_poison, candidate_poison, written);
            result.cache_k_mismatches += k_audit.mismatches;
            result.cache_v_mismatches += v_audit.mismatches;
            result.valid_bytes += k_audit.valid_bytes + v_audit.valid_bytes;
            result.hole_bytes += k_audit.hole_bytes + v_audit.hole_bytes;
            result.immutable &= immutable_inputs_clean(state, before);
            result.guards &= state.all_guards_clean();
            result.candidate_scratch &=
                state.candidate_no_scratch.download() == no_scratch_before;
        }
    }
    if (result.q_mismatches || result.k_mismatches || result.cache_k_mismatches ||
        result.cache_v_mismatches || !result.immutable || !result.guards ||
        !result.candidate_scratch || !result.disjoint) {
        die("joint exact parity, immutable input, guard, or alias failure");
    }
    return result;
}
// END joint deterministic parity

static double parse_speedup_threshold(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <min_speedup>\n", argv[0]);
        std::exit(2);
    }
    const char* input = argv[1];
    bool digit = false, dot = false, exponent = false, exponent_digit = false;
    for (size_t i = 0; input[i] != '\0'; ++i) {
        const unsigned char ch = static_cast<unsigned char>(input[i]);
        if (std::isdigit(ch)) { digit = true; if (exponent) exponent_digit = true; }
        else if (ch == '.' && !dot && !exponent) dot = true;
        else if ((ch == 'e' || ch == 'E') && digit && !exponent) exponent = true;
        else if ((ch == '+' || ch == '-') && exponent && !exponent_digit && i > 0 &&
                 (input[i - 1] == 'e' || input[i - 1] == 'E')) continue;
        else { std::fprintf(stderr, "invalid explicit numeric threshold\n"); std::exit(2); }
    }
    if (!digit || (exponent && !exponent_digit)) {
        std::fprintf(stderr, "invalid explicit numeric threshold\n"); std::exit(2);
    }
    errno = 0;
    char* end = nullptr;
    const double threshold = std::strtod(input, &end);
    if (errno != 0 || end == input || *end != '\0' || !std::isfinite(threshold) ||
        threshold <= 1.0 || threshold > 100.0) {
        std::fprintf(stderr, "invalid explicit numeric threshold\n"); std::exit(2);
    }
    return threshold;
}

// BEGIN joint ABBA timing
// kAbbaRounds = 6; whole-chain order is baseline, candidate, candidate, baseline.
struct TimingContext {
    ProbeState& state;
    const HostCase& host;
    const CaseSpec& spec;

    void reset_baseline() {
        state.baseline_q.upload(host.q); state.baseline_k.upload(host.k);
        state.q_rope.fill(kPoisonA); state.k_rope.fill(kPoisonA);
        state.k_assembled.fill(kPoisonA); state.v_assembled.fill(kPoisonA);
        state.baseline_cache.fill(kPoisonA);
    }
    void reset_candidate() {
        state.candidate_q.upload(host.q); state.candidate_k.upload(host.k);
        state.candidate_cache.fill(kPoisonB); state.candidate_no_scratch.fill(kPoisonB);
    }
    float time_baseline() {
        cudaEvent_t start, finish;
        check(cudaEventCreate(&start), "event create");
        check(cudaEventCreate(&finish), "event create");
        reset_baseline(); check(cudaEventRecord(start), "baseline event start");
        launch_baseline_chain(); check(cudaEventRecord(finish), "baseline event finish");
        check(cudaGetLastError(), "baseline timing launch");
        check(cudaEventSynchronize(finish), "baseline event sync");
        float elapsed = 0.0f;
        check(cudaEventElapsedTime(&elapsed, start, finish), "cudaEventElapsedTime");
        check(cudaEventDestroy(start), "event destroy");
        check(cudaEventDestroy(finish), "event destroy");
        return elapsed;
    }
    float time_candidate() {
        cudaEvent_t start, finish;
        check(cudaEventCreate(&start), "event create");
        check(cudaEventCreate(&finish), "event create");
        reset_candidate(); check(cudaEventRecord(start), "candidate event start");
        launch_candidate_chain(); check(cudaEventRecord(finish), "candidate event finish");
        check(cudaGetLastError(), "candidate timing launch");
        check(cudaEventSynchronize(finish), "candidate event sync");
        float elapsed = 0.0f;
        check(cudaEventElapsedTime(&elapsed, start, finish), "cudaEventElapsedTime");
        check(cudaEventDestroy(start), "event destroy");
        check(cudaEventDestroy(finish), "event destroy");
        return elapsed;
    }

private:
    void launch_baseline_chain() {
        joint_probe::launch_baseline_chain(
            state.baseline_q.data, state.baseline_k.data, state.zero_weight.data,
            state.positions.data, state.inv_freq.data, state.latent.data,
            state.slots.data, state.q_rope.data, state.k_rope.data,
            state.k_assembled.data, state.v_assembled.data,
            state.baseline_cache.k.data, state.baseline_cache.v.data,
            spec.eps, spec.mscale, spec.kScale, spec.vScale);
    }
    void launch_candidate_chain() {
        joint_probe::launch_candidate_chain(
            state.candidate_q.data, state.candidate_k.data, state.zero_weight.data,
            state.positions.data, state.inv_freq.data, state.latent.data,
            state.slots.data, state.candidate_cache.k.data,
            state.candidate_cache.v.data, spec.eps, spec.mscale,
            spec.kScale, spec.vScale);
    }
};

static void time_abba(TimingContext& context, double& baseline_ms,
                      double& candidate_ms) {
    for (int round = 0; round < kWarmupRounds; ++round) {
        context.time_baseline(); context.time_candidate();
        context.time_candidate(); context.time_baseline();
    }
    baseline_ms = 0.0;
    candidate_ms = 0.0;
    for (int round = 0; round < kAbbaRounds; ++round) {
        baseline_ms += context.time_baseline();
        candidate_ms += context.time_candidate();
        candidate_ms += context.time_candidate();
        baseline_ms += context.time_baseline();
    }
    baseline_ms /= 2 * kAbbaRounds;
    candidate_ms /= 2 * kAbbaRounds;
}
// END joint ABBA timing

static int run(int argc, char** argv) {
    const double min_speedup = parse_speedup_threshold(argc, argv);
    int driver = 0, runtime = 0, device = 0;
    cudaDeviceProp properties{};
    check(cudaDriverGetVersion(&driver), "driver version");
    check(cudaRuntimeGetVersion(&runtime), "runtime version");
    check(cudaGetDevice(&device), "active device");
    check(cudaGetDeviceProperties(&properties, device), "device properties");

    ProbeState state;
    HostCase host;
    const ParityResult parity = run_parity(state, host);
    make_case(kCases[0], host);
    upload_read_inputs(state, host);
    TimingContext timing{state, host, kCases[0]};
    double baseline_ms = 0.0, candidate_ms = 0.0;
    time_abba(timing, baseline_ms, candidate_ms);
    const double speedup = baseline_ms / candidate_ms;
    if (!std::isfinite(speedup) || speedup < min_speedup) {
        die("pre-registered joint speedup threshold not met");
    }

    char uuid[33];
    for (int i = 0; i < 16; ++i) {
        std::sprintf(uuid + 2 * i, "%02x",
                     static_cast<unsigned char>(properties.uuid.bytes[i]));
    }
    uuid[32] = '\0';
    std::printf("build_id=%s\n", V4_QB_ROPE_CACHE_JOINT_PROBE_BUILD_ID);
    std::printf("device_uuid=%s driver=%d runtime=%d\n", uuid, driver, runtime);
    std::printf("input_hash=%016llx position_hash=%016llx frequency_hash=%016llx slot_hash=%016llx scale_hash=%016llx\n",
                static_cast<unsigned long long>(parity.input_hash),
                static_cast<unsigned long long>(parity.position_hash),
                static_cast<unsigned long long>(parity.frequency_hash),
                static_cast<unsigned long long>(parity.slot_hash),
                static_cast<unsigned long long>(parity.scale_hash));
    std::printf("parity cases=2 poison_cases=2 q_mismatches=0 k_mismatches=0 cache_k_mismatches=0 cache_v_mismatches=0 valid_bytes=%zu hole_bytes=%zu\n",
                parity.valid_bytes, parity.hole_bytes);
    std::printf("invariants immutable=clean guards=clean candidate_no_scratch=clean allocations_disjoint=yes launch_abis=2\n");
    std::printf("cases main=pass compressor-yarn=pass poisons=swapped cache_tail_slots=6\n");
    std::printf("timing baseline_ms=%.6f candidate_ms=%.6f speedup=%.9f abba_rounds=6\n",
                baseline_ms, candidate_ms, speedup);
    std::printf("threshold min_speedup=%.17g\n", min_speedup);
    std::printf("result=PASS\n");
    return 0;
}
}  // namespace joint_probe

int main(int argc, char** argv) { return joint_probe::run(argc, argv); }
