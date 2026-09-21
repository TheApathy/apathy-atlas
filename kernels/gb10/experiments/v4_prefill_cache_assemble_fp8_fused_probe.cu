// SPDX-License-Identifier: AGPL-3.0-only

// Standalone GPU qualification probe for the isolated V4 prefill cache fusion.
// It is compiled and receipted offline, but is not registered with Atlas.

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#ifndef V4_CACHE_FP8_PROBE_BUILD_ID
#error "V4_CACHE_FP8_PROBE_BUILD_ID must bind this probe to a build receipt"
#endif

// BEGIN cache fusion probe kernel contract
#include "../deepseek-v4-flash/nvfp4/mla_absorbed.cu"
#include "../common/reshape_and_cache.cu"
#include "v4_prefill_cache_assemble_fp8_fused.cu"

namespace probe {
constexpr unsigned kTokens = 2410;
constexpr unsigned kKvLora = 512;
constexpr unsigned kRope = 64;
constexpr unsigned kCacheDim = 576;
constexpr unsigned kBlockSize = 16;
constexpr unsigned kThreads = 256;
constexpr unsigned kNumBlocks = 151;
constexpr unsigned long long kCacheStride = kBlockSize * kCacheDim;
constexpr int kParityCases = 3;
constexpr int kPoisonCases = 2;
constexpr int kMalformedCases = 25;
constexpr int kWarmupRounds = 2;
constexpr int kAbbaRounds = 6;
static_assert(kMalformedCases == 25, "malformed ABI census changed");

static void launch_baseline(
    const __nv_bfloat16* latent,
    const __nv_bfloat16* rope,
    __nv_bfloat16* k_scratch,
    __nv_bfloat16* v_scratch,
    unsigned char* k_cache,
    unsigned char* v_cache,
    const long long* slots,
    float k_scale,
    float v_scale) {
    mla_cache_assemble_batched<<<dim3(kTokens, 1, 1), dim3(kCacheDim, 1, 1)>>>(
        latent, rope, k_scratch, v_scratch, kKvLora, kRope, kCacheDim);
    reshape_and_cache_flash_fp8<<<dim3(kTokens, 1, 1), dim3(kThreads, 1, 1)>>>(
        k_scratch, v_scratch, k_cache, v_cache, slots, 1, kCacheDim,
        kBlockSize, k_scale, v_scale, kCacheDim, kCacheDim, kCacheStride);
}

static void launch_candidate(
    const __nv_bfloat16* latent,
    const __nv_bfloat16* rope,
    unsigned char* k_cache,
    unsigned char* v_cache,
    const long long* slots,
    float k_scale,
    float v_scale) {
    v4_prefill_cache_assemble_fp8_fused<<<dim3(kTokens, 1, 1), dim3(kThreads, 1, 1)>>>(
        latent, rope, k_cache, v_cache, slots, kTokens, kNumBlocks,
        kBlockSize, k_scale, v_scale, kCacheStride);
}
// END cache fusion probe kernel contract

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
    const auto* ptr = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < bytes; ++i) {
        hash = (hash ^ ptr[i]) * 1099511628211ull;
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
    ~Guarded() { cudaFree(allocation); }
    size_t bytes() const { return count * sizeof(T); }
    void fill(unsigned char poison) {
        check(cudaMemset(data, poison, bytes()), "poison fill");
    }
    void upload(const std::vector<T>& host) {
        if (host.size() != count) {
            die("host/device extent mismatch");
        }
        check(cudaMemcpy(data, host.data(), bytes(), cudaMemcpyHostToDevice), "upload");
    }
    std::vector<unsigned char> download() const {
        std::vector<unsigned char> host(bytes());
        check(cudaMemcpy(host.data(), data, bytes(), cudaMemcpyDeviceToHost), "download");
        return host;
    }
    bool guards_clean() const {
        std::vector<unsigned char> prefix(kGuardBytes), suffix(kGuardBytes);
        check(cudaMemcpy(prefix.data(), allocation, kGuardBytes, cudaMemcpyDeviceToHost),
              "prefix guard download");
        check(cudaMemcpy(suffix.data(), allocation + kGuardBytes + bytes(),
                         kGuardBytes, cudaMemcpyDeviceToHost),
              "suffix guard download");
        for (unsigned char byte : prefix) {
            if (byte != 0xa7) return false;
        }
        for (unsigned char byte : suffix) {
            if (byte != 0x7a) return false;
        }
        return true;
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

struct CachePair {
    Guarded<unsigned char> k{kNumBlocks * kCacheStride};
    Guarded<unsigned char> v{kNumBlocks * kCacheStride};
    void fill(unsigned char poison) { k.fill(poison); v.fill(poison); }
    bool guards_clean() const { return k.guards_clean() && v.guards_clean(); }
};

struct ProbeState {
    Guarded<__nv_bfloat16> latent{static_cast<size_t>(kTokens) * kKvLora};
    Guarded<__nv_bfloat16> rope{static_cast<size_t>(kTokens) * kRope};
    Guarded<long long> slots{kTokens};
    Guarded<__nv_bfloat16> k_scratch{static_cast<size_t>(kTokens) * kCacheDim};
    Guarded<__nv_bfloat16> v_scratch{static_cast<size_t>(kTokens) * kCacheDim};
    CachePair baseline;
    CachePair candidate;
};

// BEGIN cache fusion deterministic parity
// Contract: three independent-scale cases, two poisons, unique valid nonmonotonic slots.
// kParityCases = 3; kPoisonCases = 2.
constexpr unsigned char kPoisonA = 0x5a;
constexpr unsigned char kPoisonB = 0xc3;
struct CaseSpec {
    float kScale;
    float vScale;
    unsigned latent_salt;
    unsigned rope_salt;
};
constexpr CaseSpec kCases[kParityCases] = {
    {0.5f, 2.0f, 11, 47},
    {0.0625f, 3.25f, 29, 71},
    {1.0f, 0.125f, 43, 97},
};

static void make_case(const CaseSpec& spec,
                      std::vector<__nv_bfloat16>& latent,
                      std::vector<__nv_bfloat16>& rope,
                      std::vector<long long>& slots) {
    auto* latent_raw = reinterpret_cast<uint16_t*>(latent.data());
    auto* rope_raw = reinterpret_cast<uint16_t*>(rope.data());
    for (size_t i = 0; i < latent.size(); ++i) {
        const int value = static_cast<int>((i * 131u + spec.latent_salt * 17u) % 997u) - 498;
        float input = static_cast<float>(value) * (1.0f / 64.0f);
        if ((i + spec.latent_salt) % 131071u == 0) input = (i & 1) ? -0.0f : 0.0f;
        latent_raw[i] = bf16_bits(input);
    }
    for (size_t i = 0; i < rope.size(); ++i) {
        const int value = static_cast<int>((i * 193u + spec.rope_salt * 23u) % 509u) - 254;
        rope_raw[i] = bf16_bits(static_cast<float>(value) * (1.0f / 32.0f));
    }
    for (unsigned token = 0; token < kTokens; ++token) {
        long long slot = static_cast<long long>((token * 37u) % kTokens);
        if ((token + spec.latent_salt) % 257u == 0) slot = -1;
        slots[token] = slot;
    }
}

static std::vector<unsigned char> validate_parity_slots(
    const std::vector<long long>& slots) {
    std::vector<unsigned char> seen(kNumBlocks * kBlockSize, 0);
    size_t valid_slots = 0;
    size_t negative_slots = 0;
    for (long long slot : slots) {
        if (slot < 0) {
            ++negative_slots;
            continue;
        }
        if (static_cast<unsigned long long>(slot) >= seen.size()) {
            die("parity slot exceeds cache pool");
        }
        if (seen[static_cast<size_t>(slot)] != 0) {
            die("parity slot collision");
        }
        seen[static_cast<size_t>(slot)] = 1;
        ++valid_slots;
    }
    if (valid_slots == 0 || negative_slots == 0 ||
        valid_slots + negative_slots != slots.size()) {
        die("parity slot coverage is incomplete");
    }
    return seen;
}

struct CacheAudit {
    size_t mismatches = 0;
    size_t written_bytes = 0;
    size_t untouched_bytes = 0;
};

static CacheAudit audit_cache_bytes(
    const std::vector<unsigned char>& baseline,
    const std::vector<unsigned char>& candidate,
    unsigned char baseline_poison,
    unsigned char candidate_poison,
    const std::vector<unsigned char>& written_slots) {
    const size_t cache_bytes = kNumBlocks * kCacheStride;
    if (baseline.size() != cache_bytes || candidate.size() != cache_bytes ||
        written_slots.size() != kNumBlocks * kBlockSize ||
        baseline_poison == candidate_poison) {
        die("invalid opposite-poison cache audit");
    }
    CacheAudit audit;
    for (size_t slot = 0; slot < written_slots.size(); ++slot) {
        const size_t block_index = slot / kBlockSize;
        const size_t block_offset = slot % kBlockSize;
        const size_t row_offset = block_index * kCacheStride + block_offset * kCacheDim;
        for (size_t dim = 0; dim < kCacheDim; ++dim) {
            const size_t index = row_offset + dim;
            if (written_slots[slot] != 0) {
                audit.mismatches += baseline[index] != candidate[index];
                ++audit.written_bytes;
            } else {
                if (baseline[index] != baseline_poison) {
                    die("baseline untouched cache byte changed");
                }
                if (candidate[index] != candidate_poison) {
                    die("candidate untouched cache byte changed");
                }
                ++audit.untouched_bytes;
            }
        }
    }
    return audit;
}

struct ParityResult {
    size_t k_mismatches = 0;
    size_t v_mismatches = 0;
    bool scratch_guards_clean = true;
    bool cache_guards_clean = true;
    uint64_t input_hash = 1469598103934665603ull;
    uint64_t slot_hash = 1469598103934665603ull;
    uint64_t scale_hash = 1469598103934665603ull;
};

static ParityResult run_parity(
    ProbeState& state,
    std::vector<__nv_bfloat16>& host_latent,
    std::vector<__nv_bfloat16>& host_rope,
    std::vector<long long>& host_slots) {
    ParityResult result;
    for (const CaseSpec& spec : kCases) {
        make_case(spec, host_latent, host_rope, host_slots);
        const auto written_slots = validate_parity_slots(host_slots);
        const size_t valid_slots = static_cast<size_t>(
            std::count(written_slots.begin(), written_slots.end(), 1));
        const size_t expected_written = valid_slots * kCacheDim;
        const size_t expected_untouched =
            (written_slots.size() - valid_slots) * kCacheDim;
        state.latent.upload(host_latent);
        state.rope.upload(host_rope);
        state.slots.upload(host_slots);
        result.input_hash = fnv(result.input_hash, host_latent.data(),
                                host_latent.size() * sizeof(__nv_bfloat16));
        result.input_hash = fnv(result.input_hash, host_rope.data(),
                                host_rope.size() * sizeof(__nv_bfloat16));
        result.slot_hash = fnv(result.slot_hash, host_slots.data(),
                               host_slots.size() * sizeof(long long));
        result.scale_hash = fnv(result.scale_hash, &spec.kScale, sizeof(spec.kScale));
        result.scale_hash = fnv(result.scale_hash, &spec.vScale, sizeof(spec.vScale));
        for (int poison_pass = 0; poison_pass < kPoisonCases; ++poison_pass) {
            const unsigned char baseline_poison =
                poison_pass == 0 ? kPoisonA : kPoisonB;
            const unsigned char candidate_poison =
                poison_pass == 0 ? kPoisonB : kPoisonA;
            if (baseline_poison == candidate_poison) {
                die("opposite poison relationship collapsed");
            }
            state.k_scratch.fill(baseline_poison);
            state.v_scratch.fill(baseline_poison);
            state.baseline.fill(baseline_poison);
            state.candidate.fill(candidate_poison);
            launch_baseline(state.latent.data, state.rope.data, state.k_scratch.data,
                            state.v_scratch.data, state.baseline.k.data,
                            state.baseline.v.data, state.slots.data,
                            spec.kScale, spec.vScale);
            launch_candidate(state.latent.data, state.rope.data,
                             state.candidate.k.data, state.candidate.v.data,
                             state.slots.data, spec.kScale, spec.vScale);
            check(cudaGetLastError(), "parity launch");
            check(cudaDeviceSynchronize(), "parity sync");
            const auto baseline_k = state.baseline.k.download();
            const auto baseline_v = state.baseline.v.download();
            const auto candidate_k = state.candidate.k.download();
            const auto candidate_v = state.candidate.v.download();
            const CacheAudit k_audit =
                audit_cache_bytes(baseline_k, candidate_k, baseline_poison,
                                  candidate_poison, written_slots);
            const CacheAudit v_audit =
                audit_cache_bytes(baseline_v, candidate_v, baseline_poison,
                                  candidate_poison, written_slots);
            if (k_audit.written_bytes != expected_written ||
                v_audit.written_bytes != expected_written ||
                k_audit.untouched_bytes != expected_untouched ||
                v_audit.untouched_bytes != expected_untouched) {
                die("cache audit extent mismatch");
            }
            result.k_mismatches += k_audit.mismatches;
            result.v_mismatches += v_audit.mismatches;
            result.scratch_guards_clean &=
                state.k_scratch.guards_clean() && state.v_scratch.guards_clean();
            result.cache_guards_clean &=
                state.baseline.guards_clean() && state.candidate.guards_clean();
        }
    }
    if (result.k_mismatches != 0 || result.v_mismatches != 0 ||
        !result.scratch_guards_clean || !result.cache_guards_clean ||
        !state.latent.guards_clean() || !state.rope.guards_clean() ||
        !state.slots.guards_clean()) {
        die("cache parity or guard failure");
    }
    return result;
}
// END cache fusion deterministic parity

// BEGIN cache fusion malformed ABI
// Contract: kMalformedCases = 25 positive no-write checks.
struct CandidateArgs {
    const __nv_bfloat16* latent;
    const __nv_bfloat16* rope;
    unsigned char* k_cache;
    unsigned char* v_cache;
    const long long* slots;
    unsigned num_tokens = kTokens;
    unsigned num_blocks = kNumBlocks;
    unsigned block_size = kBlockSize;
    float k_scale = 0.5f;
    float v_scale = 2.0f;
    unsigned long long cache_stride = kCacheStride;
    dim3 grid = dim3(kTokens, 1, 1);
    dim3 block = dim3(kThreads, 1, 1);
};

static void expect_unchanged(CachePair& output, const CandidateArgs& args,
                             const char* label) {
    output.fill(0x3c);
    const auto before_k = output.k.download();
    const auto before_v = output.v.download();
    v4_prefill_cache_assemble_fp8_fused<<<args.grid, args.block>>>(
        args.latent, args.rope, args.k_cache, args.v_cache, args.slots,
        args.num_tokens, args.num_blocks, args.block_size, args.k_scale,
        args.v_scale, args.cache_stride);
    check(cudaGetLastError(), label);
    check(cudaDeviceSynchronize(), label);
    if (output.k.download() != before_k) die("full K cache changed");
    if (output.v.download() != before_v) die("full V cache changed");
    if (!output.guards_clean()) die("malformed guards_clean failure");
}

static void run_malformed(ProbeState& state, std::vector<long long>& host_slots) {
    CandidateArgs base{state.latent.data, state.rope.data, state.candidate.k.data,
                       state.candidate.v.data, state.slots.data};
    auto run = [&](CandidateArgs args, const char* label) {
        expect_unchanged(state.candidate, args, label);
    };
    { auto a = base; a.latent = nullptr; run(a, "null-latent"); }
    { auto a = base; a.rope = nullptr; run(a, "null-rope"); }
    { auto a = base; a.k_cache = nullptr; run(a, "null-k-cache"); }
    { auto a = base; a.v_cache = nullptr; run(a, "null-v-cache"); }
    { auto a = base; a.slots = nullptr; run(a, "null-slots"); }
    { auto a = base; a.v_cache = a.k_cache; run(a, "aliased-cache"); }
    { auto a = base; a.num_tokens = kTokens - 1; run(a, "wrong-tokens"); }
    { auto a = base; a.num_blocks = 0; run(a, "zero-blocks"); }
    { auto a = base; a.block_size = kBlockSize + 1; run(a, "wrong-block-size"); }
    { auto a = base; a.cache_stride = kCacheStride + 1; run(a, "wrong-cache-stride"); }
    { auto a = base; a.k_scale = 0.0f; run(a, "bad-k-scale"); }
    { auto a = base; a.v_scale = std::numeric_limits<float>::quiet_NaN(); run(a, "bad-v-scale"); }
    { auto a = base; a.block = dim3(128, 1, 1); run(a, "block-x"); }
    { auto a = base; a.block = dim3(128, 2, 1); run(a, "block-y"); }
    { auto a = base; a.block = dim3(128, 1, 2); run(a, "block-z"); }
    { auto a = base; a.grid = dim3(kTokens - 1, 1, 1); run(a, "grid-x-small"); }
    { auto a = base; a.grid = dim3(kTokens + 1, 1, 1); run(a, "grid-x-large"); }
    { auto a = base; a.grid = dim3(kTokens, 2, 1); run(a, "grid-y"); }
    { auto a = base; a.grid = dim3(kTokens, 1, 2); run(a, "grid-z"); }
    { auto a = base; a.latent = reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const unsigned char*>(a.latent) + 2); run(a, "unaligned-latent"); }
    { auto a = base; a.rope = reinterpret_cast<const __nv_bfloat16*>(reinterpret_cast<const unsigned char*>(a.rope) + 2); run(a, "unaligned-rope"); }
    { auto a = base; a.k_cache += 1; run(a, "unaligned-k-cache"); }
    { auto a = base; a.v_cache += 1; run(a, "unaligned-v-cache"); }
    { auto a = base; a.slots = reinterpret_cast<const long long*>(reinterpret_cast<const unsigned char*>(a.slots) + 4); run(a, "unaligned-slots"); }
    std::fill(host_slots.begin(), host_slots.end(), -1);
    host_slots[0] = static_cast<long long>(kNumBlocks) * kBlockSize;
    state.slots.upload(host_slots);
    run(base, "out-of-pool-slot");
}
// END cache fusion malformed ABI

static double parse_speedup_threshold(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <min_speedup>\n", argv[0]);
        std::exit(2);
    }
    const char* input = argv[1];
    bool digit = false;
    bool dot = false;
    bool exponent = false;
    bool exponent_digit = false;
    for (size_t i = 0; input[i] != '\0'; ++i) {
        const unsigned char ch = static_cast<unsigned char>(input[i]);
        if (std::isdigit(ch)) {
            digit = true;
            if (exponent) exponent_digit = true;
        } else if (ch == '.' && !dot && !exponent) {
            dot = true;
        } else if ((ch == 'e' || ch == 'E') && digit && !exponent) {
            exponent = true;
        } else if ((ch == '+' || ch == '-') && exponent && !exponent_digit &&
                   i > 0 && (input[i - 1] == 'e' || input[i - 1] == 'E')) {
            continue;
        } else {
            std::fprintf(stderr, "invalid explicit numeric threshold\n");
            std::exit(2);
        }
    }
    if (!digit || (exponent && !exponent_digit)) {
        std::fprintf(stderr, "invalid explicit numeric threshold\n");
        std::exit(2);
    }
    errno = 0;
    char* end = nullptr;
    const double min_speedup = std::strtod(input, &end);
    if (errno != 0 || end == input || *end != '\0' || !std::isfinite(min_speedup) ||
        min_speedup <= 1.0 || min_speedup > 100.0) {
        std::fprintf(stderr, "invalid explicit numeric threshold\n");
        std::exit(2);
    }
    return min_speedup;
}

// BEGIN cache fusion ABBA timing
// Contract: kWarmupRounds = 2 and kAbbaRounds = 6; baseline_ms,
// candidate_ms, and speedup are computed from paired event samples.
struct TimingContext {
    ProbeState& state;
    const std::vector<__nv_bfloat16>& host_latent;
    const std::vector<__nv_bfloat16>& host_rope;
    const std::vector<long long>& host_slots;
    const CaseSpec& spec;

    void reset_inputs() {
        state.latent.upload(host_latent);
        state.rope.upload(host_rope);
        state.slots.upload(host_slots);
    }
    void reset_outputs(bool baseline) {
        if (baseline) {
            state.k_scratch.fill(kPoisonA);
            state.v_scratch.fill(kPoisonA);
            state.baseline.fill(kPoisonA);
        } else {
            state.candidate.fill(kPoisonA);
        }
    }
    float timed(bool baseline) {
        cudaEvent_t start;
        cudaEvent_t stop;
        check(cudaEventCreate(&start), "event create");
        check(cudaEventCreate(&stop), "event create");
        auto reset = [&]() { reset_inputs(); reset_outputs(baseline); };
        reset();
        check(cudaEventRecord(start), "cudaEventRecord");
        if (baseline) {
            launch_baseline(state.latent.data, state.rope.data, state.k_scratch.data,
                            state.v_scratch.data, state.baseline.k.data,
                            state.baseline.v.data, state.slots.data,
                            spec.kScale, spec.vScale);
        } else {
            launch_candidate(state.latent.data, state.rope.data,
                             state.candidate.k.data, state.candidate.v.data,
                             state.slots.data, spec.kScale, spec.vScale);
        }
        check(cudaEventRecord(stop), "cudaEventRecord");
        check(cudaEventSynchronize(stop), "event sync");
        float milliseconds = 0.0f;
        check(cudaEventElapsedTime(&milliseconds, start, stop), "cudaEventElapsedTime");
        check(cudaEventDestroy(start), "event destroy");
        check(cudaEventDestroy(stop), "event destroy");
        return milliseconds;
    }
};

static void time_abba(TimingContext& context, double& baseline_ms,
                      double& candidate_ms) {
    // Warmups and measured samples both use baseline, candidate, candidate, baseline.
    auto timed = [&](bool baseline) { return context.timed(baseline); };
    for (int round = 0; round < kWarmupRounds; ++round) {
        timed(true); timed(false); timed(false); timed(true);
    }
    baseline_ms = 0.0;
    candidate_ms = 0.0;
    for (int round = 0; round < kAbbaRounds; ++round) {
        const float baseline_first = timed(true);
        const float candidate_first = timed(false);
        const float candidate_second = timed(false);
        const float baseline_second = timed(true);
        baseline_ms += baseline_first + baseline_second;
        candidate_ms += candidate_first + candidate_second;
    }
    baseline_ms /= 2 * kAbbaRounds;
    candidate_ms /= 2 * kAbbaRounds;
}
// END cache fusion ABBA timing

static int run(int argc, char** argv) {
    const double min_speedup = parse_speedup_threshold(argc, argv);
    int driver = 0;
    int runtime = 0;
    int device = 0;
    cudaDeviceProp properties{};
    check(cudaDriverGetVersion(&driver), "driver version");
    check(cudaRuntimeGetVersion(&runtime), "runtime version");
    check(cudaGetDevice(&device), "device");
    check(cudaGetDeviceProperties(&properties, device), "device properties");

    ProbeState state;
    std::vector<__nv_bfloat16> host_latent(static_cast<size_t>(kTokens) * kKvLora);
    std::vector<__nv_bfloat16> host_rope(static_cast<size_t>(kTokens) * kRope);
    std::vector<long long> host_slots(kTokens);
    const ParityResult parity = run_parity(state, host_latent, host_rope, host_slots);
    run_malformed(state, host_slots);

    make_case(kCases[0], host_latent, host_rope, host_slots);
    TimingContext timing{state, host_latent, host_rope, host_slots, kCases[0]};
    double baseline_ms = 0.0;
    double candidate_ms = 0.0;
    time_abba(timing, baseline_ms, candidate_ms);
    const double speedup = baseline_ms / candidate_ms;
    if (!std::isfinite(speedup) || speedup < min_speedup) {
        die("pre-registered speedup threshold not met");
    }

    char uuid[33];
    for (int i = 0; i < 16; ++i) {
        std::sprintf(uuid + 2 * i, "%02x",
                     static_cast<unsigned char>(properties.uuid.bytes[i]));
    }
    uuid[32] = '\0';
    std::printf("build_id=%s\n", V4_CACHE_FP8_PROBE_BUILD_ID);
    std::printf("device_uuid=%s driver=%d runtime=%d\n", uuid, driver, runtime);
    std::printf("input_hash=%016llx slot_hash=%016llx scale_hash=%016llx\n",
                static_cast<unsigned long long>(parity.input_hash),
                static_cast<unsigned long long>(parity.slot_hash),
                static_cast<unsigned long long>(parity.scale_hash));
    std::printf("parity cases=3 poison_cases=2 k_mismatches=0 v_mismatches=0\n");
    std::printf("guards scratch=clean cache=clean malformed_cases=25 unchanged=25\n");
    std::printf("timing baseline_ms=%.6f candidate_ms=%.6f speedup=%.9f abba_rounds=6\n",
                baseline_ms, candidate_ms, speedup);
    std::printf("threshold min_speedup=%.17g\n", min_speedup);
    std::printf("result=PASS\n");
    return 0;
}
}  // namespace probe

int main(int argc, char** argv) { return probe::run(argc, argv); }
