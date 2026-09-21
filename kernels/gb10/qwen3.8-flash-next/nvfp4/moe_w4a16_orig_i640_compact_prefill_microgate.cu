// SPDX-License-Identifier: AGPL-3.0-only
// Default-off raw gate; never referenced by Atlas production manifests.
#include "../../qwen3-next-80b-a3b/nvfp4/moe_w4a16_grouped_gemm.cu"
#include "moe_w4a16_orig_i640_compact_prefill.cu"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iterator>
#include <limits.h>
#include <map>
#include <sstream>
#include <set>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#define CUDA_OK(expr) do { cudaError_t e_ = (expr); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA failure %s:%d: %s\n", __FILE__, __LINE__, \
                 cudaGetErrorString(e_)); std::exit(2); } } while (0)

#ifndef OI640_HEADER_SHA256
#define OI640_HEADER_SHA256 "UNBOUND"
#endif
#ifndef OI640_CUDA_SHA256
#define OI640_CUDA_SHA256 "UNBOUND"
#endif
#ifndef OI640_PLAN_SHA256
#define OI640_PLAN_SHA256 "UNBOUND"
#endif
#ifndef OI640_GEMM_SHA256
#define OI640_GEMM_SHA256 "UNBOUND"
#endif
#ifndef OI640_MICROGATE_SHA256
#define OI640_MICROGATE_SHA256 "UNBOUND"
#endif
#ifndef OI640_IO_SHA256
#define OI640_IO_SHA256 "UNBOUND"
#endif
#ifndef OI640_BUFFERS_SHA256
#define OI640_BUFFERS_SHA256 "UNBOUND"
#endif
#ifndef OI640_CASES_SHA256
#define OI640_CASES_SHA256 "UNBOUND"
#endif
#ifndef OI640_TIMING_SHA256
#define OI640_TIMING_SHA256 "UNBOUND"
#endif
#ifndef OI640_PARENT_SHA256
#define OI640_PARENT_SHA256 "UNBOUND"
#endif
#ifndef OI640_CONFIG_SHA256
#define OI640_CONFIG_SHA256 "UNBOUND"
#endif
#ifndef OI640_INDEX_SHA256
#define OI640_INDEX_SHA256 "UNBOUND"
#endif
#ifndef OI640_FIRST_SHARD_SHA256
#define OI640_FIRST_SHARD_SHA256 "UNBOUND"
#endif
#ifndef OI640_LAST_SHARD_SHA256
#define OI640_LAST_SHARD_SHA256 "UNBOUND"
#endif
#ifndef OI640_SOURCE_BUNDLE_SHA256
#define OI640_SOURCE_BUNDLE_SHA256 "UNBOUND"
#endif
#ifndef OI640_CAPTURE_PRELOAD_SHA256
#define OI640_CAPTURE_PRELOAD_SHA256 "UNBOUND"
#endif
#ifndef OI640_MANIFEST_SHA256
#define OI640_MANIFEST_SHA256 "UNBOUND"
#endif
#ifndef OI640_MODEL_MANIFEST_SHA256
#define OI640_MODEL_MANIFEST_SHA256 "UNBOUND"
#endif

#include "moe_w4a16_orig_i640_compact_prefill_microgate_io.cuh"
#include "moe_w4a16_orig_i640_compact_prefill_microgate_manifest.cuh"
#include "moe_w4a16_orig_i640_compact_prefill_microgate_provenance.cuh"
#include "moe_w4a16_orig_i640_compact_prefill_microgate_buffers.cuh"
#include "moe_w4a16_orig_i640_compact_prefill_microgate_cases.cuh"
#include "moe_w4a16_orig_i640_compact_prefill_microgate_timing.cuh"

namespace gate = oi640_gate;

bool source_provenance_bound() {
    const char* digests[] = {OI640_HEADER_SHA256, OI640_CUDA_SHA256, OI640_PLAN_SHA256,
        OI640_GEMM_SHA256, OI640_MICROGATE_SHA256, OI640_IO_SHA256, OI640_BUFFERS_SHA256,
        OI640_CASES_SHA256, OI640_TIMING_SHA256, OI640_PARENT_SHA256,
        OI640_CONFIG_SHA256, OI640_INDEX_SHA256, OI640_FIRST_SHARD_SHA256,
        OI640_LAST_SHARD_SHA256, OI640_SOURCE_BUNDLE_SHA256, OI640_CAPTURE_PRELOAD_SHA256,
        OI640_MANIFEST_SHA256, OI640_MODEL_MANIFEST_SHA256};
    return std::all_of(std::begin(digests), std::end(digests),
        [](const char* value) { return gate::hex_digest(value); });
}

int main(int argc, char** argv) {
    const bool timing = argc == 2 && std::string(argv[1]) == "--timing";
    if ((argc != 1 && !timing) || !source_provenance_bound() ||
        !gate::valid_nonce(gate::required_env("ATLAS_OI640_RUN_NONCE"))) {
        std::fprintf(stderr, "unbound source/model/capture provenance; refusing qualification\n");
        return 2;
    }
    gate::Provenance proof;
    if (!gate::load_provenance(proof)) return 2;
    std::vector<int> offsets2013, offsets8192;
    std::string offset2013_sha, offset8192_sha;
    if (!gate::read_offsets("ATLAS_OI640_M2013_OFFSETS", "ATLAS_OI640_M2013_OFFSETS_SHA256",
                            oi640::M_SMALL, offsets2013, offset2013_sha) ||
        !gate::read_offsets("ATLAS_OI640_M8192_OFFSETS", "ATLAS_OI640_M8192_OFFSETS_SHA256",
                            oi640::M_LARGE, offsets8192, offset8192_sha)) return 2;
    const std::string binary_sha = proof.executable.digest;
    int ordinal = -1; CUDA_OK(cudaGetDevice(&ordinal));
    cudaDeviceProp prop{}; CUDA_OK(cudaGetDeviceProperties(&prop, ordinal));
    if (prop.major != 12 || prop.minor != 1 || prop.multiProcessorCount != 48) return 2;
    cudaStream_t stream{}; CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    gate::Weights weights;
    bool ok = true;
    gate::TimingResult t2013{}, t8192{};
    uint32_t work2013 = 0, work8192 = 0, active2013 = 0, active8192 = 0;
    uint64_t census2013 = 0, census8192 = 0, output2013 = 0, output8192 = 0;
    {
        gate::ShapeCase shape(oi640::M_SMALL, offsets2013);
        ok &= shape.correctness(weights, stream) && shape.hostile(stream);
        if (timing && ok) t2013 = gate::time_real_skew(shape, weights, stream);
        work2013 = shape.work_count; active2013 = shape.active_experts;
        census2013 = shape.census_hash; output2013 = shape.output_hash;
    }
    {
        gate::ShapeCase shape(oi640::M_LARGE, offsets8192);
        ok &= shape.correctness(weights, stream);
        if (timing && ok) t8192 = gate::time_real_skew(shape, weights, stream);
        work8192 = shape.work_count; active8192 = shape.active_experts;
        census8192 = shape.census_hash; output8192 = shape.output_hash;
    }
    ok &= !timing || (t2013.pass && t8192.pass);
    ok &= weights.safe() && gate::provenance_stable(proof);
    CUDA_OK(cudaStreamDestroy(stream));
    std::ostringstream receipt;
    receipt << "{\"schema\":2,\"candidate\":\"orig_i640_compact_prefill\","
        << "\"pass\":" << (ok ? "true" : "false") << ",\"timing\":"
        << (timing ? "true" : "false") << ",\"attestation_only\":"
        << (!timing ? "true" : "false") << ",\"qualified\":"
        << (timing && ok ? "true" : "false") << ",\"performance_claim_allowed\":"
        << (timing && ok ? "true" : "false") << ",\"binary_sha256\":\"" << binary_sha
        << "\",\"run_nonce\":\"" << gate::required_env("ATLAS_OI640_RUN_NONCE")
        << "\",\"device\":{\"ordinal\":" << ordinal << ",\"major\":" << prop.major
        << ",\"minor\":" << prop.minor << ",\"sms\":" << prop.multiProcessorCount
        << "},\"fixed_grid\":4096,\"workspace_bytes\":" << sizeof(oi640::Workspace)
        << ",\"m2013\":{\"offset_sha256\":\"" << offset2013_sha << "\",\"work\":"
        << work2013 << ",\"active\":" << active2013 << ",\"census\":" << census2013
        << ",\"output\":" << output2013 << ",\"samples\":" << t2013.samples
        << ",\"parent_median_ms\":" << t2013.parent_median
        << ",\"candidate_median_ms\":" << t2013.candidate_median
        << ",\"parent_p90_ms\":" << t2013.parent_p90 << ",\"candidate_p90_ms\":"
        << t2013.candidate_p90 << ",\"paired_median_ms\":" << t2013.paired_median
        << "},\"m8192\":{\"offset_sha256\":\"" << offset8192_sha << "\",\"work\":"
        << work8192 << ",\"active\":" << active8192 << ",\"census\":" << census8192
        << ",\"output\":" << output8192 << ",\"samples\":" << t8192.samples
        << ",\"parent_median_ms\":" << t8192.parent_median
        << ",\"candidate_median_ms\":" << t8192.candidate_median
        << ",\"parent_p90_ms\":" << t8192.parent_p90 << ",\"candidate_p90_ms\":"
        << t8192.candidate_p90 << ",\"paired_median_ms\":" << t8192.paired_median
        << "},\"config_sha256\":\"" << OI640_CONFIG_SHA256 << "\",\"index_sha256\":\""
        << OI640_INDEX_SHA256 << "\",\"first_shard_sha256\":\""
        << OI640_FIRST_SHARD_SHA256 << "\",\"last_shard_sha256\":\""
        << OI640_LAST_SHARD_SHA256 << "\",\"producer_receipt_sha256\":\""
        << proof.capture.digest
        << "\",\"parent_sha256\":\"" << OI640_PARENT_SHA256
        << "\",\"source_bundle_sha256\":\"" << OI640_SOURCE_BUNDLE_SHA256
        << "\",\"build_manifest_sha256\":\"" << proof.build.digest
        << "\",\"model_manifest_sha256\":\"" << proof.model.digest << "\"}\n";
    if (!ok || !gate::write_receipt(receipt.str())) return 1;
    std::printf("OI640_COMPACT_PREFILL pass=true timing=%s qualified=%s "
                "performance_claim_allowed=%s receipt=exclusive_mode0444\n",
                timing ? "measured" : "unrun", timing ? "true" : "false",
                timing ? "true" : "false");
    return 0;
}
