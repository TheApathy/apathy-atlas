// SPDX-License-Identifier: AGPL-3.0-only

// GPU-ready raw gate.  CPU/source checks may compile it only after the active
// foreign build/GPU owner releases; execution always requires a new exclusive
// root reservation and never enables this candidate in production.
#define main flash_next_exact_k16_v2_unused_main
#include "moe_w4a16_exact_k16_microgate.cu"
#undef main
#define FLASH_NEXT_EXACT_K16_PARENT_INCLUDED 1
#include "moe_w4a16_routed_k16_v2.cu"

#ifndef FLASH_NEXT_V2_HEADER_SHA256
#define FLASH_NEXT_V2_HEADER_SHA256 "UNBOUND"
#endif
#ifndef FLASH_NEXT_V2_CUDA_SHA256
#define FLASH_NEXT_V2_CUDA_SHA256 "UNBOUND"
#endif
#ifndef FLASH_NEXT_V2_MICROGATE_SHA256
#define FLASH_NEXT_V2_MICROGATE_SHA256 "UNBOUND"
#endif
#ifndef FLASH_NEXT_V2_EXACT_PARENT_SHA256
#define FLASH_NEXT_V2_EXACT_PARENT_SHA256 "UNBOUND"
#endif
#ifndef FLASH_NEXT_V2_EXACT_PARENT_GATE_SHA256
#define FLASH_NEXT_V2_EXACT_PARENT_GATE_SHA256 "UNBOUND"
#endif
#ifndef FLASH_NEXT_V2_BATCH_PARENT_SHA256
#define FLASH_NEXT_V2_BATCH_PARENT_SHA256 "UNBOUND"
#endif
#ifndef FLASH_NEXT_V2_SERIAL_PARENT_SHA256
#define FLASH_NEXT_V2_SERIAL_PARENT_SHA256 "UNBOUND"
#endif
#ifndef FLASH_NEXT_V2_BLEND_PARENT_SHA256
#define FLASH_NEXT_V2_BLEND_PARENT_SHA256 "UNBOUND"
#endif

using namespace flash_next_exact_k16;
namespace v2 = flash_next_routed_k16_v2;

// Routed-only exact-parent wrappers are timing controls.  Correctness below is
// additionally compared with the linked shipping batch3 and serial-K1 kernels.
extern "C" __global__ __launch_bounds__(128, 2)
void moe_w4a16_routed_k16_v2_parent_gate(
    const __nv_bfloat16* input,
    const unsigned long long* gate_packed,
    const unsigned long long* gate_scales,
    const float* gate_s2,
    __nv_bfloat16* gate_out,
    const unsigned long long* up_packed,
    const unsigned long long* up_scales,
    const float* up_s2,
    __nv_bfloat16* up_out,
    const unsigned int* routes
) {
    __shared__ float lut[16];
    if (threadIdx.x < 16) lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();
    gate_up_routed(blockIdx.x, blockIdx.y, input,
        blockIdx.y == 0 ? gate_packed : up_packed,
        blockIdx.y == 0 ? gate_scales : up_scales,
        blockIdx.y == 0 ? gate_s2 : up_s2,
        blockIdx.y == 0 ? gate_out : up_out, routes, lut);
}

extern "C" __global__ __launch_bounds__(128, 2)
void moe_w4a16_routed_k16_v2_parent_down(
    const __nv_bfloat16* gate,
    const __nv_bfloat16* up,
    const unsigned long long* packed,
    const unsigned long long* scales,
    const float* scale2,
    __nv_bfloat16* output,
    const unsigned int* routes
) {
    __shared__ float lut[16];
    extern __shared__ float activation[];
    if (threadIdx.x < 16) lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();
    down_routed(blockIdx.x, gate, up, packed, scales, scale2, output, routes, lut, activation);
}

#ifndef FLASH_NEXT_V2_PLAN_SHA256
#define FLASH_NEXT_V2_PLAN_SHA256 "UNBOUND"
#endif
#ifndef FLASH_NEXT_V2_COMPUTE_SHA256
#define FLASH_NEXT_V2_COMPUTE_SHA256 "UNBOUND"
#endif
#ifndef FLASH_NEXT_V2_MICROGATE_FIXTURE_SHA256
#define FLASH_NEXT_V2_MICROGATE_FIXTURE_SHA256 "UNBOUND"
#endif
#ifndef FLASH_NEXT_V2_MICROGATE_VALIDATION_SHA256
#define FLASH_NEXT_V2_MICROGATE_VALIDATION_SHA256 "UNBOUND"
#endif
#ifndef FLASH_NEXT_V2_MICROGATE_TIMING_SHA256
#define FLASH_NEXT_V2_MICROGATE_TIMING_SHA256 "UNBOUND"
#endif
#include "moe_w4a16_routed_k16_v2_microgate_fixture.cuh"
#include "moe_w4a16_routed_k16_v2_microgate_validation.cuh"
#include "moe_w4a16_routed_k16_v2_microgate_timing.cuh"

int main(int argc, char** argv) {
    const bool run_timing = argc == 2 && std::string(argv[1]) == "--timing";
    if (argc > 2 || (argc == 2 && !run_timing)) return 2;
    const std::string binary_sha256 = executable_sha256();
    const bool provenance_bound = !binary_sha256.empty() &&
        digest_bound(FLASH_NEXT_V2_HEADER_SHA256) &&
        digest_bound(FLASH_NEXT_V2_CUDA_SHA256) &&
        digest_bound(FLASH_NEXT_V2_MICROGATE_SHA256) &&
        digest_bound(FLASH_NEXT_V2_PLAN_SHA256) &&
        digest_bound(FLASH_NEXT_V2_COMPUTE_SHA256) &&
        digest_bound(FLASH_NEXT_V2_MICROGATE_FIXTURE_SHA256) &&
        digest_bound(FLASH_NEXT_V2_MICROGATE_VALIDATION_SHA256) &&
        digest_bound(FLASH_NEXT_V2_MICROGATE_TIMING_SHA256) &&
        digest_bound(FLASH_NEXT_V2_EXACT_PARENT_SHA256) &&
        digest_bound(FLASH_NEXT_V2_EXACT_PARENT_GATE_SHA256) &&
        digest_bound(FLASH_NEXT_V2_BATCH_PARENT_SHA256) &&
        digest_bound(FLASH_NEXT_V2_SERIAL_PARENT_SHA256) &&
        digest_bound(FLASH_NEXT_V2_BLEND_PARENT_SHA256);
    if (!provenance_bound) {
        std::fprintf(stderr, "unbound v2 source/binary provenance; refusing raw qualification\n");
        return 2;
    }
    int device = 0;
    CUDA_OK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CUDA_OK(cudaGetDeviceProperties(&properties, device));
    if (properties.major != 12 || properties.minor != 1 ||
        properties.multiProcessorCount != 48) return 2;

    Fixture fixture;
    v2_gate::State state;
    bool ok = true;
    ok &= v2_gate::fixture_case(fixture, state, v2_gate::route_case(160), "no_collision");
    ok &= v2_gate::fixture_case(fixture, state, v2_gate::mixed_tails(), "mixed_count1_2_3");
    ok &= v2_gate::fixture_case(fixture, state, v2_gate::route_case(40), "count4_only");
    ok &= v2_gate::fixture_case(fixture, state, v2_gate::route_case(0), "all_one_expert");
    ok &= v2_gate::hostile(fixture, state);
    if (run_timing && ok) ok &= v2_gate::timing(fixture, state);
    std::printf("FLASH_NEXT_ROUTED_K16_V2 version=3 rows=16 hidden=2560 intermediate=640 "
                "experts=512 top_k=10 descriptors=40 gate_grid=3200 down_grid=12800 "
                "activation_workspace=409600 shipping_batch3=exact serial_k1=exact "
                "host_launch_counts=false selected_packed_bytes=%llu selected_scale_bytes=%llu "
                "config_sha256=%s index_sha256=%s routed_first_shard_sha256=%s "
                "routed_final_shard_sha256=%s v2_header_sha256=%s v2_cuda_sha256=%s "
                "v2_microgate_sha256=%s v2_plan_sha256=%s v2_compute_sha256=%s "
                "v2_microgate_fixture_sha256=%s v2_microgate_validation_sha256=%s "
                "v2_microgate_timing_sha256=%s exact_parent_sha256=%s "
                "exact_parent_gate_sha256=%s "
                "batch_parent_sha256=%s serial_parent_sha256=%s blend_parent_sha256=%s "
                "binary_sha256=%s timing=%s pass=%s\n",
                v2::SELECTED_WEIGHT_BYTES, v2::SELECTED_SCALE_BYTES,
                CONFIG_SHA256, INDEX_SHA256, ROUTED_FIRST_SHARD_SHA256,
                ROUTED_FINAL_SHARD_SHA256, FLASH_NEXT_V2_HEADER_SHA256,
                FLASH_NEXT_V2_CUDA_SHA256, FLASH_NEXT_V2_MICROGATE_SHA256,
                FLASH_NEXT_V2_PLAN_SHA256, FLASH_NEXT_V2_COMPUTE_SHA256,
                FLASH_NEXT_V2_MICROGATE_FIXTURE_SHA256,
                FLASH_NEXT_V2_MICROGATE_VALIDATION_SHA256,
                FLASH_NEXT_V2_MICROGATE_TIMING_SHA256,
                FLASH_NEXT_V2_EXACT_PARENT_SHA256, FLASH_NEXT_V2_EXACT_PARENT_GATE_SHA256,
                FLASH_NEXT_V2_BATCH_PARENT_SHA256, FLASH_NEXT_V2_SERIAL_PARENT_SHA256,
                FLASH_NEXT_V2_BLEND_PARENT_SHA256, binary_sha256.c_str(),
                run_timing ? "measured_conditional" : "unrun", ok ? "true" : "false");
    return ok ? 0 : 1;
}
