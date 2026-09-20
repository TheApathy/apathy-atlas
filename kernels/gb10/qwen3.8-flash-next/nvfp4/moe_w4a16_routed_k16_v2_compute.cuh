// SPDX-License-Identifier: AGPL-3.0-only
namespace flash_next_routed_k16_v2 {

__device__ __forceinline__ bool compute_header(const Plan* plan, int* status) {
    if (status == nullptr || status[0] != PLANNED || !plan_header_ok(plan)) {
        fail(status, ERR_PLAN);
        return false;
    }
    return true;
}

__device__ __forceinline__ void reduce_store(
    float& value,
    unsigned long long destination,
    __nv_bfloat16* output
) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_down_sync(0xffffffff, value, offset);
    if ((threadIdx.x & 31) == 0) output[destination] = __float2bfloat16(value);
}

} // namespace flash_next_routed_k16_v2

extern "C" __global__ __launch_bounds__(512, 2)
void moe_w4a16_routed_k16_v2_fixed_gate_up(
    const __nv_bfloat16* input,
    const unsigned long long* packed_ptrs,
    const unsigned long long* scale_ptrs,
    const float* scale2,
    __nv_bfloat16* output,
    const v2::Plan* plan,
    int* status
) {
    constexpr unsigned int TILES = INTER / OUTPUTS_PER_CTA;
    constexpr unsigned int K16 = HIDDEN / 16;
    constexpr unsigned int FIXED_GRID = v2::DESCRIPTORS * TILES;
    if (!v2::compute_header(plan, status)) return;
    if (gridDim.x != FIXED_GRID || gridDim.y != 1 || gridDim.z != 1 ||
        blockDim.x != 512 || blockDim.y != 1 || blockDim.z != 1 ||
        input == nullptr || packed_ptrs == nullptr || scale_ptrs == nullptr ||
        scale2 == nullptr || output == nullptr) {
        v2::fail(status, v2::ERR_GEOMETRY_V2);
        return;
    }
    const unsigned int descriptor_index = blockIdx.x / TILES;
    const unsigned int tile = blockIdx.x % TILES;
    const v2::Descriptor descriptor = plan->descriptors[descriptor_index];
    __shared__ int valid_descriptor;
    __shared__ float lut[16];
    __shared__ unsigned long long staged_packed[8][K16];
    __shared__ unsigned char staged_scales[8][K16];
    if (threadIdx.x == 0) valid_descriptor = v2::descriptor_ok(plan, descriptor);
    if (threadIdx.x < 16) lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();
    if (!valid_descriptor) {
        v2::fail(status, v2::ERR_PLAN);
        return;
    }

    const bool homogeneous = descriptor.mode == v2::HOMOGENEOUS;
    if (homogeneous) {
        const auto* packed = reinterpret_cast<const unsigned char*>(
            packed_ptrs[descriptor.shared_expert]);
        const auto* scales = reinterpret_cast<const unsigned char*>(
            scale_ptrs[descriptor.shared_expert]);
        if (packed == nullptr || scales == nullptr || !isfinite(scale2[descriptor.shared_expert])) {
            v2::fail(status, v2::ERR_POINTER);
            return;
        }
        for (unsigned int i = threadIdx.x; i < 8 * K16; i += blockDim.x) {
            const unsigned int column = i / K16;
            const unsigned int k16 = i % K16;
            const unsigned int n = tile * OUTPUTS_PER_CTA + column;
            staged_packed[column][k16] = *reinterpret_cast<const unsigned long long*>(
                packed + static_cast<unsigned long long>(n) * (HIDDEN / 2) + k16 * 8);
            staged_scales[column][k16] =
                scales[static_cast<unsigned long long>(n) * K16 + k16];
        }
        __syncthreads();
    }

    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int route = warp >> 2;
    const unsigned int local_out = warp & 3;
    const unsigned int flat_slot = descriptor.slots[route];
    const unsigned int expert = plan->route_snapshot[flat_slot];
    const unsigned int n1 = tile * OUTPUTS_PER_CTA + local_out * 2;
    const unsigned int n2 = n1 + 1;
    const __nv_bfloat16* input_row =
        input + static_cast<unsigned long long>(flat_slot / TOP_K) * HIDDEN;
    const auto* packed = reinterpret_cast<const unsigned char*>(packed_ptrs[expert]);
    const auto* scales = reinterpret_cast<const unsigned char*>(scale_ptrs[expert]);
    if (packed == nullptr || scales == nullptr || !isfinite(scale2[expert])) {
        v2::fail(status, v2::ERR_POINTER);
        return;
    }
    const float scale_second = scale2[expert];
    float acc1 = 0.0f, acc2 = 0.0f;
    for (unsigned int k16 = lane; k16 < K16; k16 += 32) {
        const uint4 lo = reinterpret_cast<const uint4*>(input_row)[k16 * 2];
        const uint4 hi = reinterpret_cast<const uint4*>(input_row)[k16 * 2 + 1];
        const unsigned int ar[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
        const unsigned long long p1 = homogeneous ? staged_packed[local_out * 2][k16] :
            *reinterpret_cast<const unsigned long long*>(packed +
                static_cast<unsigned long long>(n1) * (HIDDEN / 2) + k16 * 8);
        const unsigned long long p2 = homogeneous ? staged_packed[local_out * 2 + 1][k16] :
            *reinterpret_cast<const unsigned long long*>(packed +
                static_cast<unsigned long long>(n2) * (HIDDEN / 2) + k16 * 8);
        __nv_fp8_e4m3 f1, f2;
        *reinterpret_cast<unsigned char*>(&f1) = homogeneous
            ? staged_scales[local_out * 2][k16]
            : scales[static_cast<unsigned long long>(n1) * K16 + k16];
        *reinterpret_cast<unsigned char*>(&f2) = homogeneous
            ? staged_scales[local_out * 2 + 1][k16]
            : scales[static_cast<unsigned long long>(n2) * K16 + k16];
        const float sc1 = static_cast<float>(f1) * scale_second;
        const float sc2 = static_cast<float>(f2) * scale_second;
        #pragma unroll
        for (int b = 0; b < 8; ++b) {
            const unsigned char v1 = static_cast<unsigned char>(p1 >> (b * 8));
            const unsigned char v2b = static_cast<unsigned char>(p2 >> (b * 8));
            __nv_bfloat16 al, ah;
            *reinterpret_cast<unsigned short*>(&al) = static_cast<unsigned short>(ar[b]);
            *reinterpret_cast<unsigned short*>(&ah) = static_cast<unsigned short>(ar[b] >> 16);
            const float afl = __bfloat162float(al), afh = __bfloat162float(ah);
            acc1 += afl * (lut[v1 & 15] * sc1) + afh * (lut[v1 >> 4] * sc1);
            acc2 += afl * (lut[v2b & 15] * sc2) + afh * (lut[v2b >> 4] * sc2);
        }
    }
    const unsigned long long destination =
        static_cast<unsigned long long>(flat_slot) * INTER + tile * OUTPUTS_PER_CTA;
    v2::reduce_store(acc1, destination + local_out * 2, output);
    v2::reduce_store(acc2, destination + local_out * 2 + 1, output);
}

extern "C" __global__ __launch_bounds__(512, 2)
void moe_w4a16_routed_k16_v2_fixed_down(
    const float* activation,
    const unsigned long long* packed_ptrs,
    const unsigned long long* scale_ptrs,
    const float* scale2,
    __nv_bfloat16* output,
    const v2::Plan* plan,
    int* status
) {
    constexpr unsigned int TILES = HIDDEN / OUTPUTS_PER_CTA;
    constexpr unsigned int K16 = INTER / 16;
    constexpr unsigned int FIXED_GRID = v2::DESCRIPTORS * TILES;
    if (!v2::compute_header(plan, status)) return;
    if (gridDim.x != FIXED_GRID || gridDim.y != 1 || gridDim.z != 1 ||
        blockDim.x != 512 || blockDim.y != 1 || blockDim.z != 1 ||
        activation == nullptr || packed_ptrs == nullptr || scale_ptrs == nullptr ||
        scale2 == nullptr || output == nullptr) {
        v2::fail(status, v2::ERR_GEOMETRY_V2);
        return;
    }
    const unsigned int descriptor_index = blockIdx.x / TILES;
    const unsigned int tile = blockIdx.x % TILES;
    const v2::Descriptor descriptor = plan->descriptors[descriptor_index];
    __shared__ int valid_descriptor;
    __shared__ float lut[16];
    __shared__ unsigned long long staged_packed[8][K16];
    __shared__ unsigned char staged_scales[8][K16];
    __shared__ float staged_activation[v2::DESCRIPTOR_WIDTH][INTER];
    if (threadIdx.x == 0) valid_descriptor = v2::descriptor_ok(plan, descriptor);
    if (threadIdx.x < 16) lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();
    if (!valid_descriptor) {
        v2::fail(status, v2::ERR_PLAN);
        return;
    }
    for (unsigned int i = threadIdx.x; i < v2::DESCRIPTOR_WIDTH * INTER; i += blockDim.x) {
        const unsigned int route = i / INTER;
        const unsigned int k = i % INTER;
        staged_activation[route][k] = activation[
            static_cast<unsigned long long>(descriptor.slots[route]) * INTER + k];
    }

    const bool homogeneous = descriptor.mode == v2::HOMOGENEOUS;
    if (homogeneous) {
        const auto* packed = reinterpret_cast<const unsigned char*>(
            packed_ptrs[descriptor.shared_expert]);
        const auto* scales = reinterpret_cast<const unsigned char*>(
            scale_ptrs[descriptor.shared_expert]);
        if (packed == nullptr || scales == nullptr || !isfinite(scale2[descriptor.shared_expert])) {
            v2::fail(status, v2::ERR_POINTER);
            return;
        }
        for (unsigned int i = threadIdx.x; i < 8 * K16; i += blockDim.x) {
            const unsigned int column = i / K16;
            const unsigned int k16 = i % K16;
            const unsigned int n = tile * OUTPUTS_PER_CTA + column;
            staged_packed[column][k16] = *reinterpret_cast<const unsigned long long*>(
                packed + static_cast<unsigned long long>(n) * (INTER / 2) + k16 * 8);
            staged_scales[column][k16] =
                scales[static_cast<unsigned long long>(n) * K16 + k16];
        }
    }
    __syncthreads();

    const unsigned int warp = threadIdx.x >> 5;
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int route = warp >> 2;
    const unsigned int local_out = warp & 3;
    const unsigned int flat_slot = descriptor.slots[route];
    const unsigned int expert = plan->route_snapshot[flat_slot];
    const unsigned int n1 = tile * OUTPUTS_PER_CTA + local_out * 2;
    const unsigned int n2 = n1 + 1;
    const auto* packed = reinterpret_cast<const unsigned char*>(packed_ptrs[expert]);
    const auto* scales = reinterpret_cast<const unsigned char*>(scale_ptrs[expert]);
    if (packed == nullptr || scales == nullptr || !isfinite(scale2[expert])) {
        v2::fail(status, v2::ERR_POINTER);
        return;
    }
    const float scale_second = scale2[expert];
    float acc1 = 0.0f, acc2 = 0.0f;
    for (unsigned int k16 = lane; k16 < K16; k16 += 32) {
        const unsigned int base = k16 * 16;
        const unsigned long long p1 = homogeneous ? staged_packed[local_out * 2][k16] :
            *reinterpret_cast<const unsigned long long*>(packed +
                static_cast<unsigned long long>(n1) * (INTER / 2) + k16 * 8);
        const unsigned long long p2 = homogeneous ? staged_packed[local_out * 2 + 1][k16] :
            *reinterpret_cast<const unsigned long long*>(packed +
                static_cast<unsigned long long>(n2) * (INTER / 2) + k16 * 8);
        __nv_fp8_e4m3 f1, f2;
        *reinterpret_cast<unsigned char*>(&f1) = homogeneous
            ? staged_scales[local_out * 2][k16]
            : scales[static_cast<unsigned long long>(n1) * K16 + k16];
        *reinterpret_cast<unsigned char*>(&f2) = homogeneous
            ? staged_scales[local_out * 2 + 1][k16]
            : scales[static_cast<unsigned long long>(n2) * K16 + k16];
        const float sc1 = static_cast<float>(f1) * scale_second;
        const float sc2 = static_cast<float>(f2) * scale_second;
        #pragma unroll
        for (int b = 0; b < 8; ++b) {
            const float al = staged_activation[route][base + b * 2];
            const float ah = staged_activation[route][base + b * 2 + 1];
            const unsigned char v1 = static_cast<unsigned char>(p1 >> (b * 8));
            const unsigned char v2b = static_cast<unsigned char>(p2 >> (b * 8));
            acc1 += al * (lut[v1 & 15] * sc1) + ah * (lut[v1 >> 4] * sc1);
            acc2 += al * (lut[v2b & 15] * sc2) + ah * (lut[v2b >> 4] * sc2);
        }
    }
    const unsigned long long destination =
        static_cast<unsigned long long>(flat_slot) * HIDDEN + tile * OUTPUTS_PER_CTA;
    v2::reduce_store(acc1, destination + local_out * 2, output);
    v2::reduce_store(acc2, destination + local_out * 2 + 1, output);
}
