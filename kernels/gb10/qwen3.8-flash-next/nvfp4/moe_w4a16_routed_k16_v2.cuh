// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#ifndef FLASH_NEXT_EXACT_K16_PARENT_INCLUDED
#include "moe_w4a16_exact_k16.cu"
#define FLASH_NEXT_EXACT_K16_PARENT_INCLUDED 1
#endif

// Raw-only, default-unrouted ABI.  The exact parent preflight must validate all
// compute buffers on the same stream before this planner runs.  The caller then
// owns one private Plan, status word, and FP32 activation range until every
// fixed-grid compute launch completes; none may be mutated in flight.
namespace flash_next_routed_k16_v2 {

using namespace flash_next_exact_k16;
constexpr unsigned long long MAGIC = 0x51464e52544b5632ULL; // QFNRTKV2
constexpr unsigned int VERSION = 3;
constexpr unsigned int ROUTES = ROWS * TOP_K;
constexpr unsigned int DESCRIPTOR_WIDTH = 4;
constexpr unsigned int DESCRIPTORS = ROUTES / DESCRIPTOR_WIDTH;
constexpr unsigned short HETEROGENEOUS_EXPERT = 0xffffU;
constexpr unsigned long long INPUT_ELEMS =
    static_cast<unsigned long long>(ROWS) * HIDDEN;
constexpr unsigned long long ROUTED_INTER_ELEMS =
    static_cast<unsigned long long>(ROUTES) * INTER;
constexpr unsigned long long ROUTED_HIDDEN_ELEMS =
    static_cast<unsigned long long>(ROUTES) * HIDDEN;
constexpr unsigned long long SELECTED_WEIGHT_BYTES =
    static_cast<unsigned long long>(INTER) * HIDDEN / 2;
constexpr unsigned long long SELECTED_SCALE_BYTES =
    static_cast<unsigned long long>(INTER) * HIDDEN / GROUP;
constexpr unsigned long long ACTIVATION_ELEMS = ROUTED_INTER_ELEMS;
constexpr unsigned long long ACTIVATION_BYTES = ACTIVATION_ELEMS * sizeof(float);

enum DescriptorMode : unsigned char { HETEROGENEOUS = 0, HOMOGENEOUS = 1 };

enum Status : int {
    BUILDING = -2,
    PLANNED = 0,
    ERR_ABI = 30,
    ERR_POINTER = 31,
    ERR_ALIGNMENT = 32,
    ERR_ALIAS = 33,
    ERR_OVERFLOW = 34,
    ERR_ROUTE = 35,
    ERR_PLAN = 36,
    ERR_GEOMETRY_V2 = 37,
    ERR_PARENT = 38,
};

struct alignas(16) Contract {
    unsigned long long magic;
    unsigned int version, rows, hidden, intermediate, experts, top_k;
    unsigned int route_count, descriptor_count;
    unsigned long long plan_bytes, activation_elems, input_elems;
    unsigned long long routed_inter_elems, routed_hidden_elems;
    unsigned long long selected_weight_bytes, selected_scale_bytes;
};

struct alignas(16) Descriptor {
    unsigned short shared_expert;
    unsigned char slots[DESCRIPTOR_WIDTH];
    unsigned char mode;
    unsigned char reserved[9];
};

struct alignas(16) Plan {
    unsigned long long ready_magic;
    unsigned int version, descriptor_count, homogeneous_count, heterogeneous_count;
    unsigned int route_count, reserved0, reserved1, reserved2;
    Descriptor descriptors[DESCRIPTORS];
    unsigned int route_snapshot[ROUTES];
};

static_assert(sizeof(Contract) == 96, "v2 contract ABI drift");
static_assert(sizeof(Descriptor) == 16, "v2 descriptor ABI drift");
static_assert(sizeof(Plan) == 1328, "v2 fixed40 plan ABI drift");
static_assert(DESCRIPTORS == 40, "v2 fixed descriptor count drift");
static_assert(INPUT_ELEMS == 40960, "v2 input extent drift");
static_assert(ROUTED_INTER_ELEMS == 102400, "v2 routed intermediate drift");
static_assert(ROUTED_HIDDEN_ELEMS == 409600, "v2 routed hidden drift");
static_assert(SELECTED_WEIGHT_BYTES == 819200, "v2 selected weight range drift");
static_assert(SELECTED_SCALE_BYTES == 102400, "v2 selected scale range drift");
static_assert(ACTIVATION_BYTES == 409600, "v2 activation workspace drift");

struct Range { const void* ptr; unsigned long long bytes; };

__device__ __forceinline__ bool end_ok(const void* ptr, unsigned long long bytes) {
    const uintptr_t begin = reinterpret_cast<uintptr_t>(ptr);
    return ptr != nullptr && bytes <= static_cast<unsigned long long>(UINTPTR_MAX - begin);
}

__device__ __forceinline__ bool aligned_to(const void* ptr, uintptr_t n) {
    return ptr != nullptr && (reinterpret_cast<uintptr_t>(ptr) & (n - 1)) == 0;
}

__device__ __forceinline__ bool overlap(const Range& a, const Range& b) {
    const uintptr_t ab = reinterpret_cast<uintptr_t>(a.ptr);
    const uintptr_t bb = reinterpret_cast<uintptr_t>(b.ptr);
    return ab < bb + b.bytes && bb < ab + a.bytes;
}

__device__ __forceinline__ bool overlap_by_distance(const Range& a, const Range& b) {
    const uintptr_t ab = reinterpret_cast<uintptr_t>(a.ptr);
    const uintptr_t bb = reinterpret_cast<uintptr_t>(b.ptr);
    return ab <= bb ? static_cast<unsigned long long>(bb - ab) < a.bytes
                    : static_cast<unsigned long long>(ab - bb) < b.bytes;
}

__device__ __forceinline__ bool contract_ok(const Contract* c) {
    return c != nullptr && c->magic == MAGIC && c->version == VERSION &&
        c->rows == ROWS && c->hidden == HIDDEN && c->intermediate == INTER &&
        c->experts == EXPERTS && c->top_k == TOP_K && c->route_count == ROUTES &&
        c->descriptor_count == DESCRIPTORS && c->plan_bytes == sizeof(Plan) &&
        c->activation_elems == ACTIVATION_ELEMS && c->input_elems == INPUT_ELEMS &&
        c->routed_inter_elems == ROUTED_INTER_ELEMS &&
        c->routed_hidden_elems == ROUTED_HIDDEN_ELEMS &&
        c->selected_weight_bytes == SELECTED_WEIGHT_BYTES &&
        c->selected_scale_bytes == SELECTED_SCALE_BYTES;
}

__device__ __forceinline__ bool plan_header_ok(const Plan* plan) {
    return plan != nullptr && plan->ready_magic == MAGIC && plan->version == VERSION &&
        plan->descriptor_count == DESCRIPTORS && plan->route_count == ROUTES &&
        plan->homogeneous_count + plan->heterogeneous_count == DESCRIPTORS &&
        plan->reserved0 == 0 && plan->reserved1 == 0 && plan->reserved2 == 0;
}

__device__ __forceinline__ bool descriptor_ok(const Plan* plan, const Descriptor& d) {
    if (d.mode > HOMOGENEOUS) return false;
    for (unsigned int i = 0; i < sizeof(d.reserved); ++i)
        if (d.reserved[i] != 0) return false;
    unsigned int first = EXPERTS;
    bool all_same = true;
    for (unsigned int i = 0; i < DESCRIPTOR_WIDTH; ++i) {
        if (d.slots[i] >= ROUTES) return false;
        const unsigned int expert = plan->route_snapshot[d.slots[i]];
        if (expert >= EXPERTS) return false;
        if (i == 0) first = expert; else all_same &= expert == first;
        for (unsigned int j = i + 1; j < DESCRIPTOR_WIDTH; ++j)
            if (d.slots[i] == d.slots[j]) return false;
    }
    return d.mode == HOMOGENEOUS
        ? all_same && d.shared_expert == first
        : !all_same && d.shared_expert == HETEROGENEOUS_EXPERT;
}

__device__ __forceinline__ void fail(int* status, int code) {
    if (status != nullptr && threadIdx.x == 0) atomicCAS(status, PLANNED, code);
}

} // namespace flash_next_routed_k16_v2

extern "C" __global__ void moe_w4a16_routed_k16_v2_plan(
    const unsigned int* routes,
    const flash_next_routed_k16_v2::Contract* contract,
    flash_next_routed_k16_v2::Plan* plan,
    float* staged_activation,
    const int* parent_status,
    int* status
);

extern "C" __global__ __launch_bounds__(128, 4)
void moe_w4a16_routed_k16_v2_silu_stage(
    const __nv_bfloat16* gate,
    const __nv_bfloat16* up,
    float* activation,
    const flash_next_routed_k16_v2::Plan* plan,
    int* status
) {
    using namespace flash_next_routed_k16_v2;
    if (status == nullptr || status[0] != PLANNED || !plan_header_ok(plan) ||
        gridDim.x != ROUTES || gridDim.y != 1 || gridDim.z != 1 ||
        blockDim.x != 128 || blockDim.y != 1 || blockDim.z != 1 ||
        blockIdx.x >= ROUTES || gate == nullptr || up == nullptr || activation == nullptr) {
        fail(status, ERR_GEOMETRY_V2); return;
    }
    const unsigned long long base = static_cast<unsigned long long>(blockIdx.x) * INTER;
    for (unsigned int i = threadIdx.x; i < INTER; i += blockDim.x) {
        const float gf = __bfloat162float(gate[base + i]);
        const float uf = __bfloat162float(up[base + i]);
        activation[base + i] = (gf / (1.0f + __expf(-gf))) * uf;
    }
}
