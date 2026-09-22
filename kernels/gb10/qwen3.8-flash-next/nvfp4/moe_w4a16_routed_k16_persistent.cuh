// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

#ifndef FLASH_NEXT_EXACT_K16_PARENT_INCLUDED
#include "moe_w4a16_exact_k16.cu"
#define FLASH_NEXT_EXACT_K16_PARENT_INCLUDED 1
#endif

// Exact Flash-Next M16 routed-MoE ABI. The worklist is deterministic:
// expert-major, then original flat token/slot order, four routes per group.
namespace flash_next_routed_k16 {

using namespace flash_next_exact_k16;
constexpr unsigned long long MAGIC = 0x51464e52544b3136ULL; // QFNRTK16
constexpr unsigned int VERSION = 1;
constexpr unsigned int ROUTES = ROWS * TOP_K;
constexpr unsigned int GROUP_WIDTH = 4;
constexpr unsigned int REQUIRED_SAVING_US = 7455;
constexpr unsigned int MEASURED_FFN_US = 27700;
// 139-register gate/up CTAs carry four parent-equivalent routes. Requiring an
// average >=3.56 active warps avoids a traffic-only false pass after the
// shared-only K16 candidate lost 0.764 ms median despite weight reuse.
constexpr unsigned int MAX_PLAUSIBLE_GROUPS = 45;

enum RoutedStatus : int {
    ROUTED_BUILDING = -2,
    ROUTED_READY = -3,
    ROUTED_VALIDATED = -4,
    ERR_ROUTED_ABI = 20,
    ERR_ROUTED_POINTER = 21,
    ERR_ROUTED_ALIGNMENT = 22,
    ERR_ROUTED_ALIAS = 23,
    ERR_ROUTED_OVERFLOW = 24,
    ERR_ROUTED_ROUTE = 25,
    ERR_ROUTED_WORKSPACE = 26,
};

struct alignas(16) RoutedContract {
    unsigned long long magic;
    unsigned int version, rows, hidden, intermediate, experts, top_k, reserved;
    unsigned long long input_elems, routed_inter_elems, routed_hidden_elems;
    unsigned long long routing_elems, pointer_table_elems, scale2_elems;
    unsigned long long workspace_bytes;
};

struct alignas(16) RouteGroup {
    unsigned short expert;
    unsigned char count, reserved;
    unsigned char slots[GROUP_WIDTH];
    unsigned char padding[8];
};

struct alignas(16) RoutedWorkspace {
    unsigned long long ready_magic;
    unsigned int version, group_count, route_count, saved_weight_loads;
    unsigned int ideal_weight_ceiling_us, production_eligible;
    unsigned int reserved0, reserved1, reserved2, reserved3;
    RouteGroup groups[ROUTES];
    unsigned char expert_counts[EXPERTS];
    unsigned char expert_cursors[EXPERTS];
    unsigned short expert_group_base[EXPERTS];
};

static_assert(sizeof(RouteGroup) == 16, "route-group ABI drift");
static_assert(sizeof(RoutedWorkspace) == 4656, "workspace ABI drift");

struct Range { const void* ptr; unsigned long long bytes; };

__device__ __forceinline__ bool end_ok(const void* ptr, unsigned long long bytes) {
    const uintptr_t begin = reinterpret_cast<uintptr_t>(ptr);
    return bytes <= static_cast<unsigned long long>(UINTPTR_MAX - begin);
}

__device__ __forceinline__ bool overlap(const Range& a, const Range& b) {
    const uintptr_t ab = reinterpret_cast<uintptr_t>(a.ptr);
    const uintptr_t bb = reinterpret_cast<uintptr_t>(b.ptr);
    return ab < bb + b.bytes && bb < ab + a.bytes;
}

__device__ __forceinline__ bool aligned_to(const void* ptr, uintptr_t n) {
    return ptr != nullptr && (reinterpret_cast<uintptr_t>(ptr) & (n - 1)) == 0;
}

} // namespace flash_next_routed_k16

extern "C" __global__ void moe_w4a16_routed_k16_preflight(
    const __nv_bfloat16* input,
    const unsigned long long* gate_packed, const unsigned long long* gate_scales,
    const float* gate_s2, __nv_bfloat16* gate_out,
    const unsigned long long* up_packed, const unsigned long long* up_scales,
    const float* up_s2, __nv_bfloat16* up_out,
    const unsigned long long* down_packed, const unsigned long long* down_scales,
    const float* down_s2, __nv_bfloat16* down_out,
    const unsigned int* routes,
    const flash_next_routed_k16::RoutedContract* contract,
    flash_next_routed_k16::RoutedWorkspace* workspace, int* status
) {
    using namespace flash_next_routed_k16;
    if (blockIdx.x != 0 || threadIdx.x != 0 || status == nullptr) return;
    constexpr unsigned long long IN_B = static_cast<unsigned long long>(ROWS) * HIDDEN * 2;
    constexpr unsigned long long RI_B = static_cast<unsigned long long>(ROUTES) * INTER * 2;
    constexpr unsigned long long RH_B = static_cast<unsigned long long>(ROUTES) * HIDDEN * 2;
    constexpr unsigned long long WT_B = static_cast<unsigned long long>(INTER) * HIDDEN / 2;
    constexpr unsigned long long WS_B = static_cast<unsigned long long>(INTER) * HIDDEN / GROUP;
    constexpr unsigned long long TAB_B = static_cast<unsigned long long>(EXPERTS) * 8;
    constexpr unsigned long long S2_B = static_cast<unsigned long long>(EXPERTS) * 4;
    const Range writes[] = {{gate_out, RI_B}, {up_out, RI_B}, {down_out, RH_B},
                            {workspace, sizeof(RoutedWorkspace)}};
    const Range reads[] = {{input, IN_B}, {routes, ROUTES * 4ULL},
        {gate_packed, TAB_B}, {gate_scales, TAB_B}, {gate_s2, S2_B},
        {up_packed, TAB_B}, {up_scales, TAB_B}, {up_s2, S2_B},
        {down_packed, TAB_B}, {down_scales, TAB_B}, {down_s2, S2_B},
        {contract, sizeof(RoutedContract)}};
    const Range status_range{status, sizeof(int)};
    if (!aligned_to(status, 4) || !end_ok(status, sizeof(int))) return;
    for (const Range* set : {writes, reads}) {
        const unsigned int n = set == writes ? 4 : 12;
        for (unsigned int i = 0; i < n; ++i) {
            if (set[i].ptr == nullptr || !end_ok(set[i].ptr, set[i].bytes)) return;
            if (overlap(status_range, set[i])) return;
        }
    }
    if (status[0] != PENDING) return;
    // Before the first status write, prove it cannot alias any selected
    // indirect expert payload. Unsafe or unrepresentable ranges stay PENDING.
    if (!aligned_to(routes, 4) || !aligned_to(gate_packed, 8) ||
        !aligned_to(gate_scales, 8) || !aligned_to(up_packed, 8) ||
        !aligned_to(up_scales, 8) || !aligned_to(down_packed, 8) ||
        !aligned_to(down_scales, 8)) return;
    for (unsigned int slot = 0; slot < ROUTES; ++slot) {
        const unsigned int e = routes[slot]; if (e >= EXPERTS) continue;
        const unsigned long long ptrs[] = {gate_packed[e], gate_scales[e], up_packed[e],
                                           up_scales[e], down_packed[e], down_scales[e]};
        for (unsigned int i = 0; i < 6; ++i) if (ptrs[i] != 0) {
            const Range indirect{reinterpret_cast<const void*>(ptrs[i]), (i & 1) == 0 ? WT_B : WS_B};
            if (!end_ok(indirect.ptr, indirect.bytes) || overlap(status_range, indirect)) return;
        }
    }
    status[0] = ROUTED_BUILDING;
    if (!aligned_to(input, 16) || !aligned_to(routes, 4) ||
        !aligned_to(gate_packed, 8) || !aligned_to(gate_scales, 8) ||
        !aligned_to(up_packed, 8) || !aligned_to(up_scales, 8) ||
        !aligned_to(down_packed, 8) || !aligned_to(down_scales, 8) ||
        !aligned_to(gate_s2, 4) || !aligned_to(up_s2, 4) || !aligned_to(down_s2, 4) ||
        !aligned_to(gate_out, 2) || !aligned_to(up_out, 2) || !aligned_to(down_out, 2) ||
        !aligned_to(contract, 16) || !aligned_to(workspace, 16)) {
        status[0] = ERR_ROUTED_ALIGNMENT; return;
    }
    if (contract->magic != MAGIC || contract->version != VERSION || contract->rows != ROWS ||
        contract->hidden != HIDDEN || contract->intermediate != INTER ||
        contract->experts != EXPERTS || contract->top_k != TOP_K || contract->reserved != 0) {
        status[0] = ERR_ROUTED_ABI; return;
    }
    if (contract->input_elems != static_cast<unsigned long long>(ROWS) * HIDDEN ||
        contract->routed_inter_elems != static_cast<unsigned long long>(ROUTES) * INTER ||
        contract->routed_hidden_elems != static_cast<unsigned long long>(ROUTES) * HIDDEN ||
        contract->routing_elems != ROUTES || contract->pointer_table_elems != EXPERTS ||
        contract->scale2_elems != EXPERTS || contract->workspace_bytes != sizeof(RoutedWorkspace)) {
        status[0] = ERR_ROUTED_WORKSPACE; return;
    }
    for (unsigned int i = 0; i < 4; ++i) for (unsigned int j = i + 1; j < 4; ++j)
        if (overlap(writes[i], writes[j])) { status[0] = ERR_ROUTED_ALIAS; return; }
    for (const auto& w : writes) for (const auto& r : reads)
        if (overlap(w, r)) { status[0] = ERR_ROUTED_ALIAS; return; }
    // Validate all selected indirect ranges before the first workspace write.
    // Route duplicates are intentionally rechecked: this keeps admission
    // independent of the not-yet-published worklist.
    for (unsigned int slot = 0; slot < ROUTES; ++slot) {
        const unsigned int e = routes[slot];
        if (e >= EXPERTS || gate_packed[e] == 0 || gate_scales[e] == 0 ||
            up_packed[e] == 0 || up_scales[e] == 0 || down_packed[e] == 0 ||
            down_scales[e] == 0 || !isfinite(gate_s2[e]) || !isfinite(up_s2[e]) ||
            !isfinite(down_s2[e])) { status[0] = ERR_ROUTED_ROUTE; return; }
        const Range indirect[] = {
            {reinterpret_cast<const void*>(gate_packed[e]), WT_B},
            {reinterpret_cast<const void*>(gate_scales[e]), WS_B},
            {reinterpret_cast<const void*>(up_packed[e]), WT_B},
            {reinterpret_cast<const void*>(up_scales[e]), WS_B},
            {reinterpret_cast<const void*>(down_packed[e]), WT_B},
            {reinterpret_cast<const void*>(down_scales[e]), WS_B}};
        for (unsigned int i = 0; i < 6; ++i) {
            if (!end_ok(indirect[i].ptr, indirect[i].bytes)) {
                status[0] = ERR_ROUTED_OVERFLOW; return;
            }
            if (!aligned_to(indirect[i].ptr, (i & 1) == 0 ? 8 : 1)) {
                status[0] = ERR_ROUTED_ALIGNMENT; return;
            }
            for (const auto& w : writes) if (overlap(w, indirect[i])) {
                status[0] = ERR_ROUTED_ALIAS; return;
            }
            if (overlap(status_range, indirect[i])) return;
        }
    }
    workspace->ready_magic = 0;
    for (unsigned int e = 0; e < EXPERTS; ++e) {
        workspace->expert_counts[e] = 0; workspace->expert_cursors[e] = 0;
        workspace->expert_group_base[e] = 0;
    }
    for (unsigned int slot = 0; slot < ROUTES; ++slot)
        ++workspace->expert_counts[routes[slot]];
    unsigned int group_count = 0;
    for (unsigned int e = 0; e < EXPERTS; ++e) {
        workspace->expert_group_base[e] = static_cast<unsigned short>(group_count);
        group_count += (workspace->expert_counts[e] + GROUP_WIDTH - 1) / GROUP_WIDTH;
    }
    if (group_count == 0 || group_count > ROUTES) { status[0] = ERR_ROUTED_ROUTE; return; }
    for (unsigned int g = 0; g < ROUTES; ++g)
        workspace->groups[g] = RouteGroup{0, 0, 0, {0, 0, 0, 0}, {0}};
    for (unsigned int slot = 0; slot < ROUTES; ++slot) {
        const unsigned int e = routes[slot], cursor = workspace->expert_cursors[e]++;
        RouteGroup& group = workspace->groups[workspace->expert_group_base[e] + cursor / GROUP_WIDTH];
        group.expert = static_cast<unsigned short>(e);
        group.slots[group.count++] = static_cast<unsigned char>(slot);
    }
    const unsigned int saved = ROUTES - group_count;
    workspace->version = VERSION; workspace->group_count = group_count;
    workspace->route_count = ROUTES; workspace->saved_weight_loads = saved;
    workspace->ideal_weight_ceiling_us = static_cast<unsigned int>(
        static_cast<unsigned long long>(MEASURED_FFN_US) * saved / ROUTES);
    workspace->production_eligible = group_count <= MAX_PLAUSIBLE_GROUPS &&
        workspace->ideal_weight_ceiling_us >= REQUIRED_SAVING_US;
    workspace->reserved0 = 0; workspace->reserved1 = 0;
    workspace->reserved2 = 0; workspace->reserved3 = 0;
    __threadfence(); workspace->ready_magic = MAGIC; __threadfence();
    status[0] = ROUTED_READY;
}
