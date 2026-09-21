// SPDX-License-Identifier: AGPL-3.0-only
// Default-unrouted: canonical expert-major groups share weights across <=4 routes;
// output addresses retain original route slots and token-local blend order.
#include "moe_w4a16_routed_k16_persistent.cuh"
namespace flash_next_routed_k16 {
__device__ __forceinline__ bool worklist_ok(const RoutedWorkspace* ws) {
    if (ws == nullptr || ws->ready_magic != MAGIC || ws->version != VERSION ||
        ws->group_count == 0 || ws->group_count > ROUTES || ws->route_count != ROUTES ||
        ws->saved_weight_loads != ROUTES - ws->group_count) return false;
    const unsigned int ceiling = static_cast<unsigned int>(
        static_cast<unsigned long long>(MEASURED_FFN_US) * ws->saved_weight_loads / ROUTES);
    return ws->ideal_weight_ceiling_us == ceiling &&
        ws->production_eligible == (ws->group_count <= MAX_PLAUSIBLE_GROUPS &&
                                     ceiling >= REQUIRED_SAVING_US) &&
        ws->reserved0 == 0 && ws->reserved1 == 0 && ws->reserved2 == 0 && ws->reserved3 == 0;
}
__device__ __forceinline__ bool worklist_strict(
    const RoutedWorkspace* ws, const unsigned int* routes
) {
    if (!worklist_ok(ws) || routes == nullptr) return false;
    for (unsigned int slot = 0; slot < ROUTES; ++slot)
        if (routes[slot] >= EXPERTS) return false;
    unsigned int expected_group = 0;
    for (unsigned int expert = 0; expert < EXPERTS; ++expert) {
        unsigned int count = 0;
        for (unsigned int slot = 0; slot < ROUTES; ++slot) count += routes[slot] == expert;
        if (ws->expert_counts[expert] != count || ws->expert_cursors[expert] != count ||
            ws->expert_group_base[expert] != expected_group) return false;
        unsigned int next_slot = 0;
        for (unsigned int consumed = 0; consumed < count; consumed += GROUP_WIDTH) {
            if (expected_group >= ws->group_count) return false;
            const RouteGroup& group = ws->groups[expected_group++];
            const unsigned int wanted = min(GROUP_WIDTH, count - consumed);
            if (group.expert != expert || group.count != wanted || group.reserved != 0) return false;
            for (unsigned int k = 0; k < GROUP_WIDTH; ++k) {
                if (k >= wanted) { if (group.slots[k] != 0) return false; continue; }
                while (next_slot < ROUTES && routes[next_slot] != expert) ++next_slot;
                if (next_slot == ROUTES || group.slots[k] != next_slot++) return false;
            }
            for (unsigned char byte : group.padding) if (byte != 0) return false;
        }
    }
    if (expected_group != ws->group_count) return false;
    for (unsigned int g = expected_group; g < ROUTES; ++g) {
        const RouteGroup& group = ws->groups[g];
        if (group.expert || group.count || group.reserved) return false;
        for (unsigned char slot : group.slots) if (slot != 0) return false;
        for (unsigned char byte : group.padding) if (byte != 0) return false;
    }
    return true;
}
__device__ __forceinline__ void fail(int* status, int code) {
    if (status != nullptr && blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0)
        atomicCAS(status, ROUTED_VALIDATED, code);
}
__device__ __forceinline__ void grouped_gate_up(
    const RouteGroup* group, unsigned int tile, const __nv_bfloat16* input,
    const unsigned long long* packed_ptrs, const unsigned long long* scale_ptrs,
    const float* scale2, __nv_bfloat16* output, float* lut,
    unsigned long long shared_packed[32][8], unsigned char shared_scales[32][8]
) {
    const unsigned int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const bool active = warp < group->count;
    const unsigned int slot = active ? group->slots[warp] : 0;
    const unsigned int token = slot / TOP_K, expert = group->expert;
    const auto* packed = reinterpret_cast<const unsigned char*>(packed_ptrs[expert]);
    const auto* scales = reinterpret_cast<const unsigned char*>(scale_ptrs[expert]);
    const __nv_bfloat16* a = input + static_cast<unsigned long long>(token) * HIDDEN;
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    constexpr unsigned int K16 = HIDDEN / 16;
    for (unsigned int it = 0; it < (K16 + 31) / 32; ++it) {
        const unsigned int k16 = it * 32 + lane; const bool valid = k16 < K16;
        if (warp == 0) for (unsigned int j = 0; j < 8; ++j) {
            const unsigned int n = tile * 8 + j;
            shared_packed[lane][j] = valid ? *reinterpret_cast<const unsigned long long*>(
                packed + static_cast<unsigned long long>(n) * (HIDDEN / 2) + k16 * 8) : 0;
            shared_scales[lane][j] = valid ? scales[
                static_cast<unsigned long long>(n) * (HIDDEN / GROUP) + k16] : 0;
        }
        __syncthreads();
        if (active && valid) {
            const uint4 lo = reinterpret_cast<const uint4*>(a)[k16 * 2];
            const uint4 hi = reinterpret_cast<const uint4*>(a)[k16 * 2 + 1];
            const unsigned int ar[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
            for (int b = 0; b < 8; ++b) {
                __nv_bfloat16 al, ah;
                *reinterpret_cast<unsigned short*>(&al) = static_cast<unsigned short>(ar[b]);
                *reinterpret_cast<unsigned short*>(&ah) = static_cast<unsigned short>(ar[b] >> 16);
                const float afl = __bfloat162float(al), afh = __bfloat162float(ah);
                for (int j = 0; j < 8; ++j) {
                    const unsigned char v = static_cast<unsigned char>(shared_packed[lane][j] >> (b * 8));
                    __nv_fp8_e4m3 sf; *reinterpret_cast<unsigned char*>(&sf) = shared_scales[lane][j];
                    const float sc = static_cast<float>(sf) * scale2[expert];
                    acc[j] += afl * (lut[v & 15] * sc) + afh * (lut[v >> 4] * sc);
                }
            }
        }
        __syncthreads();
    }
    for (int j = 0; j < 8; ++j) for (int off = 16; off > 0; off >>= 1)
        acc[j] += __shfl_down_sync(0xffffffff, acc[j], off);
    if (active && lane == 0) for (int j = 0; j < 8; ++j)
        output[static_cast<unsigned long long>(slot) * INTER + tile * 8 + j] =
            __float2bfloat16(acc[j]);
}
__device__ __forceinline__ void grouped_down(
    const RouteGroup* group, unsigned int tile, const __nv_bfloat16* gate,
    const __nv_bfloat16* up, const unsigned long long* packed_ptrs,
    const unsigned long long* scale_ptrs, const float* scale2, __nv_bfloat16* output,
    float* lut, unsigned long long shared_packed[32][8],
    unsigned char shared_scales[32][8]
) {
    const unsigned int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const bool active = warp < group->count; const unsigned int slot = active ? group->slots[warp] : 0;
    const unsigned int expert = group->expert;
    const auto* packed = reinterpret_cast<const unsigned char*>(packed_ptrs[expert]);
    const auto* scales = reinterpret_cast<const unsigned char*>(scale_ptrs[expert]);
    const __nv_bfloat16* g = gate + static_cast<unsigned long long>(slot) * INTER;
    const __nv_bfloat16* u = up + static_cast<unsigned long long>(slot) * INTER;
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    constexpr unsigned int K16 = INTER / 16;
    for (unsigned int it = 0; it < (K16 + 31) / 32; ++it) {
        const unsigned int k16 = it * 32 + lane; const bool valid = k16 < K16;
        if (warp == 0) for (unsigned int j = 0; j < 8; ++j) {
            const unsigned int n = tile * 8 + j;
            shared_packed[lane][j] = valid ? *reinterpret_cast<const unsigned long long*>(
                packed + static_cast<unsigned long long>(n) * (INTER / 2) + k16 * 8) : 0;
            shared_scales[lane][j] = valid ? scales[
                static_cast<unsigned long long>(n) * (INTER / GROUP) + k16] : 0;
        }
        __syncthreads();
        if (active && valid) for (int b = 0; b < 8; ++b) {
            const unsigned int k = k16 * 16 + b * 2;
            const float gf = __bfloat162float(g[k]), uf = __bfloat162float(u[k]);
            const float al = (gf / (1.0f + __expf(-gf))) * uf;
            const float gf1 = __bfloat162float(g[k + 1]), uf1 = __bfloat162float(u[k + 1]);
            const float ah = (gf1 / (1.0f + __expf(-gf1))) * uf1;
            for (int j = 0; j < 8; ++j) {
                const unsigned char v = static_cast<unsigned char>(shared_packed[lane][j] >> (b * 8));
                __nv_fp8_e4m3 sf; *reinterpret_cast<unsigned char*>(&sf) = shared_scales[lane][j];
                const float sc = static_cast<float>(sf) * scale2[expert];
                acc[j] += al * (lut[v & 15] * sc) + ah * (lut[v >> 4] * sc);
            }
        }
        __syncthreads();
    }
    for (int j = 0; j < 8; ++j) for (int off = 16; off > 0; off >>= 1)
        acc[j] += __shfl_down_sync(0xffffffff, acc[j], off);
    if (active && lane == 0) for (int j = 0; j < 8; ++j)
        output[static_cast<unsigned long long>(slot) * HIDDEN + tile * 8 + j] =
            __float2bfloat16(acc[j]);
}
} // namespace flash_next_routed_k16
extern "C" __global__ void moe_w4a16_routed_k16_validate(
    const flash_next_routed_k16::RoutedWorkspace* workspace,
    const unsigned int* routes, int* status
) {
    using namespace flash_next_routed_k16;
    if (blockIdx.x != 0 || threadIdx.x != 0 || status == nullptr ||
        status[0] != ROUTED_READY) return;
    status[0] = worklist_strict(workspace, routes) ? ROUTED_VALIDATED : ERR_ROUTED_WORKSPACE;
}
// Frozen routed-only launch shapes used by the raw timing gate. These call the
// approved exact parent device bodies and are not production symbols.
extern "C" __global__ __launch_bounds__(128, 2) void moe_w4a16_routed_k16_baseline_gate_up(
    const __nv_bfloat16* input, const unsigned long long* gate_packed,
    const unsigned long long* gate_scales, const float* gate_s2, __nv_bfloat16* gate_out,
    const unsigned long long* up_packed, const unsigned long long* up_scales,
    const float* up_s2, __nv_bfloat16* up_out, const unsigned int* routes
) {
    using namespace flash_next_exact_k16;
    constexpr unsigned int TASKS = ROWS * TOP_K * (INTER / 8);
    if (blockIdx.x >= TASKS || blockIdx.y >= 2) return;
    __shared__ float lut[16];
    if (threadIdx.x < 16) lut[threadIdx.x] = E2M1_LUT[threadIdx.x]; __syncthreads();
    gate_up_routed(blockIdx.x, blockIdx.y, input,
        blockIdx.y == 0 ? gate_packed : up_packed,
        blockIdx.y == 0 ? gate_scales : up_scales,
        blockIdx.y == 0 ? gate_s2 : up_s2,
        blockIdx.y == 0 ? gate_out : up_out, routes, lut);
}
extern "C" __global__ __launch_bounds__(128, 2) void moe_w4a16_routed_k16_baseline_down(
    const __nv_bfloat16* gate, const __nv_bfloat16* up,
    const unsigned long long* packed, const unsigned long long* scales,
    const float* scale2, __nv_bfloat16* output, const unsigned int* routes
) {
    using namespace flash_next_exact_k16;
    constexpr unsigned int TASKS = ROWS * TOP_K * (HIDDEN / 8);
    if (blockIdx.x >= TASKS) return;
    __shared__ float lut[16]; extern __shared__ float act[];
    if (threadIdx.x < 16) lut[threadIdx.x] = E2M1_LUT[threadIdx.x]; __syncthreads();
    down_routed(blockIdx.x, gate, up, packed, scales, scale2, output, routes, lut, act);
}
extern "C" __global__ __launch_bounds__(128, 2) void moe_w4a16_routed_k16_gate_up(
    const __nv_bfloat16* input, const unsigned long long* gate_packed,
    const unsigned long long* gate_scales, const float* gate_s2, __nv_bfloat16* gate_out,
    const unsigned long long* up_packed, const unsigned long long* up_scales,
    const float* up_s2, __nv_bfloat16* up_out,
    const flash_next_routed_k16::RoutedWorkspace* workspace, int* status
) {
    using namespace flash_next_routed_k16;
    if (status == nullptr || status[0] != ROUTED_VALIDATED || !worklist_ok(workspace)) {
        fail(status, ERR_ROUTED_WORKSPACE); return;
    }
    constexpr unsigned int TILES = INTER / 8;
    if (blockIdx.y >= 2) { fail(status, ERR_GEOMETRY); return; }
    __shared__ float lut[16]; __shared__ unsigned long long packed[32][8];
    __shared__ unsigned char scales[32][8];
    if (threadIdx.x < 16) lut[threadIdx.x] = E2M1_LUT[threadIdx.x]; __syncthreads();
    const unsigned int total = workspace->group_count * TILES;
    for (unsigned int work = blockIdx.x; work < total; work += gridDim.x) {
        const RouteGroup* group = &workspace->groups[work / TILES];
        grouped_gate_up(group, work % TILES, input,
            blockIdx.y == 0 ? gate_packed : up_packed,
            blockIdx.y == 0 ? gate_scales : up_scales,
            blockIdx.y == 0 ? gate_s2 : up_s2,
            blockIdx.y == 0 ? gate_out : up_out, lut, packed, scales);
        __syncthreads();
    }
}
extern "C" __global__ __launch_bounds__(128, 2) void moe_w4a16_routed_k16_silu_down(
    const __nv_bfloat16* gate, const __nv_bfloat16* up,
    const unsigned long long* packed, const unsigned long long* scales, const float* scale2,
    __nv_bfloat16* output, const flash_next_routed_k16::RoutedWorkspace* workspace, int* status
) {
    using namespace flash_next_routed_k16;
    if (status == nullptr || status[0] != ROUTED_VALIDATED || !worklist_ok(workspace)) {
        fail(status, ERR_ROUTED_WORKSPACE); return;
    }
    constexpr unsigned int TILES = HIDDEN / 8;
    __shared__ float lut[16]; __shared__ unsigned long long sp[32][8];
    __shared__ unsigned char ss[32][8];
    if (threadIdx.x < 16) lut[threadIdx.x] = E2M1_LUT[threadIdx.x]; __syncthreads();
    const unsigned int total = workspace->group_count * TILES;
    for (unsigned int work = blockIdx.x; work < total; work += gridDim.x) {
        const RouteGroup* group = &workspace->groups[work / TILES];
        grouped_down(group, work % TILES, gate, up, packed, scales, scale2, output, lut, sp, ss);
        __syncthreads();
    }
}

extern "C" __global__ void moe_w4a16_routed_k16_finalize(
    const flash_next_routed_k16::RoutedWorkspace* workspace, int* status
) {
    using namespace flash_next_routed_k16;
    if (blockIdx.x == 0 && threadIdx.x == 0 && status != nullptr &&
        status[0] == ROUTED_VALIDATED && worklist_ok(workspace)) status[0] = READY;
}
