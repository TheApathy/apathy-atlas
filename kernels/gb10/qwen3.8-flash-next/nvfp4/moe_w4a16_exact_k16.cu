// SPDX-License-Identifier: AGPL-3.0-only

// Qwen3.8-Flash-Next exact K16 NVFP4 MoE candidate.
//
// This translation unit is deliberately not registered in KERNEL.toml.  It is
// an ABI-separated, default-unrouted candidate for the exact Flash-Next target:
// M=16, H=2560, routed/shared I=640, 512 experts, top-10.  Routed experts retain
// the ordinary batch3 kernel's per-token/per-slot arithmetic.  The shared
// expert changes only CTA ownership: one CTA applies an identical eight-column
// GEMV to four tokens, loading each packed shared weight/scale once rather than
// four times.  Every token/output still has the parent's lane ownership,
// k16/b order, warp reduction, BF16 stage boundaries, expert order, and final
// routed-then-shared accumulation order.
//
// The preflight is a same-stream fail-closed contract gate.  Callers initialize
// status to PENDING on the same stream, then launch it before the three compute
// kernels; any status other than READY makes every compute CTA return before
// touching an output.  The candidate needs no workspace.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math.h>
#include <stdint.h>

namespace flash_next_exact_k16 {

constexpr unsigned int ROWS = 16;
constexpr unsigned int HIDDEN = 2560;
constexpr unsigned int INTER = 640;
constexpr unsigned int EXPERTS = 512;
constexpr unsigned int TOP_K = 10;
constexpr unsigned int BLOCK = 128;
constexpr unsigned int WARPS = 4;
constexpr unsigned int GROUP = 16;
constexpr unsigned int OUTPUTS_PER_WARP = 2;
constexpr unsigned int OUTPUTS_PER_CTA = WARPS * OUTPUTS_PER_WARP;
// Geometry authority: public Qwen3.8-Flash-Next config/index and the actual
// active first/final routed shards audited on 2026-08-28.  Do not substitute
// the stale layer-0 completion-sidecar digest for ROUTED_FIRST_SHARD_SHA256.
constexpr char CONFIG_SHA256[] =
    "e765305daba0951974308f4d32c075b52a6a45974730d273f2216718a994d624";
constexpr char INDEX_SHA256[] =
    "c654034a19be39baf2348dc02c818b555d3e0f2dc036346f58fe3623c1bc311d";
constexpr char ROUTED_FIRST_SHARD_SHA256[] =
    "d367f9ed6570543e49a99db0a8f88fe12ed94117f9bb3db0d90624937cb56457";
constexpr char ROUTED_FINAL_SHARD_SHA256[] =
    "38d3c582a5b60166c9e6747dae816bc0b4fc7da19a130545a055eb1261f7b90a";
constexpr unsigned long long SELECTED_PACKED_BYTES =
    static_cast<unsigned long long>(INTER) * HIDDEN / 2;
constexpr unsigned long long SELECTED_SCALE_BYTES =
    static_cast<unsigned long long>(INTER) * HIDDEN / GROUP;
static_assert(SELECTED_PACKED_BYTES == 819200, "public routed packed extent drift");
static_assert(SELECTED_SCALE_BYTES == 102400, "public routed scale extent drift");
constexpr unsigned long long ABI_MAGIC = 0x51464e4d4f454b31ULL; // QFNMOEK1
constexpr unsigned int ABI_VERSION = 1;

enum Status : int {
    PENDING = -1,
    READY = 0,
    ERR_CONTRACT = 1,
    ERR_ABI = 2,
    ERR_GEOMETRY = 3,
    ERR_CAPACITY = 4,
    ERR_WORKSPACE = 5,
    ERR_POINTER = 6,
    ERR_ALIGNMENT = 7,
    ERR_ALIAS = 8,
    ERR_ROUTE = 9,
    ERR_RANGE_OVERFLOW = 10,
};

struct alignas(16) Contract {
    unsigned long long magic;
    unsigned int abi_version;
    unsigned int rows;
    unsigned int hidden;
    unsigned int intermediate;
    unsigned int experts;
    unsigned int top_k;
    unsigned int reserved;
    unsigned long long input_elems;
    unsigned long long routed_inter_elems;
    unsigned long long routed_hidden_elems;
    unsigned long long shared_inter_elems;
    unsigned long long shared_hidden_elems;
    unsigned long long output_elems;
    unsigned long long routing_elems;
    unsigned long long expert_weight_elems;
    unsigned long long pointer_table_elems;
    unsigned long long scale2_elems;
    unsigned long long workspace_bytes;
};

__device__ __constant__ float E2M1_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

__device__ __forceinline__ bool aligned(const void* p, uintptr_t n) {
    return p != nullptr && (reinterpret_cast<uintptr_t>(p) & (n - 1)) == 0;
}

__device__ __forceinline__ bool range_end(
    const void* p,
    unsigned long long bytes,
    uintptr_t* end
) {
    const uintptr_t begin = reinterpret_cast<uintptr_t>(p);
    if (bytes > static_cast<unsigned long long>(UINTPTR_MAX - begin)) return false;
    *end = begin + static_cast<uintptr_t>(bytes);
    return true;
}

__device__ __forceinline__ bool overlaps(
    const void* a,
    unsigned long long a_bytes,
    const void* b,
    unsigned long long b_bytes,
    bool* overflow
) {
    uintptr_t ae = 0, be = 0;
    if (!range_end(a, a_bytes, &ae) || !range_end(b, b_bytes, &be)) {
        *overflow = true;
        return true;
    }
    const uintptr_t ab = reinterpret_cast<uintptr_t>(a);
    const uintptr_t bb = reinterpret_cast<uintptr_t>(b);
    return ab < be && bb < ae;
}

__device__ __forceinline__ bool contract_geometry_ok(const Contract* c) {
    return c != nullptr && c->magic == ABI_MAGIC && c->abi_version == ABI_VERSION &&
           c->rows == ROWS && c->hidden == HIDDEN && c->intermediate == INTER &&
           c->experts == EXPERTS && c->top_k == TOP_K && c->reserved == 0;
}

__device__ __forceinline__ void publish_error(int* status, int code) {
    if (status != nullptr && threadIdx.x == 0 && blockIdx.x == 0) {
        atomicCAS(status, READY, code);
    }
}

} // namespace flash_next_exact_k16

using flash_next_exact_k16::Contract;

extern "C" __global__ void moe_w4a16_exact_k16_preflight(
    const __nv_bfloat16* input,
    const unsigned long long* gate_packed_ptrs,
    const unsigned long long* gate_scale_ptrs,
    const float* gate_scale2_vals,
    __nv_bfloat16* gate_out,
    const unsigned long long* up_packed_ptrs,
    const unsigned long long* up_scale_ptrs,
    const float* up_scale2_vals,
    __nv_bfloat16* up_out,
    const unsigned int* expert_indices,
    const unsigned char* sh_gate_packed,
    const unsigned char* sh_gate_scale,
    const __nv_bfloat16* sh_gate_out,
    const unsigned char* sh_up_packed,
    const unsigned char* sh_up_scale,
    const __nv_bfloat16* sh_up_out,
    const unsigned long long* down_packed_ptrs,
    const unsigned long long* down_scale_ptrs,
    const float* down_scale2_vals,
    __nv_bfloat16* expert_down_out,
    const unsigned char* sh_down_packed,
    const unsigned char* sh_down_scale,
    __nv_bfloat16* shared_down_out,
    const float* expert_weights,
    const __nv_bfloat16* shared_gate_weight,
    __nv_bfloat16* output,
    const Contract* contract,
    const void* workspace,
    int* status
) {
    using namespace flash_next_exact_k16;
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (status == nullptr) return;

    constexpr unsigned long long INPUT = static_cast<unsigned long long>(ROWS) * HIDDEN;
    constexpr unsigned long long ROUTED_I = static_cast<unsigned long long>(ROWS) * TOP_K * INTER;
    constexpr unsigned long long ROUTED_H = static_cast<unsigned long long>(ROWS) * TOP_K * HIDDEN;
    constexpr unsigned long long SHARED_I = static_cast<unsigned long long>(ROWS) * INTER;
    constexpr unsigned long long SHARED_H = static_cast<unsigned long long>(ROWS) * HIDDEN;
    constexpr unsigned long long ROUTES = static_cast<unsigned long long>(ROWS) * TOP_K;
    constexpr unsigned long long GP_BYTES = static_cast<unsigned long long>(INTER) * HIDDEN / 2;
    constexpr unsigned long long GP_SCALE_BYTES = static_cast<unsigned long long>(INTER) * HIDDEN / GROUP;
    constexpr unsigned long long DOWN_BYTES = static_cast<unsigned long long>(HIDDEN) * INTER / 2;
    constexpr unsigned long long DOWN_SCALE_BYTES = static_cast<unsigned long long>(HIDDEN) * INTER / GROUP;
    struct Range { const void* p; unsigned long long bytes; };
    const Range writes[] = {
        {gate_out, ROUTED_I * 2}, {up_out, ROUTED_I * 2},
        {const_cast<__nv_bfloat16*>(sh_gate_out), SHARED_I * 2},
        {const_cast<__nv_bfloat16*>(sh_up_out), SHARED_I * 2},
        {expert_down_out, ROUTED_H * 2}, {shared_down_out, SHARED_H * 2},
        {output, SHARED_H * 2}
    };
    const Range direct_reads[] = {
        {input, INPUT * 2}, {expert_indices, ROUTES * 4},
        {expert_weights, ROUTES * 4}, {shared_gate_weight, HIDDEN * 2},
        {gate_packed_ptrs, EXPERTS * 8ULL}, {gate_scale_ptrs, EXPERTS * 8ULL},
        {gate_scale2_vals, EXPERTS * 4ULL}, {up_packed_ptrs, EXPERTS * 8ULL},
        {up_scale_ptrs, EXPERTS * 8ULL}, {up_scale2_vals, EXPERTS * 4ULL},
        {down_packed_ptrs, EXPERTS * 8ULL}, {down_scale_ptrs, EXPERTS * 8ULL},
        {down_scale2_vals, EXPERTS * 4ULL}, {contract, sizeof(Contract)},
        {sh_gate_packed, GP_BYTES}, {sh_gate_scale, GP_SCALE_BYTES},
        {sh_up_packed, GP_BYTES}, {sh_up_scale, GP_SCALE_BYTES},
        {sh_down_packed, DOWN_BYTES}, {sh_down_scale, DOWN_SCALE_BYTES}
    };

    // Status is the control plane for all following kernels.  Prove its range
    // is representable and isolated before the first write; on an unsafe status
    // pointer, preserve the caller's PENDING sentinel so later kernels fail shut.
    uintptr_t status_end = 0;
    if (!aligned(status, 4) || !range_end(status, sizeof(int), &status_end)) return;
    bool overflow = false;
    for (unsigned int group = 0; group < 2; ++group) {
        const Range* ranges = group == 0 ? writes : direct_reads;
        const unsigned int count = group == 0
            ? sizeof(writes) / sizeof(writes[0])
            : sizeof(direct_reads) / sizeof(direct_reads[0]);
        for (unsigned int i = 0; i < count; ++i) {
            if (ranges[i].p == nullptr) continue;
            uintptr_t protected_end = 0;
            if (!range_end(ranges[i].p, ranges[i].bytes, &protected_end)) {
                status[0] = ERR_RANGE_OVERFLOW; return;
            }
            if (overlaps(status, sizeof(int), ranges[i].p, ranges[i].bytes, &overflow)) {
                return;
            }
        }
    }
    if (status[0] != PENDING) { status[0] = ERR_CONTRACT; return; }
    if (contract == nullptr) { status[0] = ERR_CONTRACT; return; }
    if (contract->magic != ABI_MAGIC || contract->abi_version != ABI_VERSION) {
        status[0] = ERR_ABI; return;
    }
    if (!contract_geometry_ok(contract)) { status[0] = ERR_GEOMETRY; return; }

    if (contract->input_elems < INPUT || contract->routed_inter_elems < ROUTED_I ||
        contract->routed_hidden_elems < ROUTED_H || contract->shared_inter_elems < SHARED_I ||
        contract->shared_hidden_elems < SHARED_H || contract->output_elems < SHARED_H ||
        contract->routing_elems < ROUTES || contract->expert_weight_elems < ROUTES ||
        contract->pointer_table_elems < EXPERTS || contract->scale2_elems < EXPERTS) {
        status[0] = ERR_CAPACITY; return;
    }
    if (workspace != nullptr || contract->workspace_bytes != 0) {
        status[0] = ERR_WORKSPACE; return;
    }

    const void* required[] = {
        input, gate_packed_ptrs, gate_scale_ptrs, gate_scale2_vals, gate_out,
        up_packed_ptrs, up_scale_ptrs, up_scale2_vals, up_out, expert_indices,
        sh_gate_packed, sh_gate_scale, sh_gate_out, sh_up_packed, sh_up_scale,
        sh_up_out, down_packed_ptrs, down_scale_ptrs, down_scale2_vals,
        expert_down_out, sh_down_packed, sh_down_scale, shared_down_out,
        expert_weights, shared_gate_weight, output
    };
    for (unsigned int i = 0; i < sizeof(required) / sizeof(required[0]); ++i) {
        if (required[i] == nullptr) { status[0] = ERR_POINTER; return; }
    }
    if (!aligned(input, 16) || !aligned(gate_out, 2) || !aligned(up_out, 2) ||
        !aligned(expert_down_out, 2) || !aligned(sh_gate_out, 2) ||
        !aligned(sh_up_out, 2) || !aligned(shared_down_out, 2) || !aligned(output, 2) ||
        !aligned(gate_packed_ptrs, 8) || !aligned(gate_scale_ptrs, 8) ||
        !aligned(up_packed_ptrs, 8) || !aligned(up_scale_ptrs, 8) ||
        !aligned(down_packed_ptrs, 8) || !aligned(down_scale_ptrs, 8) ||
        !aligned(gate_scale2_vals, 4) || !aligned(up_scale2_vals, 4) ||
        !aligned(down_scale2_vals, 4) || !aligned(expert_indices, 4) ||
        !aligned(expert_weights, 4) || !aligned(shared_gate_weight, 16) ||
        !aligned(contract, 16) || !aligned(status, 4) ||
        !aligned(sh_gate_packed, 8) || !aligned(sh_up_packed, 8) ||
        !aligned(sh_down_packed, 8)) {
        status[0] = ERR_ALIGNMENT; return;
    }

    for (unsigned int i = 0; i < sizeof(writes) / sizeof(writes[0]); ++i) {
        uintptr_t end = 0;
        if (!range_end(writes[i].p, writes[i].bytes, &end)) {
            status[0] = ERR_RANGE_OVERFLOW; return;
        }
        for (unsigned int j = i + 1; j < sizeof(writes) / sizeof(writes[0]); ++j) {
            if (overlaps(writes[i].p, writes[i].bytes, writes[j].p, writes[j].bytes, &overflow)) {
                status[0] = overflow ? ERR_RANGE_OVERFLOW : ERR_ALIAS; return;
            }
        }
        for (unsigned int j = 0; j < sizeof(direct_reads) / sizeof(direct_reads[0]); ++j) {
            if (overlaps(writes[i].p, writes[i].bytes,
                         direct_reads[j].p, direct_reads[j].bytes, &overflow)) {
                status[0] = overflow ? ERR_RANGE_OVERFLOW : ERR_ALIAS; return;
            }
        }
    }

    for (unsigned int i = 0; i < ROWS * TOP_K; ++i) {
        const unsigned int e = expert_indices[i];
        if (e >= EXPERTS || gate_packed_ptrs[e] == 0 || gate_scale_ptrs[e] == 0 ||
            up_packed_ptrs[e] == 0 || up_scale_ptrs[e] == 0 ||
            down_packed_ptrs[e] == 0 || down_scale_ptrs[e] == 0 ||
            !isfinite(gate_scale2_vals[e]) || !isfinite(up_scale2_vals[e]) ||
            !isfinite(down_scale2_vals[e])) {
            status[0] = ERR_ROUTE; return;
        }
        const Range indirect[] = {
            {reinterpret_cast<const void*>(gate_packed_ptrs[e]), GP_BYTES},
            {reinterpret_cast<const void*>(gate_scale_ptrs[e]), GP_SCALE_BYTES},
            {reinterpret_cast<const void*>(up_packed_ptrs[e]), GP_BYTES},
            {reinterpret_cast<const void*>(up_scale_ptrs[e]), GP_SCALE_BYTES},
            {reinterpret_cast<const void*>(down_packed_ptrs[e]), DOWN_BYTES},
            {reinterpret_cast<const void*>(down_scale_ptrs[e]), DOWN_SCALE_BYTES}
        };
        for (unsigned int r = 0; r < sizeof(indirect) / sizeof(indirect[0]); ++r) {
            uintptr_t indirect_end = 0;
            if (!range_end(indirect[r].p, indirect[r].bytes, &indirect_end)) {
                status[0] = ERR_RANGE_OVERFLOW; return;
            }
            if (overlaps(status, sizeof(int), indirect[r].p, indirect[r].bytes, &overflow)) {
                return;
            }
            if (!aligned(indirect[r].p, r % 2 == 0 ? 8 : 1)) {
                status[0] = ERR_ALIGNMENT; return;
            }
            for (unsigned int w = 0; w < sizeof(writes) / sizeof(writes[0]); ++w) {
                if (overlaps(writes[w].p, writes[w].bytes,
                             indirect[r].p, indirect[r].bytes, &overflow)) {
                    status[0] = overflow ? ERR_RANGE_OVERFLOW : ERR_ALIAS; return;
                }
            }
        }
    }
    status[0] = READY;
}

namespace flash_next_exact_k16 {

__device__ __forceinline__ void gate_up_routed(
    unsigned int task,
    unsigned int projection,
    const __nv_bfloat16* input,
    const unsigned long long* packed_ptrs,
    const unsigned long long* scale_ptrs,
    const float* scale2_vals,
    __nv_bfloat16* out,
    const unsigned int* expert_indices,
    float* lut
) {
    const unsigned int tiles = INTER / OUTPUTS_PER_CTA;
    const unsigned int flat_slot = task / tiles;
    const unsigned int tile = task % tiles;
    const unsigned int token = flat_slot / TOP_K;
    const unsigned int e = expert_indices[flat_slot];
    const unsigned char* packed = reinterpret_cast<const unsigned char*>(packed_ptrs[e]);
    const unsigned char* scales = reinterpret_cast<const unsigned char*>(scale_ptrs[e]);
    const float s2 = scale2_vals[e];
    const unsigned int tid = threadIdx.x;
    const unsigned int local_out = tid / 32;
    const unsigned int lane = tid & 31;
    const unsigned int n1 = tile * OUTPUTS_PER_CTA + local_out * 2;
    const unsigned int n2 = n1 + 1;
    const __nv_bfloat16* a = input + static_cast<unsigned long long>(token) * HIDDEN;
    float acc1 = 0.0f, acc2 = 0.0f;
    for (unsigned int k16 = lane; k16 < HIDDEN / 16; k16 += 32) {
        const uint4 lo = reinterpret_cast<const uint4*>(a)[k16 * 2];
        const uint4 hi = reinterpret_cast<const uint4*>(a)[k16 * 2 + 1];
        const unsigned int ar[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
        const unsigned long long p1 = *reinterpret_cast<const unsigned long long*>(
            packed + static_cast<unsigned long long>(n1) * (HIDDEN / 2) + k16 * 8);
        const unsigned long long p2 = *reinterpret_cast<const unsigned long long*>(
            packed + static_cast<unsigned long long>(n2) * (HIDDEN / 2) + k16 * 8);
        __nv_fp8_e4m3 f1, f2;
        *reinterpret_cast<unsigned char*>(&f1) = scales[static_cast<unsigned long long>(n1) * (HIDDEN / GROUP) + k16];
        *reinterpret_cast<unsigned char*>(&f2) = scales[static_cast<unsigned long long>(n2) * (HIDDEN / GROUP) + k16];
        const float sc1 = static_cast<float>(f1) * s2;
        const float sc2 = static_cast<float>(f2) * s2;
        #pragma unroll
        for (int b = 0; b < 8; ++b) {
            const unsigned char v1 = static_cast<unsigned char>(p1 >> (b * 8));
            const unsigned char v2 = static_cast<unsigned char>(p2 >> (b * 8));
            __nv_bfloat16 al, ah;
            *reinterpret_cast<unsigned short*>(&al) = static_cast<unsigned short>(ar[b]);
            *reinterpret_cast<unsigned short*>(&ah) = static_cast<unsigned short>(ar[b] >> 16);
            const float afl = __bfloat162float(al), afh = __bfloat162float(ah);
            acc1 += afl * (lut[v1 & 15] * sc1) + afh * (lut[v1 >> 4] * sc1);
            acc2 += afl * (lut[v2 & 15] * sc2) + afh * (lut[v2 >> 4] * sc2);
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc1 += __shfl_down_sync(0xffffffff, acc1, off);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc2 += __shfl_down_sync(0xffffffff, acc2, off);
    if (lane == 0) {
        __nv_bfloat16* dst = out + static_cast<unsigned long long>(flat_slot) * INTER;
        dst[n1] = __float2bfloat16(acc1);
        dst[n2] = __float2bfloat16(acc2);
    }
    (void)projection;
}

} // namespace flash_next_exact_k16

extern "C" __global__ __launch_bounds__(128, 2)
void moe_w4a16_exact_k16_gate_up(
    const __nv_bfloat16* input,
    const unsigned long long* gate_packed_ptrs,
    const unsigned long long* gate_scale_ptrs,
    const float* gate_scale2_vals,
    __nv_bfloat16* gate_out,
    const unsigned long long* up_packed_ptrs,
    const unsigned long long* up_scale_ptrs,
    const float* up_scale2_vals,
    __nv_bfloat16* up_out,
    const unsigned int* expert_indices,
    const unsigned char* sh_gate_packed,
    const unsigned char* sh_gate_scale,
    float sh_gate_s2,
    __nv_bfloat16* sh_gate_out,
    const unsigned char* sh_up_packed,
    const unsigned char* sh_up_scale,
    float sh_up_s2,
    __nv_bfloat16* sh_up_out,
    const Contract* contract,
    int* status
) {
    using namespace flash_next_exact_k16;
    if (status == nullptr || status[0] != READY || !contract_geometry_ok(contract)) {
        publish_error(status, ERR_CONTRACT); return;
    }
    constexpr unsigned int TILES = INTER / OUTPUTS_PER_CTA;
    constexpr unsigned int ROUTED_TASKS = ROWS * TOP_K * TILES;
    constexpr unsigned int SHARED_TASKS = (ROWS / WARPS) * TILES;
    constexpr unsigned int TASKS_PER_PROJ = ROUTED_TASKS + SHARED_TASKS;
    const unsigned int projection = blockIdx.x / TASKS_PER_PROJ;
    const unsigned int task = blockIdx.x % TASKS_PER_PROJ;
    if (projection >= 2) { publish_error(status, ERR_GEOMETRY); return; }

    __shared__ float lut[16];
    __shared__ unsigned long long shared_packed[32][8];
    __shared__ unsigned char shared_scales[32][8];
    if (threadIdx.x < 16) lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();

    if (task < ROUTED_TASKS) {
        if (projection == 0) {
            gate_up_routed(task, projection, input, gate_packed_ptrs, gate_scale_ptrs,
                           gate_scale2_vals, gate_out, expert_indices, lut);
        } else {
            gate_up_routed(task, projection, input, up_packed_ptrs, up_scale_ptrs,
                           up_scale2_vals, up_out, expert_indices, lut);
        }
        return;
    }

    const unsigned int st = task - ROUTED_TASKS;
    const unsigned int token_group = st / TILES;
    const unsigned int tile = st % TILES;
    const unsigned int warp = threadIdx.x / 32;
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int token = token_group * WARPS + warp;
    const unsigned char* packed = projection == 0 ? sh_gate_packed : sh_up_packed;
    const unsigned char* scales = projection == 0 ? sh_gate_scale : sh_up_scale;
    const float s2 = projection == 0 ? sh_gate_s2 : sh_up_s2;
    __nv_bfloat16* dst = projection == 0 ? sh_gate_out : sh_up_out;
    const __nv_bfloat16* a = input + static_cast<unsigned long long>(token) * HIDDEN;
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    constexpr unsigned int K16 = HIDDEN / 16;
    constexpr unsigned int ITERS = (K16 + 31) / 32;
    for (unsigned int it = 0; it < ITERS; ++it) {
        const unsigned int k16 = it * 32 + lane;
        const bool valid = k16 < K16;
        if (warp == 0) {
            #pragma unroll
            for (int col = 0; col < 8; ++col) {
                const unsigned int n = tile * OUTPUTS_PER_CTA + col;
                shared_packed[lane][col] = valid
                    ? *reinterpret_cast<const unsigned long long*>(packed +
                          static_cast<unsigned long long>(n) * (HIDDEN / 2) + k16 * 8)
                    : 0ULL;
                if (valid) shared_scales[lane][col] =
                    scales[static_cast<unsigned long long>(n) * (HIDDEN / GROUP) + k16];
            }
        }
        __syncthreads();
        if (valid) {
            const uint4 lo = reinterpret_cast<const uint4*>(a)[k16 * 2];
            const uint4 hi = reinterpret_cast<const uint4*>(a)[k16 * 2 + 1];
            const unsigned int ar[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
            #pragma unroll
            for (int b = 0; b < 8; ++b) {
                __nv_bfloat16 al, ah;
                *reinterpret_cast<unsigned short*>(&al) = static_cast<unsigned short>(ar[b]);
                *reinterpret_cast<unsigned short*>(&ah) = static_cast<unsigned short>(ar[b] >> 16);
                const float afl = __bfloat162float(al), afh = __bfloat162float(ah);
                #pragma unroll
                for (int col = 0; col < 8; ++col) {
                    const unsigned long long pv = shared_packed[lane][col];
                    const unsigned char v = static_cast<unsigned char>(pv >> (b * 8));
                    __nv_fp8_e4m3 sf;
                    *reinterpret_cast<unsigned char*>(&sf) = shared_scales[lane][col];
                    const float sc = static_cast<float>(sf) * s2;
                    acc[col] += afl * (lut[v & 15] * sc) + afh * (lut[v >> 4] * sc);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc[j] += __shfl_down_sync(0xffffffff, acc[j], off);
    }
    if (lane == 0) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            dst[static_cast<unsigned long long>(token) * INTER + tile * OUTPUTS_PER_CTA + j] =
                __float2bfloat16(acc[j]);
        }
    }
}

namespace flash_next_exact_k16 {

__device__ __forceinline__ void down_routed(
    unsigned int task,
    const __nv_bfloat16* gate_out,
    const __nv_bfloat16* up_out,
    const unsigned long long* packed_ptrs,
    const unsigned long long* scale_ptrs,
    const float* scale2_vals,
    __nv_bfloat16* output,
    const unsigned int* expert_indices,
    float* lut,
    float* act
) {
    const unsigned int tiles = HIDDEN / OUTPUTS_PER_CTA;
    const unsigned int flat_slot = task / tiles;
    const unsigned int tile = task % tiles;
    const unsigned int e = expert_indices[flat_slot];
    const unsigned char* packed = reinterpret_cast<const unsigned char*>(packed_ptrs[e]);
    const unsigned char* scales = reinterpret_cast<const unsigned char*>(scale_ptrs[e]);
    const float s2 = scale2_vals[e];
    const __nv_bfloat16* g = gate_out + static_cast<unsigned long long>(flat_slot) * INTER;
    const __nv_bfloat16* u = up_out + static_cast<unsigned long long>(flat_slot) * INTER;
    for (unsigned int i = threadIdx.x; i < INTER; i += BLOCK) {
        const float gf = __bfloat162float(g[i]);
        const float uf = __bfloat162float(u[i]);
        act[i] = (gf / (1.0f + __expf(-gf))) * uf;
    }
    __syncthreads();
    const unsigned int local_out = threadIdx.x / 32;
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int n1 = tile * OUTPUTS_PER_CTA + local_out * 2;
    const unsigned int n2 = n1 + 1;
    float acc1 = 0.0f, acc2 = 0.0f;
    for (unsigned int k16 = lane; k16 < INTER / 16; k16 += 32) {
        const unsigned int base = k16 * 16;
        const unsigned long long p1 = *reinterpret_cast<const unsigned long long*>(
            packed + static_cast<unsigned long long>(n1) * (INTER / 2) + k16 * 8);
        const unsigned long long p2 = *reinterpret_cast<const unsigned long long*>(
            packed + static_cast<unsigned long long>(n2) * (INTER / 2) + k16 * 8);
        __nv_fp8_e4m3 f1, f2;
        *reinterpret_cast<unsigned char*>(&f1) = scales[static_cast<unsigned long long>(n1) * (INTER / GROUP) + k16];
        *reinterpret_cast<unsigned char*>(&f2) = scales[static_cast<unsigned long long>(n2) * (INTER / GROUP) + k16];
        const float sc1 = static_cast<float>(f1) * s2;
        const float sc2 = static_cast<float>(f2) * s2;
        #pragma unroll
        for (int b = 0; b < 8; ++b) {
            const float al = act[base + b * 2];
            const float ah = act[base + b * 2 + 1];
            const unsigned char v1 = static_cast<unsigned char>(p1 >> (b * 8));
            const unsigned char v2 = static_cast<unsigned char>(p2 >> (b * 8));
            acc1 += al * (lut[v1 & 15] * sc1) + ah * (lut[v1 >> 4] * sc1);
            acc2 += al * (lut[v2 & 15] * sc2) + ah * (lut[v2 >> 4] * sc2);
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc1 += __shfl_down_sync(0xffffffff, acc1, off);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc2 += __shfl_down_sync(0xffffffff, acc2, off);
    if (lane == 0) {
        __nv_bfloat16* dst = output + static_cast<unsigned long long>(flat_slot) * HIDDEN;
        dst[n1] = __float2bfloat16(acc1);
        dst[n2] = __float2bfloat16(acc2);
    }
}

} // namespace flash_next_exact_k16

extern "C" __global__ __launch_bounds__(128, 2)
void moe_w4a16_exact_k16_silu_down(
    const __nv_bfloat16* gate_out,
    const __nv_bfloat16* up_out,
    const unsigned long long* packed_ptrs,
    const unsigned long long* scale_ptrs,
    const float* scale2_vals,
    __nv_bfloat16* expert_down_out,
    const unsigned int* expert_indices,
    const __nv_bfloat16* sh_gate_in,
    const __nv_bfloat16* sh_up_in,
    const unsigned char* sh_down_packed,
    const unsigned char* sh_down_scale,
    float sh_down_s2,
    __nv_bfloat16* sh_down_out,
    const Contract* contract,
    int* status
) {
    using namespace flash_next_exact_k16;
    if (status == nullptr || status[0] != READY || !contract_geometry_ok(contract)) {
        publish_error(status, ERR_CONTRACT); return;
    }
    constexpr unsigned int TILES = HIDDEN / OUTPUTS_PER_CTA;
    constexpr unsigned int ROUTED_TASKS = ROWS * TOP_K * TILES;
    constexpr unsigned int SHARED_TASKS = (ROWS / WARPS) * TILES;
    const unsigned int task = blockIdx.x;
    __shared__ float lut[16];
    __shared__ unsigned long long shared_packed[32][8];
    __shared__ unsigned char shared_scales[32][8];
    extern __shared__ float routed_act[];
    if (threadIdx.x < 16) lut[threadIdx.x] = E2M1_LUT[threadIdx.x];
    __syncthreads();
    if (task < ROUTED_TASKS) {
        down_routed(task, gate_out, up_out, packed_ptrs, scale_ptrs, scale2_vals,
                    expert_down_out, expert_indices, lut, routed_act);
        return;
    }
    if (task >= ROUTED_TASKS + SHARED_TASKS) {
        publish_error(status, ERR_GEOMETRY); return;
    }
    const unsigned int st = task - ROUTED_TASKS;
    const unsigned int token_group = st / TILES;
    const unsigned int tile = st % TILES;
    const unsigned int warp = threadIdx.x / 32;
    const unsigned int lane = threadIdx.x & 31;
    const unsigned int token = token_group * WARPS + warp;
    const __nv_bfloat16* g = sh_gate_in + static_cast<unsigned long long>(token) * INTER;
    const __nv_bfloat16* u = sh_up_in + static_cast<unsigned long long>(token) * INTER;
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    constexpr unsigned int K16 = INTER / 16;
    constexpr unsigned int ITERS = (K16 + 31) / 32;
    for (unsigned int it = 0; it < ITERS; ++it) {
        const unsigned int k16 = it * 32 + lane;
        const bool valid = k16 < K16;
        if (warp == 0) {
            #pragma unroll
            for (int col = 0; col < 8; ++col) {
                const unsigned int n = tile * OUTPUTS_PER_CTA + col;
                shared_packed[lane][col] = valid
                    ? *reinterpret_cast<const unsigned long long*>(sh_down_packed +
                          static_cast<unsigned long long>(n) * (INTER / 2) + k16 * 8)
                    : 0ULL;
                if (valid) shared_scales[lane][col] =
                    sh_down_scale[static_cast<unsigned long long>(n) * (INTER / GROUP) + k16];
            }
        }
        __syncthreads();
        if (valid) {
            const unsigned int base = k16 * 16;
            #pragma unroll
            for (int b = 0; b < 8; ++b) {
                const float gf = __bfloat162float(g[base + b * 2]);
                const float uf = __bfloat162float(u[base + b * 2]);
                const float al = (gf / (1.0f + __expf(-gf))) * uf;
                const float gf1 = __bfloat162float(g[base + b * 2 + 1]);
                const float uf1 = __bfloat162float(u[base + b * 2 + 1]);
                const float ah = (gf1 / (1.0f + __expf(-gf1))) * uf1;
                #pragma unroll
                for (int col = 0; col < 8; ++col) {
                    const unsigned long long pv = shared_packed[lane][col];
                    const unsigned char v = static_cast<unsigned char>(pv >> (b * 8));
                    __nv_fp8_e4m3 sf;
                    *reinterpret_cast<unsigned char*>(&sf) = shared_scales[lane][col];
                    const float sc = static_cast<float>(sf) * sh_down_s2;
                    acc[col] += al * (lut[v & 15] * sc) + ah * (lut[v >> 4] * sc);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) acc[j] += __shfl_down_sync(0xffffffff, acc[j], off);
    }
    if (lane == 0) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            sh_down_out[static_cast<unsigned long long>(token) * HIDDEN + tile * OUTPUTS_PER_CTA + j] =
                __float2bfloat16(acc[j]);
        }
    }
}

extern "C" __global__ __launch_bounds__(256, 2)
void moe_w4a16_exact_k16_weighted_sum_blend(
    __nv_bfloat16* output,
    const __nv_bfloat16* expert_out,
    const float* expert_weights,
    const __nv_bfloat16* shared_out,
    const __nv_bfloat16* input,
    const __nv_bfloat16* gate_weight,
    const Contract* contract,
    int* status
) {
    using namespace flash_next_exact_k16;
    if (status == nullptr || status[0] != READY || !contract_geometry_ok(contract)) {
        publish_error(status, ERR_CONTRACT); return;
    }
    const unsigned int token = blockIdx.y;
    if (token >= ROWS) { publish_error(status, ERR_GEOMETRY); return; }
    const unsigned int tid = threadIdx.x;
    const unsigned int warp = tid / 32;
    const unsigned int lane = tid & 31;
    const __nv_bfloat16* my_input = input + static_cast<unsigned long long>(token) * HIDDEN;
    const float* my_weights = expert_weights + token * TOP_K;
    const __nv_bfloat16* my_expert = expert_out + static_cast<unsigned long long>(token) * TOP_K * HIDDEN;
    const __nv_bfloat16* my_shared = shared_out + static_cast<unsigned long long>(token) * HIDDEN;
    __nv_bfloat16* my_output = output + static_cast<unsigned long long>(token) * HIDDEN;
    __shared__ float warp_sums[8];
    __shared__ float sigmoid_value;
    float dot = 0.0f;
    for (unsigned int k8 = tid; k8 < HIDDEN / 8; k8 += 256) {
        const uint4 av = reinterpret_cast<const uint4*>(my_input)[k8];
        const uint4 wv = reinterpret_cast<const uint4*>(gate_weight)[k8];
        const unsigned int ar[4] = {av.x, av.y, av.z, av.w};
        const unsigned int wr[4] = {wv.x, wv.y, wv.z, wv.w};
        #pragma unroll
        for (int b = 0; b < 4; ++b) {
            __nv_bfloat16 al, ah, wl, wh;
            *reinterpret_cast<unsigned short*>(&al) = static_cast<unsigned short>(ar[b]);
            *reinterpret_cast<unsigned short*>(&ah) = static_cast<unsigned short>(ar[b] >> 16);
            *reinterpret_cast<unsigned short*>(&wl) = static_cast<unsigned short>(wr[b]);
            *reinterpret_cast<unsigned short*>(&wh) = static_cast<unsigned short>(wr[b] >> 16);
            dot += __bfloat162float(al) * __bfloat162float(wl);
            dot += __bfloat162float(ah) * __bfloat162float(wh);
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) dot += __shfl_down_sync(0xffffffff, dot, off);
    if (lane == 0) warp_sums[warp] = dot;
    __syncthreads();
    if (tid == 0) {
        float sum = 0.0f;
        #pragma unroll
        for (int w = 0; w < 8; ++w) sum += warp_sums[w];
        sigmoid_value = 1.0f / (1.0f + __expf(-sum));
    }
    __syncthreads();
    const unsigned int j = blockIdx.x * 256 + tid;
    if (j >= HIDDEN) return;
    float acc = 0.0f;
    for (unsigned int e = 0; e < contract->top_k; ++e) {
        acc += my_weights[e] * __bfloat162float(my_expert[static_cast<unsigned long long>(e) * HIDDEN + j]);
    }
    acc += sigmoid_value * __bfloat162float(my_shared[j]);
    my_output[j] = __float2bfloat16(acc);
}
