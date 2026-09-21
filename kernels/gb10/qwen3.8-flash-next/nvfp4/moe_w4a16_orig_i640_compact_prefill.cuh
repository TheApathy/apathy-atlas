// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math.h>
#include <stdint.h>

namespace flash_next_orig_i640_prefill {

constexpr uint64_t MAGIC = 0x4f52494749363430ULL;
constexpr uint32_t VERSION = 1;
constexpr uint32_t HIDDEN = 2560;
constexpr uint32_t INTER = 640;
constexpr uint32_t EXPERTS = 512;
constexpr uint32_t TOP_K = 10;
constexpr uint32_t TILE_M = 64;
constexpr uint32_t TILE_N = 64;
#ifndef OI640_STEP_K
#define OI640_STEP_K 16
#endif
#ifndef OI640_TRANSPOSED
#define OI640_TRANSPOSED 0
#endif
static_assert(OI640_TRANSPOSED == 0 || OI640_TRANSPOSED == 1);
constexpr uint32_t STEP_K = OI640_STEP_K;
static_assert(STEP_K == 16 || STEP_K == 32);
constexpr uint32_t PAD_K = 2;
constexpr uint32_t GROUP_K = 16;
constexpr uint32_t FIXED_GRID = 4096;
constexpr uint32_t M_SMALL = 2013;
constexpr uint32_t M_LARGE = 8192;
constexpr uint32_t MAX_EXPANDED = M_LARGE * TOP_K;
constexpr uint32_t MAX_WORK_ITEMS = (MAX_EXPANDED + TILE_M - 1) / TILE_M + EXPERTS;

enum Status : int {
    PENDING = -1,
    BUILDING = -2,
    PLANNED = 1,
    ERR_ABI = 10,
    ERR_ALIGNMENT = 11,
    ERR_ALIAS = 12,
    ERR_OFFSETS = 13,
    ERR_TOKEN = 14,
    ERR_CAPACITY = 15,
    ERR_PLAN = 16,
    ERR_GEOMETRY = 17,
    ERR_POINTER = 18,
};

struct alignas(16) Contract {
    uint64_t magic;
    uint32_t version;
    uint32_t rows;
    uint32_t hidden;
    uint32_t intermediate;
    uint32_t experts;
    uint32_t top_k;
    uint32_t expanded;
    uint32_t m_tile;
    uint32_t n_tile;
    uint32_t fixed_grid;
    uint32_t max_work_items;
    uint32_t workspace_bytes;
    uint32_t reserved0;
    uint32_t reserved1;
};

struct WorkItem {
    uint32_t expert;
    uint32_t m_tile;
};

struct alignas(16) Workspace {
    uint64_t ready_magic;
    uint32_t version;
    uint32_t rows;
    uint32_t expanded;
    uint32_t work_count;
    uint32_t active_experts;
    uint32_t max_rows;
    uint32_t reserved;
    uint64_t census_hash;
    WorkItem items[MAX_WORK_ITEMS];
};

static_assert(MAX_WORK_ITEMS == 1792);
static_assert(sizeof(Contract) == 64);
static_assert(sizeof(WorkItem) == 8);
static_assert(sizeof(Workspace) == 14384);

__host__ __device__ __forceinline__ bool supported_rows(uint32_t rows) {
#ifdef ATLAS_QWEN4_MOE_COMPACT_CONTRACT
    return rows >= 2 && rows <= 2048;
#else
    return rows == M_SMALL || rows == M_LARGE;
#endif
}

__host__ __device__ __forceinline__ bool contract_ok(const Contract* c) {
    return c != nullptr && c->magic == MAGIC && c->version == VERSION &&
        supported_rows(c->rows) && c->hidden == HIDDEN && c->intermediate == INTER &&
        c->experts == EXPERTS && c->top_k == TOP_K && c->expanded == c->rows * TOP_K &&
        c->m_tile == TILE_M && c->n_tile == TILE_N && c->fixed_grid == FIXED_GRID &&
        c->max_work_items == MAX_WORK_ITEMS && c->workspace_bytes == sizeof(Workspace) &&
        c->reserved0 == 0 && c->reserved1 == 0;
}

__device__ __forceinline__ bool workspace_ok(const Workspace* w, const Contract* c) {
    return w != nullptr && w->ready_magic == MAGIC && w->version == VERSION &&
        w->rows == c->rows && w->expanded == c->expanded &&
        w->work_count > 0 && w->work_count <= MAX_WORK_ITEMS &&
        w->active_experts > 0 && w->active_experts <= EXPERTS && w->reserved == 0;
}

__device__ __forceinline__ void fail(int* status, int code) {
    if (status != nullptr) atomicCAS(status, PLANNED, code);
}

} // namespace flash_next_orig_i640_prefill
