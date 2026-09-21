// SPDX-License-Identifier: AGPL-3.0-only

extern "C" __global__ __launch_bounds__(512, 1)
void moe_w4a16_orig_i640_compact_prefill_plan(
    const int* expert_offsets,
    const int* sorted_token_ids,
    const oi640::Contract* contract,
    oi640::Workspace* workspace,
    int* status
) {
    if (threadIdx.x != 0) return;
    if (gridDim.x != 1 || gridDim.y != 1 || gridDim.z != 1 ||
        blockDim.x != oi640::EXPERTS || blockDim.y != 1 || blockDim.z != 1 ||
        expert_offsets == nullptr || sorted_token_ids == nullptr || contract == nullptr ||
        workspace == nullptr || status == nullptr) return;
    const uintptr_t offsets_begin = reinterpret_cast<uintptr_t>(expert_offsets);
    const uintptr_t tokens_begin = reinterpret_cast<uintptr_t>(sorted_token_ids);
    const uintptr_t contract_begin = reinterpret_cast<uintptr_t>(contract);
    const uintptr_t workspace_begin = reinterpret_cast<uintptr_t>(workspace);
    const uintptr_t status_begin = reinterpret_cast<uintptr_t>(status);
    if ((status_begin & 3) || status_begin > UINTPTR_MAX - sizeof(int)) return;
    if ((offsets_begin & 3) || (tokens_begin & 3) || (contract_begin & 15) ||
        (workspace_begin & 15)) {
        status[0] = oi640::ERR_ALIGNMENT;
        return;
    }
    const uint64_t offsets_bytes = (oi640::EXPERTS + 1ULL) * sizeof(int);
    if (offsets_begin > UINTPTR_MAX - offsets_bytes ||
        contract_begin > UINTPTR_MAX - sizeof(oi640::Contract) ||
        workspace_begin > UINTPTR_MAX - sizeof(oi640::Workspace)) {
        status[0] = oi640::ERR_ALIAS;
        return;
    }
    const auto overlaps = [](uintptr_t a, uint64_t an, uintptr_t b, uint64_t bn) {
        return an > 0 && bn > 0 && a <= UINTPTR_MAX - an && b <= UINTPTR_MAX - bn &&
            a < b + bn && b < a + an;
    };
    if (overlaps(status_begin, sizeof(int), offsets_begin, offsets_bytes) ||
        overlaps(status_begin, sizeof(int), contract_begin, sizeof(oi640::Contract)) ||
        overlaps(status_begin, sizeof(int), workspace_begin, sizeof(oi640::Workspace)) ||
        overlaps(workspace_begin, sizeof(oi640::Workspace), offsets_begin, offsets_bytes) ||
        overlaps(workspace_begin, sizeof(oi640::Workspace), contract_begin,
                 sizeof(oi640::Contract))) {
        status[0] = oi640::ERR_ALIAS;
        return;
    }
    if (status[0] != oi640::PENDING || !oi640::contract_ok(contract)) {
        status[0] = oi640::ERR_ABI;
        return;
    }
    const uint64_t token_bytes = static_cast<uint64_t>(contract->expanded) * sizeof(int);
    if (tokens_begin > UINTPTR_MAX - token_bytes ||
        overlaps(status_begin, sizeof(int), tokens_begin, token_bytes) ||
        overlaps(workspace_begin, sizeof(oi640::Workspace), tokens_begin, token_bytes) ||
        overlaps(contract_begin, sizeof(oi640::Contract), tokens_begin, token_bytes)) {
        status[0] = oi640::ERR_ALIAS;
        return;
    }
    status[0] = oi640::BUILDING;
    workspace->ready_magic = 0;
    workspace->version = oi640::VERSION;
    workspace->rows = contract->rows;
    workspace->expanded = contract->expanded;
    workspace->work_count = 0;
    workspace->active_experts = 0;
    workspace->max_rows = 0;
    workspace->reserved = 0;
    workspace->census_hash = 1469598103934665603ULL;
    for (uint32_t i = 0; i < oi640::MAX_WORK_ITEMS; ++i)
        workspace->items[i] = {0xffffffffU, 0xffffffffU};
    if (expert_offsets[0] != 0 || expert_offsets[oi640::EXPERTS] != contract->expanded) {
        status[0] = oi640::ERR_OFFSETS;
        return;
    }
    uint32_t count = 0;
    for (uint32_t expert = 0; expert < oi640::EXPERTS; ++expert) {
        const int begin = expert_offsets[expert];
        const int end = expert_offsets[expert + 1];
        if (begin < 0 || end < begin || static_cast<uint32_t>(end) > contract->expanded) {
            status[0] = oi640::ERR_OFFSETS;
            return;
        }
        const uint32_t rows = static_cast<uint32_t>(end - begin);
        workspace->census_hash = (workspace->census_hash ^ rows) * 1099511628211ULL;
        if (rows == 0) continue;
        ++workspace->active_experts;
        workspace->max_rows = workspace->max_rows > rows ? workspace->max_rows : rows;
        const uint32_t tiles = (rows + oi640::TILE_M - 1) / oi640::TILE_M;
        if (count + tiles > oi640::MAX_WORK_ITEMS) {
            status[0] = oi640::ERR_CAPACITY;
            return;
        }
        for (uint32_t tile = 0; tile < tiles; ++tile)
            workspace->items[count++] = {expert, tile};
    }
    for (uint32_t i = 0; i < contract->expanded; ++i) {
        const int token = sorted_token_ids[i];
        if (token < 0 || static_cast<uint32_t>(token) >= contract->rows) {
            status[0] = oi640::ERR_TOKEN;
            return;
        }
        workspace->census_hash =
            (workspace->census_hash ^ static_cast<uint32_t>(token)) * 1099511628211ULL;
    }
    if (count == 0) {
        status[0] = oi640::ERR_PLAN;
        return;
    }
    workspace->work_count = count;
    __threadfence();
    workspace->ready_magic = oi640::MAGIC;
    __threadfence();
    status[0] = oi640::PLANNED;
}
