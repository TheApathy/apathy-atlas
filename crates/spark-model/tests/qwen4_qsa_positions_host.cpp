// SPDX-License-Identifier: AGPL-3.0-only

// Host execution of the SAME physical-index helpers included by CUDA.
// No CUDA headers, runtime, model weights, or approximation of kernel math.
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <limits>

#if __has_include("../../../kernels/gb10/common/qwen4_qsa_positions.cuh")
#include "../../../kernels/gb10/common/qwen4_qsa_positions.cuh"

using qwen4_qsa_positions::PhysicalSlot;
using qwen4_qsa_positions::physical_query;
using qwen4_qsa_positions::physical_slot;

static void text_identity_and_remapped_pages() {
    for (uint32_t position = 0; position <= 8192; ++position) {
        // Physical page ownership is deliberately not the sequence page.
        const uint64_t page = 7 + 3 * (position / 16);
        const int64_t slot = static_cast<int64_t>(page * 16 + position % 16);
        PhysicalSlot result{};
        assert(physical_slot(slot, 16, result));
        assert(result.block == page);
        assert(result.ring_row == position % 4);
        assert(result.group == (position % 16) / 4);
        uint32_t query = 99999;
        assert(physical_query(position + 1, query));
        assert(query == position);
    }
}

static void rotary_values_do_not_own_physical_indices() {
    // Repeated image T, non-linear H/W, then shifted post-image text.
    const uint32_t rotary_t[] = {10, 10, 10, 10, 10, 10, 10, 10,
                                14, 15, 16, 17, 18, 19, 20, 21};
    unsigned int endpoints = 0;
    for (uint32_t row = 0; row < 16; ++row) {
        PhysicalSlot result{};
        assert(physical_slot(112 + row, 16, result));
        assert(result.block == 7);
        assert(result.ring_row == row % 4);
        assert(result.group == row / 4);
        if (result.ring_row == 3) ++endpoints;
        uint32_t query = 0;
        assert(physical_query(2041 + row, query));
        assert(query == 2040 + row);
        assert(query != rotary_t[row]);
    }
    assert(endpoints == 4);
}

static void invalid_metadata_leaves_outputs_untouched() {
    for (uint32_t block_size : {0u, 1u, 2u, 3u, 5u, 6u, 17u}) {
        PhysicalSlot result{91, 92, 93};
        assert(!physical_slot(112, block_size, result));
        assert(result.block == 91 && result.ring_row == 92 && result.group == 93);
    }
    PhysicalSlot result{91, 92, 93};
    assert(!physical_slot(-1, 16, result));
    assert(result.block == 91 && result.ring_row == 92 && result.group == 93);
    uint32_t query = 123;
    assert(!physical_query(0, query));
    assert(query == 123);
    assert(physical_query(std::numeric_limits<uint32_t>::max(), query));
    assert(query == std::numeric_limits<uint32_t>::max() - 1);
}

static void full_width_slot_is_not_truncated() {
    PhysicalSlot result{};
    const int64_t slot = (int64_t{1} << 40) + 11;
    assert(physical_slot(slot, 16, result));
    assert(result.block == static_cast<uint64_t>(slot) / 16);
    assert(result.ring_row == 3);
    assert(result.group == 2);
}

int main() {
    text_identity_and_remapped_pages();
    rotary_values_do_not_own_physical_indices();
    invalid_metadata_leaves_outputs_untouched();
    full_width_slot_is_not_truncated();
    std::puts("QSA physical-position production helpers: 4 cases PASS");
}
#else
int main() {
    std::fputs("missing production qwen4_qsa_positions.cuh\n", stderr);
    return 1;
}
#endif
