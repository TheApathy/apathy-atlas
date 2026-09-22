// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

// Physical QSA ownership is independent of rotary T/H/W coordinates.
// Keep this helper CUDA-independent so host tests execute production logic.
#if defined(__CUDACC__)
#define QSA_POSITION_INLINE __host__ __device__ __forceinline__
#else
#define QSA_POSITION_INLINE inline
#endif

namespace qwen4_qsa_positions {

struct PhysicalSlot {
    unsigned long long block;
    unsigned int ring_row;
    unsigned int group;
};

QSA_POSITION_INLINE bool physical_slot(
    long long slot, unsigned int block_size, PhysicalSlot& output) {
    if (slot < 0 || block_size < 4 || block_size % 4 != 0) return false;
    const auto physical_slot = static_cast<unsigned long long>(slot);
    const auto token_in_block = static_cast<unsigned int>(physical_slot % block_size);
    output = {physical_slot / block_size, token_in_block % 4, token_in_block / 4};
    return true;
}

QSA_POSITION_INLINE bool physical_query(
    unsigned int sequence_length, unsigned int& output) {
    if (sequence_length == 0) return false;
    output = sequence_length - 1;
    return true;
}

}  // namespace qwen4_qsa_positions

#undef QSA_POSITION_INLINE
