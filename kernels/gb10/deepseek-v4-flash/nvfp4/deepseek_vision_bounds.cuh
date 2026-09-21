// SPDX-License-Identifier: AGPL-3.0-only

#pragma once

// Host-testable SSOT for the device's bounded sentinel scan. The full prompt
// must first pass the Rust admission validator. PAD before START is causal;
// START through END expands only raw attention, never compressed visibility.
#ifdef __CUDACC__
#define ATLAS_VISION_INLINE __device__ __forceinline__
#else
#define ATLAS_VISION_INLINE inline
#endif

ATLAS_VISION_INLINE bool deepseek_vision_raw_bounds(
    const unsigned int* ids, unsigned int count, unsigned int vocab,
    unsigned int row, unsigned int window, unsigned int& lo, unsigned int& hi
) {
    if (ids == nullptr || row >= count || window != 128u ||
        vocab == 0u || vocab > 0xfffffffau || ids[row] >= vocab + 5u)
        return false;
    lo = row >= window ? row + 1u - window : 0u;
    hi = row + 1u;
    if (ids[row] < vocab) return true;

    unsigned int start = count;
    for (unsigned int back = 0u; back <= 383u && back <= row; ++back) {
        const unsigned int position = row - back;
        const unsigned int id = ids[position];
        if (id < vocab || (back != 0u && id == vocab + 4u)) break;
        if (id >= vocab + 5u) return false;
        if (id == vocab) { start = position; break; }
    }
    if (start == count) {
        // Only pre-START compression PAD may have no preceding START.
        return ids[row] == vocab + 1u;
    }
    unsigned int end = count;
    for (unsigned int ahead = 0u; ahead <= 384u && ahead < count - row; ++ahead) {
        const unsigned int position = row + ahead;
        const unsigned int id = ids[position];
        if (id < vocab || id >= vocab + 5u || (position != start && id == vocab))
            return false;
        if (id == vocab + 4u) { end = position; break; }
    }
    if (end == count || end - start + 1u > 384u) return false;
    // Equivalent to DeepSeekImageSpan::raw_bounds(row, 128, 384).
    const unsigned int left = row - start;
    const unsigned int lookback = left > window - 1u ? left : window - 1u;
    lo = row > lookback ? row - lookback : 0u;
    hi = end + 1u;
    return true;
}

#undef ATLAS_VISION_INLINE
