// SPDX-License-Identifier: AGPL-3.0-only

// CPU execution of the actual device-header boundary helper, no CUDA runtime.
#include "../../../kernels/gb10/deepseek-v4-flash/nvfp4/deepseek_vision_bounds.cuh"
#include <algorithm>
#include <cassert>
#include <cstdio>
#include <vector>

int main() {
    constexpr unsigned int vocab = 129280;
    unsigned long checks = 0;
    for (unsigned int start : {3u, 127u, 255u, 1023u}) {
        for (unsigned int length : {3u, 16u, 128u, 256u, 381u}) {
            const unsigned int end = start + length - 1;
            std::vector<unsigned int> ids(end + 130, 42);
            ids[start - 3] = ids[start - 2] = ids[start - 1] = vocab + 1;
            ids[start] = vocab;
            for (unsigned int p = start + 1; p < end; ++p) ids[p] = vocab + 2;
            ids[end] = vocab + 4;
            for (unsigned int row = 0; row < ids.size(); ++row) {
                unsigned int lo = 0, hi = 0;
                assert(deepseek_vision_raw_bounds(ids.data(), ids.size(), vocab, row, 128, lo, hi));
                const bool image = row >= start && row <= end;
                const unsigned int left = image ? std::min(row - start, 383u) : 0;
                const unsigned int right = image ? std::min(end - row, 384u) : 0;
                const unsigned int back = std::max(127u, left);
                assert(lo == (row > back ? row - back : 0));
                assert(hi == row + right + 1);
                ++checks;
            }
        }
    }
    std::vector<unsigned int> bad = {vocab, vocab + 2, 42, vocab + 4};
    unsigned int lo = 0, hi = 0;
    assert(!deepseek_vision_raw_bounds(bad.data(), bad.size(), vocab, 1, 128, lo, hi));
    bad = {vocab, vocab + 2};
    assert(!deepseek_vision_raw_bounds(bad.data(), bad.size(), vocab, 1, 128, lo, hi));
    bad = {vocab + 5};
    assert(!deepseek_vision_raw_bounds(bad.data(), bad.size(), vocab, 0, 128, lo, hi));
    bad = {42};
    assert(!deepseek_vision_raw_bounds(bad.data(), bad.size(), vocab, 1, 128, lo, hi));
    assert(!deepseek_vision_raw_bounds(bad.data(), bad.size(), vocab, 0, 0, lo, hi));
    std::printf("DeepSeek Vision raw bounds PASS: %lu row comparisons + 5 rejection gates\n", checks);
}
