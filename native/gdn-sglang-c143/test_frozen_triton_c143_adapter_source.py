# SPDX-License-Identifier: AGPL-3.0-only
"""Dependency-free source/model tests for the fused GDN input adapter."""

from __future__ import annotations

import math
import re
import unittest
from pathlib import Path


SOURCE_PATH = Path(__file__).parent / "src/atlas_triton_adapters.cu"
SOURCE = SOURCE_PATH.read_text()


def qkv_owner(vector: int) -> tuple[str, int]:
    if not 0 <= vector < 1280:
        raise ValueError("vector out of range")
    if vector < 256:
        return "q", vector
    if vector < 512:
        return "k", vector - 256
    return "v", vector - 512


def state_owner(task: int, lane: int, offset: int) -> tuple[int, int, int]:
    if not (0 <= task < 768 and 0 <= lane < 256 and offset in (0, 8, 16, 24)):
        raise ValueError("state coordinate out of range")
    head, tile_index = divmod(task, 16)
    tile_k, tile_v = divmod(tile_index, 4)
    x, y = lane & 31, lane >> 5
    return head, tile_v * 32 + y + offset, tile_k * 32 + x


def gate_model(value: float) -> float:
    clamped = 1.0e-30 if value < 1.0e-30 else value
    return math.log(clamped)


def valid_ranges(regions: list[tuple[int, int]]) -> bool:
    for index, (begin, size) in enumerate(regions):
        if begin == 0 or begin & 15 or size <= 0 or begin + size > 2**64 - 1:
            return False
        for other_begin, other_size in regions[:index]:
            if begin < other_begin + other_size and other_begin < begin + size:
                return False
    return True


class FusedAdapterSourceTests(unittest.TestCase):
    def test_license_size_and_exact_entry(self) -> None:
        self.assertEqual(
            SOURCE.splitlines()[0], "// SPDX-License-Identifier: AGPL-3.0-only"
        )
        self.assertLessEqual(len(SOURCE.splitlines()), 250)
        self.assertIn('extern "C" int atlas_gdn_c143_pack_inputs_launch(', SOURCE)
        self.assertIn("if (m != 2079U && m != 8192U) return 1;", SOURCE)
        self.assertIn("if (stream == nullptr) return 2;", SOURCE)

    def test_one_launch_and_exact_task_geometry(self) -> None:
        self.assertEqual(SOURCE.count("<<<"), 1)
        self.assertIn("<<<m + kStateBlocks, 256, 0, stream>>>", SOURCE)
        self.assertIn("constexpr uint32_t kStateBlocks = kHeads * kStateTiles;", SOURCE)
        self.assertNotIn("atomic", SOURCE.lower())

    def test_qkv_vector_ownership_is_disjoint_and_exhaustive(self) -> None:
        owners = [qkv_owner(vector) for vector in range(1280)]
        self.assertEqual(len(set(owners)), 1280)
        self.assertEqual(sum(name == "q" for name, _ in owners), 256)
        self.assertEqual(sum(name == "k" for name, _ in owners), 256)
        self.assertEqual(sum(name == "v" for name, _ in owners), 768)
        self.assertEqual(owners[0], ("q", 0))
        self.assertEqual(owners[-1], ("v", 767))

    def test_state_transpose_ownership_is_disjoint_and_exhaustive(self) -> None:
        seen = set()
        for task in range(768):
            for lane in range(256):
                for offset in (0, 8, 16, 24):
                    owner = state_owner(task, lane, offset)
                    self.assertNotIn(owner, seen)
                    seen.add(owner)
        self.assertEqual(len(seen), 48 * 128 * 128)
        self.assertIn((0, 0, 0), seen)
        self.assertIn((47, 127, 127), seen)

    def test_gate_clamp_preserves_nan_and_bounds_nonpositive(self) -> None:
        self.assertTrue(math.isnan(gate_model(float("nan"))))
        self.assertEqual(gate_model(-float("inf")), math.log(1.0e-30))
        self.assertEqual(gate_model(-0.0), math.log(1.0e-30))
        self.assertEqual(gate_model(2.0), math.log(2.0))
        self.assertIn("value < 1.0e-30f ? 1.0e-30f : value", SOURCE)

    def test_exact_extents_are_checked_before_effect(self) -> None:
        checks = SOURCE.index("for (uint32_t i = 0; i < 9U; ++i)")
        launch = SOURCE.index("<<<")
        self.assertLess(checks, launch)
        expected = (
            "10240ULL * 2ULL",
            "96ULL * 4ULL",
            "2048ULL * 2ULL",
            "6144ULL * 2ULL",
            "48ULL * 4ULL",
            "48ULL * 128ULL * 128ULL * 4ULL",
        )
        self.assertTrue(all(value in SOURCE for value in expected))
        self.assertIn("supplied[i] != regions[i].bytes", SOURCE)

    def test_range_hostiles_fail_before_effect(self) -> None:
        good = [(0x1000, 0x100), (0x2000, 0x200), (0x4000, 0x100)]
        self.assertTrue(valid_ranges(good))
        hostiles = (
            [(0, 0x100)],
            [(0x1001, 0x100)],
            [(0x1000, 0)],
            [(0x1000, 0x200), (0x1100, 0x100)],
            [(2**64 - 16, 32)],
        )
        self.assertTrue(all(not valid_ranges(case) for case in hostiles))

    def test_source_has_vectorized_copy_and_transpose_barrier(self) -> None:
        self.assertRegex(
            SOURCE, r"const uint4 value = atlas_qkv\[source_row \+ vector\]"
        )
        self.assertEqual(SOURCE.count("__syncthreads();"), 1)
        self.assertIn("tile[y + offset][x] = atlas_state_hkv", SOURCE)
        self.assertIn("state_hvk[(head * kDim + value) * kDim + key]", SOURCE)
        self.assertIsNone(re.search(r"cuda(?:Device|Stream)Synchronize", SOURCE))


if __name__ == "__main__":
    unittest.main()
