#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""CPU/model/static/direct-SM121 gate; CUDA is compiled but never executed."""
from __future__ import annotations

import hashlib
import pathlib
import re
import shutil
import subprocess
import tempfile
import unittest


HERE = pathlib.Path(__file__).resolve().parent
GB10 = HERE.parents[1]
HEADER = HERE / "moe_w4a16_routed_k16_persistent.cuh"
CUDA = HERE / "moe_w4a16_routed_k16_persistent.cu"
GATE = HERE / "moe_w4a16_routed_k16_persistent_microgate.cu"
PARENTS = (
    GB10 / "common" / "moe_shared_expert_fused_batch3.cu",
    GB10 / "common" / "moe_shared_expert_fused.cu",
    GB10 / "common" / "moe_expert_gemv.cu",
)
MANIFEST = HERE / "KERNEL.toml"


def nvcc() -> str:
    for item in (
        shutil.which("nvcc"),
        "/usr/local/cuda-13.0/bin/nvcc",
        "/usr/local/cuda/bin/nvcc",
    ):
        if item and pathlib.Path(item).is_file():
            return item
    raise unittest.SkipTest("CUDA 13 nvcc unavailable")


def command(*sources: pathlib.Path, output: pathlib.Path, cubin: bool) -> list[str]:
    args = [
        nvcc(),
        "-std=c++17",
        "-O3",
        "--use_fast_math",
        "--fmad=false",
        "-lineinfo",
        "-gencode",
        "arch=compute_121a,code=sm_121a",
        *map(str, sources),
        "-o",
        str(output),
        "--resource-usage",
    ]
    if cubin:
        args.insert(1, "--cubin")
    return args


def groups(ids: list[int]) -> list[tuple[int, tuple[int, ...]]]:
    result: list[tuple[int, tuple[int, ...]]] = []
    for expert in range(512):
        slots = [slot for slot, value in enumerate(ids) if value == expert]
        result.extend(
            (expert, tuple(slots[i : i + 4])) for i in range(0, len(slots), 4)
        )
    return result


class RoutedK16StaticGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.header = HEADER.read_text()
        cls.source = CUDA.read_text()
        cls.gate = GATE.read_text()

    def test_files_are_isolated_small_and_default_unrouted(self) -> None:
        for path in (HEADER, CUDA, GATE, pathlib.Path(__file__)):
            self.assertLessEqual(len(path.read_text().splitlines()), 250, path.name)
        manifest = MANIFEST.read_text()
        self.assertNotIn("moe_w4a16_routed_k16", manifest)
        self.assertIn('#include "moe_w4a16_exact_k16.cu"', self.header)

    def test_cpu_worklist_covers_collision_no_collision_and_degenerate(self) -> None:
        cases = (
            ([i for i in range(160)], 160),
            ([i % 116 for i in range(160)], 116),
            ([i % 40 for i in range(160)], 40),
            ([7] * 160, 40),
        )
        for ids, expected in cases:
            work = groups(ids)
            self.assertEqual(len(work), expected)
            self.assertEqual(
                sorted(slot for _, slots in work for slot in slots), list(range(160))
            )
            self.assertTrue(all(1 <= len(slots) <= 4 for _, slots in work))
            self.assertTrue(all(list(slots) == sorted(slots) for _, slots in work))
            self.assertTrue(
                all(
                    all(ids[slot] == expert for slot in slots) for expert, slots in work
                )
            )

    def test_150_feasibility_threshold_is_exact_and_not_overclaimed(self) -> None:
        def modeled(count: int) -> int:
            return 27_700 * (160 - count) // 160

        def eligible(count: int) -> bool:
            return count <= 45 and modeled(count) >= 7_455

        self.assertEqual(modeled(116), 7617)
        self.assertFalse(eligible(116))
        self.assertEqual(modeled(40), 20_775)
        self.assertTrue(eligible(40))

    def test_preflight_is_fail_closed_before_worklist_publication(self) -> None:
        for token in (
            "PENDING",
            "ROUTED_BUILDING",
            "ROUTED_READY",
            "ROUTED_VALIDATED",
            "ready_magic = 0",
            "__threadfence(); workspace->ready_magic = MAGIC",
            "ERR_ROUTED_ALIAS",
            "ERR_ROUTED_OVERFLOW",
            "ERR_ROUTED_ROUTE",
            "ERR_ROUTED_WORKSPACE",
            "overlap(status_range",
            "contract->workspace_bytes != sizeof(RoutedWorkspace)",
        ):
            self.assertIn(token, self.header)
        ready = self.header.index("workspace->ready_magic = MAGIC")
        publish = self.header.index("status[0] = ROUTED_READY", ready)
        validate_indirect = self.header.index("const Range indirect[]")
        first_workspace_write = self.header.index("workspace->ready_magic = 0")
        status_prescan = self.header.index("const unsigned long long ptrs[]")
        first_status_write = self.header.index("status[0] = ROUTED_BUILDING")
        self.assertLess(status_prescan, first_status_write)
        self.assertLess(validate_indirect, first_workspace_write)
        self.assertLess(ready, publish)

    def test_exact_slot_and_arithmetic_order_is_preserved(self) -> None:
        self.assertGreaterEqual(self.source.count("for (int b = 0; b < 8; ++b)"), 2)
        self.assertGreaterEqual(
            self.source.count("for (int off = 16; off > 0; off >>= 1)"), 2
        )
        self.assertIn("static_cast<unsigned long long>(slot) * INTER", self.source)
        self.assertIn("static_cast<unsigned long long>(slot) * HIDDEN", self.source)
        self.assertIn("__float2bfloat16(acc[j])", self.source)
        self.assertIn("gf / (1.0f + __expf(-gf))", self.source)
        self.assertIn("work += gridDim.x", self.source)
        self.assertIn("constexpr unsigned int BLOCKS = 96", self.gate)

    def test_raw_gate_binds_full_parity_safety_and_balanced_timing(self) -> None:
        for token in (
            "f.launch_parent",
            "f.launch_serial",
            "candidate_parent",
            "parent_serial",
            "determinism",
            "no_collision",
            "collision_threshold",
            "collision_quads",
            "degenerate_one_expert",
            "workspace_alias",
            "invalid_route",
            "duplicate_slot", "same_expert_group_swap", "underfilled_split",
            "route_mismatch", "group_reserved", "group_padding", "nonzero_tail",
            "expert_regression", "expert_count", "expert_cursor", "group_base",
            "header_group_count", "hostile=24",
            "f.weights.validate()",
            "UINTPTR_MAX - 15",
            "for (int r = 0; r < 21; ++r)",
            "(r & 1) == 0",
            "dm - 3 * mad > 0",
            "frame >= 7.455f",
            "post_timing.down",
            "prop.multiProcessorCount != 48",
        ):
            self.assertIn(token, self.gate)

    def test_direct_sm121_cubin_is_deterministic_and_spill_free(self) -> None:
        with tempfile.TemporaryDirectory(prefix="routed-k16-") as tmp:
            hashes, reports = [], []
            for suffix in ("a", "b"):
                output = pathlib.Path(tmp) / f"candidate-{suffix}.cubin"
                result = subprocess.run(
                    command(CUDA, output=output, cubin=True),
                    check=True,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                reports.append(result.stdout)
                hashes.append(hashlib.sha256(output.read_bytes()).hexdigest())
            self.assertEqual(hashes[0], hashes[1])
            report = "\n".join(reports)
            self.assertNotRegex(report, r"[1-9][0-9]* bytes spill (?:stores|loads)")
            for symbol in (
                "preflight",
                "validate",
                "baseline_gate_up",
                "baseline_down",
                "gate_up",
                "silu_down",
                "finalize",
            ):
                self.assertIn(f"moe_w4a16_routed_k16_{symbol}", report)
            for symbol in ("gate_up", "silu_down"):
                match = re.search(
                    rf"Function properties for moe_w4a16_routed_k16_{symbol}\n(.*?)(?=ptxas info)",
                    reports[0],
                    re.DOTALL,
                )
                self.assertIsNotNone(match)
                self.assertIn("0 bytes stack frame", match.group(1))
            self.assertRegex(
                reports[0], r"moe_w4a16_routed_k16_gate_up[\s\S]*?Used 139 registers"
            )
            self.assertRegex(
                reports[0], r"moe_w4a16_routed_k16_silu_down[\s\S]*?Used 102 registers"
            )
            disasm = pathlib.Path(nvcc()).with_name("nvdisasm")
            if disasm.is_file():
                sass = subprocess.run(
                    [str(disasm), str(pathlib.Path(tmp) / "candidate-a.cubin")],
                    check=True,
                    text=True,
                    stdout=subprocess.PIPE,
                ).stdout
                self.assertNotRegex(sass, r"\bFFMA")

    def test_gpu_ready_microgate_links_without_execution(self) -> None:
        with tempfile.TemporaryDirectory(prefix="routed-k16-gate-") as tmp:
            output = pathlib.Path(tmp) / "microgate"
            result = subprocess.run(
                command(GATE, *PARENTS, output=output, cubin=False),
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.assertTrue(output.is_file())
            self.assertNotRegex(
                result.stdout, r"[1-9][0-9]* bytes spill (?:stores|loads)"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
