#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only

"""CPU/static and direct-sm_121 build gate for the exact K16 MoE candidate.

This test compiles but never executes CUDA code.  Runtime parity/timing stays
behind the separately reserved raw microgate binary.
"""

from __future__ import annotations

import pathlib
import re
import hashlib
import shutil
import subprocess
import tempfile
import unittest


HERE = pathlib.Path(__file__).resolve().parent
GB10 = HERE.parents[1]
CUDA = HERE / "moe_w4a16_exact_k16.cu"
MICROGATE = HERE / "moe_w4a16_exact_k16_microgate.cu"
PARENT_BATCH = GB10 / "common" / "moe_shared_expert_fused_batch3.cu"
PARENT_SERIAL = GB10 / "common" / "moe_shared_expert_fused.cu"
PARENT_BLEND = GB10 / "common" / "moe_expert_gemv.cu"
MANIFEST = HERE / "KERNEL.toml"
CONFIG_TESTS = HERE.parents[3] / "crates" / "atlas-core" / "src" / "config" / "tests.rs"


def nvcc() -> str:
    candidates = [
        shutil.which("nvcc"),
        "/usr/local/cuda-13.0/bin/nvcc",
        "/usr/local/cuda-13/bin/nvcc",
        "/usr/local/cuda/bin/nvcc",
    ]
    for candidate in candidates:
        if candidate and pathlib.Path(candidate).is_file():
            return candidate
    raise unittest.SkipTest("CUDA 13 nvcc is unavailable")


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance_definitions() -> list[str]:
    bindings = {
        "FLASH_NEXT_EXACT_SOURCE_SHA256": sha256(CUDA),
        "FLASH_NEXT_MICROGATE_SOURCE_SHA256": sha256(MICROGATE),
        "FLASH_NEXT_PARENT_BATCH_SHA256": sha256(PARENT_BATCH),
        "FLASH_NEXT_PARENT_SERIAL_SHA256": sha256(PARENT_SERIAL),
        "FLASH_NEXT_PARENT_BLEND_SHA256": sha256(PARENT_BLEND),
    }
    return [f'-D{name}=\\"{digest}\\"' for name, digest in bindings.items()]


def compile_command(
    *sources: pathlib.Path,
    output: pathlib.Path,
    cubin: bool,
    bind_provenance: bool = False,
) -> list[str]:
    command = [
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
    if bind_provenance:
        command[1:1] = provenance_definitions()
    if cubin:
        command.insert(1, "--cubin")
    return command


class ExactK16StaticGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.source = CUDA.read_text()
        cls.microgate = MICROGATE.read_text()

    def test_exact_flash_next_geometry_is_literal(self) -> None:
        literals = {
            "ROWS": 16,
            "HIDDEN": 2560,
            "INTER": 640,
            "EXPERTS": 512,
            "TOP_K": 10,
        }
        for name, value in literals.items():
            self.assertRegex(
                self.source,
                rf"constexpr unsigned int {name} = {value};",
            )
        self.assertRegex(self.source, r"ROWS\) \* TOP_K \* INTER")
        self.assertRegex(self.source, r"ROWS\) \* TOP_K \* HIDDEN")

    def test_official_i640_extents_are_exact(self) -> None:
        config_tests = CONFIG_TESTS.read_text()
        self.assertIn('"moe_intermediate_size": 640', config_tests)
        self.assertIn('"shared_expert_intermediate_size": 640', config_tests)
        rows, hidden, inter, experts, top_k = 16, 2560, 640, 512, 10
        self.assertEqual(rows * hidden * 2, 81_920)
        self.assertEqual(rows * top_k * 4, 640)
        self.assertEqual(rows * top_k * inter * 2, 204_800)
        self.assertEqual(rows * top_k * hidden * 2, 819_200)
        self.assertEqual(rows * inter * 2, 20_480)
        self.assertEqual(rows * hidden * 2, 81_920)
        self.assertEqual(inter * hidden // 2, 819_200)
        self.assertEqual(inter * hidden // 16, 102_400)
        self.assertEqual(experts * 8, 4_096)
        self.assertEqual(experts * 4, 2_048)
        for digest in (
            "e765305daba0951974308f4d32c075b52a6a45974730d273f2216718a994d624",
            "c654034a19be39baf2348dc02c818b555d3e0f2dc036346f58fe3623c1bc311d",
            "d367f9ed6570543e49a99db0a8f88fe12ed94117f9bb3db0d90624937cb56457",
            "38d3c582a5b60166c9e6747dae816bc0b4fc7da19a130545a055eb1261f7b90a",
        ):
            self.assertIn(digest, self.source)
        for symbol in (
            "CONFIG_SHA256",
            "INDEX_SHA256",
            "ROUTED_FIRST_SHARD_SHA256",
            "ROUTED_FINAL_SHARD_SHA256",
        ):
            self.assertIn(symbol, self.microgate)
        self.assertIn("SELECTED_PACKED_BYTES == 819200", self.source)
        self.assertIn("SELECTED_SCALE_BYTES == 102400", self.source)

    def test_raw_receipt_binds_sources_binary_weights_and_extents(self) -> None:
        for token in (
            "selected_packed_bytes=%llu",
            "selected_scale_bytes=%llu",
            "config_sha256=%s",
            "index_sha256=%s",
            "routed_first_shard_sha256=%s",
            "routed_final_shard_sha256=%s",
            "exact_source_sha256=%s",
            "microgate_source_sha256=%s",
            "parent_batch_sha256=%s",
            "parent_serial_sha256=%s",
            "parent_blend_sha256=%s",
            "binary_sha256=%s",
            "unbound source/binary provenance; refusing raw qualification",
            "/usr/bin/sha256sum /proc/%ld/exe",
        ):
            self.assertIn(token, self.microgate)
        self.assertNotIn("sha256sum /proc/self/exe", self.microgate)
        definitions = provenance_definitions()
        self.assertEqual(len(definitions), 5)
        for definition in definitions:
            self.assertRegex(definition, r'^-D[A-Z0-9_]+=\\"[0-9a-f]{64}\\"$')

    def test_candidate_is_abi_separated_and_default_unrouted(self) -> None:
        for symbol in (
            "moe_w4a16_exact_k16_preflight",
            "moe_w4a16_exact_k16_gate_up",
            "moe_w4a16_exact_k16_silu_down",
            "moe_w4a16_exact_k16_weighted_sum_blend",
        ):
            self.assertIn(f'extern "C" __global__', self.source)
            self.assertIn(symbol, self.source)
        self.assertNotIn("moe_w4a16_exact_k16", MANIFEST.read_text())

    def test_fail_closed_contract_covers_geometry_capacity_workspace_and_routes(self) -> None:
        for status in (
            "ERR_ABI",
            "ERR_GEOMETRY",
            "ERR_CAPACITY",
            "ERR_WORKSPACE",
            "ERR_POINTER",
            "ERR_ALIGNMENT",
            "ERR_ALIAS",
            "ERR_ROUTE",
            "ERR_RANGE_OVERFLOW",
        ):
            self.assertIn(status, self.source)
        self.assertIn("workspace != nullptr || contract->workspace_bytes != 0", self.source)
        self.assertIn("e >= EXPERTS", self.source)
        self.assertIn("status[0] != READY", self.source)
        self.assertIn("PENDING = -1", self.source)
        self.assertIn("range_end(status, sizeof(int), &status_end)", self.source)
        self.assertIn("overlaps(status, sizeof(int), ranges[i].p", self.source)

    def test_parent_order_and_bf16_boundaries_are_preserved(self) -> None:
        self.assertIn("for (unsigned int k16 = lane; k16 < HIDDEN / 16; k16 += 32)", self.source)
        self.assertIn("for (unsigned int k16 = lane; k16 < INTER / 16; k16 += 32)", self.source)
        self.assertGreaterEqual(self.source.count("for (int b = 0; b < 8; ++b)"), 4)
        self.assertGreaterEqual(self.source.count("for (int off = 16; off > 0; off >>= 1)"), 6)
        self.assertIn("dst[n1] = __float2bfloat16(acc1);", self.source)
        self.assertIn("dst[n2] = __float2bfloat16(acc2);", self.source)
        routed = self.source.index("for (unsigned int e = 0; e < contract->top_k; ++e)")
        shared = self.source.index("acc += sigmoid_value", routed)
        store = self.source.index("my_output[j] = __float2bfloat16(acc);", shared)
        self.assertLess(routed, shared)
        self.assertLess(shared, store)

    def test_shared_candidate_reuses_weights_without_changing_lane_ownership(self) -> None:
        self.assertIn("ROWS / WARPS", self.source)
        self.assertEqual(self.source.count("__shared__ unsigned long long shared_packed[32][8]"), 2)
        self.assertGreaterEqual(self.source.count("const unsigned int token = token_group * WARPS + warp"), 2)
        self.assertGreaterEqual(self.source.count("const unsigned int lane = threadIdx.x & 31"), 3)
        self.assertIn("const unsigned int flat_slot = task / tiles", self.source)

    def test_raw_gate_binds_parent_serial_canaries_immutability_and_determinism(self) -> None:
        for token in (
            "moe_expert_gate_up_shared_batch3",
            "moe_expert_silu_down_shared_batch3",
            "moe_weighted_sum_blend_batch3",
            "moe_expert_gate_up_shared",
            "moe_expert_silu_down_shared",
            "moe_weighted_sum_blend",
            "candidate_parent",
            "candidate_serial_k1",
            "candidate_determinism",
            "immutable_ok",
            "canary_ok",
            "status_alias_output",
            "status_alias_input",
            "status_range_overflow",
            "hostile_status_gates=3",
        ):
            self.assertIn(token, self.microgate)
        self.assertIn("invalid_gates=6", self.microgate)

    def test_timing_is_alternating_and_robust(self) -> None:
        self.assertIn("for (int r = 0; r < 21; ++r)", self.microgate)
        self.assertIn("(r & 1) == 0", self.microgate)
        self.assertIn("dm - 3.0f * mad > 0.0f", self.microgate)
        self.assertIn("frame_saving >= 0.5f", self.microgate)
        self.assertIn("parent_p90_ms", self.microgate)
        self.assertIn("candidate_p90_ms", self.microgate)

    def test_kernel_direct_sm121_compile_has_no_spills(self) -> None:
        with tempfile.TemporaryDirectory(prefix="flash-next-k16-static-") as tmp:
            output = pathlib.Path(tmp) / "candidate.cubin"
            result = subprocess.run(
                compile_command(CUDA, output=output, cubin=True),
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.assertTrue(output.is_file())
            self.assertNotRegex(result.stdout, r"[1-9][0-9]* bytes spill stores")
            self.assertNotRegex(result.stdout, r"[1-9][0-9]* bytes spill loads")
            for symbol in (
                "moe_w4a16_exact_k16_preflight",
                "moe_w4a16_exact_k16_gate_up",
                "moe_w4a16_exact_k16_silu_down",
                "moe_w4a16_exact_k16_weighted_sum_blend",
            ):
                self.assertIn(symbol, result.stdout)
            disassembler = pathlib.Path(nvcc()).with_name("nvdisasm")
            if disassembler.is_file():
                sass = subprocess.run(
                    [str(disassembler), str(output)],
                    check=True,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                ).stdout
                self.assertNotRegex(sass, r"\bFFMA")

    def test_gpu_ready_microgate_links_for_sm121_without_running(self) -> None:
        with tempfile.TemporaryDirectory(prefix="flash-next-k16-gate-") as tmp:
            output = pathlib.Path(tmp) / "microgate"
            result = subprocess.run(
                compile_command(
                    MICROGATE,
                    PARENT_BATCH,
                    PARENT_SERIAL,
                    PARENT_BLEND,
                    output=output,
                    cubin=False,
                    bind_provenance=True,
                ),
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            self.assertTrue(output.is_file())
            self.assertNotRegex(result.stdout, r"[1-9][0-9]* bytes spill stores")
            self.assertNotRegex(result.stdout, r"[1-9][0-9]* bytes spill loads")


if __name__ == "__main__":
    unittest.main(verbosity=2)
