#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""CPU/source and, when authorized, direct-SM121 compile checks for raw v2."""
from moe_w4a16_routed_k16_v2_static_source_cases import SourceCasesMixin
from moe_w4a16_routed_k16_v2_static_support import *


class RoutedK16V2StaticGate(SourceCasesMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.header = HEADER.read_text()
        cls.source = read_sources(CUDA_SOURCES)
        cls.gate = read_sources(GATE_SOURCES)
        cls.parent = PARENT.read_text()
        cls.parent_gate = PARENT_GATE.read_text()
        cls.all_v2 = cls.header + cls.source + cls.gate

    def test_direct_sm121_compile_resources_determinism_and_sass(self) -> None:
        if SOURCE_ONLY:
            self.skipTest("compile deferred by active foreign GPU/build owner")
        with tempfile.TemporaryDirectory(prefix="routed-k16-v2-fixed40-") as tmp:
            hashes: list[str] = []
            reports: list[str] = []
            outputs: list[pathlib.Path] = []
            for suffix in ("a", "b"):
                output = pathlib.Path(tmp) / f"{suffix}.cubin"
                result = subprocess.run(
                    command(CUDA, output=output, cubin=True), check=True, text=True,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                )
                outputs.append(output)
                reports.append(result.stdout)
                hashes.append(hashlib.sha256(output.read_bytes()).hexdigest())
            self.assertEqual(hashes[0], hashes[1])
            report = reports[0]
            self.assertNotRegex(report, r"[1-9][0-9]* bytes spill (?:stores|loads)")
            limits = {"plan": 64, "fixed_gate_up": 64, "silu_stage": 32, "fixed_down": 64}
            for symbol, limit in limits.items():
                match = re.search(
                    rf"moe_w4a16_routed_k16_v2_{symbol}[\s\S]*?Used (\d+) registers", report
                )
                self.assertIsNotNone(match, symbol)
                self.assertLessEqual(int(match.group(1)), limit)
            disassembler = pathlib.Path(nvcc()).with_name("nvdisasm")
            if disassembler.is_file():
                sass = subprocess.run(
                    [str(disassembler), str(outputs[0])], check=True, text=True,
                    stdout=subprocess.PIPE,
                ).stdout
                self.assertNotRegex(sass, r"\b(?:FFMA|HMMA|MMA)\b")

    def test_gpu_ready_microgate_links_without_execution(self) -> None:
        if SOURCE_ONLY:
            self.skipTest("link deferred by active foreign GPU/build owner")
        with tempfile.TemporaryDirectory(prefix="routed-k16-v2-fixed40-gate-") as tmp:
            output = pathlib.Path(tmp) / "microgate"
            result = subprocess.run(
                command(GATE, *PARENTS, output=output, cubin=False,
                        definitions=provenance_defines()), check=True, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            self.assertTrue(output.is_file())
            self.assertNotRegex(result.stdout, r"[1-9][0-9]* bytes spill (?:stores|loads)")



if __name__ == "__main__":
    unittest.main(verbosity=2)
