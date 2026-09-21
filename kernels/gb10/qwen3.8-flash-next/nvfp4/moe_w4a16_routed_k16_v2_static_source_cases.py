# SPDX-License-Identifier: AGPL-3.0-only
"""Source-contract cases for the raw routed-MoE v2 candidate."""
from moe_w4a16_routed_k16_v2_static_support import *


class SourceCasesMixin:
    def test_official_i640_extents_and_default_unrouted_abi(self) -> None:
        for path in ALL_SOURCE_FILES:
            self.assertIn("SPDX-License-Identifier: AGPL-3.0-only",
                          "\n".join(path.read_text().splitlines()[:2]))
            self.assertLessEqual(len(path.read_text().splitlines()), 250, path.name)
        self.assertNotIn("moe_w4a16_routed_k16_v2", (HERE / "KERNEL.toml").read_text())
        for token in (
            "constexpr unsigned int INTER = 640", "constexpr unsigned int HIDDEN = 2560",
            "constexpr unsigned int EXPERTS = 512", "constexpr unsigned int TOP_K = 10",
        ):
            self.assertIn(token, self.parent)
        for token in (
            "sizeof(Contract) == 96", "sizeof(Descriptor) == 16",
            "sizeof(Plan) == 1328", "DESCRIPTORS == 40", "INPUT_ELEMS == 40960",
            "ROUTED_INTER_ELEMS == 102400", "ROUTED_HIDDEN_ELEMS == 409600",
            "SELECTED_WEIGHT_BYTES == 819200", "SELECTED_SCALE_BYTES == 102400",
            "ACTIVATION_BYTES == 409600",
        ):
            self.assertIn(token, self.header)
        self.assertEqual(ROWS * HIDDEN, 40_960)
        self.assertEqual(ROUTES * INTER * 4, 409_600)
        self.assertEqual(INTER * HIDDEN // 2, 819_200)
        self.assertEqual(INTER * HIDDEN // 16, 102_400)

    def test_fixed40_partition_covers_every_route_once(self) -> None:
        cases = (
            route_case(160), mixed_tails(), route_case(40), route_case(0),
            [0] * 7 + [1] * 5 + [2] * 3 + [3] * 145,
        )
        for ids in cases:
            descriptors = fixed40_plan(ids)
            self.assertEqual(len(descriptors), DESCRIPTORS)
            self.assertEqual(descriptors, fixed40_plan(list(ids)))
            covered = [slot for _, _, slots in descriptors for slot in slots]
            self.assertEqual(sorted(covered), list(range(ROUTES)))
            self.assertEqual(len(set(covered)), ROUTES)
            for mode, shared, slots in descriptors:
                self.assertEqual(len(slots), WIDTH)
                if mode == "homogeneous":
                    self.assertEqual({ids[slot] for slot in slots}, {shared})
                else:
                    self.assertEqual(shared, 0xFFFF)
                    self.assertGreater(len({ids[slot] for slot in slots}), 1)
        self.assertEqual(
            tuple(sum(mode == wanted for mode, _, _ in fixed40_plan(route_case(160)))
                  for wanted in ("homogeneous", "heterogeneous")),
            (0, 40),
        )
        self.assertEqual(
            tuple(sum(mode == wanted for mode, _, _ in fixed40_plan(route_case(40)))
                  for wanted in ("homogeneous", "heterogeneous")),
            (40, 0),
        )
        self.assertEqual(
            tuple(sum(mode == wanted for mode, _, _ in fixed40_plan(route_case(0)))
                  for wanted in ("homogeneous", "heterogeneous")),
            (40, 0),
        )
        with self.assertRaises(ValueError):
            fixed40_plan([0] * 159 + [EXPERTS])

    def test_device_planner_publication_parent_gate_and_coverage(self) -> None:
        for token in (
            "__launch_bounds__(160, 1)", "parent_status[0] != READY", "ERR_PARENT",
            "leader_slots[v2::DESCRIPTORS]", "tail_slots[v2::ROUTES]",
            "homogeneous_count + tail_count / v2::DESCRIPTOR_WIDTH != v2::DESCRIPTORS",
            "coverage += plan->descriptors[d].slots[i] == slot", "ready_magic = 0",
            "__threadfence()", "plan->ready_magic = v2::MAGIC",
            "status[0] = v2::PLANNED",
        ):
            self.assertIn(token, self.source + self.header)
        self.assertNotIn("routed_k16_v2_validate", self.all_v2)
        self.assertNotIn("quad_count", self.all_v2)
        self.assertNotIn("fallback_count", self.all_v2)
        isolation = self.source.index("overlap_by_distance(sr, psr)")
        first_write = self.source.index("status[0] = v2::ERR_ALIGNMENT")
        ready = self.source.index("plan->ready_magic = v2::MAGIC")
        publish = self.source.index("status[0] = v2::PLANNED", ready)
        self.assertLess(isolation, first_write)
        self.assertLess(ready, publish)
        admission = self.source[
            self.source.index("if (slot == 0)") : self.source.index("__syncthreads()")
        ]
        self.assertNotIn("return;", admission)

    def test_all_compute_launches_are_fixed_and_census_is_receipt_only(self) -> None:
        self.assertEqual(DESCRIPTORS * (INTER // 8), 3_200)
        self.assertEqual(DESCRIPTORS * (HIDDEN // 8), 12_800)
        self.assertIn("constexpr unsigned int FIXED_GRID = v2::DESCRIPTORS * TILES", self.source)
        self.assertEqual(self.source.count("gridDim.x != FIXED_GRID"), 2)
        for symbol in (
            "moe_w4a16_routed_k16_v2_fixed_gate_up", "moe_w4a16_routed_k16_v2_fixed_down",
        ):
            self.assertIn(symbol, self.source)
        self.assertNotIn("moe_w4a16_routed_k16_v2_fallback", self.all_v2)
        self.assertNotIn("moe_w4a16_routed_k16_v2_quad", self.all_v2)
        begin = self.gate.index("void launch_v2(")
        end = self.gate.index("void launch_routed_parent", begin)
        launch = self.gate[begin:end]
        for token in (
            "GATE_GRID = v2::DESCRIPTORS * (INTER / OUTPUTS_PER_CTA)",
            "DOWN_GRID = v2::DESCRIPTORS * (HIDDEN / OUTPUTS_PER_CTA)",
            "fixed_gate_up<<<GATE_GRID", "fixed_down<<<DOWN_GRID",
        ):
            self.assertIn(token, launch)
        for forbidden in (
            "Census", "expected_census", "cudaMemcpy", "cudaStreamSynchronize",
            "homogeneous_count", "heterogeneous_count",
        ):
            self.assertNotIn(forbidden, launch)
        self.assertIn("Expected\n// descriptor census is intentionally absent", self.gate)

    def test_homogeneous_shares_weights_heterogeneous_keeps_parent_order(self) -> None:
        for token in (
            "descriptor.mode == v2::HOMOGENEOUS", "staged_packed[8][K16]",
            "staged_scales[8][K16]", "packed_ptrs[expert]", "scale_ptrs[expert]",
            "for (unsigned int k16 = lane; k16 < K16; k16 += 32)",
            "for (int b = 0; b < 8; ++b)",
            "for (int offset = 16; offset > 0; offset >>= 1)", "__float2bfloat16",
        ):
            self.assertIn(token, self.source)
        gate1 = "acc1 += afl * (lut[v1 & 15] * sc1) + afh * (lut[v1 >> 4] * sc1);"
        gate2 = "acc2 += afl * (lut[v2 & 15] * sc2) + afh * (lut[v2 >> 4] * sc2);"
        down1 = "acc1 += al * (lut[v1 & 15] * sc1) + ah * (lut[v1 >> 4] * sc1);"
        down2 = "acc2 += al * (lut[v2 & 15] * sc2) + ah * (lut[v2 >> 4] * sc2);"
        self.assertIn(gate1, self.parent)
        self.assertIn(gate1, self.source)
        self.assertIn(gate2, self.source.replace("v2b", "v2"))
        self.assertIn(down1, self.parent)
        self.assertIn(down1, self.source)
        self.assertIn(down2, self.source.replace("v2b", "v2"))
        self.assertIn("(gf / (1.0f + __expf(-gf))) * uf", self.header)
        self.assertLess(self.source.index("v2::reduce_store(acc1"),
                        self.source.index("v2::reduce_store(acc2"))

    def test_raw_gate_uses_shipping_oracles_and_adversarial_census(self) -> None:
        for token in (
            "f.launch_parent(f.parent)", "f.launch_serial(f.serial)",
            ".shipping_batch3", ".shipping_serial_k1", "plan_determinism",
            "activation_determinism", "no_collision", "mixed_count1_2_3",
            "count4_only", "all_one_expert", "DISTINCT_EXPERTS = 40",
            "V2_HOSTILE cases=7", "descriptors[0].slots[1]", "immutable_ok",
            "canary_ok", "host_launch_counts=false", "launch_candidate_blend",
            "(std::string(label) + \".final\")", "shared_source.shared_down.data",
            "FLASH_NEXT_V2_HEADER_SHA256", "FLASH_NEXT_V2_EXACT_PARENT_GATE_SHA256",
            "FLASH_NEXT_V2_PLAN_SHA256", "FLASH_NEXT_V2_COMPUTE_SHA256",
            "FLASH_NEXT_V2_MICROGATE_FIXTURE_SHA256",
            "FLASH_NEXT_V2_MICROGATE_VALIDATION_SHA256",
            "FLASH_NEXT_V2_MICROGATE_TIMING_SHA256",
            "executable_sha256()", "unbound v2 source/binary provenance",
        ):
            self.assertIn(token, self.gate)
        self.assertIn("moe_expert_gate_up_shared_batch3", self.parent_gate)
        self.assertIn("moe_expert_gate_up_shared(", self.parent_gate)
        self.assertIn("prepare_parent(f, f.candidate)", self.gate)
        self.assertIn("f.preflight(output, f.contract.data, nullptr)", self.gate)
        self.assertLess(self.gate.index("prepare_parent(f, f.candidate)"),
                        self.gate.index("launch_v2(f, f.candidate"))
        self.assertLess(self.gate.index("executable_sha256()", self.gate.index("int main")),
                        self.gate.index("cudaGetDevice(&device)", self.gate.index("int main")))
        self.assertIn('"/usr/bin/sha256sum /proc/%ld/exe"', self.parent_gate)
        self.assertIn("static_cast<long>(getpid())", self.parent_gate)
        self.assertNotIn('popen("/usr/bin/sha256sum /proc/self/exe"', self.parent_gate)
        definitions = provenance_defines()
        self.assertEqual(len(definitions), 18)
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{64}", value)
                            for value in definitions.values()))
        for name in (
            "FLASH_NEXT_V2_PLAN_SHA256", "FLASH_NEXT_V2_COMPUTE_SHA256",
            "FLASH_NEXT_V2_MICROGATE_FIXTURE_SHA256",
            "FLASH_NEXT_V2_MICROGATE_VALIDATION_SHA256",
            "FLASH_NEXT_V2_MICROGATE_TIMING_SHA256",
        ):
            self.assertIn(f"digest_bound({name})", self.gate)

    def test_timing_threshold_is_robust_scoped_and_conditional(self) -> None:
        current_ms = 1000.0 / 41.8078
        self.assertAlmostEqual(current_ms, 23.9189816, places=6)
        self.assertAlmostEqual(current_ms - 1000.0 / 150.0, 17.2523150, places=6)
        self.assertAlmostEqual(77.588 - 10.52 * 1000.0 / 150.0, 7.4546667, places=6)
        self.assertAlmostEqual(7.455 / 48.0, 0.1553125, places=7)
        for token in (
            "robust_x48 = 48.0f * (delta_median - 3.0f * mad)",
            "robust_x48 >= 7.455f", "candidate=planner_plus_fixed_launches",
            "preflight=excluded", "shared_blend=excluded", "conditional=true",
            "threshold=7.455000",
        ):
            self.assertIn(token, self.gate)
