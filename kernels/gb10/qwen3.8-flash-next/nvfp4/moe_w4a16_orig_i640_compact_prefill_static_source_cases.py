# SPDX-License-Identifier: AGPL-3.0-only
from __future__ import annotations

import json
from moe_w4a16_orig_i640_compact_prefill_static_support import (
    FILES,
    MODEL,
    ROOT,
    STEM,
    offsets_from_counts,
    planned_items,
    safetensor_header,
    sha,
    text,
)


class SourceCases:
    def test_01_caps_spdx_and_default_off(self) -> None:
        for name in FILES:
            source = text(name)
            self.assertLessEqual(len(source.splitlines()), 250, name)
            self.assertTrue(
                source.startswith("# SPDX-License-Identifier: AGPL-3.0-only")
                or source.startswith("// SPDX-License-Identifier: AGPL-3.0-only"),
                name,
            )
        integration = []
        for path in ROOT.rglob("*"):
            if path.is_file() and path.suffix in {".rs", ".json", ".toml"}:
                try:
                    if STEM in path.read_text():
                        integration.append(path)
                except UnicodeDecodeError:
                    pass
        self.assertEqual(integration, [])

    def test_02_exact_contract_and_fixed_geometry(self) -> None:
        header = text(f"{STEM}.cuh")
        for token in (
            "HIDDEN = 2560",
            "INTER = 640",
            "EXPERTS = 512",
            "TOP_K = 10",
            "M_SMALL = 2013",
            "M_LARGE = 8192",
            "FIXED_GRID = 4096",
            "MAX_WORK_ITEMS == 1792",
            "sizeof(Workspace) == 14384",
        ):
            self.assertIn(token, header)
        cases = text(f"{STEM}_microgate_cases.cuh")
        self.assertGreaterEqual(cases.count("<<<oi640::FIXED_GRID, 128"), 3)
        self.assertIn("cudaStreamNonBlocking", text(f"{STEM}_microgate.cu"))
        cuda = "".join(text(name) for name in FILES if name.endswith((".cu", ".cuh")))
        self.assertNotIn("cudaDevice" + "Synchronize", cuda)

    def test_03_original_parent_arithmetic_and_no_transposed_tables(self) -> None:
        gemm = text(f"{STEM}_gemm.cuh")
        parent = text("../../qwen3-next-80b-a3b/nvfp4/moe_w4a16_grouped_gemm.cu")
        for token in (
            "gn) * half_K + gk / 2",
            "gn) * groups + k_base / oi640::GROUP_K",
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32",
        ):
            self.assertIn(token, gemm)
        self.assertIn("gn * half_K + k_pair", parent)
        self.assertNotIn("qwen3-next-80b-a3b", text(f"{STEM}.cu"))
        self.assertIn("qwen3-next-80b-a3b", text(f"{STEM}_microgate.cu"))
        bundle = "".join(text(name) for name in FILES if name.endswith((".cu", ".cuh")))
        for forbidden in (
            "ptrs" + "_t",
            "ptrtable" + "_t",
            "gate_ptrs" + "_t",
            "up_ptrs" + "_t",
            "down_ptrs" + "_t",
        ):
            self.assertNotIn(forbidden, bundle)

    def test_04_cpu_worklist_capacity_and_coverage(self) -> None:
        for rows, counts in (
            (2013, [20130] + [0] * 511),
            (2013, [39] * 512),
            (8192, [160] * 512),
            (8192, [81920] + [0] * 511),
        ):
            delta = rows * 10 - sum(counts)
            counts[0] += delta
            offsets = offsets_from_counts(counts)
            items = planned_items(counts)
            self.assertEqual(offsets[-1], rows * 10)
            self.assertLessEqual(len(items), 1792)
            self.assertEqual(sum((count + 63) // 64 for count in counts), len(items))
            self.assertEqual(
                {expert for expert, _ in items},
                {i for i, count in enumerate(counts) if count},
            )

    def test_05_exact_local_config_and_weight_headers(self) -> None:
        self.assertEqual(
            sha(MODEL / "config.json"),
            "e765305daba0951974308f4d32c075b52a6a45974730d273f2216718a994d624",
        )
        self.assertEqual(
            sha(MODEL / "model.safetensors.index.json"),
            "c654034a19be39baf2348dc02c818b555d3e0f2dc036346f58fe3623c1bc311d",
        )
        cfg = json.loads((MODEL / "config.json").read_text())["text_config"]
        self.assertEqual(
            (
                cfg["hidden_size"],
                cfg["moe_intermediate_size"],
                cfg["num_experts"],
                cfg["num_experts_per_tok"],
            ),
            (2560, 640, 512, 10),
        )
        shard = MODEL / "layer-00000-experts-0000-0127.safetensors"
        hdr = safetensor_header(shard)
        base = "model.language_model.layers.0.mlp.experts.0."
        self.assertEqual(hdr[base + "gate_proj.weight"]["shape"], [640, 1280])
        self.assertEqual(hdr[base + "gate_proj.weight_scale"]["shape"], [640, 160])
        self.assertEqual(hdr[base + "down_proj.weight"]["shape"], [2560, 320])
        self.assertEqual(hdr[base + "down_proj.weight_scale"]["shape"], [2560, 40])

    def test_06_raw_gate_is_fail_closed_and_exact(self) -> None:
        main = text(f"{STEM}_microgate.cu")
        io = text(f"{STEM}_microgate_io.cuh")
        cases = text(f"{STEM}_microgate_cases.cuh")
        timing = text(f"{STEM}_microgate_timing.cuh")
        for token in (
            "M2013_OFFSETS",
            "M8192_OFFSETS",
            "RUN_NONCE",
            "prop.major != 12",
            "prop.minor != 1",
            "multiProcessorCount != 48",
        ):
            self.assertIn(token, main)
        self.assertIn(
            "ATLAS_OI640_CAPTURE_RECEIPT",
            text(f"{STEM}_microgate_provenance.cuh"),
        )
        self.assertIn("(file.identity.st_mode & 0222) != 0", io)
        for token in (
            "exact(parent_gate, candidate_gate)",
            "exact(parent_up, candidate_up)",
            "exact(parent_down, candidate_down)",
            "hostile",
            "guards()",
            "w.safe()",
        ):
            self.assertIn(token, cases)
        for token in (
            "22 : 32",
            "candidate_median < result.parent_median",
            "candidate_p90 < result.parent_p90",
            "paired_median > 0",
        ):
            self.assertIn(token, timing)
        self.assertIn("O_EXCL", io)
        self.assertIn("fchmod(fd, 0444)", io)
        self.assertIn("performance_claim_allowed", main)
        self.assertIn('timing && ok ? "true" : "false"', main)

    def test_07_provenance_and_hostile_planner(self) -> None:
        main = text(f"{STEM}_microgate.cu")
        for name in (
            "HEADER",
            "CUDA",
            "PLAN",
            "GEMM",
            "MICROGATE",
            "IO",
            "BUFFERS",
            "CASES",
            "TIMING",
            "PARENT",
            "CONFIG",
            "INDEX",
            "FIRST_SHARD",
            "LAST_SHARD",
            "SOURCE_BUNDLE",
            "CAPTURE_PRELOAD",
            "MANIFEST",
            "MODEL_MANIFEST",
        ):
            self.assertIn(f"OI640_{name}_SHA256", main)
        manifest = text(f"{STEM}_microgate_manifest.cuh") + text(
            f"{STEM}_microgate_provenance.cuh"
        )
        for token in (
            "oi640-build-v1",
            "oi640-capture-v1",
            "oi640-model-v1",
            "source.bundle_sha256",
            "identity_fields",
            "verify_model_manifest",
            "provenance_stable",
        ):
            self.assertIn(token, manifest)
        planner = text(f"{STEM}_plan.cuh")
        for token in (
            "ERR_ALIGNMENT",
            "ERR_ALIAS",
            "ERR_OFFSETS",
            "ERR_TOKEN",
            "ERR_CAPACITY",
            "ready_magic = 0",
            "__threadfence()",
        ):
            self.assertIn(token, planner)

    def test_08_source_syntax_shape(self) -> None:
        for name in FILES:
            source = text(name)
            self.assertNotIn("TO" + "DO", source)
            self.assertNotIn("FIX" + "ME", source)
            self.assertEqual(source.count("\t"), 0)
