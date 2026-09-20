#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only

"""Static source/host/PTX gate for the generic contiguous-prefill ABI.

Set ``ATLAS_PREFILL_ABI_PTX`` to the ``t0__inferspark_prefill.ptx`` emitted by
the atlas-kernels build test to enable the bundled-PTX assertions.  No CUDA
kernel is loaded or executed by this test.
"""

from __future__ import annotations

import os
import pathlib
import re
import unittest


HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CUDA = HERE / "inferspark_prefill.cu"
HOST = ROOT / "crates/spark-model/src/layers/ops/prefill_attn_main_a.rs"
PTX_ENV = os.environ.get("ATLAS_PREFILL_ABI_PTX")
PTX = pathlib.Path(PTX_ENV).resolve() if PTX_ENV else None

EXPECTED_ARGS = [
    "Q",
    "K",
    "V",
    "O",
    "seq_len",
    "query_start",
    "query_len_total",
    "num_q_heads",
    "num_kv_heads",
    "head_dim",
    "inv_sqrt_d",
    "causal",
    "sliding_window",
]
EXPECTED_PTX_TYPES = [
    ".u64",
    ".u64",
    ".u64",
    ".u64",
    ".u32",
    ".u32",
    ".u32",
    ".u32",
    ".u32",
    ".u32",
    ".f32",
    ".u32",
    ".u32",
]


def function_parameters(source: str, symbol: str) -> list[str]:
    match = re.search(
        rf'extern\s+"C"\s+__global__\s+void\s+{re.escape(symbol)}\s*\((.*?)\)\s*\{{',
        source,
        re.DOTALL,
    )
    if match is None:
        raise AssertionError(f"missing CUDA entry {symbol}")
    parameters = []
    for raw_parameter in match.group(1).split(","):
        parameter = re.sub(r"//.*", "", raw_parameter).strip()
        name = re.search(r"([A-Za-z_]\w*)\s*$", parameter)
        if name is None:
            raise AssertionError(f"cannot parse parameter in {symbol}: {raw_parameter!r}")
        parameters.append(name.group(1))
    return parameters


def ptx_entry(ptx: str, symbol: str) -> str:
    match = re.search(
        rf"\.visible\s+\.entry\s+{re.escape(symbol)}\s*\((.*?)\)\s*\{{",
        ptx,
        re.DOTALL,
    )
    if match is None:
        raise AssertionError(f"missing PTX entry {symbol}")
    return match.group(1)


class InfersparkPrefillAbiGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cuda = CUDA.read_text()
        cls.host = HOST.read_text()

    def test_both_cuda_entries_have_the_exact_host_ordered_abi(self) -> None:
        for symbol in ("inferspark_prefill", "inferspark_prefill_64"):
            self.assertEqual(function_parameters(self.cuda, symbol), EXPECTED_ARGS)

        host_sequence = """.arg_u32(seq_len)
        .arg_u32(0)
        .arg_u32(seq_len)
        .arg_u32(num_q_heads)
        .arg_u32(num_kv_heads)
        .arg_u32(head_dim)
        .arg_f32(inv_sqrt_d)
        .arg_u32(if causal { 1 } else { 0 })
        .arg_u32(sliding_window)"""
        self.assertGreaterEqual(self.host.count(host_sequence), 2)

    def test_br32_and_br64_are_bounded_to_the_requested_query_rows(self) -> None:
        for tile in ("BR", "BR64"):
            self.assertIn(
                f"const unsigned int q_start = query_start + q_block * {tile};",
                self.cuda,
            )
            self.assertIn(
                "if (q_start >= seq_len || "
                "q_start >= query_start + query_len_total) return;",
                self.cuda,
            )
            self.assertIn(
                f"const unsigned int q_end = min(q_start + {tile}, "
                "min(seq_len, query_start + query_len_total));",
                self.cuda,
            )
        self.assertEqual(
            self.cuda.count("q_start >= query_start + query_len_total"),
            2,
        )
        self.assertEqual(
            self.cuda.count("min(seq_len, query_start + query_len_total)"),
            2,
        )
        self.assertRegex(self.host, r"query_start\s*\.checked_add\(query_len\)")
        self.assertIn("is_some_and(|end| end <= seq_len)", self.host)

        def emitted_rows(
            seq_len: int,
            query_start: int,
            query_len: int,
            br: int,
        ) -> list[int]:
            rows = []
            # Include two deliberately over-provisioned blocks to exercise the
            # CUDA entry's explicit range-return condition as well as q_end.
            for q_block in range((query_len + br - 1) // br + 2):
                q_start = query_start + q_block * br
                if q_start >= seq_len or q_start >= query_start + query_len:
                    continue
                q_end = min(q_start + br, seq_len, query_start + query_len)
                rows.extend(range(q_start, q_end))
            return rows

        for br in (32, 64):
            for seq_len, query_start, query_len in (
                (1, 0, 1),
                (67, 0, 67),
                (100, 17, 1),
                (100, 17, 50),
                (526, 512, 14),
            ):
                self.assertEqual(
                    emitted_rows(seq_len, query_start, query_len, br),
                    list(range(query_start, query_start + query_len)),
                )

    def test_port_does_not_depend_on_dense_gate_include_macros(self) -> None:
        self.assertNotIn("ATLAS_PREFILL_64_KERNEL_NAME", self.cuda)
        self.assertNotIn("ATLAS_PREFILL_64_EXTRA_ARGS", self.cuda)
        self.assertNotIn("ATLAS_PREFILL_64_GATE_SETUP", self.cuda)
        self.assertNotIn("ATLAS_PREFILL_64_STORE_PAIR", self.cuda)

    @unittest.skipUnless(PTX is not None, "set ATLAS_PREFILL_ABI_PTX for PTX gate")
    def test_authoritative_flash_bundle_ptx_has_thirteen_parameters(self) -> None:
        assert PTX is not None
        self.assertTrue(PTX.is_file(), f"missing bundled PTX: {PTX}")
        self.assertEqual(PTX.name, "t0__inferspark_prefill.ptx")
        self.assertEqual(
            (PTX.parent / "t0__signature").read_text(),
            "gb10|qwen3.8-flash-next|nvfp4|sm_121f",
        )
        generated = (PTX.parent / "target_ptx.rs").read_text()
        self.assertIn(
            'pub const INFERSPARK_PREFILL_PTX: &str = include_str!('
            'concat!(env!("ATLAS_PTX_DIR"), "/t0__inferspark_prefill.ptx"));',
            generated,
        )

        ptx = PTX.read_text()
        for symbol in ("inferspark_prefill", "inferspark_prefill_64"):
            entry = ptx_entry(ptx, symbol)
            parameter_names = re.findall(
                rf"\b{re.escape(symbol)}_param_(\d+)\b",
                entry,
            )
            parameter_types = re.findall(r"\.param\s+(\.\w+)", entry)
            self.assertEqual(parameter_names, [str(index) for index in range(13)])
            self.assertEqual(parameter_types, EXPECTED_PTX_TYPES)
            for query_parameter in (5, 6):
                self.assertRegex(
                    ptx,
                    rf"ld\.param\.u32\s+%r\d+,\s+"
                    rf"\[{re.escape(symbol)}_param_{query_parameter}\];",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
