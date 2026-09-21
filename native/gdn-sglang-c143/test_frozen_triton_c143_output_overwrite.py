# SPDX-License-Identifier: AGPL-3.0-only
"""Hostile CPU tests for the frozen Triton output full-write proof."""

from __future__ import annotations

import dataclasses
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import frozen_triton_c143_output_overwrite as proof


UPSTREAM = Path(
    "/tmp/sglang-c14312a66420b75ca9a11bf1817c4db1fa26b097/python/sglang/kernels/ops/attention/fla/chunk_o.py"
)
ARTIFACT_DIR = Path(
    "/home/flocka/.cache/sglang/triton/KTFBEMQNV7CPTR5VWT4OH2W5U4AWJVLJ2QV3CEDL433KBK2JBWNA"
)


class OutputOverwriteProofTests(unittest.TestCase):
    def test_exact_artifacts_cover_real_shapes(self) -> None:
        for tokens, grid_y in ((2079, 33), (8192, 128)):
            result = proof.prove(UPSTREAM, ARTIFACT_DIR, tokens)
            self.assertEqual(result["grid_x"], 2)
            self.assertEqual(result["grid_y"], grid_y)
            self.assertEqual(result["grid_z"], 48)
            self.assertEqual(result["elements"], tokens * 48 * 128)

    def test_every_artifact_hash_is_fail_closed(self) -> None:
        names = ("source", "ttir", "ptx", "cubin")
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            upstream = root / "chunk_o.py"
            shutil.copyfile(UPSTREAM, upstream)
            for name in names:
                shutil.copyfile(
                    ARTIFACT_DIR / f"chunk_fwd_kernel_o.{name}",
                    root / f"chunk_fwd_kernel_o.{name}",
                )
            for name in ("upstream", *names):
                target = (
                    upstream
                    if name == "upstream"
                    else root / f"chunk_fwd_kernel_o.{name}"
                )
                original = target.read_bytes()
                target.write_bytes(original + b"\n# hostile drift\n")
                with self.assertRaisesRegex(proof.ProofError, f"{name} hash drift"):
                    proof.prove(upstream, root, 2079)
                target.write_bytes(original)

    def test_stable_read_rejects_same_inode_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            target = Path(raw) / "artifact"
            target.write_bytes(b"before")

            def mutate() -> None:
                target.write_bytes(b"after!")

            with self.assertRaisesRegex(proof.ProofError, "drift"):
                proof._stable_read(target, _test_after_read=mutate)

    def test_stable_read_rejects_path_swap(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            target = Path(raw) / "artifact"
            displaced = Path(raw) / "displaced"
            target.write_bytes(b"before")

            def swap() -> None:
                os.rename(target, displaced)
                target.write_bytes(b"before")

            with self.assertRaisesRegex(proof.ProofError, "drift"):
                proof._stable_read(target, _test_after_read=swap)

    def test_stable_read_rejects_symlink_and_oversize(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            target = root / "artifact"
            target.write_bytes(b"ok")
            link = root / "link"
            link.symlink_to(target)
            with self.assertRaisesRegex(proof.ProofError, "regular file"):
                proof._stable_read(link)
            target.write_bytes(b"x" * (proof.MAX_ARTIFACT_BYTES + 1))
            with self.assertRaisesRegex(proof.ProofError, "size cap"):
                proof._stable_read(target)

    def test_geometry_drift_is_rejected(self) -> None:
        fields = {
            "batch": 2,
            "heads": 47,
            "grouped_heads": 8,
            "key_dim": 64,
            "value_dim": 192,
            "block_tokens": 32,
            "block_key": 64,
            "block_value": 32,
            "varlen": False,
        }
        for field, value in fields.items():
            with self.subTest(field=field):
                geometry = dataclasses.replace(proof.EXACT_GEOMETRY, **{field: value})
                with self.assertRaisesRegex(proof.ProofError, "geometry drift"):
                    proof._validate_geometry(geometry, 2079)

    def test_nonpositive_tokens_are_rejected(self) -> None:
        for tokens in (0, -1):
            with self.assertRaisesRegex(proof.ProofError, "positive"):
                proof._validate_geometry(proof.EXACT_GEOMETRY, tokens)

    def test_upstream_output_read_and_atomic_are_rejected(self) -> None:
        source = UPSTREAM.read_text()
        with self.assertRaisesRegex(proof.ProofError, "output read"):
            proof._validate_upstream(source + "\nt = tl.load(p_o)\n")
        with self.assertRaisesRegex(proof.ProofError, "atomic"):
            proof._validate_upstream(source + "\ntl.atomic_add(p_o, 1)\n")

    def test_ttir_output_read_extra_store_and_atomic_are_rejected(self) -> None:
        text = (ARTIFACT_DIR / "chunk_fwd_kernel_o.ttir").read_text()
        with self.assertRaisesRegex(proof.ProofError, "output read"):
            proof._validate_ttir(
                text.replace("tt.return", "%x = tt.load %15\n    tt.return")
            )
        with self.assertRaisesRegex(proof.ProofError, "store count"):
            proof._validate_ttir(
                text.replace("tt.return", "tt.store %15, %13, %b_v_128\n    tt.return")
            )
        with self.assertRaisesRegex(proof.ProofError, "atomic"):
            proof._validate_ttir(text + "\natomic_rmw\n")

    def test_ptx_output_read_store_width_and_atomic_are_rejected(self) -> None:
        text = (ARTIFACT_DIR / "chunk_fwd_kernel_o.ptx").read_text()
        with self.assertRaisesRegex(proof.ProofError, "output-address load"):
            proof._validate_ptx(
                text.replace("ret;", "ld.global.b32 %r1, [%rd68];\n\tret;")
            )
        with self.assertRaisesRegex(proof.ProofError, "store width"):
            proof._validate_ptx(text.replace("st.global.v4.b32", "st.global.b32", 1))
        with self.assertRaisesRegex(proof.ProofError, "atomic"):
            proof._validate_ptx(text + "\natom.global.add.u32 %r1, [%rd68], 1;\n")


if __name__ == "__main__":
    unittest.main()
