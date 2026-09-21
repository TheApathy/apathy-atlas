# SPDX-License-Identifier: AGPL-3.0-only
"""Behavior and mutation-resistant source contracts for the fused candidate."""

from __future__ import annotations

import ast
import hashlib
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
PACKAGE = HERE / "frozen_triton_fused_nozero"
ADAPTER = PACKAGE / "adapter.py"
CASE = PACKAGE / "case.py"


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _method(tree: ast.Module, owner: str | None, name: str) -> ast.FunctionDef:
    scope: list[ast.stmt] = tree.body
    if owner is not None:
        classes = [
            node
            for node in scope
            if isinstance(node, ast.ClassDef) and node.name == owner
        ]
        if len(classes) != 1:
            raise AssertionError(f"expected one class {owner}")
        scope = classes[0].body
    found = [
        node
        for node in scope
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(found) != 1:
        raise AssertionError(f"expected one function {owner}.{name}")
    return found[0]


def _digest(nodes: list[ast.AST]) -> str:
    wire = "\n".join(ast.dump(node, include_attributes=False) for node in nodes)
    return hashlib.sha256(wire.encode("ascii")).hexdigest()


def _guard_raises(node: ast.FunctionDef, expression: str) -> bool:
    expected = ast.dump(
        ast.parse(expression, mode="eval").body, include_attributes=False
    )
    matches = []
    for item in ast.walk(node):
        if not isinstance(item, ast.If):
            continue
        current = ast.dump(item.test, include_attributes=False)
        if current == expected:
            matches.append(item)
    return len(matches) == 1 and any(
        isinstance(item, ast.Raise) for item in matches[0].body
    )


class FusedNozeroContractTests(unittest.TestCase):
    def test_raw_equal_is_flat_exact_storage_equality(self) -> None:
        from frozen_triton_fused_nozero import contract  # noqa: F401
        import torch

        from frozen_triton_fused_nozero.adapter import _raw_equal

        left = torch.arange(16, dtype=torch.float32).to(torch.bfloat16).view(2, 2, 4)
        right = left.clone().view(2, 8)
        self.assertTrue(_raw_equal(left, right))
        changed = right.clone()
        changed_bytes = changed.view(torch.uint8).reshape(-1)
        changed_bytes[-1] = int(changed_bytes[-1]) ^ 1
        self.assertFalse(_raw_equal(left, changed))
        self.assertFalse(_raw_equal(left, right[:, :-1]))

    def test_fused_abi_status_stream_and_phase_ast_are_exact(self) -> None:
        tree = _tree(ADAPTER)
        abi = [
            _method(tree, "FusedInputLibrary", name) for name in ("__init__", "launch")
        ]
        phase = [
            _method(tree, "FusedBuffers", name)
            for name in (
                "bind_stream",
                "reset_mutable",
                "adapter_qkv",
                "adapter_gate",
                "adapter_state_in",
                "_capture_exact_intermediates",
                "adapter_checks",
            )
        ]
        self.assertEqual(
            _digest(abi),
            "a1bd995484f0c9da77e63d66ca01b1018f43e1da227895659ad170d15d0cf2fe",
        )
        self.assertEqual(
            _digest(phase),
            "df898e6179a1762defea04e960198055eae6d49df6cf7f0f45ada2f71437427f",
        )
        source = ADAPTER.read_text()
        mutants = (
            source.replace("pair * 9", "pair * 8", 1),
            source.replace("ctypes.c_void_p(stream)", "ctypes.c_void_p(0)", 1),
            source.replace("if status != 0:", "if False and status != 0:", 1),
            source.replace(
                '            v["q_bf16"],\n            v["k_bf16"],',
                '            v["k_bf16"],\n            v["q_bf16"],',
                1,
            ),
            source.replace("self._adapter_phase = 1", "self._adapter_phase = 2", 1),
        )
        expected = _digest([*abi, *phase])
        for mutant in mutants:
            changed = ast.parse(mutant)
            changed_abi = [
                _method(changed, "FusedInputLibrary", name)
                for name in ("__init__", "launch")
            ]
            changed_phase = [
                _method(changed, "FusedBuffers", name)
                for name in (
                    "bind_stream",
                    "reset_mutable",
                    "adapter_qkv",
                    "adapter_gate",
                    "adapter_state_in",
                    "_capture_exact_intermediates",
                    "adapter_checks",
                )
            ]
            self.assertNotEqual(_digest([*changed_abi, *changed_phase]), expected)

    def test_exact_and_timing_aggregates_are_strict_and_mutation_bound(self) -> None:
        source = CASE.read_text()
        tree = ast.parse(source, filename=str(CASE))
        correctness = _method(tree, None, "correctness")
        timing = _method(tree, None, "timing")
        self.assertTrue(_guard_raises(correctness, "not exact"))
        self.assertTrue(_guard_raises(timing, "not all(predicates.values())"))
        self.assertEqual(
            _digest([correctness]),
            "6007802380133331338348bc836e6ce2e73fbed4cd439785b8adbd2aa2fa1dfe",
        )
        self.assertEqual(
            _digest([timing]),
            "3faad42f189dfcba4452ef1f1d391fc3d0259fe108c0678bb89cce0dd72421cd",
        )
        mutants = (
            source.replace("if not exact:", "if False and not exact:", 1),
            source.replace(
                "if not all(predicates.values()):",
                "if False and not all(predicates.values()):",
                1,
            ),
        )
        self.assertNotEqual(
            _digest([_method(ast.parse(mutants[0]), None, "correctness")]),
            _digest([correctness]),
        )
        self.assertNotEqual(
            _digest([_method(ast.parse(mutants[1]), None, "timing")]),
            _digest([timing]),
        )


if __name__ == "__main__":
    unittest.main()
