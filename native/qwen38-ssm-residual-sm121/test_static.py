# SPDX-License-Identifier: AGPL-3.0-only
"""Body-scoped source authority for the private SSM residual C ABI."""

from __future__ import annotations

import re
import hashlib
import stat
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HEADER = ROOT / "include" / "atlas_qwen38_ssm_residual.h"
CUDA = ROOT / "src" / "atlas_qwen38_ssm_residual.cu"
ADMISSION = ROOT / "src" / "atlas_qwen38_ssm_residual_admission.cuh"
EXPORTS = ROOT / "exports.map"
EXPECTED_CUDA_SHA256 = "79b247013ecde27dbe56ebc7216164523a3da2388025e2a1e854e0a5a24bbd8e"
EXPECTED_ADMISSION_SHA256 = "82b694c86b5f2bbb8cfd73f0967bcd096a691bb6d451cce566fad100c4730d12"


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def body(source: str, signature: str) -> str:
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 0
    for index in range(brace, len(source)):
        depth += source[index] == "{"
        depth -= source[index] == "}"
        if depth == 0:
            return source[start : index + 1]
    raise AssertionError(f"unterminated function {signature}")


def exact_count(source: str, token: str, count: int = 1) -> bool:
    return compact(source).count(compact(token)) == count


def ordered(source: str, first: str, second: str) -> bool:
    compacted, left, right = compact(source), compact(first), compact(second)
    return left in compacted and right in compacted and compacted.index(left) < compacted.index(right)


def parameters(source: str, name: str) -> str:
    start = source.index(name) + len(name)
    opening = source.index("(", start)
    depth = 0
    for index in range(opening, len(source)):
        depth += source[index] == "("
        depth -= source[index] == ")"
        if depth == 0:
            return compact(source[opening + 1 : index])
    raise AssertionError(f"unterminated declaration {name}")


def source_contract(cuda: str, admission: str) -> list[str]:
    residual = body(cuda, 'extern "C" int atlas_qwen38_ssm_residual_bf16(')
    add = body(cuda, 'extern "C" int atlas_qwen38_ssm_add_bf16(')
    sync = body(cuda, 'extern "C" int atlas_qwen38_ssm_stream_synchronize(')
    pending = body(cuda, 'extern "C" unsigned long long atlas_qwen38_ssm_pending_receipt(')
    kernel = body(cuda, "__global__ void residual_kernel(")
    checks = {
        "exact CUDA source bytes": hashlib.sha256(cuda.encode()).hexdigest() == EXPECTED_CUDA_SHA256,
        "exact admission source bytes": hashlib.sha256(admission.encode()).hexdigest() == EXPECTED_ADMISSION_SHA256,
        "residual exact ABI lengths": all(exact_count(residual, name, 2) for name in ["residual_bf16_bytes", "input_bf16_bytes", "packed_e2m1_bytes", "physical_e4m3_scales_bytes"]),
        "add exact ABI lengths": all(exact_count(add, name, 2) for name in ["output_bf16_bytes", "first_bf16_bytes", "second_bf16_bytes"]),
        "exact allocation residual4": exact_count(residual, "exact_cuda_allocation(", 4),
        "exact allocation add3": exact_count(add, "exact_cuda_allocation(", 3),
        "residual disjoint polarity": exact_count(residual, "!pairwise_disjoint(spans, 4)"),
        "add disjoint polarity": exact_count(add, "!pairwise_disjoint(spans, 3)"),
        "finite positive scale": exact_count(residual, "!std::isfinite(scale2) || scale2 <= 0.0f"),
        "physical scale extent": exact_count(residual, "size_t scale_bytes = padded_rows * (cols / kGroupSize);"),
        "physical scale read": exact_count(kernel, "physical_scales[scale_offset_128x4(row, col / kGroupSize, groups)]"),
        "full residual write": exact_count(kernel, "residual[index] = __float2bfloat16_rn(original - dequantized)"),
        "prelaunch residual": exact_count(residual, "if (!prelaunch_status())"),
        "prelaunch add": exact_count(add, "if (!prelaunch_status())"),
        "postlaunch residual": exact_count(residual, "return publish_after_launch(cuda_stream, nonce)"),
        "postlaunch add": exact_count(add, "return publish_after_launch(cuda_stream, nonce)"),
        "nondefault residual": exact_count(residual, "stream == nullptr"),
        "nondefault add": exact_count(add, "stream == nullptr"),
        "sync same stream": exact_count(sync, "cudaStreamSynchronize(cuda_stream)"),
        "sync rejects default": exact_count(sync, "stream == nullptr"),
        "pending receipt only when active": exact_count(pending, "g_pending.active ? g_pending.nonce : 0"),
        "enqueue refuses pending residual": exact_count(residual, "g_pending.active"),
        "enqueue refuses pending add": exact_count(add, "g_pending.active"),
        "sync exact pending": all(exact_count(sync, token) for token in ["!g_pending.active", "g_pending.stream != cuda_stream", "g_pending.nonce != nonce"]),
        "sync consumes before effect": ordered(sync, "g_pending = {};", "cudaStreamSynchronize"),
        "device or managed": all(token in admission for token in ["cudaMemoryTypeDevice", "cudaMemoryTypeManaged"]),
        "exact base and bytes": all(token in admission for token in ["CU_POINTER_ATTRIBUTE_RANGE_START_ADDR", "CU_POINTER_ATTRIBUTE_RANGE_SIZE", "base != address", "allocation_bytes != claimed_bytes"]),
    }
    return [name for name, passed in checks.items() if not passed]


EXPECTED_ABI = {
    "atlas_qwen38_ssm_residual_bf16",
    "atlas_qwen38_ssm_add_bf16",
    "atlas_qwen38_ssm_stream_synchronize",
    "atlas_qwen38_ssm_pending_receipt",
    "atlas_qwen38_ssm_residual_last_error",
}
EXPECTED_PARAMETERS = {
    "atlas_qwen38_ssm_residual_bf16": compact(
        "void* residual_bf16, size_t residual_bf16_bytes, const void* input_bf16, size_t input_bf16_bytes, const unsigned char* packed_e2m1, size_t packed_e2m1_bytes, const unsigned char* physical_e4m3_scales, size_t physical_e4m3_scales_bytes, float scale2, int rows, int cols, void* stream"
    ),
    "atlas_qwen38_ssm_add_bf16": compact("void* output_bf16, size_t output_bf16_bytes, const void* first_bf16, size_t first_bf16_bytes, const void* second_bf16, size_t second_bf16_bytes, int rows, int cols, void* stream"),
    "atlas_qwen38_ssm_stream_synchronize": compact("void* stream, unsigned long long nonce"),
    "atlas_qwen38_ssm_pending_receipt": compact("void"),
    "atlas_qwen38_ssm_residual_last_error": compact("void"),
}


class SourceContractTests(unittest.TestCase):
    def test_files_spdx_modes_and_caps(self) -> None:
        for path in [HEADER, CUDA, ADMISSION, EXPORTS, ROOT / "build.sh", ROOT / "test-static.sh", ROOT / "build_provenance.py", ROOT / "test_build_provenance.py", Path(__file__)]:
            text = path.read_text()
            self.assertLessEqual(len(text.splitlines()), 250, path)
            self.assertIn("SPDX-License-Identifier: AGPL-3.0-only", "\n".join(text.splitlines()[:2]))
        for path in [ROOT / "build.sh", ROOT / "test-static.sh", ROOT / "build_provenance.py"]:
            self.assertTrue(path.stat().st_mode & stat.S_IXUSR, path)

    def test_header_and_export_abi_are_exact(self) -> None:
        header = HEADER.read_text()
        exports = EXPORTS.read_text()
        declared = set(re.findall(r"ATLAS_Q38_API (?:int|unsigned long long|const char\*)\s+(atlas_qwen38_[a-z0-9_]+)\(", header))
        exported = set(re.findall(r"^\s+(atlas_qwen38_[a-z0-9_]+);$", exports, re.M))
        self.assertEqual(declared, EXPECTED_ABI)
        self.assertEqual(exported, EXPECTED_ABI)
        cuda = CUDA.read_text()
        for name, expected in EXPECTED_PARAMETERS.items():
            self.assertEqual(parameters(header, name), expected)
            self.assertEqual(parameters(cuda, name), expected)
        self.assertEqual(header.count("atlas_qwen38_ssm_stream_synchronize("), 1)

    def test_active_bodies_bind_numeric_allocation_and_status_contract(self) -> None:
        self.assertEqual(source_contract(CUDA.read_text(), ADMISSION.read_text()), [])

    def test_all_seven_reported_mutants_are_rejected(self) -> None:
        cuda = CUDA.read_text()
        admission = ADMISSION.read_text()
        mutants = [
            (cuda, "physical_scales[scale_offset_128x4(row, col / kGroupSize, groups)]", "physical_scales[row * groups + col / kGroupSize]", admission),
            (cuda, "!pairwise_disjoint(spans, 4)", "pairwise_disjoint(spans, 4)", admission),
            (cuda, "return publish_after_launch(cuda_stream, nonce);", "return ATLAS_Q38_OK;", admission),
            (cuda, "residual[index] = __float2bfloat16_rn", "residual[0] = __float2bfloat16_rn", admission),
            (cuda, "padded_rows * (cols / kGroupSize)", "padded_rows * (cols / kGroupSize) / 2", admission),
            (cuda, "!std::isfinite(scale2) || scale2 <= 0.0f", "false", admission),
        ]
        for source, old, new, adm in mutants:
            self.assertIn(old, source)
            self.assertTrue(source_contract(source.replace(old, new, 1), adm), old)
        header = HEADER.read_text().replace("ATLAS_Q38_API const char* atlas_qwen38_ssm_residual_last_error(void);", "", 1)
        declared = set(re.findall(r"ATLAS_Q38_API (?:int|unsigned long long|const char\*)\s+(atlas_qwen38_[a-z0-9_]+)\(", header))
        self.assertNotEqual(declared, EXPECTED_ABI)

    def test_additional_extent_domain_stream_and_sync_mutants_reject(self) -> None:
        cuda, admission = CUDA.read_text(), ADMISSION.read_text()
        for source, old, new, adm in [
            (cuda, "exact_cuda_allocation(", "make_span(", admission),
            (cuda, "stream == nullptr", "false", admission),
            (cuda, "return publish_after_launch(cuda_stream, nonce);", "return ATLAS_Q38_OK;", admission),
            (admission, "cudaMemoryTypeManaged", "cudaMemoryTypeHost", admission),
            (admission, "allocation_bytes != claimed_bytes", "allocation_bytes < claimed_bytes", admission),
        ]:
            self.assertIn(old, source)
            candidate_cuda = source.replace(old, new, 1) if source is cuda else cuda
            candidate_admission = source.replace(old, new, 1) if source is admission else adm
            self.assertTrue(source_contract(candidate_cuda, candidate_admission), old)

    def test_receipt_idle_cross_stream_replay_failure_and_thread_hostiles_reject(self) -> None:
        cuda, admission = CUDA.read_text(), ADMISSION.read_text()
        for old, new in [
            ("!g_pending.active", "false"),
            ("g_pending.stream != cuda_stream", "false"),
            ("g_pending.nonce != nonce", "false"),
            ("g_pending = {};", "/* receipt remains replayable */"),
            ("thread_local PendingReceipt g_pending", "PendingReceipt g_pending"),
            ("return publish_after_launch(cuda_stream, nonce);", "g_pending = {cuda_stream, nonce, true}; return launch_status();"),
        ]:
            self.assertIn(old, cuda)
            self.assertTrue(source_contract(cuda.replace(old, new, 1), admission), old)


if __name__ == "__main__":
    unittest.main()
