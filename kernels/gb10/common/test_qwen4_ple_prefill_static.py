# SPDX-License-Identifier: AGPL-3.0-only

import math
import pathlib
import struct
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[3]
RUST = ROOT / "crates/spark-model/src/layers/qwen4_ple.rs"
PREFILL_A = ROOT / "crates/spark-model/src/model/trait_impl/prefill_a.rs"
PREFILL_B = ROOT / "crates/spark-model/src/model/trait_impl/prefill_b/forward_layers.rs"
BUILD = ROOT / "crates/spark-model/src/factory/build.rs"
DROP = ROOT / "crates/spark-model/src/model/drop.rs"
CUDA = ROOT / "kernels/gb10/common/qwen4_hyper.cu"


def serial_history(values: list[int], entry: list[int], reset: bool):
    state = [0] * 9 if reset else list(entry)
    taps = []
    for value in values:
        taps.append((state[0], state[3], state[6], value))
        state = state[1:] + [value]
    return taps, state


def parallel_history(values: list[int], entry: list[int], reset: bool):
    state = [0] * 9 if reset else list(entry)

    def prior(token: int, dilation: int):
        return (
            values[token - dilation]
            if token >= dilation
            else state[token + 9 - dilation]
        )

    taps = [
        (prior(t, 9), prior(t, 6), prior(t, 3), value) for t, value in enumerate(values)
    ]
    return taps, (state + values)[-9:]


def bf16(value: float) -> float:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    bits = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def silu(value: float) -> float:
    return value / (1.0 + math.exp(-value))


def ple_terms(query, key, value, norm_query, norm_key, norm_conv, eps):
    width = len(query)

    def normalized(vector, weight):
        inv = 1.0 / math.sqrt(sum(item * item for item in vector) / width + eps)
        return [item * inv * (1.0 + scale) for item, scale in zip(vector, weight)]

    q_norm = normalized(query, norm_query)
    k_norm = normalized(key, norm_key)
    dot = sum(q * k for q, k in zip(q_norm, k_norm)) / math.sqrt(width)
    root = math.copysign(math.sqrt(max(abs(dot), 1.0e-6)), dot)
    gate = 1.0 / (1.0 + math.exp(-root))
    gated = [gate * item for item in value]
    conv_inv = 1.0 / math.sqrt(sum(item * item for item in gated) / width + eps)
    conv = [item * conv_inv * (1.0 + scale) for item, scale in zip(gated, norm_conv)]
    return gated, conv


def serial_arithmetic(inputs, norms, weights, entry, reset):
    state = [[0.0] * 9 for _ in entry] if reset else [list(row) for row in entry]
    output = []
    for query, key, value in inputs:
        gated, conv = ple_terms(query, key, value, *norms)
        row = []
        for h, current in enumerate(conv):
            old = state[h]
            kernel = (
                old[0] * weights[h][0]
                + old[3] * weights[h][1]
                + old[6] * weights[h][2]
                + current * weights[h][3]
            )
            row.append(bf16(query[h] + gated[h] + silu(kernel)))
            state[h] = old[1:] + [bf16(current)]
        output.append(row)
    return output, state


def parallel_arithmetic(inputs, norms, weights, entry, reset):
    terms = [ple_terms(query, key, value, *norms) for query, key, value in inputs]
    output = []
    for token, ((query, _, _), (gated, conv)) in enumerate(zip(inputs, terms)):
        row = []
        for h, current in enumerate(conv):

            def prior(dilation):
                if token >= dilation:
                    return bf16(terms[token - dilation][1][h])
                return 0.0 if reset else entry[h][token + 9 - dilation]

            kernel = (
                prior(9) * weights[h][0]
                + prior(6) * weights[h][1]
                + prior(3) * weights[h][2]
                + current * weights[h][3]
            )
            row.append(bf16(query[h] + gated[h] + silu(kernel)))
        output.append(row)
    initial = [[0.0] * 9 for _ in entry] if reset else entry
    state = [
        (list(initial[h]) + [bf16(item[1][h]) for item in terms])[-9:]
        for h in range(len(entry))
    ]
    return output, state


def require_fragments(source: str, fragments: tuple[str, ...]):
    missing = [fragment for fragment in fragments if source.count(fragment) != 1]
    if missing:
        raise AssertionError(f"missing/non-unique source contracts: {missing}")


class PlePrefillStaticTests(unittest.TestCase):
    def test_parallel_history_matches_serial(self):
        entry = [100 + index for index in range(9)]
        for length in (1, 3, 8, 9, 17, 2013):
            values = [1000 + index for index in range(length)]
            for reset in (False, True):
                self.assertEqual(
                    parallel_history(values, entry, reset),
                    serial_history(values, entry, reset),
                )

    def test_full_arithmetic_matches_serial_with_bf16_history_only(self):
        width = 4
        inputs = []
        for token in range(17):
            inputs.append(
                tuple(
                    [bf16((token + 1) * factor + h * 0.03125) for h in range(width)]
                    for factor in (0.071, -0.053, 0.097)
                )
            )
        norms = (
            [bf16(0.01 * (h + 1)) for h in range(width)],
            [bf16(-0.02 * (h + 1)) for h in range(width)],
            [bf16(0.03 * (h + 1)) for h in range(width)],
            1.0e-6,
        )
        weights = [
            [bf16(0.11 + 0.01 * h), bf16(-0.17), bf16(0.23), bf16(0.563)]
            for h in range(width)
        ]
        entry = [
            [bf16(-0.4 + 0.013 * (h * 9 + index)) for index in range(9)]
            for h in range(width)
        ]
        for reset in (False, True):
            self.assertEqual(
                parallel_arithmetic(inputs, norms, weights, entry, reset),
                serial_arithmetic(inputs, norms, weights, entry, reset),
            )

    def test_legacy_early_bf16_rounding_has_exact_counterexample(self):
        serial = bf16(-2.0 - 1.881 + silu(-1.195 * 0.563))
        legacy = bf16(-2.0 + bf16(-1.881) + silu(bf16(-1.195) * 0.563))
        self.assertEqual(serial, -4.09375)
        self.assertEqual(legacy, -4.125)
        self.assertNotEqual(serial, legacy)

    def test_literal_million_token_residual_index_requires_u64(self):
        last = 1_000_000 * 10_240 - 1
        self.assertGreater(last, 2**32 - 1)
        self.assertLess(last, 2**64)

    def test_cuda_contract_is_complete(self):
        source = CUDA.read_text()
        require_fragments(
            source,
            (
                "void qwen4_ple_dequant_prefill(",
                "void qwen4_ple_prepare_prefill(",
                "void qwen4_ple_conv_inject_prefill(",
                "void qwen4_ple_commit_prefill_state(",
                "state[token]",
                "state[token + 3u]",
                "state[token + 6u]",
                "old[num_tokens + j]",
            ),
        )
        self.assertLess(
            source.index("void qwen4_ple_prepare_prefill("),
            source.index("void qwen4_ple_conv_inject_prefill("),
        )
        self.assertLess(
            source.index("void qwen4_ple_conv_inject_prefill("),
            source.index("void qwen4_ple_commit_prefill_state("),
        )
        self.assertEqual(source.count("qwen4_ple_history_bf16("), 4)
        self.assertEqual(source.count("const unsigned long long residual_width"), 2)
        self.assertIn("float* __restrict__ gated_value", source)
        self.assertIn("const float* __restrict__ gated_value", source)
        self.assertIn("const float* __restrict__ conv_input", source)
        self.assertIn("const float x = conv_input[out];", source)
        self.assertIn("__bfloat162float(hyper[out]) + gated_value[out] + c", source)
        self.assertIn(
            "next = __float2bfloat16(\n"
            "                conv_input[(unsigned long long)source_token * residual_width + i]);",
            source,
        )

    def test_rust_path_is_default_off_and_bounded(self):
        source = RUST.read_text()
        require_fragments(
            source,
            (
                "const PLE_IO_BATCH_ROWS: usize = 32;",
                "pub fn forward_prefill(",
                "num_tokens <= scratch.max_tokens",
                "request.chunks(PLE_IO_BATCH_ROWS)",
                '"qwen4_ple_dequant_prefill"',
                '"qwen4_ple_prepare_prefill"',
                '"qwen4_ple_conv_inject_prefill"',
                '"qwen4_ple_commit_prefill_state"',
                '"QWEN4_PREFILL_ENGAGED"',
                'projection = "cublaslt_bf16_non_bit_exact"',
                "projection_parity_required = true",
            ),
        )
        self.assertEqual(source.count("performance_claim_allowed = false"), 2)
        self.assertEqual(source.count('"performance_claim_allowed": false'), 1)
        self.assertEqual(source.count("ops::cublas_bf16_proj_dense("), 2)
        load_flag = source.index('std::env::var("ATLAS_QWEN4_PLE_PREFILL_BATCH")')
        allocation = source.index(
            "self.prefill = Some(Qwen4PlePrefillScratch::allocate("
        )
        self.assertLess(load_flag, allocation)
        execution = source.index(
            "let execution =", source.index("pub fn forward_prefill(")
        )
        sync = source.index("gpu.synchronize(stream)", execution)
        prefill = source.index("pub fn forward_prefill(")
        receipt = source.index('"QWEN4_PREFILL_ENGAGED"', prefill)
        fence = source.index("gpu.record_event(self.scratch_event, stream)", prefill)
        self.assertLess(fence, receipt)
        self.assertLess(sync, receipt)
        self.assertIn("scratch_poisoned.store(true, Ordering::Release)", source)
        self.assertIn("Qwen4PlePrefillScratch::allocate(", source)
        self.assertIn("scratch.release(gpu)", source)
        teardown = source.index("pub fn destroy_owned_resources(")
        poisoned = source.index(
            "!self.scratch_poisoned.load(Ordering::Acquire)", teardown
        )
        release = source.index("scratch.release(gpu)", teardown)
        self.assertLess(
            source.index("gpu.stream_wait_event(teardown_stream", teardown),
            release,
        )
        self.assertLess(poisoned, release)

    def test_drop_owns_prefill_scratch_release(self):
        source = DROP.read_text()
        self.assertIn("ple.destroy_owned_resources(self.gpu.as_ref())", source)
        self.assertLess(
            source.index("synchronize(default_stream)"),
            source.index("destroy_owned_resources"),
        )

    def test_both_prefill_callers_retain_serial_fallback(self):
        for path in (PREFILL_A, PREFILL_B):
            source = path.read_text()
            self.assertEqual(source.count("ATLAS_QWEN4_PLE_PREFILL_BATCH"), 1)
            self.assertEqual(source.count("ple.forward_prefill("), 1)
            self.assertEqual(source.count("ple.forward_token("), 1)
            self.assertLess(
                source.index("ple.forward_prefill("), source.index("ple.forward_token(")
            )

    def test_factory_binds_workspace_to_configured_arena(self):
        source = BUILD.read_text()
        call = source[source.index("Qwen4PleLayer::load(") :]
        call = call[: call.index(")?;")]
        self.assertIn("max_batch_size", call)
        init = "model.initialize_qwen4_ple_prefill(max_batch_tokens)?;"
        self.assertIn(init, source)
        self.assertGreater(source.index(init), source.index("TransformerModel::new("))

    def test_capture_publication_guards_first_admission_failures(self):
        source = RUST.read_text()
        contracts = (
            "struct OwnedCreatedFile {",
            "let mut created = OwnedCreatedFile::new(path.clone());",
            "created.file = Some(",
            "let opened = admit(created.file.as_ref()",
            "impl Drop for OwnedCreatedFile {",
            "drop(self.file.take());",
            'fs::metadata(format!("/proc/self/fd/{}", file.as_raw_fd()))',
            "struct OwnedPleParityFrame {",
            "fn random_private_suffix() -> Result<String>",
            "fn rename_noreplace(source: &Path, destination: &Path)",
            "fn private_staging_path(final_path: &Path)",
            "let staging_path = private_staging_path(&final_path)?;",
            "let mut owned = OwnedPleParityFrame::new(staging_path.clone(), final_path);",
            "builder.create(&staging_path).with_context",
            "owned.handle = Some(",
            "owned.identity = Some((handle_metadata.dev(), handle_metadata.ino()));",
            "fn create_owned_frame_with_open_and_admission<O, A>(",
            "fn handle_is_owned(&self) -> bool",
            'let admitted = admit(&staging_path).context("stat new PLE parity staging frame")?;',
            "fn publish_noreplace(&mut self) -> Result<()>",
            "rename_noreplace(&self.path, &published_path)?;",
            "owned_frame.publish_noreplace().context(",
            "impl Drop for OwnedPleParityFrame {",
            "first_artifact_metadata_failure_is_absent_and_retryable",
            "frame_admission_failures_are_absent_and_retryable",
            "post_mkdir_open_failure_is_absent_and_retryable",
            "foreign_replacements_survive_cleanup_guards",
        )
        require_fragments(source, contracts)
        self.assertLess(source.index(contracts[1]), source.index(contracts[2]))
        self.assertLess(source.index(contracts[10]), source.index(contracts[16]))
        self.assertLess(source.index(contracts[16]), source.index(contracts[11]))
        self.assertLess(source.index(contracts[11]), source.index(contracts[12]))
        self.assertLess(source.index(contracts[12]), source.index(contracts[13]))
        self.assertLess(source.index(contracts[13]), source.index(contracts[14]))
        self.assertLess(source.index(contracts[14]), source.index(contracts[15]))
        self.assertLess(source.index(contracts[15]), source.index(contracts[18]))
        self.assertLess(source.index(contracts[19]), source.index(contracts[20]))
        for fragment in contracts:
            with self.assertRaises(AssertionError):
                require_fragments(
                    source.replace(fragment, "BROKEN_GUARD", 1), contracts
                )

    def test_source_contract_mutations_fail(self):
        source = CUDA.read_text()
        critical = (
            "state[token + 3u]",
            "state[token + 6u]",
            "old[num_tokens + j]",
        )
        require_fragments(source, critical)
        for fragment in critical:
            mutated = source.replace(fragment, "BROKEN_HISTORY_EXPRESSION", 1)
            with self.assertRaises(AssertionError):
                require_fragments(mutated, critical)

        arithmetic = (
            "    float* __restrict__ gated_value,",
            "    const float* __restrict__ gated_value,",
            "qwen4_ple_history_bf16(float value)",
            "const unsigned long long out = (unsigned long long)token * residual_width + i;",
            "const float x = conv_input[out];",
            "__bfloat162float(hyper[out]) + gated_value[out] + c",
            "conv_input[(unsigned long long)source_token * residual_width + i]",
        )
        require_fragments(source, arithmetic)
        for fragment in arithmetic:
            mutated = source.replace(fragment, "BROKEN_ARITHMETIC_CONTRACT", 1)
            with self.assertRaises(AssertionError):
                require_fragments(mutated, arithmetic)


if __name__ == "__main__":
    unittest.main()
