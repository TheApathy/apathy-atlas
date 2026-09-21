// SPDX-License-Identifier: AGPL-3.0-only
// Read-only final-encoder gate, not a teacher runner or model-quality benchmark.
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { PINS, validateManifests, validateStageManifest } from './contract.mjs';
import { compareBf16, passesGate, THRESHOLDS } from './metrics.mjs';
import { parseArgs, readBoundedFile, readPinnedFile, sha256, validateDirectory } from './io.mjs';

const MANIFEST_LIMIT = 1024 * 1024;
const INPUT_LIMIT = 3456 * 588 * 4;
const OUTPUT_LIMIT = 384 * 4096 * 2;

function document(file, expectedSha256) {
  const options = { maxBytes: MANIFEST_LIMIT };
  const bytes = expectedSha256 === undefined ? readBoundedFile(file, options)
    : readPinnedFile(file, { ...options, sha256: expectedSha256 });
  return { value: JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes)),
    sha256: sha256(bytes) };
}

function checkInputs(corpus, item) {
  const file = `${item.name}.input.f32`;
  const options = { sha256: item.input_sha256, maxBytes: INPUT_LIMIT,
    exactBytes: item.patches * 588 * 4 };
  const actual = readPinnedFile(join(corpus, file), options);
  const baseline = readPinnedFile(join(PINS.corpus, file), options);
  if (!actual.equals(baseline)) throw new Error('native input bytes differ from pinned baseline');
  for (let index = 0; index < actual.length / 4; index++) {
    const value = actual.readFloatLE(index * 4);
    const expected = ((index * 37 + Math.floor(index / 588) * 17) % 257 - 128) / 128;
    if (!Number.isFinite(value) || value !== expected) {
      throw new Error('input does not match the pinned deterministic dyadic corpus');
    }
  }
}

function checkObservedOutput(corpus, item, actualBytes) {
  const directory = join(corpus, `${item.name}-stages`);
  const metadata = document(join(directory, 'manifest.json'));
  const final = validateStageManifest(metadata.value, item);
  const observed = readPinnedFile(join(directory, final.file), {
    sha256: final.sha256, maxBytes: OUTPUT_LIMIT, exactBytes: actualBytes.length,
  });
  if (!observed.equals(actualBytes)) throw new Error('observed and normal final outputs differ');
  return { metadata_sha256: metadata.sha256, validated_entries: metadata.value.stages.length,
    declared_total_bytes: metadata.value.total_bytes, final_output_byte_equal: true,
    final_output_sha256: final.sha256,
    intermediate_payloads_rehashed: false };
}

export function compareCorpus({ corpus, manifestSha256 }) {
  validateDirectory(corpus);
  validateDirectory(PINS.corpus);
  const baseline = document(join(PINS.corpus, 'atlas-manifest.json'), PINS.baseline_sha256);
  const reference = document(join(PINS.corpus, 'reference-cuda-math.json'), PINS.reference_sha256);
  const actual = document(join(corpus, 'atlas-manifest.json'), manifestSha256);
  const cases = validateManifests(actual.value, baseline.value, reference.value);
  const results = [];
  for (const item of cases) {
    checkInputs(corpus, item);
    const extent = { maxBytes: OUTPUT_LIMIT, exactBytes: item.aligned_rows * 4096 * 2 };
    const candidate = readPinnedFile(join(corpus, `${item.name}.atlas.bf16`), {
      ...extent, sha256: item.output_sha256,
    });
    const teacher = readPinnedFile(join(PINS.corpus, `${item.name}.reference-cuda-math.bf16`), {
      ...extent, sha256: item.reference_sha256,
    });
    const observed = actual.value.diagnostic_stage_capture
      ? checkObservedOutput(corpus, item, candidate) : null;
    const metrics = compareBf16(candidate, teacher, item.aligned_rows, 4096);
    results.push({ name: item.name, grid: [item.grid_h, item.grid_w],
      shape: [item.aligned_rows, 4096], bytes: candidate.length,
      input_sha256: item.input_sha256, native_output_sha256: item.output_sha256,
      reference_output_sha256: item.reference_sha256,
      input_byte_equal_to_pinned_corpus: true, observed,
      encoder_with_upload_ms_reported: item.timings.map(timing => timing.encoder_with_upload_ms),
      metrics, passed: passesGate(metrics) });
  }
  const passed = results.every(item => item.passed);
  return {
    schema: 'atlas-deepseek-final-encoder-comparison-v1',
    status: passed ? 'PASS' : 'GATE_FAIL', passed,
    gate: 'same-input final encoder BF16 numerical comparison',
    corpus, actual_manifest_sha256: actual.sha256,
    baseline_manifest_sha256: baseline.sha256, reference_manifest_sha256: reference.sha256,
    model_revision: PINS.model_revision, config_sha256: PINS.config_sha256,
    index_sha256: PINS.index_sha256, official_code_sha256: PINS.official_code_sha256,
    visual_encoder_payload_sha256: PINS.visual_encoder_payload_sha256,
    producer_binary_sha256_reported: actual.value.binary_sha256,
    selected_tensors_reported: actual.value.checkpoint.selected_tensors,
    selected_bytes_reported: actual.value.checkpoint.selected_bytes,
    diagnostic_stage_capture: actual.value.diagnostic_stage_capture,
    thresholds: THRESHOLDS,
    evidence_limits: {
      repeat_equality: 'producer receipt only; second normal output is not retained',
      checkpoint: 'pinned header/download receipts; comparator does not rehash model weights',
      producer_binary: 'manifest receipt; binary is not rehashed by this comparator',
      reductions: 'FP64 equations with compensated sums; parallel Torch reductions may differ in last bits',
      timings: 'reported isolated encoder plus upload; not measured here and not LLM prefill',
      qualification: 'no decoder, image semantics, speculative-mode or performance qualification',
    },
    comparator_executed_gpu: false, generated_reference: false, cases: results,
  };
}

if (process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url) {
  try {
    const result = compareCorpus(parseArgs(process.argv.slice(2)));
    process.stdout.write(`${JSON.stringify(result)}\n`);
    process.exitCode = result.passed ? 0 : 2;
  } catch (error) {
    process.stdout.write(`${JSON.stringify({ schema: 'atlas-deepseek-final-encoder-comparison-v1',
      status: 'INVALID', passed: false, error: error instanceof Error ? error.message : String(error),
      comparator_executed_gpu: false, generated_reference: false })}\n`);
    process.exitCode = 1;
  }
}
