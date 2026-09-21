// SPDX-License-Identifier: AGPL-3.0-only
import test from 'node:test';
import assert from 'node:assert/strict';
import { validateManifests, validateStageManifest } from './contract.mjs';
import { fixtures, stageFixture, HASH } from './fixtures.mjs';

test('three exact corpus cases are admitted independent of order', () => {
  const f = fixtures();
  f.actual.cases.reverse();
  const rows = validateManifests(f.actual, f.baseline, f.reference);
  assert.equal(rows.length, 3);
  assert.deepEqual(new Set(rows.map(c => c.name)),
    new Set(['grid-3x3', 'grid-4x5', 'grid-54x54']));
});

test('model/config/index/selected weights pins and source oracle contract are mandatory', () => {
  for (const key of ['model_revision', 'config_sha256', 'index_sha256',
    'selected_tensors', 'selected_bytes', 'complete_shards']) {
    const f = fixtures();
    f.actual.checkpoint[key] = null;
    assert.throws(() => validateManifests(f.actual, f.baseline, f.reference), key);
  }
  for (const key of ['config_sha256', 'index_sha256', 'official_code_sha256',
    'visual_encoder_payload_sha256', 'device', 'sdpa', 'attention_mode', 'thresholds']) {
    const f = fixtures();
    f.reference[key] = null;
    assert.throws(() => validateManifests(f.actual, f.baseline, f.reference), key);
  }
});

test('missing, duplicate, renamed, excessive, fractional and swapped geometry fail closed', () => {
  for (const mutate of [
    a => a.cases.pop(), a => a.cases.push(a.cases[0]),
    a => a.cases[1] = structuredClone(a.cases[0]),
    a => a.cases[0].name = '../grid-3x3',
    a => a.cases[0].grid_h = 3.5, a => a.cases[0].grid_w = 99999,
    a => a.cases[0].aligned_rows = 2, a => a.cases[0].hidden_size = 4095,
    a => a.cases[0].patches = 20, a => a.cases[0].input_sha256 = 'b'.repeat(64),
    a => a.cases[0].output_sha256 = 'not-a-hash',
  ]) {
    const f = fixtures(); mutate(f.actual);
    assert.throws(() => validateManifests(f.actual, f.baseline, f.reference));
  }
});

test('stage/repeat receipts must be explicit and non-timing metadata cannot promote a benchmark', () => {
  for (const mutate of [a => delete a.diagnostic_stage_capture,
    a => a.diagnostic_stage_capture = 'false', a => a.cases[0].repeat_byte_equal = false,
    a => a.cases[0].timings.pop(), a => a.cases[0].timings[1].repeat = 0,
    a => a.cases[0].timings[1].encoder_with_upload_ms = NaN,
    a => a.timing_scope = 'LLM prefill tok/s', a => a.binary_sha256 = null]) {
    const f = fixtures(); mutate(f.actual);
    assert.throws(() => validateManifests(f.actual, f.baseline, f.reference));
  }
  const f = fixtures(); f.actual.diagnostic_stage_capture = false;
  assert.equal(validateManifests(f.actual, f.baseline, f.reference).length, 3);
});

test('stage final payload has exact geometry/hash and bounded unique safe file entries', () => {
  const c = fixtures().actual.cases[0];
  assert.equal(validateStageManifest(stageFixture(c), c).sha256, HASH);
  for (const mutate of [s => s.output_byte_equal = false, s => s.grid[0] = 4,
    s => s.stages[0].sha256 = 'b'.repeat(64), s => s.stages[0].dtype = 'f32',
    s => s.stages[0].bytes++, s => s.stages[0].file = '../output.bf16',
    s => s.stages[0].shape = [385, 4096], s => s.total_bytes++,
    s => s.stages.push(structuredClone(s.stages[0])),
    s => s.stages = Array(65).fill(s.stages[0])]) {
    const stage = stageFixture(c); mutate(stage);
    assert.throws(() => validateStageManifest(stage, c));
  }
});
