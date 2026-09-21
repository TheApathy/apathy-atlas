// SPDX-License-Identifier: AGPL-3.0-only
// Actual comparator admission; synthetic metadata only, no GPU or payload I/O.
import test from 'node:test';
import assert from 'node:assert/strict';
import { validateManifests, validateStageManifest } from './contract.mjs';
import { fixtures, stageFixture, HASH } from './fixtures.mjs';

const KEY = 'selected_detail_block';
const INVALID = [-1, 32, 8.5, Number.MAX_SAFE_INTEGER, Infinity, -Infinity, NaN,
  undefined, '8', true, false, [], {}, new Number(8)];

function admitted(f) {
  return validateManifests(f.actual, f.baseline, f.reference);
}

test('legacy run and stage manifests remain selector-free and byte-unmodified', () => {
  for (const capture of [false, true]) {
    const f = fixtures();
    f.actual.diagnostic_stage_capture = capture;
    const before = structuredClone(f);
    const rows = admitted(f);
    assert.equal(rows.length, 3);
    assert.ok(rows.every(row => !Object.hasOwn(row, KEY)));
    assert.deepEqual(f, before);
    if (capture) {
      const stage = stageFixture(rows[0]);
      assert.equal(validateStageManifest(stage, rows[0]).sha256, HASH);
      assert.ok(!Object.hasOwn(stage, KEY));
    }
  }
});

test('non-capture native manifest admits only explicit null and carries its identity', () => {
  const f = fixtures();
  f.actual.diagnostic_stage_capture = false;
  f.actual[KEY] = null;
  const before = structuredClone(f);
  const rows = admitted(f);
  assert.equal(rows.length, 3);
  assert.ok(rows.every(row => Object.hasOwn(row, KEY) && row[KEY] === null));
  assert.deepEqual(f, before);
});

test('captured native selectors 0, 8 and 31 propagate without relabeling legacy baseline', () => {
  for (const block of [0, 8, 31]) {
    const f = fixtures();
    f.actual[KEY] = block;
    const before = structuredClone(f);
    const rows = admitted(f);
    assert.equal(rows.length, 3);
    assert.ok(rows.every(row => Object.hasOwn(row, KEY) && row[KEY] === block));
    assert.ok(rows.every(row => row.reference_sha256 === HASH));
    assert.ok(!Object.hasOwn(f.baseline, KEY));
    assert.deepEqual(f, before);
  }
});

test('non-capture native rejects every non-null selector without coercion', () => {
  for (const block of [0, 8, 31, ...INVALID]) {
    const f = fixtures();
    f.actual.diagnostic_stage_capture = false;
    f.actual[KEY] = block;
    assert.throws(() => admitted(f), `accepted run selector ${String(block)}`);
  }
});

test('captured native rejects null, unsafe, fractional and non-number selectors', () => {
  for (const block of [null, ...INVALID]) {
    const f = fixtures();
    f.actual[KEY] = block;
    assert.throws(() => admitted(f), `accepted stage selector ${String(block)}`);
  }
});

test('optional stage selector accepts bounded integers with a legacy case receipt', () => {
  const c = fixtures().actual.cases[1];
  for (const block of [0, 8, 31]) {
    const stage = stageFixture(c);
    stage[KEY] = block;
    const before = structuredClone(stage);
    assert.equal(validateStageManifest(stage, c).sha256, HASH);
    assert.deepEqual(stage, before);
    assert.ok(!Object.hasOwn(c, KEY));
  }
});

test('optional stage selector rejects null and every invalid integer representation', () => {
  const c = fixtures().actual.cases[1];
  for (const block of [null, ...INVALID]) {
    const stage = stageFixture(c);
    stage[KEY] = block;
    assert.throws(() => validateStageManifest(stage, c));
  }
});

test('native selector is cross-bound to every corresponding stage manifest', () => {
  for (const block of [0, 8, 31]) {
    const f = fixtures();
    f.actual[KEY] = block;
    for (const c of admitted(f)) {
      const stage = stageFixture(c);
      stage[KEY] = block;
      assert.equal(validateStageManifest(stage, c).sha256, HASH);
    }
  }
});

test('an explicit native selector rejects missing or different stage selectors', () => {
  for (const block of [0, 8, 31]) {
    const c = { ...fixtures().actual.cases[1], [KEY]: block };
    assert.throws(() => validateStageManifest(stageFixture(c), c),
      'explicit native selector silently downgraded to a legacy stage');
    for (const other of [0, 8, 31].filter(value => value !== block)) {
      const stage = stageFixture(c);
      stage[KEY] = other;
      assert.throws(() => validateStageManifest(stage, c), 'selector mismatch admitted');
    }
  }
});

test('stage comparison rejects explicitly null or invalid case selectors', () => {
  for (const block of [null, ...INVALID]) {
    const c = { ...fixtures().actual.cases[1], [KEY]: block };
    assert.throws(() => validateStageManifest(stageFixture(c), c));
    const stage = stageFixture(c);
    stage[KEY] = block;
    assert.throws(() => validateStageManifest(stage, c));
  }
});

test('selector admission retains complete corpus, shape, pins and fixed thresholds', () => {
  const mutations = [
    f => f.actual.cases.pop(),
    f => f.actual.cases.push(structuredClone(f.actual.cases[0])),
    f => f.actual.cases[1] = structuredClone(f.actual.cases[0]),
    f => f.actual.cases[0].hidden_size = 4095,
    f => f.actual.cases[0].input_sha256 = 'b'.repeat(64),
    f => f.actual.cases[0][KEY] = 8,
    f => f.actual.checkpoint.config_sha256 = 'b'.repeat(64),
    f => f.actual.checkpoint.index_sha256 = 'b'.repeat(64),
    f => f.actual.checkpoint.model_revision = 'b'.repeat(40),
    f => f.reference.official_code_sha256 = 'b'.repeat(64),
    f => f.reference.thresholds.cosine_min = 0.998,
    f => f.reference.thresholds.worst_row_min = 0.994,
    f => f.reference.thresholds.relative_l2_max = 0.06,
    f => f.actual.unrecognized = 8,
  ];
  for (const mutate of mutations) {
    const f = fixtures();
    f.actual[KEY] = 8;
    assert.equal(admitted(f).length, 3);
    mutate(f);
    assert.throws(() => admitted(f));
  }
});

test('stage selector cannot bypass final hash, extent, output receipt or stage bounds', () => {
  for (const mutate of [
    s => s.stages[0].sha256 = 'b'.repeat(64),
    s => s.stages[0].shape[0]++,
    s => s.total_bytes++,
    s => s.output_byte_equal = false,
    s => s.stages.push(structuredClone(s.stages[0])),
    s => s.stages = Array(65).fill(s.stages[0]),
    s => s.unrecognized = 8,
  ]) {
    const c = { ...fixtures().actual.cases[1], [KEY]: 8 };
    const stage = stageFixture(c);
    stage[KEY] = 8;
    assert.equal(validateStageManifest(stage, c).sha256, HASH);
    mutate(stage);
    assert.throws(() => validateStageManifest(stage, c));
  }
});
