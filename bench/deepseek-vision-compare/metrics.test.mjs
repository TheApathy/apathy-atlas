// SPDX-License-Identifier: AGPL-3.0-only
import test from 'node:test';
import assert from 'node:assert/strict';
import { compareBf16, passesGate, THRESHOLDS } from './metrics.mjs';

function bf16(values) {
  const result = Buffer.alloc(values.length * 2);
  const word = Buffer.alloc(4);
  values.forEach((value, index) => {
    word.writeFloatLE(value);
    assert.equal(word.readUInt16LE(0), 0, 'test fixture must be exact BF16');
    result.writeUInt16LE(word.readUInt16LE(2), index * 2);
  });
  return result;
}
const near = (actual, expected) => assert.ok(Math.abs(actual - expected) <= 1e-14,
  `${actual} differs from ${expected}`);

test('identity is exact and comparison does not mutate either operand', () => {
  const a = bf16([3, 4, 0, 12]);
  const saved = Buffer.from(a);
  const result = compareBf16(a, a, 2, 2);
  assert.deepEqual(result, { cosine: 1, worst_row_cosine: 1, relative_l2: 0,
    max_abs_error: 0, exact_fraction: 1 });
  assert.ok(passesGate(result));
  assert.deepEqual(a, saved);
});

test('hand-computable FP64 norms and asymmetric denominator bind operand order', () => {
  const a = bf16([6, 8]);
  const b = bf16([0, 5]);
  const forward = compareBf16(a, b, 1, 2);
  const reverse = compareBf16(b, a, 1, 2);
  near(forward.cosine, 0.8);
  near(forward.worst_row_cosine, 0.8);
  near(forward.relative_l2, Math.sqrt(45) / 5);
  near(reverse.relative_l2, Math.sqrt(45) / 10);
  assert.equal(forward.max_abs_error, 6);
  assert.equal(forward.exact_fraction, 0);
  assert.equal(passesGate(forward), false);
});

test('worst row and torch FP32 exact-fraction reduction are preserved', () => {
  const result = compareBf16(bf16([1, 0, 0, 1]), bf16([1, 0, 0, -1]), 2, 2);
  assert.equal(result.cosine, 0);
  assert.equal(result.worst_row_cosine, -1);
  assert.equal(result.exact_fraction, 0.75);
  const thirds = compareBf16(bf16([1, 2, 3]), bf16([1, 4, 6]), 1, 3);
  assert.equal(thirds.exact_fraction, Math.fround(1 / 3));
});

test('nonfinite BF16 values and any zero-norm row are invalid', () => {
  for (const word of [0x7f80, 0xff80, 0x7fc1]) {
    const bad = Buffer.alloc(2);
    bad.writeUInt16LE(word);
    assert.throws(() => compareBf16(bad, bf16([1]), 1, 1));
    assert.throws(() => compareBf16(bf16([1]), bad, 1, 1));
  }
  assert.throws(() => compareBf16(bf16([0, 0]), bf16([1, 1]), 1, 2));
  assert.throws(() => compareBf16(bf16([1, 0]), bf16([1, 1]), 2, 1));
  assert.throws(() => compareBf16(bf16([1, 1]), bf16([1, 0]), 2, 1));
});

test('shape, exact byte extent and element caps reject before iteration', () => {
  const a = bf16([1, 2]);
  for (const [rows, cols] of [[0, 2], [1.5, 2], [385, 4096], [1, 4097],
    [Number.MAX_SAFE_INTEGER, 2], [true, 2]]) {
    assert.throws(() => compareBf16(a, a, rows, cols));
  }
  assert.throws(() => compareBf16(a, a.subarray(0, 2), 1, 2));
  assert.throws(() => compareBf16(Buffer.from([0]), Buffer.from([0]), 1, 1));
  assert.throws(() => compareBf16([1, 2], [1, 2], 1, 2));
});

test('unchanged gates are conjunctive, inclusive and never accept invalid metrics', () => {
  assert.deepEqual(THRESHOLDS, { cosine_min: 0.999, worst_row_min: 0.995,
    relative_l2_max: 0.05 });
  assert.ok(Object.isFrozen(THRESHOLDS));
  const edge = { cosine: 0.999, worst_row_cosine: 0.995, relative_l2: 0.05,
    max_abs_error: 0.01, exact_fraction: 0.5 };
  assert.equal(passesGate(edge), true);
  for (const [key, value] of [['cosine', 0.998999], ['worst_row_cosine', 0.994999],
    ['relative_l2', 0.050001], ['cosine', NaN], ['relative_l2', -1]]) {
    assert.equal(passesGate({ ...edge, [key]: value }), false);
  }
});
