// SPDX-License-Identifier: AGPL-3.0-only
// Same equations as the pinned compare_reference.py:36-50; no teacher execution.
export const THRESHOLDS = Object.freeze({ cosine_min: 0.999, worst_row_min: 0.995,
  relative_l2_max: 0.05 });

const FIELDS = ['cosine', 'worst_row_cosine', 'relative_l2', 'max_abs_error', 'exact_fraction'];

// Compensated FP64 accumulation avoids a long serial reduction losing small
// terms. CUDA's parallel FP64 reduction can still differ in its last bits.
class Sum {
  value = 0;
  correction = 0;
  add(value) {
    const next = this.value + value;
    this.correction += Math.abs(this.value) >= Math.abs(value)
      ? (this.value - next) + value : (value - next) + this.value;
    this.value = next;
  }
  total() { return this.value + this.correction; }
}

export function compareBf16(actual, reference, rows, cols) {
  if (!Number.isSafeInteger(rows) || rows < 1 || rows > 384
    || !Number.isSafeInteger(cols) || cols < 1 || cols > 4096
    || !Buffer.isBuffer(actual) || !Buffer.isBuffer(reference)
    || actual.length !== rows * cols * 2 || reference.length !== actual.length) {
    throw new Error('invalid BF16 comparison geometry or exact byte extent');
  }
  const word = new DataView(new ArrayBuffer(4));
  const decode = (buffer, index) => {
    word.setUint32(0, buffer.readUInt16LE(index * 2) * 65536, true);
    const value = word.getFloat32(0, true);
    if (!Number.isFinite(value)) throw new Error('nonfinite BF16 comparison value');
    return value;
  };
  const a2 = new Sum(), b2 = new Sum(), dot = new Sum(), difference2 = new Sum();
  let worst = Infinity, maxError = 0, equal = 0;
  for (let row = 0; row < rows; row++) {
    const ra2 = new Sum(), rb2 = new Sum(), rdot = new Sum();
    for (let col = 0; col < cols; col++) {
      const index = row * cols + col;
      const a = decode(actual, index), b = decode(reference, index);
      ra2.add(a * a); rb2.add(b * b); rdot.add(a * b);
      const difference = a - b;
      difference2.add(difference * difference);
      maxError = Math.max(maxError, Math.abs(difference));
      if (a === b) equal++; // Numeric equality, including +0 versus -0, as Torch.
    }
    const aa = ra2.total(), bb = rb2.total(), ab = rdot.total();
    if (!(aa > 0) || !(bb > 0)) throw new Error('zero norm comparison row');
    worst = Math.min(worst, ab / (Math.sqrt(aa) * Math.sqrt(bb)));
    a2.add(aa); b2.add(bb); dot.add(ab);
  }
  const referenceNorm = Math.sqrt(b2.total());
  const result = {
    cosine: dot.total() / (Math.sqrt(a2.total()) * referenceNorm),
    worst_row_cosine: worst,
    relative_l2: Math.sqrt(difference2.total()) / referenceNorm,
    max_abs_error: maxError,
    exact_fraction: Math.fround(equal / (rows * cols)),
  };
  if (!FIELDS.every(key => Number.isFinite(result[key]))) {
    throw new Error('nonfinite FP64 comparison metric');
  }
  return result;
}

export function passesGate(metrics) {
  if (!metrics || !FIELDS.every(key => typeof metrics[key] === 'number'
    && Number.isFinite(metrics[key]))) return false;
  // A sqrt-product quotient may round one representable step above one.
  // This is a metric-domain guard, not a relaxation of any acceptance gate.
  if (Math.abs(metrics.cosine) > 1 + Number.EPSILON
    || Math.abs(metrics.worst_row_cosine) > 1 + Number.EPSILON
    || metrics.relative_l2 < 0 || metrics.max_abs_error < 0
    || metrics.exact_fraction < 0 || metrics.exact_fraction > 1) return false;
  return metrics.cosine >= THRESHOLDS.cosine_min
    && metrics.worst_row_cosine >= THRESHOLDS.worst_row_min
    && metrics.relative_l2 <= THRESHOLDS.relative_l2_max;
}
