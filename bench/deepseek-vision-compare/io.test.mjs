// SPDX-License-Identifier: AGPL-3.0-only
import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync, symlinkSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createHash } from 'node:crypto';
import { readPinnedFile, parseArgs } from './io.mjs';

test('bounded reads validate complete payload hash and extent; never follow symlinks', () => {
  const directory = mkdtempSync(join(tmpdir(), 'atlas-vision-compare-test-'));
  try {
    const file = join(directory, 'tensor.bf16');
    const data = Buffer.from([0x80, 0x3f, 0, 0x40]);
    const hash = createHash('sha256').update(data).digest('hex');
    writeFileSync(file, data, { flag: 'wx' });
    const options = { sha256: hash, maxBytes: 4, exactBytes: 4 };
    assert.deepEqual(readPinnedFile(file, options), data);
    assert.throws(() => readPinnedFile(file, { ...options, sha256: '0'.repeat(64) }));
    assert.throws(() => readPinnedFile(file, { ...options, maxBytes: 3 }));
    assert.throws(() => readPinnedFile(file, { ...options, exactBytes: 2 }));
    const link = join(directory, 'link.bf16');
    symlinkSync(file, link);
    assert.throws(() => readPinnedFile(link, options));
    assert.throws(() => readPinnedFile(directory, options));
    assert.throws(() => readPinnedFile(file, { ...options, maxBytes: Infinity }));
  } finally {
    rmSync(directory, { recursive: true });
  }
});

test('CLI requires explicit absolute corpus and SHA with no tolerance or duplicate knobs', () => {
  const good = ['--corpus', '/explicit/new/probe', '--manifest-sha256', 'a'.repeat(64)];
  assert.deepEqual(parseArgs(good), { corpus: '/explicit/new/probe', manifestSha256: 'a'.repeat(64) });
  for (const args of [[], good.slice(0, 2), [...good, '--tolerance', '1'],
    [...good, '--corpus', '/other'], ['--corpus', '../relative', '--manifest-sha256', 'a'.repeat(64)],
    ['--corpus', '/explicit/new/probe', '--manifest-sha256', 'not-sha']]) {
    assert.throws(() => parseArgs(args));
  }
});
