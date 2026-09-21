// SPDX-License-Identifier: AGPL-3.0-only
// The sole filesystem boundary: no writes, network, model imports or payload discovery.
import { constants, openSync, closeSync, readSync, fstatSync, lstatSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { isAbsolute, normalize, dirname, parse, join } from 'node:path';

const MAX_READ_BYTES = 64 * 1024 * 1024;
const SHA = /^[0-9a-f]{64}$/;
export const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');

function absolutePath(path) {
  if (typeof path !== 'string' || path.length > 4096 || path.includes('\0')
    || !isAbsolute(path) || normalize(path) !== path) {
    throw new Error('path must be explicit, normalized and absolute');
  }
  return path;
}

function directoryChain(path) {
  absolutePath(path);
  const root = parse(path).root;
  const parts = path.slice(root.length).split('/').filter(Boolean);
  if (parts.length > 64) throw new Error('directory depth exceeds bound');
  let current = root;
  const result = [];
  for (const part of ['', ...parts]) {
    if (part) current = join(current, part);
    const stat = lstatSync(current, { bigint: true });
    if (!stat.isDirectory() || stat.isSymbolicLink()) {
      throw new Error('non-directory or symbolic-link path component');
    }
    result.push({ path: current, stat });
  }
  return result;
}

function sameIdentity(a, b) {
  return a.dev === b.dev && a.ino === b.ino && a.mode === b.mode;
}

function unchangedFile(a, b) {
  return sameIdentity(a, b) && a.size === b.size && a.mtimeNs === b.mtimeNs
    && a.ctimeNs === b.ctimeNs && a.nlink === b.nlink;
}

export function validateDirectory(path) {
  directoryChain(path);
  return path;
}

export function readBoundedFile(file, { maxBytes, exactBytes } = {}) {
  absolutePath(file);
  if (!Number.isSafeInteger(maxBytes) || maxBytes < 1 || maxBytes > MAX_READ_BYTES
    || (exactBytes !== undefined && (!Number.isSafeInteger(exactBytes)
      || exactBytes < 1 || exactBytes > maxBytes))) {
    throw new Error('invalid bounded read extent');
  }
  const parents = directoryChain(dirname(file));
  const before = lstatSync(file, { bigint: true });
  if (!before.isFile() || before.isSymbolicLink() || before.size < 1n
    || before.size > BigInt(maxBytes)
    || (exactBytes !== undefined && before.size !== BigInt(exactBytes))) {
    throw new Error('non-regular file or incorrect bounded extent');
  }
  const fd = openSync(file, constants.O_RDONLY | constants.O_NOFOLLOW | constants.O_NONBLOCK);
  try {
    const opened = fstatSync(fd, { bigint: true });
    if (!opened.isFile() || !unchangedFile(before, opened)) {
      throw new Error('file identity changed before read');
    }
    const bytes = Buffer.alloc(Number(opened.size));
    let offset = 0;
    while (offset < bytes.length) {
      const count = readSync(fd, bytes, offset, bytes.length - offset, offset);
      if (count === 0) throw new Error('truncated file during read');
      offset += count;
    }
    if (readSync(fd, Buffer.alloc(1), 0, 1, bytes.length) !== 0
      || !unchangedFile(opened, fstatSync(fd, { bigint: true }))
      || !unchangedFile(opened, lstatSync(file, { bigint: true }))) {
      throw new Error('file changed during read');
    }
    for (const parent of parents) {
      if (!sameIdentity(parent.stat, lstatSync(parent.path, { bigint: true }))) {
        throw new Error('parent directory identity changed during read');
      }
    }
    return bytes;
  } finally {
    closeSync(fd);
  }
}

export function readPinnedFile(file, options = {}) {
  if (typeof options.sha256 !== 'string' || !SHA.test(options.sha256)) {
    throw new Error('complete lowercase SHA256 pin is required');
  }
  const bytes = readBoundedFile(file, options);
  if (sha256(bytes) !== options.sha256) throw new Error('complete file SHA256 mismatch');
  return bytes;
}

export function parseArgs(args) {
  if (!Array.isArray(args) || args.length !== 4) {
    throw new Error('usage: main.mjs --corpus /ABS/PROBE --manifest-sha256 SHA256');
  }
  const values = new Map();
  for (let index = 0; index < args.length; index += 2) {
    const key = args[index], value = args[index + 1];
    if (!['--corpus', '--manifest-sha256'].includes(key) || values.has(key)
      || typeof value !== 'string') throw new Error('unknown or duplicate argument');
    values.set(key, value);
  }
  const corpus = absolutePath(values.get('--corpus'));
  const manifestSha256 = values.get('--manifest-sha256');
  if (!SHA.test(manifestSha256)) throw new Error('invalid manifest SHA256');
  return { corpus, manifestSha256 };
}
