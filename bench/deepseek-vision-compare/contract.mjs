// SPDX-License-Identifier: AGPL-3.0-only
// Pure metadata admission; CLI authenticates the baseline and teacher documents.
import { THRESHOLDS } from './metrics.mjs';

export const PINS = Object.freeze({
  corpus: '/var/tmp/atlas-deepseek-vision-parity-v7-20260905T0440Z',
  baseline_sha256: 'caff41128f7e6266d3127e51d3a4aca6a0a6c9237559b08851e1a094c214e660',
  reference_sha256: '2a32cf81b2e22bea22d6cac362587bbc2a6463904e4310a1e160235f6ddc58ae',
  model_revision: 'c171bea574201ff25530256fbd63626c7fd20f3c',
  config_sha256: '28a07138554196d7de70cfb193eb63bf51c39bb42ae4cd4303ba16610b5b1bf5',
  index_sha256: 'f4df075b9b9d77af5fe1482624a33466a7b5418f96c9d31f53339c34d72338d8',
  official_code_sha256: 'a4f089069310398d42ca17fd4496cec82da64cbbfde9b0230679ce1537cc0bb1',
  visual_encoder_payload_sha256: 'fc9d54d790826fb7d8f60b06c8ae17a45b3cade7bfee44e7ae3f74dc948922b0',
});
const GRIDS = [[3, 3], [4, 5], [54, 54]];
const HASH = /^[0-9a-f]{64}$/;
const SCOPE = 'isolated encoder plus input upload; not LLM prompt prefill';
const RECEIPTS = 'download receipts; no full-shard rehash in this probe';
const CASE_KEYS = ['name', 'grid_h', 'grid_w', 'patches', 'aligned_rows', 'hidden_size',
  'input_sha256', 'output_sha256', 'stats', 'timings', 'repeat_byte_equal'];

function require(condition, message) {
  if (!condition) throw new Error(message);
}
function object(value) {
  require(value !== null && typeof value === 'object' && !Array.isArray(value)
    && [Object.prototype, null].includes(Object.getPrototypeOf(value)), 'expected plain object');
}
function keys(value, expected) {
  object(value);
  require(Object.keys(value).length === expected.length
    && expected.every(key => Object.hasOwn(value, key)), 'missing or extra metadata fields');
}
function selectorKeys(value, expected) {
  object(value);
  keys(value, Object.hasOwn(value, 'selected_detail_block')
    ? [...expected, 'selected_detail_block'] : expected);
}
function hash(value) {
  require(typeof value === 'string' && HASH.test(value), 'invalid SHA256');
}
function integer(value, maximum, minimum = 1) {
  require(Number.isSafeInteger(value) && value >= minimum && value <= maximum,
    'invalid bounded integer');
}
function finite(value) {
  require(typeof value === 'number' && Number.isFinite(value), 'nonfinite metadata number');
}
function checkpoint(value) {
  keys(value, ['model_revision', 'config_sha256', 'index_sha256', 'selected_tensors',
    'selected_bytes', 'complete_shards', 'payload_integrity']);
  for (const key of ['model_revision', 'config_sha256', 'index_sha256']) {
    require(value[key] === PINS[key], `incorrect checkpoint ${key}`);
  }
  require(value.selected_tensors === 267 && value.selected_bytes === 932786176,
    'incorrect visual tensor selection');
  require(value.payload_integrity === RECEIPTS, 'incorrect payload-integrity scope');
  require(Array.isArray(value.complete_shards) && value.complete_shards.length === 10,
    'incomplete checkpoint shard receipts');
  const shards = new Map();
  for (const shard of value.complete_shards) {
    keys(shard, ['file', 'bytes', 'header_sha256', 'download_etag']);
    require(/^model-000(?:0[1-9]|10)-of-00010\.safetensors$/.test(shard.file)
      && !shards.has(shard.file), 'invalid or duplicate shard');
    integer(shard.bytes, 32 * 1024 ** 3);
    hash(shard.header_sha256); hash(shard.download_etag);
    shards.set(shard.file, shard);
  }
  return shards;
}
function nativeCase(value, h, w) {
  keys(value, CASE_KEYS);
  require(value.name === `grid-${h}x${w}` && value.grid_h === h && value.grid_w === w
    && value.patches === h * w && value.aligned_rows === Math.ceil(h / 3) * Math.ceil(w / 3)
    && value.hidden_size === 4096, 'incorrect corpus case geometry');
  hash(value.input_sha256); hash(value.output_sha256);
  keys(value.stats, ['count', 'max_abs', 'mean', 'sum_squares']);
  require(value.stats.count === value.aligned_rows * 4096, 'incorrect native statistics extent');
  for (const key of ['max_abs', 'mean', 'sum_squares']) finite(value.stats[key]);
  require(value.stats.max_abs > 0 && value.stats.sum_squares > 0, 'invalid native statistics');
  require(value.repeat_byte_equal === true && Array.isArray(value.timings)
    && value.timings.length === 2, 'missing exact-repeat receipt');
  value.timings.forEach((timing, index) => {
    keys(timing, ['repeat', 'encoder_with_upload_ms']);
    finite(timing.encoder_with_upload_ms);
    require(timing.repeat === index && timing.encoder_with_upload_ms > 0,
      'invalid repeat or encoder timing');
  });
}
function nativeManifest(value) {
  selectorKeys(value, ['checkpoint', 'binary_sha256', 'diagnostic_stage_capture', 'binary',
    'scratch_bytes', 'cases', 'timing_scope', 'reference_parity']);
  const shards = checkpoint(value.checkpoint);
  hash(value.binary_sha256);
  require(typeof value.binary === 'string' && value.binary.startsWith('/')
    && value.binary.length <= 4096 && !value.binary.includes('\0'), 'invalid producer binary path');
  require(typeof value.diagnostic_stage_capture === 'boolean', 'missing stage-capture mode');
  const hasSelector = Object.hasOwn(value, 'selected_detail_block');
  if (hasSelector) {
    if (value.diagnostic_stage_capture) integer(value.selected_detail_block, 31, 0);
    else require(value.selected_detail_block === null, 'non-capture selector must be null');
  }
  require(value.scratch_bytes === 206275584, 'unexpected encoder arena contract');
  require(value.timing_scope === SCOPE
    && value.reference_parity === 'pending separate pinned official torch oracle',
  'incorrect encoder-only receipt scope');
  require(Array.isArray(value.cases) && value.cases.length === GRIDS.length,
    'missing or extra native corpus cases');
  const cases = new Map();
  for (const [h, w] of GRIDS) {
    const found = value.cases.filter(item => item?.name === `grid-${h}x${w}`);
    require(found.length === 1, 'missing or duplicate native corpus case');
    nativeCase(found[0], h, w);
    // Preserve legacy absence; only the admitted native top-level receipt may
    // supply selector identity to later stage checks, never a raw case field.
    cases.set(found[0].name, hasSelector
      ? { ...found[0], selected_detail_block: value.selected_detail_block } : found[0]);
  }
  return { shards, cases };
}
function referenceManifest(value) {
  keys(value, ['official_code_sha256', 'config_sha256', 'index_sha256',
    'visual_encoder_payload_sha256', 'torch_version', 'device', 'sdpa', 'attention_mode',
    'diagnostic_only', 'bf16_reduced_precision_reduction', 'thresholds', 'cases', 'passed']);
  for (const key of ['official_code_sha256', 'config_sha256', 'index_sha256',
    'visual_encoder_payload_sha256']) require(value[key] === PINS[key], `incorrect oracle ${key}`);
  require(value.torch_version === '2.10.0+cu130' && value.device === 'cuda'
    && value.sdpa === 'math' && value.attention_mode === 'official'
    && value.diagnostic_only === false && value.bf16_reduced_precision_reduction === true,
  'incorrect retained official oracle mode');
  keys(value.thresholds, Object.keys(THRESHOLDS));
  for (const key of Object.keys(THRESHOLDS)) {
    require(value.thresholds[key] === THRESHOLDS[key], 'changed acceptance thresholds');
  }
  require(typeof value.passed === 'boolean' && Array.isArray(value.cases)
    && value.cases.length === GRIDS.length, 'invalid reference cases');
  const cases = new Map();
  for (const [h, w] of GRIDS) {
    const found = value.cases.filter(item => item?.name === `grid-${h}x${w}`);
    require(found.length === 1, 'missing or duplicate reference case');
    const item = found[0];
    keys(item, ['name', 'reference_sha256', 'reference_ms', 'metrics', 'passed']);
    hash(item.reference_sha256); finite(item.reference_ms);
    require(item.reference_ms > 0 && typeof item.passed === 'boolean', 'invalid reference receipt');
    keys(item.metrics, ['cosine', 'worst_row_cosine', 'relative_l2', 'max_abs_error', 'exact_fraction']);
    Object.values(item.metrics).forEach(finite);
    cases.set(item.name, item); // Stored metrics/passed are provenance, never new evidence.
  }
  return cases;
}

export function validateManifests(actual, baseline, reference) {
  const source = nativeManifest(baseline), candidate = nativeManifest(actual);
  const teacher = referenceManifest(reference);
  for (const [name, shard] of candidate.shards) {
    const original = source.shards.get(name);
    for (const key of ['bytes', 'header_sha256', 'download_etag']) {
      require(shard[key] === original[key], 'checkpoint shard receipt differs from baseline');
    }
  }
  return [...candidate.cases.values()].map(item => {
    require(item.input_sha256 === source.cases.get(item.name).input_sha256,
      'native input differs from pinned reference input');
    return { ...item, reference_sha256: teacher.get(item.name).reference_sha256 };
  });
}

export function validateStageManifest(value, caseInfo) {
  selectorKeys(value, ['scope', 'grid', 'total_bytes', 'output_byte_equal', 'stages']);
  const hasSelector = Object.hasOwn(value, 'selected_detail_block');
  if (hasSelector) integer(value.selected_detail_block, 31, 0);
  if (Object.hasOwn(caseInfo, 'selected_detail_block')) {
    integer(caseInfo.selected_detail_block, 31, 0);
    require(hasSelector && value.selected_detail_block === caseInfo.selected_detail_block,
      'stage selector is missing or differs from native capture');
  }
  require(value.scope === 'diagnostic only; not a timing run' && value.output_byte_equal === true,
    'invalid observed-output receipt');
  require(Array.isArray(value.grid) && value.grid.length === 2
    && value.grid[0] === caseInfo.grid_h && value.grid[1] === caseInfo.grid_w,
  'stage grid does not match corpus');
  require(Array.isArray(value.stages) && value.stages.length > 0 && value.stages.length <= 64,
    'missing or excessive stage metadata');
  integer(value.total_bytes, 512 * 1024 * 1024);
  const names = new Set(), files = new Set();
  let total = 0, final;
  for (const stage of value.stages) {
    keys(stage, ['name', 'file', 'shape', 'dtype', 'bytes', 'sha256']);
    require(typeof stage.name === 'string' && /^[A-Za-z0-9-]{1,80}$/.test(stage.name)
      && !names.has(stage.name) && !files.has(stage.file), 'unsafe or duplicate stage name');
    require(['bf16', 'f32'].includes(stage.dtype)
      && stage.file === `${stage.name}.${stage.dtype}`, 'unsafe stage filename or dtype');
    require(Array.isArray(stage.shape) && stage.shape.length === 2, 'invalid stage rank');
    stage.shape.forEach(dimension => integer(dimension, 3456 * 16));
    integer(stage.bytes, 3456 * 3456 * 4);
    require(stage.bytes === stage.shape[0] * stage.shape[1] * (stage.dtype === 'bf16' ? 2 : 4),
      'stage byte extent does not match dtype/geometry');
    hash(stage.sha256);
    names.add(stage.name); files.add(stage.file); total += stage.bytes;
    require(total <= 512 * 1024 * 1024, 'cumulative stage metadata exceeds cap');
    if (stage.name === 'aligner-output') final = stage;
  }
  require(total === value.total_bytes && final !== undefined, 'incomplete stage metadata');
  require(final.dtype === 'bf16' && final.shape[0] === caseInfo.aligned_rows
    && final.shape[1] === 4096 && final.sha256 === caseInfo.output_sha256,
  'final observed stage differs from native final output');
  return final;
}
