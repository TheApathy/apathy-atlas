// SPDX-License-Identifier: AGPL-3.0-only
// Synthetic metadata only. No checkpoint, reference or GPU execution.
export const HASH = 'a'.repeat(64);
export const CONFIG = '28a07138554196d7de70cfb193eb63bf51c39bb42ae4cd4303ba16610b5b1bf5';
export const INDEX = 'f4df075b9b9d77af5fe1482624a33466a7b5418f96c9d31f53339c34d72338d8';
export const REVISION = 'c171bea574201ff25530256fbd63626c7fd20f3c';
export function fixtures() {
  const checkpoint = { model_revision: REVISION, config_sha256: CONFIG, index_sha256: INDEX,
    selected_tensors: 267, selected_bytes: 932786176,
    complete_shards: Array.from({ length: 10 }, (_, index) => ({
      file: `model-${String(index + 1).padStart(5, '0')}-of-00010.safetensors`,
      bytes: 10000, header_sha256: HASH, download_etag: HASH,
    })), payload_integrity: 'download receipts; no full-shard rehash in this probe' };
  const cases = [[3, 3], [4, 5], [54, 54]].map(([h, w]) => ({
    name: `grid-${h}x${w}`, grid_h: h, grid_w: w, patches: h * w,
    aligned_rows: Math.ceil(h / 3) * Math.ceil(w / 3), hidden_size: 4096,
    input_sha256: HASH, output_sha256: HASH,
    stats: { count: Math.ceil(h / 3) * Math.ceil(w / 3) * 4096,
      max_abs: 1, mean: 1, sum_squares: 1 },
    timings: [{ repeat: 0, encoder_with_upload_ms: 1 },
      { repeat: 1, encoder_with_upload_ms: 2 }], repeat_byte_equal: true,
  }));
  const baseline = { checkpoint, binary_sha256: HASH, diagnostic_stage_capture: true,
    binary: '/synthetic/probe', scratch_bytes: 206275584, cases,
    timing_scope: 'isolated encoder plus input upload; not LLM prompt prefill',
    reference_parity: 'pending separate pinned official torch oracle' };
  const reference = {
    official_code_sha256: 'a4f089069310398d42ca17fd4496cec82da64cbbfde9b0230679ce1537cc0bb1',
    config_sha256: CONFIG, index_sha256: INDEX,
    visual_encoder_payload_sha256: 'fc9d54d790826fb7d8f60b06c8ae17a45b3cade7bfee44e7ae3f74dc948922b0',
    torch_version: '2.10.0+cu130', device: 'cuda', sdpa: 'math', attention_mode: 'official',
    diagnostic_only: false, bf16_reduced_precision_reduction: true,
    thresholds: { cosine_min: 0.999, worst_row_min: 0.995, relative_l2_max: 0.05 },
    cases: cases.map(({ name }) => ({ name, reference_sha256: HASH, reference_ms: 1,
      metrics: { cosine: 1, worst_row_cosine: 1, relative_l2: 0, max_abs_error: 0,
        exact_fraction: 1 }, passed: true })), passed: true,
  };
  return { actual: structuredClone(baseline), baseline, reference };
}
export function stageFixture(caseInfo) {
  const bytes = caseInfo.aligned_rows * caseInfo.hidden_size * 2;
  return { scope: 'diagnostic only; not a timing run', grid: [caseInfo.grid_h, caseInfo.grid_w],
    total_bytes: bytes, output_byte_equal: true, stages: [{ name: 'aligner-output',
      file: 'aligner-output.bf16', shape: [caseInfo.aligned_rows, 4096], dtype: 'bf16',
      bytes, sha256: caseInfo.output_sha256 }] };
}
