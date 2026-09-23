// SPDX-License-Identifier: AGPL-3.0-only

//! Typed launches of the V4.1 indexer kernels (`kernels/gb10/deepseek-v4.1/attn/sparse_index.cu`).
//!
//! Three device steps on an index-source layer, in this order:
//!
//! 1. [`IndexOps::score`] — `score[t, n] = sum_h relu(bf16(q[t,h,:] . ik[n,:])) * w[t,h]`,
//!    `-inf` past `compress_lens[t]` and off the candidate mask.
//! 2. [`IndexOps::select_candidates`] — layer 20 only, from the score it just produced.
//! 3. [`IndexOps::topk`] — the ascending top `min(512, n_c)` finite columns, padded to
//!    EXACTLY 512 with -1.
//!
//! `compress_lens` is never passed as a buffer: it is `(pos0 + t + 1) / ratio` with ABSOLUTE
//! positions, computed in the kernel from `pos0`. A chunk-relative `pos0` is the bug this
//! shape prevents (the oracle check `compress_lens(abs pos)` fails on it).

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

/// Kernel module compiled from `sparse_index.cu`.
pub const INDEX_MODULE: &str = "dsv41_sparse_index";

/// `index_n_heads`, `index_head_dim`: compiled into the kernel.
pub const INDEX_HEADS: usize = 32;
pub const INDEX_HEAD_DIM: usize = 128;
/// `index_topk`: the output width, a numerics contract (see `seams::INDEX_TOPK`).
pub const INDEX_TOPK: usize = 512;
/// Score columns are padded to a multiple of this (the reference's `KEY_BLOCK`).
pub const KEY_BLOCK: usize = 512;
const SCORE_KEYS_PER_BLOCK: usize = 128;
const SCORE_THREADS: u32 = 128;
const SELECT_THREADS: u32 = 512;
/// [`IndexOps::gemm_f32`] takes `dsv41_gemv_f32_nt` at M <= this (bit-identical, N/8 CTAs not N/64).
pub const GEMV_F32_MAX_M: usize = 16;
const GEMV_F32_KC: usize = 128;

/// A/B switch for the small-M fp32 GEMV (`ATLAS_DSV41_GEMV_F32=0` at core load, or a gate):
/// `true` forces `dsv41_gemm_f32_nt` at every M.
pub static GEMV_F32_OFF: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// A/B switch for the prefill-M fp32 GEMM (`ATLAS_DSV41_GEMM_F32_V1=1` at core load, or a gate):
/// `true` forces the original 64x64 `dsv41_gemm_f32_nt` above the GEMV's M.
pub static GEMM_F32_V1: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Score width for `n_c` visible compressed rows: `max(512, ceil(n_c / 512) * 512)`.
///
/// This is `n_pad` in `engine/model.py::_indexer`. It is also the candidate mask's width,
/// which is why layers 24..36 must be scored at the SAME `n_pad` as layer 20 in a pass.
pub fn score_width(n_c: usize) -> usize {
    n_c.div_ceil(KEY_BLOCK).max(1) * KEY_BLOCK
}

/// How many columns the top-k may take: `min(index_topk, n_c)`.
pub fn topk_width(n_c: usize) -> usize {
    INDEX_TOPK.min(n_c)
}

#[derive(Clone, Copy, Debug)]
pub struct IndexKernels {
    pub score: KernelHandle,
    pub topk: KernelHandle,
    pub candidates: KernelHandle,
    pub combine2: KernelHandle,
    pub wts: KernelHandle,
    pub iota: KernelHandle,
    pub gemm_f32: KernelHandle,
    pub gemm_bf16_smalln: KernelHandle,
    pub gemv_f32: KernelHandle,
    pub gemm_f32_v2: KernelHandle,
    /// Shape-static decode entries: the pass start is read from device memory.
    pub score_dev: KernelHandle,
    pub topk_dev: KernelHandle,
    pub candidates_dev: KernelHandle,
    pub iota_dev: KernelHandle,
    pub comp2_pending_in: KernelHandle,
    pub combine2_dev: KernelHandle,
    pub comp2_pending_out: KernelHandle,
    pub publish_rows: KernelHandle,
}

impl IndexKernels {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        let k = |name: &str| {
            gpu.kernel(INDEX_MODULE, name).with_context(|| {
                format!(
                    "{INDEX_MODULE}::{name} is not in the compiled PTX. It is built only for \
                     the (gb10, deepseek-v4.1, cb3) target; a NEW .cu file may also need \
                     `touch crates/atlas-kernels/build.rs` before cargo notices it."
                )
            })
        };
        Ok(Self {
            score: k("dsv41_index_score")?,
            topk: k("dsv41_index_topk")?,
            candidates: k("dsv41_select_candidates")?,
            combine2: k("dsv41_compress_combine2")?,
            wts: k("dsv41_index_wts")?,
            iota: k("dsv41_iota_i32")?,
            gemm_f32: k("dsv41_gemm_f32_nt")?,
            gemm_bf16_smalln: k("dsv41_gemm_bf16_smalln")?,
            gemv_f32: k("dsv41_gemv_f32_nt")?,
            gemm_f32_v2: k("dsv41_gemm_f32_nt_v2")?,
            score_dev: k("dsv41_index_score_dev")?,
            topk_dev: k("dsv41_index_topk_dev")?,
            candidates_dev: k("dsv41_select_candidates_dev")?,
            iota_dev: k("dsv41_iota_dev")?,
            comp2_pending_in: k("dsv41_comp2_pending_in")?,
            combine2_dev: k("dsv41_compress_combine2_dev")?,
            comp2_pending_out: k("dsv41_comp2_pending_out")?,
            publish_rows: k("dsv41_publish_rows")?,
        })
    }
}

/// Bytes of one candidate-mask row of `ld` score columns: one BIT per 8-column block (the
/// selection keeps whole blocks), packed in u32 words: `[rows, ld / 256]` u32.
pub fn cand_row_bytes(ld: usize) -> usize {
    debug_assert!(ld % 256 == 0, "candidate width {ld} not a multiple of 256");
    ld / 64
}

/// Host: a per-column u8 mask `[rows, ld]` (a capture's `cand_in` / `cand_out`) -> the device bit
/// layout. Errors if a block is not uniform (then the bit form could not represent it).
pub fn pack_cand(cols: &[u8], rows: usize, ld: usize) -> Result<Vec<u8>> {
    ensure!(cols.len() == rows * ld && ld % 256 == 0, "pack_cand: {} bytes for [{rows}, {ld}]", cols.len());
    let mut words = vec![0u32; rows * ld / 256];
    for r in 0..rows {
        for b in 0..ld / 8 {
            let blk = &cols[r * ld + b * 8..r * ld + b * 8 + 8];
            ensure!(blk.iter().all(|&v| (v != 0) == (blk[0] != 0)), "row {r} block {b}: mask not block-uniform");
            if blk[0] != 0 {
                words[r * ld / 256 + b / 32] |= 1 << (b % 32);
            }
        }
    }
    Ok(words.iter().flat_map(|w| w.to_le_bytes()).collect())
}

/// Host: the device bit layout `[rows, ld / 256]` u32 -> a per-column u8 mask `[rows, ld]`.
pub fn unpack_cand(bits: &[u8], rows: usize, ld: usize) -> Vec<u8> {
    (0..rows * ld)
        .map(|i| {
            let (r, c) = (i / ld, i % ld);
            let w = r * ld / 256 + c / 256;
            let word = u32::from_le_bytes(bits[w * 4..w * 4 + 4].try_into().unwrap());
            u8::from((word >> ((c / 8) % 32)) & 1 == 1)
        })
        .collect()
}

/// Rows the index scratch (fp32 score + candidate block scores) holds at once: the largest
/// multiple of 16 whose scratch fits [`SCORE_BUDGET`] at `max_seq`, within [16, max_chunk]. The
/// indexer walks a chunk in blocks of this many rows (per-row score, candidates and top-k, so
/// the result is the same bytes). 1M context: 224 rows; <= ~200K: the whole chunk.
pub const SCORE_BUDGET: usize = 1 << 30;
pub fn score_rows_per_block(max_chunk: usize, max_seq: usize) -> usize {
    let w = score_width(max_seq);
    let per_row = w * 4 + w.div_ceil(8) * 4;
    (SCORE_BUDGET / per_row / 16 * 16).clamp(16, max_chunk.max(16))
}

/// Candidate mask from layer 20: [`cand_row_bytes`]`(ld)` bytes per token row, bit = keep block.
#[derive(Clone, Copy, Debug)]
pub struct CandidateMask {
    pub mask: DevicePtr,
    pub ld: usize,
}

pub struct IndexOps<'a> {
    pub gpu: &'a dyn GpuBackend,
    pub k: &'a IndexKernels,
    pub stream: u64,
}

impl IndexOps<'_> {
    /// `q` `[t, 32, 128]` bf16 (RoPE'd with freqs_c), `ik` `[n_keys, 128]` bf16, `w` `[t, 32]`
    /// fp32 (already scaled by `128^-0.5 * 32^-0.5`), `out` `[t, n_pad]` fp32.
    #[allow(clippy::too_many_arguments)]
    pub fn score(
        &self,
        q: DevicePtr,
        ik: DevicePtr,
        w: DevicePtr,
        cand: Option<CandidateMask>,
        out: DevicePtr,
        t: usize,
        n_keys: usize,
        n_pad: usize,
        pos0: usize,
        ratio: usize,
    ) -> Result<()> {
        ensure!(n_pad % SCORE_KEYS_PER_BLOCK == 0, "n_pad {n_pad} not a multiple of {SCORE_KEYS_PER_BLOCK}");
        ensure!(ratio >= 1, "the indexer never runs on a ratio-0 layer");
        if t == 0 {
            return Ok(());
        }
        let (cand_ptr, cand_ld) = match cand {
            Some(c) => (c.mask, c.ld),
            None => (DevicePtr(0), 0),
        };
        KernelLaunch::new(self.gpu, self.k.score)
            .grid([t as u32, (n_pad / SCORE_KEYS_PER_BLOCK) as u32, 1])
            .block([SCORE_THREADS, 1, 1])
            .arg_ptr(q).arg_ptr(ik).arg_ptr(w).arg_ptr(cand_ptr).arg_ptr(out)
            .arg_i32(n_keys as i32).arg_i32(n_pad as i32).arg_u64(pos0 as u64)
            .arg_i32(ratio as i32).arg_i32(cand_ld as i32)
            .launch(self.stream)
    }

    /// `score` `[t, n_pad]` fp32 -> `out` `[t, 512]` i64.
    pub fn topk(&self, score: DevicePtr, out: DevicePtr, t: usize, n_pad: usize, n_c: usize) -> Result<()> {
        if t == 0 {
            return Ok(());
        }
        KernelLaunch::new(self.gpu, self.k.topk)
            .grid([t as u32, 1, 1])
            .block([SELECT_THREADS, 1, 1])
            .arg_ptr(score).arg_ptr(out)
            .arg_i32(n_pad as i32).arg_i32(topk_width(n_c) as i32)
            .launch(self.stream)
    }

    /// Layer 20 only. `block_scratch` holds `t * n_pad / block_size` fp32; `cand` is
    /// `[t, n_pad]` u8.
    #[allow(clippy::too_many_arguments)]
    pub fn select_candidates(
        &self,
        score: DevicePtr,
        block_scratch: DevicePtr,
        cand: DevicePtr,
        t: usize,
        n_pad: usize,
        pos0: usize,
        ratio: usize,
        topk_blocks: usize,
        block_size: usize,
    ) -> Result<()> {
        if t == 0 {
            return Ok(());
        }
        KernelLaunch::new(self.gpu, self.k.candidates)
            .grid([t as u32, 1, 1])
            .block([SELECT_THREADS, 1, 1])
            .arg_ptr(score).arg_ptr(block_scratch).arg_ptr(cand)
            .arg_i32(n_pad as i32).arg_u64(pos0 as u64).arg_i32(ratio as i32)
            .arg_i32(topk_blocks as i32).arg_i32(block_size as i32)
            .launch(self.stream)
    }
}

impl IndexOps<'_> {
    fn grid_1d(n: usize) -> Result<u32> {
        u32::try_from(n.div_ceil(256)).context("1-D grid overflow")
    }

    /// Ratio-2 gated combine: `kv`, `sc` `[2 * n_pairs, d]` fp32 -> `out` `[n_pairs, d]` bf16.
    pub fn combine2(&self, kv: DevicePtr, sc: DevicePtr, out: DevicePtr, n_pairs: usize, d: usize) -> Result<()> {
        if n_pairs == 0 {
            return Ok(());
        }
        KernelLaunch::new(self.gpu, self.k.combine2)
            .grid([Self::grid_1d(n_pairs * d)?, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(kv).arg_ptr(sc).arg_ptr(out).arg_i32(n_pairs as i32).arg_i32(d as i32)
            .launch(self.stream)
    }

    /// `out[i] = float(raw[i]) * scale`, `n` values.
    pub fn wts(&self, raw: DevicePtr, out: DevicePtr, scale: f32, n: usize) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        KernelLaunch::new(self.gpu, self.k.wts)
            .grid([Self::grid_1d(n)?, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(raw).arg_ptr(out).arg_f32(scale).arg_i32(n as i32)
            .launch(self.stream)
    }

    /// `c[m, n] = a[m, k] @ b[n, k]^T` in TRUE fp32, one sequential fmaf chain per output
    /// (row-invariant by construction). `k % 16 == 0`.
    pub fn gemm_f32(&self, a: DevicePtr, b: DevicePtr, c: DevicePtr, m: usize, n: usize, k: usize) -> Result<()> {
        ensure!(k % 16 == 0, "gemm_f32: K {k} not a multiple of 16");
        if m == 0 {
            return Ok(());
        }
        if m <= GEMV_F32_MAX_M && k % GEMV_F32_KC == 0 && !GEMV_F32_OFF.load(std::sync::atomic::Ordering::Relaxed) {
            // The same fmaf chain per output, parallel over N only: bit-identical at any M.
            return KernelLaunch::new(self.gpu, self.k.gemv_f32)
                .grid([n.div_ceil(8) as u32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(a).arg_ptr(b).arg_ptr(c).arg_i32(m as i32).arg_i32(n as i32).arg_i32(k as i32)
                .launch(self.stream);
        }
        if !GEMM_F32_V1.load(std::sync::atomic::Ordering::Relaxed) {
            // 128x64 register-blocked tile, the same chain per output: bit-identical.
            return KernelLaunch::new(self.gpu, self.k.gemm_f32_v2)
                .grid([n.div_ceil(64) as u32, m.div_ceil(128) as u32, 1])
                .block([256, 1, 1])
                .arg_ptr(a).arg_ptr(b).arg_ptr(c).arg_i32(m as i32).arg_i32(n as i32).arg_i32(k as i32)
                .launch(self.stream);
        }
        KernelLaunch::new(self.gpu, self.k.gemm_f32)
            .grid([n.div_ceil(64) as u32, m.div_ceil(64) as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(a).arg_ptr(b).arg_ptr(c).arg_i32(m as i32).arg_i32(n as i32).arg_i32(k as i32)
            .launch(self.stream)
    }

    /// `out[m, n] = bf16(x[m, :] . w[n, :])`, fp32 accumulate, one block per row (small N).
    pub fn gemm_bf16_smalln(&self, x: DevicePtr, w: DevicePtr, out: DevicePtr, m: usize, n: usize, k: usize) -> Result<()> {
        ensure!(k % 64 == 0, "gemm_bf16_smalln: K {k} not a multiple of 64");
        if m == 0 {
            return Ok(());
        }
        KernelLaunch::new(self.gpu, self.k.gemm_bf16_smalln)
            .grid([m as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x).arg_ptr(w).arg_ptr(out).arg_i32(n as i32).arg_i32(k as i32)
            .launch(self.stream)
    }

    /// `out[i] = start + i * stride` as i32: absolute RoPE positions.
    pub fn iota(&self, out: DevicePtr, start: usize, stride: usize, n: usize) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let last = start + stride * (n - 1);
        ensure!(last <= i32::MAX as usize, "position {last} overflows the i32 RoPE index");
        KernelLaunch::new(self.gpu, self.k.iota)
            .grid([Self::grid_1d(n)?, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(out).arg_i32(start as i32).arg_i32(stride as i32).arg_i32(n as i32)
            .launch(self.stream)
    }
}

/// Shape-static decode (`ATLAS_DSV41_CORE_STATIC`): the same operations with the pass START read
/// from `dstart` (one device i32) instead of a launch argument, and FIXED geometry, so a captured
/// graph replays at any position. `ld` is the static score width; the kernels stop at
/// `score_width(n_c)` for the position they read. See `sparse_index.cu`.
impl IndexOps<'_> {
    #[allow(clippy::too_many_arguments)]
    pub fn score_dev(&self, q: DevicePtr, ik: DevicePtr, w: DevicePtr, cand: Option<CandidateMask>, out: DevicePtr, t: usize, ld: usize, dstart: DevicePtr, ratio: usize) -> Result<()> {
        ensure!(ld % KEY_BLOCK == 0 && ratio >= 1 && t > 0, "score_dev: ld {ld}, ratio {ratio}, t {t}");
        let (cand_ptr, cand_ld) = match cand {
            Some(c) => (c.mask, c.ld),
            None => (DevicePtr(0), 0),
        };
        KernelLaunch::new(self.gpu, self.k.score_dev)
            .grid([t as u32, (ld / SCORE_KEYS_PER_BLOCK) as u32, 1])
            .block([SCORE_THREADS, 1, 1])
            .arg_ptr(q).arg_ptr(ik).arg_ptr(w).arg_ptr(cand_ptr).arg_ptr(out)
            .arg_i32(ld as i32).arg_ptr(dstart).arg_i32(ratio as i32).arg_i32(cand_ld as i32)
            .launch(self.stream)
    }

    pub fn topk_dev(&self, score: DevicePtr, out: DevicePtr, t: usize, ld: usize, dstart: DevicePtr, ratio: usize) -> Result<()> {
        KernelLaunch::new(self.gpu, self.k.topk_dev)
            .grid([t as u32, 1, 1])
            .block([SELECT_THREADS, 1, 1])
            .arg_ptr(score).arg_ptr(out).arg_i32(ld as i32).arg_ptr(dstart).arg_i32(ratio as i32)
            .launch(self.stream)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn select_candidates_dev(&self, score: DevicePtr, block_scratch: DevicePtr, cand: DevicePtr, t: usize, ld: usize, dstart: DevicePtr, ratio: usize, topk_blocks: usize, block_size: usize) -> Result<()> {
        KernelLaunch::new(self.gpu, self.k.candidates_dev)
            .grid([t as u32, 1, 1])
            .block([SELECT_THREADS, 1, 1])
            .arg_ptr(score).arg_ptr(block_scratch).arg_ptr(cand).arg_i32(ld as i32).arg_ptr(dstart)
            .arg_i32(ratio as i32).arg_i32(topk_blocks as i32).arg_i32(block_size as i32)
            .launch(self.stream)
    }

    /// `out[i] = base + i * stride`, base = start, or start & !1 when `even_floor`.
    pub fn iota_dev(&self, out: DevicePtr, dstart: DevicePtr, even_floor: bool, stride: usize, n: usize) -> Result<()> {
        KernelLaunch::new(self.gpu, self.k.iota_dev)
            .grid([Self::grid_1d(n)?, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(out).arg_ptr(dstart).arg_i32(i32::from(even_floor)).arg_i32(stride as i32).arg_i32(n as i32)
            .launch(self.stream)
    }

    /// Ratio 2: copy the pending `(kv, sc)` row to slot 0 of `kv`/`sc` when the start is odd.
    pub fn comp2_pending_in(&self, pending: DevicePtr, kv: DevicePtr, sc: DevicePtr, d: usize, dstart: DevicePtr) -> Result<()> {
        KernelLaunch::new(self.gpu, self.k.comp2_pending_in)
            .grid([Self::grid_1d(d)?, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(pending).arg_ptr(kv).arg_ptr(sc).arg_i32(d as i32).arg_ptr(dstart)
            .launch(self.stream)
    }

    /// Ratio 2: combine `n_pairs` pairs starting at slot `1 - (start & 1)`.
    pub fn combine2_dev(&self, kv: DevicePtr, sc: DevicePtr, out: DevicePtr, n_pairs: usize, d: usize, dstart: DevicePtr) -> Result<()> {
        KernelLaunch::new(self.gpu, self.k.combine2_dev)
            .grid([Self::grid_1d(n_pairs * d)?, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(kv).arg_ptr(sc).arg_ptr(out).arg_i32(n_pairs as i32).arg_i32(d as i32).arg_ptr(dstart)
            .launch(self.stream)
    }

    /// Ratio 2: slot `t` becomes the pending row when `t + (start & 1)` is odd.
    pub fn comp2_pending_out(&self, kv: DevicePtr, sc: DevicePtr, pending: DevicePtr, d: usize, t: usize, dstart: DevicePtr) -> Result<()> {
        KernelLaunch::new(self.gpu, self.k.comp2_pending_out)
            .grid([Self::grid_1d(d)?, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(kv).arg_ptr(sc).arg_ptr(pending).arg_i32(d as i32).arg_i32(t as i32).arg_ptr(dstart)
            .launch(self.stream)
    }

    /// Copy the pass's published compressed rows from `src` (row 0..) to `dst` at row `j0`.
    /// `max_rows` is the grid; the kernel publishes `(t + p) / 2` (ratio 2) or `t` rows.
    #[allow(clippy::too_many_arguments)]
    pub fn publish_rows(&self, src: DevicePtr, dst: DevicePtr, row_bytes: usize, max_rows: usize, ratio: usize, t: usize, dstart: DevicePtr) -> Result<()> {
        ensure!(row_bytes % 16 == 0 && (ratio == 1 || ratio == 2), "publish_rows: row {row_bytes} B, ratio {ratio}");
        KernelLaunch::new(self.gpu, self.k.publish_rows)
            .grid([max_rows as u32, 1, 1])
            .block([64, 1, 1])
            .arg_ptr(src).arg_ptr(dst).arg_i32((row_bytes / 16) as i32).arg_i32(ratio as i32).arg_i32(t as i32).arg_ptr(dstart)
            .launch(self.stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `n_pad` values the oracle recorded (runD/runE `n_pad` taps): n_c 256 -> 512,
    /// 512 -> 512, 1024 -> 1024. And n_c = 0 still gets one block, as the reference's `max`.
    #[test]
    fn score_width_matches_the_oracle_n_pad() {
        assert_eq!(score_width(256), 512);
        assert_eq!(score_width(512), 512);
        assert_eq!(score_width(513), 1024);
        assert_eq!(score_width(1024), 1024);
        assert_eq!(score_width(0), 512);
    }

    #[test]
    fn topk_width_clamps_to_visible_rows() {
        assert_eq!(topk_width(256), 256);
        assert_eq!(topk_width(1024), 512);
    }

    /// The selection contract's precondition (see sparse_index.cu): selecting only FINITE
    /// columns equals the reference's `where(idx < compress_lens)` only while the candidate
    /// pool can never be smaller than the top-k. Pinned so a config change trips it.
    /// Long context: the scratch plan stays inside the budget at 1M and every kernel's i32
    /// argument fits; short context keeps the whole chunk in one block.
    #[test]
    fn score_blocks_fit_the_budget_at_1m() {
        let (chunk, seq) = (3968usize, 1usize << 20);
        let r = score_rows_per_block(chunk, seq);
        assert_eq!(r, 224);
        let w = score_width(seq);
        assert!(r * (w * 4 + w / 8 * 4) <= SCORE_BUDGET);
        assert!(w <= i32::MAX as usize && (w / 128) <= 65535, "score grid.y {} over the limit", w / 128);
        assert_eq!(score_rows_per_block(2048, 8192), 2048);
        assert_eq!(score_rows_per_block(2048, 65536), 2048);
        assert!(score_rows_per_block(8, 1 << 20) >= 8, "decode T must fit one block");
        assert_eq!(cand_row_bytes(w), 16384);
    }

    #[test]
    fn candidate_bits_round_trip_and_reject_ragged_blocks() {
        let (rows, ld) = (3usize, 512usize);
        let cols: Vec<u8> = (0..rows * ld).map(|i| u8::from(((i % ld) / 8 + i / ld) % 3 == 0)).collect();
        let bits = pack_cand(&cols, rows, ld).unwrap();
        assert_eq!(bits.len(), rows * cand_row_bytes(ld));
        assert_eq!(unpack_cand(&bits, rows, ld), cols);
        let mut ragged = cols.clone();
        ragged[3] ^= 1;
        assert!(pack_cand(&ragged, rows, ld).is_err());
    }

    #[test]
    fn candidate_pool_covers_the_topk() {
        let (blocks, block_size) = (2048usize, 8usize);
        assert!(blocks * block_size >= INDEX_TOPK);
    }
}
