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
        })
    }
}

/// Candidate mask from layer 20: `[num_tokens, ld]` u8, 1 = keep.
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
        KernelLaunch::new(self.gpu, self.k.gemm_f32)
            .grid([n.div_ceil(64) as u32, m.div_ceil(64) as u32, 1])
            .block([256, 1, 1])
            .arg_ptr(a).arg_ptr(b).arg_ptr(c).arg_i32(m as i32).arg_i32(n as i32).arg_i32(k as i32)
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
    #[test]
    fn candidate_pool_covers_the_topk() {
        let (blocks, block_size) = (2048usize, 8usize);
        assert!(blocks * block_size >= INDEX_TOPK);
    }
}
