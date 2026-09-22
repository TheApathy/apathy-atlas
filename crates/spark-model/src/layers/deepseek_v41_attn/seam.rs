// SPDX-License-Identifier: AGPL-3.0-only

//! The attention lane's implementation of `weight_loader::deepseek_v41::seams::Dsv41Attention`.
//!
//! Split with the integrator (ruled 2026-09-22): the integrator's `V41AttentionBlock` owns
//! the q/kv projections, RoPE on q/kv, the window ring, `wpos`, the inverse RoPE and the
//! output projections. This file owns the three seam calls, in the order the block makes
//! them:
//!
//! 1. [`Dsv41Attention::compress`] on kv-source layers {2,8,14,20}: the compressor, writing
//!    this chunk's rows of `ckv`/`ik` at ABSOLUTE offsets into caches THIS OBJECT OWNS, then
//!    publishing them. The ratio-2 unpaired `pending` row lives here too. Nothing else can
//!    write these caches, so "ckv cleared per chunk" cannot be expressed.
//! 2. [`Dsv41Attention::sparse_index_select`] on index-source layers: q/wts projections,
//!    score, (layer 20) candidates, top-k. Layer 20 also keeps the last-128-rows tail of its
//!    top-k and candidates for the decoder replay (CED + SWA bounded replay).
//! 3. [`Dsv41Attention::sparse_attention`] on every layer (window-only on 0/1).
//!
//! Every recipe here is pinned by a CPU check against runF_faithful
//! (`kernels/gb10/deepseek-v4.1/attn/check_compressor.py`, `check_index_projection.py`) and
//! every kernel by a gate (SPEC.md 8d, 8e).

use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::kv_cache::KvCacheDtype;
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::index::{CandidateMask, INDEX_HEAD_DIM, INDEX_HEADS, IndexKernels, IndexOps, score_width};
use crate::layer::ForwardContext;
use crate::weight_loader::deepseek_v41::ops::{Dsv41Kernels, Fp8Linear, Ops, RopeSpec, RopeTable, bf16_tensor};
use crate::weight_loader::deepseek_v41::seams::{
    CANDIDATE_SOURCE_LAYER, Dsv41Attention, Dsv41AttentionLoader, INDEX_SOURCE_LAYERS, INDEX_TOPK,
    KV_SOURCE_LAYERS, SparseShared,
};

/// Kernel module compiled from `cb3/dsv41_sparse_attn.cu`.
pub const ATTN_MODULE: &str = "dsv41_sparse_attn";
const HEADS_PER_BLOCK: usize = 8;
const THREADS: u32 = 256;
pub const HEAD_DIM: usize = 512;
pub const WINDOW: usize = 128;
const HIDDEN: usize = 5120;
const Q_LORA: usize = 1280;
/// `wts` scale: `index_head_dim^-0.5 * index_n_heads^-0.5` = 1/64, exact in fp32.
const WTS_SCALE: f32 = 1.0 / 64.0;
const CANDIDATE_TOPK_BLOCKS: usize = 2048;
const CANDIDATE_BLOCK_SIZE: usize = 8;
/// Rows the decoder replay runs over (`window_size`).
pub const REPLAY_ROWS: usize = 128;

/// `engine/model.py` MM_TILE. The reference runs EVERY activation GEMM on exactly this many
/// rows (`v41_ref.mm`), because cuBLAS picks its algorithm from M and a GEMM row is otherwise
/// not the same for every chunk length. Measured here: with M = chunk length, splitting the
/// same 1024 tokens as [0,511,1000,1023,1024] instead of 2 x 512 changed L20's replay-tail
/// top-k on 19 of 128 rows. The GEMMs this file issues are therefore tiled the same way.
const MM_TILE: usize = 16;

/// A device buffer that grows on demand. `free` synchronises the device, so growing never
/// pulls memory out from under a kernel still in flight.
#[derive(Default)]
struct Buf {
    ptr: Option<DevicePtr>,
    bytes: usize,
}

impl Buf {
    fn ensure(&mut self, gpu: &dyn GpuBackend, bytes: usize) -> Result<DevicePtr> {
        if let Some(p) = self.ptr
            && bytes <= self.bytes
        {
            return Ok(p);
        }
        if let Some(p) = self.ptr.take() {
            gpu.free(p)?;
        }
        let want = bytes.max(256).next_power_of_two();
        let p = gpu.alloc(want)?;
        self.ptr = Some(p);
        self.bytes = want;
        Ok(p)
    }
}

/// Compressor weights and the sequence-lifetime caches of one kv-source layer.
struct Compressor {
    ratio: usize,
    /// r=2: fp32 copies (the reference runs this projection in TRUE fp32). r=1: bf16.
    wkv: DevicePtr,
    wgate: Option<DevicePtr>,
    norm: DevicePtr,
    wk: DevicePtr,
    k_norm: DevicePtr,
    /// `[rows_cap, 512]` bf16 and `[rows_cap, 128]` bf16, `rows_cap = max_seq / r + 1`.
    ckv: DevicePtr,
    ik: DevicePtr,
    rows_cap: usize,
    /// r=2 unpaired position carried across chunks: kv `[512]` f32 then score `[512]` f32.
    pending: DevicePtr,
}

struct IndexerW {
    wq_b: Fp8Linear,
    weights_proj: DevicePtr,
}

/// Layer 20's replay tail: the last <= 128 prompt rows of its top-k and candidates, which
/// can span two chunks (`engine/model.py` `_rep_keep` / `_rep_tail`). Ping-pong buffers:
/// each chunk writes the new tail into the idle one.
struct ReplayTail {
    topk: [DevicePtr; 2],
    cand: [DevicePtr; 2],
    cand_cap: usize,
    cur: usize,
    rows: usize,
    ld: usize,
}

#[derive(Default)]
struct Scratch {
    xf: Buf,
    kvl: Buf,
    sc: Buf,
    latent: Buf,
    pos: Buf,
    deq: Buf,
    qi: Buf,
    wraw: Buf,
    wts: Buf,
    score: Buf,
    topk: Buf,
    cand: Buf,
    blocks: Buf,
    pad_in: Buf,
    pad_out: Buf,
}

/// Run `gemm(x_tile, out_tile)` over `m` rows in fixed MM_TILE-row tiles; the tail tile is
/// zero-padded to MM_TILE rows so every call has the same M.
#[allow(clippy::too_many_arguments)]
fn tiled(
    gpu: &dyn GpuBackend,
    pad_in: &mut Buf,
    pad_out: &mut Buf,
    x: DevicePtr,
    out: DevicePtr,
    m: usize,
    in_row: usize,
    out_row: usize,
    stream: u64,
    gemm: impl Fn(DevicePtr, DevicePtr) -> Result<()>,
) -> Result<()> {
    let full = m / MM_TILE;
    for i in 0..full {
        gemm(x.offset(i * MM_TILE * in_row), out.offset(i * MM_TILE * out_row))?;
    }
    let rem = m % MM_TILE;
    if rem > 0 {
        let xin = pad_in.ensure(gpu, MM_TILE * in_row)?;
        let tmp = pad_out.ensure(gpu, MM_TILE * out_row)?;
        gpu.memset_async(xin, 0, MM_TILE * in_row, stream)?;
        gpu.copy_d2d_async(x.offset(full * MM_TILE * in_row), xin, rem * in_row, stream)?;
        gemm(xin, tmp)?;
        gpu.copy_d2d_async(tmp, out.offset(full * MM_TILE * out_row), rem * out_row, stream)?;
    }
    Ok(())
}

struct State {
    has_pending: bool,
    s: Scratch,
    tail: Option<ReplayTail>,
}

pub struct Dsv41SparseAttention {
    pub layer: usize,
    pub num_heads: usize,
    /// `compress_ratios[layer]` from the config (0 on layers 0/1).
    pub ratio: usize,
    eps: f32,
    attn_kernel: KernelHandle,
    idx: IndexKernels,
    fwd: Dsv41Kernels,
    rope_c: RopeTable,
    compressor: Option<Compressor>,
    indexer: Option<IndexerW>,
    state: Mutex<State>,
}

fn dev_f32_copy(gpu: &dyn GpuBackend, fwd: &Dsv41Kernels, bf: DevicePtr, n: usize) -> Result<DevicePtr> {
    let out = gpu.alloc(n * 4)?;
    let stream = gpu.default_stream();
    Ops { gpu, k: fwd, stream }.bf16_to_f32(bf, out, n)?;
    gpu.synchronize(stream)?;
    Ok(out)
}

impl Dsv41SparseAttention {
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        gpu: &dyn GpuBackend,
        store: &WeightStore,
        config: &ModelConfig,
        layer: usize,
        max_seq: usize,
        rope_c: RopeTable,
    ) -> Result<Self> {
        let num_heads = config.num_attention_heads;
        ensure!(num_heads % HEADS_PER_BLOCK == 0, "dsv41_sparse_attn needs heads in groups of {HEADS_PER_BLOCK}");
        let ratio = *config
            .compress_ratios
            .get(layer)
            .with_context(|| format!("compress_ratios has no entry for layer {layer}"))?;
        let attn_kernel = gpu
            .kernel(ATTN_MODULE, "dsv41_sparse_attn")
            .with_context(|| format!("{ATTN_MODULE}::dsv41_sparse_attn is not in the compiled PTX"))?;
        let idx = IndexKernels::load(gpu)?;
        let fwd = Dsv41Kernels::load(gpu)?;
        let p = |s: &str| format!("layers.{layer}.attn.{s}");

        let compressor = if KV_SOURCE_LAYERS.contains(&layer) {
            ensure!(ratio == 1 || ratio == 2, "kv-source layer {layer} has ratio {ratio}; V4.1 uses 1 or 2");
            let has_gate = store.contains(&p("compressor.wgate.weight"));
            ensure!(
                has_gate == (ratio == 2),
                "layer {layer}: compressor.wgate present={has_gate} but ratio={ratio} (the gate exists exactly on ratio-2 layers)"
            );
            let wkv_bf = bf16_tensor(store, &p("compressor.wkv.weight"), &[HEAD_DIM, HIDDEN])?;
            let (wkv, wgate) = if ratio == 2 {
                let g = bf16_tensor(store, &p("compressor.wgate.weight"), &[HEAD_DIM, HIDDEN])?;
                (dev_f32_copy(gpu, &fwd, wkv_bf, HEAD_DIM * HIDDEN)?, Some(dev_f32_copy(gpu, &fwd, g, HEAD_DIM * HIDDEN)?))
            } else {
                (wkv_bf, None)
            };
            let rows_cap = max_seq / ratio + 1;
            let ckv = gpu.alloc(rows_cap * HEAD_DIM * 2)?;
            let ik = gpu.alloc(rows_cap * INDEX_HEAD_DIM * 2)?;
            gpu.memset(ckv, 0, rows_cap * HEAD_DIM * 2)?;
            gpu.memset(ik, 0, rows_cap * INDEX_HEAD_DIM * 2)?;
            let pending = gpu.alloc(2 * HEAD_DIM * 4)?;
            Some(Compressor {
                ratio,
                wkv,
                wgate,
                norm: bf16_tensor(store, &p("compressor.norm.weight"), &[HEAD_DIM])?,
                wk: bf16_tensor(store, &p("indexer.wk.weight"), &[INDEX_HEAD_DIM, HEAD_DIM])?,
                k_norm: bf16_tensor(store, &p("indexer.k_norm.weight"), &[INDEX_HEAD_DIM])?,
                ckv,
                ik,
                rows_cap,
                pending,
            })
        } else {
            None
        };

        let indexer = if INDEX_SOURCE_LAYERS.contains(&layer) {
            let wq_b = Fp8Linear::load(store, &p("indexer.wq_b"), INDEX_HEADS * INDEX_HEAD_DIM, Q_LORA)?;
            let wt = store.get(&p("indexer.weights_proj.weight"))?;
            ensure!(wt.dtype == WeightDtype::BF16 && wt.shape == [INDEX_HEADS, HIDDEN], "indexer.weights_proj: {:?} {:?}", wt.dtype, wt.shape);
            Some(IndexerW { wq_b, weights_proj: wt.ptr })
        } else {
            None
        };

        let tail = if layer == CANDIDATE_SOURCE_LAYER {
            // Widest candidate row the replay can see: n_pad at the longest prompt.
            let cand_cap = score_width(max_seq / ratio.max(1) + 1);
            let topk = [gpu.alloc(REPLAY_ROWS * INDEX_TOPK * 8)?, gpu.alloc(REPLAY_ROWS * INDEX_TOPK * 8)?];
            let cand = [gpu.alloc(REPLAY_ROWS * cand_cap)?, gpu.alloc(REPLAY_ROWS * cand_cap)?];
            Some(ReplayTail { topk, cand, cand_cap, cur: 0, rows: 0, ld: 0 })
        } else {
            None
        };

        Ok(Self {
            layer,
            num_heads,
            ratio,
            eps: config.rms_norm_eps as f32,
            attn_kernel,
            idx,
            fwd,
            rope_c,
            compressor,
            indexer,
            state: Mutex::new(State { has_pending: false, s: Scratch::default(), tail }),
        })
    }

    /// This layer's sequence-lifetime `(ckv, ik)` caches, if it is a kv-source layer.
    pub fn caches(&self) -> Option<(DevicePtr, DevicePtr)> {
        self.compressor.as_ref().map(|c| (c.ckv, c.ik))
    }

    /// NEGATIVE CONTROL ONLY: forget the carried ratio-2 row, as a port that dropped it would.
    /// Exists so a gate can watch the pending path fail; nothing in the forward calls it.
    pub fn control_drop_pending(&self) {
        self.state.lock().expect("attention state poisoned").has_pending = false;
    }

    /// Append this chunk's rows of L20's top-k / candidates to the replay tail.
    #[allow(clippy::too_many_arguments)]
    fn keep_tail(
        gpu: &dyn GpuBackend,
        tail: &mut ReplayTail,
        chunk_start: usize,
        t: usize,
        topk: DevicePtr,
        cand: DevicePtr,
        n_pad: usize,
        stream: u64,
    ) -> Result<()> {
        if chunk_start == 0 {
            // A new prompt: nothing from the previous one may leak into its replay.
            tail.rows = 0;
            tail.ld = 0;
        }
        ensure!(n_pad <= tail.cand_cap, "candidate width {n_pad} exceeds the replay tail's {}", tail.cand_cap);
        ensure!(n_pad >= tail.ld, "candidate width shrank within a prompt ({} -> {n_pad})", tail.ld);
        let take = t.min(REPLAY_ROWS);
        let keep = (tail.rows + take).min(REPLAY_ROWS) - take;
        let (src, dst) = (tail.cur, 1 - tail.cur);
        // Older rows were scored against fewer columns; pad them with 0 (model.py pads False).
        gpu.memset_async(tail.cand[dst], 0, REPLAY_ROWS * n_pad, stream)?;
        if keep > 0 {
            let from = tail.rows - keep;
            gpu.copy_d2d_async(tail.topk[src].offset(from * INDEX_TOPK * 8), tail.topk[dst], keep * INDEX_TOPK * 8, stream)?;
            gpu.copy_d2d_2d_async(tail.cand[src].offset(from * tail.ld), tail.ld, tail.cand[dst], n_pad, tail.ld, keep, stream)?;
        }
        let from = t - take;
        gpu.copy_d2d_async(topk.offset(from * INDEX_TOPK * 8), tail.topk[dst].offset(keep * INDEX_TOPK * 8), take * INDEX_TOPK * 8, stream)?;
        gpu.copy_d2d_2d_async(cand.offset(from * n_pad), n_pad, tail.cand[dst].offset(keep * n_pad), n_pad, n_pad, take, stream)?;
        tail.cur = dst;
        tail.rows = keep + take;
        tail.ld = n_pad;
        Ok(())
    }
}

/// The seam bodies take the backend directly (the trait passes `ctx.gpu`), so a harness can
/// drive them without a full `ForwardContext`.
impl Dsv41SparseAttention {
    #[allow(clippy::too_many_arguments)]
    pub fn compress_with(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        layer: usize,
        chunk_start: usize,
        num_tokens: usize,
        shared: &mut SparseShared,
        stream: u64,
    ) -> Result<()> {
        ensure!(layer == self.layer, "attention object for layer {} called as layer {layer}", self.layer);
        let c = self
            .compressor
            .as_ref()
            .with_context(|| format!("compress() called on layer {layer}, which is not a kv-source layer {KV_SOURCE_LAYERS:?}"))?;
        let ops = Ops { gpu, k: &self.fwd, stream };
        let iops = IndexOps { gpu, k: &self.idx, stream };
        let mut st = self.state.lock().expect("attention state poisoned");
        if chunk_start == 0 {
            st.has_pending = false; // a new sequence: nothing is carried
        }
        let t = num_tokens;
        let r = c.ratio;
        let row = HEAD_DIM * 4;
        let (j0, nj, latent) = if r == 2 {
            let pend = usize::from(st.has_pending);
            let n = t + pend;
            let xf = st.s.xf.ensure(gpu, t * HIDDEN * 4)?;
            let kvl = st.s.kvl.ensure(gpu, n * row)?;
            let sc = st.s.sc.ensure(gpu, n * row)?;
            ops.bf16_to_f32(x, xf, t * HIDDEN)?;
            if pend == 1 {
                gpu.copy_d2d_async(c.pending, kvl, row, stream)?;
                gpu.copy_d2d_async(c.pending.offset(row), sc, row, stream)?;
            }
            let wgate = c.wgate.expect("ratio-2 has a gate");
            let sc_ = &mut st.s;
            tiled(gpu, &mut sc_.pad_in, &mut sc_.pad_out, xf, kvl.offset(pend * row), t, HIDDEN * 4, row, stream,
                  |a, o| ops.linear_f32(a, c.wkv, o, MM_TILE, HEAD_DIM, HIDDEN))?;
            tiled(gpu, &mut sc_.pad_in, &mut sc_.pad_out, xf, sc.offset(pend * row), t, HIDDEN * 4, row, stream,
                  |a, o| ops.linear_f32(a, wgate, o, MM_TILE, HEAD_DIM, HIDDEN))?;
            if n % 2 == 1 {
                gpu.copy_d2d_async(kvl.offset((n - 1) * row), c.pending, row, stream)?;
                gpu.copy_d2d_async(sc.offset((n - 1) * row), c.pending.offset(row), row, stream)?;
            }
            st.has_pending = n % 2 == 1;
            let n_pairs = n / 2;
            let latent = st.s.latent.ensure(gpu, n_pairs.max(1) * HEAD_DIM * 2)?;
            iops.combine2(kvl, sc, latent, n_pairs, HEAD_DIM)?;
            ((chunk_start - pend) / 2, n_pairs, latent)
        } else {
            let latent = st.s.latent.ensure(gpu, t * HEAD_DIM * 2)?;
            let sc_ = &mut st.s;
            tiled(gpu, &mut sc_.pad_in, &mut sc_.pad_out, x, latent, t, HIDDEN * 2, HEAD_DIM * 2, stream,
                  |a, o| ops.linear_bf16(a, c.wkv, o, MM_TILE, HEAD_DIM, HIDDEN))?;
            (chunk_start, t, latent)
        };
        ensure!(j0 + nj <= c.rows_cap, "layer {layer}: compressed row {} exceeds the cache ({} rows; max_seq too small)", j0 + nj, c.rows_cap);
        if nj > 0 {
            ops.rmsnorm(latent, c.norm, latent, nj, HEAD_DIM, self.eps)?;
            let pos = st.s.pos.ensure(gpu, nj * 4)?;
            iops.iota(pos, j0 * r, r, nj)?; // compressed row j sits at absolute position j*r
            // Index key from the PRE-RoPE latent, written straight into the cache.
            let ik = c.ik.offset(j0 * INDEX_HEAD_DIM * 2);
            let sc_ = &mut st.s;
            tiled(gpu, &mut sc_.pad_in, &mut sc_.pad_out, latent, ik, nj, HEAD_DIM * 2, INDEX_HEAD_DIM * 2, stream,
                  |a, o| ops.linear_bf16(a, c.wk, o, MM_TILE, INDEX_HEAD_DIM, HEAD_DIM))?;
            ops.rmsnorm(ik, c.k_norm, ik, nj, INDEX_HEAD_DIM, self.eps)?;
            ops.rope_tail(ik, pos, &self.rope_c, nj, 1, INDEX_HEAD_DIM, false)?;
            let ckv = c.ckv.offset(j0 * HEAD_DIM * 2);
            gpu.copy_d2d_async(latent, ckv, nj * HEAD_DIM * 2, stream)?;
            ops.rope_tail(ckv, pos, &self.rope_c, nj, 1, HEAD_DIM, false)?;
        }
        shared.publish_compressed(c.ckv, c.ik, j0 + nj, r);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sparse_index_select_with(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        qr: DevicePtr,
        layer: usize,
        num_tokens: usize,
        chunk_start: usize,
        _chunk_len: usize,
        shared: &mut SparseShared,
        stream: u64,
    ) -> Result<()> {
        ensure!(layer == self.layer, "attention object for layer {} called as layer {layer}", self.layer);
        let w = self
            .indexer
            .as_ref()
            .with_context(|| format!("sparse_index_select on layer {layer}, not an index-source layer {INDEX_SOURCE_LAYERS:?}"))?;
        // The reference asserts sh.ratio == w.ratio on every layer: a layer must never read a
        // cache built at another stride.
        ensure!(
            shared.ratio == self.ratio,
            "layer {layer} (ratio {}) would read a compressed cache built at ratio {}",
            self.ratio,
            shared.ratio
        );
        let (ik, ckv_rows) = (shared.ik.context("no compressed cache published")?, shared.ckv_rows);
        let ops = Ops { gpu, k: &self.fwd, stream };
        let iops = IndexOps { gpu, k: &self.idx, stream };
        let mut st = self.state.lock().expect("attention state poisoned");
        let t = num_tokens;
        let r = self.ratio;
        let n_c = (chunk_start + t) / r;
        ensure!(n_c <= ckv_rows, "layer {layer}: {n_c} compressed rows visible but only {ckv_rows} published");
        let topk = st.s.topk.ensure(gpu, t.max(1) * INDEX_TOPK * 8)?;
        if n_c == 0 || t == 0 {
            gpu.memset_async(topk, 0xFF, t * INDEX_TOPK * 8, stream)?; // all -1
            shared.topk = Some(topk);
            return Ok(());
        }
        let n_pad = score_width(n_c);

        // q_i = rope_c(bf16(qr @ dequant(wq_b)^T)), wts = bf16(x @ weights_proj^T) * 1/64
        let deq = st.s.deq.ensure(gpu, w.wq_b.bf16_bytes())?;
        let qi = st.s.qi.ensure(gpu, t * INDEX_HEADS * INDEX_HEAD_DIM * 2)?;
        ops.dequant(&w.wq_b, deq)?;
        let sc_ = &mut st.s;
        tiled(gpu, &mut sc_.pad_in, &mut sc_.pad_out, qr, qi, t, Q_LORA * 2, INDEX_HEADS * INDEX_HEAD_DIM * 2, stream,
              |a, o| ops.linear_bf16(a, deq, o, MM_TILE, w.wq_b.n, w.wq_b.k))?;
        let pos = st.s.pos.ensure(gpu, t * 4)?;
        iops.iota(pos, chunk_start, 1, t)?;
        ops.rope_tail(qi, pos, &self.rope_c, t, INDEX_HEADS, INDEX_HEAD_DIM, false)?;
        let wraw = st.s.wraw.ensure(gpu, t * INDEX_HEADS * 2)?;
        let wts = st.s.wts.ensure(gpu, t * INDEX_HEADS * 4)?;
        let sc_ = &mut st.s;
        tiled(gpu, &mut sc_.pad_in, &mut sc_.pad_out, x, wraw, t, HIDDEN * 2, INDEX_HEADS * 2, stream,
              |a, o| ops.linear_bf16(a, w.weights_proj, o, MM_TILE, INDEX_HEADS, HIDDEN))?;
        iops.wts(wraw, wts, WTS_SCALE, t * INDEX_HEADS)?;

        // Candidates prune only layers AFTER the source (use_cand in model.py).
        let use_cand = layer > CANDIDATE_SOURCE_LAYER && shared.candidates.is_some();
        let cand_in = if use_cand {
            Some(CandidateMask { mask: shared.candidates.expect("checked"), ld: shared.candidates_ld })
        } else {
            None
        };
        let score = st.s.score.ensure(gpu, t * n_pad * 4)?;
        iops.score(qi, ik, wts, cand_in, score, t, ckv_rows, n_pad, chunk_start, r)?;

        if layer == CANDIDATE_SOURCE_LAYER {
            let cand = st.s.cand.ensure(gpu, t * n_pad)?;
            let blocks = st.s.blocks.ensure(gpu, t * n_pad.div_ceil(CANDIDATE_BLOCK_SIZE) * 4)?;
            iops.select_candidates(score, blocks, cand, t, n_pad, chunk_start, r, CANDIDATE_TOPK_BLOCKS, CANDIDATE_BLOCK_SIZE)?;
            shared.candidates = Some(cand);
            shared.candidates_ld = n_pad;
        }
        iops.topk(score, topk, t, n_pad, n_c)?;
        shared.topk = Some(topk);

        if layer == CANDIDATE_SOURCE_LAYER {
            let cand = st.s.cand.ptr.expect("allocated above");
            let tail = st.tail.as_mut().expect("layer 20 owns the replay tail");
            Self::keep_tail(gpu, tail, chunk_start, t, topk, cand, n_pad, stream)?;
            shared.replay_topk = Some(tail.topk[tail.cur]);
            shared.replay_candidates = Some(tail.cand[tail.cur]);
            shared.replay_candidates_ld = tail.ld;
            shared.replay_rows = tail.rows;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn sparse_attention_with(
        &self,
        gpu: &dyn GpuBackend,
        q: DevicePtr,
        ring: DevicePtr,
        ring_len: usize,
        wpos: DevicePtr,
        win_lo: usize,
        ckv: Option<DevicePtr>,
        ckv_rows: usize,
        cidx: Option<DevicePtr>,
        sink: DevicePtr,
        scale: f32,
        num_tokens: usize,
        out: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        // ckv and cidx travel together: rows without a selection (or a selection without
        // rows) is a caller bug that would otherwise read garbage or silently drop keys.
        ensure!(
            ckv.is_some() == cidx.is_some(),
            "layer {}: ckv and cidx must both be present (ratio != 0) or both absent (layers 0/1)",
            self.layer
        );
        ensure!(ckv.is_some() == (self.ratio != 0), "layer {} has ratio {} but ckv present={}", self.layer, self.ratio, ckv.is_some());
        ensure!(ckv.is_none() || ckv_rows > 0, "layer {}: compressed cache with 0 rows", self.layer);
        ensure!(ring_len > 0, "layer {}: empty window ring", self.layer);
        if num_tokens == 0 {
            return Ok(());
        }
        KernelLaunch::new(gpu, self.attn_kernel)
            .grid([num_tokens as u32, (self.num_heads / HEADS_PER_BLOCK) as u32, 1])
            .block([THREADS, 1, 1])
            .arg_ptr(q)
            .arg_ptr(ring)
            .arg_ptr(wpos)
            .arg_ptr(ckv.unwrap_or(DevicePtr::NULL))
            .arg_ptr(cidx.unwrap_or(DevicePtr::NULL))
            .arg_ptr(sink)
            .arg_ptr(out)
            .arg_i32(num_tokens as i32)
            .arg_i32(self.num_heads as i32)
            .arg_i32(HEAD_DIM as i32)
            .arg_i32(WINDOW as i32)
            .arg_i32(INDEX_TOPK as i32)
            .arg_i32(ring_len as i32)
            .arg_i32(win_lo as i32)
            .arg_f32(scale)
            .launch(stream)
    }
}

impl Dsv41Attention for Dsv41SparseAttention {
    fn compress(&self, x: DevicePtr, layer: usize, chunk_start: usize, num_tokens: usize, shared: &mut SparseShared, ctx: &ForwardContext, stream: u64) -> Result<()> {
        self.compress_with(ctx.gpu, x, layer, chunk_start, num_tokens, shared, stream)
    }

    fn sparse_index_select(
        &self,
        x: DevicePtr,
        qr: DevicePtr,
        layer: usize,
        num_tokens: usize,
        chunk_start: usize,
        chunk_len: usize,
        shared: &mut SparseShared,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.sparse_index_select_with(ctx.gpu, x, qr, layer, num_tokens, chunk_start, chunk_len, shared, stream)
    }

    fn sparse_attention(
        &self,
        q: DevicePtr,
        ring: DevicePtr,
        ring_len: usize,
        wpos: DevicePtr,
        win_lo: usize,
        ckv: Option<DevicePtr>,
        ckv_rows: usize,
        cidx: Option<DevicePtr>,
        sink: DevicePtr,
        scale: f32,
        num_tokens: usize,
        out: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.sparse_attention_with(ctx.gpu, q, ring, ring_len, wpos, win_lo, ckv, ckv_rows, cidx, sink, scale, num_tokens, out, stream)
    }
}

/// Registered once by the binary: `register_attention_loader(Box::leak(Box::new(loader)))`.
pub struct Dsv41SparseAttentionLoader {
    /// Longest sequence the KV cache holds; sizes ckv/ik and the RoPE table.
    pub max_seq: usize,
    rope_c: OnceLock<RopeTable>,
}

impl Dsv41SparseAttentionLoader {
    pub fn new(max_seq: usize) -> Self {
        Self { max_seq, rope_c: OnceLock::new() }
    }

    /// `freqs_c`: theta 160000 WITH YaRN (original_seq_len 65536, factor 16, beta 32/1).
    /// Used for every compressed row and every indexer query on every layer (SPEC sec 6).
    fn rope_c(&self, gpu: &dyn GpuBackend) -> Result<RopeTable> {
        if let Some(t) = self.rope_c.get() {
            return Ok(*t);
        }
        let spec = RopeSpec { dim: 64, original_seq_len: 65536, base: 160000.0, factor: 16.0, beta_fast: 32.0, beta_slow: 1.0 };
        let t = spec.upload(gpu, self.max_seq + 8)?;
        Ok(*self.rope_c.get_or_init(|| t))
    }
}

impl Dsv41AttentionLoader for Dsv41SparseAttentionLoader {
    fn load_attention(
        &self,
        layer: usize,
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        _kv_dtype: KvCacheDtype,
    ) -> Result<Box<dyn Dsv41Attention>> {
        let rope_c = self.rope_c(gpu)?;
        Ok(Box::new(Dsv41SparseAttention::load(gpu, store, config, layer, self.max_seq, rope_c)?))
    }
}
