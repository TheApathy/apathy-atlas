// SPDX-License-Identifier: AGPL-3.0-only

//! The attention lane's side of the V4.1 forward: `attn_block::AttnCore` + `forward::PassHook`.
//!
//! Split (lead ruling, 2026-09-22): `attn_block.rs` (integrator) computes qr, q, the window kv
//! and ring writes, `wpos`, the inverse RoPE and wo_a/wo_b. This file does, per layer:
//!
//! 1. the COMPRESSOR on kv-source layers {2,8,14,20}: this chunk's rows of `ckv`/`ik`, written
//!    at ABSOLUTE offsets into caches this object owns, plus the ratio-2 `pending` row. Nothing
//!    else can write them, so "ckv cleared per chunk" cannot be expressed;
//! 2. the INDEXER on index-source layers {2,8,14,20,24,28,32,36}: its own q/wts projections,
//!    the score, layer 20's candidate blocks, the top-k. Layer 20 also keeps the last-128-rows
//!    tail of its top-k and candidates, which the decoder replay installs before layer 21;
//! 3. the one-pass SPARSE ATTENTION on every layer (window-only on 0/1).
//!
//! The per-sequence state lives here because `Dsv41Model` serves ONE sequence. If it ever
//! serves several, this state moves into `V41Seq` by `&mut` — that is the trigger to watch.
//!
//! Every recipe is pinned against runF_faithful: `kernels/gb10/deepseek-v4.1/attn/` —
//! `check_compressor.py`, `check_index_projection.py`, and `examples/dsv41_attn_seam.rs`
//! (SPEC.md 8d-8f).

use std::sync::Mutex;

use anyhow::{Context, Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::index::{CandidateMask, INDEX_HEAD_DIM, INDEX_HEADS, INDEX_TOPK, IndexKernels, IndexOps, score_width};
use crate::weight_loader::deepseek_v41::attn_block::{AttnCore, CoreArgs, RING, WINDOW};
use crate::weight_loader::deepseek_v41::forward::{PassHook, PassKind};
use crate::weight_loader::deepseek_v41::ops::{Dsv41Kernels, Fp8Linear, Ops, RopeTable, bf16_tensor};

/// Kernel module compiled from `cb3/dsv41_sparse_attn.cu`.
pub const ATTN_MODULE: &str = "dsv41_sparse_attn";
pub const KV_SOURCE_LAYERS: [usize; 4] = [2, 8, 14, 20];
pub const INDEX_SOURCE_LAYERS: [usize; 8] = [2, 8, 14, 20, 24, 28, 32, 36];
pub const CANDIDATE_SOURCE_LAYER: usize = 20;
/// Rows the decoder replay runs over (`window_size`).
pub const REPLAY_ROWS: usize = WINDOW;
pub const HEAD_DIM: usize = 512;
const HEADS_PER_BLOCK: usize = 8;
const ATTN_THREADS: u32 = 256;
const HIDDEN: usize = 5120;
const Q_LORA: usize = 1280;
/// `wts` scale: `index_head_dim^-0.5 * index_n_heads^-0.5` = 1/64, exact in fp32.
const WTS_SCALE: f32 = 1.0 / 64.0;
const CANDIDATE_TOPK_BLOCKS: usize = 2048;
const CANDIDATE_BLOCK_SIZE: usize = 8;
/// `engine/model.py` MM_TILE. The reference runs EVERY activation GEMM on exactly this many
/// rows (`v41_ref.mm`), because cuBLAS picks its algorithm from M. Measured: with M = chunk
/// length, re-chunking the same prompt changed L20's replay-tail top-k on 19 of 128 rows.
const MM_TILE: usize = 16;

/// Bytes of the ONE fp32 score scratch all index layers share: `[max_chunk, n_pad(max_seq)]`.
/// Layers >= 20 run at ratio 1, so `n_c` reaches `max_seq`.
pub fn score_scratch_bytes(max_chunk: usize, max_seq: usize) -> usize {
    max_chunk * score_width(max_seq) * 4
}

/// Run `gemm(x_tile, out_tile)` over `m` rows in MM_TILE-row tiles; the tail tile is zero-padded
/// so every call has the same M.
#[allow(clippy::too_many_arguments)]
fn tiled(
    gpu: &dyn GpuBackend,
    pad_in: DevicePtr,
    pad_out: DevicePtr,
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
        gpu.memset_async(pad_in, 0, MM_TILE * in_row, stream)?;
        gpu.copy_d2d_async(x.offset(full * MM_TILE * in_row), pad_in, rem * in_row, stream)?;
        gemm(pad_in, pad_out)?;
        gpu.copy_d2d_async(pad_out, out.offset(full * MM_TILE * out_row), rem * out_row, stream)?;
    }
    Ok(())
}

/// Compressor weights + sequence-lifetime caches of one kv-source layer.
struct KvSource {
    ratio: usize,
    /// r=2: fp32 copies (the reference runs this projection in TRUE fp32). r=1: bf16.
    wkv: DevicePtr,
    wgate: Option<DevicePtr>,
    norm: DevicePtr,
    wk: DevicePtr,
    k_norm: DevicePtr,
    /// `[rows_cap, 512]` / `[rows_cap, 128]` bf16, `rows_cap = max_seq / r + 1`.
    ckv: DevicePtr,
    ik: DevicePtr,
    rows_cap: usize,
    /// r=2 unpaired position carried across chunks: kv `[512]` f32 then score `[512]` f32.
    pending: DevicePtr,
}

struct Indexer {
    wq_b: Fp8Linear,
    weights_proj: DevicePtr,
    /// This layer's selection `[max_chunk, 512]` i64; inheriting layers read it.
    topk: DevicePtr,
}

struct Layer {
    ratio: usize,
    kv: Option<KvSource>,
    idx: Option<Indexer>,
}

/// Fixed scratch, sized at load.
struct Scratch {
    xf: DevicePtr,
    kvl: DevicePtr,
    sc: DevicePtr,
    latent: DevicePtr,
    pos: DevicePtr,
    deq: DevicePtr,
    qi: DevicePtr,
    wraw: DevicePtr,
    wts: DevicePtr,
    score: DevicePtr,
    score_bytes: usize,
    cand: DevicePtr,
    blocks: DevicePtr,
    pad_in: DevicePtr,
    pad_out: DevicePtr,
}

/// Mutable per-sequence / per-pass state.
#[derive(Default)]
struct SeqState {
    /// Per kv-source layer index into `layers`: is a ratio-2 row pending?
    has_pending: Vec<bool>,
    /// Handles the current kv-source layer published: (ckv, ik, rows, ratio).
    published: Option<(DevicePtr, DevicePtr, usize, usize)>,
    /// The selection the next attention reads, and the candidate mask the next indexer reads.
    topk: Option<DevicePtr>,
    cand: Option<(DevicePtr, usize)>,
    kind: Option<PassKind>,
    pass_start: usize,
    pass_t: usize,
    tail: TailState,
}

#[derive(Default)]
struct TailState {
    cur: usize,
    rows: usize,
    ld: usize,
}

struct Tail {
    topk: [DevicePtr; 2],
    cand: [DevicePtr; 2],
    cand_cap: usize,
}

pub struct Dsv41SparseCore {
    layers: Vec<Layer>,
    eps: f32,
    max_chunk: usize,
    max_seq: usize,
    attn_kernel: KernelHandle,
    idx: IndexKernels,
    rope_c: RopeTable,
    tail: Tail,
    s: Scratch,
    st: Mutex<SeqState>,
}

fn dev_f32_copy(gpu: &dyn GpuBackend, fwd: &Dsv41Kernels, bf: DevicePtr, n: usize) -> Result<DevicePtr> {
    let out = gpu.alloc(n * 4)?;
    let stream = gpu.default_stream();
    Ops { gpu, k: fwd, stream }.bf16_to_f32(bf, out, n)?;
    gpu.synchronize(stream)?;
    Ok(out)
}

impl Dsv41SparseCore {
    /// `freqs_c` is the forward's YaRN table (the compressed rows and every indexer query use
    /// it on every layer, SPEC sec 6). `max_seq` sizes the caches; `max_chunk` the scratch.
    pub fn load(
        gpu: &dyn GpuBackend,
        store: &WeightStore,
        config: &ModelConfig,
        max_seq: usize,
        max_chunk: usize,
        freqs_c: RopeTable,
    ) -> Result<Self> {
        ensure!(config.num_attention_heads == 64, "V4.1 attention is 64 heads, config says {}", config.num_attention_heads);
        ensure!(max_chunk <= RING - WINDOW, "chunk {max_chunk} would overwrite window rows still in use");
        ensure!(freqs_c.positions >= max_seq, "freqs_c covers {} positions, max_seq is {max_seq}", freqs_c.positions);
        let fwd = Dsv41Kernels::load(gpu)?;
        let idx = IndexKernels::load(gpu)?;
        let attn_kernel = gpu
            .kernel(ATTN_MODULE, "dsv41_sparse_attn_w32")
            .with_context(|| format!("{ATTN_MODULE}::dsv41_sparse_attn_w32 is not in the compiled PTX"))?;
        let total = std::cell::Cell::new(0usize);
        let alloc = |bytes: usize| -> Result<DevicePtr> {
            total.set(total.get() + bytes);
            gpu.alloc(bytes.max(256))
        };

        let mut layers = Vec::with_capacity(40);
        for layer in 0..40 {
            let ratio = *config.compress_ratios.get(layer).with_context(|| format!("no compress_ratio for layer {layer}"))?;
            let p = |s: &str| format!("layers.{layer}.attn.{s}");
            let kv = if KV_SOURCE_LAYERS.contains(&layer) {
                ensure!(ratio == 1 || ratio == 2, "kv-source layer {layer} has ratio {ratio}");
                ensure!(
                    store.contains(&p("compressor.wgate.weight")) == (ratio == 2),
                    "layer {layer}: compressor.wgate must exist exactly on ratio-2 layers"
                );
                let wkv_bf = bf16_tensor(store, &p("compressor.wkv.weight"), &[HEAD_DIM, HIDDEN])?;
                let (wkv, wgate) = if ratio == 2 {
                    let g = bf16_tensor(store, &p("compressor.wgate.weight"), &[HEAD_DIM, HIDDEN])?;
                    total.set(total.get() + 2 * HEAD_DIM * HIDDEN * 4);
                    (dev_f32_copy(gpu, &fwd, wkv_bf, HEAD_DIM * HIDDEN)?, Some(dev_f32_copy(gpu, &fwd, g, HEAD_DIM * HIDDEN)?))
                } else {
                    (wkv_bf, None)
                };
                let rows_cap = max_seq / ratio + 1;
                let ckv = alloc(rows_cap * HEAD_DIM * 2)?;
                let ik = alloc(rows_cap * INDEX_HEAD_DIM * 2)?;
                gpu.memset(ckv, 0, rows_cap * HEAD_DIM * 2)?;
                gpu.memset(ik, 0, rows_cap * INDEX_HEAD_DIM * 2)?;
                Some(KvSource {
                    ratio,
                    wkv,
                    wgate,
                    norm: bf16_tensor(store, &p("compressor.norm.weight"), &[HEAD_DIM])?,
                    wk: bf16_tensor(store, &p("indexer.wk.weight"), &[INDEX_HEAD_DIM, HEAD_DIM])?,
                    k_norm: bf16_tensor(store, &p("indexer.k_norm.weight"), &[INDEX_HEAD_DIM])?,
                    ckv,
                    ik,
                    rows_cap,
                    pending: alloc(2 * HEAD_DIM * 4)?,
                })
            } else {
                None
            };
            let idx = if INDEX_SOURCE_LAYERS.contains(&layer) {
                let wq_b = Fp8Linear::load(store, &p("indexer.wq_b"), INDEX_HEADS * INDEX_HEAD_DIM, Q_LORA)?;
                let wt = store.get(&p("indexer.weights_proj.weight"))?;
                ensure!(
                    wt.dtype == WeightDtype::BF16 && wt.shape == [INDEX_HEADS, HIDDEN],
                    "indexer.weights_proj: {:?} {:?}",
                    wt.dtype,
                    wt.shape
                );
                Some(Indexer { wq_b, weights_proj: wt.ptr, topk: alloc(max_chunk * INDEX_TOPK * 8)? })
            } else {
                None
            };
            layers.push(Layer { ratio, kv, idx });
        }

        let n_pad_max = score_width(max_seq);
        let score_bytes = score_scratch_bytes(max_chunk, max_seq);
        let row = HEAD_DIM * 4;
        let s = Scratch {
            xf: alloc(max_chunk * HIDDEN * 4)?,
            kvl: alloc((max_chunk + 1) * row)?,
            sc: alloc((max_chunk + 1) * row)?,
            latent: alloc((max_chunk + 1) * HEAD_DIM * 2)?,
            pos: alloc(max_chunk * 4)?,
            deq: alloc(INDEX_HEADS * INDEX_HEAD_DIM * Q_LORA * 2)?,
            qi: alloc(max_chunk * INDEX_HEADS * INDEX_HEAD_DIM * 2)?,
            wraw: alloc(max_chunk * INDEX_HEADS * 2)?,
            wts: alloc(max_chunk * INDEX_HEADS * 4)?,
            score: alloc(score_bytes)?,
            score_bytes,
            cand: alloc(max_chunk * n_pad_max)?,
            blocks: alloc(max_chunk * n_pad_max.div_ceil(CANDIDATE_BLOCK_SIZE) * 4)?,
            pad_in: alloc(MM_TILE * HIDDEN * 4)?,
            pad_out: alloc(MM_TILE * INDEX_HEADS * INDEX_HEAD_DIM * 4)?,
        };
        let tail = Tail {
            topk: [alloc(REPLAY_ROWS * INDEX_TOPK * 8)?, alloc(REPLAY_ROWS * INDEX_TOPK * 8)?],
            cand: [alloc(REPLAY_ROWS * n_pad_max)?, alloc(REPLAY_ROWS * n_pad_max)?],
            cand_cap: n_pad_max,
        };
        tracing::info!(
            "dsv41 sparse core: {:.2} GB device (score scratch {:.1} MB = [{max_chunk}, {n_pad_max}] fp32; \
             ckv/ik for max_seq {max_seq})",
            total.get() as f64 / 1e9,
            score_bytes as f64 / 1e6
        );
        Ok(Self {
            layers,
            eps: config.rms_norm_eps as f32,
            max_chunk,
            max_seq,
            attn_kernel,
            idx,
            rope_c: freqs_c,
            tail,
            s,
            st: Mutex::new(SeqState { has_pending: vec![false; 40], ..SeqState::default() }),
        })
    }

    /// This layer's `(ckv, ik)` caches, if it is a kv-source layer (for gates).
    pub fn caches(&self, layer: usize) -> Option<(DevicePtr, DevicePtr)> {
        self.layers.get(layer)?.kv.as_ref().map(|k| (k.ckv, k.ik))
    }

    /// The selection the current pass's attention reads, and L20's replay tail
    /// `(topk, cand, cand_ld, rows)` (for gates).
    pub fn current_topk(&self) -> Option<DevicePtr> {
        self.st.lock().expect("core state poisoned").topk
    }
    pub fn current_candidates(&self) -> Option<(DevicePtr, usize)> {
        self.st.lock().expect("core state poisoned").cand
    }
    pub fn replay_tail(&self) -> (DevicePtr, DevicePtr, usize, usize) {
        let st = self.st.lock().expect("core state poisoned");
        (self.tail.topk[st.tail.cur], self.tail.cand[st.tail.cur], st.tail.ld, st.tail.rows)
    }

    /// NEGATIVE CONTROL ONLY: forget the carried ratio-2 row at `layer`, as a port that
    /// dropped it would. Nothing in the forward calls it.
    pub fn control_drop_pending(&self, layer: usize) {
        self.st.lock().expect("core state poisoned").has_pending[layer] = false;
    }

    fn compress(&self, ops: &Ops, a: &CoreArgs, st: &mut SeqState) -> Result<()> {
        let (layer, t, start, stream) = (a.layer, a.t, a.start, ops.stream);
        let c = self.layers[layer].kv.as_ref().expect("kv-source");
        let gpu = ops.gpu;
        let iops = IndexOps { gpu, k: &self.idx, stream };
        let s = &self.s;
        let r = c.ratio;
        let row = HEAD_DIM * 4;
        let (j0, nj) = if r == 2 {
            let pend = usize::from(st.has_pending[layer]);
            let n = t + pend;
            ops.bf16_to_f32(a.x, s.xf, t * HIDDEN)?;
            if pend == 1 {
                gpu.copy_d2d_async(c.pending, s.kvl, row, stream)?;
                gpu.copy_d2d_async(c.pending.offset(row), s.sc, row, stream)?;
            }
            let wgate = c.wgate.expect("ratio-2 has a gate");
            tiled(gpu, s.pad_in, s.pad_out, s.xf, s.kvl.offset(pend * row), t, HIDDEN * 4, row, stream, |x, o| {
                ops.linear_f32(x, c.wkv, o, MM_TILE, HEAD_DIM, HIDDEN)
            })?;
            tiled(gpu, s.pad_in, s.pad_out, s.xf, s.sc.offset(pend * row), t, HIDDEN * 4, row, stream, |x, o| {
                ops.linear_f32(x, wgate, o, MM_TILE, HEAD_DIM, HIDDEN)
            })?;
            if n % 2 == 1 {
                gpu.copy_d2d_async(s.kvl.offset((n - 1) * row), c.pending, row, stream)?;
                gpu.copy_d2d_async(s.sc.offset((n - 1) * row), c.pending.offset(row), row, stream)?;
            }
            st.has_pending[layer] = n % 2 == 1;
            iops.combine2(s.kvl, s.sc, s.latent, n / 2, HEAD_DIM)?;
            ((start - pend) / 2, n / 2)
        } else {
            tiled(gpu, s.pad_in, s.pad_out, a.x, s.latent, t, HIDDEN * 2, HEAD_DIM * 2, stream, |x, o| {
                ops.linear_bf16(x, c.wkv, o, MM_TILE, HEAD_DIM, HIDDEN)
            })?;
            (start, t)
        };
        ensure!(
            j0 + nj <= c.rows_cap,
            "layer {layer}: compressed row {} past the cache ({} rows, max_seq {})",
            j0 + nj,
            c.rows_cap,
            self.max_seq
        );
        if nj > 0 {
            ops.rmsnorm(s.latent, c.norm, s.latent, nj, HEAD_DIM, self.eps)?;
            iops.iota(s.pos, j0 * r, r, nj)?; // compressed row j sits at absolute position j*r
            // The index key from the PRE-RoPE latent, straight into the cache.
            let ik = c.ik.offset(j0 * INDEX_HEAD_DIM * 2);
            tiled(gpu, s.pad_in, s.pad_out, s.latent, ik, nj, HEAD_DIM * 2, INDEX_HEAD_DIM * 2, stream, |x, o| {
                ops.linear_bf16(x, c.wk, o, MM_TILE, INDEX_HEAD_DIM, HEAD_DIM)
            })?;
            ops.rmsnorm(ik, c.k_norm, ik, nj, INDEX_HEAD_DIM, self.eps)?;
            ops.rope_tail(ik, s.pos, &self.rope_c, nj, 1, INDEX_HEAD_DIM, false)?;
            let ckv = c.ckv.offset(j0 * HEAD_DIM * 2);
            gpu.copy_d2d_async(s.latent, ckv, nj * HEAD_DIM * 2, stream)?;
            ops.rope_tail(ckv, s.pos, &self.rope_c, nj, 1, HEAD_DIM, false)?;
        }
        st.published = Some((c.ckv, c.ik, j0 + nj, r));
        Ok(())
    }

    fn index(&self, ops: &Ops, a: &CoreArgs, st: &mut SeqState) -> Result<()> {
        let (layer, t, start, stream) = (a.layer, a.t, a.start, ops.stream);
        let lw = &self.layers[layer];
        let w = lw.idx.as_ref().expect("index-source");
        let (_, ik, rows, ratio) = st.published.context("no compressed cache published before the indexer")?;
        // The reference asserts sh.ratio == w.ratio on every layer.
        ensure!(ratio == lw.ratio, "layer {layer} (ratio {}) would read a cache built at ratio {ratio}", lw.ratio);
        let gpu = ops.gpu;
        let iops = IndexOps { gpu, k: &self.idx, stream };
        let s = &self.s;
        let n_c = (start + t) / ratio;
        ensure!(n_c <= rows, "layer {layer}: {n_c} compressed rows visible, {rows} published");
        if n_c == 0 {
            gpu.memset_async(w.topk, 0xFF, t * INDEX_TOPK * 8, stream)?; // all -1
            st.topk = Some(w.topk);
            return Ok(());
        }
        let n_pad = score_width(n_c);
        ensure!(t * n_pad * 4 <= s.score_bytes, "score [{t}, {n_pad}] exceeds the scratch");

        // q_i = rope_c(bf16(qr @ dequant(wq_b)^T)); wts = bf16(x @ weights_proj^T) * 1/64
        ops.dequant(&w.wq_b, s.deq)?;
        let qrow = INDEX_HEADS * INDEX_HEAD_DIM * 2;
        tiled(gpu, s.pad_in, s.pad_out, a.qr, s.qi, t, Q_LORA * 2, qrow, stream, |x, o| {
            ops.linear_bf16(x, s.deq, o, MM_TILE, w.wq_b.n, w.wq_b.k)
        })?;
        iops.iota(s.pos, start, 1, t)?;
        ops.rope_tail(s.qi, s.pos, &self.rope_c, t, INDEX_HEADS, INDEX_HEAD_DIM, false)?;
        tiled(gpu, s.pad_in, s.pad_out, a.x, s.wraw, t, HIDDEN * 2, INDEX_HEADS * 2, stream, |x, o| {
            ops.linear_bf16(x, w.weights_proj, o, MM_TILE, INDEX_HEADS, HIDDEN)
        })?;
        iops.wts(s.wraw, s.wts, WTS_SCALE, t * INDEX_HEADS)?;

        // Candidates prune only the layers AFTER the source (use_cand in model.py).
        let cand_in = match st.cand {
            Some((mask, ld)) if layer > CANDIDATE_SOURCE_LAYER => Some(CandidateMask { mask, ld }),
            _ => None,
        };
        iops.score(s.qi, ik, s.wts, cand_in, s.score, t, rows, n_pad, start, ratio)?;
        if layer == CANDIDATE_SOURCE_LAYER {
            iops.select_candidates(s.score, s.blocks, s.cand, t, n_pad, start, ratio, CANDIDATE_TOPK_BLOCKS, CANDIDATE_BLOCK_SIZE)?;
            st.cand = Some((s.cand, n_pad));
        }
        iops.topk(s.score, w.topk, t, n_pad, n_c)?;
        st.topk = Some(w.topk);
        if layer == CANDIDATE_SOURCE_LAYER && st.kind == Some(PassKind::EncoderChunk) {
            self.keep_tail(gpu, st, start, t, w.topk, n_pad, stream)?;
        }
        Ok(())
    }

    /// Append this encoder chunk's L20 rows to the replay tail (`Model._rep_keep`): the last
    /// <= 128 prompt rows, which can span chunks. Ping-pong buffers.
    #[allow(clippy::too_many_arguments)]
    fn keep_tail(&self, gpu: &dyn GpuBackend, st: &mut SeqState, start: usize, t: usize, topk: DevicePtr, n_pad: usize, stream: u64) -> Result<()> {
        let tl = &mut st.tail;
        if start == 0 {
            *tl = TailState::default();
        }
        ensure!(n_pad <= self.tail.cand_cap, "candidate width {n_pad} exceeds the tail's {}", self.tail.cand_cap);
        ensure!(n_pad >= tl.ld, "candidate width shrank within a prompt ({} -> {n_pad})", tl.ld);
        let take = t.min(REPLAY_ROWS);
        let keep = (tl.rows + take).min(REPLAY_ROWS) - take;
        let (src, dst) = (tl.cur, 1 - tl.cur);
        let (tk, cd) = (&self.tail.topk, &self.tail.cand);
        // Older rows were scored against fewer columns: pad with 0, as _rep_tail pads False.
        gpu.memset_async(cd[dst], 0, REPLAY_ROWS * n_pad, stream)?;
        if keep > 0 {
            let from = tl.rows - keep;
            gpu.copy_d2d_async(tk[src].offset(from * INDEX_TOPK * 8), tk[dst], keep * INDEX_TOPK * 8, stream)?;
            gpu.copy_d2d_2d_async(cd[src].offset(from * tl.ld), tl.ld, cd[dst], n_pad, tl.ld, keep, stream)?;
        }
        let from = t - take;
        gpu.copy_d2d_async(topk.offset(from * INDEX_TOPK * 8), tk[dst].offset(keep * INDEX_TOPK * 8), take * INDEX_TOPK * 8, stream)?;
        gpu.copy_d2d_2d_async(self.s.cand.offset(from * n_pad), n_pad, cd[dst].offset(keep * n_pad), n_pad, n_pad, take, stream)?;
        *tl = TailState { cur: dst, rows: keep + take, ld: n_pad };
        Ok(())
    }

    fn attend(&self, ops: &Ops, a: &CoreArgs, st: &SeqState) -> Result<()> {
        let ratio = self.layers[a.layer].ratio;
        let (ckv, cidx, rows) = if ratio == 0 {
            (DevicePtr::NULL, DevicePtr::NULL, 0)
        } else {
            let (ckv, _, rows, r) = st.published.context("sparse attention before any compressed cache")?;
            ensure!(r == ratio, "layer {} (ratio {ratio}) would attend over a cache built at ratio {r}", a.layer);
            (ckv, st.topk.context("sparse attention before any top-k in this pass")?, rows)
        };
        ensure!(rows > 0 || ratio == 0, "layer {}: empty compressed cache", a.layer);
        if a.t == 0 {
            return Ok(());
        }
        KernelLaunch::new(ops.gpu, self.attn_kernel)
            .grid([a.t as u32, (64 / HEADS_PER_BLOCK) as u32, 1])
            .block([ATTN_THREADS, 1, 1])
            .arg_ptr(a.q)
            .arg_ptr(a.ring)
            .arg_ptr(a.wpos)
            .arg_ptr(ckv)
            .arg_ptr(cidx)
            .arg_ptr(a.sink)
            .arg_ptr(a.out)
            .arg_i32(a.t as i32)
            .arg_i32(64)
            .arg_i32(HEAD_DIM as i32)
            .arg_i32(WINDOW as i32)
            .arg_i32(INDEX_TOPK as i32)
            .arg_i32(RING as i32)
            .arg_i32(a.win_lo as i32)
            .arg_f32((HEAD_DIM as f32).powf(-0.5))
            .launch(ops.stream)
    }
}

impl PassHook for Dsv41SparseCore {
    fn begin_pass(&self, kind: PassKind, start: usize, t: usize) -> Result<()> {
        ensure!(t <= self.max_chunk, "pass of {t} rows exceeds max_chunk {}", self.max_chunk);
        let mut st = self.st.lock().expect("core state poisoned");
        if start == 0 && kind != PassKind::Replay {
            // A new sequence: nothing is carried.
            st.has_pending = vec![false; 40];
            st.published = None;
            st.tail = TailState::default();
        }
        st.kind = Some(kind);
        st.pass_start = start;
        st.pass_t = t;
        match kind {
            PassKind::Replay => {
                // Layers 21..23 attend with L20's tail selection; 24..36 score inside its
                // candidate pool. Both belong to the QUERIES, carried from the encoder pass.
                ensure!(st.tail.rows == t, "replay over {t} rows but the L20 tail holds {}", st.tail.rows);
                st.topk = Some(self.tail.topk[st.tail.cur]);
                st.cand = Some((self.tail.cand[st.tail.cur], st.tail.ld));
            }
            PassKind::EncoderChunk | PassKind::FullChunk | PassKind::Decode => {
                st.topk = None;
                st.cand = None;
            }
        }
        Ok(())
    }
}

impl AttnCore for Dsv41SparseCore {
    fn run(&self, ops: &Ops, a: &CoreArgs) -> Result<()> {
        ensure!(a.layer < self.layers.len(), "layer {} out of range", a.layer);
        let mut st = self.st.lock().expect("core state poisoned");
        ensure!(
            st.kind.is_some() && a.start == st.pass_start && a.t == st.pass_t,
            "core.run(layer {}, start {}, t {}) outside the pass begun at ({}, {})",
            a.layer,
            a.start,
            a.t,
            st.pass_start,
            st.pass_t
        );
        let lw = &self.layers[a.layer];
        if lw.kv.is_some() {
            self.compress(ops, a, &mut st)?;
        }
        if lw.idx.is_some() {
            self.index(ops, a, &mut st)?;
        }
        self.attend(ops, a, &st)
    }
}
