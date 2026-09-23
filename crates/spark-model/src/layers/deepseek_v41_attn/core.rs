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

use std::collections::BTreeMap;
use std::sync::Mutex;
use std::time::Instant;

use anyhow::{Context, Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::index::{CandidateMask, INDEX_HEAD_DIM, INDEX_HEADS, INDEX_TOPK, IndexKernels, IndexOps, cand_row_bytes, score_rows_per_block, score_width};
use crate::weight_loader::deepseek_v41::attn_block::{AttnCore, CoreArgs, RING, WINDOW};
use crate::weight_loader::deepseek_v41::device_allocs::{DeviceAllocs, SharedGpu};
use crate::weight_loader::deepseek_v41::forward::{PassHook, PassKind};
use crate::weight_loader::deepseek_v41::ops::{FWD_MODULE, Fp8Linear, Ops, RopeTable, bf16_tensor};

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
/// Decode (T <= DECODE_MAX_T) uses the SPLIT-KV entry: keys cut into slices of this many,
/// one CTA per (row, head group, slice), then a combine. Measured at T=1 on runC_2048 L2's real
/// tensors: one-pass 527 us, split 82 us (6.4x; sweep 8/16/32/64 -> 16 fastest). Only the
/// within-row accumulation order changes: 99.98% of bf16 outputs identical to the one-pass
/// kernel, the rest 1 ulp (attn gate log). Used ONLY for Decode passes, so every PREFILL row
/// stays chunk-invariant bit-for-bit.
const DECODE_SLICE: usize = 16;
const DECODE_MAX_T: usize = 8;
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

/// Bytes of the ONE fp32 score scratch all index layers share: `[rows, n_pad(max_seq)]`, rows =
/// [`score_rows_per_block`] (the whole chunk up to ~200K context; 224 rows at 1M). Layers >= 20 run
/// at ratio 1, so `n_c` reaches `max_seq`.
pub fn score_scratch_bytes(max_chunk: usize, max_seq: usize) -> usize {
    score_rows_per_block(max_chunk, max_seq) * score_width(max_seq) * 4
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

/// Per-phase wall time, ONLY when `ATLAS_DSV41_CORE_PROF=1`: every mark synchronizes the stream,
/// so a profiled run is slower and its total is not a throughput number. Off, it costs nothing.
struct Prof {
    on: bool,
    inner: Mutex<(Instant, BTreeMap<&'static str, (f64, u64)>)>,
}

impl Prof {
    fn new() -> Self {
        let on = std::env::var("ATLAS_DSV41_CORE_PROF").as_deref() == Ok("1");
        Self { on, inner: Mutex::new((Instant::now(), BTreeMap::new())) }
    }
    fn begin(&self, ops: &Ops) -> Result<()> {
        if self.on {
            ops.gpu.synchronize(ops.stream)?;
            self.inner.lock().expect("prof").0 = Instant::now();
        }
        Ok(())
    }
    fn mark(&self, ops: &Ops, name: &'static str) -> Result<()> {
        if self.on {
            ops.gpu.synchronize(ops.stream)?;
            let mut g = self.inner.lock().expect("prof");
            let now = Instant::now();
            let dt = now.duration_since(g.0).as_secs_f64() * 1e3;
            g.0 = now;
            let e = g.1.entry(name).or_insert((0.0, 0));
            e.0 += dt;
            e.1 += 1;
        }
        Ok(())
    }
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
    /// ROLLBACK (DSpark verify): the last pass's compressor inputs, as `Caches._chunk_inputs`:
    /// its projected kv/score rows `[max_chunk, 512]` f32 each, and the pending row it started
    /// from. Only ratio-2 layers carry state across positions, so only they keep these.
    save_kv: DevicePtr,
    save_sc: DevicePtr,
    before: DevicePtr,
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
    /// Rows the score/block scratch holds: the indexer walks a chunk in blocks of this many.
    score_rows: usize,
    cand: DevicePtr,
    blocks: DevicePtr,
    pad_in: DevicePtr,
    pad_out: DevicePtr,
    /// Static decode: the pass's index-key rows before they are published `[DECODE_MAX_T, 128]`.
    iks: DevicePtr,
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
    /// Per layer: the last compressor pass `(start, t, pending_before)` -- what rollback restores
    /// from. `None` until the layer has compressed in this sequence.
    last: Vec<Option<(usize, usize, bool)>>,
    /// Positions the core has absorbed (end of the last pass, or the rollback point).
    len: usize,
    pass_start: usize,
    pass_t: usize,
    tail: TailState,
    /// Static decode: where the pass start lives on the device when the caller owns it (graph
    /// replay), and whether the core has written its own copy for the current pass otherwise.
    dstart_ext: Option<DevicePtr>,
    dstart_written: bool,
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
    split_kernel: KernelHandle,
    /// Prefill/replay: the tensor-core entry (`dsv41_sparse_attn_mma`, 16 heads x 128 threads per
    /// token). On the real runC fixture: 12.5x faster than the one-pass fp32 kernel, 99.90% of
    /// bf16 outputs identical to it, no added error vs the fp32 reference (attn gate log).
    /// `ATLAS_DSV41_ATTN_MMA=0` restores the one-pass kernel (A/B arm only).
    mma_kernel: Option<(KernelHandle, usize)>,
    combine_kernel: KernelHandle,
    /// `[DECODE_MAX_T, 64, S, 2 + 512]` fp32 split partials.
    part: DevicePtr,
    idx: IndexKernels,
    rope_c: RopeTable,
    tail: Tail,
    s: Scratch,
    st: Mutex<SeqState>,
    prof: Prof,
    /// `ATLAS_DSV41_ATTN_SPLIT=0` forces the one-pass kernel for decode too (A/B arm only).
    split_on: bool,
    /// `ATLAS_DSV41_COMP_CUBLAS=1`: the ratio-2 compressor's fp32 projections and the indexer's
    /// weights_proj as the reference issues them (cuBLASLt, 16-row tiles) instead of the
    /// deterministic `dsv41_gemm_f32_nt` / `dsv41_gemm_bf16_smalln` (A/B arm only).
    comp_cublas_tiled: bool,
    /// `ATLAS_DSV41_CORE_STATIC=1` (or [`Self::set_static_decode`]): Decode/Verify passes with
    /// T <= DECODE_MAX_T run shape-static -- no launch reads a host value that changes per step.
    static_on: std::sync::atomic::AtomicBool,
    /// The core's own device copy of the pass start, used when no caller pointer is set.
    own_dstart: DevicePtr,
    /// Static decode score width: `score_width(max_seq)`, the widest any pass can need.
    static_ld: usize,
    /// Owns every device allocation above; frees them when the core drops.
    allocs: DeviceAllocs,
}

/// Widen a bf16 weight to an fp32 copy at load. Launches `dsv41_fwd::dsv41_bf16_to_f32` directly
/// rather than through `Dsv41Kernels::load`, which also allocates decode scratch this core would
/// not own (caught by the drop test: 46 live allocations, 45 recorded).
fn dev_f32_copy(gpu: &dyn GpuBackend, cvt: KernelHandle, bf: DevicePtr, out: DevicePtr, n: usize) -> Result<DevicePtr> {
    let stream = gpu.default_stream();
    KernelLaunch::new(gpu, cvt)
        .grid([u32::try_from(n.div_ceil(256))?, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(bf)
        .arg_ptr(out)
        .arg_u64(n as u64)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    Ok(out)
}

impl Dsv41SparseCore {
    /// `freqs_c` is the forward's YaRN table (the compressed rows and every indexer query use
    /// it on every layer, SPEC sec 6). `max_seq` sizes the caches; `max_chunk` the scratch.
    pub fn load(
        gpu: &SharedGpu,
        store: &WeightStore,
        config: &ModelConfig,
        max_seq: usize,
        max_chunk: usize,
        freqs_c: RopeTable,
    ) -> Result<Self> {
        Self::load_prefix(gpu, store, config, max_seq, max_chunk, freqs_c, 40)
    }

    /// Layers `0..n_layers` only — for drivers that load a layer prefix of the checkpoint.
    pub fn load_prefix(
        shared: &SharedGpu,
        store: &WeightStore,
        config: &ModelConfig,
        max_seq: usize,
        max_chunk: usize,
        freqs_c: RopeTable,
        n_layers: usize,
    ) -> Result<Self> {
        ensure!((1..=40).contains(&n_layers), "n_layers {n_layers} outside 1..=40");
        let gpu: &dyn GpuBackend = shared.as_ref();
        ensure!(config.num_attention_heads == 64, "V4.1 attention is 64 heads, config says {}", config.num_attention_heads);
        ensure!(max_chunk <= RING - WINDOW, "chunk {max_chunk} would overwrite window rows still in use");
        ensure!(freqs_c.positions >= max_seq, "freqs_c covers {} positions, max_seq is {max_seq}", freqs_c.positions);
        let cvt = gpu.kernel(FWD_MODULE, "dsv41_bf16_to_f32").context("dsv41_fwd::dsv41_bf16_to_f32 not in the PTX")?;
        let idx = IndexKernels::load(gpu)?;
        let split_kernel = gpu.kernel(ATTN_MODULE, "dsv41_sparse_attn_split").context("dsv41_sparse_attn_split not in the PTX")?;
        // Heads per CTA for the tensor-core entry: 16 (4 warps), 32 or 64 (the same warps fused,
        // sharing one K gather per tile). All three are BYTE-IDENTICAL (attn gate); the choice is
        // speed only. `ATLAS_DSV41_ATTN_HPC` overrides the default for A/B.
        let hpc: usize = std::env::var("ATLAS_DSV41_ATTN_HPC").ok().and_then(|v| v.parse().ok()).unwrap_or(32);
        let mma_kernel = if std::env::var("ATLAS_DSV41_ATTN_MMA").as_deref() == Ok("0") {
            None
        } else {
            // ATLAS_DSV41_ATTN_FA=1: the FA2-style restructure (softmax in registers, 2 syncs per
            // tile), byte-identical to the mma entries by construction (attn gate). 16/32 only.
            let fa = std::env::var("ATLAS_DSV41_ATTN_FA").as_deref() == Ok("1");
            let name = match (hpc, fa) {
                (16, false) => "dsv41_sparse_attn_mma",
                (32, false) => "dsv41_sparse_attn_mma32h",
                (64, false) => "dsv41_sparse_attn_mma64",
                (16, true) => "dsv41_sparse_attn_fa16h",
                (32, true) => "dsv41_sparse_attn_fa32h",
                (o, _) => anyhow::bail!("ATLAS_DSV41_ATTN_HPC={o}: use 16, 32 or 64 (16 or 32 with ATLAS_DSV41_ATTN_FA=1)"),
            };
            Some((gpu.kernel(ATTN_MODULE, name).with_context(|| format!("{name} not in the PTX"))?, hpc))
        };
        let combine_kernel = gpu.kernel(ATTN_MODULE, "dsv41_sparse_attn_combine").context("dsv41_sparse_attn_combine not in the PTX")?;
        let attn_kernel = gpu
            .kernel(ATTN_MODULE, "dsv41_sparse_attn_w32")
            .with_context(|| format!("{ATTN_MODULE}::dsv41_sparse_attn_w32 is not in the compiled PTX"))?;
        let total = std::cell::Cell::new(0usize);
        // Every allocation goes through ONE owned guard: freed when the core drops, and freed
        // with the error if this constructor fails half-way (TUI model swap).
        let allocs = std::cell::RefCell::new(DeviceAllocs::owned(shared.clone()));
        let alloc = |bytes: usize| -> Result<DevicePtr> {
            total.set(total.get() + bytes);
            allocs.borrow_mut().alloc(gpu, bytes.max(256))
        };

        let mut layers = Vec::with_capacity(n_layers);
        for layer in 0..n_layers {
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
                    (
                        dev_f32_copy(gpu, cvt, wkv_bf, alloc(HEAD_DIM * HIDDEN * 4)?, HEAD_DIM * HIDDEN)?,
                        Some(dev_f32_copy(gpu, cvt, g, alloc(HEAD_DIM * HIDDEN * 4)?, HEAD_DIM * HIDDEN)?),
                    )
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
                    save_kv: alloc(max_chunk * HEAD_DIM * 4)?,
                    save_sc: alloc(max_chunk * HEAD_DIM * 4)?,
                    before: alloc(2 * HEAD_DIM * 4)?,
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
        // ATLAS_DSV41_SCORE_ROWS: GATE ONLY -- force smaller row blocks to prove blocking changes no
        // byte (runJ digest A/B). Never larger than the budget allows.
        let score_rows = match std::env::var("ATLAS_DSV41_SCORE_ROWS").ok().and_then(|v| v.parse::<usize>().ok()) {
            Some(r) => r.clamp(DECODE_MAX_T, score_rows_per_block(max_chunk, max_seq)),
            None => score_rows_per_block(max_chunk, max_seq),
        };
        ensure!(score_rows >= DECODE_MAX_T, "score blocks of {score_rows} rows cannot hold a decode pass");
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
            score_rows,
            cand: alloc(max_chunk * cand_row_bytes(n_pad_max))?,
            blocks: alloc(score_rows * n_pad_max.div_ceil(CANDIDATE_BLOCK_SIZE) * 4)?,
            pad_in: alloc(MM_TILE * HIDDEN * 4)?,
            pad_out: alloc(MM_TILE * INDEX_HEADS * INDEX_HEAD_DIM * 4)?,
            iks: alloc(DECODE_MAX_T * INDEX_HEAD_DIM * 2)?,
        };
        let own_dstart = alloc(4)?;
        let part = alloc(DECODE_MAX_T * 64 * (WINDOW + INDEX_TOPK).div_ceil(DECODE_SLICE) * (2 + HEAD_DIM) * 4)?;
        let tail = Tail {
            topk: [alloc(REPLAY_ROWS * INDEX_TOPK * 8)?, alloc(REPLAY_ROWS * INDEX_TOPK * 8)?],
            cand: [alloc(REPLAY_ROWS * cand_row_bytes(n_pad_max))?, alloc(REPLAY_ROWS * cand_row_bytes(n_pad_max))?],
            cand_cap: n_pad_max,
        };
        tracing::info!(
            "dsv41 sparse core: {:.2} GB device (score scratch {:.1} MB = [{score_rows}, {n_pad_max}] fp32; \
             ckv/ik for max_seq {max_seq})",
            total.get() as f64 / 1e9,
            score_bytes as f64 / 1e6
        );
        if std::env::var("ATLAS_DSV41_GEMM_F32_V1").as_deref() == Ok("1") {
            // A/B arm only: the compressor's prefill projections back on the 64x64 GEMM.
            super::index::GEMM_F32_V1.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        if std::env::var("ATLAS_DSV41_GEMV_F32").as_deref() == Ok("0") {
            // A/B arm only: the ratio-2 compressor's small-M projections back on the 64x64 GEMM.
            super::index::GEMV_F32_OFF.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        Ok(Self {
            layers,
            eps: config.rms_norm_eps as f32,
            max_chunk,
            max_seq,
            attn_kernel,
            split_kernel,
            mma_kernel,
            combine_kernel,
            part,
            idx,
            rope_c: freqs_c,
            tail,
            s,
            st: Mutex::new(SeqState { has_pending: vec![false; 40], last: vec![None; 40], ..SeqState::default() }),
            prof: Prof::new(),
            allocs: allocs.into_inner(),
            split_on: std::env::var("ATLAS_DSV41_ATTN_SPLIT").as_deref() != Ok("0"),
            comp_cublas_tiled: std::env::var("ATLAS_DSV41_COMP_CUBLAS").as_deref() == Ok("1"),
            static_on: std::sync::atomic::AtomicBool::new(std::env::var("ATLAS_DSV41_CORE_STATIC").as_deref() == Ok("1")),
            own_dstart,
            static_ld: n_pad_max,
        })
    }

    /// Device allocations this core owns (freed on drop).
    pub fn allocation_count(&self) -> usize {
        self.allocs.len()
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

    /// `ATLAS_DSV41_CORE_PROF=1` breakdown so far: (phase, total ms, calls), then resets it.
    pub fn prof_take(&self) -> Vec<(&'static str, f64, u64)> {
        let mut g = self.prof.inner.lock().expect("prof");
        std::mem::take(&mut g.1).into_iter().map(|(k, (ms, n))| (k, ms, n)).collect()
    }

    /// DSpark ROLLBACK: discard every position >= `n` (`engine/model.py` `Caches.rollback`).
    ///
    /// Only the ratio-2 compressor carries state across positions: its unpaired `pending` row.
    /// It is restored from the LAST pass's saved inputs: `n` even -> nothing pending; else the
    /// row for position n-1, which is either in the last pass (its saved projection) or the one
    /// that pass started from. Everything else is append-only at absolute offsets -- ckv/ik rows,
    /// L20's cache, the rings (the caller's) -- so rows >= n are simply rewritten by the next
    /// pass before anything reads them (compressed row j is visible only to positions >= j*r,
    /// and the next pass writes row j before attending). Rolling back past the last pass's
    /// start is refused: those inputs are no longer kept.
    pub fn rollback(&self, gpu: &dyn GpuBackend, n: usize, stream: u64) -> Result<()> {
        self.rollback_inner(gpu, n, true, stream)
    }

    /// NEGATIVE CONTROL ONLY: truncate the length but SKIP the pending restore -- the bug the
    /// rollback gate must be able to see.
    pub fn control_rollback_without_restore(&self, gpu: &dyn GpuBackend, n: usize, stream: u64) -> Result<()> {
        self.rollback_inner(gpu, n, false, stream)
    }

    fn rollback_inner(&self, gpu: &dyn GpuBackend, n: usize, restore: bool, stream: u64) -> Result<()> {
        let mut st = self.st.lock().expect("core state poisoned");
        ensure!(n <= st.len, "rollback({n}) past the {} positions held", st.len);
        let row = HEAD_DIM * 4;
        for (layer, lw) in self.layers.iter().enumerate() {
            let Some(c) = lw.kv.as_ref() else { continue };
            if c.ratio != 2 || !restore {
                continue;
            }
            let Some((s0, t0, pend_before)) = st.last[layer] else {
                // Never compressed in this sequence (a gate driving a subset of layers): it holds
                // no pending row to restore.
                ensure!(!st.has_pending[layer], "layer {layer}: pending row with no compressor pass recorded");
                continue;
            };
            if n % 2 == 0 {
                st.has_pending[layer] = false;
                continue;
            }
            let p = n - 1; // the position left unpaired at n
            if p >= s0 && p < s0 + t0 {
                gpu.copy_d2d_async(c.save_kv.offset((p - s0) * row), c.pending, row, stream)?;
                gpu.copy_d2d_async(c.save_sc.offset((p - s0) * row), c.pending.offset(row), row, stream)?;
            } else if p + 1 == s0 {
                ensure!(pend_before, "layer {layer}: position {p} should have been pending before the pass at {s0}");
                gpu.copy_d2d_async(c.before, c.pending, 2 * row, stream)?;
            } else {
                anyhow::bail!(
                    "rollback({n}) reaches before the last pass (start {s0}); the compressor input for position {p} is no longer kept"
                );
            }
            st.has_pending[layer] = true;
        }
        st.len = n;
        st.kind = None; // the next pass starts clean
        Ok(())
    }

    /// NEGATIVE CONTROL ONLY: forget the carried ratio-2 row at `layer`, as a port that
    /// dropped it would. Nothing in the forward calls it.
    pub fn control_drop_pending(&self, layer: usize) {
        self.st.lock().expect("core state poisoned").has_pending[layer] = false;
    }

    /// GATE SCAFFOLDING ONLY: install a captured state for the decoder replay — L20's
    /// compressed cache (`rows` rows of ckv/ik, copied into the core's OWN caches) and the replay
    /// tail's candidate mask `[tail_rows, cand_ld]` — so a capture holding only replay layers
    /// (runI: L24/L28) can be gated teacher-forced. Nothing in the forward calls it.
    #[allow(clippy::too_many_arguments)]
    pub fn gate_seed_replay(
        &self,
        gpu: &dyn GpuBackend,
        ckv: DevicePtr,
        ik: DevicePtr,
        rows: usize,
        cand: DevicePtr,
        cand_ld: usize,
        tail_rows: usize,
    ) -> Result<()> {
        let c = self.layers.get(CANDIDATE_SOURCE_LAYER).and_then(|l| l.kv.as_ref()).context("layer 20 not loaded")?;
        ensure!(rows <= c.rows_cap && cand_ld <= self.tail.cand_cap && tail_rows <= REPLAY_ROWS, "seed exceeds the caches");
        gpu.copy_d2d(ckv, c.ckv, rows * HEAD_DIM * 2)?;
        gpu.copy_d2d(ik, c.ik, rows * INDEX_HEAD_DIM * 2)?;
        let mut st = self.st.lock().expect("core state poisoned");
        let cur = st.tail.cur;
        gpu.copy_d2d(cand, self.tail.cand[cur], tail_rows * cand_row_bytes(cand_ld))?;
        gpu.memset(self.tail.topk[cur], 0xFF, REPLAY_ROWS * INDEX_TOPK * 8)?;
        st.tail = TailState { cur, rows: tail_rows, ld: cand_ld };
        st.published = Some((c.ckv, c.ik, rows, c.ratio));
        Ok(())
    }

    fn compress(&self, ops: &Ops, a: &CoreArgs, st: &mut SeqState) -> Result<()> {
        let (layer, t, start, stream) = (a.layer, a.t, a.start, ops.stream);
        let c = self.layers[layer].kv.as_ref().expect("kv-source");
        let gpu = ops.gpu;
        let iops = IndexOps { gpu, k: &self.idx, stream };
        let s = &self.s;
        let r = c.ratio;
        let row = HEAD_DIM * 4;
        self.prof.begin(ops)?;
        let (j0, nj) = if r == 2 {
            let pend = usize::from(st.has_pending[layer]);
            let n = t + pend;
            ensure!(t <= self.max_chunk, "compress: {t} rows exceed max_chunk {}", self.max_chunk);
            if pend == 1 {
                gpu.copy_d2d_async(c.pending, c.before, 2 * row, stream)?; // for rollback
            }
            st.last[layer] = Some((start, t, pend == 1));
            ops.bf16_to_f32(a.x, s.xf, t * HIDDEN)?;
            self.prof.mark(ops, "compress.x_to_f32")?;
            if pend == 1 {
                gpu.copy_d2d_async(c.pending, s.kvl, row, stream)?;
                gpu.copy_d2d_async(c.pending.offset(row), s.sc, row, stream)?;
            }
            let wgate = c.wgate.expect("ratio-2 has a gate");
            if self.comp_cublas_tiled {
                tiled(gpu, s.pad_in, s.pad_out, s.xf, s.kvl.offset(pend * row), t, HIDDEN * 4, row, stream, |x, o| {
                    ops.linear_f32(x, c.wkv, o, MM_TILE, HEAD_DIM, HIDDEN)
                })?;
                tiled(gpu, s.pad_in, s.pad_out, s.xf, s.sc.offset(pend * row), t, HIDDEN * 4, row, stream, |x, o| {
                    ops.linear_f32(x, wgate, o, MM_TILE, HEAD_DIM, HIDDEN)
                })?;
            } else {
                iops.gemm_f32(s.xf, c.wkv, s.kvl.offset(pend * row), t, HEAD_DIM, HIDDEN)?;
                iops.gemm_f32(s.xf, wgate, s.sc.offset(pend * row), t, HEAD_DIM, HIDDEN)?;
            }
            self.prof.mark(ops, if self.comp_cublas_tiled { "compress.gemm_f32_x2.cublas16" } else { "compress.gemm_f32_x2" })?;
            gpu.copy_d2d_async(s.kvl.offset(pend * row), c.save_kv, t * row, stream)?; // for rollback
            gpu.copy_d2d_async(s.sc.offset(pend * row), c.save_sc, t * row, stream)?;
            if n % 2 == 1 {
                gpu.copy_d2d_async(s.kvl.offset((n - 1) * row), c.pending, row, stream)?;
                gpu.copy_d2d_async(s.sc.offset((n - 1) * row), c.pending.offset(row), row, stream)?;
            }
            st.has_pending[layer] = n % 2 == 1;
            iops.combine2(s.kvl, s.sc, s.latent, n / 2, HEAD_DIM)?;
            self.prof.mark(ops, "compress.combine")?;
            ((start - pend) / 2, n / 2)
        } else {
            tiled(gpu, s.pad_in, s.pad_out, a.x, s.latent, t, HIDDEN * 2, HEAD_DIM * 2, stream, |x, o| {
                ops.linear_bf16(x, c.wkv, o, MM_TILE, HEAD_DIM, HIDDEN)
            })?;
            self.prof.mark(ops, "compress.gemm_bf16")?;
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
            self.prof.mark(ops, "compress.norm")?;
            // The index key from the PRE-RoPE latent, straight into the cache.
            let ik = c.ik.offset(j0 * INDEX_HEAD_DIM * 2);
            tiled(gpu, s.pad_in, s.pad_out, s.latent, ik, nj, HEAD_DIM * 2, INDEX_HEAD_DIM * 2, stream, |x, o| {
                ops.linear_bf16(x, c.wk, o, MM_TILE, INDEX_HEAD_DIM, HEAD_DIM)
            })?;
            self.prof.mark(ops, "compress.ik_gemm")?;
            ops.rmsnorm(ik, c.k_norm, ik, nj, INDEX_HEAD_DIM, self.eps)?;
            ops.rope_tail(ik, s.pos, &self.rope_c, nj, 1, INDEX_HEAD_DIM, false)?;
            let ckv = c.ckv.offset(j0 * HEAD_DIM * 2);
            gpu.copy_d2d_async(s.latent, ckv, nj * HEAD_DIM * 2, stream)?;
            ops.rope_tail(ckv, s.pos, &self.rope_c, nj, 1, HEAD_DIM, false)?;
            self.prof.mark(ops, "compress.norm_rope_write")?;
        }
        st.published = Some((c.ckv, c.ik, j0 + nj, r));
        Ok(())
    }

    /// Shape-static form of [`Self::compress`] for Decode/Verify (`ATLAS_DSV41_CORE_STATIC`): the
    /// same kernels on FIXED shapes, the parity, row offsets and row counts read on the device
    /// from `dstart`. The projections always land at slots 1..=t, a pending row at slot 0, and
    /// the pairs start at slot 1 - p; up to `ceil(t / 2)` rows are normalised and RoPE'd, and only
    /// the pass's `(t + p) / 2` are published. Host bookkeeping is the dynamic path's.
    fn compress_static(&self, ops: &Ops, a: &CoreArgs, st: &mut SeqState, dstart: DevicePtr) -> Result<()> {
        let (layer, t, start, stream) = (a.layer, a.t, a.start, ops.stream);
        let c = self.layers[layer].kv.as_ref().expect("kv-source");
        let gpu = ops.gpu;
        let iops = IndexOps { gpu, k: &self.idx, stream };
        let s = &self.s;
        let r = c.ratio;
        let row = HEAD_DIM * 4;
        // The unpublished rows are RoPE'd too: their positions must stay inside the table.
        ensure!(start + t <= self.rope_c.positions, "static compress at {start}+{t} past freqs_c ({})", self.rope_c.positions);
        let nmax = if r == 2 {
            // Every pass is contiguous from 0 (begin_pass), so the pending row exists exactly when
            // start is odd -- the device's p. Derived, not asserted: the skip-restore rollback
            // CONTROL leaves has_pending stale on purpose and must diverge, not stop.
            let pend = start % 2 == 1;
            gpu.copy_d2d_async(c.pending, c.before, 2 * row, stream)?; // for rollback (read only if pend)
            st.last[layer] = Some((start, t, pend));
            ops.bf16_to_f32(a.x, s.xf, t * HIDDEN)?;
            iops.comp2_pending_in(c.pending, s.kvl, s.sc, HEAD_DIM, dstart)?;
            let wgate = c.wgate.expect("ratio-2 has a gate");
            let (kv1, sc1) = (s.kvl.offset(row), s.sc.offset(row));
            if self.comp_cublas_tiled {
                tiled(gpu, s.pad_in, s.pad_out, s.xf, kv1, t, HIDDEN * 4, row, stream, |x, o| {
                    ops.linear_f32(x, c.wkv, o, MM_TILE, HEAD_DIM, HIDDEN)
                })?;
                tiled(gpu, s.pad_in, s.pad_out, s.xf, sc1, t, HIDDEN * 4, row, stream, |x, o| {
                    ops.linear_f32(x, wgate, o, MM_TILE, HEAD_DIM, HIDDEN)
                })?;
            } else {
                iops.gemm_f32(s.xf, c.wkv, kv1, t, HEAD_DIM, HIDDEN)?;
                iops.gemm_f32(s.xf, wgate, sc1, t, HEAD_DIM, HIDDEN)?;
            }
            gpu.copy_d2d_async(kv1, c.save_kv, t * row, stream)?; // for rollback
            gpu.copy_d2d_async(sc1, c.save_sc, t * row, stream)?;
            iops.comp2_pending_out(s.kvl, s.sc, c.pending, HEAD_DIM, t, dstart)?;
            st.has_pending[layer] = (t + usize::from(pend)) % 2 == 1;
            iops.combine2_dev(s.kvl, s.sc, s.latent, t.div_ceil(2), HEAD_DIM, dstart)?;
            t.div_ceil(2)
        } else {
            tiled(gpu, s.pad_in, s.pad_out, a.x, s.latent, t, HIDDEN * 2, HEAD_DIM * 2, stream, |x, o| {
                ops.linear_bf16(x, c.wkv, o, MM_TILE, HEAD_DIM, HIDDEN)
            })?;
            t
        };
        let (j0, nj) = if r == 2 { ((start - start % 2) / 2, (t + start % 2) / 2) } else { (start, t) };
        ensure!(j0 + nj <= c.rows_cap, "layer {layer}: compressed row {} past the cache ({} rows)", j0 + nj, c.rows_cap);
        ops.rmsnorm(s.latent, c.norm, s.latent, nmax, HEAD_DIM, self.eps)?;
        iops.iota_dev(s.pos, dstart, r == 2, r, nmax)?;
        tiled(gpu, s.pad_in, s.pad_out, s.latent, s.iks, nmax, HEAD_DIM * 2, INDEX_HEAD_DIM * 2, stream, |x, o| {
            ops.linear_bf16(x, c.wk, o, MM_TILE, INDEX_HEAD_DIM, HEAD_DIM)
        })?;
        ops.rmsnorm(s.iks, c.k_norm, s.iks, nmax, INDEX_HEAD_DIM, self.eps)?;
        ops.rope_tail(s.iks, s.pos, &self.rope_c, nmax, 1, INDEX_HEAD_DIM, false)?;
        iops.publish_rows(s.iks, c.ik, INDEX_HEAD_DIM * 2, nmax, r, t, dstart)?;
        ops.rope_tail(s.latent, s.pos, &self.rope_c, nmax, 1, HEAD_DIM, false)?;
        iops.publish_rows(s.latent, c.ckv, HEAD_DIM * 2, nmax, r, t, dstart)?;
        st.published = Some((c.ckv, c.ik, j0 + nj, r));
        Ok(())
    }

    /// The device pass start when this pass runs shape-static, else `None`. Without a caller
    /// pointer the core writes its own copy once per pass (fine ungraphed; a graph must use
    /// [`Self::set_device_start`], or the write would be captured with a frozen value).
    fn static_start(&self, ops: &Ops, st: &mut SeqState) -> Result<Option<DevicePtr>> {
        let on = self.static_on.load(std::sync::atomic::Ordering::Relaxed)
            && matches!(st.kind, Some(PassKind::Decode | PassKind::Verify))
            && st.pass_t <= DECODE_MAX_T;
        if !on {
            return Ok(None);
        }
        if let Some(d) = st.dstart_ext {
            return Ok(Some(d));
        }
        if !st.dstart_written {
            let v = u32::try_from(st.pass_start).context("start overflows the device i32")?;
            ops.gpu.memset_u32_async(self.own_dstart, v, 1, ops.stream)?;
            st.dstart_written = true;
        }
        Ok(Some(self.own_dstart))
    }

    /// Turn shape-static Decode/Verify passes on or off (gates A/B both arms on one core).
    pub fn set_static_decode(&self, on: bool) {
        self.static_on.store(on, std::sync::atomic::Ordering::Relaxed);
    }

    /// GRAPH REPLAY: the device i32 holding each pass's start (the position of row 0), written by
    /// the caller BEFORE the core's first launch of the pass and not changed until its last. The
    /// POINTER may be changed between passes (any call; a graph bakes in the one it captured
    /// with), never within a pass. One scalar can serve Decode and Verify alike. `None` = the
    /// core's own copy, written by the core itself (ungraphed runs only).
    pub fn set_device_start(&self, ptr: Option<DevicePtr>) {
        self.st.lock().expect("core state poisoned").dstart_ext = ptr;
    }

    /// GRAPH REPLAY: the host bookkeeping of a static Decode/Verify pass that was REPLAYED rather
    /// than run (the core's `run` is not called): `begin_pass` plus each ratio-2 layer's pending
    /// and rollback record, exactly as `compress_static` leaves them. Launches nothing.
    pub fn replay_step(&self, kind: PassKind, start: usize, t: usize) -> Result<()> {
        ensure!(
            self.static_on.load(std::sync::atomic::Ordering::Relaxed) && matches!(kind, PassKind::Decode | PassKind::Verify) && t <= DECODE_MAX_T,
            "replay_step({kind:?}, {start}, {t}) is not a static pass"
        );
        PassHook::begin_pass(self, kind, start, t)?;
        let mut st = self.st.lock().expect("core state poisoned");
        ensure!(st.dstart_ext.is_some(), "replay_step without a caller-owned device start");
        for (layer, lw) in self.layers.iter().enumerate() {
            // Only layers the forward actually drives (every ratio-2 layer in the model; a gate
            // driving a subset must not gain rollback records for layers it never ran).
            if lw.kv.as_ref().is_some_and(|c| c.ratio == 2) && st.last[layer].is_some() {
                let pend = start % 2 == 1; // as compress_static
                st.last[layer] = Some((start, t, pend));
                st.has_pending[layer] = (t + usize::from(pend)) % 2 == 1;
            }
        }
        Ok(())
    }

    fn index(&self, ops: &Ops, a: &CoreArgs, st: &mut SeqState, dstart: Option<DevicePtr>) -> Result<()> {
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
        if n_c == 0 && dstart.is_none() {
            gpu.memset_async(w.topk, 0xFF, t * INDEX_TOPK * 8, stream)?; // all -1
            st.topk = Some(w.topk);
            return Ok(());
        }
        // Static: one fixed width; the kernels stop at score_width(n_c) for the start they read.
        let n_pad = if dstart.is_some() { self.static_ld } else { score_width(n_c) };
        ensure!(t.min(s.score_rows) * n_pad * 4 <= s.score_bytes, "score [{}, {n_pad}] exceeds the scratch", t.min(s.score_rows));

        // q_i = rope_c(bf16(qr @ dequant(wq_b)^T)); wts = bf16(x @ weights_proj^T) * 1/64
        self.prof.begin(ops)?;
        ops.dequant(&w.wq_b, s.deq)?;
        self.prof.mark(ops, "index.dequant_wq_b")?;
        let qrow = INDEX_HEADS * INDEX_HEAD_DIM * 2;
        tiled(gpu, s.pad_in, s.pad_out, a.qr, s.qi, t, Q_LORA * 2, qrow, stream, |x, o| {
            ops.linear_bf16(x, s.deq, o, MM_TILE, w.wq_b.n, w.wq_b.k)
        })?;
        self.prof.mark(ops, "index.q_gemm")?;
        match dstart {
            Some(d) => iops.iota_dev(s.pos, d, false, 1, t)?,
            None => iops.iota(s.pos, start, 1, t)?,
        }
        ops.rope_tail(s.qi, s.pos, &self.rope_c, t, INDEX_HEADS, INDEX_HEAD_DIM, false)?;
        if self.comp_cublas_tiled {
            tiled(gpu, s.pad_in, s.pad_out, a.x, s.wraw, t, HIDDEN * 2, INDEX_HEADS * 2, stream, |x, o| {
                ops.linear_bf16(x, w.weights_proj, o, MM_TILE, INDEX_HEADS, HIDDEN)
            })?;
        } else {
            iops.gemm_bf16_smalln(a.x, w.weights_proj, s.wraw, t, INDEX_HEADS, HIDDEN)?;
        }
        iops.wts(s.wraw, s.wts, WTS_SCALE, t * INDEX_HEADS)?;
        self.prof.mark(ops, "index.q_rope_wts")?;

        // Candidates prune only the layers AFTER the source (use_cand in model.py).
        let cand_in = match st.cand {
            Some((mask, ld)) if layer > CANDIDATE_SOURCE_LAYER => Some(CandidateMask { mask, ld }),
            _ => None,
        };
        let is_src = layer == CANDIDATE_SOURCE_LAYER;
        if let Some(d) = dstart {
            // Static decode/verify: t <= DECODE_MAX_T <= score_rows, one block.
            iops.score_dev(s.qi, ik, s.wts, cand_in, s.score, t, n_pad, d, ratio)?;
            self.prof.mark(ops, "index.score")?;
            if is_src {
                iops.select_candidates_dev(s.score, s.blocks, s.cand, t, n_pad, d, ratio, CANDIDATE_TOPK_BLOCKS, CANDIDATE_BLOCK_SIZE)?;
                self.prof.mark(ops, "index.candidates")?;
            }
            iops.topk_dev(s.score, w.topk, t, n_pad, d, ratio)?;
        } else {
            // Row blocks of score_rows: score, candidates and top-k are all per row (row r's
            // compress_lens comes from its absolute position start + r0 + r), so blocking changes
            // no byte; it bounds the scratch at long context.
            for r0 in (0..t).step_by(s.score_rows) {
                let r = s.score_rows.min(t - r0);
                let cand_b = cand_in.map(|c| CandidateMask { mask: c.mask.offset(r0 * cand_row_bytes(c.ld)), ld: c.ld });
                iops.score(s.qi.offset(r0 * qrow), ik, s.wts.offset(r0 * INDEX_HEADS * 4), cand_b, s.score, r, rows, n_pad, start + r0, ratio)?;
                self.prof.mark(ops, "index.score")?;
                if is_src {
                    let cand_out = s.cand.offset(r0 * cand_row_bytes(n_pad));
                    iops.select_candidates(s.score, s.blocks, cand_out, r, n_pad, start + r0, ratio, CANDIDATE_TOPK_BLOCKS, CANDIDATE_BLOCK_SIZE)?;
                    self.prof.mark(ops, "index.candidates")?;
                }
                iops.topk(s.score, w.topk.offset(r0 * INDEX_TOPK * 8), r, n_pad, n_c)?;
                if r0 + r < t {
                    self.prof.mark(ops, "index.topk")?;
                }
            }
        }
        if is_src {
            st.cand = Some((s.cand, n_pad));
        }
        st.topk = Some(w.topk);
        self.prof.mark(ops, "index.topk")?;
        if layer == CANDIDATE_SOURCE_LAYER && st.kind == Some(PassKind::EncoderChunk) {
            self.keep_tail(gpu, st, start, t, w.topk, n_pad, stream)?;
            self.prof.mark(ops, "index.keep_tail")?;
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
        // Bit rows (cand_row_bytes): a narrower older row's bits are its leading blocks, and the
        // zeroed rest reads as "not a candidate", exactly as the byte form padded False.
        let (old_rb, new_rb) = (cand_row_bytes(tl.ld), cand_row_bytes(n_pad));
        gpu.memset_async(cd[dst], 0, REPLAY_ROWS * new_rb, stream)?;
        if keep > 0 {
            let from = tl.rows - keep;
            gpu.copy_d2d_async(tk[src].offset(from * INDEX_TOPK * 8), tk[dst], keep * INDEX_TOPK * 8, stream)?;
            gpu.copy_d2d_2d_async(cd[src].offset(from * old_rb), old_rb, cd[dst], new_rb, old_rb, keep, stream)?;
        }
        let from = t - take;
        gpu.copy_d2d_async(topk.offset(from * INDEX_TOPK * 8), tk[dst].offset(keep * INDEX_TOPK * 8), take * INDEX_TOPK * 8, stream)?;
        gpu.copy_d2d_2d_async(self.s.cand.offset(from * new_rb), new_rb, cd[dst].offset(keep * new_rb), new_rb, new_rb, take, stream)?;
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
        let scale = (HEAD_DIM as f32).powf(-0.5);
        if matches!(st.kind, Some(PassKind::Decode | PassKind::Verify)) && a.t <= DECODE_MAX_T && self.split_on {
            let keys = WINDOW + if ratio == 0 { 0 } else { INDEX_TOPK };
            let splits = keys.div_ceil(DECODE_SLICE);
            KernelLaunch::new(ops.gpu, self.split_kernel)
                .grid([a.t as u32, (64 / HEADS_PER_BLOCK) as u32, splits as u32])
                .block([ATTN_THREADS, 1, 1])
                .arg_ptr(a.q)
                .arg_ptr(a.ring)
                .arg_ptr(a.wpos)
                .arg_ptr(ckv)
                .arg_ptr(cidx)
                .arg_ptr(self.part)
                .arg_i32(a.t as i32)
                .arg_i32(64)
                .arg_i32(HEAD_DIM as i32)
                .arg_i32(WINDOW as i32)
                .arg_i32(INDEX_TOPK as i32)
                .arg_i32(RING as i32)
                .arg_i32(a.win_lo as i32)
                .arg_f32(scale)
                .arg_i32(DECODE_SLICE as i32)
                .launch(ops.stream)?;
            return KernelLaunch::new(ops.gpu, self.combine_kernel)
                .grid([a.t as u32, 64, 1])
                .block([ATTN_THREADS, 1, 1])
                .arg_ptr(self.part)
                .arg_ptr(a.sink)
                .arg_ptr(a.out)
                .arg_i32(64)
                .arg_i32(HEAD_DIM as i32)
                .arg_i32(splits as i32)
                .launch(ops.stream);
        }
        let (kernel, heads_per_cta, threads) = match self.mma_kernel {
            Some((k, hpc)) => (k, hpc, (hpc / 16 * 128) as u32),
            None => (self.attn_kernel, HEADS_PER_BLOCK, ATTN_THREADS),
        };
        KernelLaunch::new(ops.gpu, kernel)
            .grid([a.t as u32, (64 / heads_per_cta) as u32, 1])
            .block([threads, 1, 1])
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
            .arg_f32(scale)
            .launch(ops.stream)
    }
}

impl PassHook for Dsv41SparseCore {
    /// DSpark: `V41Forward::rollback` calls this after a Verify pass. NEGATIVE CONTROL ONLY:
    /// `ATLAS_DSV41_CONTROL_SKIP_ROLLBACK_RESTORE=1` rolls the length back without restoring the
    /// ratio-2 compressor carry (the gate must then fail).
    fn rollback(&self, ops: &Ops, n: usize) -> Result<()> {
        let restore = std::env::var("ATLAS_DSV41_CONTROL_SKIP_ROLLBACK_RESTORE").as_deref() != Ok("1");
        self.rollback_inner(ops.gpu, n, restore, ops.stream)
    }

    fn enable_graph(&self, dstart: DevicePtr) -> Result<()> {
        self.set_static_decode(true);
        self.set_device_start(Some(dstart));
        Ok(())
    }

    fn replay_step(&self, kind: PassKind, start: usize, t: usize) -> Result<()> {
        Dsv41SparseCore::replay_step(self, kind, start, t)
    }

    fn begin_pass(&self, kind: PassKind, start: usize, t: usize) -> Result<()> {
        ensure!(t <= self.max_chunk, "pass of {t} rows exceeds max_chunk {}", self.max_chunk);
        let mut st = self.st.lock().expect("core state poisoned");
        if start == 0 && matches!(kind, PassKind::EncoderChunk | PassKind::FullChunk) {
            // A new prompt: nothing is carried. NEVER on Decode or Replay, which continue the
            // sequence -- a reset there would wipe ckv's pending row and the replay tail.
            st.has_pending = vec![false; 40];
            st.last = vec![None; 40];
            st.len = 0;
            st.published = None;
            st.tail = TailState::default();
        }
        let new_prompt = start == 0 && kind == PassKind::EncoderChunk;
        if self.prof.on && (kind == PassKind::Replay || new_prompt || (kind == PassKind::Decode && st.kind != Some(PassKind::Decode))) {
            // Report what the core spent since the last report (the encoder chunks before the
            // replay; the replay before the first decode step).
            let rows = self.prof_take();
            let total: f64 = rows.iter().map(|r| r.1).sum();
            eprintln!("dsv41 core profile before {kind:?} (synchronized per phase): total {total:.1} ms");
            for (k, ms, n) in rows {
                eprintln!("  {k:28} {ms:9.2} ms  {n:5} calls  {:8.3} ms/call", ms / n as f64);
            }
        }
        if kind != PassKind::Replay {
            // Encoder/decode passes continue the sequence exactly where it stands (a rolled-back
            // sequence resumes AT the rollback point, never past it).
            ensure!(start == st.len || start == 0, "pass at {start} but the core holds {} positions", st.len);
            st.len = start + t;
        }
        st.kind = Some(kind);
        st.pass_start = start;
        st.pass_t = t;
        st.dstart_written = false;
        match kind {
            PassKind::Replay => {
                // Layers 21..23 attend with L20's tail selection; 24..36 score inside its
                // candidate pool. Both belong to the QUERIES, carried from the encoder pass.
                ensure!(st.tail.rows == t, "replay over {t} rows but the L20 tail holds {}", st.tail.rows);
                st.topk = Some(self.tail.topk[st.tail.cur]);
                st.cand = Some((self.tail.cand[st.tail.cur], st.tail.ld));
            }
            PassKind::EncoderChunk | PassKind::FullChunk | PassKind::Decode | PassKind::Verify => {
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
        let dstart = self.static_start(ops, &mut st)?;
        if lw.kv.is_some() {
            match dstart {
                Some(d) => self.compress_static(ops, a, &mut st, d)?,
                None => self.compress(ops, a, &mut st)?,
            }
        }
        if lw.idx.is_some() {
            self.index(ops, a, &mut st, dstart)?;
        }
        self.prof.begin(ops)?;
        self.attend(ops, a, &st)?;
        let split = matches!(st.kind, Some(PassKind::Decode | PassKind::Verify)) && a.t <= DECODE_MAX_T && self.split_on;
        self.prof.mark(
            ops,
            match (lw.ratio == 0, split) {
                (true, false) => "attn.window_only",
                (true, true) => "attn.window_only.split",
                (false, false) => if self.mma_kernel.is_some() { "attn.sparse.mma" } else { "attn.sparse" },
                (false, true) => "attn.sparse.split",
            },
        )
    }
}

/// `Dsv41Model` holds the hook and the core in two separate boxes; one shared core serves both:
/// `let c = Arc::new(Dsv41SparseCore::load(..)?); hook: Box::new(c.clone()), core: Box::new(c)`.
impl PassHook for std::sync::Arc<Dsv41SparseCore> {
    fn begin_pass(&self, kind: PassKind, start: usize, t: usize) -> Result<()> {
        self.as_ref().begin_pass(kind, start, t)
    }
    fn rollback(&self, ops: &Ops, n: usize) -> Result<()> {
        PassHook::rollback(self.as_ref(), ops, n)
    }
    fn enable_graph(&self, dstart: DevicePtr) -> Result<()> {
        PassHook::enable_graph(self.as_ref(), dstart)
    }
    fn replay_step(&self, kind: PassKind, start: usize, t: usize) -> Result<()> {
        PassHook::replay_step(self.as_ref(), kind, start, t)
    }
}

impl AttnCore for std::sync::Arc<Dsv41SparseCore> {
    fn run(&self, ops: &Ops, a: &CoreArgs) -> Result<()> {
        self.as_ref().run(ops, a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The model boxes the core as `Box<dyn AttnCore + Send + Sync>`.
    #[test]
    fn the_core_can_be_shared_across_the_models_two_boxes() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Dsv41SparseCore>();
        assert_send_sync::<std::sync::Arc<Dsv41SparseCore>>();
    }

    use spark_runtime::gpu::mock::MockGpuBackend;
    use spark_runtime::weights::WeightTensor;
    use std::collections::HashMap;
    use std::sync::Arc;

    /// A store holding exactly the tensors the core looks up for layers `0..n_layers`, at fake
    /// device addresses (the core never allocates weights, so they do not enter the count).
    fn fake_store(skip_layer: Option<usize>, n_layers: usize) -> WeightStore {
        let mut m = HashMap::new();
        let mut next = 0x7000_0000u64;
        let mut put = |name: String, dtype: WeightDtype, shape: Vec<usize>| {
            next += 0x10_0000;
            m.insert(name, WeightTensor { ptr: DevicePtr(next), shape, dtype });
        };
        for l in (0..n_layers).filter(|l| Some(*l) != skip_layer) {
            let p = |x: &str| format!("layers.{l}.attn.{x}");
            if KV_SOURCE_LAYERS.contains(&l) {
                put(p("compressor.wkv.weight"), WeightDtype::BF16, vec![HEAD_DIM, HIDDEN]);
                if l < 20 {
                    put(p("compressor.wgate.weight"), WeightDtype::BF16, vec![HEAD_DIM, HIDDEN]);
                }
                put(p("compressor.norm.weight"), WeightDtype::BF16, vec![HEAD_DIM]);
                put(p("indexer.wk.weight"), WeightDtype::BF16, vec![INDEX_HEAD_DIM, HEAD_DIM]);
                put(p("indexer.k_norm.weight"), WeightDtype::BF16, vec![INDEX_HEAD_DIM]);
            }
            if INDEX_SOURCE_LAYERS.contains(&l) {
                put(p("indexer.wq_b.weight"), WeightDtype::FP8E4M3, vec![INDEX_HEADS * INDEX_HEAD_DIM, Q_LORA]);
                put(p("indexer.wq_b.scale"), WeightDtype::FP8E8M0, vec![INDEX_HEADS * INDEX_HEAD_DIM / 32, Q_LORA / 32]);
                put(p("indexer.weights_proj.weight"), WeightDtype::BF16, vec![INDEX_HEADS, HIDDEN]);
            }
        }
        WeightStore::from_map(m)
    }

    fn v41_config() -> ModelConfig {
        let mut c = ModelConfig::qwen3_next_80b_nvfp4();
        c.num_attention_heads = 64;
        c.rms_norm_eps = 1e-20;
        c.compress_ratios = [0usize, 0].into_iter().chain(std::iter::repeat(2).take(18)).chain(std::iter::repeat(1).take(20)).collect();
        c
    }

    fn rope() -> RopeTable {
        RopeTable { cos: DevicePtr(0x6000_0000), sin: DevicePtr(0x6800_0000), half: 32, positions: 4096 }
    }

    /// TUI model swap: every device allocation the core makes is freed when it drops.
    /// NEGATIVE CONTROL: a core that is leaked (never dropped) keeps its allocations, so the
    /// count below can fail -- and does, for the leaking arm.
    #[test]
    fn the_core_frees_every_allocation_on_drop() {
        let mock = Arc::new(MockGpuBackend::new());
        let shared: SharedGpu = mock.clone();
        let store = fake_store(None, 40);
        let core = Dsv41SparseCore::load(&shared, &store, &v41_config(), 1024, 64, rope()).unwrap();
        let n = mock.alloc_count();
        assert!(n > 20, "the core allocates its caches, tail and scratch ({n})");
        assert_eq!(core.allocation_count(), n, "every live allocation is recorded by the core's guard");
        drop(core);
        assert_eq!(mock.alloc_count(), 0, "dropping the core must free everything it allocated");

        let leaked = Dsv41SparseCore::load(&shared, &store, &v41_config(), 1024, 64, rope()).unwrap();
        std::mem::forget(leaked); // CONTROL: the leak the swap test must be able to see
        assert_eq!(mock.alloc_count(), n, "a leaked core keeps all {n} allocations");
    }

    /// A load that fails half-way (layer 14's tensors missing, after layers 2 and 8 allocated
    /// their caches) frees what it already took.
    #[test]
    fn a_failed_load_frees_its_partial_allocations() {
        let mock = Arc::new(MockGpuBackend::new());
        let shared: SharedGpu = mock.clone();
        let err = Dsv41SparseCore::load(&shared, &fake_store(Some(14), 40), &v41_config(), 1024, 64, rope());
        assert!(err.is_err(), "layer 14 has no tensors, the load must fail");
        assert_eq!(mock.alloc_count(), 0, "the half-built core must free layers 2/8's caches");
    }

    /// Score scratch sizing: n_pad rounds max_seq up to 512 columns.
    #[test]
    fn score_scratch_is_one_chunk_by_the_widest_row() {
        assert_eq!(score_scratch_bytes(512, 8192), 512 * 8192 * 4);
        assert_eq!(score_scratch_bytes(512, 8193), 512 * 8704 * 4);
        // 1M context: row blocks, not the chunk -- under the 1 GiB budget at any chunk.
        assert_eq!(score_scratch_bytes(3968, 1 << 20), 224 * (1 << 20) * 4);
    }
}
