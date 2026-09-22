// SPDX-License-Identifier: AGPL-3.0-only

//! The DeepSeek-V4.1 model forward: embedding -> 40 blocks -> head, in the reference's
//! prefill shape, **CED + Decoder SWA Bounded Replay** (lead ruling; tech report §2.2/§3.2.2,
//! `engine/model.py::forward(encoder_only=True)` + `decoder_replay`):
//!
//! ```text
//! prefill, per chunk:  layers 0..=20 over the chunk      (the ENCODER: everything that
//!                                                         writes global KV)
//! prefill, once:       layers 21..=39 over the LAST min(128, P) prompt tokens, window
//!                      truncated to that segment (win_lo = P - n), reusing L20's
//!                      top-k / candidate rows for those tokens            (the REPLAY)
//! decode:              all 40 layers, one token
//! ```
//!
//! The replay is exact for the prompt's last row and is what the production engine serves;
//! a full-prompt decoder is a debug mode only (`PrefillMode::Full`).

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::WeightStore;

use super::attn_block::{self, AttnCore, AttnScratch, HEAD_DIM, RING, V41AttnWeights, WINDOW, compress_ratio};
use super::fwd::{BlockControl, PassScratch, Tap, V41AttentionBlock, V41BlockWeights, V41Dims, V41RoutedMoe, block};
use super::ops::{Ops, RopeSpec, RopeTable, bf16_tensor, bytemuck_u32, prof};
use crate::layers::deepseek_v41_engram::{
    EngramGather, EngramHashState, N_HEAD_COLS, engram_dead_heads, engram_dead_heads_with_carry,
    update_dead_carry,
};

/// Set to dump `engram_in` (the block's `h` going INTO the engram projection),
/// `engram_rows_masked` (the post-mask, cast-to-bf16 rows the wkv linear
/// actually reads) alongside the existing `engram_out` tap, and to log the
/// masked-head count per chunk from the EXECUTING path (host-side, from the
/// same buffer that gets uploaded to the device -- not a separate recompute).
/// Off by default: the extra taps cost a D2H copy nobody wants paying for on
/// every run.
pub const ENGRAM_DEBUG_ENV: &str = "ATLAS_DSV41_ENGRAM_DEBUG";

fn engram_debug() -> bool {
    std::env::var(ENGRAM_DEBUG_ENV).is_ok()
}

/// `candidate_source_layer`: the last encoder layer.
pub const ENCODER_LAST: usize = 20;
pub const N_LAYERS: usize = 40;

/// Which pass a core call belongs to. The attention lane's state machine keys on it: the
/// replay must install L20's tail selection before layer 21 and must not recompute it on
/// the inheriting layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PassKind {
    /// Layers 0..=20 over one prompt chunk.
    EncoderChunk,
    /// Layers 21..=39 over the last min(128, P) prompt rows.
    Replay,
    /// One generated token through all 40 layers.
    Decode,
    /// Debug: all 40 layers over a whole prompt chunk (no replay).
    FullChunk,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrefillMode {
    /// THE prefill path.
    Replay,
    /// Debug only: every layer over every prompt token.
    Full,
}

/// Per-pass hook for the attention lane, called once before the first layer of a pass.
pub trait PassHook {
    fn begin_pass(&self, kind: PassKind, start: usize, t: usize) -> Result<()>;
}

/// Per-sequence state this module owns. (ckv / ik / pending belong to the attention lane.)
pub struct V41Seq {
    /// One window ring per layer, `[RING, 512]` bf16, slot = pos % RING.
    pub rings: Vec<DevicePtr>,
    pub hash: EngramHashState,
    /// Positions absorbed so far.
    pub len: usize,
    /// The last min(128, len) rows of the ENCODER output stream and its pre_mix, carried
    /// across chunks for the replay (`Model._rep_keep`).
    pub tail_h: DevicePtr,
    pub tail_pre: DevicePtr,
    pub tail_rows: usize,
    /// Token ids of those tail rows (the replay's MoE routes by them).
    pub tail_ids: Vec<u32>,
    /// The trailing up-to-`MAX_LOOKBACK` raw ids from the most recently processed
    /// chunk/token, carried so `engram_dead_heads_with_carry` sees look-back
    /// across a chunk boundary the way `engine/v41_engine.py:755` does (it hashes
    /// the WHOLE image-expanded prompt once, then slices per chunk) rather than
    /// `engine/model.py:747`'s local fallback, which only applies to decode/text
    /// (dsv41-parity, 2026-09-22).
    pub dead_carry: Vec<u32>,
}

impl V41Seq {
    pub fn new(gpu: &dyn GpuBackend, dims: &V41Dims, hash: EngramHashState) -> Result<Self> {
        let rings = (0..N_LAYERS)
            .map(|_| {
                let r = gpu.alloc(RING * HEAD_DIM * 2)?;
                gpu.memset(r, 0, RING * HEAD_DIM * 2)?;
                Ok(r)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            rings,
            hash,
            len: 0,
            tail_h: gpu.alloc(WINDOW * dims.hc * dims.hidden * 2)?,
            tail_pre: gpu.alloc(WINDOW * dims.hc * 4)?,
            tail_rows: 0,
            tail_ids: Vec::new(),
            dead_carry: Vec::new(),
        })
    }

    pub fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        for r in self.rings {
            gpu.free(r)?;
        }
        gpu.free(self.tail_h)?;
        gpu.free(self.tail_pre)
    }
}

/// Everything resident for the forward, independent of any one sequence.
pub struct V41Forward {
    pub dims: V41Dims,
    pub vocab: usize,
    pub blocks: Vec<V41BlockWeights>,
    pub attn: Vec<V41AttnWeights>,
    pub embed: DevicePtr,
    pub norm: DevicePtr,
    pub head: DevicePtr,
    pub freqs_c: RopeTable,
    pub freqs_w: RopeTable,
    pub scratch: PassScratch,
    pub attn_scratch: AttnScratch,
    pub engram: Vec<(usize, EngramGather)>,
    pub ids_dev: DevicePtr,
    pub max_chunk: usize,
    pub max_seq: usize,
}

pub fn rope_specs() -> (RopeSpec, RopeSpec) {
    let w = RopeSpec { dim: 64, original_seq_len: 0, base: 10000.0, factor: 16.0, beta_fast: 32.0, beta_slow: 1.0 };
    let c = RopeSpec { original_seq_len: 65536, base: 160000.0, ..w };
    (c, w)
}

impl V41Forward {
    /// `n_layers < 40` is for the driver's partial runs; serving always passes 40.
    pub fn load(
        store: &WeightStore,
        ops: &Ops,
        dims: V41Dims,
        vocab: usize,
        n_layers: usize,
        max_chunk: usize,
        max_seq: usize,
        model_dir: &std::path::Path,
        engram_threads: usize,
    ) -> Result<Self> {
        let blocks = (0..n_layers).map(|l| V41BlockWeights::load(store, l, &dims, ops)).collect::<Result<Vec<_>>>()?;
        let attn = (0..n_layers).map(|l| V41AttnWeights::load(store, l, dims.hidden)).collect::<Result<Vec<_>>>()?;
        let largest = blocks
            .iter()
            .flat_map(|b| [b.shared.w1, b.shared.w2, b.shared.w3].into_iter().chain(b.engram.as_ref().map(|e| e.wkv)))
            .map(|w| w.n * w.k)
            .chain(attn.iter().map(|a| a.largest_weight()))
            .max()
            .unwrap_or(0);
        let (c, w) = rope_specs();
        // FP8 dense GEMMs at M > 16 are issued at ONE M for every chunk (see ops::fp8_fixed_m);
        // every activation buffer below holds tiled_rows(max(max_chunk, 128)) rows.
        super::ops::set_fp8_fixed_m(super::ops::tiled_rows(max_chunk.max(WINDOW)));
        let engram = blocks
            .iter()
            .filter(|b| b.engram.is_some())
            .map(|b| Ok((b.layer, EngramGather::open(model_dir, b.layer, engram_threads)?)))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            dims,
            vocab,
            embed: bf16_tensor(store, "embed.weight", &[vocab, dims.hidden])?,
            norm: bf16_tensor(store, "norm.weight", &[dims.hidden])?,
            head: bf16_tensor(store, "head.weight", &[vocab, dims.hidden])?,
            freqs_c: c.upload(ops.gpu, max_seq + 8)?,
            freqs_w: w.upload(ops.gpu, max_seq + 8)?,
            scratch: PassScratch::new(ops.gpu, &dims, max_chunk.max(WINDOW), largest)?,
            attn_scratch: AttnScratch::new(ops.gpu, max_chunk.max(WINDOW))?,
            ids_dev: ops.gpu.alloc(super::ops::tiled_rows(max_chunk.max(WINDOW)) * 4)?,
            blocks,
            attn,
            engram,
            max_chunk,
            max_seq,
        })
    }

    fn rope(&self, layer: usize) -> &RopeTable {
        if compress_ratio(layer) != 0 { &self.freqs_c } else { &self.freqs_w }
    }

    /// Layers `layers` over `t` rows at absolute `start`, in place on `scratch.h/pre_mix`.
    #[allow(clippy::too_many_arguments)]
    fn run_layers(
        &self,
        ops: &Ops,
        seq: &mut V41Seq,
        layers: std::ops::Range<usize>,
        t: usize,
        start: usize,
        win_lo: usize,
        hashes: Option<&[i64]>,
        core: &dyn AttnCore,
        moe: &dyn V41RoutedMoe,
        tap: &Tap,
    ) -> Result<()> {
        let s = &self.scratch;
        for l in layers {
            let w = &self.blocks[l];
            if let Some(e) = &w.engram {
                let all = hashes.context("engram layer reached without hashes")?;
                let li = if l == 1 { 0 } else { 1 };
                let rows: Vec<i64> = (0..t).flat_map(|tok| all[(tok * 2 + li) * 24..(tok * 2 + li + 1) * 24].iter().copied()).collect();
                let g = &self.engram.iter().find(|(el, _)| *el == l).context("no engram gather")?.1;
                prof(ops, "engram.gather", || g.gather_rows_gpu(&rows, t, s.engram_rows, ops.gpu, ops.stream))?;
                let debug = engram_debug();
                if debug {
                    tap.bf16(ops, "engram_in", l, s.h, &[t, self.dims.hc, self.dims.hidden])?;
                    tap.f32(ops, "engram_rows_premask", l, s.engram_rows, &[t, N_HEAD_COLS, 256])?;
                }
                // `pass` uploaded this chunk's dead-head mask into `s.engram_dead`.
                prof(ops, "engram.proj", || e.forward(ops, s.h, s.engram_rows, s.engram_dead, t, s, &self.dims))?;
                if debug {
                    // The mask+cast the wkv linear actually reads, AFTER dsv41_engram_rows_bf16
                    // runs -- if this doesn't differ from engram_rows_premask on a chunk the
                    // executing mask log says has masked cells, the mask isn't reaching the GEMM.
                    tap.bf16(ops, "engram_rows_masked", l, s.engram_rows_bf16, &[t, N_HEAD_COLS, 256])?;
                }
                tap.bf16(ops, "engram_out", l, s.h, &[t, self.dims.hc, self.dims.hidden])?;
            }
            let adapter = AttnAdapter { fwd: self, ring: seq.rings[l], win_lo, core, tap };
            block(ops, w, &self.dims, s, t, start, &adapter, moe, tap, BlockControl::None)?;
        }
        Ok(())
    }

    /// Embed `ids` into the stream, hash them for the engram, run `layers`.
    #[allow(clippy::too_many_arguments)]
    fn pass(
        &self,
        ops: &Ops,
        seq: &mut V41Seq,
        ids: &[u32],
        start: usize,
        layers: std::ops::Range<usize>,
        kind: PassKind,
        hook: &dyn PassHook,
        core: &dyn AttnCore,
        moe: &dyn V41RoutedMoe,
        tap: &Tap,
    ) -> Result<()> {
        let t = ids.len();
        ensure!(seq.len == start, "sequence holds {} positions, pass starts at {start}", seq.len);
        ensure!(start + t <= self.max_seq, "position {} exceeds max_seq {}", start + t, self.max_seq);
        let hashes = seq.hash.forward(ids, start, None)?;
        // PREFILL (`engine/v41_engine.py:755`) hashes the WHOLE image-expanded prompt once and
        // slices per chunk -- carrying the trailing MAX_LOOKBACK raw ids across chunks is the
        // exact equivalent (see deepseek_v41_engram::dead_heads's module doc). DECODE does NOT
        // carry: `m.forward(block, pos, prefill=False)` (v41_engine.py:912, 1037) passes no
        // dead_heads, so `model.py:746-747`'s fallback computes it fresh from that call's own
        // (single-token) ids with no history. Corrected 2026-09-22 by dsv41-parity: an earlier
        // version of this carried on decode too, which is wrong at exactly the position right
        // after an image-ending prompt's first generated token. Replay never reaches an engram
        // layer, so it never exercises this branch either way.
        let dead: Vec<u8> = match kind {
            PassKind::Decode => engram_dead_heads(ids),
            _ => engram_dead_heads_with_carry(&seq.dead_carry, ids),
        }
        .into_iter()
        .map(u8::from)
        .collect();
        if kind != PassKind::Decode {
            update_dead_carry(&mut seq.dead_carry, ids);
        }
        if engram_debug() {
            // Straight off the host buffer this chunk is about to upload -- proves the mask
            // reaches the point of upload, not merely that it was computed somewhere upstream.
            let masked_cells = dead.iter().filter(|&&d| d != 0).count();
            let masked_positions = dead.chunks(N_HEAD_COLS).filter(|row| row.iter().any(|&d| d != 0)).count();
            eprintln!(
                "ENGRAM_DEBUG pass start={start} t={t}: masked_cells={masked_cells}/{} masked_positions={masked_positions}/{t}",
                dead.len()
            );
        }
        ops.gpu.copy_h2d_async(&dead, self.scratch.engram_dead, ops.stream)?;
        ops.gpu.copy_h2d_async(bytemuck_u32(ids), self.ids_dev, ops.stream)?;
        ops.embed(self.embed, self.ids_dev, self.scratch.x, t, self.dims.hidden)?;
        ops.hc_expand(self.scratch.x, self.scratch.h, self.scratch.pre_mix, t, self.dims.hidden)?;
        hook.begin_pass(kind, start, t)?;
        moe.begin_pass(ids)?;
        self.run_layers(ops, seq, layers, t, start, 0, Some(&hashes), core, moe, tap)?;
        seq.len = start + t;
        Ok(())
    }

    /// Keep the last `min(128, …)` encoder output rows across chunks (`_rep_keep`).
    fn keep_tail(&self, ops: &Ops, seq: &mut V41Seq, ids: &[u32]) -> Result<()> {
        let t = ids.len();
        seq.tail_ids.extend_from_slice(ids);
        let drop = seq.tail_ids.len().saturating_sub(WINDOW);
        seq.tail_ids.drain(..drop);
        let (hc, d) = (self.dims.hc, self.dims.hidden);
        let (hrow, prow) = (hc * d * 2, hc * 4);
        let take = t.min(WINDOW);
        let keep_old = (WINDOW - take).min(seq.tail_rows);
        if keep_old > 0 && keep_old < seq.tail_rows {
            // shift the newest `keep_old` old rows to the front (non-overlapping when
            // keep_old <= tail_rows - keep_old; otherwise go through the x scratch)
            let from = seq.tail_rows - keep_old;
            let tmp_h = self.scratch.engram_kv;
            ops.gpu.copy_d2d_async(seq.tail_h.offset(from * hrow), tmp_h, keep_old * hrow, ops.stream)?;
            ops.gpu.copy_d2d_async(tmp_h, seq.tail_h, keep_old * hrow, ops.stream)?;
            let tmp_p = self.scratch.engram_rows;
            ops.gpu.copy_d2d_async(seq.tail_pre.offset(from * prow), tmp_p, keep_old * prow, ops.stream)?;
            ops.gpu.copy_d2d_async(tmp_p, seq.tail_pre, keep_old * prow, ops.stream)?;
        }
        let src = t - take;
        ops.gpu.copy_d2d_async(self.scratch.h.offset(src * hrow), seq.tail_h.offset(keep_old * hrow), take * hrow, ops.stream)?;
        ops.gpu.copy_d2d_async(self.scratch.pre_mix.offset(src * prow), seq.tail_pre.offset(keep_old * prow), take * prow, ops.stream)?;
        seq.tail_rows = keep_old + take;
        Ok(())
    }

    /// One prompt chunk at `seq.len`. Replay mode runs only the encoder (layers 0..=20) and
    /// keeps the stream's tail; Full mode runs every layer. Call [`Self::finish_prefill`]
    /// after the LAST chunk.
    #[allow(clippy::too_many_arguments)]
    pub fn prefill_chunk(
        &self,
        ops: &Ops,
        seq: &mut V41Seq,
        chunk: &[u32],
        mode: PrefillMode,
        hook: &dyn PassHook,
        core: &dyn AttnCore,
        moe: &dyn V41RoutedMoe,
        tap: &Tap,
    ) -> Result<()> {
        ensure!(!chunk.is_empty() && chunk.len() <= self.max_chunk, "chunk of {} (max {})", chunk.len(), self.max_chunk);
        let start = seq.len;
        if start == 0 {
            seq.tail_rows = 0;
            seq.tail_ids.clear();
            // Robustness (dsv41-parity): V41Seq is rebuilt per request today
            // (Dsv41Model::alloc_sequence), so this is currently unreachable with a nonempty
            // carry -- but if a sequence slot is ever reused, a stale dead_carry from the
            // PREVIOUS request would otherwise silently leak into this request's first chunk.
            seq.dead_carry.clear();
        }
        let n = self.blocks.len();
        match mode {
            PrefillMode::Replay => {
                let enc = 0..(ENCODER_LAST + 1).min(n);
                self.pass(ops, seq, chunk, start, enc, PassKind::EncoderChunk, hook, core, moe, tap)?;
                self.keep_tail(ops, seq, chunk)
            }
            PrefillMode::Full => {
                self.pass(ops, seq, chunk, start, 0..n, PassKind::FullChunk, hook, core, moe, tap)?;
                seq.tail_rows = chunk.len(); // rows of the last chunk live in scratch.h
                Ok(())
            }
        }
    }

    /// After the last prompt chunk: the decoder replay (Replay mode), then bf16 logits
    /// `[vocab]` of the prompt's last row.
    #[allow(clippy::too_many_arguments)]
    pub fn finish_prefill(
        &self,
        ops: &Ops,
        seq: &mut V41Seq,
        mode: PrefillMode,
        hook: &dyn PassHook,
        core: &dyn AttnCore,
        moe: &dyn V41RoutedMoe,
        tap: &Tap,
        logits: DevicePtr,
    ) -> Result<()> {
        let n = self.blocks.len();
        let t = seq.tail_rows;
        ensure!(t > 0, "finish_prefill before any prompt chunk");
        if mode == PrefillMode::Replay && n > ENCODER_LAST + 1 {
            // decoder_replay: layers 21..39 over the tail, window truncated to it.
            let start = seq.len - t;
            let (hrow, prow) = (self.dims.hc * self.dims.hidden * 2, self.dims.hc * 4);
            ops.gpu.copy_d2d_async(seq.tail_h, self.scratch.h, t * hrow, ops.stream)?;
            ops.gpu.copy_d2d_async(seq.tail_pre, self.scratch.pre_mix, t * prow, ops.stream)?;
            hook.begin_pass(PassKind::Replay, start, t)?;
            ensure!(seq.tail_ids.len() == t, "replay tail holds {} ids for {t} rows", seq.tail_ids.len());
            moe.begin_pass(&seq.tail_ids)?;
            self.run_layers(ops, seq, (ENCODER_LAST + 1)..n, t, start, start, None, core, moe, tap)?;
        }
        prof(ops, "head", || super::fwd::final_logits_last_row(ops, &self.dims, &self.scratch, t, self.norm, self.head, self.vocab, logits))
    }

    /// The whole prompt: chunks of `max_chunk`, then [`Self::finish_prefill`].
    #[allow(clippy::too_many_arguments)]
    pub fn prefill(
        &self,
        ops: &Ops,
        seq: &mut V41Seq,
        ids: &[u32],
        mode: PrefillMode,
        hook: &dyn PassHook,
        core: &dyn AttnCore,
        moe: &dyn V41RoutedMoe,
        tap: &Tap,
        logits: DevicePtr,
    ) -> Result<()> {
        ensure!(!ids.is_empty(), "empty prompt");
        for chunk in ids.chunks(self.max_chunk) {
            self.prefill_chunk(ops, seq, chunk, mode, hook, core, moe, tap)?;
        }
        self.finish_prefill(ops, seq, mode, hook, core, moe, tap, logits)
    }

    /// One decode step at position `seq.len`; bf16 logits `[vocab]`.
    #[allow(clippy::too_many_arguments)]
    pub fn decode(
        &self,
        ops: &Ops,
        seq: &mut V41Seq,
        token: u32,
        hook: &dyn PassHook,
        core: &dyn AttnCore,
        moe: &dyn V41RoutedMoe,
        tap: &Tap,
        logits: DevicePtr,
    ) -> Result<()> {
        let start = seq.len;
        self.pass(ops, seq, &[token], start, 0..self.blocks.len(), PassKind::Decode, hook, core, moe, tap)?;
        prof(ops, "head", || super::fwd::final_logits_last_row(ops, &self.dims, &self.scratch, 1, self.norm, self.head, self.vocab, logits))
    }
}

/// `V41AttentionBlock` for one layer's call: the projections here, the core from the lane.
struct AttnAdapter<'a> {
    fwd: &'a V41Forward,
    ring: DevicePtr,
    win_lo: usize,
    core: &'a dyn AttnCore,
    tap: &'a Tap,
}

impl V41AttentionBlock for AttnAdapter<'_> {
    fn forward(&self, ops: &Ops, layer: usize, x: DevicePtr, out: DevicePtr, t: usize, start: usize) -> Result<()> {
        attn_block::attention(
            ops,
            &self.fwd.attn[layer],
            &self.fwd.attn_scratch,
            self.fwd.scratch.wscratch,
            self.fwd.rope(layer),
            self.ring,
            x,
            out,
            t,
            start,
            self.win_lo,
            self.fwd.dims.norm_eps,
            self.core,
            self.tap,
        )
    }
}
