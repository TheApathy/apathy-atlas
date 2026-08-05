// SPDX-License-Identifier: AGPL-3.0-only

//! Per-call context for multi-sequence batched decode. Bundles the
//! shared scalars and buffer pointers that every phase reads.

use spark_runtime::gpu::DevicePtr;

use crate::layer::{ForwardContext, TreeReseed};
use crate::layers::qwen3_attention::Qwen3AttentionLayer;

/// DDTree: the split-cache-write instruction for a batched tree-verify
/// forward. Rows `[0, spine_rows)` are the bonus + spine and write their K/V
/// into canonical blocks; rows `[spine_rows, n)` are branch rows and write
/// into scratch blocks, after `reseed` has copied this layer's canonical
/// blocks into that scratch.
///
/// `None` (the default) means the ordinary flat verify: one cache write over
/// all `n` rows, no re-seed — bit-identical to the pre-DDTree path.
#[derive(Clone, Copy)]
pub(super) struct TreeSplit<'a> {
    pub spine_rows: usize,
    pub reseed: &'a TreeReseed<'a>,
}

/// Shared scalars / buffer pointers for `super::decode_multi_seq_inner`.
/// Built once in the orchestrator, then handed to each phase by `&self`.
#[allow(dead_code)]
pub(super) struct MultiSeqCtx<'a> {
    /// Wrapping forward-pass context (kernels, buffers, gpu, config…).
    pub fwd: &'a ForwardContext<'a>,
    /// Per-token hidden state buffer (input).
    pub hidden: DevicePtr,
    /// Per-token residual buffer (FP32 or BF16 depending on config).
    pub residual: DevicePtr,
    /// Number of sequences in this batched decode.
    pub n: usize,
    /// CUDA stream.
    pub stream: u64,

    // Cached config / per-layer scalars.
    pub h: usize,
    pub nq: u32,
    pub nkv: u32,
    pub hd: u32,
    pub eps: f32,
    pub bs: u32,
    pub bf16: usize,
    pub q_dim: u32,
    pub q_proj_dim: u32,
    pub q_proj_bytes: usize,
    pub per_seq_qkv: usize,

    // Buffer pointers used by ≥2 phases.
    /// Buffer holding RMS-normed hidden (output of phase 1, input of QKV).
    pub normed: DevicePtr,
    /// Per-token concatenated QKV output [Q | K | V] strided by `per_seq_qkv`.
    pub qkv_buf: DevicePtr,
    /// M2 per-request LoRA routing: `[n]` i32 adapter-slot buffer for this
    /// batched step (from `AttnMetadataDev.seq_slot`; `DevicePtr(0)` = no
    /// routing). Set by the orchestrator after fetching `meta`. Read by the
    /// K/V/O bgmv apply sites; `0` (or a base-only route) makes them no-op.
    pub seq_slot: DevicePtr,
    /// ATLAS_FUSED_ELEMWISE=1 flat-verify epilogue: when true the wide (n>3)
    /// QKV branches SKIP the per-row scatter into `qkv_buf`, the per-head
    /// norms, per-row rope, per-row cache write and the Q gather — a single
    /// fused kernel (bit-identical) consumes the contiguous GEMM outputs
    /// instead. Set ONLY by `decode_multi_seq_inner` when
    /// [`Qwen3AttentionLayer::ms_fused_epilogue_eligible`] holds; defaults to
    /// false so the tree-verify and mHC paths are untouched.
    pub fused_qk_epilogue: bool,
    /// DDTree split cache write, armed ONLY by
    /// [`Qwen3AttentionLayer::decode_multi_seq_tree_inner`]. Every other
    /// entry point leaves it `None`, so the flat verify is untouched.
    pub tree: Option<TreeSplit<'a>>,
    /// Absolute position of row 0, when this batch is a contiguous γ-verify
    /// window over ONE sequence (host `seq_lens[r] == seq_lens[0] + r`).
    /// `None` for a genuine multi-sequence co-batch, where the rows are
    /// unrelated streams and no single compressor frontier applies.
    ///
    /// Read only by the V4 compressed-arm speculation
    /// (`v4_compress_speculate`), which needs a HOST-side position: the
    /// `AttnMetadataDev` positions live on device, and the compressor's window
    /// bookkeeping is host logic.
    pub verify_base_pos: Option<usize>,
}

impl<'a> MultiSeqCtx<'a> {
    pub(super) fn new(
        layer: &Qwen3AttentionLayer,
        fwd: &'a ForwardContext<'a>,
        hidden: DevicePtr,
        residual: DevicePtr,
        n: usize,
        bs: u32,
        stream: u64,
    ) -> Self {
        let h = fwd.config.hidden_size;
        let nq = layer
            .num_q_heads_override
            .unwrap_or(fwd.config.num_attention_heads) as u32;
        let nkv = layer
            .num_kv_heads_override
            .unwrap_or(fwd.config.num_key_value_heads) as u32;
        let hd = layer.head_dim_override.unwrap_or(fwd.config.head_dim) as u32;
        let eps = fwd.config.rms_norm_eps as f32;
        let bf16 = 2usize;
        let q_dim = nq * hd;
        let q_proj_dim = if layer.gated { q_dim * 2 } else { q_dim };
        let q_proj_bytes = q_proj_dim as usize * bf16;
        let per_seq_qkv = q_proj_bytes + (nkv * hd) as usize * bf16 * 2;
        let normed = fwd.buffers.norm_output();
        let qkv_buf = fwd.buffers.qkv_output();
        Self {
            fwd,
            hidden,
            residual,
            n,
            stream,
            h,
            nq,
            nkv,
            hd,
            eps,
            bs,
            bf16,
            q_dim,
            q_proj_dim,
            q_proj_bytes,
            per_seq_qkv,
            normed,
            qkv_buf,
            seq_slot: DevicePtr(0),
            fused_qk_epilogue: false,
            tree: None,
            verify_base_pos: None,
        }
    }
}

/// Recognise a contiguous single-sequence γ-verify window from the HOST
/// `seq_lens` the batched decode was called with, returning row 0's absolute
/// FORWARD position.
///
/// The verify builder emits `seq_lens[r] = seq.seq_len + r` (`verify_d.rs`),
/// where `seq.seq_len` COUNTS the last emitted-but-not-yet-forwarded token —
/// so row 0's forward position is `seq_lens[0] - 1`, not `seq_lens[0]`.
/// Measured (task #45, the compressed-arm divergence): plain decode appends
/// token X's normed at ring slot `pos % ratio` with `pos = seq_len_at_decode
/// = seq_lens[0] - 1`; a replay based at `seq_lens[0]` wrote every row one
/// slot late and left one slot holding stale bytes, so every replayed pool
/// block differed from plain's ([12,13,residue,row0] vs [12,13,row0,row1]).
///
/// Requiring the exact `+r` stride is what distinguishes a γ window from a
/// genuine multi-sequence co-batch, where advancing one compressor frontier
/// over all rows would be meaningless.
pub(super) fn verify_base_pos_of(seq_lens: &[usize], n: usize) -> Option<usize> {
    if n == 0 || seq_lens.len() < n {
        return None;
    }
    let base = seq_lens[0];
    if base == 0 {
        return None;
    }
    seq_lens[..n]
        .iter()
        .enumerate()
        .all(|(r, &l)| l == base + r)
        .then_some(base - 1)
}
