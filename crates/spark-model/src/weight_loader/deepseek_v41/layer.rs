// SPDX-License-Identifier: AGPL-3.0-only

//! The DeepSeek-V4.1 transformer layer.
//!
//! Holds everything a V4.1 layer needs and is CONSTRUCTIBLE today: both norms, the router
//! (weight + both biases), the FP8 shared expert, the per-layer mHC tensors, a handle into
//! the resident CB3 arena, and the attention / engram seams.
//!
//! ## What the forward does NOT do, and why it says so instead of doing it
//! `decode` and `prefill` hard-stop. Three pieces are genuinely absent — attention
//! (`dsv41-attention`), engram (`dsv41-engram`), and the routing + combine that turns the
//! router's logits into a weighted sum over experts (mine, not done). The per-expert
//! compute those would drive IS built and shape-checked (`super::moe`), and the weights
//! are resident, but a layer that ran with any of the three stubbed would emit fluent,
//! wrong tokens and no error. That is the exact failure this port has spent the week
//! refusing, so the forward refuses instead.
//!
//! Constructing the layer is nevertheless the milestone it looks like: it is what proves
//! the 71.7 GB pack is addressable, the weight names resolve, and the geometry agrees.

use anyhow::Result;
use std::sync::{Arc, Mutex};


use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::PagedKvCache;

use crate::layer::{EmptyLayerState, ForwardContext, LayerState, TransformerLayer};
use crate::weight_map::DenseWeight;

use super::cb3_arena::Cb3ExpertArena;
use super::moe::{Cb3Matrix, Cb3Reconstruct};
use super::seams::{Dsv41Attention, Dsv41Engram, SparseShared};

/// The router. V4.1 carries **two** biases, which is not a duplicate.
///
/// `gate.bias` is the noaux_tc correction bias applied to the routing scores. `gate.bias_vl`
/// is a SEPARATE vision-language bias the multimodal path selects instead. Loading one as
/// the other would shift every routing score by a constant vector — a change that reorders
/// top-k near ties and shows up as slightly-different text, never as an error. Both are
/// held, and which one applies is a forward-path decision, not a load-time one.
pub struct V41Router {
    pub weight: DenseWeight,
    pub bias: DenseWeight,
    pub bias_vl: DenseWeight,
}

/// The always-on shared expert. **FP8 block-quantised in the main shards, NOT CB3.**
///
/// The routed experts live in the `k154-cb3` pack; this one does not, and it is dense and
/// always active. Decoding it with the CB3 path would read FP8 bytes as CB3 planes.
pub struct V41SharedExpert {
    pub w1: DenseWeight,
    pub w1_scale: DenseWeight,
    pub w2: DenseWeight,
    pub w2_scale: DenseWeight,
    pub w3: DenseWeight,
    pub w3_scale: DenseWeight,
}

/// Per-layer mHC tensors.
///
/// V4-Flash-0731 has ONE model-level `hc_head` replicated to every layer. V4.1 has
/// **per-layer** `hc_attn_*` and `hc_ffn_*`, on all 40 layers — counted against the index:
/// 43 `hc_attn_fn` tensors, being 40 layers plus 3 MTP modules. Modelling this as a
/// model-level constant, as the V4 loader does, would share one layer's coefficients
/// across all forty.
///
/// `attn` is an `Option` only because the MTP modules may differ; for the main stack it is
/// always `Some`, and `load_layers` builds it unconditionally.
pub struct V41HyperConnections {
    pub ffn_fn: DenseWeight,
    pub ffn_base: DenseWeight,
    pub ffn_scale: DenseWeight,
    pub attn: Option<V41HcAttn>,
}

pub struct V41HcAttn {
    pub attn_fn: DenseWeight,
    pub base: DenseWeight,
    pub scale: DenseWeight,
}

pub struct DeepSeekV41Layer {
    pub layer: usize,
    pub input_norm: DenseWeight,
    pub post_attn_norm: DenseWeight,
    pub router: V41Router,
    pub shared_expert: V41SharedExpert,
    pub hyper_connections: V41HyperConnections,
    /// Shared across all 40 layers; each layer addresses its own slice.
    pub arena: Arc<Cb3ExpertArena>,
    pub reconstruct: Cb3Reconstruct,
    pub expert_matrices: [Cb3Matrix; 3],
    pub attention: Box<dyn Dsv41Attention>,
    /// Cross-layer sparse state for ONE forward, shared by all 40 layers and RESET BY
    /// LAYER 0. See [`SparseShared`] for why it lives here rather than on `ForwardContext`,
    /// and why layer 0 is a sound reset point (it runs once per forward and has
    /// `compress_ratio == 0`, so it has no sparse work a reset could destroy).
    pub sparse: Arc<Mutex<SparseShared>>,
    /// Layers 1 and 14 only; `None` elsewhere, which is the correct answer there.
    pub engram: Option<Box<dyn Dsv41Engram>>,
}

impl DeepSeekV41Layer {
    /// One message, so both forward entry points say the same true thing.
    fn forward_unimplemented(&self, phase: &str) -> anyhow::Error {
        anyhow::anyhow!(
            "DeepSeek-V4.1 layer {} cannot run {phase} yet. WIRED: the layer is constructed, \
             all dense weights resolve, the {} resident CB3 experts are on the device \
             ({:.1} GB across {} layers), the reconstruct kernel is compiled, and the \
             per-expert CB3 -> bf16 -> cuBLASLt path is built and shape-checked. MISSING, \
             three things: (1) attention — OWNER dsv41-attention, see \
             `seams::Dsv41Attention`; (2) engram on layers 1 and 14 — OWNER dsv41-engram, \
             see `seams::Dsv41Engram`; (3) routing and combine — masked top-k over the \
             router's allow-list and the weighted sum over selected experts — OWNER \
             dsv41-engine, NOT DONE. This is a deliberate hard stop. Running with any of \
             the three stubbed would produce fluent, wrong tokens and no error.",
            self.layer,
            self.arena.packed_keep(),
            self.arena.resident_bytes() as f64 / 1e9,
            self.arena.num_layers(),
        )
    }
}

impl TransformerLayer for DeepSeekV41Layer {
    fn decode(
        &self,
        _hidden: DevicePtr,
        _residual: DevicePtr,
        _state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        _seq_len: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<()> {
        Err(self.forward_unimplemented("decode"))
    }

    fn prefill(
        &self,
        _hidden: DevicePtr,
        _residual: DevicePtr,
        _num_tokens: usize,
        _state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        _seq_len_start: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        _kv_write_start: usize,
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<()> {
        Err(self.forward_unimplemented("prefill"))
    }

    fn alloc_state(&self, _gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        // V4.1 attention is not recurrent: KV lives in the paged cache, so the per-layer
        // state is empty. This is the same answer every attention layer in the tree gives.
        Ok(Box::new(EmptyLayerState))
    }

    fn as_any(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
}
