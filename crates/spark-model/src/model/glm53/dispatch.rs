// SPDX-License-Identifier: AGPL-3.0-only

//! Event dispatch for the GLM-5.3 target walk.
//!
//! `executor.rs` proves every schedule event resolves to an in-tree op.
//! `workspace_binding.rs` turns the workspace plan into device buffers. This is
//! where the two meet: an event plus bound buffers becomes an actual kernel
//! launch.
//!
//! Dispatch is added **per event kind, each with a test**, rather than all at
//! once. Wired today:
//!
//! * `ExpandMhc`     — collapsed hidden -> 4 widened mHC streams
//! * `OrderedMean`   — 4 streams -> collapsed hidden (the inverse reduction)
//! * `PreAttention`  — mHC pre (attention branch) + `attn_norm`
//! * `PostAttention` — mHC post, then mHC pre (FFN branch) + `ffn_norm`
//! * `PostFfn`       — mHC post
//! * `FinalNormF32`  — `output_norm` over the collapsed hidden state
//! * `Ffn` (MoE)     — router + 8 routed experts + shared expert, serially
//! * `Ffn` (dense)   — gate/up/SwiGLU/down over a 12,288-wide intermediate
//! * `CaptureWidenedMhc` — post-layer stream state into a DFlash2 arena slot
//! * `Attention` (KDA) — conv, forget gate, recurrence, gated norm, projection
//! * `Attention` (DSA) — absorbed MLA with the k-pool indexer and top-k select
//! * `LmHeadF32`   — `output.weight` over the normalized hidden state
//!
//! That is every one of the 234 events.
//!
//! Refusal paths are retained for events whose operands do not resolve — a
//! dispatcher that silently skipped an event would produce fluent, wrong output
//! instead of an error, the failure mode that cost this project days on
//! Flash-Next.
//!
//! # Why `PostAttention` does three things
//!
//! The reference graph (`llama.cpp` `src/models/glm5next.cpp`, the per-layer
//! loop) calls `build_hc_pre` **twice** per layer — once with `hc_attn_*`
//! before the attention site and once with `hc_ffn_*` before the FFN site —
//! and each call overwrites the `post`/`comb` mixing coefficients that the
//! following `build_hc_post` consumes:
//!
//! ```text
//! residual = inpL;  cur = hc_pre(inpL, attn_fn..) -> post, comb
//! cur = norm(cur, attn_norm);  cur = attention(cur)
//! inpL = hc_post(cur, residual, post, comb)
//! residual = inpL;  cur = hc_pre(inpL, ffn_fn..)  -> post, comb   (overwrites)
//! cur = norm(cur, ffn_norm);   cur = ffn(cur)
//! inpL = hc_post(cur, residual, post, comb)
//! ```
//!
//! The pinned 234-event schedule has no `PreFfn` event, so the FFN-side
//! `hc_pre` + `ffn_norm` has to live somewhere. It is placed at the **end** of
//! `PostAttention` rather than the start of `Ffn` so that the `Ffn` seam stays
//! exactly the FFN compute the executor's seam map names, and so the mixing
//! coefficients are already in the workspace before any FFN work begins. The
//! composed order is asserted against the sequence above in `dispatch_tests`.

use anyhow::{Result, bail, ensure};
use spark_runtime::gpu::GpuBackend;

use crate::layers::ops::{
    GgmlIqBuffer, Glm53HyperKernels, Glm53HyperPlan, Glm53HyperPostBuffers, Glm53HyperPreBuffers,
    Glm53RouterBuffers, Glm53RouterKernels, Glm53RouterPlan,
};
use crate::layers::{
    Glm53SerialMoeKernels, Glm53TargetAttentionKind, Glm53TargetEvent, Glm53TargetFfnKind,
};
use crate::weight_loader::{
    Glm53AttentionWeights, Glm53FfnWeights, Glm53GgufMatrix, Glm53HyperWeights, Glm53LayerNorms,
    Glm53TargetLayerWeights,
};

use super::capture_slots::Glm53CaptureSlots;
use super::dense_ffn::Glm53DenseFfnKernels;
use super::dsa_attention::{Glm53DsaAttentionKernels, Glm53DsaCacheSlots, Glm53DsaLayerGeometry};
use super::kda_attention::{Glm53KdaAttentionKernels, Glm53KdaConvSlots};
use super::kda_state_binding::Glm53KdaScratchState;
use super::mhc_expansion::{Glm53HyperBranch, Glm53MhcExpanded};
use super::walk_scratch::Glm53WalkScratch;
use super::workspace_binding::Glm53BoundWorkspace;

/// Layers carrying mHC connections.
const LAYERS: usize = 45;
/// Gated-delta-net layers.
const KDA_LAYERS: usize = 34;
/// Sparse-latent layers, at `layer % 4 == 3`.
const DSA_LAYERS: usize = 11;

/// Per-layer attention resources for one walk.
///
/// KDA and DSA layers are indexed by their own **ordinals**, not by target
/// layer id: the recurrent commit copies by ordinal, and indexing those arrays
/// by layer would stripe the wrong slots.
pub struct Glm53AttentionBinding {
    pub kda_states: Vec<Glm53KdaScratchState>,
    pub kda_conv: Vec<Glm53KdaConvSlots>,
    pub dsa_cache: Vec<Glm53DsaCacheSlots>,
    pub geometry: Glm53DsaLayerGeometry,
}

/// KDA ordinal of a target layer, or `None` if it is a DSA layer.
const fn kda_ordinal(layer: usize) -> Option<usize> {
    if layer % 4 == 3 || layer >= LAYERS {
        None
    } else {
        // Subtract the DSA layers that precede it.
        Some(layer - (layer + 1) / 4)
    }
}

/// DSA ordinal of a target layer, or `None` if it is a KDA layer.
const fn dsa_ordinal(layer: usize) -> Option<usize> {
    if layer % 4 == 3 && layer < LAYERS {
        Some((layer - 3) / 4)
    } else {
        None
    }
}
/// `[MIX]` mixing bias and `[3]` scale, as F32.
const BASE_ELEMENTS: usize = 24;
const SCALE_ELEMENTS: usize = 3;
/// RMS norm weights are F32 `[HIDDEN]`.
const HIDDEN: usize = 4096;

/// One layer's mHC operands, resolved once so dispatch is pure plumbing.
///
/// `function` points into the expanded F32 arena region, not at the Q8_0
/// checkpoint tensor — see [`super::mhc_expansion`].
#[derive(Debug, Clone, Copy)]
pub struct Glm53LayerHyperOperands {
    pub attn_function: GgmlIqBuffer,
    pub attn_base: GgmlIqBuffer,
    pub attn_scale: GgmlIqBuffer,
    pub attn_norm: GgmlIqBuffer,
    pub ffn_function: GgmlIqBuffer,
    pub ffn_base: GgmlIqBuffer,
    pub ffn_scale: GgmlIqBuffer,
    pub ffn_norm: GgmlIqBuffer,
}

fn f32_buffer(ptr: spark_runtime::gpu::DevicePtr, bytes: usize) -> GgmlIqBuffer {
    GgmlIqBuffer { ptr, bytes }
}

impl Glm53LayerHyperOperands {
    /// Resolve one layer from its catalog weights plus the expanded region.
    pub fn resolve(
        layer: usize,
        hyper: &Glm53HyperWeights,
        norms: &Glm53LayerNorms,
        expanded: &Glm53MhcExpanded,
    ) -> Result<Self> {
        let check = |name: &str, got: usize, want: usize| -> Result<()> {
            ensure!(
                got == want,
                "GLM layer {layer} {name} is {got} F32 elements, expected {want}"
            );
            Ok(())
        };
        check(
            "hc base (attn)",
            hyper.attention.base.elements(),
            BASE_ELEMENTS,
        )?;
        check("hc base (ffn)", hyper.ffn.base.elements(), BASE_ELEMENTS)?;
        check(
            "hc scale (attn)",
            hyper.attention.scale.elements(),
            SCALE_ELEMENTS,
        )?;
        check("hc scale (ffn)", hyper.ffn.scale.elements(), SCALE_ELEMENTS)?;
        check("attn_norm", norms.attention.elements(), HIDDEN)?;
        check("ffn_norm", norms.ffn.elements(), HIDDEN)?;

        Ok(Self {
            attn_function: expanded.slot(layer, Glm53HyperBranch::Attention)?,
            attn_base: f32_buffer(hyper.attention.base.ptr(), hyper.attention.base.bytes()),
            attn_scale: f32_buffer(hyper.attention.scale.ptr(), hyper.attention.scale.bytes()),
            attn_norm: f32_buffer(norms.attention.ptr(), norms.attention.bytes()),
            ffn_function: expanded.slot(layer, Glm53HyperBranch::Ffn)?,
            ffn_base: f32_buffer(hyper.ffn.base.ptr(), hyper.ffn.base.bytes()),
            ffn_scale: f32_buffer(hyper.ffn.scale.ptr(), hyper.ffn.scale.bytes()),
            ffn_norm: f32_buffer(norms.ffn.ptr(), norms.ffn.bytes()),
        })
    }
}

/// Kernels and operands for one walk.
///
/// FFN weights are **borrowed**, never copied: they carry device pointers, and
/// `glm53_moe_serial`'s own source contract pins `weights: &Glm53MoeWeights`
/// through execution so a copy cannot silently outlive its store.
pub struct Glm53Dispatcher<'w> {
    hyper: Glm53HyperKernels,
    plan: Glm53HyperPlan,
    bound: Glm53BoundWorkspace,
    layers: Vec<Glm53LayerHyperOperands>,
    output_norm: GgmlIqBuffer,
    moe: Glm53SerialMoeKernels,
    dense: Glm53DenseFfnKernels,
    router: Glm53RouterKernels,
    scratch: Glm53WalkScratch,
    captures: Glm53CaptureSlots,
    catalog: &'w [Glm53TargetLayerWeights],
    kda: Glm53KdaAttentionKernels,
    dsa: Glm53DsaAttentionKernels,
    attention: Glm53AttentionBinding,
    lm_head: &'w Glm53GgufMatrix,
    logits: GgmlIqBuffer,
}

impl<'w> Glm53Dispatcher<'w> {
    /// `tokens` must match the geometry the workspace was planned for; the
    /// hyper plan re-derives its extents from it and the op layer validates
    /// every buffer against those extents on each launch.
    pub fn new(
        gpu: &dyn GpuBackend,
        tokens: u32,
        bound: Glm53BoundWorkspace,
        layers: Vec<Glm53LayerHyperOperands>,
        output_norm: GgmlIqBuffer,
        scratch: Glm53WalkScratch,
        captures: Glm53CaptureSlots,
        catalog: &'w [Glm53TargetLayerWeights],
        attention: Glm53AttentionBinding,
        lm_head: &'w Glm53GgufMatrix,
        logits: GgmlIqBuffer,
    ) -> Result<Self> {
        ensure!(
            layers.len() == LAYERS,
            "GLM dispatch needs exactly {LAYERS} layers of mHC operands, got {}",
            layers.len()
        );
        ensure!(
            catalog.len() == LAYERS,
            "GLM dispatch needs exactly {LAYERS} layers of catalog weights, got {}",
            catalog.len()
        );
        // The catalog's own layer topology must match the schedule's, or a
        // layer scheduled as KDA would be handed DSA weights.
        for (index, layer) in catalog.iter().enumerate() {
            let scheduled_dsa = index % 4 == 3;
            let holds_dsa = matches!(layer.attention, Glm53AttentionWeights::Dsa(_));
            ensure!(
                scheduled_dsa == holds_dsa,
                "GLM layer {index} is scheduled {} but the catalog holds {}",
                if scheduled_dsa { "DSA" } else { "KDA" },
                if holds_dsa { "DSA" } else { "KDA" }
            );
            let scheduled_dense = index < 3;
            let holds_dense = matches!(layer.ffn, Glm53FfnWeights::Dense(_));
            ensure!(
                scheduled_dense == holds_dense,
                "GLM layer {index} FFN kind disagrees with the schedule"
            );
        }
        ensure!(
            attention.kda_states.len() == KDA_LAYERS && attention.kda_conv.len() == KDA_LAYERS,
            "GLM dispatch needs {KDA_LAYERS} KDA state and conv slots, got {} and {}",
            attention.kda_states.len(),
            attention.kda_conv.len()
        );
        ensure!(
            attention.dsa_cache.len() == DSA_LAYERS,
            "GLM dispatch needs {DSA_LAYERS} DSA cache slots, got {}",
            attention.dsa_cache.len()
        );
        // Every KDA ordinal must be distinct, or two layers share one 4 MiB
        // recurrent slot and silently overwrite each other's state.
        let mut ordinals: Vec<usize> = attention.kda_states.iter().map(|s| s.ordinal()).collect();
        ordinals.sort_unstable();
        ordinals.dedup();
        ensure!(
            ordinals.len() == KDA_LAYERS,
            "GLM dispatch KDA states must cover {KDA_LAYERS} distinct ordinals, got {}",
            ordinals.len()
        );
        attention.geometry.validate()?;
        // The head must produce one BF16 logit per vocabulary entry; a short
        // buffer would truncate the distribution rather than fail.
        let head_plan = lm_head.plan(1)?;
        ensure!(
            logits.bytes >= head_plan.output_bytes,
            "GLM logits buffer is {} bytes, the {}-wide head needs {}",
            logits.bytes,
            lm_head.columns(),
            head_plan.output_bytes
        );
        Ok(Self {
            hyper: Glm53HyperKernels::load(gpu)?,
            // H4096 / hc4 / Sinkhorn20 is the only geometry GLM-5.3 admits; the
            // plan constructor rejects anything else rather than scaling.
            plan: Glm53HyperPlan::new(tokens, 4096, 4, 20)?,
            bound,
            layers,
            output_norm,
            moe: Glm53SerialMoeKernels::load(gpu)?,
            dense: Glm53DenseFfnKernels::load(gpu)?,
            router: Glm53RouterKernels::load(gpu)?,
            scratch,
            captures,
            catalog,
            kda: Glm53KdaAttentionKernels::load(gpu)?,
            dsa: Glm53DsaAttentionKernels::load(gpu)?,
            attention,
            lm_head,
            logits,
        })
    }

    fn layer(&self, layer: u32) -> Result<&Glm53LayerHyperOperands> {
        self.layers
            .get(usize::try_from(layer)?)
            .ok_or_else(|| anyhow::anyhow!("GLM dispatch: layer {layer} has no mHC operands"))
    }

    /// mHC pre for one branch, followed by that branch's RMS norm.
    ///
    /// Writes `collapsed` (the normalized block input) plus the `post`/`comb`
    /// mixing coefficients the matching `hc_post` will consume.
    fn pre_and_norm(
        &self,
        gpu: &dyn GpuBackend,
        function: GgmlIqBuffer,
        base: GgmlIqBuffer,
        scale: GgmlIqBuffer,
        norm_weight: GgmlIqBuffer,
        stream: u64,
    ) -> Result<()> {
        self.hyper.pre(
            gpu,
            self.plan,
            Glm53HyperPreBuffers {
                streams_f32: self.bound.widened_hc,
                function_f32: function,
                base_f32: base,
                scale_f32: scale,
                collapsed_bf16: self.bound.collapsed,
                post_bf16: self.bound.hyper_post,
                comb_bf16: self.bound.hyper_comb,
            },
            stream,
        )?;
        {
            // Bring-up diagnostic (ATLAS_GLM53_DUMP_DIR): the mHC mixing
            // coefficients. `comb` is Sinkhorn-normalized and should be doubly
            // stochastic (rows and columns sum to 1). If it is not
            // norm-preserving the residual streams amplify every layer, which
            // is the shape of the remaining divergence (flat to L11, then
            // ~1.29x per layer).
            use std::sync::atomic::{AtomicU32, Ordering};
            static HC_CALLS: AtomicU32 = AtomicU32::new(0);
            let call = HC_CALLS.fetch_add(1, Ordering::Relaxed);
            super::walk_dump::Glm53WalkDump::from_env().write(
                gpu,
                stream,
                self.attention.geometry.position,
                &format!("hc{call:03}"),
                &[
                    ("comb", self.bound.hyper_comb, super::walk_dump::Glm53DumpDtype::Bf16),
                    ("post", self.bound.hyper_post, super::walk_dump::Glm53DumpDtype::Bf16),
                ],
            )?;
        }
        // In-place: the op permits output == input for norm.
        self.hyper.norm(
            gpu,
            self.plan,
            self.bound.collapsed,
            norm_weight,
            self.bound.collapsed,
            stream,
        )
    }

    /// mHC post: recombine a block output with the residual streams.
    ///
    /// Writes the streams in place, matching the reference graph where
    /// `inpL = hc_post(cur, residual, ..)` rebinds the same tensor.
    fn post(&self, gpu: &dyn GpuBackend, block_output: GgmlIqBuffer, stream: u64) -> Result<()> {
        self.hyper.post(
            gpu,
            self.plan,
            Glm53HyperPostBuffers {
                block_output_bf16: block_output,
                residual_streams_f32: self.bound.widened_hc,
                post_bf16: self.bound.hyper_post,
                comb_bf16: self.bound.hyper_comb,
                output_streams_f32: self.bound.widened_hc,
            },
            stream,
        )
    }

    /// Execute one event.
    ///
    /// Returns `Err` for any event whose operands are not yet bound. There is
    /// deliberately no silent-skip arm.
    pub fn dispatch(
        &self,
        gpu: &dyn GpuBackend,
        event: &Glm53TargetEvent,
        stream: u64,
    ) -> Result<()> {
        match event {
            // Widen the collapsed hidden state into the four mHC streams.
            Glm53TargetEvent::ExpandMhc => self.hyper.expand(
                gpu,
                self.plan,
                self.bound.collapsed,
                self.bound.widened_hc,
                stream,
            ),
            // The inverse: reduce the four streams back to one hidden vector.
            // Order is load-bearing — an unordered reduction changes output.
            Glm53TargetEvent::OrderedMean => self.hyper.mean(
                gpu,
                self.plan,
                self.bound.widened_hc,
                self.bound.collapsed,
                stream,
            ),
            Glm53TargetEvent::PreAttention { layer } => {
                let operands = self.layer(*layer)?;
                self.pre_and_norm(
                    gpu,
                    operands.attn_function,
                    operands.attn_base,
                    operands.attn_scale,
                    operands.attn_norm,
                    stream,
                )
            }
            // Close the attention site, then open the FFN site: see the module
            // docs for why the FFN-side pre lives here.
            Glm53TargetEvent::PostAttention { layer } => {
                let operands = self.layer(*layer)?;
                self.post(gpu, self.bound.hidden_a, stream)?;
                self.pre_and_norm(
                    gpu,
                    operands.ffn_function,
                    operands.ffn_base,
                    operands.ffn_scale,
                    operands.ffn_norm,
                    stream,
                )
            }
            Glm53TargetEvent::PostFfn { layer } => {
                // Bounds-check the layer even though the post itself is
                // layer-independent, so an out-of-range walk fails here.
                self.layer(*layer)?;
                self.post(gpu, self.bound.hidden_b, stream)
            }
            Glm53TargetEvent::FinalNormF32 => self.hyper.norm(
                gpu,
                self.plan,
                self.bound.collapsed,
                self.output_norm,
                self.bound.collapsed,
                stream,
            ),
            Glm53TargetEvent::Attention {
                layer,
                kind: Glm53TargetAttentionKind::Kda,
            } => {
                let index = usize::try_from(*layer)?;
                let ordinal = kda_ordinal(index).ok_or_else(|| {
                    anyhow::anyhow!("GLM dispatch: layer {layer} is not a KDA layer")
                })?;
                let Some(Glm53AttentionWeights::Kda(weights)) =
                    self.catalog.get(index).map(|l| &l.attention)
                else {
                    bail!("GLM dispatch: layer {layer} is scheduled KDA but holds DSA weights")
                };
                self.kda
                    .stage(
                        gpu,
                        weights,
                        self.bound.collapsed,
                        self.scratch.kda_buffers(),
                        &self.attention.kda_states[ordinal],
                        self.attention.kda_conv[ordinal],
                        self.attention.geometry.position,
                        self.attention.geometry.capacity,
                        self.attention.geometry.nonce,
                        self.bound.hidden_a,
                        stream,
                    )
                    .map(|_launches| ())
            }
            Glm53TargetEvent::Attention {
                layer,
                kind: Glm53TargetAttentionKind::Dsa,
            } => {
                let index = usize::try_from(*layer)?;
                let ordinal = dsa_ordinal(index).ok_or_else(|| {
                    anyhow::anyhow!("GLM dispatch: layer {layer} is not a DSA layer")
                })?;
                let Some(Glm53AttentionWeights::Dsa(weights)) =
                    self.catalog.get(index).map(|l| &l.attention)
                else {
                    bail!("GLM dispatch: layer {layer} is scheduled DSA but holds KDA weights")
                };
                self.dsa
                    .stage(
                        gpu,
                        weights,
                        self.bound.collapsed,
                        self.scratch.dsa_buffers(),
                        self.attention.dsa_cache[ordinal],
                        self.attention.geometry,
                        self.bound.hidden_a,
                        stream,
                    )
                    .map(|_launches| ())
            }
            Glm53TargetEvent::Ffn {
                layer,
                kind: Glm53TargetFfnKind::Moe,
            } => {
                let index = usize::try_from(*layer)?;
                let weights = self.catalog.get(index).ok_or_else(|| {
                    anyhow::anyhow!("GLM dispatch: layer {layer} has no catalog weights")
                })?;
                let Glm53FfnWeights::Moe(moe) = &weights.ffn else {
                    bail!("GLM dispatch: layer {layer} is scheduled MoE but holds dense weights")
                };
                // The serial executor READS its routed expert ids back from
                // device memory; it never computes them. The router has to run
                // first or the layer dispatches to whatever ids happen to be in
                // the buffer — eight arbitrary experts, fluently wrong.
                let (logits, indices, route_weights) = self.scratch.router_scratch();
                let (router_probs, router_biased) = self.scratch.router_scores();
                self.router.launch(
                    gpu,
                    Glm53RouterPlan::new(1, 4096, 288, 8)?,
                    Glm53RouterBuffers {
                        input_bf16: self.bound.collapsed,
                        router_f32: GgmlIqBuffer {
                            ptr: moe.router.ptr(),
                            bytes: moe.router.bytes(),
                        },
                        bias_f32: GgmlIqBuffer {
                            ptr: moe.expert_bias.ptr(),
                            bytes: moe.expert_bias.bytes(),
                        },
                        logits_f32: logits,
                        indices_u32: indices,
                        weights_f32: route_weights,
                        probs_f32: router_probs,
                        biased_f32: router_biased,
                    },
                    stream,
                )?;
                // Reads the normalized block input written by the FFN-side
                // hc_pre + ffn_norm, writes the block output PostFfn folds back
                // into the streams.
                let result = self
                    .moe
                    .execute(
                        gpu,
                        moe,
                        self.scratch
                            .moe_buffers(self.bound.collapsed, self.bound.hidden_b),
                        stream,
                    )
                    .map(|_receipt| ());
                {
                    // Bring-up diagnostic (ATLAS_GLM53_DUMP_DIR): MoE routing.
                    // 42 of 45 layers are MoE, so wrong routing is fluent-but-
                    // wrong output. Check: 8 distinct ids in 0..288, weights
                    // summing to ~1.
                    let b = self
                        .scratch
                        .moe_buffers(self.bound.collapsed, self.bound.hidden_b);
                    super::walk_dump::Glm53WalkDump::from_env().write(
                        gpu,
                        stream,
                        self.attention.geometry.position,
                        &format!("moe-layer{layer}"),
                        &[
                            ("route_ids_u32", b.route_ids_u32, super::walk_dump::Glm53DumpDtype::U32),
                            // Full 288-expert score vectors, one named file
                            // each. Names mirror llama's ffn_moe_probs /
                            // ffn_moe_probs_biased so the two engines' files
                            // pair 1:1 and a comparison script cannot cross
                            // them. NOT interleaved: an interleaved blob makes
                            // every consumer know a stride, and a stride
                            // mistake is silent -- it yields plausible numbers
                            // that are simply the wrong ranking.
                            ("route_scores_logits", logits, super::walk_dump::Glm53DumpDtype::F32),
                            ("route_scores_prebias", router_probs, super::walk_dump::Glm53DumpDtype::F32),
                            // The score selection actually ranks on. Sort this
                            // to get Atlas's own 8th-vs-9th margin.
                            ("route_scores_biased", router_biased, super::walk_dump::Glm53DumpDtype::F32),
                            ("routed", b.routed_bf16, super::walk_dump::Glm53DumpDtype::Bf16),
                            ("shared", b.shared_bf16, super::walk_dump::Glm53DumpDtype::Bf16),
                            ("route_weights_f32", b.route_weights_f32, super::walk_dump::Glm53DumpDtype::F32),
                            ("out", self.bound.hidden_b, super::walk_dump::Glm53DumpDtype::Bf16),
                        ],
                    )?;
                }
                result
            }
            Glm53TargetEvent::Ffn {
                layer,
                kind: Glm53TargetFfnKind::Dense,
            } => {
                let index = usize::try_from(*layer)?;
                let weights = self.catalog.get(index).ok_or_else(|| {
                    anyhow::anyhow!("GLM dispatch: layer {layer} has no catalog weights")
                })?;
                let Glm53FfnWeights::Dense(dense) = &weights.ffn else {
                    bail!("GLM dispatch: layer {layer} is scheduled dense but holds MoE weights")
                };
                self.dense
                    .execute(
                        gpu,
                        dense,
                        self.bound.collapsed,
                        self.scratch.dense_buffers(),
                        self.bound.hidden_b,
                        stream,
                    )
                    .map(|_launches| ())
            }
            Glm53TargetEvent::CaptureWidenedMhc { layer, slot } => {
                self.captures
                    .capture(gpu, *layer, *slot, self.bound.widened_hc, stream)
            }
            // The head reads the collapsed hidden state that FinalNormF32 just
            // normalized in place.
            Glm53TargetEvent::LmHeadF32 => self.dense.project(
                gpu,
                self.lm_head,
                self.bound.collapsed,
                self.scratch.dense_buffers().q8_activation,
                GgmlIqBuffer {
                    ptr: self.logits.ptr,
                    bytes: self.lm_head.plan(1)?.output_bytes,
                },
                stream,
            ),
        }
    }

    /// Event kinds this dispatcher can execute today.
    pub fn dispatchable(event: &Glm53TargetEvent) -> bool {
        matches!(
            event,
            Glm53TargetEvent::ExpandMhc
                | Glm53TargetEvent::OrderedMean
                | Glm53TargetEvent::PreAttention { .. }
                | Glm53TargetEvent::PostAttention { .. }
                | Glm53TargetEvent::PostFfn { .. }
                | Glm53TargetEvent::FinalNormF32
                | Glm53TargetEvent::Ffn { .. }
                | Glm53TargetEvent::CaptureWidenedMhc { .. }
                | Glm53TargetEvent::Attention { .. }
                | Glm53TargetEvent::LmHeadF32
        )
    }
}

#[cfg(test)]
#[path = "dispatch_tests.rs"]
mod tests;
