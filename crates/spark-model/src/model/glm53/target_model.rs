// SPDX-License-Identifier: AGPL-3.0-only

//! The GLM-5.3 target model: embedding, the 234-event walk, and logits.
//!
//! # Admission
//!
//! Construction is gated. [`Glm53KernelAdmission`] is the reviewed control and
//! is still `[false; 6]`, so the default path refuses. Bring-up is reachable
//! only through the explicit `ATLAS_GLM53_UNVALIDATED_BRINGUP=1` escape, which
//! logs a loud warning and exists for one purpose: to run the port against the
//! llama.cpp `glm5next` oracle so the capability reviews have evidence to work
//! from. **Output from that path has been compared to nothing and must not be
//! quoted as a result.** Precedent for a named experimental gate is
//! `ATLAS_EXPERIMENTAL_NATIVE_QWEN4_DFLASH`.
//!
//! # Commit
//!
//! The walk *stages*: the recurrence runs against scratch, the convolution and
//! the DSA latent write transaction overlays. Persistent state only advances
//! when a step is accepted. This model runs without speculation, so every step
//! is accepted and [`Glm53Model::commit_accepted`] performs the accept-path
//! effects directly rather than through the receipt-verified transaction in
//! `device_completion` / `t1_state_transaction`. That shortcut is the single
//! largest reason the bring-up path is not a validated path: it publishes state
//! without the completion receipts the design requires. It is confined to this
//! function and is unreachable with admission closed.

use std::sync::Mutex;

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::gguf::GgmlType;

use crate::factory::Glm53TargetRuntimeWeights;
use crate::layers::ops::{
    GgmlIqBuffer, GgmlQ4EmbeddingBuffers, GgmlQ4EmbeddingKernel, GgmlQ4EmbeddingPlan,
    GgmlQ5EmbeddingBuffers, GgmlQ5EmbeddingKernel, GgmlQ5EmbeddingPlan, GgmlQ8F32Kernel,
};
use crate::layers::{Glm53TargetGeometry, Glm53TargetSchedule};

use super::arena::{GLM53_KNOWN_ARENA_BYTES, Glm53ArenaPlan};
use super::capture_slots::Glm53CaptureSlots;
use super::dispatch::{Glm53AttentionBinding, Glm53Dispatcher, Glm53LayerHyperOperands};
use super::dsa_attention::{Glm53DsaCacheSlots, Glm53DsaLayerGeometry};
use super::kda_attention::Glm53KdaConvSlots;
use super::kda_state_binding::Glm53KdaScratchState;
use super::kernels::Glm53KernelAdmission;
use super::mhc_expansion::Glm53MhcExpanded;
use super::walk_dump::{Glm53DumpDtype, Glm53WalkDump};

/// The KDA convolution-state carry across positions.
///
/// `true` is production. `false` reproduces the pre-fix behaviour, in which the
/// conv shift-register never advanced and the model emitted degenerate text from
/// the second token onward -- kept only so the defect can be re-measured.
const ATLAS_KDA_CONV_COMMIT_DIRECT: bool = true;

/// Steps staged per commit on this walk. The walk is non-speculative: it stages
/// exactly one token and accepts it. Threaded into the commit as real values
/// rather than assumed, so that wiring speculation forces the call site to pass
/// the true counts and trips the acceptance guard instead of silently
/// committing the wrong window.
const GLM53_WALK_STAGED_QUERIES: u32 = 1;
const GLM53_WALK_ACCEPTED_QUERIES: u32 = 1;

/// Build-identity marker, force-emitted so `strings` can POSITIVELY name which
/// side of the gate a binary was built on.
///
/// Long and delimited on purpose: a 3-byte literal is packed contiguously and
/// an exact-match grep silently returns nothing, so the check would measure the
/// tool rather than the code. Read it with:
///   strings -a <bin> | grep -oE 'ATLAS_KDA_CONV_COMMIT=(direct-guarded|off-NO-CONV-CARRY)'
#[used]
static ATLAS_KDA_CONV_COMMIT_MARKER: &str = if ATLAS_KDA_CONV_COMMIT_DIRECT {
    "ATLAS_KDA_CONV_COMMIT=direct-guarded"
} else {
    "ATLAS_KDA_CONV_COMMIT=off-NO-CONV-CARRY"
};
use super::walk_scratch::Glm53WalkScratch;

/// The env escape that admits the unvalidated bring-up path.
pub const GLM53_BRINGUP_ENV: &str = "ATLAS_GLM53_UNVALIDATED_BRINGUP";

const VOCAB: u32 = 154_880;
const HIDDEN: u32 = 4_096;
const KDA_ORDINALS: usize = 34;
const DSA_ORDINALS: usize = 11;

/// `token_embd.weight` is Q5_K under UD-Q2_K_XL and Q4_K under UD-IQ2_XXS.
enum Glm53Embedding {
    Q4(GgmlQ4EmbeddingKernel),
    Q5(GgmlQ5EmbeddingKernel),
}

/// Mutable per-walk state. One sequence AT A TIME; batching is not wired.
///
/// `position` is not per-request unless something resets it. It was not, and
/// the result was cross-request contamination on a live server: request 3
/// answered its own prompt and then resumed request 1 mid-sentence, because
/// every carried region -- the KDA conv shift-register, the KDA recurrent
/// state, the DSA pools and latent cache -- continued the earlier sequence.
/// `Glm53Model::reset_sequence` is what makes this per-request; see there.
struct Glm53WalkState {
    position: u32,
    nonce: u64,
    /// Walks since the last sequence boundary. Zero means the carried state is
    /// at its construction values and a fresh sequence may start.
    walks_since_boundary: u64,
}

pub struct Glm53Model {
    gpu: Box<dyn GpuBackend>,
    weights: Glm53TargetRuntimeWeights,
    arena: DevicePtr,
    scratch_allocation: DevicePtr,
    scratch: Glm53WalkScratch,
    mhc: Glm53MhcExpanded,
    captures: Glm53CaptureSlots,
    hyper: Vec<Glm53LayerHyperOperands>,
    kda_states: Vec<Glm53KdaScratchState>,
    kda_conv: Vec<Glm53KdaConvSlots>,
    dsa_cache: Vec<Glm53DsaCacheSlots>,
    schedule: Glm53TargetSchedule,
    embedding: Glm53Embedding,
    logits: DevicePtr,
    token_ids: DevicePtr,
    capacity: u32,
    plan: Glm53ArenaPlan,
    state: Mutex<Glm53WalkState>,
    /// Sequences the scheduler currently holds against this model.
    ///
    /// The arena is planned for batch 1: `Glm53ArenaPlan::for_context` sizes
    /// every region for ONE sequence and there is a single `Glm53WalkState`
    /// behind one mutex, so two concurrent sequences would interleave one
    /// position counter with no error at all. `reset_sequence` fixes
    /// SEQUENTIAL reuse; this refuses CONCURRENT use. Shipping either alone
    /// leaves a hole: the reset without this leaves concurrency corrupting
    /// silently, this without the reset leaves the contamination unfixed.
    live_sequences: std::sync::atomic::AtomicUsize,
}

impl Glm53Model {
    /// Whether the unvalidated bring-up escape is set.
    pub fn bringup_escape_set() -> bool {
        std::env::var(GLM53_BRINGUP_ENV).is_ok_and(|value| value == "1")
    }

    /// Refuse unless admission is open or the bring-up escape is set.
    ///
    /// Kept separate from construction so a caller can report the reason
    /// without allocating an arena first.
    pub fn admit() -> Result<()> {
        if Glm53KernelAdmission::current().validate().is_ok() {
            return Ok(());
        }
        if Self::bringup_escape_set() {
            tracing::warn!(
                "GLM-5.3 is running through {GLM53_BRINGUP_ENV}=1. Kernel admission is CLOSED: \
                 no capability has been reviewed, the accept path publishes state without \
                 completion receipts, and this output has been compared against nothing. Do not \
                 quote any accuracy or speed number produced on this path."
            );
            return Ok(());
        }
        bail!(
            "GLM-5.3 kernel admission is closed and {GLM53_BRINGUP_ENV} is not set; \
             see Glm53KernelAdmission for the capabilities awaiting review"
        );
    }

    /// Device bytes this model allocates beyond the checkpoint, at full 1M.
    pub fn runtime_allocation_bytes() -> u64 {
        GLM53_KNOWN_ARENA_BYTES + Glm53WalkScratch::required_bytes() + u64::from(VOCAB) * 2 + 4
    }

    /// The same, at a chosen context length.
    pub fn runtime_allocation_bytes_for(positions: u32) -> Result<u64> {
        Ok(Glm53ArenaPlan::for_context(positions)?.known_bytes
            + Glm53WalkScratch::required_bytes()
            + u64::from(VOCAB) * 2
            + 4)
    }

    /// Build the model with `positions` of context.
    ///
    /// Full 1M does not fit alongside the UD-IQ2_XXS checkpoint on a 119 GiB
    /// box: the device preflight refused at 115,000,477,564 required against
    /// 114,901,217,280 free, a 95 MiB overrun that is almost entirely the DSA
    /// latent cache. The caller picks a context that fits, with margin —
    /// over-allocating unified memory on GB10 takes the host down rather than
    /// returning an error.
    pub fn new(
        gpu: Box<dyn GpuBackend>,
        weights: Glm53TargetRuntimeWeights,
        positions: u32,
    ) -> Result<Self> {
        Self::admit()?;
        let plan = Glm53ArenaPlan::for_context(positions)?;
        let arena_bytes = usize::try_from(plan.known_bytes)?;

        let arena = gpu
            .alloc(arena_bytes)
            .context("GLM arena allocation failed")?;
        // The recurrence and the caches read state before anything writes it,
        // so the arena must start zeroed rather than holding allocator debris.
        gpu.memset(arena, 0, arena_bytes)?;

        let scratch_bytes = Glm53WalkScratch::required_bytes();
        let scratch_allocation = gpu.alloc(usize::try_from(scratch_bytes)? + 256)?;
        // Same reason the arena is zeroed: the recurrence stages into this
        // scratch and the DSA tail is read out of it before every write path
        // has run, so allocator debris here reaches the model as state. It also
        // makes a run reproducible — leaving it dirty made the second token
        // depend on arena capacity, which is how this was found.
        gpu.memset(scratch_allocation, 0, usize::try_from(scratch_bytes)? + 256)?;
        let scratch = Glm53WalkScratch::bind(
            DevicePtr((scratch_allocation.0 + 255) & !255),
            scratch_bytes,
        )?;

        let at = |region_offset: u64| DevicePtr(arena.0 + region_offset);
        let mhc = Glm53MhcExpanded::bind(
            at(plan.mhc_expanded_f32.offset_bytes),
            plan.mhc_expanded_f32.allocation_bytes,
        )?;
        let captures = Glm53CaptureSlots::bind(
            at(plan.dflash_captures.offset_bytes),
            plan.dflash_captures.allocation_bytes,
        )?;

        // One-time dequantization of the 90 mHC mixing functions.
        let q8_f32 = GgmlQ8F32Kernel::load(gpu.as_ref())?;
        // Borrowed, never copied: these views carry device pointers and the
        // catalog owns them.
        let hyper_weights: Vec<&crate::weight_loader::Glm53HyperWeights> =
            weights.target_layers().iter().map(|l| &l.hyper).collect();
        mhc.expand(gpu.as_ref(), &q8_f32, &hyper_weights, gpu.default_stream())?;

        let hyper = weights
            .target_layers()
            .iter()
            .enumerate()
            .map(|(index, layer)| {
                Glm53LayerHyperOperands::resolve(index, &layer.hyper, &layer.norms, &mhc)
            })
            .collect::<Result<Vec<_>>>()?;

        let embedding = match weights.token_embedding().kind() {
            GgmlType::Q4_K => Glm53Embedding::Q4(GgmlQ4EmbeddingKernel::load(gpu.as_ref())?),
            GgmlType::Q5_K => Glm53Embedding::Q5(GgmlQ5EmbeddingKernel::load(gpu.as_ref())?),
            other => bail!("GLM token_embd.weight is {other:?}; no embedding gather exists for it"),
        };

        let logits = gpu.alloc(usize::try_from(VOCAB)? * 2)?;
        let token_ids = gpu.alloc(4)?;

        let (kda_states, kda_conv, dsa_cache) = Self::bind_state(&plan, arena)?;

        Ok(Self {
            gpu,
            weights,
            arena,
            scratch_allocation,
            scratch,
            mhc,
            captures,
            hyper,
            kda_states,
            kda_conv,
            dsa_cache,
            schedule: Glm53TargetSchedule::new(Glm53TargetGeometry::exact(1))?,
            embedding,
            logits,
            token_ids,
            capacity: positions,
            plan,
            live_sequences: std::sync::atomic::AtomicUsize::new(0),
            state: Mutex::new(Glm53WalkState {
                position: 0,
                walks_since_boundary: 0,
                nonce: 1,
            }),
        })
    }

    pub(super) fn bind_state(
        plan: &Glm53ArenaPlan,
        arena: DevicePtr,
    ) -> Result<(
        Vec<Glm53KdaScratchState>,
        Vec<Glm53KdaConvSlots>,
        Vec<Glm53DsaCacheSlots>,
    )> {
        let layout = super::t1_state_transaction::Glm53T1StateLayout::exact()
            .map_err(|error| anyhow::anyhow!("GLM T1 layout: {error:?}"))?;
        let t1 = plan.t1_transaction.offset_bytes;
        let context = &plan.context;

        let kda_states = (0..KDA_ORDINALS)
            .map(|ordinal| Glm53KdaScratchState::stage(arena, t1, &layout, context, ordinal))
            .collect::<Result<Vec<_>>>()?;

        // Conv state is 3 streams x 8192 channels x 4 taps x 4 bytes per layer.
        const CONV_BYTES: u64 = 3 * 8_192 * 4 * 4;
        let conv_persistent = context.kda_conv_f32.offset_bytes;
        let conv_staged = t1 + layout.kda_conv_f32.offset_bytes;
        let kda_conv = (0..KDA_ORDINALS)
            .map(|ordinal| {
                let step = ordinal as u64 * CONV_BYTES;
                Ok(Glm53KdaConvSlots {
                    persistent_state_f32: buffer(arena, conv_persistent + step, CONV_BYTES)?,
                    staged_state_f32: buffer(arena, conv_staged + step, CONV_BYTES)?,
                    published_ends_u32: buffer(
                        arena,
                        t1 + layout.published_ends_u32.offset_bytes + ordinal as u64 * 4,
                        4,
                    )?,
                    published_nonces_u64: buffer(
                        arena,
                        t1 + layout.published_nonces_u64.offset_bytes + ordinal as u64 * 8,
                        8,
                    )?,
                    logical_lengths_u32: buffer(
                        arena,
                        t1 + layout.logical_lengths_u32.offset_bytes + ordinal as u64 * 4,
                        4,
                    )?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let latent_row = 512u64 * 2;
        let dsa_cache = (0..DSA_ORDINALS)
            .map(|ordinal| {
                let ordinal = ordinal as u64;
                Ok(Glm53DsaCacheSlots {
                    latent_cache_bf16: buffer(
                        arena,
                        context.dsa_latent.offset_bytes
                            + ordinal * context.dsa_latent.payload_bytes / DSA_ORDINALS as u64,
                        context.dsa_latent.payload_bytes / DSA_ORDINALS as u64,
                    )?,
                    latent_overlay_bf16: buffer(
                        arena,
                        t1 + layout.dsa_latent_overlay.offset_bytes + ordinal * latent_row,
                        latent_row,
                    )?,
                    pool_keys_bf16: buffer(
                        arena,
                        context.dsa_pooled_index.offset_bytes
                            + ordinal * context.dsa_pooled_index.payload_bytes
                                / DSA_ORDINALS as u64,
                        context.dsa_pooled_index.payload_bytes / DSA_ORDINALS as u64,
                    )?,
                    pool_validity_u8: buffer(
                        arena,
                        context.dsa_pool_validity.offset_bytes
                            + ordinal * context.dsa_pool_validity.payload_bytes
                                / DSA_ORDINALS as u64,
                        context.dsa_pool_validity.payload_bytes / DSA_ORDINALS as u64,
                    )?,
                    prior_tail_keys_bf16: buffer(
                        arena,
                        context.dsa_tail_keys.offset_bytes + ordinal * 3 * 128 * 2,
                        3 * 128 * 2,
                    )?,
                    prior_tail_gates_bf16: buffer(
                        arena,
                        context.dsa_tail_gates.offset_bytes + ordinal * 3 * 128 * 2,
                        3 * 128 * 2,
                    )?,
                    prior_tail_validity_u8: buffer(
                        arena,
                        context.dsa_tail_validity.offset_bytes + ordinal * 3,
                        3,
                    )?,
                    out_tail_validity_u8: buffer(
                        arena,
                        t1 + layout.dsa_pool_tail_overlay.offset_bytes + ordinal * 3,
                        3,
                    )?,
                    sequence_lengths_u32: buffer(
                        arena,
                        t1 + layout.logical_lengths_u32.offset_bytes
                            + (KDA_ORDINALS as u64 + ordinal) * 4,
                        4,
                    )?,
                    query_positions_u32: buffer(
                        arena,
                        t1 + layout.published_ends_u32.offset_bytes
                            + (KDA_ORDINALS as u64 + ordinal) * 4,
                        4,
                    )?,
                    query_validity_u8: buffer(
                        arena,
                        t1 + layout.dsa_pool_tail_overlay.offset_bytes
                            + (DSA_ORDINALS as u64 + ordinal) * 3,
                        1,
                    )?,
                    published_ends_u32: buffer(
                        arena,
                        t1 + layout.published_ends_u32.offset_bytes
                            + (KDA_ORDINALS as u64 + DSA_ORDINALS as u64 + ordinal) * 4,
                        4,
                    )?,
                    published_nonces_u64: buffer(
                        arena,
                        t1 + layout.published_nonces_u64.offset_bytes
                            + (KDA_ORDINALS as u64 + ordinal) * 8,
                        8,
                    )?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok((kda_states, kda_conv, dsa_cache))
    }

    /// Gather one token's embedding into the walk's collapsed hidden buffer.
    fn embed(&self, token: u32, collapsed: GgmlIqBuffer, stream: u64) -> Result<()> {
        self.gpu.copy_h2d(&token.to_le_bytes(), self.token_ids)?;
        let table = self.weights.token_embedding();
        let source = GgmlIqBuffer {
            ptr: table.buffer().ptr,
            bytes: table.buffer().bytes,
        };
        let ids = GgmlIqBuffer {
            ptr: self.token_ids,
            bytes: 4,
        };
        match &self.embedding {
            Glm53Embedding::Q4(kernel) => kernel.launch(
                self.gpu.as_ref(),
                GgmlQ4EmbeddingPlan::new(1, VOCAB, HIDDEN)?,
                GgmlQ4EmbeddingBuffers {
                    source_q4_k: source,
                    token_ids_u32: ids,
                    destination_bf16: collapsed,
                },
                stream,
            ),
            Glm53Embedding::Q5(kernel) => kernel.launch(
                self.gpu.as_ref(),
                GgmlQ5EmbeddingPlan::new(1, VOCAB, HIDDEN)?,
                GgmlQ5EmbeddingBuffers {
                    source_q5_k: source,
                    token_ids_u32: ids,
                    destination_bf16: collapsed,
                },
                stream,
            ),
        }
    }

    /// Publish this step's staged state.
    ///
    /// **This bypasses the receipt-verified transaction on purpose** — see the
    /// module docs. It performs the `accepted == 1` effects directly, which is
    /// only sound because this path never speculates: every staged step is
    /// accepted, so there is no rejection to roll back. Wiring speculation
    /// without first routing this through `Glm53CompletionAuthority` would
    /// reintroduce exactly the corruption the staging split exists to prevent.
    fn commit_accepted(&self, stream: u64) -> Result<()> {
        for state in &self.kda_states {
            let staged = state.buffer();
            // Stream-ordered: the decode kernel wrote `staged` on `stream`, and
            // the walk stream is CU_STREAM_NON_BLOCKING, so a plain copy_d2d
            // issues on the default stream and does NOT wait for it -- it would
            // publish the buffer's PREVIOUS contents as the committed state.
            self.gpu
                .copy_d2d_async(staged.ptr, state.persistent().ptr, staged.bytes, stream)?;
        }
        self.commit_conv(
            stream,
            GLM53_WALK_STAGED_QUERIES,
            GLM53_WALK_ACCEPTED_QUERIES,
        )?;
        Ok(())
    }

    /// Commit the KDA convolution shift-register from staged to persistent.
    ///
    /// This is the SAME operation the transactional kernel performs on its
    /// accepted path: `launch_commit`'s `accepted_count == query_count` branch
    /// is literally `persistent_state[element] = staged_state[element]`, and
    /// `launch_stage` writes the fully-updated post-token window into staged.
    /// The kernel's q/k/v arguments exist only for the PARTIAL-acceptance
    /// branch, which rebuilds the window from persistent over the accepted
    /// prefix. So for a fully-accepted stream this is not an approximation.
    ///
    /// The precondition is ENFORCED, not documented. `launch_commit`'s
    /// transactional path is simply not wired: nothing calls it.
    ///
    /// The publish step DOES run -- `atlas_glm53_kda_conv_finalize` is launched
    /// from inside `launch_stage` (ops/glm53_kda_conv.rs:199), not from a
    /// separate wrapper, so end_position and the nonce are published every
    /// token. Note the protocol is a closed chain: finalize refuses to publish
    /// unless `logical_lengths == start_position`, and `logical_lengths` is
    /// advanced only by the commit kernel. It therefore bootstraps correctly
    /// only if commit runs every token from position 0; wiring it mid-stream
    /// would find stale metadata and silently decline.
    ///
    /// Rather than leave a comment saying this is only valid under full
    /// acceptance, the guard below fails loudly the moment it is not, so
    /// enabling speculative decode cannot silently corrupt the window.
    ///
    /// Without this the conv shift-register never advances: both buffers come
    /// from the memset-once arena, so persistent stays zeros forever. At
    /// position 0 a zero window is the CORRECT answer, which is why every
    /// single-token parity run passed while the model produced degenerate text
    /// from position 1 onward.
    ///
    /// Stream-ordered: the conv kernel writes `staged` on `stream`, and the
    /// walk stream is CU_STREAM_NON_BLOCKING, so a plain `copy_d2d` would issue
    /// on the default stream without waiting and publish stale contents.
    fn commit_conv(&self, stream: u64, queries: u32, accepted: u32) -> Result<()> {
        ensure!(
            accepted == queries,
            "GLM KDA conv commit: the direct staged->persistent path requires full \
             acceptance (accepted={accepted}, queries={queries}). The transactional \
             replay path (Glm53KdaConvKernel::launch_commit) is not wired -- nothing \
             calls it. Wiring it for speculation means passing a real accepted count \
             and the same nonce and start/end positions that launch_stage published."
        );
        if !ATLAS_KDA_CONV_COMMIT_DIRECT {
            return Ok(());
        }
        for conv in &self.kda_conv {
            self.gpu.copy_d2d_async(
                conv.staged_state_f32.ptr,
                conv.persistent_state_f32.ptr,
                conv.staged_state_f32.bytes,
                stream,
            )?;
        }
        Ok(())
    }

    /// The acceptance precondition the direct conv commit enforces.
    ///
    /// Exposed for the test below: the guard is the only thing standing between
    /// enabling speculation and silently committing a window that reflects
    /// rejected tokens, so it needs to be asserted rather than assumed.
    pub(super) fn conv_commit_precondition(queries: u32, accepted: u32) -> Result<()> {
        ensure!(
            accepted == queries,
            "GLM KDA conv commit: the direct staged->persistent path requires full \
             acceptance (accepted={accepted}, queries={queries})"
        );
        Ok(())
    }

    /// Run the whole 234-event walk for one token.
    /// OBSERVE the conv commit chain, rather than deriving it from reading.
    ///
    /// `commit_conv` is justified by a chain nobody has ever watched run:
    /// `atlas_glm53_kda_conv_finalize` publishes only when
    /// `logical_lengths == start_position`, and `logical_lengths` is advanced
    /// only by the commit kernel, so the protocol bootstraps correctly ONLY if
    /// commit runs every token from position 0. Everything speculative decode
    /// will be built on rests on that, and it has been read, not measured.
    ///
    /// Enabled by `ATLAS_GLM53_COMMIT_PROBE=1`. It runs AFTER the walk's final
    /// synchronize, so the copies it issues add a readback but never a barrier
    /// the walk did not already pay for -- and when the variable is unset it
    /// costs one `OnceLock` read per token.
    ///
    /// What to look for: `end` and `len` must BOTH equal position+1 after the
    /// token at `position`, on all 34 KDA layers, with no layer lagging. A
    /// layer stuck at 0 means finalize declined; a layer stuck at 1 means it
    /// published once and the commit never advanced the length after.
    fn commit_probe(&self, position: u32) -> Result<()> {
        static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ENABLED.get_or_init(|| {
            std::env::var("ATLAS_GLM53_COMMIT_PROBE").is_ok_and(|value| value == "1")
        }) {
            return Ok(());
        }
        let expected = position + 1;
        let mut word = [0u8; 4];
        let mut lagging = Vec::new();
        let mut first = None;
        for (ordinal, conv) in self.kda_conv.iter().enumerate() {
            self.gpu.copy_d2h(conv.published_ends_u32.ptr, &mut word)?;
            let end = u32::from_le_bytes(word);
            self.gpu.copy_d2h(conv.logical_lengths_u32.ptr, &mut word)?;
            let length = u32::from_le_bytes(word);
            if first.is_none() {
                first = Some((end, length));
            }
            if end != expected || length != expected {
                lagging.push(format!("L{ordinal}(end={end},len={length})"));
            }
        }
        let (end, length) = first.unwrap_or((0, 0));
        if lagging.is_empty() {
            eprintln!(
                "GLM commit probe p{position}: end={end} len={length} on all {} KDA \
                 layers, expected {expected}: ADVANCING",
                self.kda_conv.len()
            );
        } else {
            eprintln!(
                "GLM commit probe p{position}: expected end=len={expected}, {} of {} \
                 layers disagree: {}",
                lagging.len(),
                self.kda_conv.len(),
                lagging.join(" ")
            );
        }
        Ok(())
    }

    /// Claim the single sequence slot, or refuse.
    ///
    /// One-model-per-sequence is not the alternative: the arena is multi-GB on
    /// top of resident weights, so the scheduler cannot hold N of them on this
    /// box. Refusing is the honest answer.
    ///
    /// A leaked claim produces a LOUD refusal rather than silent interleaving,
    /// which is the correct direction to fail in: the released state is one
    /// sequence's worth and there is no way to tell two apart once they have
    /// both advanced the same counter.
    pub fn claim_sequence(&self) -> Result<()> {
        claim_only_sequence_slot(&self.live_sequences)
    }

    /// Release the sequence slot. Saturating at zero rather than wrapping.
    pub fn release_sequence(&self) {
        release_only_sequence_slot(&self.live_sequences);
    }

    /// Return the model to its construction state so the NEXT request is a
    /// fresh sequence.
    ///
    /// WHAT THIS FIXES, observed on a live server: three unrelated prompts in
    /// one process. Request 2 produced correct Python and then trailed into
    /// "...to famous historical figures like Napoleon". Request 3 answered
    /// "391" and then continued ", cheese, and fashion. The Eiffel Tower is
    /// located in Paris" -- resuming request 1's sentence, which had ended
    /// "France is known for its cuisine, wine,". Nothing zeroed `position`, so
    /// it advanced monotonically for the life of the process and every carried
    /// region went with it.
    ///
    /// No parity test could have caught it: every dump runs one sequence in a
    /// fresh process. Single-token parity hid the conv carry the same way.
    ///
    /// THE REGIONS ARE DERIVED FROM THE PLAN, NOT ENUMERATED. Enumerating them
    /// is how a reset misses one and leaves a subtler version of this bug. The
    /// arena is zeroed WHOLE at construction, so the faithful reset is the same
    /// memset -- except for `mhc_expanded_f32`, which is a load-time
    /// precomputation from the weights and not sequence state. So this zeroes
    /// everything on either side of it and ASSERTS that it is the only gap; a
    /// future layout that moves another load-time region out of the zeroed span
    /// trips the assert instead of silently surviving the reset.
    ///
    /// Synchronous, not stream-ordered: this is a boundary between requests,
    /// the caller is not mid-pipeline, and a reset that raced the previous
    /// sequence's tail kernels would restore exactly the state it is clearing.
    pub fn reset_sequence(&self) -> Result<()> {
        let plan = &self.plan;
        let mhc = plan.mhc_expanded_f32;
        let carried_end = plan.context.total_bytes;
        let after = plan.t1_transaction.offset_bytes;
        // The mHC block is the ONLY thing between the two zeroed spans.
        ensure!(
            carried_end <= mhc.offset_bytes
                && mhc.offset_bytes + mhc.allocation_bytes == after
                && after < plan.known_bytes,
            "GLM sequence reset cannot derive its spans: the arena layout is              context[0, {carried_end}) mhc[{}, {}) t1[{after}, ..) known={}.              The reset zeroes everything except mhc, so mhc must be the single              gap between the two spans.",
            mhc.offset_bytes,
            mhc.offset_bytes + mhc.allocation_bytes,
            plan.known_bytes
        );
        self.gpu.memset(
            self.arena,
            0,
            usize::try_from(carried_end).context("GLM reset carried span")?,
        )?;
        self.gpu.memset(
            DevicePtr(self.arena.0 + after),
            0,
            usize::try_from(plan.known_bytes - after).context("GLM reset tail span")?,
        )?;
        // The scratch allocation stages the recurrence and is zeroed at
        // construction for the same reason the arena is.
        self.gpu.memset(
            self.scratch_allocation,
            0,
            usize::try_from(Glm53WalkScratch::required_bytes())
                .context("GLM reset scratch span")?
                + 256,
        )?;
        let mut state = self.state.lock().unwrap();
        state.position = 0;
        state.walks_since_boundary = 0;
        // The nonce stays MONOTONIC across the boundary. It exists to make a
        // stale transaction detectable, so restarting it would make a carried
        // record from the previous sequence look current.
        Ok(())
    }

    /// A fresh sequence must actually START fresh, verified rather than assumed.
    ///
    /// `reset_sequence` being CALLED is not the same as the state being clear:
    /// a memset that failed, a region the layout moved out of the zeroed spans,
    /// or a caller that reset the wrong model would all leave `position` at 0
    /// with live carried state, which is the contamination bug wearing a
    /// correct-looking position counter.
    ///
    /// So this reads back the transaction metadata that gates the whole carry
    /// protocol -- `published_ends` and `logical_lengths` for the KDA conv
    /// layers -- and requires them zero. It is a few hundred bytes, not 12 GB:
    /// the check is EXACT for the metadata and does not certify every byte of
    /// the latent cache. It is placed where it can fire, which is more than the
    /// comment it replaces did.
    fn ensure_sequence_boundary(&self, walks_since_boundary: u64) -> Result<()> {
        sequence_boundary_is_clean(walks_since_boundary)?;
        let mut word = [0u8; 4];
        for (ordinal, conv) in self.kda_conv.iter().enumerate() {
            for (name, buffer) in [
                ("published_ends", conv.published_ends_u32),
                ("logical_lengths", conv.logical_lengths_u32),
            ] {
                self.gpu.copy_d2h(buffer.ptr, &mut word)?;
                let value = u32::from_le_bytes(word);
                ensure!(
                    value == 0,
                    "GLM sequence boundary is not clean: KDA layer {ordinal}                      {name} is {value}, expected 0 at position 0. The reset did                      not reach this region."
                );
            }
        }
        Ok(())
    }

    fn walk(&self, token: u32, stream: u64) -> Result<DevicePtr> {
        super::walk_timing::walk(|| self.walk_inner(token, stream))
    }

    fn walk_inner(&self, token: u32, stream: u64) -> Result<DevicePtr> {
        let (position, nonce, walks_since_boundary) = {
            let mut state = self.state.lock().unwrap();
            let position = state.position;
            state.nonce = state.nonce.wrapping_add(1).max(1);
            (position, state.nonce, state.walks_since_boundary)
        };
        ensure!(
            position < self.capacity,
            "GLM sequence position {position} reached capacity {}",
            self.capacity
        );
        if position == 0 {
            self.ensure_sequence_boundary(walks_since_boundary)?;
        }

        let workspace = Glm53BoundWorkspaceRef::bind(self)?;
        self.embed(token, workspace.collapsed, stream)?;

        let dispatcher = Glm53Dispatcher::new(
            self.gpu.as_ref(),
            1,
            workspace.inner,
            self.hyper.clone(),
            GgmlIqBuffer {
                ptr: self.weights.output_norm().ptr(),
                bytes: self.weights.output_norm().bytes(),
            },
            self.scratch,
            self.captures,
            self.weights.target_layers(),
            Glm53AttentionBinding {
                kda_states: self.kda_states.clone(),
                kda_conv: self.kda_conv.clone(),
                dsa_cache: self.dsa_cache.clone(),
                geometry: Glm53DsaLayerGeometry {
                    position,
                    capacity: self.capacity,
                    nonce,
                },
            },
            self.weights.output(),
            GgmlIqBuffer {
                ptr: self.logits,
                bytes: usize::try_from(VOCAB)? * 2,
            },
        )?;

        let dump = Glm53WalkDump::from_env();
        dump.write(
            self.gpu.as_ref(),
            stream,
            position,
            "embed",
            &[("collapsed", workspace.collapsed, Glm53DumpDtype::Bf16)],
        )?;
        for (index, event) in self.schedule.events().iter().enumerate() {
            dispatcher
                .dispatch(self.gpu.as_ref(), event, stream)
                .with_context(|| format!("GLM walk failed at {event:?}"))?;
            dump.write(
                self.gpu.as_ref(),
                stream,
                position,
                &format!("{index:03}-{event:?}"),
                &[
                    ("collapsed", workspace.inner.collapsed, Glm53DumpDtype::Bf16),
                    // The mHC residual streams are F32 as of the streams change.
                    (
                        "widened_hc",
                        workspace.inner.widened_hc,
                        Glm53DumpDtype::F32,
                    ),
                    ("hidden_a", workspace.inner.hidden_a, Glm53DumpDtype::Bf16),
                    ("hidden_b", workspace.inner.hidden_b, Glm53DumpDtype::Bf16),
                    // KDA intermediates, so a divergence lands on one stage
                    // rather than on "the recurrence". Each has a named
                    // counterpart in the llama.cpp reference dump.
                    (
                        "kda_q_conv",
                        self.scratch.kda_buffers().q_conv_bf16,
                        Glm53DumpDtype::Bf16,
                    ),
                    (
                        "kda_k_conv",
                        self.scratch.kda_buffers().k_conv_bf16,
                        Glm53DumpDtype::Bf16,
                    ),
                    (
                        "kda_v_conv",
                        self.scratch.kda_buffers().v_conv_bf16,
                        Glm53DumpDtype::Bf16,
                    ),
                    (
                        "kda_log_decay",
                        self.scratch.kda_buffers().log_decay_f32,
                        Glm53DumpDtype::F32,
                    ),
                    (
                        "kda_beta",
                        self.scratch.kda_buffers().beta_bf16,
                        Glm53DumpDtype::Bf16,
                    ),
                    (
                        "kda_recurrent_out",
                        self.scratch.kda_buffers().recurrent_out_bf16,
                        Glm53DumpDtype::Bf16,
                    ),
                    (
                        "kda_gated",
                        self.scratch.kda_buffers().gated_bf16,
                        Glm53DumpDtype::Bf16,
                    ),
                ],
            )?;
        }
        self.commit_accepted(stream)?;
        // Whatever the GPU had not finished when the host stopped enqueuing.
        // A lower bound on real kernel execution, and the term that separates
        // sync-bound from launch-bound.
        super::walk_timing::blocked_final(|| self.gpu.synchronize(stream))?;

        self.commit_probe(position)?;

        {
            let mut state = self.state.lock().unwrap();
            state.position = position + 1;
            state.walks_since_boundary += 1;
        }
        super::walk_timing::report();
        Ok(self.logits)
    }

    /// Free every device allocation this model owns.
    pub fn free(self) -> Result<()> {
        self.gpu.free(self.token_ids)?;
        self.gpu.free(self.logits)?;
        self.gpu.free(self.scratch_allocation)?;
        self.gpu.free(self.arena)?;
        self.weights
            .free(self.gpu.as_ref())
            .map_err(|error| anyhow::anyhow!("GLM weight store free failed: {error}"))
    }

    pub fn vocab(&self) -> usize {
        usize::try_from(VOCAB).expect("vocab fits usize")
    }

    pub fn logits_ptr(&self) -> DevicePtr {
        self.logits
    }

    pub fn bind_thread(&self) -> Result<()> {
        self.gpu.bind_to_thread()
    }

    /// Copy the BF16 logit row to the host.
    pub fn copy_logits(&self, logits_ptr: DevicePtr, dst: &mut [u8]) -> Result<()> {
        let needed = usize::try_from(VOCAB)? * 2;
        ensure!(
            dst.len() >= needed,
            "GLM logits need {needed} bytes, destination holds {}",
            dst.len()
        );
        self.gpu.copy_d2h(logits_ptr, &mut dst[..needed])
    }

    /// Top-k `(token_id, logit)` over the BF16 logit row, descending.
    ///
    /// Parity diagnostic: an argmax mismatch against llama.cpp says the output
    /// is wrong; whether the reference token sits at rank 2 or rank 20,000 says
    /// how wrong, and that is what distinguishes a numerics drift from a
    /// dead layer.
    pub fn top_k_host(
        &self,
        logits_ptr: DevicePtr,
        k: usize,
        stream: u64,
    ) -> Result<Vec<(u32, f32)>> {
        self.gpu.synchronize(stream)?;
        let mut bytes = vec![0u8; usize::try_from(VOCAB)? * 2];
        self.gpu.copy_d2h(logits_ptr, &mut bytes)?;
        let mut scored: Vec<(u32, f32)> = bytes
            .chunks_exact(2)
            .enumerate()
            .map(|(index, chunk)| {
                (
                    index as u32,
                    half::bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32(),
                )
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        Ok(scored)
    }

    /// Greedy argmax over the BF16 logit row.
    ///
    /// Performed on the host: there is no GLM-specific device argmax, and a
    /// wrong one would be indistinguishable from a wrong model. The copy is
    /// 309,760 bytes per token, which is real overhead and a known cost of the
    /// bring-up path rather than a design choice.
    pub fn argmax_host(&self, logits_ptr: DevicePtr, stream: u64) -> Result<u32> {
        self.gpu.synchronize(stream)?;
        let mut bytes = vec![0u8; usize::try_from(VOCAB)? * 2];
        self.gpu.copy_d2h(logits_ptr, &mut bytes)?;
        let mut best = 0u32;
        let mut best_value = f32::NEG_INFINITY;
        for (index, chunk) in bytes.chunks_exact(2).enumerate() {
            let value = half::bf16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32();
            if value > best_value {
                best_value = value;
                best = u32::try_from(index)?;
            }
        }
        ensure!(
            best_value.is_finite(),
            "GLM logits are entirely non-finite; the walk produced no usable distribution"
        );
        Ok(best)
    }

    /// Run one token and return the logits pointer.
    pub fn decode_token(&self, token: u32, stream: u64) -> Result<DevicePtr> {
        self.walk(token, stream)
    }

    /// Run a prompt token by token, returning the last position's logits.
    ///
    /// This is a sequential decode, not a batched prefill: the schedule is
    /// built for exactly one token. Correct, and roughly `prompt_len` times
    /// slower than a real prefill would be.
    pub fn prefill_tokens(&self, tokens: &[u32], stream: u64) -> Result<DevicePtr> {
        ensure!(!tokens.is_empty(), "GLM prefill needs at least one token");
        let mut last = self.logits;
        for token in tokens {
            last = self.walk(*token, stream)?;
        }
        Ok(last)
    }

    /// Bring-up probe: embed one token and return the hidden row as f32.
    ///
    /// Operation zero. If this disagrees with an independent dequantization of
    /// `token_embd.weight`, every later layer is comparing against garbage and
    /// localizing a divergence further in is wasted effort.
    pub fn embed_probe(&self, token: u32, stream: u64) -> Result<Vec<f32>> {
        let workspace = Glm53BoundWorkspaceRef::bind(self)?;
        self.embed(token, workspace.collapsed, stream)?;
        self.gpu.synchronize(stream)?;
        let mut bytes = vec![0u8; workspace.collapsed.bytes];
        self.gpu.copy_d2h(workspace.collapsed.ptr, &mut bytes)?;
        Ok(bytes
            .chunks_exact(2)
            .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect())
    }
}

fn buffer(arena: DevicePtr, offset: u64, bytes: u64) -> Result<GgmlIqBuffer> {
    Ok(GgmlIqBuffer {
        ptr: DevicePtr(
            arena
                .0
                .checked_add(offset)
                .context("GLM arena address overflow")?,
        ),
        bytes: usize::try_from(bytes)?,
    })
}

/// The workspace regions, bound against the arena's transient prefix.
struct Glm53BoundWorkspaceRef {
    inner: super::workspace_binding::Glm53BoundWorkspace,
    collapsed: GgmlIqBuffer,
}

impl Glm53BoundWorkspaceRef {
    fn bind(model: &Glm53Model) -> Result<Self> {
        let plan = &model.plan;
        let inner = super::workspace_binding::Glm53BoundWorkspace::bind(
            &plan.workspace,
            DevicePtr(model.arena.0 + plan.transient.offset_bytes),
            plan.workspace.arena_bytes,
        )?;
        Ok(Self {
            collapsed: inner.collapsed,
            inner,
        })
    }
}

/// Claim the single sequence slot, or refuse. Free so it can be pinned without
/// a device; the model method is a one-line forward to it.
fn claim_only_sequence_slot(live: &std::sync::atomic::AtomicUsize) -> Result<()> {
    use std::sync::atomic::Ordering;
    let held = live.fetch_add(1, Ordering::AcqRel);
    if held != 0 {
        // Undo the speculative increment, or one refusal poisons the slot for
        // the life of the process.
        live.fetch_sub(1, Ordering::AcqRel);
        bail!(
            "GLM-5.3 serves ONE sequence at a time and {held} is already live. The \
             arena is planned for batch 1 -- every region is sized for one sequence \
             and there is a single position counter -- so admitting a second would \
             interleave both into the same state with no error. Sequential requests \
             are fine; each resets the carried state."
        );
    }
    Ok(())
}

/// Saturating at zero, not wrapping: a wrap to usize::MAX would refuse every
/// subsequent request forever.
fn release_only_sequence_slot(live: &std::sync::atomic::AtomicUsize) {
    use std::sync::atomic::Ordering;
    let _ = live.fetch_update(Ordering::AcqRel, Ordering::Acquire, |held| {
        held.checked_sub(1)
    });
}

/// The structural half of the sequence-boundary check, split out so it can be
/// pinned without a device.
///
/// The other half reads the transaction metadata back off the GPU. This half
/// is the one that catches the reported bug: a request that begins while a
/// previous sequence's walks are still counted has not been reset, whatever
/// the position counter says.
fn sequence_boundary_is_clean(walks_since_boundary: u64) -> Result<()> {
    ensure!(
        walks_since_boundary == 0,
        "GLM walk is at position 0 after {walks_since_boundary} walks with no \
         sequence reset. The carried state (KDA conv, KDA recurrent, DSA \
         pools and latent cache) still holds the previous request, so this \
         sequence would continue it -- observed live as request 3 answering \
         its own prompt and then resuming request 1 mid-sentence. Call \
         Glm53Model::reset_sequence at the start of every request."
    );
    Ok(())
}
#[cfg(test)]
#[path = "target_model_tests.rs"]
mod tests;

#[cfg(test)]
mod conv_commit_tests {
    use super::{GLM53_WALK_ACCEPTED_QUERIES, GLM53_WALK_STAGED_QUERIES, Glm53Model};

    /// The direct conv commit is only correct under full acceptance, and the
    /// guard must FIRE rather than merely be documented -- a precondition that
    /// is only a comment is what let the conv carry go missing in the first
    /// place.
    #[test]
    fn conv_commit_refuses_partial_acceptance() {
        // The walk's own values must pass.
        Glm53Model::conv_commit_precondition(
            GLM53_WALK_STAGED_QUERIES,
            GLM53_WALK_ACCEPTED_QUERIES,
        )
        .expect("the non-speculative walk must satisfy its own precondition");
        assert_eq!(GLM53_WALK_STAGED_QUERIES, GLM53_WALK_ACCEPTED_QUERIES);

        // Anything speculation would produce must be refused, loudly.
        for (queries, accepted) in [(4u32, 3u32), (4, 0), (8, 7), (2, 1)] {
            let error = Glm53Model::conv_commit_precondition(queries, accepted)
                .expect_err("partial acceptance must be refused, not silently committed");
            let text = format!("{error}");
            assert!(
                text.contains("requires full acceptance"),
                "guard must name the precondition it enforced: {text}"
            );
        }
    }
}
