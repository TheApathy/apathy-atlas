// SPDX-License-Identifier: AGPL-3.0-only

//! The one buffer KDA decode/prefill are allowed to mutate.
//!
//! # The hazard
//!
//! `ops::glm53_kda_decode` and `ops::glm53_kda_prefill` take `state_f32` and
//! **mutate it in place** — vector-decaying the rows and delta-writing the new
//! key/value outer product. That is correct against scratch and is silent
//! corruption against persistent state: a speculative step that is later
//! rejected has already advanced the recurrent state, and there is no inverse
//! to undo a decay-and-accumulate. The sequence keeps generating, fluently,
//! from a state that never existed.
//!
//! Nothing about the op signatures prevents this. Worse, the two candidate
//! sources are near-identical:
//!
//! ```text
//! Glm53ContextPlan  { kda_recurrent_f32: Glm53ContextRegion,  .. }  // PERSISTENT
//! Glm53T1StateLayout{ kda_recurrent_f32: Glm53T1StateRegion,  .. }  // STAGED
//! ```
//!
//! Same field name, same three `u64` fields, same units — one is the thing you
//! must never hand to a KDA kernel, the other is the thing you must always hand
//! it. `Glm53TargetT1StateTransaction::record_kda_layer` already refuses a
//! receipt whose `persistent_state_untouched` flag is false, but that flag is
//! an *attestation*: the caller asserts it, nothing verifies it.
//!
//! # The guard
//!
//! [`Glm53KdaScratchState`] is the only type in this module that yields a
//! buffer for `state_f32`, and it can only be built from the T1 staged region.
//! Construction re-derives the staged extent and **proves disjointness from the
//! whole persistent context region**, so a binding that aliases persistent
//! state cannot be constructed at all — the attestation becomes a check.
//!
//! Persistent state advances only through
//! [`super::kda_recurrent_commit`], on `accepted == 1`.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layers::ops::GgmlIqBuffer;
use crate::weight_loader::Glm53ContextPlan;

use super::kda_recurrent_commit::{
    GLM53_KDA_RECURRENT_ORDINAL_BYTES, GLM53_KDA_RECURRENT_ORDINALS,
};
use super::t1_state_transaction::Glm53T1StateLayout;

/// A recurrent-state buffer that is provably not persistent state.
///
/// Hand this — and only this — to `Glm53KdaDecodeBuffers::state_f32` or
/// `Glm53KdaPrefillBuffers::state_f32`.
#[derive(Debug, Clone, Copy)]
pub struct Glm53KdaScratchState {
    buffer: GgmlIqBuffer,
    /// The persistent slot this ordinal commits to. Held so priming and
    /// committing both derive it from the same validated arithmetic instead of
    /// each caller recomputing it against the wrong `kda_recurrent_f32`.
    persistent: GgmlIqBuffer,
    ordinal: usize,
}

/// Byte span `[start, end)`.
type Span = (u64, u64);

fn intersects(left: Span, right: Span) -> bool {
    left.0 < right.1 && right.0 < left.1
}

impl Glm53KdaScratchState {
    /// Bind the staged recurrent slot for one KDA ordinal.
    ///
    /// `arena_base` is the arena allocation start, `t1_offset` the T1
    /// transaction region's offset within it, and `context` the persistent
    /// plan whose regions this binding must avoid. Ordinals are KDA ordinals
    /// `0..34`, never target-layer ids — the recurrent copies are ordered by
    /// ordinal, and indexing them by layer would stripe the wrong slots.
    pub fn stage(
        arena_base: DevicePtr,
        t1_offset: u64,
        layout: &Glm53T1StateLayout,
        context: &Glm53ContextPlan,
        ordinal: usize,
    ) -> Result<Self> {
        ensure!(!arena_base.is_null(), "GLM KDA scratch arena base is NULL");
        ensure!(
            ordinal < GLM53_KDA_RECURRENT_ORDINALS,
            "GLM KDA ordinal {ordinal} is outside 0..{GLM53_KDA_RECURRENT_ORDINALS}"
        );

        let staged = layout.kda_recurrent_f32;
        ensure!(
            staged.payload_bytes
                == GLM53_KDA_RECURRENT_ORDINAL_BYTES * GLM53_KDA_RECURRENT_ORDINALS as u64,
            "GLM T1 staged recurrent payload {} does not tile {GLM53_KDA_RECURRENT_ORDINALS} \
             ordinals of {GLM53_KDA_RECURRENT_ORDINAL_BYTES} bytes",
            staged.payload_bytes
        );

        let slot_offset = t1_offset
            .checked_add(staged.offset_bytes)
            .and_then(|base| {
                base.checked_add((ordinal as u64).checked_mul(GLM53_KDA_RECURRENT_ORDINAL_BYTES)?)
            })
            .context("GLM KDA scratch slot offset overflow")?;
        let slot_end = slot_offset
            .checked_add(GLM53_KDA_RECURRENT_ORDINAL_BYTES)
            .context("GLM KDA scratch slot extent overflow")?;

        // The slot must lie wholly inside the staged recurrent payload.
        let staged_start = t1_offset + staged.offset_bytes;
        ensure!(
            slot_offset >= staged_start && slot_end <= staged_start + staged.payload_bytes,
            "GLM KDA ordinal {ordinal} falls outside the staged recurrent region"
        );

        // The load-bearing check: it must not touch persistent state anywhere.
        for (name, region) in [
            ("kda_recurrent_f32", context.kda_recurrent_f32),
            ("kda_conv_f32", context.kda_conv_f32),
            ("dsa_latent", context.dsa_latent),
            ("dsa_pooled_index", context.dsa_pooled_index),
            ("dsa_pool_validity", context.dsa_pool_validity),
            ("dsa_tail_keys", context.dsa_tail_keys),
            ("dsa_tail_gates", context.dsa_tail_gates),
            ("dsa_tail_validity", context.dsa_tail_validity),
        ] {
            let persistent = (
                region.offset_bytes,
                region
                    .offset_bytes
                    .checked_add(region.allocation_bytes)
                    .context("GLM persistent region extent overflow")?,
            );
            ensure!(
                !intersects((slot_offset, slot_end), persistent),
                "GLM KDA scratch ordinal {ordinal} aliases persistent {name}; KDA kernels \
                 mutate this buffer in place and a rejected step could not be rolled back"
            );
        }

        // The persistent slot this ordinal will commit into, derived once here.
        //
        // Persistent recurrent state holds `kda_pool_slots` slots of 34
        // ordinals, not 34 ordinals total: the active sequence's slot 0 plus a
        // fixed padding/dummy slot that `SsmStatePool` requires and that must
        // **never be written**. So the region is a multiple of the per-slot
        // span, and every ordinal here addresses slot 0 only.
        let persistent_region = context.kda_recurrent_f32;
        let slot_span = GLM53_KDA_RECURRENT_ORDINAL_BYTES * GLM53_KDA_RECURRENT_ORDINALS as u64;
        ensure!(
            persistent_region.payload_bytes.is_multiple_of(slot_span)
                && persistent_region.payload_bytes >= slot_span,
            "GLM persistent recurrent payload {} is not a whole number of {slot_span}-byte slots",
            persistent_region.payload_bytes
        );
        let persistent_offset = persistent_region
            .offset_bytes
            .checked_add((ordinal as u64) * GLM53_KDA_RECURRENT_ORDINAL_BYTES)
            .context("GLM persistent recurrent slot offset overflow")?;
        // Never address past the active slot: the padding slot begins exactly
        // at `slot_span` and writing it corrupts the pool's dummy entry.
        ensure!(
            persistent_offset + GLM53_KDA_RECURRENT_ORDINAL_BYTES
                <= persistent_region.offset_bytes + slot_span,
            "GLM KDA ordinal {ordinal} would address the padding slot, which must never be written"
        );

        Ok(Self {
            persistent: GgmlIqBuffer {
                ptr: DevicePtr(
                    arena_base
                        .0
                        .checked_add(persistent_offset)
                        .context("GLM persistent recurrent address overflow")?,
                ),
                bytes: usize::try_from(GLM53_KDA_RECURRENT_ORDINAL_BYTES)?,
            },
            buffer: GgmlIqBuffer {
                ptr: DevicePtr(
                    arena_base
                        .0
                        .checked_add(slot_offset)
                        .context("GLM KDA scratch address overflow")?,
                ),
                bytes: usize::try_from(GLM53_KDA_RECURRENT_ORDINAL_BYTES)?,
            },
            ordinal,
        })
    }

    /// The buffer to pass as `state_f32`.
    pub fn buffer(&self) -> GgmlIqBuffer {
        self.buffer
    }

    /// KDA ordinal this slot serves, for receipt ordering.
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// The persistent slot this ordinal commits into.
    ///
    /// Exposed for the commit phase only. Never hand this to a KDA kernel: they
    /// mutate their state argument in place, which is the hazard this whole
    /// module exists to prevent.
    pub fn persistent(&self) -> GgmlIqBuffer {
        self.persistent
    }

    /// Copy the last committed state into scratch before the recurrence runs.
    ///
    /// The decode/prefill kernels read *and* mutate `state_f32`, so scratch must
    /// start as a faithful copy of persistent state or the recurrence continues
    /// from stale or zeroed rows — fluent, wrong output rather than an error.
    /// This costs one 4 MiB device-to-device copy per KDA layer, so 136 MiB per
    /// token across all 34; that is the price of never letting a rejected step
    /// touch persistent state, and it is a known throughput ceiling rather than
    /// an oversight.
    pub fn prime(&self, gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
        // Stream-ordered: `persistent` is written by the commit copy on
        // `stream`, and the walk stream is CU_STREAM_NON_BLOCKING.
        gpu.copy_d2d_async(
            self.persistent.ptr,
            self.buffer.ptr,
            self.buffer.bytes,
            stream,
        )
    }
}

#[cfg(test)]
#[path = "kda_state_binding_tests.rs"]
mod tests;
