// SPDX-License-Identifier: AGPL-3.0-only

//! All-34 KDA recurrent commit: the phase that moves staged recurrent state
//! into persistent state under a verified completion receipt.
//!
//! # Why this module exists
//!
//! `ops::glm53_kda_decode` / `glm53_kda_prefill` mutate the recurrent F32 state
//! they are handed, in place. That is correct for a scratch buffer and is a
//! silent corruption if the caller points them at persistent state, because a
//! rejected speculative step would leave the mutation behind with no way to
//! roll back. The executor here is the reason a driver never has to: KDA runs
//! against the T1 transaction's staged region, and persistent state advances
//! only through this phase, only on `accepted == 1`, and only after the
//! per-layer conv commits have verified.
//!
//! # Ordering contract
//!
//! `Glm53CompletionAuthority` issues `KdaRecurrentCommit` only as the successor
//! of `KdaConvCommit(accepted = 1)`, so possession of the authorization already
//! proves the exact-34 conv slots verified. Effects are enqueued on the
//! caller's exclusive stream in this order, and the marker is last so the
//! readback cannot observe success ahead of the payload:
//!
//! 1. clear the single recurrent completion slot (slot 34)
//! 2. enqueue ordinals `0..=33`, 4 MiB each, staged -> persistent
//! 3. launch the one-thread marker that publishes slot 34
//! 4. exact-slot `copy_d2h_on_stream` (CUDA orders this behind the work above)
//!
//! `accepted == 0` performs none of the above: no clear, no copy, no marker and
//! no readback. It retires the rejected phase and leaves persistent state
//! exactly as it was.

use anyhow::{Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::arena_owner::Glm53ArenaView;
use super::device_completion::{
    GLM53_COMPLETION_SLOT_BYTES, GLM53_COMPLETION_SLOTS, Glm53CompletionExpectation,
};
use super::t1_state_transaction::{GLM53_KDA_T1_LAYERS, Glm53T1StateLayout};

/// Completion slot the `KdaRecurrentCommit` phase publishes and reads back.
/// It sits immediately after the 34 per-layer conv slots (`0..=33`).
pub(super) const GLM53_KDA_RECURRENT_COMMIT_SLOT: usize = 34;

/// Per-ordinal recurrent extent. The T1 layout's `kda_recurrent_f32` payload is
/// exactly `34 * 4 MiB`; the const assertion below pins that relationship so a
/// layout change cannot silently reshape the copies.
pub(super) const GLM53_KDA_RECURRENT_ORDINAL_BYTES: u64 = 4 * 1024 * 1024;

/// Number of recurrent ordinals copied by one commit.
pub(super) const GLM53_KDA_RECURRENT_ORDINALS: usize = GLM53_KDA_T1_LAYERS.len();

const GLM53_KDA_RECURRENT_TOTAL_BYTES: u64 =
    GLM53_KDA_RECURRENT_ORDINAL_BYTES * GLM53_KDA_RECURRENT_ORDINALS as u64;

const _: () = {
    assert!(GLM53_KDA_RECURRENT_ORDINALS == 34);
    assert!(GLM53_KDA_RECURRENT_COMMIT_SLOT == GLM53_KDA_RECURRENT_ORDINALS);
    assert!(GLM53_KDA_RECURRENT_COMMIT_SLOT < GLM53_COMPLETION_SLOTS);
    assert!(GLM53_KDA_RECURRENT_TOTAL_BYTES == 142_606_336);
};

/// Preloaded marker handle. Loading is separate from execution so a failure to
/// resolve the kernel cannot happen midway through a commit.
#[derive(Clone, Copy, Debug)]
pub(super) struct Glm53KdaRecurrentCommitKernel {
    marker: KernelHandle,
}

impl Glm53KdaRecurrentCommitKernel {
    pub(super) fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            marker: gpu.kernel(
                "glm53_kda_recurrent_commit",
                "atlas_glm53_kda_recurrent_commit_marker",
            )?,
        })
    }
}

/// Device regions this phase touches. Held by reference so the arena owner's
/// borrow outlives every enqueue.
#[derive(Clone, Copy)]
pub(super) struct Glm53KdaRecurrentCommitViews<'owner> {
    /// Staged recurrent state inside the T1 transaction region.
    pub(super) staged: &'owner Glm53ArenaView<'owner>,
    /// Persistent recurrent state in the context arena.
    pub(super) persistent: &'owner Glm53ArenaView<'owner>,
    /// Completion slab base (45 slots x 48 bytes).
    pub(super) completion_slab: &'owner Glm53ArenaView<'owner>,
}

/// Crate-private, non-Clone executor: one value performs at most one commit.
#[must_use = "a recurrent commit executor must be consumed by exactly one phase"]
pub(super) struct Glm53KdaRecurrentCommitExecutor {
    kernel: Glm53KdaRecurrentCommitKernel,
    stream: u64,
}

impl Glm53KdaRecurrentCommitExecutor {
    pub(super) fn new(kernel: Glm53KdaRecurrentCommitKernel, stream: u64) -> Result<Self> {
        ensure!(
            stream != 0,
            "GLM KDA recurrent commit requires a nonzero exclusive stream"
        );
        Ok(Self { kernel, stream })
    }

    /// Consumes the executor. `expectation` must be the one this phase issued
    /// for [`GLM53_KDA_RECURRENT_COMMIT_SLOT`]; its fields are the sole source
    /// of the marker's published identity, so a forged expectation cannot make
    /// the host accept a slot it did not ask for.
    ///
    /// Returns the raw readback bytes for the caller to verify. This function
    /// deliberately does not interpret them: verification belongs to
    /// `Glm53CompletionPending`, and splitting publish from verify keeps this
    /// module unable to declare its own success.
    pub(super) fn commit(
        self,
        gpu: &dyn GpuBackend,
        layout: Glm53T1StateLayout,
        views: Glm53KdaRecurrentCommitViews<'_>,
        expectation: Glm53CompletionExpectation,
        readback: &mut [u8],
    ) -> Result<()> {
        layout
            .validate()
            .map_err(|error| anyhow::anyhow!("{error:?}"))?;

        let (
            slot,
            tag,
            layer_id,
            accepted,
            exclusive_end,
            capacity,
            phase_incarnation,
            owner_generation,
            publish_nonce,
        ) = expectation.kernel_fields();

        ensure!(
            slot == GLM53_KDA_RECURRENT_COMMIT_SLOT,
            "GLM KDA recurrent commit received another phase's completion slot"
        );
        ensure!(
            accepted == 1,
            "GLM KDA recurrent commit must not run for a rejected decision"
        );
        ensure!(
            exclusive_end != 0 && exclusive_end <= capacity,
            "GLM KDA recurrent commit exclusive end is outside the capacity"
        );
        ensure!(
            phase_incarnation != 0 && owner_generation != 0 && publish_nonce != 0,
            "GLM KDA recurrent commit identity is unset"
        );
        ensure!(
            readback.len() == GLM53_COMPLETION_SLOT_BYTES,
            "GLM KDA recurrent commit reads back exactly one completion slot"
        );

        // Layout: the staged region must be exactly the 34x4 MiB recurrent
        // payload the T1 layout declares, and persistent must be able to hold
        // it. A mismatch here is the corruption this phase prevents, so it is
        // checked before any enqueue rather than trusted from the caller.
        ensure!(
            layout.kda_recurrent_f32.payload_bytes == GLM53_KDA_RECURRENT_TOTAL_BYTES,
            "GLM T1 recurrent payload is not the exact 34x4MiB extent"
        );
        ensure!(
            views.staged.payload_bytes() >= GLM53_KDA_RECURRENT_TOTAL_BYTES
                && views.persistent.payload_bytes() >= GLM53_KDA_RECURRENT_TOTAL_BYTES,
            "GLM KDA recurrent views are smaller than the committed extent"
        );

        let slab_bytes = (GLM53_COMPLETION_SLOTS * GLM53_COMPLETION_SLOT_BYTES) as u64;
        ensure!(
            views.completion_slab.payload_bytes() >= slab_bytes,
            "GLM completion slab view is smaller than the slot table"
        );

        let staged_base = views.staged.device_ptr();
        let persistent_base = views.persistent.device_ptr();
        let slab_base = views.completion_slab.device_ptr();

        // Mutual non-alias across the full committed extent. Staged and
        // persistent overlapping would make the copies read their own output.
        ensure!(
            disjoint(
                staged_base,
                persistent_base,
                GLM53_KDA_RECURRENT_TOTAL_BYTES
            )?,
            "GLM KDA staged and persistent recurrent views alias"
        );
        ensure!(
            disjoint_spans(
                staged_base,
                GLM53_KDA_RECURRENT_TOTAL_BYTES,
                slab_base,
                slab_bytes
            )? && disjoint_spans(
                persistent_base,
                GLM53_KDA_RECURRENT_TOTAL_BYTES,
                slab_base,
                slab_bytes
            )?,
            "GLM completion slab aliases a recurrent view"
        );

        // 4 MiB alignment keeps every ordinal copy a whole-extent transfer.
        ensure!(
            staged_base.0 % GLM53_KDA_RECURRENT_ORDINAL_BYTES == 0
                && persistent_base.0 % GLM53_KDA_RECURRENT_ORDINAL_BYTES == 0,
            "GLM KDA recurrent views are not ordinal-aligned"
        );

        let slot_offset = (slot * GLM53_COMPLETION_SLOT_BYTES) as u64;
        let slot_ptr = DevicePtr(
            slab_base
                .0
                .checked_add(slot_offset)
                .ok_or_else(|| anyhow::anyhow!("GLM completion slot address overflow"))?,
        );

        // 1. Clear only this phase's slot. Clearing the slab would erase the
        //    conv receipts this phase's authority depends on.
        gpu.memset_async(slot_ptr, 0, GLM53_COMPLETION_SLOT_BYTES, self.stream)?;

        // 2. Ordinals 0..=33, staged -> persistent, one common stream.
        for ordinal in 0..GLM53_KDA_RECURRENT_ORDINALS {
            let offset = GLM53_KDA_RECURRENT_ORDINAL_BYTES
                .checked_mul(ordinal as u64)
                .ok_or_else(|| anyhow::anyhow!("GLM KDA recurrent ordinal overflow"))?;
            let src = DevicePtr(
                staged_base
                    .0
                    .checked_add(offset)
                    .ok_or_else(|| anyhow::anyhow!("GLM KDA staged ordinal overflow"))?,
            );
            let dst = DevicePtr(
                persistent_base
                    .0
                    .checked_add(offset)
                    .ok_or_else(|| anyhow::anyhow!("GLM KDA persistent ordinal overflow"))?,
            );
            gpu.copy_d2d_async(
                src,
                dst,
                GLM53_KDA_RECURRENT_ORDINAL_BYTES as usize,
                self.stream,
            )?;
        }

        // 3. Marker last, same stream, so success cannot precede the payload.
        KernelLaunch::new(gpu, self.kernel.marker)
            .grid([1, 1, 1])
            .block([1, 1, 1])
            .arg_ptr(slab_base)
            .arg_u32(slot as u32)
            .arg_u32(tag)
            .arg_u32(layer_id)
            .arg_u32(accepted)
            .arg_u32(exclusive_end)
            .arg_u32(capacity)
            .arg_u32(phase_incarnation)
            .arg_u64(owner_generation)
            .arg_u64(publish_nonce)
            .arg_u64(GLM53_KDA_RECURRENT_TOTAL_BYTES)
            .launch(self.stream)?;

        // 4. Exact-slot readback. CUDA enqueues this behind the work above and
        //    synchronizes, so the bytes reflect a retired commit.
        gpu.copy_d2h_on_stream(slot_ptr, readback, self.stream)?;
        Ok(())
    }

    /// Rejected decision: retire without touching persistent state. Present so
    /// the accepted-0 path is a call rather than an omission a caller can
    /// forget.
    pub(super) fn retire_rejected(self, accepted: u32) -> Result<()> {
        if accepted != 0 {
            bail!("GLM KDA recurrent retire requires a rejected decision");
        }
        Ok(())
    }
}

fn disjoint(a: DevicePtr, b: DevicePtr, bytes: u64) -> Result<bool> {
    disjoint_spans(a, bytes, b, bytes)
}

fn disjoint_spans(a: DevicePtr, a_bytes: u64, b: DevicePtr, b_bytes: u64) -> Result<bool> {
    let a_end =
        a.0.checked_add(a_bytes)
            .ok_or_else(|| anyhow::anyhow!("GLM KDA span overflow"))?;
    let b_end =
        b.0.checked_add(b_bytes)
            .ok_or_else(|| anyhow::anyhow!("GLM KDA span overflow"))?;
    Ok(a_end <= b.0 || b_end <= a.0)
}

#[cfg(test)]
#[path = "kda_recurrent_commit_tests.rs"]
mod kda_recurrent_commit_tests;
