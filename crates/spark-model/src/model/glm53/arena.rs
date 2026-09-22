// SPDX-License-Identifier: AGPL-3.0-only

//! Side-effect-free arena accounting for the first GLM-5.3 target runtime.

use anyhow::{Context, Result, ensure};

use crate::layers::{Glm53TargetGeometry, Glm53TargetSchedule, Glm53TargetWorkspace};
use crate::weight_loader::{GLM53_MAX_CONTEXT_TOKENS, Glm53ContextPlan, Glm53DsaStorage};

const ALIGNMENT_BYTES: u64 = 256;
pub const GLM53_MHC_EXPANDED_F32_BYTES: u64 = 141_557_760;
pub const GLM53_FULL_1M_PERSISTENT_BYTES: u64 = 12_864_209_664;
pub const GLM53_T1_WORKSPACE_BYTES: u64 = 90_624;
pub const GLM53_T1_TRANSACTION_BYTES: u64 = 156_008_192;
pub const GLM53_T1_TRANSIENT_BYTES: u64 = 1_139_200;
pub const GLM53_DFLASH_CAPTURE_BYTES: u64 = 327_680;
pub const GLM53_KNOWN_ARENA_BYTES: u64 = 13_163_242_496;
pub const GLM53_Q2_TENSOR_BYTES: u64 = 108_710_550_904;
pub const GLM53_KNOWN_Q2_RESIDENCY_BYTES: u64 = 121_873_793_400;
/// UD-IQ2_XXS tensor payload, from the published shard headers.
pub const GLM53_IQ2_XXS_TENSOR_BYTES: u64 = 101_835_431_288;
/// UD-IQ2_XXS weights plus the full-1M known arena.
///
/// This is the profile that actually fits one 128 GB Spark. Atlas holds every
/// tensor resident rather than mmap-ing it, so the comparison that matters is
/// against total RAM, not against what llama.cpp gets away with via page cache:
/// Q2_K_XL lands at 113.50 GiB of a ~119 GiB box, leaving 5.50 GiB for the host
/// and the CUDA context, whereas IQ2_XXS lands at 107.10 GiB and leaves 11.90.
/// The 6.40 GiB the quant swap recovers is what makes full 1M context reachable
/// without shrinking the DSA latent cache.
pub const GLM53_KNOWN_IQ2_XXS_RESIDENCY_BYTES: u64 = 114_998_673_784;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53ArenaRegion {
    pub offset_bytes: u64,
    pub allocation_bytes: u64,
}

impl Glm53ArenaRegion {
    fn end(self) -> Result<u64> {
        self.offset_bytes
            .checked_add(self.allocation_bytes)
            .context("GLM target arena extent overflow")
    }
}

/// Exact checked lower-bound allocations for B1/T1/full-1M/BF16.
///
/// The transaction region conservatively retains every layer's speculative
/// state at once. The transient region is one liveness-reused arena and begins
/// with the 90,624-byte target-schedule workspace. This still excludes device
/// allocator overhead and effectful ABI deltas, so it is not a runnable peak.
/// [`super::kernels::Glm53KernelAdmission`] keeps execution disabled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53ArenaPlan {
    pub context: Glm53ContextPlan,
    pub mhc_expanded_f32: Glm53ArenaRegion,
    pub t1_transaction: Glm53ArenaRegion,
    pub transient: Glm53ArenaRegion,
    pub dflash_captures: Glm53ArenaRegion,
    pub workspace: Glm53TargetWorkspace,
    pub known_bytes: u64,
}

impl Glm53ArenaPlan {
    /// The pinned full-1M plan.
    pub fn exact_full_1m_b1() -> Result<Self> {
        let plan = Self::for_context(GLM53_MAX_CONTEXT_TOKENS)?;
        ensure!(
            plan.context.total_bytes == GLM53_FULL_1M_PERSISTENT_BYTES,
            "GLM target persistent full-1M state drift"
        );
        plan.validate()?;
        Ok(plan)
    }

    /// The same layout at a smaller context.
    ///
    /// Full 1M costs 12.26 GiB of arena, and on a 119 GiB box that is 95 MiB
    /// *more* than the UD-IQ2_XXS checkpoint leaves free — measured, not
    /// estimated: the device preflight refused at
    /// `115,000,477,564 required` vs `114,901,217,280 free`. The overrun is
    /// almost entirely the DSA latent cache, which is linear in context, so a
    /// shorter context is what makes the model loadable at all.
    ///
    /// Only the persistent context region scales. Every other region — the mHC
    /// expansion, the T1 transaction, the transient arena, the capture slots —
    /// is context-independent and keeps its pinned size, so this cannot be used
    /// to quietly shrink anything else.
    pub fn for_context(positions: u32) -> Result<Self> {
        ensure!(
            positions > 0 && positions <= GLM53_MAX_CONTEXT_TOKENS,
            "GLM context {positions} is outside 1..={GLM53_MAX_CONTEXT_TOKENS}"
        );
        let context = Glm53ContextPlan::new(1, positions, Glm53DsaStorage::BF16)?;
        ensure!(context.dense_kv_bytes == 0, "dense DSA K/V is forbidden");

        let schedule = Glm53TargetSchedule::new(Glm53TargetGeometry::exact(1))?;
        let workspace = schedule.workspace;
        ensure!(
            workspace.arena_bytes == GLM53_T1_WORKSPACE_BYTES,
            "GLM target T1 workspace drift"
        );

        let mut cursor = align_up(context.total_bytes)?;
        let mhc_expanded_f32 = place(&mut cursor, GLM53_MHC_EXPANDED_F32_BYTES)?;
        let t1_transaction = place(&mut cursor, GLM53_T1_TRANSACTION_BYTES)?;
        let transient = place(&mut cursor, GLM53_T1_TRANSIENT_BYTES)?;
        let dflash_captures = place(&mut cursor, GLM53_DFLASH_CAPTURE_BYTES)?;
        let plan = Self {
            context,
            mhc_expanded_f32,
            t1_transaction,
            transient,
            dflash_captures,
            workspace,
            known_bytes: cursor,
        };
        plan.validate_layout()?;
        Ok(plan)
    }

    /// Full validation, including the pinned full-1M totals.
    pub fn validate(self) -> Result<()> {
        self.validate_layout()?;
        ensure!(
            self.context.positions == GLM53_MAX_CONTEXT_TOKENS,
            "GLM target admission must preserve full 1M context"
        );
        ensure!(
            self.known_bytes == GLM53_KNOWN_ARENA_BYTES,
            "GLM target known arena total drift"
        );
        ensure!(
            GLM53_Q2_TENSOR_BYTES
                .checked_add(self.known_bytes)
                .context("GLM target known Q2 residency overflow")?
                == GLM53_KNOWN_Q2_RESIDENCY_BYTES,
            "GLM target known Q2 residency subtotal drift"
        );
        Ok(())
    }

    /// Layout validation that holds at any admitted context.
    pub fn validate_layout(self) -> Result<()> {
        let expected_context =
            Glm53ContextPlan::new(1, self.context.positions, Glm53DsaStorage::BF16)?;
        ensure!(
            self.context == expected_context,
            "GLM target BF16 context layout drift"
        );
        let expected_workspace = Glm53TargetSchedule::new(Glm53TargetGeometry::exact(1))?.workspace;
        self.workspace.validate()?;
        ensure!(
            self.workspace == expected_workspace,
            "GLM target B1/T1 workspace layout drift"
        );
        ensure!(
            self.context.storage == Glm53DsaStorage::BF16,
            "GLM target runtime admits BF16 DSA state only"
        );
        ensure!(
            self.mhc_expanded_f32.offset_bytes == self.context.total_bytes,
            "GLM target mHC expansion placement drift"
        );
        ensure!(
            self.mhc_expanded_f32.allocation_bytes == GLM53_MHC_EXPANDED_F32_BYTES,
            "GLM target mHC expansion size drift"
        );
        ensure!(
            self.t1_transaction.offset_bytes == self.mhc_expanded_f32.end()?,
            "GLM target transaction placement drift"
        );
        ensure!(
            self.t1_transaction.allocation_bytes == GLM53_T1_TRANSACTION_BYTES,
            "GLM target transaction size drift"
        );
        ensure!(
            self.transient.offset_bytes == self.t1_transaction.end()?
                && self.transient.allocation_bytes == GLM53_T1_TRANSIENT_BYTES,
            "GLM target reusable transient arena drift"
        );
        ensure!(
            self.workspace.arena_bytes == GLM53_T1_WORKSPACE_BYTES
                && self.workspace.arena_bytes <= self.transient.allocation_bytes,
            "GLM target schedule workspace does not fit the transient arena"
        );
        ensure!(
            self.dflash_captures.offset_bytes == self.transient.end()?
                && self.dflash_captures.allocation_bytes == GLM53_DFLASH_CAPTURE_BYTES,
            "GLM target retained-capture arena drift"
        );
        ensure!(
            self.known_bytes == self.dflash_captures.end()?,
            "GLM target known arena total drift"
        );
        for region in [
            self.mhc_expanded_f32,
            self.t1_transaction,
            self.transient,
            self.dflash_captures,
        ] {
            ensure!(
                region.offset_bytes % ALIGNMENT_BYTES == 0
                    && region.allocation_bytes % ALIGNMENT_BYTES == 0,
                "GLM target arena alignment drift"
            );
        }
        Ok(())
    }
}

fn align_up(value: u64) -> Result<u64> {
    value
        .checked_add(ALIGNMENT_BYTES - 1)
        .map(|rounded| rounded & !(ALIGNMENT_BYTES - 1))
        .context("GLM target arena alignment overflow")
}

fn place(cursor: &mut u64, bytes: u64) -> Result<Glm53ArenaRegion> {
    let offset_bytes = align_up(*cursor)?;
    let allocation_bytes = align_up(bytes)?;
    *cursor = offset_bytes
        .checked_add(allocation_bytes)
        .context("GLM target arena placement overflow")?;
    Ok(Glm53ArenaRegion {
        offset_bytes,
        allocation_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_known_full_1m_arena_is_pinned_without_dense_kv() {
        let plan = Glm53ArenaPlan::exact_full_1m_b1().unwrap();
        assert_eq!(plan.context.kda_bytes(), 311_951_360);
        assert_eq!(plan.context.dsa_bytes(), 12_552_258_304);
        assert_eq!(plan.context.dense_kv_bytes, 0);
        assert_eq!(plan.mhc_expanded_f32.offset_bytes, 12_864_209_664);
        assert_eq!(plan.t1_transaction.offset_bytes, 13_005_767_424);
        assert_eq!(plan.transient.offset_bytes, 13_161_775_616);
        assert_eq!(plan.dflash_captures.offset_bytes, 13_162_914_816);
        assert_eq!(plan.known_bytes, 13_163_242_496);
        assert_eq!(
            GLM53_Q2_TENSOR_BYTES + plan.known_bytes,
            GLM53_KNOWN_Q2_RESIDENCY_BYTES
        );
        // The same full-1M arena under the UD-IQ2_XXS profile. Both residency
        // totals are pinned against the one arena so a change to either the
        // arena or a checkpoint payload cannot silently move the fit.
        assert_eq!(
            GLM53_IQ2_XXS_TENSOR_BYTES + plan.known_bytes,
            GLM53_KNOWN_IQ2_XXS_RESIDENCY_BYTES
        );
        // IQ2_XXS must stay strictly the smaller resident footprint, since that
        // is the entire reason the profile exists.
        assert!(GLM53_KNOWN_IQ2_XXS_RESIDENCY_BYTES < GLM53_KNOWN_Q2_RESIDENCY_BYTES);
        // A 128 GB Spark reports ~119 GiB usable. Refuse to let the pinned
        // profile drift back above a 10 GiB host/CUDA-context reserve.
        const USABLE_BYTES: u64 = 119 * (1 << 30);
        const HOST_RESERVE_BYTES: u64 = 10 * (1 << 30);
        assert!(
            GLM53_KNOWN_IQ2_XXS_RESIDENCY_BYTES + HOST_RESERVE_BYTES <= USABLE_BYTES,
            "UD-IQ2_XXS no longer leaves a 10 GiB host reserve"
        );
        assert!(
            GLM53_KNOWN_Q2_RESIDENCY_BYTES + HOST_RESERVE_BYTES > USABLE_BYTES,
            "Q2_K_XL unexpectedly fits; re-derive which profile to serve"
        );
    }

    #[test]
    fn exact_regions_are_aligned_ordered_and_nonoverlapping() {
        let plan = Glm53ArenaPlan::exact_full_1m_b1().unwrap();
        let regions = [
            plan.mhc_expanded_f32,
            plan.t1_transaction,
            plan.transient,
            plan.dflash_captures,
        ];
        for region in regions {
            assert_eq!(region.offset_bytes % ALIGNMENT_BYTES, 0);
            assert_eq!(region.allocation_bytes % ALIGNMENT_BYTES, 0);
        }
        for pair in regions.windows(2) {
            assert_eq!(pair[0].end().unwrap(), pair[1].offset_bytes);
        }
        assert_eq!(plan.workspace.arena_bytes, 90_624);
        assert!(plan.workspace.arena_bytes < plan.transient.allocation_bytes);
        assert!(align_up(u64::MAX).is_err());
        let mut cursor = u64::MAX - 255;
        assert!(place(&mut cursor, 256).is_err());
    }

    #[test]
    fn nested_public_context_and_workspace_forgery_is_rejected() {
        let exact = Glm53ArenaPlan::exact_full_1m_b1().unwrap();

        let mut dense_kv = exact;
        dense_kv.context.dense_kv_bytes = 1;
        assert!(dense_kv.validate().is_err());

        let mut context_region = exact;
        context_region.context.dsa_tail_validity.offset_bytes += 256;
        assert!(context_region.validate().is_err());

        let mut workspace_geometry = exact;
        workspace_geometry.workspace.geometry.hidden_size = 2048;
        assert!(workspace_geometry.validate().is_err());

        let mut workspace_region = exact;
        workspace_region.workspace.hidden_b.offset_bytes += 256;
        assert!(workspace_region.validate().is_err());
    }
}
