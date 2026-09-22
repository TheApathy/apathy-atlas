// SPDX-License-Identifier: AGPL-3.0-only

//! Pure B1/T1 transaction contract for the future GLM-5.3 DSA composer.
//!
//! This module launches nothing and does not mutate a cache. It seals the
//! ordering required around the effectful primitives that are still missing.

use std::fmt;

use spark_runtime::kv_cache::{Glm53DsaAppendPlan, Glm53DsaSequenceHandle};

pub const GLM53_DSA_T1_LAYERS: [u32; 11] = [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43];
pub const GLM53_DSA_T1_MAX_POSITIONS: u32 = 1_048_576;
pub const GLM53_DSA_T1_EXECUTION_IMPLEMENTED: bool = false;
const ALL_LAYERS_READY: u16 = (1u16 << GLM53_DSA_T1_LAYERS.len()) - 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Glm53DsaT1MissingCapability {
    ExactNormAndIndexerProjection,
    CurrentTokenVisibilityMaterializer,
    AllLayerIndexCommit,
    DsaLayerComposer,
    CompleteScratchPlan,
    FalliblePoisonOwner,
}

pub const GLM53_DSA_T1_MISSING_CAPABILITIES: [Glm53DsaT1MissingCapability; 6] = [
    Glm53DsaT1MissingCapability::ExactNormAndIndexerProjection,
    Glm53DsaT1MissingCapability::CurrentTokenVisibilityMaterializer,
    Glm53DsaT1MissingCapability::AllLayerIndexCommit,
    Glm53DsaT1MissingCapability::DsaLayerComposer,
    Glm53DsaT1MissingCapability::CompleteScratchPlan,
    Glm53DsaT1MissingCapability::FalliblePoisonOwner,
];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53DsaT1Visibility {
    pub committed_len: u32,
    pub visible_len: u32,
    pub query_position: u32,
    pub future_latent_row: u32,
    pub future_pool_row: Option<u32>,
    pub committed_complete_pools: u32,
    pub visible_complete_pools: u32,
    pub score_pool_count: u32,
    pub initial_tail_len: u8,
    pub staged_tail_len: u8,
    /// The post-token tail is transaction-private until acceptance.
    pub use_staged_tail: bool,
    /// A composer must never overwrite retry-critical committed tail rows here.
    pub persistent_tail_write_allowed: bool,
}

/// Visibility permission for exactly one staged DSA layer.
///
/// This is emitted only after that layer's latent, index, and future-slot
/// materialization receipts all validate against the active transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53DsaT1LayerVisibility {
    pub layer: u32,
    pub view: Glm53DsaT1Visibility,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53DsaT1LayerReady {
    pub owner: Glm53DsaSequenceHandle,
    pub layer: u32,
    pub transaction_nonce: u64,
    pub exclusive_end: u32,
    pub latent_ready: bool,
    pub index_ready: bool,
    pub current_visibility_ready: bool,
    pub device_status: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Glm53DsaT1DeviceOutcome {
    Success,
    LaunchFailure,
    AsyncFailure,
    StatusFailure(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Glm53DsaT1Effect {
    /// Accepted zero: retire markers without touching persistent tail or length.
    RetireRejectedMarkers,
    /// Accepted one: validate all 11 layers, then publish the staged index tail.
    CommitIndexAllLayers,
    /// Must follow index commit; uses the reviewed latent commit as batch 11.
    CommitLatentAllLayers,
    /// Required after every accepted/rejected device effect and before CPU state.
    FinalStreamSync,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Glm53DsaT1CpuPublication {
    append: Glm53DsaAppendPlan,
    accepted: u32,
}

impl Glm53DsaT1CpuPublication {
    /// The caller may invoke `Glm53DsaCache::commit_append` only with these parts.
    pub fn into_parts(self) -> (Glm53DsaAppendPlan, u32) {
        (self.append, self.accepted)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum Glm53DsaT1Transition {
    Device(Glm53DsaT1Effect),
    PublishCpu(Glm53DsaT1CpuPublication),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Glm53DsaT1CpuOutcome {
    Success,
    /// Includes stale generation, stale nonce, or any cache publication error.
    Rejected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Glm53DsaT1Phase {
    Staging,
    AwaitingForwardSync,
    CommitEligible,
    Applying(Glm53DsaT1Effect),
    CpuPublicationPending,
    Complete,
    Poisoned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53DsaT1Error(&'static str);

impl fmt::Display for Glm53DsaT1Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.0)
    }
}

impl std::error::Error for Glm53DsaT1Error {}

#[derive(Debug, PartialEq, Eq)]
pub struct Glm53DsaT1Transaction {
    append: Glm53DsaAppendPlan,
    visibility: Glm53DsaT1Visibility,
    ready_mask: u16,
    next_layer: usize,
    accepted: Option<u32>,
    phase: Glm53DsaT1Phase,
}

impl Glm53DsaT1Transaction {
    pub fn begin(append: Glm53DsaAppendPlan) -> Result<Self, Glm53DsaT1Error> {
        let expected_end = append
            .start_position
            .checked_add(1)
            .ok_or(Glm53DsaT1Error("DSA T1 position overflow"))?;
        let before_pools = append.start_position / 4;
        let after_pools = expected_end / 4;
        let initial_tail = (append.start_position % 4) as u8;
        let final_tail = (expected_end % 4) as u8;
        if append.token_count != 1
            || append.nonce == 0
            || append.end_position != expected_end
            || expected_end > GLM53_DSA_T1_MAX_POSITIONS
            || append.first_pool_to_write != before_pools
            || append.complete_pools_to_write != after_pools - before_pools
            || append.initial_tail_len != initial_tail
            || append.final_tail_len != final_tail
        {
            return Err(Glm53DsaT1Error("invalid or forged B1/T1 DSA append plan"));
        }
        Ok(Self {
            append,
            visibility: Glm53DsaT1Visibility {
                committed_len: append.start_position,
                visible_len: expected_end,
                query_position: append.start_position,
                future_latent_row: append.start_position,
                future_pool_row: (initial_tail == 3).then_some(before_pools),
                committed_complete_pools: before_pools,
                visible_complete_pools: after_pools,
                score_pool_count: after_pools.max(1),
                initial_tail_len: initial_tail,
                staged_tail_len: final_tail,
                use_staged_tail: true,
                persistent_tail_write_allowed: false,
            },
            ready_mask: 0,
            next_layer: 0,
            accepted: None,
            phase: Glm53DsaT1Phase::Staging,
        })
    }

    pub fn phase(&self) -> Glm53DsaT1Phase {
        self.phase
    }

    /// Arithmetic geometry only; this does not authorize a layer to consume
    /// the future row. That permission is returned by `record_layer_ready`.
    pub fn planned_visibility(&self) -> Glm53DsaT1Visibility {
        self.visibility
    }

    pub fn ready_mask(&self) -> u16 {
        self.ready_mask
    }

    pub fn is_poisoned(&self) -> bool {
        self.phase == Glm53DsaT1Phase::Poisoned
    }

    /// Identifies the complete sequence allocation that a future fallible
    /// owner must invalidate after any transaction failure.
    pub fn poisoned_owner(&self) -> Option<Glm53DsaSequenceHandle> {
        self.is_poisoned().then_some(self.append.handle)
    }

    pub fn record_layer_ready(
        &mut self,
        receipt: Glm53DsaT1LayerReady,
    ) -> Result<Glm53DsaT1LayerVisibility, Glm53DsaT1Error> {
        if self.phase != Glm53DsaT1Phase::Staging {
            return self.poison("DSA layer readiness arrived outside staging");
        }
        let Some(&expected_layer) = GLM53_DSA_T1_LAYERS.get(self.next_layer) else {
            return self.poison("too many DSA layer readiness receipts");
        };
        if receipt.owner != self.append.handle
            || receipt.layer != expected_layer
            || receipt.transaction_nonce != self.append.nonce
            || receipt.exclusive_end != self.append.end_position
            || !receipt.latent_ready
            || !receipt.index_ready
            || !receipt.current_visibility_ready
            || receipt.device_status != 0
        {
            return self.poison("stale, reordered, incomplete, or failed DSA layer receipt");
        }
        self.ready_mask |= 1u16 << self.next_layer;
        self.next_layer += 1;
        if self.ready_mask == ALL_LAYERS_READY {
            self.phase = Glm53DsaT1Phase::AwaitingForwardSync;
        }
        Ok(Glm53DsaT1LayerVisibility {
            layer: receipt.layer,
            view: self.visibility,
        })
    }

    pub fn confirm_forward_sync(
        &mut self,
        outcome: Glm53DsaT1DeviceOutcome,
    ) -> Result<(), Glm53DsaT1Error> {
        if self.phase != Glm53DsaT1Phase::AwaitingForwardSync || self.ready_mask != ALL_LAYERS_READY
        {
            return self.poison("forward sync attempted before all 11 DSA layers were ready");
        }
        if outcome != Glm53DsaT1DeviceOutcome::Success {
            return self.poison("DSA forward synchronization or device status failed");
        }
        self.phase = Glm53DsaT1Phase::CommitEligible;
        Ok(())
    }

    pub fn decide(&mut self, accepted: u32) -> Result<Glm53DsaT1Transition, Glm53DsaT1Error> {
        if self.phase != Glm53DsaT1Phase::CommitEligible || accepted > 1 {
            return self.poison("DSA T1 decision is premature or not in 0..=1");
        }
        self.accepted = Some(accepted);
        let effect = if accepted == 0 {
            Glm53DsaT1Effect::RetireRejectedMarkers
        } else {
            Glm53DsaT1Effect::CommitIndexAllLayers
        };
        self.phase = Glm53DsaT1Phase::Applying(effect);
        Ok(Glm53DsaT1Transition::Device(effect))
    }

    pub fn complete_effect(
        &mut self,
        effect: Glm53DsaT1Effect,
        outcome: Glm53DsaT1DeviceOutcome,
    ) -> Result<Glm53DsaT1Transition, Glm53DsaT1Error> {
        if self.phase != Glm53DsaT1Phase::Applying(effect) {
            return self.poison("DSA commit effects were omitted, duplicated, or reordered");
        }
        if outcome != Glm53DsaT1DeviceOutcome::Success {
            return self.poison("DSA commit launch, asynchronous execution, or status failed");
        }
        let next = match effect {
            Glm53DsaT1Effect::RetireRejectedMarkers => Glm53DsaT1Effect::FinalStreamSync,
            Glm53DsaT1Effect::CommitIndexAllLayers => Glm53DsaT1Effect::CommitLatentAllLayers,
            Glm53DsaT1Effect::CommitLatentAllLayers => Glm53DsaT1Effect::FinalStreamSync,
            Glm53DsaT1Effect::FinalStreamSync => {
                let accepted = self
                    .accepted
                    .ok_or(Glm53DsaT1Error("missing DSA acceptance decision"))?;
                self.phase = Glm53DsaT1Phase::CpuPublicationPending;
                return Ok(Glm53DsaT1Transition::PublishCpu(Glm53DsaT1CpuPublication {
                    append: self.append,
                    accepted,
                }));
            }
        };
        self.phase = Glm53DsaT1Phase::Applying(next);
        Ok(Glm53DsaT1Transition::Device(next))
    }

    /// Records the result of the final CPU cache effect. A rejection after
    /// device effects makes the whole sequence allocation unusable.
    pub fn confirm_cpu_publication(
        &mut self,
        outcome: Glm53DsaT1CpuOutcome,
    ) -> Result<(), Glm53DsaT1Error> {
        if self.phase != Glm53DsaT1Phase::CpuPublicationPending {
            return self.poison("CPU DSA publication acknowledgement was reordered");
        }
        if outcome != Glm53DsaT1CpuOutcome::Success {
            return self.poison("CPU DSA publication rejected a stale generation or nonce");
        }
        self.phase = Glm53DsaT1Phase::Complete;
        Ok(())
    }

    fn poison<T>(&mut self, reason: &'static str) -> Result<T, Glm53DsaT1Error> {
        self.phase = Glm53DsaT1Phase::Poisoned;
        Err(Glm53DsaT1Error(reason))
    }
}

#[cfg(test)]
#[path = "glm53_dsa_t1_transaction_tests.rs"]
mod tests;
