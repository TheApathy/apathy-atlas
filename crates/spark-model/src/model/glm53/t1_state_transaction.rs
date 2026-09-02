// SPDX-License-Identifier: AGPL-3.0-only

//! Pure, effect-free B1/T1 state transaction for the future GLM-5.3 executor.

use std::fmt;

use spark_runtime::kv_cache::{Glm53DsaAppendPlan, Glm53DsaSequenceHandle};

const ALIGNMENT: u64 = 256;
const ALL_KDA_READY: u64 = (1u64 << GLM53_KDA_T1_LAYERS.len()) - 1;
const ALL_DSA_READY: u16 = (1u16 << GLM53_DSA_T1_LAYERS.len()) - 1;

pub const GLM53_TARGET_T1_LAYERS: u32 = 45;
pub const GLM53_T1_MAX_POSITIONS: u32 = 1_048_576;
pub const GLM53_TARGET_T1_TRANSACTION_BYTES: u64 = 156_008_192;
pub const GLM53_KDA_PERSISTENT_B1_WITH_DUMMY_BYTES: u64 = 311_951_360;
pub const GLM53_TARGET_T1_STATE_EXECUTION_IMPLEMENTED: bool = false;
pub const GLM53_KDA_T1_LAYERS: [u32; 34] = [
    0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25, 26, 28, 29, 30, 32, 33,
    34, 36, 37, 38, 40, 41, 42, 44,
];
pub const GLM53_DSA_T1_LAYERS: [u32; 11] =
    crate::layers::glm53_dsa_t1_transaction::GLM53_DSA_T1_LAYERS;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53T1StateRegion {
    pub offset_bytes: u64,
    pub payload_bytes: u64,
    pub allocation_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53T1StateLayout {
    pub kda_recurrent_f32: Glm53T1StateRegion,
    pub kda_conv_f32: Glm53T1StateRegion,
    pub dsa_latent_overlay: Glm53T1StateRegion,
    pub dsa_pool_tail_overlay: Glm53T1StateRegion,
    pub published_ends_u32: Glm53T1StateRegion,
    pub published_nonces_u64: Glm53T1StateRegion,
    pub logical_lengths_u32: Glm53T1StateRegion,
    pub total_bytes: u64,
}

impl Glm53T1StateLayout {
    pub fn exact() -> Result<Self, Glm53TargetT1Error> {
        build_layout()
    }

    pub fn validate(self) -> Result<(), Glm53TargetT1Error> {
        if self != build_layout()? {
            return Err(Glm53TargetT1Error("forged GLM target T1 state layout"));
        }
        Ok(())
    }

    pub fn kda_candidate_bytes(self) -> u64 {
        self.kda_recurrent_f32.payload_bytes + self.kda_conv_f32.payload_bytes
    }
}

fn build_layout() -> Result<Glm53T1StateLayout, Glm53TargetT1Error> {
    let mut cursor = 0;
    let kda_recurrent_f32 = place(&mut cursor, 142_606_336)?;
    let kda_conv_f32 = place(&mut cursor, 13_369_344)?;
    let dsa_latent_overlay = place(&mut cursor, 11_264)?;
    let dsa_pool_tail_overlay = place(&mut cursor, 20_224)?;
    let published_ends_u32 = place(&mut cursor, u64::from(GLM53_TARGET_T1_LAYERS) * 4)?;
    let published_nonces_u64 = place(&mut cursor, u64::from(GLM53_TARGET_T1_LAYERS) * 8)?;
    let logical_lengths_u32 = place(&mut cursor, u64::from(GLM53_TARGET_T1_LAYERS) * 4)?;
    if cursor != GLM53_TARGET_T1_TRANSACTION_BYTES {
        return Err(Glm53TargetT1Error("GLM target T1 arena byte drift"));
    }
    Ok(Glm53T1StateLayout {
        kda_recurrent_f32,
        kda_conv_f32,
        dsa_latent_overlay,
        dsa_pool_tail_overlay,
        published_ends_u32,
        published_nonces_u64,
        logical_lengths_u32,
        total_bytes: cursor,
    })
}

fn place(cursor: &mut u64, payload_bytes: u64) -> Result<Glm53T1StateRegion, Glm53TargetT1Error> {
    let offset_bytes = align_up(*cursor)?;
    let allocation_bytes = align_up(payload_bytes)?;
    *cursor = offset_bytes
        .checked_add(allocation_bytes)
        .ok_or(Glm53TargetT1Error("GLM target T1 arena extent overflow"))?;
    Ok(Glm53T1StateRegion {
        offset_bytes,
        payload_bytes,
        allocation_bytes,
    })
}

fn align_up(value: u64) -> Result<u64, Glm53TargetT1Error> {
    value
        .checked_add(ALIGNMENT - 1)
        .map(|rounded| rounded & !(ALIGNMENT - 1))
        .ok_or(Glm53TargetT1Error("GLM target T1 alignment overflow"))
}

#[derive(Debug, PartialEq, Eq)]
pub struct Glm53T1ExclusiveStreamLease {
    owner: Glm53DsaSequenceHandle,
    transaction_nonce: u64,
    stream: u64,
    capture_observed: bool,
}

impl Glm53T1ExclusiveStreamLease {
    pub(crate) fn new(
        owner: Glm53DsaSequenceHandle,
        transaction_nonce: u64,
        stream: u64,
        capture_observed: bool,
    ) -> Result<Self, Glm53TargetT1Error> {
        if transaction_nonce == 0 || stream == 0 || capture_observed {
            return Err(Glm53TargetT1Error(
                "GLM target T1 requires a nonzero nonce and exclusive noncapturing stream",
            ));
        }
        Ok(Self {
            owner,
            transaction_nonce,
            stream,
            capture_observed,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53KdaT1LayerReady {
    pub owner: Glm53DsaSequenceHandle,
    pub layer: u32,
    pub transaction_nonce: u64,
    pub exclusive_end: u32,
    pub stream: u64,
    pub capture_observed: bool,
    pub staged_h_ready: bool,
    pub staged_conv_ready: bool,
    pub persistent_state_untouched: bool,
    pub device_status: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53TargetT1DsaLayerReady {
    pub owner: Glm53DsaSequenceHandle,
    pub layer: u32,
    pub transaction_nonce: u64,
    pub exclusive_end: u32,
    pub stream: u64,
    pub capture_observed: bool,
    pub latent_ready: bool,
    pub index_ready: bool,
    pub current_visibility_ready: bool,
    pub device_status: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Glm53TargetT1DeviceOutcome {
    Success,
    LaunchFailure,
    AsyncFailure,
    StatusFailure(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Glm53TargetT1CpuOutcome {
    Success,
    Rejected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Glm53TargetT1Effect {
    RetireRejectedMarkers,
    CommitDsaIndexAllLayers,
    CommitDsaLatentAllLayers,
    CommitKdaAllLayers,
    FinalStreamSync,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53TargetT1DeviceEffect {
    pub kind: Glm53TargetT1Effect,
    pub owner: Glm53DsaSequenceHandle,
    pub transaction_nonce: u64,
    pub exclusive_end: u32,
    pub stream: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Glm53TargetT1Phase {
    Staging,
    AwaitingForwardSync,
    CommitEligible,
    Applying(Glm53TargetT1Effect),
    CpuPublicationPending,
    Complete,
    Poisoned,
}

#[derive(Debug, PartialEq, Eq)]
pub struct Glm53TargetT1CpuPublication {
    append: Glm53DsaAppendPlan,
    accepted: u32,
}

impl Glm53TargetT1CpuPublication {
    pub fn into_parts(self) -> (Glm53DsaAppendPlan, u32) {
        (self.append, self.accepted)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum Glm53TargetT1Transition {
    Device(Glm53TargetT1DeviceEffect),
    PublishCpu(Glm53TargetT1CpuPublication),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53TargetT1Error(&'static str);

impl fmt::Display for Glm53TargetT1Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.0)
    }
}

impl std::error::Error for Glm53TargetT1Error {}

pub struct Glm53TargetT1StateTransaction {
    append: Glm53DsaAppendPlan,
    stream: u64,
    kda_ready: u64,
    dsa_ready: u16,
    next_target_layer: u32,
    accepted: Option<u32>,
    phase: Glm53TargetT1Phase,
}

impl Glm53TargetT1StateTransaction {
    pub fn begin(
        append: Glm53DsaAppendPlan,
        lease: Glm53T1ExclusiveStreamLease,
        layout: Glm53T1StateLayout,
    ) -> Result<Self, Glm53TargetT1Error> {
        layout.validate()?;
        if append.handle.slot() != 0
            || lease.owner != append.handle
            || lease.transaction_nonce != append.nonce
            || (lease.stream == 0 || lease.capture_observed)
        {
            return Err(Glm53TargetT1Error(
                "GLM target T1 owner, slot, nonce, or stream lease mismatch",
            ));
        }
        let expected_end = append
            .start_position
            .checked_add(1)
            .ok_or(Glm53TargetT1Error("GLM target T1 position overflow"))?;
        if append.token_count != 1
            || append.nonce == 0
            || append.end_position != expected_end
            || expected_end > GLM53_T1_MAX_POSITIONS
            || append.first_pool_to_write != append.start_position / 4
            || append.complete_pools_to_write != expected_end / 4 - append.start_position / 4
            || append.initial_tail_len != (append.start_position % 4) as u8
            || append.final_tail_len != (expected_end % 4) as u8
        {
            return Err(Glm53TargetT1Error("forged GLM target B1/T1 append"));
        }
        Ok(Self {
            append,
            stream: lease.stream,
            kda_ready: 0,
            dsa_ready: 0,
            next_target_layer: 0,
            accepted: None,
            phase: Glm53TargetT1Phase::Staging,
        })
    }

    pub fn phase(&self) -> Glm53TargetT1Phase {
        self.phase
    }

    pub fn is_poisoned(&self) -> bool {
        self.phase == Glm53TargetT1Phase::Poisoned
    }

    pub fn poisoned_owner(&self) -> Option<Glm53DsaSequenceHandle> {
        self.is_poisoned().then_some(self.append.handle)
    }

    pub fn ready_counts(&self) -> (u32, u32) {
        (self.kda_ready.count_ones(), self.dsa_ready.count_ones())
    }

    pub fn record_kda_layer(
        &mut self,
        receipt: Glm53KdaT1LayerReady,
    ) -> Result<(), Glm53TargetT1Error> {
        let Some(ordinal) = GLM53_KDA_T1_LAYERS
            .iter()
            .position(|&layer| layer == receipt.layer)
        else {
            return self.poison("DSA layer presented as KDA readiness");
        };
        if self.phase != Glm53TargetT1Phase::Staging
            || receipt.layer != self.next_target_layer
            || !self.common_receipt_valid(
                receipt.owner,
                receipt.transaction_nonce,
                receipt.exclusive_end,
                receipt.stream,
                receipt.capture_observed,
                receipt.device_status,
            )
            || !receipt.staged_h_ready
            || !receipt.staged_conv_ready
            || !receipt.persistent_state_untouched
            || self.kda_ready & (1u64 << ordinal) != 0
        {
            return self.poison("stale, forged, reordered, or effectful KDA readiness");
        }
        self.kda_ready |= 1u64 << ordinal;
        self.advance_layer()
    }

    pub fn record_dsa_layer(
        &mut self,
        receipt: Glm53TargetT1DsaLayerReady,
    ) -> Result<(), Glm53TargetT1Error> {
        let Some(ordinal) = GLM53_DSA_T1_LAYERS
            .iter()
            .position(|&layer| layer == receipt.layer)
        else {
            return self.poison("KDA layer presented as DSA readiness");
        };
        if self.phase != Glm53TargetT1Phase::Staging
            || receipt.layer != self.next_target_layer
            || !self.common_receipt_valid(
                receipt.owner,
                receipt.transaction_nonce,
                receipt.exclusive_end,
                receipt.stream,
                receipt.capture_observed,
                receipt.device_status,
            )
            || self.dsa_ready & (1u16 << ordinal) != 0
        {
            return self.poison("stale, forged, or reordered DSA readiness");
        }
        if !receipt.latent_ready || !receipt.index_ready || !receipt.current_visibility_ready {
            return self.poison("incomplete DSA target-layer readiness");
        }
        self.dsa_ready |= 1u16 << ordinal;
        self.advance_layer()
    }

    pub fn confirm_forward_sync(
        &mut self,
        outcome: Glm53TargetT1DeviceOutcome,
    ) -> Result<(), Glm53TargetT1Error> {
        if self.phase != Glm53TargetT1Phase::AwaitingForwardSync
            || self.kda_ready != ALL_KDA_READY
            || self.dsa_ready != ALL_DSA_READY
            || outcome != Glm53TargetT1DeviceOutcome::Success
        {
            return self.poison("GLM target forward sync or all-layer readiness failed");
        }
        self.phase = Glm53TargetT1Phase::CommitEligible;
        Ok(())
    }

    pub fn decide(&mut self, accepted: u32) -> Result<Glm53TargetT1Transition, Glm53TargetT1Error> {
        if self.phase != Glm53TargetT1Phase::CommitEligible || accepted > 1 {
            return self.poison("GLM target T1 decision is premature or not in 0..=1");
        }
        self.accepted = Some(accepted);
        let effect = if accepted == 0 {
            Glm53TargetT1Effect::RetireRejectedMarkers
        } else {
            Glm53TargetT1Effect::CommitDsaIndexAllLayers
        };
        self.phase = Glm53TargetT1Phase::Applying(effect);
        Ok(Glm53TargetT1Transition::Device(self.bind_effect(effect)))
    }

    pub fn complete_effect(
        &mut self,
        effect: Glm53TargetT1DeviceEffect,
        outcome: Glm53TargetT1DeviceOutcome,
    ) -> Result<Glm53TargetT1Transition, Glm53TargetT1Error> {
        if self.phase != Glm53TargetT1Phase::Applying(effect.kind)
            || effect.owner != self.append.handle
            || effect.transaction_nonce != self.append.nonce
            || effect.exclusive_end != self.append.end_position
            || effect.stream != self.stream
        {
            return self.poison("GLM target T1 effects were omitted, duplicated, or reordered");
        }
        if outcome != Glm53TargetT1DeviceOutcome::Success {
            return self.poison("GLM target T1 launch, async execution, or status failed");
        }
        let next = match effect.kind {
            Glm53TargetT1Effect::RetireRejectedMarkers => Glm53TargetT1Effect::FinalStreamSync,
            Glm53TargetT1Effect::CommitDsaIndexAllLayers => {
                Glm53TargetT1Effect::CommitDsaLatentAllLayers
            }
            Glm53TargetT1Effect::CommitDsaLatentAllLayers => {
                Glm53TargetT1Effect::CommitKdaAllLayers
            }
            Glm53TargetT1Effect::CommitKdaAllLayers => Glm53TargetT1Effect::FinalStreamSync,
            Glm53TargetT1Effect::FinalStreamSync => return self.finish_device_transaction(),
        };
        self.phase = Glm53TargetT1Phase::Applying(next);
        Ok(Glm53TargetT1Transition::Device(self.bind_effect(next)))
    }

    pub fn confirm_cpu_publication(
        &mut self,
        outcome: Glm53TargetT1CpuOutcome,
    ) -> Result<(), Glm53TargetT1Error> {
        if self.phase != Glm53TargetT1Phase::CpuPublicationPending {
            return self.poison("GLM target CPU publication acknowledgement was reordered");
        }
        if outcome != Glm53TargetT1CpuOutcome::Success {
            return self.poison("GLM target CPU publication rejected owner generation or nonce");
        }
        self.phase = Glm53TargetT1Phase::Complete;
        Ok(())
    }

    fn common_receipt_valid(
        &self,
        owner: Glm53DsaSequenceHandle,
        nonce: u64,
        end: u32,
        stream: u64,
        capture_observed: bool,
        status: u32,
    ) -> bool {
        owner == self.append.handle
            && nonce == self.append.nonce
            && end == self.append.end_position
            && stream == self.stream
            && !capture_observed
            && status == 0
    }

    fn advance_layer(&mut self) -> Result<(), Glm53TargetT1Error> {
        self.next_target_layer = self
            .next_target_layer
            .checked_add(1)
            .ok_or(Glm53TargetT1Error("GLM target layer cursor overflow"))?;
        if self.next_target_layer == GLM53_TARGET_T1_LAYERS {
            if self.kda_ready != ALL_KDA_READY || self.dsa_ready != ALL_DSA_READY {
                return self.poison("GLM target reached layer45 without all readiness receipts");
            }
            self.phase = Glm53TargetT1Phase::AwaitingForwardSync;
        }
        Ok(())
    }

    fn finish_device_transaction(&mut self) -> Result<Glm53TargetT1Transition, Glm53TargetT1Error> {
        let accepted = self
            .accepted
            .ok_or(Glm53TargetT1Error("missing GLM target acceptance decision"))?;
        let append = self.append;
        self.phase = Glm53TargetT1Phase::CpuPublicationPending;
        Ok(Glm53TargetT1Transition::PublishCpu(
            Glm53TargetT1CpuPublication { append, accepted },
        ))
    }

    fn bind_effect(&self, kind: Glm53TargetT1Effect) -> Glm53TargetT1DeviceEffect {
        Glm53TargetT1DeviceEffect {
            kind,
            owner: self.append.handle,
            transaction_nonce: self.append.nonce,
            exclusive_end: self.append.end_position,
            stream: self.stream,
        }
    }

    fn poison<T>(&mut self, message: &'static str) -> Result<T, Glm53TargetT1Error> {
        self.phase = Glm53TargetT1Phase::Poisoned;
        Err(Glm53TargetT1Error(message))
    }
}

#[cfg(test)]
#[path = "t1_state_transaction_tests.rs"]
mod tests;
