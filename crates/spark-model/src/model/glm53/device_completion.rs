// SPDX-License-Identifier: AGPL-3.0-only

//! Dormant host contract for exact GLM-5.3 B1/T1 device completion receipts.

use std::sync::atomic::{AtomicU32, Ordering};

use spark_runtime::kv_cache::{Glm53DsaAppendPlan, Glm53DsaSequenceHandle};

use super::t1_state_transaction::{GLM53_DSA_T1_LAYERS, GLM53_KDA_T1_LAYERS};

pub(super) const GLM53_COMPLETION_ABI_VERSION: u32 = 1;
pub(super) const GLM53_COMPLETION_SUCCESS: u32 = 0x354d_4c47;
pub(super) const GLM53_COMPLETION_SLOT_BYTES: usize = 48;
pub(super) const GLM53_COMPLETION_SLOTS: usize = 45;
pub(super) const GLM53_COMPLETION_SLAB_PAYLOAD_BYTES: usize = 2_160;
pub(super) const GLM53_COMPLETION_SLAB_ALLOCATION_BYTES: usize = 2_304;
pub(super) const GLM53_KDA_STAGE_MARKERS: usize = 96;
pub(super) const GLM53_KDA_STAGE_MARKER_BYTES: usize = 768;
pub(super) const GLM53_COMPLETION_TOTAL_BYTES: usize = 3_072;

const ALIGNMENT: usize = 256;
const NO_U32: u32 = u32::MAX;
const CAPACITY: u32 = 1_048_576;
static NEXT_PHASE_INCARNATION: AtomicU32 = AtomicU32::new(1);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub(super) enum Glm53CompletionTag {
    KdaConvStage = 1,
    DsaCurrentVisibility = 2,
    DsaIndexCommit = 3,
    DsaLatentCommit = 4,
    KdaConvCommit = 5,
    KdaRecurrentCommit = 6,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(super) enum Glm53CompletionPhase {
    Forward,
    DsaIndexCommit,
    DsaLatentCommit,
    KdaConvCommit,
    KdaRecurrentCommit,
}

/// Linear issuer rooted in one exact runtime append and its Lane-A handle.
#[must_use = "completion authority owns phase-issuance state"]
pub(super) struct Glm53CompletionAuthority {
    append: Glm53DsaAppendPlan,
    stream: u64,
    next_phase: Glm53CompletionPhase,
}

impl Glm53CompletionAuthority {
    pub(super) fn claim(
        append: Glm53DsaAppendPlan,
        stream: u64,
        capture_observed: bool,
    ) -> Result<Self, Glm53CompletionAdmissionError> {
        let end = append
            .start_position
            .checked_add(1)
            .ok_or("GLM completion position overflow")?;
        if append.handle.slot() != 0
            || append.handle.generation() == 0
            || append.nonce == 0
            || stream == 0
            || capture_observed
            || append.token_count != 1
            || append.end_position != end
            || end > CAPACITY
            || append.first_pool_to_write != append.start_position / 4
            || append.complete_pools_to_write != end / 4 - append.start_position / 4
            || append.initial_tail_len != (append.start_position % 4) as u8
            || append.final_tail_len != (end % 4) as u8
        {
            return Err("invalid GLM completion owner, append, or stream");
        }
        let _device_identity = append.handle.device_identity();
        Ok(Self {
            append,
            stream,
            next_phase: Glm53CompletionPhase::Forward,
        })
    }

    pub(super) fn authorize(
        self,
        phase: Glm53CompletionPhase,
        accepted: Option<u32>,
    ) -> Result<Glm53CompletionAuthorization, Glm53CompletionAdmissionError> {
        if self.next_phase != phase || !decision_is_valid(phase, accepted) {
            return Err("GLM completion phase was forged, duplicated, or reordered");
        }
        let phase_incarnation = issue_phase_incarnation()?;
        let next_phase = match (phase, accepted) {
            (Glm53CompletionPhase::Forward, None) => Some(Glm53CompletionPhase::DsaIndexCommit),
            (Glm53CompletionPhase::DsaIndexCommit, Some(1)) => {
                Some(Glm53CompletionPhase::DsaLatentCommit)
            }
            (Glm53CompletionPhase::DsaLatentCommit, Some(1)) => {
                Some(Glm53CompletionPhase::KdaConvCommit)
            }
            (Glm53CompletionPhase::KdaConvCommit, Some(1)) => {
                Some(Glm53CompletionPhase::KdaRecurrentCommit)
            }
            _ => None,
        };
        Ok(Glm53CompletionAuthorization {
            identity: Glm53CompletionIdentity {
                owner: self.append.handle,
                transaction_nonce: self.append.nonce,
                exclusive_end: self.append.end_position,
                capacity: CAPACITY,
                stream: self.stream,
                phase,
                accepted,
                phase_incarnation,
            },
            next_authority: next_phase.map(|next_phase| Self { next_phase, ..self }),
        })
    }
}

#[must_use = "completion authorization must be consumed into one pending phase"]
pub(super) struct Glm53CompletionAuthorization {
    identity: Glm53CompletionIdentity,
    next_authority: Option<Glm53CompletionAuthority>,
}

impl Glm53CompletionAuthorization {
    pub(super) fn begin(self) -> Glm53CompletionPending {
        Glm53CompletionPending::from_authorization(self)
    }
}

struct Glm53CompletionIdentity {
    owner: Glm53DsaSequenceHandle,
    transaction_nonce: u64,
    exclusive_end: u32,
    capacity: u32,
    stream: u64,
    phase: Glm53CompletionPhase,
    accepted: Option<u32>,
    phase_incarnation: u32,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct Glm53CompletionExpectation {
    slot: usize,
    tag: Glm53CompletionTag,
    layer_id: u32,
    accepted: u32,
    exclusive_end: u32,
    capacity: u32,
    owner_generation: u64,
    transaction_nonce: u64,
    phase_incarnation: u32,
}

impl Glm53CompletionExpectation {
    pub(super) const fn kernel_fields(self) -> (usize, u32, u32, u32, u32, u32, u32, u64, u64) {
        (
            self.slot,
            self.tag as u32,
            self.layer_id,
            self.accepted,
            self.exclusive_end,
            self.capacity,
            self.phase_incarnation,
            self.owner_generation,
            self.transaction_nonce,
        )
    }

    fn matches(self, raw: RawCompletionSlot) -> bool {
        raw.abi_version == GLM53_COMPLETION_ABI_VERSION
            && raw.status == GLM53_COMPLETION_SUCCESS
            && raw.tag == self.tag as u32
            && raw.layer_id == self.layer_id
            && raw.accepted == self.accepted
            && raw.exclusive_end == self.exclusive_end
            && raw.capacity == self.capacity
            && raw.phase_incarnation == self.phase_incarnation
            && raw.owner_generation == self.owner_generation
            && raw.publish_nonce == self.transaction_nonce
    }
}

#[must_use = "pending completion must be verified or poisoned"]
pub(super) struct Glm53CompletionPending {
    identity: Glm53CompletionIdentity,
    next_authority: Option<Glm53CompletionAuthority>,
    first_slot: usize,
    slot_count: usize,
    issued_slots: u64,
    expectations: [Option<Glm53CompletionExpectation>; GLM53_COMPLETION_SLOTS],
}

impl Glm53CompletionPending {
    fn from_authorization(authorization: Glm53CompletionAuthorization) -> Self {
        let Glm53CompletionAuthorization {
            identity,
            next_authority,
        } = authorization;
        let mut expectations = [None; GLM53_COMPLETION_SLOTS];
        let (first_slot, slot_count) = match identity.phase {
            Glm53CompletionPhase::Forward => {
                for (slot, &layer) in GLM53_KDA_T1_LAYERS.iter().enumerate() {
                    expectations[slot] = Some(expectation(
                        slot,
                        Glm53CompletionTag::KdaConvStage,
                        layer,
                        NO_U32,
                        &identity,
                    ));
                }
                for (ordinal, &layer) in GLM53_DSA_T1_LAYERS.iter().enumerate() {
                    let slot = GLM53_KDA_T1_LAYERS.len() + ordinal;
                    expectations[slot] = Some(expectation(
                        slot,
                        Glm53CompletionTag::DsaCurrentVisibility,
                        layer,
                        NO_U32,
                        &identity,
                    ));
                }
                (0, GLM53_COMPLETION_SLOTS)
            }
            Glm53CompletionPhase::KdaConvCommit => {
                for (slot, &layer) in GLM53_KDA_T1_LAYERS.iter().enumerate() {
                    expectations[slot] = Some(expectation(
                        slot,
                        Glm53CompletionTag::KdaConvCommit,
                        layer,
                        1,
                        &identity,
                    ));
                }
                (0, GLM53_KDA_T1_LAYERS.len())
            }
            phase => {
                let (slot, tag) = match phase {
                    Glm53CompletionPhase::DsaIndexCommit => (0, Glm53CompletionTag::DsaIndexCommit),
                    Glm53CompletionPhase::DsaLatentCommit => {
                        (0, Glm53CompletionTag::DsaLatentCommit)
                    }
                    Glm53CompletionPhase::KdaRecurrentCommit => {
                        (34, Glm53CompletionTag::KdaRecurrentCommit)
                    }
                    _ => unreachable!("handled completion phase"),
                };
                expectations[slot] = Some(expectation(
                    slot,
                    tag,
                    NO_U32,
                    identity.accepted.expect("commit decision"),
                    &identity,
                ));
                (slot, 1)
            }
        };
        Self {
            identity,
            next_authority,
            first_slot,
            slot_count,
            issued_slots: 0,
            expectations,
        }
    }

    pub(super) const fn stream(&self) -> u64 {
        self.identity.stream
    }

    pub(super) const fn readback_offset_bytes(&self) -> usize {
        self.first_slot * GLM53_COMPLETION_SLOT_BYTES
    }

    pub(super) const fn readback_bytes(&self) -> usize {
        self.slot_count * GLM53_COMPLETION_SLOT_BYTES
    }

    pub(super) const fn marker_bytes(&self) -> usize {
        if matches!(self.identity.phase, Glm53CompletionPhase::Forward) {
            GLM53_KDA_STAGE_MARKER_BYTES
        } else {
            0
        }
    }

    pub(super) fn take_expectation(&mut self, slot: usize) -> Option<Glm53CompletionExpectation> {
        let bit = 1_u64.checked_shl(slot.try_into().ok()?)?;
        let expectation = self.expectations.get(slot).copied().flatten()?;
        (self.issued_slots & bit == 0).then(|| {
            self.issued_slots |= bit;
            expectation
        })
    }

    pub(super) fn verify_readback(
        self,
        readback: &[u8],
    ) -> Result<Glm53CompletionVerified, Glm53CompletionPoison> {
        let expected_mask = ((1_u64 << self.slot_count) - 1) << self.first_slot;
        if self.issued_slots != expected_mask {
            return Err(self.into_poison(Glm53CompletionFailure::UnissuedSlot));
        }
        if readback.len() != self.readback_bytes() {
            return Err(self.into_poison(Glm53CompletionFailure::ReadbackLength));
        }
        for slot in self.first_slot..self.first_slot + self.slot_count {
            let local = (slot - self.first_slot) * GLM53_COMPLETION_SLOT_BYTES;
            let raw =
                RawCompletionSlot::decode(&readback[local..local + GLM53_COMPLETION_SLOT_BYTES]);
            if !self.expectations[slot].is_some_and(|expected| expected.matches(raw)) {
                return Err(self.into_poison(Glm53CompletionFailure::SlotMismatch(slot as u32)));
            }
        }
        Ok(Glm53CompletionVerified {
            identity: self.identity,
            next_authority: self.next_authority,
            slot_count: self.slot_count,
        })
    }

    pub(super) fn poison(self, failure: Glm53CompletionFailure) -> Glm53CompletionPoison {
        self.into_poison(failure)
    }

    fn into_poison(self, failure: Glm53CompletionFailure) -> Glm53CompletionPoison {
        Glm53CompletionPoison {
            identity: self.identity,
            failure,
        }
    }
}

fn expectation(
    slot: usize,
    tag: Glm53CompletionTag,
    layer_id: u32,
    accepted: u32,
    identity: &Glm53CompletionIdentity,
) -> Glm53CompletionExpectation {
    Glm53CompletionExpectation {
        slot,
        tag,
        layer_id,
        accepted,
        exclusive_end: identity.exclusive_end,
        capacity: identity.capacity,
        owner_generation: identity.owner.generation(),
        transaction_nonce: identity.transaction_nonce,
        phase_incarnation: identity.phase_incarnation,
    }
}

#[must_use = "verified completion must be consumed by the effect driver"]
pub(super) struct Glm53CompletionVerified {
    identity: Glm53CompletionIdentity,
    next_authority: Option<Glm53CompletionAuthority>,
    slot_count: usize,
}

impl Glm53CompletionVerified {
    pub(super) const fn parts(
        &self,
    ) -> (
        Glm53DsaSequenceHandle,
        Glm53CompletionPhase,
        Option<u32>,
        u32,
        u64,
        usize,
    ) {
        (
            self.identity.owner,
            self.identity.phase,
            self.identity.accepted,
            self.identity.phase_incarnation,
            self.identity.transaction_nonce,
            self.slot_count,
        )
    }

    pub(super) fn into_next(self) -> Option<Glm53CompletionAuthority> {
        self.next_authority
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Glm53CompletionFailure {
    Clear,
    Launch,
    ReadbackOrSync,
    ReadbackLength,
    UnissuedSlot,
    SlotMismatch(u32),
}

#[must_use = "poisoned completion retains the exact owner and incarnation"]
pub(super) struct Glm53CompletionPoison {
    identity: Glm53CompletionIdentity,
    failure: Glm53CompletionFailure,
}

impl Glm53CompletionPoison {
    pub(super) const fn owner(&self) -> Glm53DsaSequenceHandle {
        self.identity.owner
    }

    pub(super) const fn phase_incarnation(&self) -> u32 {
        self.identity.phase_incarnation
    }

    pub(super) const fn failure(&self) -> Glm53CompletionFailure {
        self.failure
    }
}

fn decision_is_valid(phase: Glm53CompletionPhase, accepted: Option<u32>) -> bool {
    matches!(
        (phase, accepted),
        (Glm53CompletionPhase::Forward, None)
            | (Glm53CompletionPhase::DsaIndexCommit, Some(0 | 1))
            | (
                Glm53CompletionPhase::DsaLatentCommit
                    | Glm53CompletionPhase::KdaConvCommit
                    | Glm53CompletionPhase::KdaRecurrentCommit,
                Some(1)
            )
    )
}

fn issue_phase_incarnation() -> Result<u32, Glm53CompletionAdmissionError> {
    NEXT_PHASE_INCARNATION
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            incarnation_step(current).map(|(_, next)| next)
        })
        .map_err(|_| "GLM completion phase incarnation exhausted")
}

fn incarnation_step(current: u32) -> Option<(u32, u32)> {
    (current != 0)
        .then_some(current)
        .zip(current.checked_add(1))
}

#[derive(Clone, Copy)]
#[repr(C)]
struct RawCompletionSlot {
    abi_version: u32,
    status: u32,
    tag: u32,
    layer_id: u32,
    accepted: u32,
    exclusive_end: u32,
    capacity: u32,
    phase_incarnation: u32,
    owner_generation: u64,
    publish_nonce: u64,
}

impl RawCompletionSlot {
    fn decode(bytes: &[u8]) -> Self {
        let u32_at = |offset| u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap());
        let u64_at = |offset| u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
        Self {
            abi_version: u32_at(0),
            status: u32_at(4),
            tag: u32_at(8),
            layer_id: u32_at(12),
            accepted: u32_at(16),
            exclusive_end: u32_at(20),
            capacity: u32_at(24),
            phase_incarnation: u32_at(28),
            owner_generation: u64_at(32),
            publish_nonce: u64_at(40),
        }
    }
}

pub(super) type Glm53CompletionAdmissionError = &'static str;

const _: () = {
    assert!(GLM53_COMPLETION_SLOTS * GLM53_COMPLETION_SLOT_BYTES == 2_160);
    assert!(GLM53_COMPLETION_SLAB_ALLOCATION_BYTES % ALIGNMENT == 0);
    assert!(GLM53_KDA_STAGE_MARKERS * 8 == GLM53_KDA_STAGE_MARKER_BYTES);
    assert!(GLM53_COMPLETION_SLAB_ALLOCATION_BYTES + GLM53_KDA_STAGE_MARKER_BYTES == 3_072);
};

#[cfg(test)]
#[path = "device_completion_tests.rs"]
mod tests;
