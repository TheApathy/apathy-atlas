// SPDX-License-Identifier: AGPL-3.0-only

//! Dormant fail-closed scheduler contract for GLM-5.3 DFlash2.
//!
//! The live scheduler is intentionally untouched. Future integration must
//! obtain this unit's typed admission before any prefill, allocation, target,
//! or drafter effect and must preserve its target->capture->emit ordering.

use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Result, bail, ensure};

pub const GLM53_DFLASH2_BLOCK_TOKENS: u8 = 8;
pub const GLM53_DFLASH2_RETURNED_DRAFTS: u8 = 7;
pub const GLM53_DFLASH2_WINDOW: u16 = 2_048;
pub const GLM53_MAX_POSITION_EXCLUSIVE: u64 = 1_048_576;
pub const GLM53_MAX_POSITION_INCLUSIVE: u64 = GLM53_MAX_POSITION_EXCLUSIVE - 1;

static NEXT_POLICY_OWNER_ID: AtomicU64 = AtomicU64::new(1);

fn allocate_owner_from(counter: &AtomicU64) -> Result<u64> {
    counter
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |owner| {
            (owner != 0).then(|| owner.checked_add(1)).flatten()
        })
        .map_err(|_| anyhow::anyhow!("GLM DFlash2 policy owner ID exhausted"))
}

fn allocate_policy_owner_id() -> Result<u64> {
    allocate_owner_from(&NEXT_POLICY_OWNER_ID)
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PreAdmissionEffects {
    pub prefills: u32,
    pub sequence_allocations: u32,
    pub target_effects: u32,
    pub drafter_effects: u32,
}

impl PreAdmissionEffects {
    const fn is_empty(self) -> bool {
        self.prefills == 0
            && self.sequence_allocations == 0
            && self.target_effects == 0
            && self.drafter_effects == 0
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Glm53Dflash2SchedulerRequest {
    pub contract_epoch: u64,
    pub temperature: f32,
    pub anchor_position: u64,
    pub block_tokens: u8,
    pub returned_drafts: u8,
    pub window: u16,
}

impl Glm53Dflash2SchedulerRequest {
    pub const fn greedy(contract_epoch: u64, anchor_position: u64) -> Self {
        Self {
            contract_epoch,
            temperature: 0.0,
            anchor_position,
            block_tokens: GLM53_DFLASH2_BLOCK_TOKENS,
            returned_drafts: GLM53_DFLASH2_RETURNED_DRAFTS,
            window: GLM53_DFLASH2_WINDOW,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Glm53VerifyCandidate {
    DflashGamma,
    /// Representable only so the contract can explicitly reject the current
    /// generic `len>=3` K4 fallthrough for a seven-draft bootstrap.
    GenericK4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Glm53VerifyRoute {
    DflashGamma8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SchedulerTransaction {
    contract_epoch: u64,
    policy_owner_id: u64,
    sequence: u64,
}

impl SchedulerTransaction {
    pub const fn contract_epoch(self) -> u64 {
        self.contract_epoch
    }

    pub const fn policy_owner_id(self) -> u64 {
        self.policy_owner_id
    }

    pub const fn sequence(self) -> u64 {
        self.sequence
    }

    #[cfg(test)]
    pub(crate) const fn forged_for_test(
        contract_epoch: u64,
        policy_owner_id: u64,
        sequence: u64,
    ) -> Self {
        Self {
            contract_epoch,
            policy_owner_id,
            sequence,
        }
    }
}

/// Non-cloneable policy with a process-minted opaque owner ID and an
/// independent checked transaction sequence.
#[derive(Debug)]
pub struct Glm53Dflash2SchedulerPolicy {
    contract_epoch: u64,
    policy_owner_id: u64,
    next_sequence: u64,
}

impl Glm53Dflash2SchedulerPolicy {
    pub fn new(contract_epoch: u64) -> Result<Self> {
        ensure!(
            contract_epoch != 0,
            "GLM DFlash2 contract epoch must be nonzero"
        );
        let policy_owner_id = allocate_policy_owner_id()?;
        Ok(Self {
            contract_epoch,
            policy_owner_id,
            next_sequence: 0,
        })
    }

    pub const fn contract_epoch(&self) -> u64 {
        self.contract_epoch
    }

    pub const fn policy_owner_id(&self) -> u64 {
        self.policy_owner_id
    }

    /// This pure gate must run before any scheduler/model effect.
    pub fn admit(
        &mut self,
        request: Glm53Dflash2SchedulerRequest,
        prior_effects: PreAdmissionEffects,
    ) -> Result<Glm53Dflash2Decision> {
        ensure!(
            prior_effects.is_empty(),
            "GLM DFlash2 admission must precede every scheduler/model effect"
        );
        ensure!(
            request.temperature.is_finite() && request.temperature == 0.0,
            "GLM DFlash2 scheduler is greedy-only"
        );
        ensure!(
            request.contract_epoch == self.contract_epoch
                && request.block_tokens == GLM53_DFLASH2_BLOCK_TOKENS
                && request.returned_drafts == GLM53_DFLASH2_RETURNED_DRAFTS
                && request.window == GLM53_DFLASH2_WINDOW,
            "stale GLM DFlash2 scheduler geometry"
        );
        let final_position = request
            .anchor_position
            .checked_add(u64::from(GLM53_DFLASH2_BLOCK_TOKENS - 1))
            .ok_or_else(|| anyhow::anyhow!("GLM DFlash2 position overflow"))?;
        ensure!(
            final_position <= GLM53_MAX_POSITION_INCLUSIVE,
            "GLM DFlash2 block exceeds full context"
        );
        self.next_sequence = self
            .next_sequence
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("GLM DFlash2 scheduler sequence overflow"))?;
        Ok(Glm53Dflash2Decision {
            transaction: SchedulerTransaction {
                contract_epoch: self.contract_epoch,
                policy_owner_id: self.policy_owner_id,
                sequence: self.next_sequence,
            },
            anchor_position: request.anchor_position,
            route: Glm53VerifyRoute::DflashGamma8,
            phase: DecisionPhase::Admitted,
            accepted_rows: None,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecisionPhase {
    Admitted,
    EffectsStarted,
    Verified,
    TargetCommitted,
    CaptureCommitted,
    Emitted,
    Failed,
}

#[derive(Clone, Copy, Debug)]
pub struct VerificationReceipt {
    pub accepted_drafts: u8,
    pub target_status: u32,
    pub drafter_status: u32,
    pub async_complete: bool,
}

#[derive(Debug)]
pub struct Glm53Dflash2Decision {
    transaction: SchedulerTransaction,
    anchor_position: u64,
    route: Glm53VerifyRoute,
    phase: DecisionPhase,
    accepted_rows: Option<u8>,
}

impl Glm53Dflash2Decision {
    pub const fn transaction(&self) -> SchedulerTransaction {
        self.transaction
    }

    pub const fn anchor_position(&self) -> u64 {
        self.anchor_position
    }

    pub const fn route(&self) -> Glm53VerifyRoute {
        self.route
    }

    pub const fn phase(&self) -> DecisionPhase {
        self.phase
    }

    pub const fn accepted_rows(&self) -> Option<u8> {
        self.accepted_rows
    }

    pub fn begin_effects(
        &mut self,
        policy: &Glm53Dflash2SchedulerPolicy,
        transaction: SchedulerTransaction,
    ) -> Result<()> {
        self.authorize(policy, transaction)?;
        self.require_phase(DecisionPhase::Admitted)?;
        self.phase = DecisionPhase::EffectsStarted;
        Ok(())
    }

    pub fn verify(
        &mut self,
        policy: &Glm53Dflash2SchedulerPolicy,
        transaction: SchedulerTransaction,
        candidate: Glm53VerifyCandidate,
        returned_drafts: u8,
        receipt: VerificationReceipt,
    ) -> Result<u8> {
        self.authorize(policy, transaction)?;
        self.require_phase(DecisionPhase::EffectsStarted)?;
        if candidate != Glm53VerifyCandidate::DflashGamma
            || returned_drafts != GLM53_DFLASH2_RETURNED_DRAFTS
        {
            return self.fail("seven GLM drafts must use the DFlash gamma verifier before K4");
        }
        if receipt.accepted_drafts > GLM53_DFLASH2_RETURNED_DRAFTS
            || receipt.target_status != 0
            || receipt.drafter_status != 0
            || !receipt.async_complete
        {
            return self.fail("GLM DFlash2 verification receipt failed");
        }
        let accepted_rows = receipt.accepted_drafts + 1;
        self.accepted_rows = Some(accepted_rows);
        self.phase = DecisionPhase::Verified;
        Ok(accepted_rows)
    }

    pub fn commit_target(
        &mut self,
        policy: &Glm53Dflash2SchedulerPolicy,
        transaction: SchedulerTransaction,
        committed_rows: u8,
        status: u32,
        async_complete: bool,
    ) -> Result<()> {
        self.authorize(policy, transaction)?;
        self.require_phase(DecisionPhase::Verified)?;
        self.require_commit(committed_rows, status, async_complete, "target")?;
        self.phase = DecisionPhase::TargetCommitted;
        Ok(())
    }

    pub fn commit_capture(
        &mut self,
        policy: &Glm53Dflash2SchedulerPolicy,
        transaction: SchedulerTransaction,
        committed_rows: u8,
        status: u32,
        async_complete: bool,
    ) -> Result<()> {
        self.authorize(policy, transaction)?;
        self.require_phase(DecisionPhase::TargetCommitted)?;
        self.require_commit(committed_rows, status, async_complete, "capture")?;
        self.phase = DecisionPhase::CaptureCommitted;
        Ok(())
    }

    pub fn emit(
        &mut self,
        policy: &Glm53Dflash2SchedulerPolicy,
        transaction: SchedulerTransaction,
        emitted_rows: u8,
    ) -> Result<()> {
        self.authorize(policy, transaction)?;
        self.require_phase(DecisionPhase::CaptureCommitted)?;
        if Some(emitted_rows) != self.accepted_rows {
            return self.fail("emitted rows do not match committed verification prefix");
        }
        self.phase = DecisionPhase::Emitted;
        Ok(())
    }

    fn authorize(
        &self,
        policy: &Glm53Dflash2SchedulerPolicy,
        transaction: SchedulerTransaction,
    ) -> Result<()> {
        ensure!(
            transaction == self.transaction,
            "stale GLM DFlash2 scheduler transaction"
        );
        ensure!(
            policy.contract_epoch == self.transaction.contract_epoch
                && policy.policy_owner_id == self.transaction.policy_owner_id,
            "stale GLM DFlash2 scheduler contract or policy owner"
        );
        Ok(())
    }

    fn require_phase(&mut self, expected: DecisionPhase) -> Result<()> {
        if self.phase != expected {
            return self.fail("GLM DFlash2 scheduler effect order violation");
        }
        Ok(())
    }

    fn require_commit(
        &mut self,
        committed_rows: u8,
        status: u32,
        async_complete: bool,
        label: &str,
    ) -> Result<()> {
        if Some(committed_rows) != self.accepted_rows || status != 0 || !async_complete {
            return self.fail(&format!("GLM DFlash2 {label} commit failed"));
        }
        Ok(())
    }

    fn fail<T>(&mut self, message: &str) -> Result<T> {
        self.phase = DecisionPhase::Failed;
        bail!(message.to_owned())
    }
}

#[cfg(test)]
fn section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let start = source.find(start).unwrap();
    let end = source[start..].find(end).unwrap() + start;
    &source[start..end]
}

#[cfg(test)]
fn ordered(source: &str, tokens: &[&str]) -> bool {
    let mut cursor = 0;
    for token in tokens {
        let Some(offset) = source[cursor..].find(token) else {
            return false;
        };
        cursor += offset + token.len();
    }
    true
}

#[cfg(test)]
fn source_contract(source: &str) -> bool {
    let source = source
        .split_once("#[cfg(test)]\nfn section")
        .map_or(source, |(production, _)| production);
    let allocator = section(
        source,
        "fn allocate_owner_from(",
        "fn allocate_policy_owner_id(",
    );
    let constructor = section(source, "pub fn new(", "pub const fn contract_epoch");
    let admit = section(source, "pub fn admit(", "pub enum DecisionPhase");
    let begin = section(source, "pub fn begin_effects(", "pub fn verify(");
    let verify = section(source, "pub fn verify(", "pub fn commit_target(");
    let target = section(source, "pub fn commit_target(", "pub fn commit_capture(");
    let capture = section(source, "pub fn commit_capture(", "pub fn emit(");
    let emit = section(source, "pub fn emit(", "fn authorize(");
    let authorize = section(source, "fn authorize(", "fn require_phase(");
    source
        .matches("static NEXT_POLICY_OWNER_ID: AtomicU64 = AtomicU64::new(1);")
        .count()
        == 1
        && ordered(
            allocator,
            &[
                ".fetch_update(Ordering::Relaxed, Ordering::Relaxed",
                "owner != 0",
                "owner.checked_add(1)",
                ".map_err",
            ],
        )
        && ordered(
            constructor,
            &[
                "contract_epoch != 0",
                "let policy_owner_id = allocate_policy_owner_id()?;",
                "policy_owner_id",
                "next_sequence: 0",
            ],
        )
        && ordered(
            admit,
            &[
                "prior_effects.is_empty()",
                "request.temperature.is_finite() && request.temperature == 0.0",
                "request.contract_epoch == self.contract_epoch",
                "let final_position",
                "self.next_sequence = self",
                ".checked_add(1)",
                "policy_owner_id: self.policy_owner_id",
                "sequence: self.next_sequence",
            ],
        )
        && ordered(
            begin,
            &[
                "self.authorize",
                "DecisionPhase::Admitted",
                "DecisionPhase::EffectsStarted",
            ],
        )
        && ordered(
            verify,
            &[
                "self.authorize",
                "DecisionPhase::EffectsStarted",
                "candidate != Glm53VerifyCandidate::DflashGamma",
                "receipt.accepted_drafts > GLM53_DFLASH2_RETURNED_DRAFTS",
                "let accepted_rows = receipt.accepted_drafts + 1",
                "DecisionPhase::Verified",
            ],
        )
        && ordered(
            target,
            &[
                "self.authorize",
                "DecisionPhase::Verified",
                "self.require_commit",
                "DecisionPhase::TargetCommitted",
            ],
        )
        && ordered(
            capture,
            &[
                "self.authorize",
                "DecisionPhase::TargetCommitted",
                "self.require_commit",
                "DecisionPhase::CaptureCommitted",
            ],
        )
        && ordered(
            emit,
            &[
                "self.authorize",
                "DecisionPhase::CaptureCommitted",
                "Some(emitted_rows) != self.accepted_rows",
                "DecisionPhase::Emitted",
            ],
        )
        && authorize.contains("transaction == self.transaction")
        && authorize.contains("policy.contract_epoch == self.transaction.contract_epoch")
        && authorize.contains("policy.policy_owner_id == self.transaction.policy_owner_id")
        && !source.contains("Glm53VerifyRoute::GenericK4")
}

#[cfg(test)]
#[path = "glm53_dflash2_admission_tests.rs"]
mod tests;
