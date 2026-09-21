// SPDX-License-Identifier: AGPL-3.0-only

//! Fail-closed receipt for a future exact-raw-argmax proposer.
//!
//! This module is deliberately not consumed by the scheduler yet. Merely
//! selecting [`TargetTokenAuthority::ExactRawArgmax`] must not make an
//! existing proposal eligible: a future integration has to issue this receipt
//! at publication and re-admit it immediately before target verification.

#![allow(dead_code)]

use anyhow::{Result, ensure};
use spark_model::speculative::TargetTokenAuthority;

const RECEIPT_VERSION: u8 = 1;
pub(super) const MAX_EXACT_DRAFTS: usize = 31;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ExactDraftSource {
    GhostModelOnly,
    Retrieval,
    AsyncCollection,
    EarlyExit,
    CfgJumpForward,
    Portfolio,
    Tree,
    Ngram,
    PolicySplice,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ExactTargetRowSource {
    RawTargetArgmax,
    ServingPolicy,
    GrammarMasked,
    Resampled,
    TypicalOrRelaxed,
}

/// Complete census of scheduler transforms that can change or observe the
/// serving token relative to the target model's unmodified argmax.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct ExactServingTransforms(u32);

impl ExactServingTransforms {
    pub const SAMPLER_FILTER: Self = Self(1 << 0);
    pub const HISTORY_PENALTY: Self = Self(1 << 1);
    pub const LOGIT_BIAS: Self = Self(1 << 2);
    pub const THINKING_POLICY: Self = Self(1 << 3);
    pub const CONTENT_POLICY: Self = Self(1 << 4);
    pub const TOOL_POLICY: Self = Self(1 << 5);
    pub const GRAMMAR: Self = Self(1 << 6);
    pub const ADAPTIVE_SAMPLING: Self = Self(1 << 7);
    pub const LOGPROBS: Self = Self(1 << 8);
    pub const TARGET_MASK_OR_SUPPRESSION: Self = Self(1 << 9);
    pub const TYPICAL_OR_RELAXED_ACCEPT: Self = Self(1 << 10);
    pub const WATCHDOG_OR_ROLLBACK: Self = Self(1 << 11);

    pub const fn none() -> Self {
        Self(0)
    }

    fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// Scalar-bit policy identity. Requiring the canonical inactive value for
/// every field avoids accepting NaNs, signed zero, or a future sampler whose
/// nominal equality happens to hide a different bit pattern.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ExactGreedyPolicyKey {
    pub temperature_bits: u32,
    pub top_k: u32,
    pub top_p_bits: u32,
    pub top_n_sigma_bits: u32,
    pub min_p_bits: u32,
    pub repetition_penalty_bits: u32,
    pub repetition_penalty_window: u32,
    pub presence_penalty_bits: u32,
    pub frequency_penalty_bits: u32,
    pub lz_penalty_bits: u32,
    pub dry_multiplier_bits: u32,
    pub dry_base_bits: u32,
    pub dry_allowed_length: u32,
    pub logit_bias_len: usize,
    pub dry_sequence_breakers_len: usize,
    pub seed: Option<u64>,
}

impl ExactGreedyPolicyKey {
    pub(super) fn canonical() -> Self {
        Self {
            temperature_bits: 0.0f32.to_bits(),
            top_k: 0,
            top_p_bits: 1.0f32.to_bits(),
            top_n_sigma_bits: 0.0f32.to_bits(),
            min_p_bits: 0.0f32.to_bits(),
            repetition_penalty_bits: 1.0f32.to_bits(),
            repetition_penalty_window: 0,
            presence_penalty_bits: 0.0f32.to_bits(),
            frequency_penalty_bits: 0.0f32.to_bits(),
            lz_penalty_bits: 0.0f32.to_bits(),
            dry_multiplier_bits: 0.0f32.to_bits(),
            dry_base_bits: 1.75f32.to_bits(),
            dry_allowed_length: 2,
            logit_bias_len: 0,
            dry_sequence_breakers_len: 0,
            seed: None,
        }
    }

    fn validate(self) -> Result<()> {
        ensure!(
            self == Self::canonical(),
            "non-canonical exact-greedy policy"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ExactGreedyFrameFacts {
    pub position: usize,
    pub fed_len: usize,
    pub logical_len: usize,
    pub prompt_len: usize,
    pub pending_token: u32,
    pub vocab_size: u32,
    pub canonical_fed_prefix_digest: [u8; 32],
    pub target_identity: [u8; 32],
    pub proposer_identity: [u8; 32],
    pub config_identity: [u8; 32],
    pub raw_argmax_identity: [u8; 32],
}

impl ExactGreedyFrameFacts {
    fn validate(self, max_context: usize) -> Result<()> {
        ensure!(
            self.position == self.fed_len,
            "position differs from target-fed prefix"
        );
        ensure!(
            self.prompt_len <= self.fed_len,
            "prompt exceeds target-fed prefix"
        );
        let logical_len = self
            .fed_len
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("exact-greedy logical length overflow"))?;
        ensure!(
            logical_len == self.logical_len,
            "logical prefix length mismatch"
        );
        ensure!(
            self.logical_len <= max_context,
            "logical prefix exceeds exact-greedy context"
        );
        ensure!(self.vocab_size != 0, "zero exact-greedy vocabulary");
        ensure!(
            self.pending_token < self.vocab_size,
            "pending token exceeds exact-greedy vocabulary"
        );
        ensure!(
            self.canonical_fed_prefix_digest != [0; 32],
            "zero canonical target-fed prefix digest"
        );
        for (name, identity) in [
            ("target", self.target_identity),
            ("proposer", self.proposer_identity),
            ("config", self.config_identity),
            ("raw argmax", self.raw_argmax_identity),
        ] {
            ensure!(identity != [0; 32], "zero {name} identity");
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ExactGreedyFrameKey {
    pub session_nonce: u64,
    pub issuer_instance_nonce: u64,
    pub receipt_nonce: u64,
    pub target_commit_epoch: u64,
    pub proposal_epoch: u64,
    pub max_context: usize,
    pub facts: ExactGreedyFrameFacts,
}

impl ExactGreedyFrameKey {
    fn validate(self) -> Result<()> {
        ensure!(self.session_nonce != 0, "zero exact-greedy session nonce");
        ensure!(
            self.issuer_instance_nonce != 0,
            "zero exact-greedy issuer-instance nonce"
        );
        ensure!(self.receipt_nonce != 0, "zero exact-greedy receipt nonce");
        ensure!(self.target_commit_epoch != 0, "zero target commit epoch");
        ensure!(self.proposal_epoch != 0, "zero proposal epoch");
        ensure!(self.max_context != 0, "zero exact-greedy max context");
        self.facts.validate(self.max_context)
    }
}

#[path = "exact_greedy_admission/issuer.rs"]
mod issuer;
pub(super) use issuer::ExactGreedyReceiptIssuer;

fn validate_frame_extent(key: ExactGreedyFrameKey, draft_len: usize) -> Result<usize> {
    ensure!(
        (1..=MAX_EXACT_DRAFTS).contains(&draft_len),
        "bad exact-greedy draft extent"
    );
    let verify_rows = draft_len
        .checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("exact-greedy verify-row extent overflow"))?;
    let state_end = key
        .facts
        .fed_len
        .checked_add(verify_rows)
        .ok_or_else(|| anyhow::anyhow!("exact-greedy target-state extent overflow"))?;
    let output_end = key
        .facts
        .logical_len
        .checked_add(draft_len)
        .ok_or_else(|| anyhow::anyhow!("exact-greedy output extent overflow"))?;
    ensure!(state_end == output_end, "inconsistent exact-greedy extents");
    ensure!(
        output_end <= key.max_context,
        "exact-greedy frame exceeds max context"
    );
    Ok(verify_rows)
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct ExactGreedyFrameReceipt {
    version: u8,
    authority: TargetTokenAuthority,
    source: ExactDraftSource,
    transforms: ExactServingTransforms,
    policy: ExactGreedyPolicyKey,
    key: ExactGreedyFrameKey,
    draft_len: u8,
    drafts: [u32; MAX_EXACT_DRAFTS],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ExactGreedyVerifyOutcome {
    pub num_accepted: usize,
    pub bonus: u32,
}

#[derive(Clone, Copy)]
pub(super) struct ExactGreedyPreverify<'a> {
    pub authority: TargetTokenAuthority,
    pub source: ExactDraftSource,
    pub transforms: ExactServingTransforms,
    pub policy: ExactGreedyPolicyKey,
    pub key: ExactGreedyFrameKey,
    pub drafts: &'a [u32],
}

/// Linear capability returned only after the scheduler frame and receipt
/// reclose. It must be consumed to admit target rows, preventing a receipt
/// from being reused across multiple target launches.
#[derive(Debug, PartialEq, Eq)]
#[must_use]
pub(super) struct ExactGreedyTargetPermit {
    draft_len: u8,
    drafts: [u32; MAX_EXACT_DRAFTS],
    key: ExactGreedyFrameKey,
}

fn validate_admission(
    authority: TargetTokenAuthority,
    source: ExactDraftSource,
    transforms: ExactServingTransforms,
    policy: ExactGreedyPolicyKey,
    key: ExactGreedyFrameKey,
) -> Result<()> {
    ensure!(
        authority == TargetTokenAuthority::ExactRawArgmax,
        "proposer lacks exact raw-argmax authority"
    );
    ensure!(
        source == ExactDraftSource::GhostModelOnly,
        "non-Ghost exact draft source"
    );
    ensure!(transforms.is_empty(), "serving transform active");
    policy.validate()?;
    key.validate()?;
    Ok(())
}

impl ExactGreedyFrameReceipt {
    pub(super) fn issue(
        issuer: &mut ExactGreedyReceiptIssuer,
        authority: TargetTokenAuthority,
        source: ExactDraftSource,
        transforms: ExactServingTransforms,
        policy: ExactGreedyPolicyKey,
        facts: ExactGreedyFrameFacts,
        drafts: &[u32],
    ) -> Result<Self> {
        issuer.require_idle()?;
        let next_receipt_nonce = issuer
            .next_receipt_nonce
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("exact-greedy receipt nonce exhausted"))?;
        let next_proposal_epoch = issuer
            .next_proposal_epoch
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("proposal epoch exhausted"))?;
        let key = ExactGreedyFrameKey {
            session_nonce: issuer.session_nonce,
            issuer_instance_nonce: issuer.instance_nonce,
            receipt_nonce: issuer.next_receipt_nonce,
            target_commit_epoch: issuer.target_commit_epoch,
            proposal_epoch: issuer.next_proposal_epoch,
            max_context: issuer.max_context,
            facts,
        };
        validate_admission(authority, source, transforms, policy, key)?;
        ensure!(!drafts.is_empty(), "empty exact-greedy proposal");
        ensure!(
            drafts.len() <= MAX_EXACT_DRAFTS,
            "exact-greedy proposal exceeds fixed receipt capacity"
        );
        ensure!(
            drafts.iter().all(|token| *token < key.facts.vocab_size),
            "draft token exceeds exact-greedy vocabulary"
        );
        let _verify_rows = validate_frame_extent(key, drafts.len())?;
        let mut fixed = [0; MAX_EXACT_DRAFTS];
        fixed[..drafts.len()].copy_from_slice(drafts);
        let receipt = Self {
            version: RECEIPT_VERSION,
            authority,
            source,
            transforms,
            policy,
            key,
            draft_len: drafts.len() as u8,
            drafts: fixed,
        };
        issuer.publish(key)?;
        issuer.next_receipt_nonce = next_receipt_nonce;
        issuer.next_proposal_epoch = next_proposal_epoch;
        Ok(receipt)
    }

    pub(super) fn frame_key(&self) -> ExactGreedyFrameKey {
        self.key
    }

    /// Re-admit a published frame immediately before target effects. The
    /// returned linear permit is the only value that may later admit target
    /// rows for this frame.
    pub(super) fn admit_preverify(
        self,
        issuer: &mut ExactGreedyReceiptIssuer,
        observed: ExactGreedyPreverify<'_>,
    ) -> Result<ExactGreedyTargetPermit> {
        ensure!(
            self.version == RECEIPT_VERSION,
            "unknown exact receipt version"
        );
        validate_admission(
            observed.authority,
            observed.source,
            observed.transforms,
            observed.policy,
            observed.key,
        )?;
        ensure!(
            self.authority == observed.authority,
            "authority receipt mismatch"
        );
        ensure!(
            self.source == observed.source,
            "draft source receipt mismatch"
        );
        ensure!(
            self.transforms == observed.transforms,
            "transform receipt mismatch"
        );
        ensure!(self.policy == observed.policy, "policy receipt mismatch");
        ensure!(self.key == observed.key, "frame key receipt mismatch");
        let draft_len = usize::from(self.draft_len);
        ensure!(
            draft_len != 0 && draft_len <= MAX_EXACT_DRAFTS,
            "bad receipt width"
        );
        ensure!(
            observed.drafts == &self.drafts[..draft_len],
            "draft receipt mismatch"
        );
        let _verify_rows = validate_frame_extent(self.key, draft_len)?;
        issuer.begin_verify(self.key)?;
        Ok(ExactGreedyTargetPermit {
            draft_len: self.draft_len,
            drafts: self.drafts,
            key: self.key,
        })
    }
}

impl ExactGreedyTargetPermit {
    /// Consume the preverify permit and accept only raw target argmax rows.
    /// Success leaves the issuer awaiting a strict next-epoch target commit;
    /// no new receipt can be minted before that commit is recorded.
    pub(super) fn finish_raw_target(
        self,
        issuer: &mut ExactGreedyReceiptIssuer,
        source: ExactTargetRowSource,
        observed_key: ExactGreedyFrameKey,
        raw_target_tokens: &[u32],
    ) -> Result<ExactGreedyVerifyOutcome> {
        ensure!(
            source == ExactTargetRowSource::RawTargetArgmax,
            "target rows are not raw argmax"
        );
        observed_key.validate()?;
        ensure!(self.key == observed_key, "target frame key mismatch");
        let draft_len = usize::from(self.draft_len);
        let verify_rows = validate_frame_extent(self.key, draft_len)?;
        ensure!(
            raw_target_tokens.len() == verify_rows,
            "raw target row count mismatch"
        );
        ensure!(
            raw_target_tokens
                .iter()
                .all(|token| *token < self.key.facts.vocab_size),
            "raw target token exceeds exact-greedy vocabulary"
        );
        let num_accepted = self.drafts[..draft_len]
            .iter()
            .zip(raw_target_tokens)
            .take_while(|(draft, target)| draft == target)
            .count();
        issuer.finish_verify(self.key)?;
        Ok(ExactGreedyVerifyOutcome {
            num_accepted,
            bonus: raw_target_tokens[num_accepted],
        })
    }
}

#[cfg(test)]
#[path = "exact_greedy_admission_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "exact_greedy_admission_replay_tests.rs"]
mod replay_tests;
