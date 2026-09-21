// SPDX-License-Identifier: AGPL-3.0-only

//! Quarantined admission protocol for exact-suffix wide retrieval.
//!
//! This module is intentionally not registered or consumed by the scheduler.
//! It proves only that a 16..=31-token flat proposal is copied from an exact
//! earlier suffix continuation and that raw target argmax remains the sole
//! output authority. Retrieval availability and throughput are conditional.

#![allow(dead_code)]

use anyhow::{Result, ensure};

const RECEIPT_VERSION: u8 = 1;
pub(super) const MIN_WIDE_DRAFTS: usize = 16;
pub(super) const MAX_WIDE_DRAFTS: usize = 31;
const MIN_VERIFY_K: usize = MIN_WIDE_DRAFTS + 1;
const MAX_VERIFY_K: usize = MAX_WIDE_DRAFTS + 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ExactWideDraftSource {
    ExactSuffixHistory,
    NeuralOrSynthetic,
    Portfolio,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ExactWideTopology {
    FlatChain,
    Tree,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ExactWideIdentities {
    pub target: [u8; 32],
    pub target_config: [u8; 32],
    pub target_state: [u8; 32],
    pub retriever: [u8; 32],
    pub retrieval_config: [u8; 32],
    pub raw_argmax: [u8; 32],
}

impl ExactWideIdentities {
    fn validate(self) -> Result<()> {
        for (name, identity) in [
            ("target", self.target),
            ("target config", self.target_config),
            ("target state", self.target_state),
            ("retriever", self.retriever),
            ("retrieval config", self.retrieval_config),
            ("raw argmax", self.raw_argmax),
        ] {
            ensure!(identity != [0; 32], "zero exact-wide {name} identity");
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub(super) struct ExactWideFrame<'a> {
    pub source: ExactWideDraftSource,
    pub topology: ExactWideTopology,
    pub position: usize,
    pub prompt_len: usize,
    pub pending_token: u32,
    pub vocab_size: u32,
    pub identities: ExactWideIdentities,
    pub match_start: usize,
    pub suffix_len: usize,
    pub canonical_tokens: &'a [u32],
    pub drafts: &'a [u32],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ExactWideFrameKey {
    session_nonce: u64,
    issuer_instance_nonce: u64,
    receipt_nonce: u64,
    target_commit_epoch: u64,
    proposal_epoch: u64,
    max_context: usize,
    physical_verify_k: usize,
    verify_k: usize,
    source: ExactWideDraftSource,
    topology: ExactWideTopology,
    position: usize,
    prompt_len: usize,
    pending_token: u32,
    vocab_size: u32,
    canonical_prefix_digest: [u8; 32],
    identities: ExactWideIdentities,
    match_start: usize,
    suffix_len: usize,
}

impl ExactWideFrameKey {
    fn validate(self) -> Result<()> {
        ensure!(self.session_nonce != 0, "zero exact-wide session nonce");
        ensure!(
            self.issuer_instance_nonce != 0,
            "zero exact-wide issuer-instance nonce"
        );
        ensure!(self.receipt_nonce != 0, "zero exact-wide receipt nonce");
        ensure!(self.target_commit_epoch != 0, "zero target commit epoch");
        ensure!(self.proposal_epoch != 0, "zero proposal epoch");
        ensure!(self.max_context != 0, "zero exact-wide max context");
        ensure!(
            (MIN_VERIFY_K..=MAX_VERIFY_K).contains(&self.physical_verify_k),
            "physical verify K is outside exact-wide K17..32"
        );
        ensure!(
            (MIN_VERIFY_K..=MAX_VERIFY_K).contains(&self.verify_k),
            "frame verify K is outside exact-wide K17..32"
        );
        ensure!(
            self.verify_k <= self.physical_verify_k,
            "frame verify K exceeds physical capacity"
        );
        ensure!(
            self.source == ExactWideDraftSource::ExactSuffixHistory,
            "non-suffix exact-wide source"
        );
        ensure!(
            self.topology == ExactWideTopology::FlatChain,
            "non-flat exact-wide topology"
        );
        ensure!(self.vocab_size != 0, "zero exact-wide vocabulary");
        ensure!(
            self.canonical_prefix_digest != [0; 32],
            "zero canonical prefix digest"
        );
        self.identities.validate()
    }
}

#[path = "exact_wide_retrieval_admission/issuer.rs"]
mod issuer;
use issuer::canonical_prefix_digest;
pub(super) use issuer::{ExactWideRawTargetReceipt, ExactWideReceiptIssuer};

fn validate_exact_suffix(frame: ExactWideFrame<'_>) -> Result<()> {
    ensure!(frame.suffix_len != 0, "empty retrieval suffix");
    let current_start = frame
        .canonical_tokens
        .len()
        .checked_sub(frame.suffix_len)
        .ok_or_else(|| anyhow::anyhow!("suffix exceeds canonical prefix"))?;
    let match_end = frame
        .match_start
        .checked_add(frame.suffix_len)
        .ok_or_else(|| anyhow::anyhow!("retrieval match extent overflow"))?;
    ensure!(
        match_end <= current_start,
        "retrieval match is not an earlier non-overlapping suffix"
    );
    ensure!(
        &frame.canonical_tokens[frame.match_start..match_end]
            == &frame.canonical_tokens[current_start..],
        "conditional retrieval suffix miss"
    );
    let tail_end = match_end
        .checked_add(frame.drafts.len())
        .ok_or_else(|| anyhow::anyhow!("retrieval tail extent overflow"))?;
    ensure!(
        tail_end <= frame.canonical_tokens.len(),
        "conditional retrieval tail is unavailable"
    );
    ensure!(
        &frame.canonical_tokens[match_end..tail_end] == frame.drafts,
        "conditional retrieval tail mismatch"
    );
    Ok(())
}

fn validate_frame(frame: ExactWideFrame<'_>, key: ExactWideFrameKey) -> Result<()> {
    key.validate()?;
    ensure!(frame.source == key.source, "draft source receipt drift");
    ensure!(frame.topology == key.topology, "topology receipt drift");
    ensure!(
        frame.position == key.position,
        "target position receipt drift"
    );
    ensure!(
        frame.prompt_len == key.prompt_len,
        "prompt length receipt drift"
    );
    ensure!(
        frame.pending_token == key.pending_token,
        "pending token receipt drift"
    );
    ensure!(
        frame.vocab_size == key.vocab_size,
        "vocabulary receipt drift"
    );
    ensure!(
        canonical_prefix_digest(frame.canonical_tokens)? == key.canonical_prefix_digest,
        "canonical prefix content receipt drift"
    );
    ensure!(frame.identities == key.identities, "identity receipt drift");
    ensure!(
        frame.match_start == key.match_start,
        "match-start receipt drift"
    );
    ensure!(
        frame.suffix_len == key.suffix_len,
        "suffix-length receipt drift"
    );
    let logical_len = frame
        .position
        .checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("logical prefix length overflow"))?;
    ensure!(
        frame.canonical_tokens.len() == logical_len,
        "canonical prefix length differs from target position"
    );
    ensure!(
        frame.prompt_len <= frame.position,
        "prompt exceeds target-fed prefix"
    );
    ensure!(
        frame.canonical_tokens.last() == Some(&frame.pending_token),
        "pending token differs from canonical prefix"
    );
    ensure!(
        frame
            .canonical_tokens
            .iter()
            .chain(frame.drafts)
            .all(|token| *token < frame.vocab_size),
        "exact-wide token exceeds vocabulary"
    );
    let verify_k = frame
        .drafts
        .len()
        .checked_add(1)
        .ok_or_else(|| anyhow::anyhow!("verify K overflow"))?;
    ensure!(verify_k == key.verify_k, "verify K receipt drift");
    let target_state_end = frame
        .position
        .checked_add(verify_k)
        .ok_or_else(|| anyhow::anyhow!("target state extent overflow"))?;
    let output_end = logical_len
        .checked_add(frame.drafts.len())
        .ok_or_else(|| anyhow::anyhow!("output extent overflow"))?;
    ensure!(
        target_state_end == output_end,
        "inconsistent exact-wide extents"
    );
    ensure!(
        output_end <= key.max_context,
        "exact-wide frame exceeds max context"
    );
    validate_exact_suffix(frame)
}

#[derive(Debug, PartialEq, Eq)]
pub(super) struct ExactWideFrameReceipt {
    version: u8,
    key: ExactWideFrameKey,
    draft_len: u8,
    drafts: [u32; MAX_WIDE_DRAFTS],
}

impl ExactWideFrameReceipt {
    /// Validate exact retrieval provenance before invoking the caller's first
    /// external effect. Target execution still requires `admit_preverify`.
    pub(super) fn issue_before_first_effect<T>(
        issuer: &mut ExactWideReceiptIssuer,
        frame: ExactWideFrame<'_>,
        first_effect: impl FnOnce() -> Result<T>,
    ) -> Result<(Self, T)> {
        issuer.require_idle()?;
        let verify_k = frame
            .drafts
            .len()
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("verify K overflow"))?;
        let key = ExactWideFrameKey {
            session_nonce: issuer.session_nonce,
            issuer_instance_nonce: issuer.instance_nonce,
            receipt_nonce: issuer.next_receipt_nonce,
            target_commit_epoch: issuer.target_commit_epoch,
            proposal_epoch: issuer.next_proposal_epoch,
            max_context: issuer.max_context,
            physical_verify_k: issuer.physical_verify_k,
            verify_k,
            source: frame.source,
            topology: frame.topology,
            position: frame.position,
            prompt_len: frame.prompt_len,
            pending_token: frame.pending_token,
            vocab_size: frame.vocab_size,
            canonical_prefix_digest: canonical_prefix_digest(frame.canonical_tokens)?,
            identities: issuer.identities,
            match_start: frame.match_start,
            suffix_len: frame.suffix_len,
        };
        validate_frame(frame, key)?;
        let next_receipt_nonce = issuer
            .next_receipt_nonce
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("receipt nonce exhausted"))?;
        let next_proposal_epoch = issuer
            .next_proposal_epoch
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("proposal epoch exhausted"))?;
        let mut drafts = [0; MAX_WIDE_DRAFTS];
        drafts[..frame.drafts.len()].copy_from_slice(frame.drafts);
        let receipt = Self {
            version: RECEIPT_VERSION,
            key,
            draft_len: frame.drafts.len() as u8,
            drafts,
        };
        issuer.publish(key)?;
        issuer.next_receipt_nonce = next_receipt_nonce;
        issuer.next_proposal_epoch = next_proposal_epoch;
        let effect = first_effect()?;
        Ok((receipt, effect))
    }

    #[cfg(test)]
    pub(super) fn frame_key(&self) -> ExactWideFrameKey {
        self.key
    }

    pub(super) fn admit_preverify(
        self,
        issuer: &mut ExactWideReceiptIssuer,
        observed: ExactWideFrame<'_>,
    ) -> Result<ExactWideTargetPermit> {
        ensure!(
            self.version == RECEIPT_VERSION,
            "unknown exact-wide receipt version"
        );
        validate_frame(observed, self.key)?;
        let draft_len = usize::from(self.draft_len);
        ensure!(
            (MIN_WIDE_DRAFTS..=MAX_WIDE_DRAFTS).contains(&draft_len)
                && draft_len.checked_add(1) == Some(self.key.verify_k),
            "bad exact-wide receipt width"
        );
        ensure!(
            observed.drafts == &self.drafts[..draft_len],
            "exact-wide draft receipt mismatch"
        );
        issuer.begin_verify(self.key)?;
        Ok(ExactWideTargetPermit {
            key: self.key,
            draft_len: self.draft_len,
            drafts: self.drafts,
        })
    }
}

#[derive(Debug, PartialEq, Eq)]
#[must_use]
pub(super) struct ExactWideTargetPermit {
    key: ExactWideFrameKey,
    draft_len: u8,
    drafts: [u32; MAX_WIDE_DRAFTS],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct ExactWideVerifyOutcome {
    pub num_accepted: usize,
    pub bonus: u32,
}

impl ExactWideTargetPermit {
    pub(super) fn seal_raw_target(
        &self,
        issuer: &mut ExactWideReceiptIssuer,
        raw_target_tokens: &[u32],
    ) -> Result<ExactWideRawTargetReceipt> {
        issuer.seal_raw_target(self.key, raw_target_tokens)
    }

    pub(super) fn finish_raw_target(
        self,
        issuer: &mut ExactWideReceiptIssuer,
        receipt: ExactWideRawTargetReceipt,
    ) -> Result<ExactWideVerifyOutcome> {
        ensure!(receipt.key() == self.key, "target frame key receipt drift");
        let raw_target_tokens = receipt.raw_target_tokens();
        ensure!(
            raw_target_tokens.len() == self.key.verify_k,
            "raw target row count mismatch"
        );
        ensure!(
            raw_target_tokens
                .iter()
                .all(|token| *token < self.key.vocab_size),
            "raw target token exceeds vocabulary"
        );
        let draft_len = usize::from(self.draft_len);
        ensure!(
            (MIN_WIDE_DRAFTS..=MAX_WIDE_DRAFTS).contains(&draft_len)
                && draft_len.checked_add(1) == Some(self.key.verify_k),
            "bad exact-wide target permit width"
        );
        let num_accepted = self.drafts[..draft_len]
            .iter()
            .zip(raw_target_tokens)
            .take_while(|(draft, target)| draft == target)
            .count();
        let bonus = raw_target_tokens[num_accepted];
        issuer.consume_raw_target(self.key, receipt)?;
        Ok(ExactWideVerifyOutcome {
            num_accepted,
            bonus,
        })
    }
}

#[cfg(test)]
#[path = "exact_wide_retrieval_admission_tests.rs"]
mod tests;
