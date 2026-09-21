// SPDX-License-Identifier: AGPL-3.0-only

use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Result, ensure};

use super::ExactGreedyFrameKey;

static NEXT_EXACT_GREEDY_ISSUER_NONCE: AtomicU64 = AtomicU64::new(1);

fn mint_issuer_instance_nonce() -> Result<u64> {
    NEXT_EXACT_GREEDY_ISSUER_NONCE
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |nonce| {
            nonce.checked_add(1)
        })
        .map_err(|_| anyhow::anyhow!("exact-greedy issuer-instance nonce exhausted"))
}

/// Scheduler-owned minting authority. A process-unique instance nonce,
/// non-`Clone` state, and monotonically increasing per-instance counters
/// prevent duplicate frame identities.
#[derive(Debug)]
pub(in crate::scheduler) struct ExactGreedyReceiptIssuer {
    pub(super) session_nonce: u64,
    pub(super) instance_nonce: u64,
    pub(super) max_context: usize,
    pub(super) target_commit_epoch: u64,
    pub(super) next_receipt_nonce: u64,
    pub(super) next_proposal_epoch: u64,
    phase: ExactGreedyIssuerPhase,
}

#[derive(Debug, PartialEq, Eq)]
enum ExactGreedyIssuerPhase {
    Idle,
    Published(ExactGreedyFrameKey),
    Verifying(ExactGreedyFrameKey),
    AwaitingCommit(ExactGreedyFrameKey),
}

impl ExactGreedyReceiptIssuer {
    pub(in crate::scheduler) fn new(
        session_nonce: u64,
        max_context: usize,
        target_commit_epoch: u64,
        last_proposal_epoch: u64,
    ) -> Result<Self> {
        ensure!(session_nonce != 0, "zero exact-greedy session nonce");
        ensure!(max_context != 0, "zero exact-greedy max context");
        ensure!(target_commit_epoch != 0, "zero target commit epoch");
        let next_proposal_epoch = last_proposal_epoch
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("proposal epoch exhausted"))?;
        let instance_nonce = mint_issuer_instance_nonce()?;
        Ok(Self {
            session_nonce,
            instance_nonce,
            max_context,
            target_commit_epoch,
            next_receipt_nonce: 1,
            next_proposal_epoch,
            phase: ExactGreedyIssuerPhase::Idle,
        })
    }

    /// Advance only from the exact target state this issuer currently owns.
    pub(in crate::scheduler) fn record_target_commit(
        &mut self,
        expected_current_epoch: u64,
        next_epoch: u64,
    ) -> Result<()> {
        let frame_epoch = match &self.phase {
            ExactGreedyIssuerPhase::AwaitingCommit(key) => key.target_commit_epoch,
            _ => anyhow::bail!("no finished exact frame awaiting target commit"),
        };
        ensure!(
            expected_current_epoch == self.target_commit_epoch,
            "stale target commit authority"
        );
        ensure!(
            frame_epoch == self.target_commit_epoch,
            "finished frame target epoch drift"
        );
        let expected_next = self
            .target_commit_epoch
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("target commit epoch exhausted"))?;
        ensure!(
            next_epoch == expected_next,
            "out-of-order target commit epoch"
        );
        self.target_commit_epoch = next_epoch;
        self.phase = ExactGreedyIssuerPhase::Idle;
        Ok(())
    }

    pub(super) fn require_idle(&self) -> Result<()> {
        ensure!(
            self.phase == ExactGreedyIssuerPhase::Idle,
            "exact-greedy receipt already active"
        );
        Ok(())
    }

    pub(super) fn publish(&mut self, key: ExactGreedyFrameKey) -> Result<()> {
        self.require_idle()?;
        self.phase = ExactGreedyIssuerPhase::Published(key);
        Ok(())
    }

    pub(super) fn begin_verify(&mut self, key: ExactGreedyFrameKey) -> Result<()> {
        ensure!(
            self.phase == ExactGreedyIssuerPhase::Published(key),
            "stale or replayed exact-greedy receipt"
        );
        self.phase = ExactGreedyIssuerPhase::Verifying(key);
        Ok(())
    }

    pub(super) fn finish_verify(&mut self, key: ExactGreedyFrameKey) -> Result<()> {
        ensure!(
            self.phase == ExactGreedyIssuerPhase::Verifying(key),
            "stale or out-of-order exact target permit"
        );
        self.phase = ExactGreedyIssuerPhase::AwaitingCommit(key);
        Ok(())
    }
}
