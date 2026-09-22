// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Result, bail, ensure};

use super::{
    CaptureReadSegment, CaptureSegment, GLM53_DFLASH2_RETURNED_DRAFTS, GLM53_DFLASH2_WINDOW,
    GLM53_MAX_POSITION_EXCLUSIVE, Glm53Dflash2Admission, ProposalPlan, ReleaseReceipt, RingCursor,
    TransactionId,
};

const ATTENTION_VISIBLE_TOKENS: u16 = GLM53_DFLASH2_WINDOW - 1;
const CAPTURE_LAYER_COUNT: usize = 5;

#[derive(Clone, Copy, Debug)]
pub struct CaptureAppendPlan {
    txn: TransactionId,
    base_position: u64,
    end_position: u64,
    source_skip_rows: u32,
    write_rows: u16,
    segments: [CaptureSegment; 2],
}

impl CaptureAppendPlan {
    pub const fn transaction(&self) -> TransactionId {
        self.txn
    }

    pub const fn base_position(&self) -> u64 {
        self.base_position
    }

    pub const fn end_position(&self) -> u64 {
        self.end_position
    }

    pub const fn source_skip_rows(&self) -> u32 {
        self.source_skip_rows
    }

    pub const fn write_rows(&self) -> u16 {
        self.write_rows
    }

    pub const fn segments(&self) -> [CaptureSegment; 2] {
        self.segments
    }
}

#[derive(Clone, Copy, Debug)]
struct ExpectedCapture {
    base_position: u64,
    rows: u8,
}

#[derive(Clone, Copy, Debug)]
struct OutstandingProposal {
    txn: TransactionId,
    drafts: u8,
}

#[derive(Clone, Copy, Debug)]
enum Pending {
    Capture {
        plan: CaptureAppendPlan,
        next: RingCursor,
    },
    Proposal {
        plan: ProposalPlan,
        next: RingCursor,
    },
}

#[derive(Debug)]
pub struct Glm53Dflash2SequenceState {
    lease_id: Option<u64>,
    last_released_lease: Option<u64>,
    generation: u64,
    next_nonce: u64,
    capture: RingCursor,
    layer_caches: [RingCursor; CAPTURE_LAYER_COUNT],
    expected_capture: Option<ExpectedCapture>,
    outstanding: Option<OutstandingProposal>,
    pending: Option<Pending>,
}

impl Glm53Dflash2SequenceState {
    pub fn new(lease_id: u64, _admission: &Glm53Dflash2Admission) -> Result<Self> {
        ensure!(lease_id != 0, "lease id must be nonzero");
        Ok(Self {
            lease_id: Some(lease_id),
            last_released_lease: None,
            generation: 1,
            next_nonce: 0,
            capture: RingCursor::default(),
            layer_caches: [RingCursor::default(); CAPTURE_LAYER_COUNT],
            expected_capture: None,
            outstanding: None,
            pending: None,
        })
    }

    pub fn capture_cursor(&self) -> Result<RingCursor> {
        self.ensure_live()?;
        Ok(self.capture)
    }

    pub fn layer_cache_cursors(&self) -> Result<[RingCursor; CAPTURE_LAYER_COUNT]> {
        self.ensure_live()?;
        Ok(self.layer_caches)
    }

    pub fn expected_capture(&self) -> Result<Option<(u64, u8)>> {
        self.ensure_live()?;
        Ok(self
            .expected_capture
            .map(|expected| (expected.base_position, expected.rows)))
    }

    pub fn outstanding_drafts(&self) -> Result<Option<u8>> {
        self.ensure_live()?;
        Ok(self.outstanding.map(|proposal| proposal.drafts))
    }

    pub fn begin_capture(&mut self, base_position: u64, rows: u64) -> Result<CaptureAppendPlan> {
        self.ensure_idle()?;
        ensure!(
            self.outstanding.is_none(),
            "verification is still outstanding"
        );
        ensure!(rows != 0, "capture append must contain rows");
        ensure!(
            base_position == self.capture.absolute_end,
            "capture append is not contiguous"
        );
        if let Some(expected) = self.expected_capture {
            ensure!(
                base_position == expected.base_position && rows == u64::from(expected.rows),
                "capture append does not match verified acceptance"
            );
        }
        let end_position = base_position
            .checked_add(rows)
            .ok_or_else(|| anyhow::anyhow!("capture position overflow"))?;
        ensure!(
            end_position <= GLM53_MAX_POSITION_EXCLUSIVE,
            "capture end exceeds GLM-5.3 context"
        );
        let write_rows_u64 = rows.min(u64::from(GLM53_DFLASH2_WINDOW));
        let write_rows = u16::try_from(write_rows_u64)?;
        let source_skip_rows = u32::try_from(rows - write_rows_u64)?;
        let (destination_row, next) = if write_rows == GLM53_DFLASH2_WINDOW {
            (
                0,
                RingCursor {
                    absolute_end: end_position,
                    head: 0,
                    retained: GLM53_DFLASH2_WINDOW,
                },
            )
        } else {
            let destination_row =
                (self.capture.head + self.capture.retained) % GLM53_DFLASH2_WINDOW;
            let total = u32::from(self.capture.retained) + u32::from(write_rows);
            let overflow = total.saturating_sub(u32::from(GLM53_DFLASH2_WINDOW));
            let head = (u32::from(self.capture.head) + overflow) % u32::from(GLM53_DFLASH2_WINDOW);
            (
                destination_row,
                RingCursor {
                    absolute_end: end_position,
                    head: u16::try_from(head)?,
                    retained: u16::try_from(total.min(u32::from(GLM53_DFLASH2_WINDOW)))?,
                },
            )
        };
        let first_rows = write_rows.min(GLM53_DFLASH2_WINDOW - destination_row);
        let txn = self.next_transaction()?;
        let plan = CaptureAppendPlan {
            txn,
            base_position,
            end_position,
            source_skip_rows,
            write_rows,
            segments: [
                CaptureSegment {
                    destination_row,
                    rows: first_rows,
                },
                CaptureSegment {
                    destination_row: 0,
                    rows: write_rows - first_rows,
                },
            ],
        };
        self.pending = Some(Pending::Capture { plan, next });
        Ok(plan)
    }

    pub fn commit_capture(&mut self, txn: TransactionId, published_end: u64) -> Result<()> {
        let Pending::Capture { plan, next } = self.match_pending(txn)? else {
            bail!("transaction is not a capture append");
        };
        ensure!(
            published_end == plan.end_position,
            "published capture end mismatches plan"
        );
        self.capture = next;
        self.pending = None;
        self.expected_capture = None;
        Ok(())
    }

    pub fn rollback_capture(&mut self, txn: TransactionId) -> Result<()> {
        let Pending::Capture { .. } = self.match_pending(txn)? else {
            bail!("transaction is not a capture append");
        };
        self.pending = None;
        Ok(())
    }

    pub fn begin_proposal(&mut self, admission: &Glm53Dflash2Admission) -> Result<ProposalPlan> {
        self.ensure_idle()?;
        ensure!(
            self.outstanding.is_none(),
            "verification is still outstanding"
        );
        ensure!(
            self.expected_capture.is_none(),
            "verified rows have not been captured"
        );
        ensure!(
            admission.anchor_position() == self.capture.absolute_end,
            "admission anchor does not match committed captures"
        );
        let cache = self.coherent_cache()?;
        let new_context = self
            .capture
            .absolute_end
            .checked_sub(cache.absolute_end)
            .ok_or_else(|| anyhow::anyhow!("cache is ahead of captures"))?;
        ensure!(new_context != 0, "proposal has no new target context");
        let target_tail = new_context.min(u64::from(ATTENTION_VISIBLE_TOKENS));
        let kept_past =
            u64::from(cache.retained).min(u64::from(ATTENTION_VISIBLE_TOKENS) - target_tail);
        let local_context = kept_past + target_tail;
        let capture_oldest_position = self
            .capture
            .absolute_end
            .checked_sub(u64::from(self.capture.retained))
            .ok_or_else(|| anyhow::anyhow!("capture cursor underflow"))?;
        let target_source_start_position = self.capture.absolute_end - target_tail;
        let source_offset = target_source_start_position
            .checked_sub(capture_oldest_position)
            .ok_or_else(|| anyhow::anyhow!("target tail is no longer retained"))?;
        ensure!(
            source_offset + target_tail <= u64::from(self.capture.retained),
            "target tail exceeds retained capture ring"
        );
        let source_offset = u16::try_from(source_offset)?;
        let source_row = (self.capture.head + source_offset) % GLM53_DFLASH2_WINDOW;
        let target_tail_rows = u16::try_from(target_tail)?;
        let first_rows = target_tail_rows.min(GLM53_DFLASH2_WINDOW - source_row);
        let txn = self.next_transaction()?;
        let plan = ProposalPlan {
            txn,
            anchor_position: admission.anchor_position(),
            logical_new_context_rows: u32::try_from(new_context)?,
            attention_new_context_rows: target_tail_rows,
            absolute_context_end: self.capture.absolute_end,
            capture_oldest_position,
            capture_retained_rows: self.capture.retained,
            target_source_start_position,
            capture_source_offset_rows: source_offset,
            capture_source_segments: [
                CaptureReadSegment {
                    source_row,
                    rows: first_rows,
                },
                CaptureReadSegment {
                    source_row: 0,
                    rows: target_tail_rows - first_rows,
                },
            ],
            kept_past_rows: u16::try_from(kept_past)?,
            past_drop_rows: cache.retained - u16::try_from(kept_past)?,
            local_context_rows: u16::try_from(local_context)?,
            route: admission.route(),
        };
        let next = RingCursor {
            absolute_end: self.capture.absolute_end,
            head: 0,
            retained: plan.local_context_rows,
        };
        self.pending = Some(Pending::Proposal { plan, next });
        Ok(plan)
    }

    pub fn commit_proposal(
        &mut self,
        txn: TransactionId,
        topk_status: &[u32; 8],
        selector_status: u32,
    ) -> Result<()> {
        let Pending::Proposal { plan, next } = self.match_pending(txn)? else {
            bail!("transaction is not a proposal");
        };
        ensure!(
            topk_status.iter().all(|status| *status == 0),
            "top-k device status rejected proposal"
        );
        ensure!(
            selector_status == 0,
            "selector device status rejected proposal"
        );
        self.layer_caches = [next; CAPTURE_LAYER_COUNT];
        self.pending = None;
        self.outstanding = Some(OutstandingProposal {
            txn: plan.txn,
            drafts: GLM53_DFLASH2_RETURNED_DRAFTS,
        });
        Ok(())
    }

    pub fn rollback_proposal(&mut self, txn: TransactionId) -> Result<()> {
        let Pending::Proposal { .. } = self.match_pending(txn)? else {
            bail!("transaction is not a proposal");
        };
        self.pending = None;
        Ok(())
    }

    pub fn after_verify(&mut self, proposal_txn: TransactionId, accepted_drafts: u8) -> Result<()> {
        self.ensure_live()?;
        ensure!(self.pending.is_none(), "transaction is pending");
        let outstanding = self
            .outstanding
            .ok_or_else(|| anyhow::anyhow!("no proposal awaits verification"))?;
        ensure!(
            proposal_txn == outstanding.txn,
            "stale proposal verification"
        );
        ensure!(
            accepted_drafts <= outstanding.drafts,
            "accepted drafts exceed proposal"
        );
        let rows = accepted_drafts + 1;
        self.expected_capture = Some(ExpectedCapture {
            base_position: self.capture.absolute_end,
            rows,
        });
        self.outstanding = None;
        Ok(())
    }

    /// Abort all in-flight metadata before the owner frees every sequence buffer.
    pub fn release(&mut self) -> Result<ReleaseReceipt> {
        let lease_id = self
            .lease_id
            .take()
            .ok_or_else(|| anyhow::anyhow!("sequence is already released"))?;
        self.last_released_lease = Some(lease_id);
        self.pending = None;
        self.outstanding = None;
        self.expected_capture = None;
        self.capture = RingCursor::default();
        self.layer_caches = [RingCursor::default(); CAPTURE_LAYER_COUNT];
        Ok(ReleaseReceipt {
            lease_id,
            generation: self.generation,
        })
    }

    pub fn reset_for_reuse(
        &mut self,
        lease_id: u64,
        _admission: &Glm53Dflash2Admission,
    ) -> Result<()> {
        ensure!(self.lease_id.is_none(), "sequence is still live");
        ensure!(lease_id != 0, "lease id must be nonzero");
        ensure!(
            Some(lease_id) != self.last_released_lease,
            "released lease id cannot be reused"
        );
        self.generation = self
            .generation
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("sequence generation overflow"))?;
        self.next_nonce = 0;
        self.lease_id = Some(lease_id);
        self.capture = RingCursor::default();
        self.layer_caches = [RingCursor::default(); CAPTURE_LAYER_COUNT];
        Ok(())
    }

    fn ensure_live(&self) -> Result<()> {
        ensure!(self.lease_id.is_some(), "sequence is released");
        Ok(())
    }

    fn ensure_idle(&self) -> Result<()> {
        self.ensure_live()?;
        ensure!(self.pending.is_none(), "another transaction is pending");
        Ok(())
    }

    fn coherent_cache(&self) -> Result<RingCursor> {
        let first = self.layer_caches[0];
        ensure!(
            self.layer_caches.iter().all(|cursor| *cursor == first),
            "capture-layer caches are incoherent"
        );
        Ok(first)
    }

    fn next_transaction(&mut self) -> Result<TransactionId> {
        self.next_nonce = self
            .next_nonce
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("transaction nonce overflow"))?;
        Ok(TransactionId {
            generation: self.generation,
            nonce: self.next_nonce,
        })
    }

    fn match_pending(&self, txn: TransactionId) -> Result<Pending> {
        self.ensure_live()?;
        let pending = self
            .pending
            .ok_or_else(|| anyhow::anyhow!("no transaction is pending"))?;
        let pending_txn = match pending {
            Pending::Capture { plan, .. } => plan.txn,
            Pending::Proposal { plan, .. } => plan.txn,
        };
        ensure!(pending_txn == txn, "stale transaction id");
        Ok(pending)
    }
}
