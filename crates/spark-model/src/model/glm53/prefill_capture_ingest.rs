// SPDX-License-Identifier: AGPL-3.0-only

//! Complete-five-tap ingestion with one success-only context publication.
//! Failure is terminal even after a successful drain: projected rows may already
//! be written. The owner must poison/reset the target and drafter, not retry or
//! infer that the persistent target state was rolled back by this helper.

use anyhow::{Result, ensure};

use super::prefill_capture_plan::{
    BoundCapturePlan, CopyStep, DeviceSpan, GATHER_ROWS, ProjectionStep,
};
use crate::model::glm53::GLM53_CAPTURE_LAYERS;

/// Implementations enqueue copies and FC+RMSNorm on exactly the supplied stream.
/// `project_norm` must not publish/advance drafter context. `synchronize` must
/// fence that stream's completion, not merely submit it. All errors propagate.
pub(crate) trait PrefillCaptureIo {
    fn copy(&mut self, step: CopyStep, stream: u64) -> Result<()>;
    fn project_norm(&mut self, step: ProjectionStep, stream: u64) -> Result<()>;
    fn synchronize(&mut self, stream: u64) -> Result<()>;
}

#[derive(Debug)]
enum Status {
    Pending,
    Failed,
    Published(u32),
}

/// Not cloneable: one completed capture owns at most one publication attempt.
pub(crate) struct CaptureReceipt {
    bound: BoundCapturePlan,
    next_tap: usize,
    status: Status,
}

impl CaptureReceipt {
    pub(crate) fn new(bound: BoundCapturePlan) -> Self {
        Self {
            bound,
            next_tap: 0,
            status: Status::Pending,
        }
    }

    pub(crate) fn validate_staged(&self, rows: u32, position: u32) -> Result<()> {
        ensure!(
            matches!(self.status, Status::Pending) && self.next_tap == 0,
            "staged capture requires a fresh receipt"
        );
        ensure!(
            rows == self.bound.plan().rows() && position == self.bound.plan().start_position(),
            "staged capture geometry/position differs from its checked binding"
        );
        Ok(())
    }

    /// A real dispatcher tap must finish enqueueing before its receipt is recorded.
    /// Partial enqueue failure is terminal; the owning bank performs the drain.
    pub(crate) fn enqueue_tap(
        &mut self,
        layer: u32,
        slot: usize,
        enqueue: impl FnOnce(DeviceSpan) -> Result<()>,
    ) -> Result<()> {
        ensure!(
            matches!(self.status, Status::Pending),
            "capture receipt is terminal"
        );
        ensure!(
            slot == self.next_tap,
            "capture taps must enqueue once in canonical order"
        );
        let destination = self.bound.tap_destination(layer, slot)?;
        if let Err(error) = enqueue(destination) {
            self.status = Status::Failed;
            return Err(error.context("capture tap enqueue failed; owner must drain/reset"));
        }
        self.record_tap(layer, slot)
    }

    /// The dispatcher calls this only after the corresponding capture operation
    /// successfully enqueues on the ingestion stream. No missing tap is inferred.
    pub(crate) fn record_tap(&mut self, layer: u32, slot: usize) -> Result<()> {
        ensure!(
            matches!(self.status, Status::Pending),
            "capture receipt is terminal"
        );
        ensure!(
            slot == self.next_tap,
            "capture taps must complete once in canonical order"
        );
        self.bound.tap_destination(layer, slot)?;
        self.next_tap += 1;
        Ok(())
    }

    pub(crate) fn published_position(&self) -> Option<u32> {
        match self.status {
            Status::Published(position) => Some(position),
            _ => None,
        }
    }

    pub(crate) fn failed(&self) -> bool {
        matches!(self.status, Status::Failed)
    }

    /// Target is already at the whole chunk's end; drafter remains at its start.
    /// The caller updates context only with this method's successful return value.
    pub(crate) fn ingest(
        &mut self,
        io: &mut impl PrefillCaptureIo,
        target_position: u32,
        context_tokens: u32,
        stream: u64,
    ) -> Result<u32> {
        ensure!(
            self.published_position().is_none() && !self.failed(),
            "capture receipt is terminal"
        );
        ensure!(
            self.next_tap == GLM53_CAPTURE_LAYERS.len(),
            "capture has incomplete taps"
        );
        let plan = self.bound.plan();
        ensure!(
            target_position == plan.end_position(),
            "target is not at completed capture end"
        );
        ensure!(
            context_tokens == plan.start_position(),
            "drafter context changed before ingestion"
        );
        let end = plan.end_position();

        // Preflight every address and slice before the first I/O effect. At most
        // 256 slices / 10235 copy descriptors; no additional device allocation.
        let slices = (0..plan.rows())
            .step_by(GATHER_ROWS as usize)
            .map(|first| {
                self.bound
                    .slice(first, GATHER_ROWS.min(plan.rows() - first))
            })
            .collect::<Result<Vec<_>>>()?;

        // A partial-write failure (or backend panic) cannot leave a reusable
        // receipt. Only the completed final fence promotes it to Published.
        self.status = Status::Failed;
        let execution = (|| -> Result<()> {
            for slice in slices {
                for copy in slice.copies {
                    io.copy(copy, stream)?;
                }
                io.project_norm(slice.projection, stream)?;
            }
            io.synchronize(stream)
        })();
        if let Err(error) = execution {
            // Even an enqueue failure can follow prior asynchronous writes.
            // Preserve terminal failure regardless of whether this drain works.
            return match io.synchronize(stream) {
                Ok(()) => {
                    Err(error.context("capture ingestion failed; stream drained; reset required"))
                }
                Err(drain) => Err(error.context(format!(
                    "capture ingestion failed; stream drain also failed ({drain:#}); reset required"
                ))),
            };
        }
        self.status = Status::Published(end);
        Ok(end)
    }
}
