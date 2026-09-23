// SPDX-License-Identifier: AGPL-3.0-only

//! Large target capture ingestion reuses the installed drafter's eight-row FC arena.
use super::*;
use crate::model::glm53::prefill_capture_ingest::{CaptureReceipt, PrefillCaptureIo};
use crate::model::glm53::prefill_capture_plan::{
    CopyStep, DeviceSpan, PrefillCapturePlan, ProjectionStep,
};

struct CaptureIngestIo<'a> {
    runtime: &'a Glm53Dflash2Runtime,
    gpu: &'a dyn GpuBackend,
}

impl PrefillCaptureIo for CaptureIngestIo<'_> {
    fn copy(&mut self, step: CopyStep, stream: u64) -> Result<()> {
        self.gpu.copy_d2d_async(
            DevicePtr(step.source),
            DevicePtr(step.destination),
            step.bytes,
            stream,
        )
    }

    fn project_norm(&mut self, step: ProjectionStep, stream: u64) -> Result<()> {
        ensure!(
            step.input.address == self.runtime.capture_input.0
                && (1..=QUERY_TOKENS).contains(&step.rows),
            "capture FC staging contract drift"
        );
        dense(
            DevicePtr(step.input.address),
            self.runtime.weights.fc.weight,
            DevicePtr(step.output.address),
            step.rows,
            HIDDEN,
            5 * HIDDEN,
            stream,
        )?;
        ops::rms_norm(
            self.gpu,
            self.runtime.rms_norm,
            DevicePtr(step.output.address),
            &self.runtime.weights.hidden_norm,
            DevicePtr(step.output.address),
            step.rows,
            HIDDEN,
            1.0e-5,
            stream,
        )
    }

    fn synchronize(&mut self, stream: u64) -> Result<()> {
        self.gpu.synchronize(stream)
    }
}

impl Glm53Dflash2Runtime {
    pub(in crate::model::glm53) fn bind_prefill_capture(
        &self,
        plan: PrefillCapturePlan,
        bank: DeviceSpan,
    ) -> Result<CaptureReceipt> {
        ensure!(
            plan.start_position() == self.context_tokens,
            "capture binding cursor drift"
        );
        let projected = self.region(self.plan.projected_target);
        let bound = plan.bind(
            bank,
            DeviceSpan {
                address: self.capture_input.0,
                bytes: QUERY_TOKENS as usize * 5 * HIDDEN as usize * 2,
            },
            DeviceSpan {
                address: projected.ptr.0,
                bytes: projected.bytes,
            },
        )?;
        Ok(CaptureReceipt::new(bound))
    }

    pub(in crate::model::glm53) fn observe_prefill_capture(
        &mut self,
        receipt: &mut CaptureReceipt,
        gpu: &dyn GpuBackend,
        target_position: u32,
        stream: u64,
    ) -> Result<()> {
        let end = {
            let mut io = CaptureIngestIo { runtime: self, gpu };
            receipt.ingest(&mut io, target_position, self.context_tokens, stream)?
        };
        // No intermediate FC slice advances this cursor. ingest returns only
        // after all five taps, every ordered slice and the final stream fence.
        self.context_tokens = end;
        Ok(())
    }
}
