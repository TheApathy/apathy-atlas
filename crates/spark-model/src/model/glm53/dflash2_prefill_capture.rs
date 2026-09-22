// SPDX-License-Identifier: AGPL-3.0-only

//! Large target capture ingestion uses a selector-bounded FC staging arena.
use super::*;
use crate::model::glm53::prefill_capture_ingest::{CaptureReceipt, PrefillCaptureIo};
use crate::model::glm53::prefill_capture_plan::{
    CaptureTransferMode, CopyStep, DeviceSpan, GatherStep, PrefillCapturePlan, ProjectionStep,
};
use spark_runtime::kernel_args::KernelLaunch;

pub(super) fn capture_tile_mode() -> Result<CaptureTransferMode> {
    CaptureTransferMode::parse(std::env::var_os("ATLAS_GLM53_DFLASH2_CAPTURE_TILE128").as_deref())
}

pub(super) struct CaptureTileResources {
    pub(super) input: DevicePtr,
    pub(super) bytes: usize,
    pub(super) mode: CaptureTransferMode,
    pub(super) gather: Option<KernelHandle>,
}

impl CaptureTileResources {
    pub(super) fn load(gpu: &dyn GpuBackend, mode: CaptureTransferMode) -> Result<Self> {
        let bytes = mode.staging_bytes(HIDDEN as usize * 2)?;
        let gather = match mode {
            CaptureTransferMode::Copies8 => None,
            CaptureTransferMode::Gather128 => Some(gpu.kernel(
                "glm53_dflash2_capture",
                "atlas_glm53_dflash2_gather_capture_tile",
            )?),
        };
        Ok(Self {
            input: gpu.alloc(bytes)?,
            bytes,
            mode,
            gather,
        })
    }
}

struct CaptureIngestIo<'a> {
    runtime: &'a Glm53Dflash2Runtime,
    gpu: &'a dyn GpuBackend,
}

impl PrefillCaptureIo for CaptureIngestIo<'_> {
    fn copy(&mut self, step: CopyStep, stream: u64) -> Result<()> {
        ensure!(
            self.runtime.capture.mode == CaptureTransferMode::Copies8,
            "capture copy path is disabled by tile selector"
        );
        self.gpu.copy_d2d_async(
            DevicePtr(step.source),
            DevicePtr(step.destination),
            step.bytes,
            stream,
        )
    }

    fn gather(&mut self, step: GatherStep, stream: u64) -> Result<()> {
        ensure!(
            self.runtime.capture.mode == CaptureTransferMode::Gather128
                && step.destination.address == self.runtime.capture.input.0
                && (1..=128).contains(&step.rows)
                && step.row_bytes == HIDDEN as usize * 2
                && step.slot_stride_bytes % step.row_bytes == 0,
            "capture gather staging contract drift"
        );
        let slot_stride_elements = u32::try_from(step.slot_stride_bytes / 2)
            .context("capture gather slot stride exceeds u32")?;
        let total_elements = step
            .rows
            .checked_mul(5 * HIDDEN)
            .context("capture gather element count overflow")?;
        let kernel = self
            .runtime
            .capture
            .gather
            .context("capture gather kernel is unavailable")?;
        KernelLaunch::new(self.gpu, kernel)
            .grid([total_elements.div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(DevicePtr(step.source.address))
            .arg_ptr(DevicePtr(step.destination.address))
            .arg_u32(step.first_row)
            .arg_u32(step.rows)
            .arg_u32(slot_stride_elements)
            .launch(stream)
    }

    fn project_norm(&mut self, step: ProjectionStep, stream: u64) -> Result<()> {
        ensure!(
            step.input.address == self.runtime.capture.input.0
                && (1..=self.runtime.capture.mode.rows()).contains(&step.rows)
                && step.input.bytes == step.rows as usize * 5 * HIDDEN as usize * 2
                && step.output.bytes == step.rows as usize * HIDDEN as usize * 2,
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
                address: self.capture.input.0,
                bytes: self.capture.bytes,
            },
            DeviceSpan {
                address: projected.ptr.0,
                bytes: projected.bytes,
            },
            self.capture.mode,
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
