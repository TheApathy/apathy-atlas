// SPDX-License-Identifier: AGPL-3.0-only

//! Prepared output ownership and reset barriers for the C1 GLM target.

use super::super::prefill_input_exl3::{InputRegion, PreparedRows};
use super::super::prefill_owner_exl3::OwnerIo;
use super::*;

struct GpuOwnerIo<'a>(&'a dyn GpuBackend);
impl OwnerIo for GpuOwnerIo<'_> {
    fn drain(&mut self, stream: u64) -> Result<()> {
        self.0.synchronize(stream)
    }
    fn free(&mut self, address: u64) -> Result<()> {
        self.0.free(DevicePtr(address))
    }
}

impl Glm53Exl3Model {
    pub(super) fn retry_prepared_vision(&self) -> Result<()> {
        self.prepared_vision
            .lock()
            .unwrap()
            .retry_quarantine(&mut GpuOwnerIo(self.gpu.as_ref()))
    }

    pub(super) fn clear_prepared_vision(&self) -> Result<()> {
        self.drain_policy_readback()?;
        // Reject an active owner before any other effects. Ready outputs have
        // an encoder completion receipt; quarantined outputs must retry drain.
        self.prepared_vision
            .lock()
            .unwrap()
            .clear(&mut GpuOwnerIo(self.gpu.as_ref()))?;
        if let Some(stream) = self.state.lock().unwrap().poisoned_stream {
            self.gpu
                .synchronize(stream)
                .context("GLM poisoned work must drain before preparation/shutdown")?;
        }
        Ok(())
    }

    pub(super) fn arm_prepared_vision(&self, stream: u64) -> Result<()> {
        self.prepared_vision.lock().unwrap().arm(stream)
    }

    pub(super) fn finish_prepared_vision(&self) -> Result<()> {
        self.prepared_vision
            .lock()
            .unwrap()
            .finish(&mut GpuOwnerIo(self.gpu.as_ref()))
    }

    pub(in crate::model::glm53) fn reset_sequence(&self) -> Result<()> {
        {
            let mut state = self.state.lock().unwrap();
            state.generation = state
                .generation
                .checked_add(1)
                .context("GLM request generation exhausted before reset")?;
        }
        self.drain_policy_readback()?;
        self.retry_prepared_vision()?;
        self.release_prefill_capture_bank()?;
        let failed_stream = self.state.lock().unwrap().poisoned_stream;
        if let Some(stream) = failed_stream {
            self.gpu
                .synchronize(stream)
                .context("GLM poisoned stream must drain before reset")?;
        }
        self.state.lock().unwrap().poisoned_stream = Some(self.gpu.default_stream());
        if let Some(runtime) = self
            .dflash2
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_mut()
        {
            runtime.reset_context(self.gpu.as_ref())?;
        }
        self.dflash2.clear_poison();
        self.gpu.memset(
            self.arena,
            0,
            usize::try_from(self.plan.known_bytes).context("GLM EXL3 reset arena span")?,
        )?;
        self.gpu
            .memset(self.scratch_allocation, 0, self.scratch_bytes)?;
        self.gpu
            .memset(self.wide_workspace_allocation, 0, self.wide_workspace_bytes)?;
        self.gpu.synchronize(self.gpu.default_stream())?;
        let mut state = self.state.lock().unwrap();
        state.position = 0;
        state.poisoned_stream = None;
        Ok(())
    }

    pub(in crate::model::glm53) fn prepare_vision_images(
        &self,
        images: &[(Vec<f32>, usize, usize)],
    ) -> Result<()> {
        self.clear_prepared_vision()?;
        if images.is_empty() {
            return Ok(());
        }
        ensure!(
            images.len() <= self.capacity as usize,
            "GLM image count exceeds the configured target context"
        );
        let catalog = self
            .vision_catalog
            .as_ref()
            .context("GLM EXL3 image input requires an owned vision catalog")?;
        let plans = images
            .iter()
            .map(|(pixels, height, width)| {
                let plan = Glm53VisionPlan::new(*height, *width)?;
                let expected = usize::try_from(u64::from(plan.rows) * 3 * 2 * 14 * 14)?;
                ensure!(
                    pixels.len() == expected,
                    "GLM vision pixel extent does not match grid"
                );
                Ok(plan)
            })
            .collect::<Result<Vec<_>>>()?;
        let total_rows = plans.iter().try_fold(0usize, |rows, plan| {
            rows.checked_add(usize::try_from(plan.merged_rows)?)
                .context("GLM EXL3 prepared vision row count overflow")
        })?;
        ensure!(
            total_rows > 0 && total_rows <= self.capacity as usize,
            "GLM prepared vision exceeds the configured target context"
        );
        let total_bytes = total_rows
            .checked_mul(HIDDEN as usize * 2)
            .context("GLM EXL3 prepared vision byte count overflow")?;
        // Keep this mutex through encoding: no caller can borrow a partially
        // prepared output, and the encoder never calls back into this model.
        let mut owner = self.prepared_vision.lock().unwrap();
        ensure!(
            !owner.has_owner(),
            "GLM prepared owner changed during admission"
        );
        let allocation = self.gpu.alloc(total_bytes)?;
        if let Err(error) = owner.publish(PreparedRows {
            region: InputRegion {
                address: allocation.0,
                bytes: total_bytes,
            },
            rows: total_rows,
        }) {
            // No work was queued. This is a backend allocation-contract failure.
            return match self.gpu.free(allocation) {
                Ok(()) => Err(error),
                Err(cleanup) => Err(error.context(format!(
                    "GLM invalid output allocation cleanup failed: {cleanup:#}"
                ))),
            };
        }
        ensure!(
            owner.begin()?.is_some(),
            "GLM new output owner was not retained"
        );
        let stream = self.gpu.default_stream();
        owner.arm(stream)?;
        let encode = (|| -> Result<()> {
            let mut row_offset = 0usize;
            for ((pixels, height, width), plan) in images.iter().zip(&plans) {
                let rows = usize::try_from(plan.merged_rows)?;
                let bytes = rows * HIDDEN as usize * 2;
                let receipt = forward_glm53_exl3_image(
                    self.gpu.as_ref(),
                    catalog,
                    pixels,
                    *height,
                    *width,
                    Glm53Exl3Buffer {
                        ptr: allocation.offset(row_offset * HIDDEN as usize * 2),
                        bytes,
                    },
                    stream,
                )?;
                ensure!(
                    usize::try_from(receipt.rows)? == rows
                        && receipt.width == HIDDEN
                        && receipt.bytes == bytes,
                    "GLM EXL3 vision forward receipt drift"
                );
                row_offset += rows;
            }
            ensure!(
                row_offset == total_rows,
                "GLM EXL3 prepared vision row packing drift"
            );
            Ok(())
        })();
        match encode {
            Ok(()) => {
                let result = owner.complete_preparation(&mut GpuOwnerIo(self.gpu.as_ref()));
                if result.is_err() {
                    self.state.lock().unwrap().poisoned_stream = Some(stream);
                }
                result
            }
            Err(error) => {
                self.state.lock().unwrap().poisoned_stream = Some(stream);
                match owner.finish(&mut GpuOwnerIo(self.gpu.as_ref())) {
                    Ok(()) => Err(error),
                    Err(cleanup) => Err(error.context(format!(
                        "GLM vision output cleanup failed; owner retained: {cleanup:#}"
                    ))),
                }
            }
        }
    }
}
