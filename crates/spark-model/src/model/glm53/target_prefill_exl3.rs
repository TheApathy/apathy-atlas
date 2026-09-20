// SPDX-License-Identifier: AGPL-3.0-only

//! Borrowed mixed prompt inputs; target state/capture arithmetic remains shared.

use super::super::prefill_exl3::{
    WidePrefillConfig, mixed_prefill_rows, requested_wide_prefill, validate_prefill_mode,
};
use super::super::prefill_input_exl3::{DraftContext, InputCopyIo, InputPlan, InputRegion};
use super::*;
use crate::layers::ops::with_glm53_exact_wide_prefill;

struct GpuInputIo<'a>(&'a dyn GpuBackend);
impl InputCopyIo for GpuInputIo<'_> {
    fn copy(&mut self, source: u64, destination: u64, bytes: usize, stream: u64) -> Result<()> {
        self.0
            .copy_d2d_async(DevicePtr(source), DevicePtr(destination), bytes, stream)
    }
}

impl Glm53Exl3Model {
    pub(in crate::model::glm53) fn prefill_request(
        &self,
        tokens: &[u32],
        stream: u64,
    ) -> Result<DevicePtr> {
        self.retry_prepared_vision()?;
        let prepared = self.prepared_vision.lock().unwrap().begin()?;
        let mut effects_started = false;
        let result = (|| {
            let config = requested_wide_prefill(prepared.is_some())?;
            // This is admission for a NEW sequence. The prior runtime cursor is
            // reset below, but its actual capacity is authoritative before reset.
            let draft = self
                .dflash2
                .lock()
                .unwrap()
                .as_ref()
                .map(|runtime| DraftContext {
                    position: 0,
                    capacity: runtime.context_capacity(),
                });
            let plan = InputPlan::new(
                tokens,
                prepared,
                InputRegion {
                    address: self.weights.embedding.ptr().0,
                    bytes: self.weights.embedding.bytes(),
                },
                0,
                self.capacity,
                draft,
            )?;
            let large_capture = self.preflight_prefill_capture(config, tokens.len())?;
            if !large_capture {
                validate_prefill_mode(config, draft.is_some())?;
            }
            let mixed_rows = if prepared.is_some() {
                Some(mixed_prefill_rows(config)?)
            } else {
                None
            };
            if let Some(rows) = mixed_rows {
                self.preflight_mixed_inputs(&plan, rows)?;
            }
            effects_started = true;
            self.reset_sequence()?;
            if let Some(rows) = mixed_rows {
                self.arm_prepared_vision(stream)?;
                self.prefill_inputs_wide(&plan, config, rows, stream)
            } else if let Some(config) = config {
                self.prefill_tokens_wide(tokens, config, stream)
            } else {
                let mut logits = self.logits;
                for &token in tokens {
                    self.preflight_prefill_rows(self.position(), 1)?;
                    logits = self.walk(token, stream)?;
                }
                Ok(logits)
            }
        })();
        if result.is_err() && effects_started {
            // Reset uses the default stream; retain its poison if that failed.
            let mut state = self.state.lock().unwrap();
            if state.poisoned_stream.is_none() {
                state.poisoned_stream = Some(stream);
            }
        }
        let cleanup = self.finish_prepared_vision();
        if cleanup.is_err() {
            let mut state = self.state.lock().unwrap();
            if state.poisoned_stream.is_none() {
                state.poisoned_stream = Some(stream);
            }
        }
        match (result, cleanup) {
            (Ok(logits), Ok(())) => Ok(logits),
            (Err(error), Ok(())) => Err(error),
            (Ok(_), Err(error)) => Err(error).context("GLM prepared output cleanup failed"),
            (Err(error), Err(cleanup)) => Err(error.context(format!(
                "GLM prepared cleanup also failed; owner retained: {cleanup:#}"
            ))),
        }
    }

    pub(super) fn preflight_prefill_rows(&self, position: u32, rows: u32) -> Result<()> {
        ensure!(
            position == self.position()
                && rows > 0
                && position
                    .checked_add(rows)
                    .is_some_and(|end| end <= self.capacity),
            "GLM prefill chunk target cursor/capacity drift"
        );
        if let Some(runtime) = self.dflash2.lock().unwrap().as_ref() {
            runtime.preflight_target_rows(position, rows)?;
        }
        Ok(())
    }

    fn mixed_input_destination(&self, rows: usize) -> Result<GgmlIqBuffer> {
        ensure!(
            (1..=2_048).contains(&rows),
            "GLM mixed input row geometry is invalid"
        );
        if rows == 1 {
            return Ok(self.workspace()?.collapsed);
        }
        let schedule = Glm53TargetSchedule::new(Glm53TargetGeometry::exact(u32::try_from(rows)?))?;
        let base = self
            .wide_workspace_allocation
            .0
            .checked_add(255)
            .context("GLM wide workspace address overflow")?
            & !255;
        let alignment = usize::try_from(base - self.wide_workspace_allocation.0)?;
        ensure!(
            usize::try_from(schedule.workspace.arena_bytes)?
                .checked_add(alignment)
                .is_some_and(|end| end <= self.wide_workspace_bytes),
            "GLM mixed input workspace exceeds its owned allocation"
        );
        Ok(Glm53BoundWorkspace::bind(
            &schedule.workspace,
            DevicePtr(base),
            schedule.workspace.arena_bytes,
        )?
        .collapsed)
    }

    fn preflight_mixed_inputs(&self, plan: &InputPlan<'_>, max_rows: usize) -> Result<()> {
        ensure!(
            matches!(
                max_rows,
                1 | 2 | 4 | 8 | 16 | 32 | 64 | 128 | 256 | 512 | 1_024 | 2_048
            ),
            "GLM mixed input chunk geometry is invalid"
        );
        // Bounded temporary descriptors only; no prompt-sized/device allocation.
        for start in (0..plan.rows()).step_by(max_rows) {
            let end = (start + max_rows).min(plan.rows());
            let destination = self.mixed_input_destination(end - start)?;
            let region = InputRegion {
                address: destination.ptr.0,
                bytes: destination.bytes,
            };
            if end - start <= 8 {
                plan.chunk(start..end, region)?;
            } else {
                plan.token_slice(start..end)?;
                plan.preflight_vision_overwrite(start..end, region)?;
            }
        }
        Ok(())
    }

    pub(super) fn prefill_inputs_wide(
        &self,
        plan: &InputPlan<'_>,
        config: Option<WidePrefillConfig>,
        max_rows: usize,
        stream: u64,
    ) -> Result<DevicePtr> {
        ensure!(
            max_rows == mixed_prefill_rows(config)?,
            "GLM mixed input row configuration drift"
        );
        let layer_major = config.is_some_and(WidePrefillConfig::is_layer_major);
        let mut final_logits = self.logits;
        for start in (0..plan.rows()).step_by(max_rows) {
            let end = (start + max_rows).min(plan.rows());
            let rows = end - start;
            if !layer_major {
                self.preflight_prefill_rows(self.position(), u32::try_from(rows)?)?;
            }
            if rows == 1 {
                let destination = self.mixed_input_destination(1)?;
                let batch = plan.chunk(
                    start..end,
                    InputRegion {
                        address: destination.ptr.0,
                        bytes: destination.bytes,
                    },
                )?;
                ensure!(
                    batch.start_position() == self.position(),
                    "GLM mixed T1 position drift"
                );
                let source = batch.single_source()?;
                final_logits = self.walk_input(
                    WalkInput::ExternalEmbedding(DevicePtr(source.address)),
                    stream,
                )?;
                continue;
            }
            let logits = if layer_major {
                self.prefill_layer_major_dflash2_with_fill(
                    rows,
                    max_rows,
                    stream,
                    |destination, position| {
                        let region = InputRegion {
                            address: destination.ptr.0,
                            bytes: destination.bytes,
                        };
                        ensure!(
                            position == self.position(),
                            "GLM mixed layer-major position drift"
                        );
                        self.embed_dflash_tokens(
                            plan.token_slice(start..end)?,
                            destination,
                            stream,
                        )?;
                        plan.overwrite_vision(
                            start..end,
                            region,
                            &mut GpuInputIo(self.gpu.as_ref()),
                            stream,
                        )
                    },
                )?
            } else {
                let logits = with_glm53_exact_wide_prefill(|| -> Result<DevicePtr> {
                    let (logits, position, count) =
                        self.verify_inputs_staged(rows, stream, |destination, position| {
                            let batch = plan.chunk(
                                start..end,
                                InputRegion {
                                    address: destination.ptr.0,
                                    bytes: destination.bytes,
                                },
                            )?;
                            ensure!(
                                batch.start_position() == position && batch.rows() == rows,
                                "GLM mixed wide position/row drift"
                            );
                            batch.enqueue(&mut GpuInputIo(self.gpu.as_ref()), stream)
                        })?;
                    self.commit_accepted(stream)?;
                    self.gpu.synchronize(stream)?;
                    self.state.lock().unwrap().position = position + count;
                    Ok(logits)
                })?;
                self.observe_committed_wide_rows(u32::try_from(rows)?, stream)?;
                logits
            };
            final_logits = logits.offset((rows - 1) * VOCAB as usize * 2);
        }
        Ok(final_logits)
    }
}
