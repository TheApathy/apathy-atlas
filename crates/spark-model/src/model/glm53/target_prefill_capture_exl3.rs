// SPDX-License-Identifier: AGPL-3.0-only

//! Owned, bounded layer-major TEXT capture path. Ordinary K8 slots remain separate.
use super::*;
use crate::layers::ops::with_glm53_layer_major_prefill;
use crate::model::glm53::prefill_capture_owner::CaptureBankIo;
use crate::model::glm53::prefill_capture_plan::PrefillCapturePlan;
use crate::model::glm53::prefill_exl3::WidePrefillConfig;

struct GpuCaptureBankIo<'a>(&'a dyn GpuBackend);
impl CaptureBankIo for GpuCaptureBankIo<'_> {
    fn allocate(&mut self, bytes: usize) -> Result<u64> {
        self.0.alloc(bytes).map(|ptr| ptr.0)
    }
    fn synchronize(&mut self, stream: u64) -> Result<()> {
        self.0.synchronize(stream)
    }
    fn free(&mut self, address: u64) -> Result<()> {
        self.0.free(DevicePtr(address))
    }
}

impl Glm53Exl3Model {
    /// Pure admission for a fresh request. Checks EVERY future chunk against
    /// actual installed context capacity before reset, allocation or model effects.
    pub(in crate::model::glm53) fn preflight_prefill_capture(
        &self,
        config: Option<WidePrefillConfig>,
        prompt_rows: usize,
    ) -> Result<bool> {
        let Some(capacity) = config.and_then(|config| config.layer_major_rows()) else {
            return Ok(false);
        };
        let guard = self.dflash2.lock().unwrap();
        let Some(runtime) = guard.as_ref() else {
            return Ok(false);
        };
        let context_limit = runtime.context_capacity();
        let total = u32::try_from(prompt_rows)?;
        let capacity = u32::try_from(capacity)?;
        ensure!(
            total > 0 && total <= context_limit && total <= self.capacity,
            "GLM layer-major DFlash2 prompt exceeds installed target/drafter capacity"
        );
        PrefillCapturePlan::new(
            capacity,
            capacity.min(total),
            0,
            0,
            context_limit,
            self.capacity,
        )?;
        self.preflight_capture_workspace(capacity.min(total))?;
        for start in (capacity..total).step_by(capacity as usize) {
            PrefillCapturePlan::new(
                capacity,
                capacity.min(total - start),
                start,
                start,
                context_limit,
                self.capacity,
            )?;
            self.preflight_capture_workspace(capacity.min(total - start))?;
        }
        Ok(true)
    }

    fn preflight_capture_workspace(&self, rows: u32) -> Result<()> {
        if rows == 1 {
            return Ok(());
        }
        let schedule = Glm53TargetSchedule::new(Glm53TargetGeometry::exact(rows))?;
        let base = self
            .wide_workspace_allocation
            .0
            .checked_add(255)
            .context("capture workspace alignment overflow")?
            & !255;
        let padding = usize::try_from(base - self.wide_workspace_allocation.0)?;
        ensure!(
            usize::try_from(schedule.workspace.arena_bytes)?
                .checked_add(padding)
                .is_some_and(|required| required <= self.wide_workspace_bytes),
            "capture chunk exceeds the model's owned wide workspace"
        );
        Ok(())
    }

    pub(in crate::model::glm53) fn release_prefill_capture_bank(&self) -> Result<()> {
        self.prefill_capture_bank
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .release(&mut GpuCaptureBankIo(self.gpu.as_ref()))
    }

    /// One complete multirow chunk. Both locks stay held through target forward,
    /// persistent commit, capture ingestion and the owner completion fence.
    pub(in crate::model::glm53) fn prefill_layer_major_dflash2(
        &self,
        tokens: &[u32],
        capacity_rows: usize,
        stream: u64,
    ) -> Result<DevicePtr> {
        self.prefill_layer_major_dflash2_with_fill(
            tokens.len(),
            capacity_rows,
            stream,
            |destination, _position| self.embed_dflash_tokens(tokens, destination, stream),
        )
    }

    pub(in crate::model::glm53) fn prefill_layer_major_dflash2_with_fill(
        &self,
        input_rows: usize,
        capacity_rows: usize,
        stream: u64,
        fill: impl FnOnce(GgmlIqBuffer, u32) -> Result<()>,
    ) -> Result<DevicePtr> {
        self.ensure_verify_healthy()?;
        ensure!(
            input_rows >= 2,
            "single-row tail must use the ordinary walk observer"
        );
        let mut owner = self
            .prefill_capture_bank
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let mut runtime_guard = self.dflash2.lock().unwrap();
        let runtime = runtime_guard
            .as_mut()
            .context("layer-major capture requires installed DFlash2")?;
        let plan = PrefillCapturePlan::new(
            u32::try_from(capacity_rows)?,
            u32::try_from(input_rows)?,
            self.position(),
            runtime.context_tokens(),
            runtime.context_capacity(),
            self.capacity,
        )?;
        self.preflight_capture_workspace(u32::try_from(input_rows)?)?;
        let mut io = GpuCaptureBankIo(self.gpu.as_ref());
        let bank = owner.begin(&mut io, &plan, stream)?;
        let result = (|| -> Result<DevicePtr> {
            let mut receipt = runtime.bind_prefill_capture(plan, bank)?;
            let logits = with_glm53_layer_major_prefill(|| -> Result<DevicePtr> {
                let (logits, position, rows) = self.verify_inputs_staged_with_capture(
                    input_rows,
                    stream,
                    Some(&mut receipt),
                    fill,
                )?;
                self.commit_accepted(stream)?;
                self.gpu.synchronize(stream)?;
                self.state.lock().unwrap().position = position + rows;
                Ok(logits)
            })?;
            runtime.observe_prefill_capture(
                &mut receipt,
                self.gpu.as_ref(),
                self.position(),
                stream,
            )?;
            owner.complete(&mut io)?;
            Ok(logits)
        })();
        match result {
            Ok(logits) => Ok(logits),
            Err(error) => {
                self.state.lock().unwrap().poisoned_stream = Some(stream);
                match owner.abort(&mut io) {
                    Ok(()) => Err(error.context("GLM layer-major capture failed; target/drafter reset required")),
                    Err(drain) => Err(error.context(format!(
                        "GLM layer-major capture failed; bank retained after drain failure: {drain:#}"
                    ))),
                }
            }
        }
    }
}
