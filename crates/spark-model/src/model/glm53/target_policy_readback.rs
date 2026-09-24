// SPDX-License-Identifier: AGPL-3.0-only

//! Device readback through storage that outlives uncertain CUDA completion.

use super::*;
use crate::model::glm53::owned_verify_readback::ReadbackIo;
use crate::model::glm53::state_read_plan::StateReadPlan;

struct DeviceReadback<'a> {
    gpu: &'a dyn GpuBackend,
    source: DevicePtr,
}

impl ReadbackIo for DeviceReadback<'_> {
    fn copy(&mut self, destination: &mut [u8], stream: u64) -> Result<()> {
        self.gpu
            .copy_d2h_on_stream(self.source, destination, stream)
    }
    fn drain(&mut self, stream: u64) -> Result<()> {
        self.gpu.synchronize(stream)
    }
}

impl Glm53Exl3Model {
    pub(super) fn copy_state_probe_region(
        &self,
        plan: &StateReadPlan,
        destination: &mut [u8],
        stream: u64,
    ) -> Result<()> {
        self.ensure_verify_healthy()?;
        ensure!(
            destination.len() == plan.logical_range().len(),
            "state probe destination extent changed"
        );
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<()> {
            let mut owner = self
                .verify_readback
                .lock()
                .map_err(|_| anyhow::anyhow!("state readback owner panicked; reset required"))?;
            let mut io = DeviceReadback {
                gpu: self.gpu.as_ref(),
                source: DevicePtr(plan.source()),
            };
            let owned = owner.read(plan.physical_bytes(), stream, &mut io)?;
            destination.copy_from_slice(&owned[plan.logical_range()]);
            Ok(())
        }));
        match result {
            Ok(Ok(())) => Ok(()),
            Ok(Err(error)) => {
                self.poison_verify(stream);
                Err(error.context("state probe readback failed; sequence poisoned"))
            }
            Err(_) => {
                self.poison_verify(stream);
                bail!("state probe readback panicked; model and pending owner retained for reset")
            }
        }
    }

    pub(super) fn copy_policy_logits(
        &self,
        source: DevicePtr,
        bytes: usize,
        destination: &mut [u8],
        stream: u64,
    ) -> Result<()> {
        self.ensure_verify_healthy()?;
        ensure!(
            bytes == destination.len(),
            "GLM policy readback destination extent changed"
        );
        let offset = source
            .0
            .checked_sub(self.logits.0)
            .context("GLM logits source precedes allocation")?;
        let offset = usize::try_from(offset)?;
        let allocation = self.scratch.max_wide_rows() as usize * VOCAB as usize * 2;
        ensure!(
            offset % 2 == 0
                && offset
                    .checked_add(bytes)
                    .is_some_and(|end| end <= allocation),
            "GLM logits source span exceeds its BF16 allocation"
        );
        let mut owner = self
            .verify_readback
            .lock()
            .map_err(|_| anyhow::anyhow!("GLM readback owner panicked; reset required"))?;
        let mut io = DeviceReadback {
            gpu: self.gpu.as_ref(),
            source,
        };
        let read = owner.read(bytes, stream, &mut io);
        match read {
            Ok(owned) => {
                destination.copy_from_slice(owned);
                Ok(())
            }
            Err(error) => {
                self.poison_verify(stream);
                Err(error.context("GLM logits readback failed; sequence poisoned"))
            }
        }
    }

    pub(super) fn drain_policy_readback(&self) -> Result<()> {
        // A caught I/O panic poisons the mutex, but its owned buffer and pending
        // stream are precisely the state reset must retain and explicitly drain.
        let mut owner = self
            .verify_readback
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        owner.drain(&mut DeviceReadback {
            gpu: self.gpu.as_ref(),
            source: DevicePtr::NULL,
        })?;
        self.verify_readback.clear_poison();
        Ok(())
    }
}
