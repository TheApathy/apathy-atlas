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
    pub(super) fn copy_policy_argmax(
        &self,
        source: DevicePtr,
        rows: usize,
        destination: &mut [u32],
        excluded: [u32; 2],
        stream: u64,
    ) -> Result<()> {
        self.ensure_verify_healthy()?;
        ensure!(
            (2..=DFLASH2_MAX_ROWS).contains(&rows) && destination.len() == rows,
            "GLM policy compact argmax extent changed"
        );
        ensure!(
            excluded
                .iter()
                .all(|token| *token == u32::MAX || *token < VOCAB),
            "GLM policy compact exclusion is outside vocabulary"
        );
        let bytes = rows
            .checked_mul(VOCAB as usize)
            .and_then(|values| values.checked_mul(2))
            .context("GLM policy compact logits extent overflow")?;
        let offset = source
            .0
            .checked_sub(self.logits.0)
            .context("GLM compact logits source precedes allocation")?;
        ensure!(
            offset == 0 && bytes <= GLM53_EXL3_MAX_WIDE_ROWS as usize * VOCAB as usize * 2,
            "GLM compact logits source span exceeds allocation"
        );
        let kernel = self
            .gpu
            .kernel("glm53_kda", "atlas_glm53_policy_argmax_bf16_rows")?;
        let values = self
            .dflash2_argmax_allocation
            .offset(DFLASH2_ARGMAX_INDEX_BYTES);
        KernelLaunch::new(self.gpu.as_ref(), kernel)
            .grid([u32::try_from(rows)?, 1, 1])
            .block([1024, 1, 1])
            .arg_ptr(source)
            .arg_ptr(self.dflash2_argmax_allocation)
            .arg_ptr(values)
            .arg_u32(u32::try_from(rows)?)
            .arg_u32(VOCAB)
            .arg_u32(excluded[0])
            .arg_u32(excluded[1])
            .launch(stream)?;
        let mut receipt = [0u8; DFLASH2_ARGMAX_BYTES];
        self.gpu
            .copy_d2h_on_stream(self.dflash2_argmax_allocation, &mut receipt, stream)?;
        for (row, token) in destination.iter_mut().enumerate() {
            let index_offset = row * size_of::<u32>();
            let value_offset = DFLASH2_ARGMAX_INDEX_BYTES + row * size_of::<f32>();
            *token = u32::from_le_bytes(
                receipt[index_offset..index_offset + size_of::<u32>()]
                    .try_into()
                    .unwrap(),
            );
            let value = f32::from_le_bytes(
                receipt[value_offset..value_offset + size_of::<f32>()]
                    .try_into()
                    .unwrap(),
            );
            ensure!(
                *token < VOCAB && !value.is_nan(),
                "GLM compact policy argmax row {row} returned invalid token or NaN"
            );
        }
        Ok(())
    }

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
        let allocation = GLM53_EXL3_MAX_WIDE_ROWS as usize * VOCAB as usize * 2;
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
