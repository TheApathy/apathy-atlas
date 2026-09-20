// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::contract::{CANARY, REDZONE};

pub(super) struct Guarded {
    allocation: DevicePtr,
    payload_len: usize,
    image: Vec<u8>,
}

impl Guarded {
    pub(super) fn input(gpu: &dyn GpuBackend, payload: &[u8]) -> Result<Self> {
        let mut image = vec![CANARY; REDZONE + payload.len() + REDZONE];
        image[REDZONE..REDZONE + payload.len()].copy_from_slice(payload);
        let allocation = gpu.alloc(image.len())?;
        gpu.copy_h2d(&image, allocation)?;
        Ok(Self {
            allocation,
            payload_len: payload.len(),
            image,
        })
    }

    pub(super) fn output(gpu: &dyn GpuBackend, payload_len: usize, fill: u8) -> Result<Self> {
        Self::input(gpu, &vec![fill; payload_len])
    }

    pub(super) fn payload_ptr(&self) -> DevicePtr {
        self.allocation.offset(REDZONE)
    }

    pub(super) fn reset(&self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.copy_h2d(&self.image, self.allocation)
    }

    fn read(&self, gpu: &dyn GpuBackend) -> Result<Vec<u8>> {
        let mut bytes = vec![0u8; self.image.len()];
        gpu.copy_d2h(self.allocation, &mut bytes)?;
        Ok(bytes)
    }

    pub(super) fn verify_immutable(&self, gpu: &dyn GpuBackend, label: &str) -> Result<()> {
        ensure!(
            self.read(gpu)? == self.image,
            "{label}: input bytes changed"
        );
        Ok(())
    }

    pub(super) fn output_payload(&self, gpu: &dyn GpuBackend, label: &str) -> Result<Vec<u8>> {
        let bytes = self.read(gpu)?;
        ensure!(
            bytes[..REDZONE].iter().all(|&byte| byte == CANARY),
            "{label}: leading 4-KiB redzone changed"
        );
        let suffix = REDZONE + self.payload_len;
        ensure!(
            bytes[suffix..].iter().all(|&byte| byte == CANARY),
            "{label}: trailing 4-KiB redzone changed"
        );
        Ok(bytes[REDZONE..suffix].to_vec())
    }

    pub(super) fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.free(self.allocation)
    }
}
