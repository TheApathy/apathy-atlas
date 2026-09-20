// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result, ensure};
use half::bf16;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::REDZONE;

pub(super) struct Guarded {
    pub(super) base: DevicePtr,
    pub(super) len: usize,
    pub(super) salt: u8,
    pub(super) original: Option<Vec<u8>>,
}

impl Guarded {
    fn new(gpu: &dyn GpuBackend, payload: Vec<u8>, salt: u8, immutable: bool) -> Result<Self> {
        let mut image = vec![0xa5 ^ salt; REDZONE + payload.len() + REDZONE];
        image[REDZONE..REDZONE + payload.len()].copy_from_slice(&payload);
        image[REDZONE + payload.len()..].fill(0x5a ^ salt);
        let base = gpu.alloc(image.len())?;
        gpu.copy_h2d(&image, base)?;
        Ok(Self {
            base,
            len: payload.len(),
            salt,
            original: immutable.then_some(payload),
        })
    }

    pub(super) fn input(gpu: &dyn GpuBackend, payload: Vec<u8>, salt: u8) -> Result<Self> {
        Self::new(gpu, payload, salt, true)
    }

    pub(super) fn output(gpu: &dyn GpuBackend, len: usize, salt: u8) -> Result<Self> {
        Self::new(gpu, vec![0x7d; len], salt, false)
    }

    pub(super) fn ptr(&self) -> DevicePtr {
        self.base.offset(REDZONE)
    }

    pub(super) fn span(&self) -> Result<(usize, usize)> {
        let start = usize::try_from(self.base.0)?;
        Ok((
            start,
            start
                .checked_add(REDZONE * 2 + self.len)
                .context("device span overflow")?,
        ))
    }

    pub(super) fn read(&self, gpu: &dyn GpuBackend, label: &str) -> Result<Vec<u8>> {
        let mut image = vec![0; REDZONE + self.len + REDZONE];
        gpu.copy_d2h(self.base, &mut image)?;
        ensure!(
            image[..REDZONE]
                .iter()
                .all(|value| *value == (0xa5 ^ self.salt)),
            "{label} prefix redzone changed"
        );
        ensure!(
            image[REDZONE + self.len..]
                .iter()
                .all(|value| *value == (0x5a ^ self.salt)),
            "{label} suffix redzone changed"
        );
        Ok(image[REDZONE..REDZONE + self.len].to_vec())
    }

    pub(super) fn immutable(&self, gpu: &dyn GpuBackend, label: &str) -> Result<()> {
        ensure!(
            self.original.as_ref().context("buffer is mutable")? == &self.read(gpu, label)?,
            "{label} changed"
        );
        Ok(())
    }

    pub(super) fn free(&self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.free(self.base)
    }
}

pub(super) fn input(elements: usize, up: bool) -> Vec<u8> {
    const G: [f32; 8] = [-6.0, -2.9, -0.4, -0.0, 0.1, 0.9, 3.0, 6.0];
    const U: [f32; 8] = [0.5, -1.5, 2.0, -3.0, 0.0, -0.25, 4.0, -6.0];
    let values = if up { &U } else { &G };
    let mut out = Vec::with_capacity(elements * 2);
    for index in 0..elements {
        out.extend_from_slice(
            &bf16::from_f32(values[(index * 13 + index / 127) % 8])
                .to_bits()
                .to_le_bytes(),
        );
    }
    out
}
