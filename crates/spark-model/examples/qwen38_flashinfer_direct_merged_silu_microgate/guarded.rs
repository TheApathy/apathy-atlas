// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result, ensure};
use half::bf16;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::REDZONE;
use super::owner::AllocationOwner;

pub(super) struct Guarded {
    pub(super) base: DevicePtr,
    pub(super) len: usize,
    pub(super) salt: u8,
    pub(super) original: Option<Vec<u8>>,
}

impl Guarded {
    fn new(
        owner: &mut AllocationOwner,
        gpu: &dyn GpuBackend,
        payload: Vec<u8>,
        salt: u8,
        immutable: bool,
    ) -> Result<Self> {
        let image_len = REDZONE
            .checked_add(payload.len())
            .and_then(|len| len.checked_add(REDZONE))
            .context("guarded allocation extent overflow")?;
        let mut image = Vec::new();
        image.try_reserve_exact(image_len)?;
        image.resize(image_len, 0xa5 ^ salt);
        image[REDZONE..REDZONE + payload.len()].copy_from_slice(&payload);
        image[REDZONE + payload.len()..].fill(0x5a ^ salt);
        // Ownership is registered before the fallible copy. The outer case
        // teardown therefore retains cleanup authority on copy failure.
        let base = owner.allocate(gpu, image.len())?;
        gpu.copy_h2d(&image, base)?;
        Ok(Self {
            base,
            len: payload.len(),
            salt,
            original: immutable.then_some(payload),
        })
    }

    pub(super) fn input(
        owner: &mut AllocationOwner,
        gpu: &dyn GpuBackend,
        payload: Vec<u8>,
        salt: u8,
    ) -> Result<Self> {
        Self::new(owner, gpu, payload, salt, true)
    }

    pub(super) fn output(
        owner: &mut AllocationOwner,
        gpu: &dyn GpuBackend,
        len: usize,
        salt: u8,
    ) -> Result<Self> {
        Self::new(owner, gpu, output_sentinel(len, salt), salt, false)
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

    pub(super) fn replace_payload(
        &mut self,
        gpu: &dyn GpuBackend,
        payload: Vec<u8>,
        immutable: bool,
    ) -> Result<()> {
        ensure!(payload.len() == self.len, "replacement extent changed");
        gpu.copy_h2d(&payload, self.ptr())?;
        self.original = immutable.then_some(payload);
        Ok(())
    }
}

pub(super) fn output_sentinel(len: usize, salt: u8) -> Vec<u8> {
    (0..len)
        .map(|index| {
            let mut value = (index as u64).wrapping_add(0x9e37_79b9_7f4a_7c15);
            value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            ((value ^ (value >> 31)) as u8) ^ salt
        })
        .collect()
}

pub(super) fn disjoint_sentinel(expected: &[u8], salt: u8) -> Vec<u8> {
    let mut sentinel = output_sentinel(expected.len(), salt);
    for (actual, expected) in sentinel.iter_mut().zip(expected) {
        if actual == expected {
            *actual ^= 0xff;
        }
    }
    debug_assert!(
        sentinel
            .iter()
            .zip(expected)
            .all(|(left, right)| left != right)
    );
    sentinel
}

pub(super) fn merged_input(rows: u32, cols: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(rows as usize * cols as usize * 4);
    for row in 0..rows as usize {
        for side in 0..2usize {
            for col in 0..cols as usize {
                let coordinate = ((row as u64) << 33) ^ ((col as u64) << 1) ^ side as u64;
                let mut mixed = coordinate.wrapping_add(0x6a09_e667_f3bc_c909);
                mixed = (mixed ^ (mixed >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
                mixed ^= mixed >> 31;
                let bits = if row == 0 && col < 8 {
                    [
                        0x0000, 0x8000, 0x0001, 0x8001, 0x3f80, 0xbf80, 0x40c0, 0xc0c0,
                    ][col]
                        ^ ((side as u16) << 6)
                } else {
                    let sign = ((mixed >> 63) as u16) << 15;
                    let exponent = (118 + ((mixed >> 7) % 18) as u16) << 7;
                    sign | exponent | (mixed as u16 & 0x7f)
                };
                out.extend_from_slice(&bf16::from_bits(bits).to_bits().to_le_bytes());
            }
        }
    }
    out
}
