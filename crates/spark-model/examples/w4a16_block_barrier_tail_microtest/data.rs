// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::cases::Case;

const REDZONE_WORDS: usize = 16;
pub(crate) const CANARY: u16 = 0xa55a;

pub(crate) fn u16s_to_le(values: &[u16]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

pub(crate) fn upload(gpu: &dyn GpuBackend, bytes: &[u8]) -> Result<DevicePtr> {
    let ptr = gpu.alloc(bytes.len().max(1))?;
    gpu.copy_h2d(bytes, ptr)?;
    Ok(ptr)
}

pub(crate) struct GuardedOutput {
    allocation: DevicePtr,
    payload_words: usize,
}

impl GuardedOutput {
    pub(crate) fn new(gpu: &dyn GpuBackend, payload_words: usize) -> Result<Self> {
        let words = vec![CANARY; REDZONE_WORDS * 2 + payload_words];
        Ok(Self {
            allocation: upload(gpu, &u16s_to_le(&words))?,
            payload_words,
        })
    }

    pub(crate) fn payload_ptr(&self) -> DevicePtr {
        self.allocation.offset(REDZONE_WORDS * size_of::<u16>())
    }

    pub(crate) fn free(&self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.free(self.allocation)
    }

    pub(crate) fn read_and_check(
        &self,
        gpu: &dyn GpuBackend,
        stream: u64,
        label: &str,
        written: &[bool],
    ) -> Result<Vec<u16>> {
        ensure!(written.len() == self.payload_words, "bad written mask");
        let total_words = REDZONE_WORDS * 2 + self.payload_words;
        let mut raw = vec![0u8; total_words * size_of::<u16>()];
        gpu.copy_d2h_on_stream(self.allocation, &mut raw, stream)?;
        let words: Vec<u16> = raw
            .chunks_exact(2)
            .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
            .collect();

        ensure!(
            words[..REDZONE_WORDS].iter().all(|&word| word == CANARY),
            "{label}: leading redzone modified"
        );
        ensure!(
            words[REDZONE_WORDS + self.payload_words..]
                .iter()
                .all(|&word| word == CANARY),
            "{label}: trailing redzone modified"
        );
        let payload = &words[REDZONE_WORDS..REDZONE_WORDS + self.payload_words];
        if let Some(index) = payload
            .iter()
            .zip(written)
            .position(|(&word, &is_written)| !is_written && word != CANARY)
        {
            bail!(
                "{label}: output hole {index} changed from canary to 0x{:04x}",
                payload[index]
            );
        }
        Ok(payload.to_vec())
    }
}

pub(crate) fn written_mask(case: Case, launch_n: usize) -> Vec<bool> {
    let mut mask = vec![false; case.output_words()];
    for row in 0..case.rows {
        for logical_n in 0..launch_n {
            mask[case.output_index(launch_n, row, logical_n)] = true;
        }
    }
    mask
}

pub(crate) fn gather(case: Case, launch_n: usize, logical_n: usize, output: &[u16]) -> Vec<u16> {
    let mut gathered = Vec::with_capacity(case.rows * logical_n);
    for row in 0..case.rows {
        for column in 0..logical_n {
            gathered.push(output[case.output_index(launch_n, row, column)]);
        }
    }
    gathered
}
