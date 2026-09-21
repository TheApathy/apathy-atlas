// SPDX-License-Identifier: AGPL-3.0-only

//! Checked, allocation-free row layout for serial-exact native FP8 verification.

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;

pub(super) struct NativeSharedVerifyPlan;

impl NativeSharedVerifyPlan {
    /// Inputs retain the caller's packed `[rows, hidden]` contract. Output
    /// capacities are the actual arena allocations (gate, up, down), not
    /// assumed multiples of max_batch_tokens: logits may be sized separately.
    pub(super) fn new(
        rows: u32,
        max_rows: usize,
        hidden: u32,
        intermediate: u32,
        pointers: [DevicePtr; 4],
        output_capacities: [usize; 3],
    ) -> Result<Self> {
        ensure!(
            rows >= 2
                && rows <= 8
                && rows as usize <= max_rows
                && hidden == 4096
                && intermediate == 2048,
            "native FP8 shared verify requires bounded actual-Vision packed rows"
        );
        let strides = [8192_u64, 4096, 4096, 8192];
        let mut ends = [0; 4];
        for i in 0..4 {
            let bytes = u64::from(rows) * strides[i];
            ensure!(
                pointers[i].0 != 0 && pointers[i].0 % 16 == 0,
                "native FP8 shared verify pointer is null or not vector-load aligned"
            );
            ends[i] = pointers[i].0.checked_add(bytes).ok_or_else(|| {
                anyhow::anyhow!("native FP8 shared verify pointer extent overflow")
            })?;
            if i > 0 {
                ensure!(
                    bytes <= output_capacities[i - 1] as u64,
                    "native FP8 shared verify output exceeds its arena allocation"
                );
            }
        }
        for i in 0..4 {
            for j in i + 1..4 {
                ensure!(
                    ends[i] <= pointers[j].0 || ends[j] <= pointers[i].0,
                    "native FP8 shared verify activation regions overlap"
                );
            }
        }
        Ok(Self)
    }
}
