// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::super::types::TransformerModel;
use crate::model::k1_stage_diag;

impl TransformerModel {
    pub(super) fn capture_k1_stage(
        &self,
        stage: &str,
        ptr: DevicePtr,
        rows: usize,
        row_bytes: usize,
        stream: u64,
    ) -> Result<()> {
        if !k1_stage_diag::serial_active() && !k1_stage_diag::batch_active() {
            return Ok(());
        }
        if !k1_stage_diag::capture_stage(stage) {
            return Ok(());
        }
        self.gpu.synchronize(stream)?;
        let mut bytes = vec![0u8; rows * row_bytes];
        self.gpu.copy_d2h(ptr, &mut bytes)?;
        if k1_stage_diag::serial_active() {
            if rows != 1 {
                anyhow::bail!("DFLASH_K1_STAGE_DIAG serial capture has {rows} rows");
            }
            k1_stage_diag::record_serial(stage, bytes)
        } else {
            k1_stage_diag::record_batch(stage, &bytes, row_bytes)
        }
    }
}
