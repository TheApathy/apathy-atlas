// SPDX-License-Identifier: AGPL-3.0-only

use super::{Qwen4HyperConnection, prefill_exact_plan::Layout};
use anyhow::{Result, ensure};
use spark_runtime::{buffers::BufferArena, gpu::DevicePtr};

impl Qwen4HyperConnection {
    /// Check the complete tiled path before its first normalization/state write.
    pub(crate) fn validate_prefill_exact(
        &self,
        hyper: DevicePtr,
        residual: DevicePtr,
        rows: usize,
        buffers: &BufferArena,
        eps: f32,
    ) -> Result<()> {
        let sizes = buffers.sizes();
        let p = Layout::new(
            rows,
            self.hidden_size,
            self.hc_count,
            self.rank,
            buffers.max_batch_tokens(),
            [
                sizes.ssm_ba,
                sizes.qkv_output,
                sizes.norm_output,
                sizes.hidden_states,
                sizes.residual,
            ],
        )
        .map_err(anyhow::Error::msg)?;
        ensure!(
            eps.is_finite() && eps > 0.0,
            "invalid exact HC normalization epsilon"
        );
        ensure!(
            !self.norm.weight.is_null(),
            "missing exact HC normalization weight"
        );
        for weight in [&self.down, &self.up]
            .into_iter()
            .chain(self.inject.as_ref())
        {
            ensure!(
                !weight.weight.is_null()
                    && !weight.weight_scale.is_null()
                    && weight.weight_scale_2.is_finite()
                    && weight.weight_scale_2 > 0.0,
                "invalid exact HC projection weight"
            );
        }
        for kernel in [
            self.group_norm_k,
            self.silu_div_k,
            self.mix_k,
            self.save_inject_k,
            self.inject_saved_k,
            self.stage_mixed_k,
            self.pack_mixed_k,
            self.w4a16_gemv_k,
            self.w4a16_gemv_exact_m4_k,
            self.w4a16_gemv_exact_m8_k,
            self.w4a16_gemv_exact_m17_k,
            self.w4a16_gemv_exact_m32_k,
        ] {
            ensure!(kernel.0 != 0, "missing exact HC kernel");
        }
        let ranges = [
            (hyper, p.residual_bytes),
            (residual, p.residual_bytes),
            (buffers.ssm_ba(), sizes.ssm_ba),
            (buffers.qkv_output(), p.tile * p.row_bytes),
            (buffers.norm_output(), p.input_bytes),
        ];
        for (index, &(ptr, bytes)) in ranges.iter().enumerate() {
            ensure!(
                !ptr.is_null() && ptr.0.is_multiple_of(2),
                "invalid exact HC pointer"
            );
            let end = ptr
                .0
                .checked_add(bytes as u64)
                .ok_or_else(|| anyhow::anyhow!("exact HC device extent overflow"))?;
            for &(other, count) in &ranges[..index] {
                let other_end = other
                    .0
                    .checked_add(count as u64)
                    .ok_or_else(|| anyhow::anyhow!("exact HC device extent overflow"))?;
                ensure!(
                    end <= other.0 || other_end <= ptr.0,
                    "exact HC buffers overlap"
                );
            }
        }
        Ok(())
    }
}
