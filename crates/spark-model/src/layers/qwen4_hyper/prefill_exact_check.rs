// SPDX-License-Identifier: AGPL-3.0-only

//! Explicit same-input numerical diagnostic. Never use checked runs for timing.

use super::*;

fn finite_bf16(bytes: &[u8]) -> Result<()> {
    anyhow::ensure!(
        bytes.len().is_multiple_of(2),
        "HC check has partial BF16 element"
    );
    for value in bytes.chunks_exact(2) {
        let bits = u16::from_ne_bytes([value[0], value[1]]);
        anyhow::ensure!(bits & 0x7f80 != 0x7f80, "HC check produced nonfinite BF16");
    }
    Ok(())
}

impl Qwen4HyperConnection {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn check_prefill_exact(
        &self,
        hyper: DevicePtr,
        residual: DevicePtr,
        rows: usize,
        buffers: &BufferArena,
        gpu: &dyn GpuBackend,
        eps: f32,
        stream: u64,
    ) -> Result<()> {
        if !crate::layers::qwen4_prefill_moe::hyper_check_selected()? {
            return Ok(());
        }
        // Validate before readback; the caller has already performed the same
        // checks before preparing any rows. This diagnostic covers F8 only.
        anyhow::ensure!(
            rows <= 2048 && self.inject.is_some(),
            "unsupported HC check geometry"
        );
        self.validate_prefill_exact(hyper, residual, rows, buffers, eps)?;
        let row_bytes = self.residual_width() * 2;
        let core_bytes = self.hidden_size * 2;
        let tail = row_bytes - self.hc_count * 2;
        let mut candidate = vec![0u8; rows * core_bytes];
        let mut saved_residual = vec![0u8; rows * row_bytes];
        gpu.copy_d2h_on_stream(buffers.norm_output(), &mut candidate, stream)?;
        gpu.copy_d2h_on_stream(residual, &mut saved_residual, stream)?;
        finite_bf16(&candidate)?;
        let mut reference = vec![0u8; core_bytes];
        let mut injection = vec![0u8; self.hc_count * 2];
        for row in 0..rows {
            let residual_row = residual.offset(row * row_bytes);
            let (mixed, scales) = self.prepare_decode(
                hyper.offset(row * row_bytes),
                residual_row,
                buffers,
                gpu,
                eps,
                stream,
            )?;
            gpu.copy_d2h_on_stream(mixed, &mut reference, stream)?;
            gpu.copy_d2h_on_stream(
                scales.ok_or_else(|| anyhow::anyhow!("HC check lost injection"))?,
                &mut injection,
                stream,
            )?;
            finite_bf16(&reference)?;
            finite_bf16(&injection)?;
            anyhow::ensure!(
                reference == candidate[row * core_bytes..(row + 1) * core_bytes],
                "HC exact check: mixed row {row} differs from serial decode"
            );
            let at = row * row_bytes + tail;
            anyhow::ensure!(
                injection == saved_residual[at..at + injection.len()],
                "HC exact check: saved injection row {row} differs from serial decode"
            );
        }
        // Serial replay reused both buffers. Restore the fully staged candidate
        // (including first-H row staging consumed by serial-core callers).
        gpu.copy_h2d_group_on_stream(
            &[
                spark_runtime::gpu::HostToDeviceCopy::new(&candidate, buffers.norm_output()),
                spark_runtime::gpu::HostToDeviceCopy::new(&saved_residual, residual),
            ],
            stream,
        )?;
        tracing::info!(
            "QWEN4_HC_EXACT_CHECK rows={rows} mixed=byte_exact saved_inject=byte_exact timing_eligible=false"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::finite_bf16;
    #[test]
    fn rejects_all_nonfinite_and_partial_bf16() {
        for value in [0u16, 0x8000, 0x3f80, 0xbf80, 0x7f7f] {
            assert!(finite_bf16(&value.to_ne_bytes()).is_ok());
        }
        for value in [0x7f80u16, 0xff80, 0x7fc0, 0xffc1] {
            assert!(finite_bf16(&value.to_ne_bytes()).is_err());
        }
        assert!(finite_bf16(&[0]).is_err());
    }
}
