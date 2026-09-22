// SPDX-License-Identifier: AGPL-3.0-only

//! Exact full-tile packing for Flash-Next attention prefill.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::Qwen4HyperConnection;

const ROWS: usize = 16;

fn span(ptr: DevicePtr, bytes: usize) -> Result<(u64, u64)> {
    ensure!(
        ptr != DevicePtr::NULL && ptr.0.is_multiple_of(2),
        "HC16 pointer is invalid"
    );
    let end = ptr
        .0
        .checked_add(u64::try_from(bytes)?)
        .ok_or_else(|| anyhow::anyhow!("HC16 pointer extent overflow"))?;
    Ok((ptr.0, end))
}

fn disjoint(left: (u64, u64), right: (u64, u64)) -> bool {
    left.1 <= right.0 || right.1 <= left.0
}

impl Qwen4HyperConnection {
    pub(crate) fn preflight_saved_tile(
        &self,
        hidden: DevicePtr,
        packed: DevicePtr,
        residual: DevicePtr,
    ) -> Result<()> {
        ensure!(
            self.hidden_size == 2560 && self.hc_count == 4,
            "HC16 requires canonical Flash-Next geometry"
        );
        let hidden = span(hidden, ROWS * self.residual_width() * 2)?;
        let packed = span(packed, ROWS * self.hidden_size * 2)?;
        let residual = span(residual, ROWS * self.residual_width() * 2)?;
        ensure!(
            disjoint(hidden, packed) && disjoint(hidden, residual) && disjoint(packed, residual),
            "HC16 spans must be disjoint"
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn pack_saved_tile_or_rows(
        &self,
        selected: bool,
        residual: DevicePtr,
        packed: DevicePtr,
        row_bytes: usize,
        core_bytes: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        if !selected {
            for row in 0..ROWS {
                gpu.copy_d2d_async(
                    residual.offset(row * row_bytes),
                    packed.offset(row * core_bytes),
                    core_bytes,
                    stream,
                )?;
            }
            return Ok(());
        }
        KernelLaunch::new(gpu, self.pack_mixed_k)
            .grid([div_ceil((ROWS * self.hidden_size) as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(residual)
            .arg_ptr(packed)
            .arg_u32(ROWS as u32)
            .arg_u32(self.hidden_size as u32)
            .arg_u32(self.hc_count as u32)
            .launch(stream)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn inject_saved_tile_or_rows(
        &self,
        selected: bool,
        hidden: DevicePtr,
        packed: DevicePtr,
        residual: DevicePtr,
        row_bytes: usize,
        core_bytes: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        if selected {
            return self.inject_saved_batched(hidden, packed, residual, ROWS, gpu, stream);
        }
        for row in 0..ROWS {
            let residual_row = residual.offset(row * row_bytes);
            self.inject_decode(
                hidden.offset(row * row_bytes),
                packed.offset(row * core_bytes),
                self.saved_inject(residual_row),
                gpu,
                stream,
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{DevicePtr, disjoint, span};

    #[test]
    fn checked_spans_reject_null_misalignment_overflow_and_alias() {
        assert!(span(DevicePtr::NULL, 2).is_err());
        assert!(span(DevicePtr(3), 2).is_err());
        assert!(span(DevicePtr(u64::MAX - 1), 4).is_err());
        let a = span(DevicePtr(2), 16).unwrap();
        let b = span(DevicePtr(18), 2).unwrap();
        let overlap = span(DevicePtr(16), 4).unwrap();
        assert!(disjoint(a, b));
        assert!(!disjoint(a, overlap));
    }
}
