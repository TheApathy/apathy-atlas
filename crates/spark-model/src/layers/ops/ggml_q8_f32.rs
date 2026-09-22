// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::GgmlIqBuffer;

const Q8_BLOCK_VALUES: usize = 32;
const Q8_BLOCK_BYTES: usize = 34;
const THREADS: u32 = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GgmlQ8F32Plan {
    pub rows: u32,
    pub columns: u32,
    pub source_bytes: usize,
    pub destination_bytes: usize,
}

impl GgmlQ8F32Plan {
    pub fn new(rows: u32, columns: u32) -> Result<Self> {
        if rows == 0 || columns == 0 {
            bail!("GGML Q8_0 F32 materialization requires nonzero rows and columns");
        }
        if rows > i32::MAX as u32 || columns > i32::MAX as u32 {
            bail!("GGML Q8_0 F32 materialization exceeds CUDA grid or loop bounds");
        }
        let columns_usize = usize::try_from(columns)?;
        if !columns_usize.is_multiple_of(Q8_BLOCK_VALUES) {
            bail!("GGML Q8_0 F32 columns must be divisible by 32");
        }
        let elements = usize::try_from(rows)?
            .checked_mul(columns_usize)
            .context("GGML Q8_0 F32 element count overflow")?;
        let source_bytes = elements
            .checked_div(Q8_BLOCK_VALUES)
            .and_then(|blocks| blocks.checked_mul(Q8_BLOCK_BYTES))
            .context("GGML Q8_0 F32 source byte count overflow")?;
        let destination_bytes = elements
            .checked_mul(4)
            .context("GGML Q8_0 F32 destination byte count overflow")?;
        Ok(Self {
            rows,
            columns,
            source_bytes,
            destination_bytes,
        })
    }
}

pub struct GgmlQ8F32Kernel {
    dequantize: KernelHandle,
}

impl GgmlQ8F32Kernel {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            dequantize: gpu.kernel("ggml_q8_f32", "atlas_q8_0_to_f32")?,
        })
    }

    pub fn launch(
        &self,
        gpu: &dyn GpuBackend,
        plan: GgmlQ8F32Plan,
        source: GgmlIqBuffer,
        destination: GgmlIqBuffer,
        stream: u64,
    ) -> Result<()> {
        if source.bytes != plan.source_bytes || destination.bytes != plan.destination_bytes {
            bail!("GGML Q8_0 F32 device buffer extent mismatch");
        }
        if source.ptr == DevicePtr::NULL || destination.ptr == DevicePtr::NULL {
            bail!("GGML Q8_0 F32 requires nonnull device buffers");
        }
        let source_end = source
            .ptr
            .0
            .checked_add(u64::try_from(source.bytes)?)
            .context("GGML Q8_0 F32 source address overflow")?;
        let destination_end = destination
            .ptr
            .0
            .checked_add(u64::try_from(destination.bytes)?)
            .context("GGML Q8_0 F32 destination address overflow")?;
        if source.ptr.0 < destination_end && destination.ptr.0 < source_end {
            bail!("GGML Q8_0 F32 source and destination overlap");
        }
        KernelLaunch::new(gpu, self.dequantize)
            .grid([plan.rows, 1, 1])
            .block([THREADS, 1, 1])
            .arg_ptr(source.ptr)
            .arg_ptr(destination.ptr)
            .arg_u32(plan.rows)
            .arg_u32(plan.columns)
            .launch(stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::mock::MockGpuBackend;

    #[test]
    fn exact_mhc_geometry_and_invalid_shapes_fail_closed() {
        let plan = GgmlQ8F32Plan::new(24, 16_384).unwrap();
        assert_eq!(plan.source_bytes, 24 * (16_384 / 32) * 34);
        assert_eq!(plan.destination_bytes, 24 * 16_384 * 4);
        assert_eq!(plan.destination_bytes * 90, 141_557_760);
        assert!(GgmlQ8F32Plan::new(0, 16_384).is_err());
        assert!(GgmlQ8F32Plan::new(24, 16_383).is_err());
        assert!(GgmlQ8F32Plan::new(u32::MAX, 32).is_err());
    }

    #[test]
    fn launch_checks_extents_and_aliases_before_enqueue() {
        let gpu = MockGpuBackend::new();
        let kernel = GgmlQ8F32Kernel::load(&gpu).unwrap();
        let plan = GgmlQ8F32Plan::new(2, 32).unwrap();
        let buffer = |ptr, bytes| GgmlIqBuffer {
            ptr: DevicePtr(ptr),
            bytes,
        };
        assert!(
            kernel
                .launch(
                    &gpu,
                    plan,
                    buffer(0x1000, plan.source_bytes - 1),
                    buffer(0x4000, plan.destination_bytes),
                    0,
                )
                .is_err()
        );
        assert!(
            kernel
                .launch(
                    &gpu,
                    plan,
                    buffer(0x1000, plan.source_bytes),
                    buffer(0x1040, plan.destination_bytes),
                    0,
                )
                .is_err()
        );
        assert_eq!(gpu.launch_count(), 0);
        kernel
            .launch(
                &gpu,
                plan,
                buffer(0x1000, plan.source_bytes),
                buffer(0x4000, plan.destination_bytes),
                0,
            )
            .unwrap();
        assert_eq!(gpu.launch_count(), 1);
    }
}
