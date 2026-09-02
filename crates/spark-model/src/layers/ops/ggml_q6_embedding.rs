// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::{GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::GgmlIqBuffer;

const Q6_BLOCK_VALUES: usize = 256;
const Q6_BLOCK_BYTES: usize = 210;
const THREADS: u32 = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GgmlQ6EmbeddingPlan {
    pub rows: u32,
    pub columns: u32,
    pub source_bytes: usize,
    pub destination_bytes: usize,
}

impl GgmlQ6EmbeddingPlan {
    pub fn new(rows: u32, columns: u32) -> Result<Self> {
        if rows == 0 || columns == 0 {
            bail!("GGML Q6_K embedding requires nonzero rows and columns");
        }
        if rows > i32::MAX as u32 || columns > i32::MAX as u32 {
            bail!("GGML Q6_K embedding exceeds CUDA grid or loop bounds");
        }
        let columns_usize = usize::try_from(columns)?;
        if !columns_usize.is_multiple_of(Q6_BLOCK_VALUES) {
            bail!("GGML Q6_K embedding columns must be divisible by 256");
        }
        let elements = usize::try_from(rows)?
            .checked_mul(columns_usize)
            .context("GGML Q6_K embedding element count overflow")?;
        let source_bytes = elements
            .checked_div(Q6_BLOCK_VALUES)
            .and_then(|blocks| blocks.checked_mul(Q6_BLOCK_BYTES))
            .context("GGML Q6_K embedding source byte count overflow")?;
        let destination_bytes = elements
            .checked_mul(2)
            .context("GGML Q6_K embedding destination byte count overflow")?;
        Ok(Self {
            rows,
            columns,
            source_bytes,
            destination_bytes,
        })
    }
}

pub struct GgmlQ6EmbeddingKernel {
    dequantize: KernelHandle,
}

impl GgmlQ6EmbeddingKernel {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            dequantize: gpu.kernel("ggml_q6_embedding", "atlas_q6_k_embedding_bf16")?,
        })
    }

    pub fn launch(
        &self,
        gpu: &dyn GpuBackend,
        plan: GgmlQ6EmbeddingPlan,
        source: GgmlIqBuffer,
        destination: GgmlIqBuffer,
        stream: u64,
    ) -> Result<()> {
        if source.bytes != plan.source_bytes || destination.bytes != plan.destination_bytes {
            bail!("GGML Q6_K embedding device buffer extent mismatch");
        }
        if source.ptr == spark_runtime::gpu::DevicePtr::NULL
            || destination.ptr == spark_runtime::gpu::DevicePtr::NULL
        {
            bail!("GGML Q6_K embedding requires nonnull device buffers");
        }
        let source_end = source
            .ptr
            .0
            .checked_add(u64::try_from(source.bytes)?)
            .context("GGML Q6_K embedding source address overflow")?;
        let destination_end = destination
            .ptr
            .0
            .checked_add(u64::try_from(destination.bytes)?)
            .context("GGML Q6_K embedding destination address overflow")?;
        if source.ptr.0 < destination_end && destination.ptr.0 < source_end {
            bail!("GGML Q6_K embedding source and destination overlap");
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
    use spark_runtime::gpu::{DevicePtr, mock::MockGpuBackend};

    #[test]
    fn exact_glm_embedding_geometry_and_overflow_fail_closed() {
        let plan = GgmlQ6EmbeddingPlan::new(154_880, 4_096).unwrap();
        assert_eq!(plan.source_bytes, 154_880 * 16 * 210);
        assert_eq!(plan.destination_bytes, 154_880 * 4_096 * 2);
        assert!(GgmlQ6EmbeddingPlan::new(0, 4_096).is_err());
        assert!(GgmlQ6EmbeddingPlan::new(1, 128).is_err());
        assert!(GgmlQ6EmbeddingPlan::new(u32::MAX, u32::MAX - 255).is_err());
    }

    #[test]
    fn launch_rejects_extents_before_enqueue() {
        let gpu = MockGpuBackend::new();
        let kernel = GgmlQ6EmbeddingKernel::load(&gpu).unwrap();
        let plan = GgmlQ6EmbeddingPlan::new(2, 256).unwrap();
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
        assert!(
            kernel
                .launch(
                    &gpu,
                    plan,
                    buffer(0x1000, plan.source_bytes),
                    buffer(0x1100, plan.destination_bytes),
                    0,
                )
                .is_err()
        );
        assert_eq!(gpu.launch_count(), 1);
    }
}
