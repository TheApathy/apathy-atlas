// SPDX-License-Identifier: AGPL-3.0-only
//! Separate GEMV arithmetic family; this helper does not grant cache authority.
use super::gemv_contract::{GemvPlan, HIDDEN, OUTPUTS, validate_handles};
use anyhow::Result;
use spark_model::{layers::ops, weight_map::DenseWeight};
use spark_runtime::{
    gpu::{DevicePtr, GpuBackend, KernelHandle},
    kernel_args::KernelLaunch,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GemvMode {
    Sequential,
    Gather,
    Batch2,
}
impl GemvMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Sequential => "sequential-gemv",
            Self::Gather => "gather-gemv",
            Self::Batch2 => "batch2-with-m1-gemv",
        }
    }
}

pub struct GemvKernels {
    single: KernelHandle,
    gather: KernelHandle,
    pair: KernelHandle,
}
impl GemvKernels {
    /// Resolve once, before metadata submission, projection and timing.
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        let single = gpu.kernel("gemv", "dense_gemv_bf16")?;
        let gather = gpu.kernel("lora_bgmv", "lora_bgmv_shrink")?;
        let pair = gpu.kernel("dense_gemv_bf16_batch2", "dense_gemv_bf16_batch2")?;
        validate_handles([single.0, gather.0, pair.0])?;
        Ok(Self {
            single,
            gather,
            pair,
        })
    }

    /// Submit only; the caller retains all plan owners through same-stream drain.
    /// Metadata must already be initialized and fenced by the session. Existing
    /// typed launch-builder host allocations are included in operator timing;
    /// no device buffers or metadata are allocated here.
    pub fn launch(
        &self,
        gpu: &dyn GpuBackend,
        mode: GemvMode,
        plan: &GemvPlan,
        stream: u64,
    ) -> Result<()> {
        let input = plan.input();
        let output = plan.output();
        let weight = DenseWeight {
            weight: DevicePtr(plan.weight().ptr),
        };
        match mode {
            GemvMode::Gather => {
                let slots = plan.slots();
                let table = plan.table();
                // The gather table contains one real weight pointer, all slots
                // are zero. OUTPUTS is solely the kernel's output dimension.
                KernelLaunch::new(gpu, self.gather)
                    .grid([OUTPUTS.div_ceil(4), plan.rows(), 1])
                    .block([256, 1, 1])
                    .arg_ptr(DevicePtr(input.ptr))
                    .arg_ptr(DevicePtr(slots.ptr))
                    .arg_ptr(DevicePtr(table.ptr))
                    .arg_ptr(DevicePtr(output.ptr))
                    .arg_u32(plan.rows())
                    .arg_u32(OUTPUTS)
                    .arg_u32(HIDDEN)
                    .arg_u32(HIDDEN)
                    .launch(stream)
            }
            GemvMode::Batch2 if plan.rows() == 2 => ops::dense_gemv_batch2(
                gpu,
                self.pair,
                DevicePtr(input.ptr),
                &weight,
                DevicePtr(output.ptr),
                OUTPUTS,
                HIDDEN,
                OUTPUTS,
                stream,
            ),
            GemvMode::Sequential | GemvMode::Batch2 => {
                // Plan admission bounds both selected rows, so these fixed
                // per-row offsets remain inside the checked immutable spans.
                for row in 0..plan.rows() {
                    ops::dense_gemv(
                        gpu,
                        self.single,
                        DevicePtr(input.ptr + u64::from(row) * u64::from(HIDDEN) * 2),
                        &weight,
                        DevicePtr(output.ptr + u64::from(row) * u64::from(OUTPUTS) * 2),
                        OUTPUTS,
                        HIDDEN,
                        stream,
                    )?;
                }
                Ok(())
            }
        }
    }
}
