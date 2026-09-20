// SPDX-License-Identifier: AGPL-3.0-only
//! Separate diagnostic adapter; no serving selector or arithmetic substitution.
use super::batchm_contract::{BatchMKind, BatchMLaunch, BatchMPlan, validate_kernel_rows};
use super::gemv_contract::{HIDDEN, OUTPUTS};
use anyhow::{Result, ensure};
use spark_model::{layers::ops, weight_map::DenseWeight};
use spark_runtime::{
    gpu::{DevicePtr, GpuBackend, KernelHandle},
    kernel_args::KernelLaunch,
};

pub struct BatchMKernels {
    single: KernelHandle,
    pair: KernelHandle,
    wide: KernelHandle,
}
impl BatchMKernels {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        let single = gpu.kernel("gemv", "dense_gemv_bf16")?;
        let pair = gpu.kernel("dense_gemv_bf16_batch2", "dense_gemv_bf16_batch2")?;
        let wide = gpu.kernel("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm")?;
        ensure!(
            [single.0, pair.0, wide.0].iter().all(|h| *h != 0),
            "batchM null kernel handle"
        );
        Ok(Self { single, pair, wide })
    }
    pub fn handles(&self) -> [u64; 3] {
        [self.single.0, self.pair.0, self.wide.0]
    }
    /// Submit only: caller retains all owners and drains this same stream.
    pub fn launch(&self, gpu: &dyn GpuBackend, launch: BatchMLaunch, stream: u64) -> Result<()> {
        validate_kernel_rows(launch.rows)?;
        // Re-admit even an independently constructed descriptor before effects.
        let _ = BatchMPlan::new(
            launch.rows,
            0,
            launch.rows,
            launch.input,
            launch.weight,
            launch.output,
            self.handles(),
        )?;
        let weight = DenseWeight {
            weight: DevicePtr(launch.weight.ptr),
        };
        match launch.kind {
            BatchMKind::Wide => KernelLaunch::new(gpu, self.wide)
                .grid([OUTPUTS.div_ceil(4), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(DevicePtr(launch.input.ptr))
                .arg_ptr(DevicePtr(launch.weight.ptr))
                .arg_ptr(DevicePtr(launch.output.ptr))
                .arg_u32(launch.rows)
                .arg_u32(OUTPUTS)
                .arg_u32(HIDDEN)
                .arg_u32(HIDDEN)
                .arg_u32(OUTPUTS)
                .launch(stream),
            BatchMKind::Pair => {
                ensure!(launch.rows == 2, "pair descriptor row mismatch");
                ops::dense_gemv_batch2(
                    gpu,
                    self.pair,
                    DevicePtr(launch.input.ptr),
                    &weight,
                    DevicePtr(launch.output.ptr),
                    OUTPUTS,
                    HIDDEN,
                    OUTPUTS,
                    stream,
                )
            }
            BatchMKind::Single => {
                ensure!(launch.rows == 1, "single descriptor row mismatch");
                ops::dense_gemv(
                    gpu,
                    self.single,
                    DevicePtr(launch.input.ptr),
                    &weight,
                    DevicePtr(launch.output.ptr),
                    OUTPUTS,
                    HIDDEN,
                    stream,
                )
            }
        }
    }
}
