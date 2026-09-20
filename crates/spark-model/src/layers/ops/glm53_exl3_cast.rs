// SPDX-License-Identifier: AGPL-3.0-only

//! Checked dtype bridge between Atlas BF16 activations and EXL3 F16 GEMM.

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::glm53_exl3::Glm53Exl3Buffer;

const THREADS: u32 = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53Exl3CastPlan {
    pub rows: u32,
    pub width: u32,
    pub elements: u32,
    pub blocks: u32,
    pub bytes: usize,
}

impl Glm53Exl3CastPlan {
    pub fn new(rows: u32, width: u32) -> Result<Self> {
        ensure!(
            rows != 0 && width != 0,
            "GLM EXL3 cast dimensions must be nonzero"
        );
        let elements = rows
            .checked_mul(width)
            .context("GLM EXL3 cast element count overflow")?;
        let blocks = elements
            .checked_add(THREADS - 1)
            .context("GLM EXL3 cast grid overflow")?
            / THREADS;
        let bytes = usize::try_from(elements)?
            .checked_mul(2)
            .context("GLM EXL3 cast byte count overflow")?;
        Ok(Self {
            rows,
            width,
            elements,
            blocks,
            bytes,
        })
    }
}

pub struct Glm53Exl3CastKernels {
    bf16_to_f16: KernelHandle,
    f16_to_bf16: KernelHandle,
}

impl Glm53Exl3CastKernels {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            bf16_to_f16: gpu.kernel("glm53_exl3_cast", "atlas_glm53_exl3_bf16_to_f16")?,
            f16_to_bf16: gpu.kernel("glm53_exl3_cast", "atlas_glm53_exl3_f16_to_bf16")?,
        })
    }

    pub fn bf16_to_f16(
        &self,
        gpu: &dyn GpuBackend,
        plan: Glm53Exl3CastPlan,
        input_bf16: Glm53Exl3Buffer,
        output_f16: Glm53Exl3Buffer,
        stream: u64,
    ) -> Result<()> {
        validate_pair(plan, input_bf16, output_f16)?;
        KernelLaunch::new(gpu, self.bf16_to_f16)
            .grid([plan.blocks, 1, 1])
            .block([THREADS, 1, 1])
            .arg_ptr(input_bf16.ptr)
            .arg_ptr(output_f16.ptr)
            .arg_u32(plan.elements)
            .launch(stream)
    }

    pub fn f16_to_bf16(
        &self,
        gpu: &dyn GpuBackend,
        plan: Glm53Exl3CastPlan,
        input_f16: Glm53Exl3Buffer,
        output_bf16: Glm53Exl3Buffer,
        stream: u64,
    ) -> Result<()> {
        validate_pair(plan, input_f16, output_bf16)?;
        KernelLaunch::new(gpu, self.f16_to_bf16)
            .grid([plan.blocks, 1, 1])
            .block([THREADS, 1, 1])
            .arg_ptr(input_f16.ptr)
            .arg_ptr(output_bf16.ptr)
            .arg_u32(plan.elements)
            .launch(stream)
    }
}

fn validate_pair(
    plan: Glm53Exl3CastPlan,
    input: Glm53Exl3Buffer,
    output: Glm53Exl3Buffer,
) -> Result<()> {
    for (name, buffer) in [("input", input), ("output", output)] {
        if buffer.ptr == DevicePtr::NULL || buffer.bytes != plan.bytes {
            bail!("GLM EXL3 cast {name} is null or has the wrong exact extent");
        }
        buffer
            .ptr
            .0
            .checked_add(u64::try_from(buffer.bytes)?)
            .with_context(|| format!("GLM EXL3 cast {name} address overflow"))?;
    }
    ensure!(input.ptr != output.ptr, "GLM EXL3 cast cannot run in place");
    Ok(())
}

#[cfg(test)]
mod tests {
    use spark_runtime::gpu::mock::MockGpuBackend;

    use super::*;

    fn at(ptr: u64, bytes: usize) -> Glm53Exl3Buffer {
        Glm53Exl3Buffer {
            ptr: DevicePtr(ptr),
            bytes,
        }
    }

    #[test]
    fn exact_plan_and_both_launches_are_checked() {
        let gpu = MockGpuBackend::new();
        let kernels = Glm53Exl3CastKernels::load(&gpu).unwrap();
        let plan = Glm53Exl3CastPlan::new(3, 4096).unwrap();
        assert_eq!(
            (plan.elements, plan.blocks, plan.bytes),
            (12_288, 48, 24_576)
        );
        kernels
            .bf16_to_f16(&gpu, plan, at(1, plan.bytes), at(2, plan.bytes), 0)
            .unwrap();
        kernels
            .f16_to_bf16(&gpu, plan, at(2, plan.bytes), at(3, plan.bytes), 0)
            .unwrap();
        assert_eq!(gpu.launch_count(), 2);
    }

    #[test]
    fn zero_overflow_alias_and_extent_fail_closed() {
        assert!(Glm53Exl3CastPlan::new(0, 4096).is_err());
        assert!(Glm53Exl3CastPlan::new(u32::MAX, 2).is_err());
        let gpu = MockGpuBackend::new();
        let kernels = Glm53Exl3CastKernels::load(&gpu).unwrap();
        let plan = Glm53Exl3CastPlan::new(1, 4096).unwrap();
        assert!(
            kernels
                .bf16_to_f16(&gpu, plan, at(1, plan.bytes), at(1, plan.bytes), 0)
                .is_err()
        );
        assert!(
            kernels
                .f16_to_bf16(&gpu, plan, at(2, plan.bytes - 2), at(3, plan.bytes), 0)
                .is_err()
        );
        assert_eq!(gpu.launch_count(), 0);
    }
}
