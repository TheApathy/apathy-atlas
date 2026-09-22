// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::GgmlIqBuffer;

const HIDDEN_SIZE: u32 = 4096;
const GROUP_SIZE: u32 = 16;
const GROUPS: u32 = HIDDEN_SIZE / GROUP_SIZE;
const KERNEL_SIZE: u32 = 2;
const PHASES: u32 = 2;
const MAX_BLOCK_TOKENS: u32 = 8;
const THREADS: u32 = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Glm53Dflash2ConvPhase {
    Prepare = 0,
    Finish = 1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53Dflash2ConvPlan {
    pub batch: u32,
    pub tokens: u32,
    pub blocks: u32,
    pub hidden_bytes: usize,
    pub dynamic_bytes: usize,
    pub base_bytes: usize,
}

impl Glm53Dflash2ConvPlan {
    pub fn new(
        batch: u32,
        tokens: u32,
        hidden_size: u32,
        group_size: u32,
        kernel_size: u32,
    ) -> Result<Self> {
        if batch == 0 || tokens == 0 || tokens > MAX_BLOCK_TOKENS {
            bail!("GLM DFlash2 convolution requires batch>0 and 1..=8 tokens");
        }
        if hidden_size != HIDDEN_SIZE || group_size != GROUP_SIZE || kernel_size != KERNEL_SIZE {
            bail!("GLM DFlash2 convolution requires exact H4096/group16/kernel2");
        }
        let rows = u64::from(batch)
            .checked_mul(u64::from(tokens))
            .context("GLM DFlash2 convolution row overflow")?;
        let hidden_elements = rows
            .checked_mul(u64::from(hidden_size))
            .context("GLM DFlash2 convolution hidden element overflow")?;
        let blocks = hidden_elements
            .checked_add(u64::from(THREADS - 1))
            .context("GLM DFlash2 convolution grid overflow")?
            / u64::from(THREADS);
        let dynamic_elements = rows
            .checked_mul(u64::from(PHASES))
            .and_then(|count| count.checked_mul(u64::from(kernel_size)))
            .and_then(|count| count.checked_mul(u64::from(GROUPS)))
            .context("GLM DFlash2 convolution dynamic element overflow")?;
        let base_elements = u64::from(PHASES)
            .checked_mul(u64::from(kernel_size))
            .and_then(|count| count.checked_mul(u64::from(hidden_size)))
            .context("GLM DFlash2 convolution base element overflow")?;
        Ok(Self {
            batch,
            tokens,
            blocks: u32::try_from(blocks).context("GLM DFlash2 convolution CUDA grid overflow")?,
            hidden_bytes: usize::try_from(hidden_elements)?
                .checked_mul(2)
                .context("GLM DFlash2 convolution hidden byte overflow")?,
            dynamic_bytes: usize::try_from(dynamic_elements)?
                .checked_mul(2)
                .context("GLM DFlash2 convolution dynamic byte overflow")?,
            base_bytes: usize::try_from(base_elements)?
                .checked_mul(2)
                .context("GLM DFlash2 convolution base byte overflow")?,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Glm53Dflash2ConvBuffers {
    pub input_bf16: GgmlIqBuffer,
    pub dynamic_bf16: GgmlIqBuffer,
    pub base_bf16: GgmlIqBuffer,
    pub output_bf16: GgmlIqBuffer,
}

pub struct Glm53Dflash2ConvKernel {
    grouped_causal: KernelHandle,
}

impl Glm53Dflash2ConvKernel {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            grouped_causal: gpu.kernel(
                "glm53_dflash2_conv",
                "atlas_glm53_dflash2_grouped_causal_conv",
            )?,
        })
    }

    pub fn launch(
        &self,
        gpu: &dyn GpuBackend,
        plan: Glm53Dflash2ConvPlan,
        phase: Glm53Dflash2ConvPhase,
        buffers: Glm53Dflash2ConvBuffers,
        stream: u64,
    ) -> Result<()> {
        validate_buffers(plan, buffers)?;
        KernelLaunch::new(gpu, self.grouped_causal)
            .grid([plan.blocks, 1, 1])
            .block([THREADS, 1, 1])
            .arg_ptr(buffers.input_bf16.ptr)
            .arg_ptr(buffers.dynamic_bf16.ptr)
            .arg_ptr(buffers.base_bf16.ptr)
            .arg_ptr(buffers.output_bf16.ptr)
            .arg_u32(plan.batch)
            .arg_u32(plan.tokens)
            .arg_u32(HIDDEN_SIZE)
            .arg_u32(GROUP_SIZE)
            .arg_u32(KERNEL_SIZE)
            .arg_u32(phase as u32)
            .launch(stream)
    }
}

fn validate_buffers(plan: Glm53Dflash2ConvPlan, buffers: Glm53Dflash2ConvBuffers) -> Result<()> {
    let named = [
        ("input", buffers.input_bf16, plan.hidden_bytes),
        ("dynamic", buffers.dynamic_bf16, plan.dynamic_bytes),
        ("base", buffers.base_bf16, plan.base_bytes),
        ("output", buffers.output_bf16, plan.hidden_bytes),
    ];
    // Sized from the table it mirrors, never a literal: a hand-written
    // length silently goes out of bounds the moment the table grows.
    let mut ranges = Vec::with_capacity(named.len());
    for (name, buffer, expected) in named.iter().copied() {
        if buffer.ptr == DevicePtr::NULL || buffer.bytes != expected {
            bail!("GLM DFlash2 convolution {name} buffer is null or has the wrong extent");
        }
        let end = buffer
            .ptr
            .0
            .checked_add(u64::try_from(buffer.bytes)?)
            .with_context(|| format!("GLM DFlash2 convolution {name} address overflow"))?;
        ranges.push((buffer.ptr.0, end));
    }
    for left in 0..ranges.len() {
        for right in left + 1..ranges.len() {
            if ranges[left].0 < ranges[right].1 && ranges[right].0 < ranges[left].1 {
                bail!("GLM DFlash2 convolution device buffers overlap");
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use half::bf16;
    use spark_runtime::gpu::mock::MockGpuBackend;

    use super::*;

    fn rounded(value: f32) -> f32 {
        bf16::from_f32(value).to_f32()
    }

    fn scheduled(values: [f32; 2], base: [f32; 2], dynamic: [f32; 2]) -> f32 {
        let mut output = 0.0f32;
        for offset in 0..2 {
            let base_product = rounded(base[offset] * values[offset]);
            output = rounded(output + base_product);
            output = rounded(output + dynamic[offset] * values[offset]);
        }
        output
    }

    #[test]
    fn exact_geometry_and_bf16_operation_boundaries_are_pinned() {
        let plan = Glm53Dflash2ConvPlan::new(2, 8, 4096, 16, 2).unwrap();
        assert_eq!(plan.blocks, 256);
        assert_eq!(plan.hidden_bytes, 2 * 8 * 4096 * 2);
        assert_eq!(plan.dynamic_bytes, 2 * 8 * 2 * 2 * 256 * 2);
        assert_eq!(plan.base_bytes, 2 * 2 * 4096 * 2);
        assert!(Glm53Dflash2ConvPlan::new(1, 0, 4096, 16, 2).is_err());
        assert!(Glm53Dflash2ConvPlan::new(1, 9, 4096, 16, 2).is_err());
        assert!(Glm53Dflash2ConvPlan::new(1, 8, 4096, 32, 2).is_err());

        let mut divergence = None;
        for seed in 1..4096 {
            let sample = |factor: usize| rounded(((seed * factor) % 257) as f32 / 37.0 - 3.0);
            let values = [sample(17), sample(29)];
            let base = [sample(43), sample(61)];
            let dynamic = [sample(73), sample(89)];
            let exact = scheduled(values, base, dynamic);
            let fused =
                rounded((base[0] + dynamic[0]) * values[0] + (base[1] + dynamic[1]) * values[1]);
            if exact.to_bits() != fused.to_bits() {
                divergence = Some((exact, fused));
                break;
            }
        }
        assert!(divergence.is_some());

        // Upstream materializes causal padding as BF16 +0 and still evaluates
        // both tensor operations. Skipping the padded tap would change its
        // signed-zero and non-finite behavior.
        assert!(scheduled([1.0, 0.0], [0.0, f32::INFINITY], [0.0, 0.0]).is_nan());
    }

    #[test]
    fn launches_both_phases_and_rejects_alias_before_effect() {
        let gpu = MockGpuBackend::new();
        let kernel = Glm53Dflash2ConvKernel::load(&gpu).unwrap();
        let plan = Glm53Dflash2ConvPlan::new(1, 8, 4096, 16, 2).unwrap();
        let at = |ptr, bytes| GgmlIqBuffer {
            ptr: DevicePtr(ptr),
            bytes,
        };
        let valid = Glm53Dflash2ConvBuffers {
            input_bf16: at(0x10_0000, plan.hidden_bytes),
            dynamic_bf16: at(0x20_0000, plan.dynamic_bytes),
            base_bf16: at(0x30_0000, plan.base_bytes),
            output_bf16: at(0x40_0000, plan.hidden_bytes),
        };
        assert!(
            kernel
                .launch(
                    &gpu,
                    plan,
                    Glm53Dflash2ConvPhase::Prepare,
                    Glm53Dflash2ConvBuffers {
                        output_bf16: valid.input_bf16,
                        ..valid
                    },
                    0,
                )
                .is_err()
        );
        assert_eq!(gpu.launch_count(), 0);
        kernel
            .launch(&gpu, plan, Glm53Dflash2ConvPhase::Prepare, valid, 0)
            .unwrap();
        kernel
            .launch(&gpu, plan, Glm53Dflash2ConvPhase::Finish, valid, 0)
            .unwrap();
        assert_eq!(gpu.launch_count(), 2);
    }
}
