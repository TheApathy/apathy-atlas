// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::GgmlIqBuffer;

const HEADS: u32 = 64;
const HEAD_DIM: u32 = 128;
const QKV_DIM: u32 = HEADS * HEAD_DIM;
const THREADS: u32 = 128;
const ELEMENT_THREADS: u32 = 256;
const NORM_EPS: f32 = 1.0e-5;
const GATE_LOWER_BOUND: f32 = -5.0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53KdaNormPlan {
    pub tokens: u32,
    pub groups: u32,
    pub values_bytes: usize,
    pub weight_bytes: usize,
}

impl Glm53KdaNormPlan {
    pub fn new(tokens: u32, heads: u32, head_dim: u32) -> Result<Self> {
        if tokens == 0 {
            bail!("GLM KDA gated norm requires at least one token");
        }
        if heads != HEADS || head_dim != HEAD_DIM {
            bail!("GLM KDA gated norm requires exact 64x128 head geometry");
        }
        let groups = tokens
            .checked_mul(heads)
            .context("GLM KDA gated norm CUDA grid overflow")?;
        let values = usize::try_from(groups)?
            .checked_mul(usize::try_from(head_dim)?)
            .context("GLM KDA gated norm element count overflow")?;
        Ok(Self {
            tokens,
            groups,
            values_bytes: values
                .checked_mul(2)
                .context("GLM KDA gated norm value byte overflow")?,
            weight_bytes: usize::try_from(head_dim)?
                .checked_mul(4)
                .context("GLM KDA gated norm weight byte overflow")?,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53KdaForgetPlan {
    pub tokens: u32,
    pub blocks: u32,
    pub input_bytes: usize,
    pub dt_bias_bytes: usize,
    pub a_log_bytes: usize,
    pub output_bytes: usize,
}

impl Glm53KdaForgetPlan {
    pub fn new(tokens: u32, heads: u32, head_dim: u32) -> Result<Self> {
        if tokens == 0 {
            bail!("GLM KDA forget gate requires at least one token");
        }
        if heads != HEADS || head_dim != HEAD_DIM {
            bail!("GLM KDA forget gate requires exact 64x128 geometry");
        }
        let values = u64::from(tokens)
            .checked_mul(u64::from(QKV_DIM))
            .context("GLM KDA forget gate element overflow")?;
        let blocks = values
            .checked_add(u64::from(ELEMENT_THREADS - 1))
            .context("GLM KDA forget gate grid overflow")?
            / u64::from(ELEMENT_THREADS);
        let values = usize::try_from(values)?;
        Ok(Self {
            tokens,
            blocks: u32::try_from(blocks).context("GLM KDA forget gate CUDA grid overflow")?,
            input_bytes: values
                .checked_mul(2)
                .context("GLM KDA forget input byte overflow")?,
            dt_bias_bytes: usize::try_from(QKV_DIM)?
                .checked_mul(4)
                .context("GLM KDA dt bias byte overflow")?,
            a_log_bytes: usize::try_from(HEADS)?
                .checked_mul(4)
                .context("GLM KDA A log byte overflow")?,
            output_bytes: values
                .checked_mul(4)
                .context("GLM KDA forget output byte overflow")?,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53KdaBetaPlan {
    pub tokens: u32,
    pub blocks: u32,
    pub values_bytes: usize,
}

impl Glm53KdaBetaPlan {
    pub fn new(tokens: u32, heads: u32) -> Result<Self> {
        if tokens == 0 || heads != HEADS {
            bail!("GLM KDA beta requires nonzero tokens and exactly 64 heads");
        }
        let values = u64::from(tokens)
            .checked_mul(u64::from(heads))
            .context("GLM KDA beta element overflow")?;
        let blocks = values
            .checked_add(u64::from(ELEMENT_THREADS - 1))
            .context("GLM KDA beta grid overflow")?
            / u64::from(ELEMENT_THREADS);
        Ok(Self {
            tokens,
            blocks: u32::try_from(blocks).context("GLM KDA beta CUDA grid overflow")?,
            values_bytes: usize::try_from(values)?
                .checked_mul(2)
                .context("GLM KDA beta byte overflow")?,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Glm53KdaNormBuffers {
    pub input_bf16: GgmlIqBuffer,
    pub weight_f32: GgmlIqBuffer,
    pub gate_bf16: GgmlIqBuffer,
    pub output_bf16: GgmlIqBuffer,
}

#[derive(Debug, Clone, Copy)]
pub struct Glm53KdaForgetBuffers {
    pub projected_bf16: GgmlIqBuffer,
    pub dt_bias_f32: GgmlIqBuffer,
    pub a_log_f32: GgmlIqBuffer,
    pub output_f32: GgmlIqBuffer,
}

#[derive(Debug, Clone, Copy)]
pub struct Glm53KdaBetaBuffers {
    pub projected_bf16: GgmlIqBuffer,
    pub output_bf16: GgmlIqBuffer,
}

pub struct Glm53KdaKernels {
    forget_gate: KernelHandle,
    beta: KernelHandle,
    gated_norm: KernelHandle,
}

impl Glm53KdaKernels {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            forget_gate: gpu.kernel("glm53_kda", "atlas_glm53_kda_forget_gate")?,
            beta: gpu.kernel("glm53_kda", "atlas_glm53_kda_beta_sigmoid")?,
            gated_norm: gpu.kernel("glm53_kda", "atlas_glm53_kda_gated_rms_norm")?,
        })
    }

    pub fn forget_gate(
        &self,
        gpu: &dyn GpuBackend,
        plan: Glm53KdaForgetPlan,
        buffers: Glm53KdaForgetBuffers,
        stream: u64,
    ) -> Result<()> {
        validate_set(&[
            (
                "forget projection",
                buffers.projected_bf16,
                plan.input_bytes,
            ),
            ("forget dt bias", buffers.dt_bias_f32, plan.dt_bias_bytes),
            ("forget A log", buffers.a_log_f32, plan.a_log_bytes),
            ("forget output", buffers.output_f32, plan.output_bytes),
        ])?;
        KernelLaunch::new(gpu, self.forget_gate)
            .grid([plan.blocks, 1, 1])
            .block([ELEMENT_THREADS, 1, 1])
            .arg_ptr(buffers.projected_bf16.ptr)
            .arg_ptr(buffers.dt_bias_f32.ptr)
            .arg_ptr(buffers.a_log_f32.ptr)
            .arg_ptr(buffers.output_f32.ptr)
            .arg_u32(plan.tokens)
            .arg_u32(HEADS)
            .arg_u32(HEAD_DIM)
            .arg_f32(GATE_LOWER_BOUND)
            .launch(stream)
    }

    pub fn beta(
        &self,
        gpu: &dyn GpuBackend,
        plan: Glm53KdaBetaPlan,
        buffers: Glm53KdaBetaBuffers,
        stream: u64,
    ) -> Result<()> {
        validate_set(&[
            ("beta projection", buffers.projected_bf16, plan.values_bytes),
            ("beta output", buffers.output_bf16, plan.values_bytes),
        ])?;
        KernelLaunch::new(gpu, self.beta)
            .grid([plan.blocks, 1, 1])
            .block([ELEMENT_THREADS, 1, 1])
            .arg_ptr(buffers.projected_bf16.ptr)
            .arg_ptr(buffers.output_bf16.ptr)
            .arg_u32(plan.tokens)
            .arg_u32(HEADS)
            .launch(stream)
    }

    pub fn gated_norm(
        &self,
        gpu: &dyn GpuBackend,
        plan: Glm53KdaNormPlan,
        buffers: Glm53KdaNormBuffers,
        stream: u64,
    ) -> Result<()> {
        validate_buffers(plan, buffers)?;
        KernelLaunch::new(gpu, self.gated_norm)
            .grid([plan.groups, 1, 1])
            .block([THREADS, 1, 1])
            .arg_ptr(buffers.input_bf16.ptr)
            .arg_ptr(buffers.weight_f32.ptr)
            .arg_ptr(buffers.gate_bf16.ptr)
            .arg_ptr(buffers.output_bf16.ptr)
            .arg_u32(plan.tokens)
            .arg_u32(HEADS)
            .arg_u32(HEAD_DIM)
            .arg_f32(NORM_EPS)
            .launch(stream)
    }
}

fn validate_buffers(plan: Glm53KdaNormPlan, buffers: Glm53KdaNormBuffers) -> Result<()> {
    validate_set(&[
        ("input", buffers.input_bf16, plan.values_bytes),
        ("weight", buffers.weight_f32, plan.weight_bytes),
        ("gate", buffers.gate_bf16, plan.values_bytes),
        ("output", buffers.output_bf16, plan.values_bytes),
    ])
}

fn validate_set(named: &[(&str, GgmlIqBuffer, usize)]) -> Result<()> {
    let mut ranges = Vec::with_capacity(named.len());
    for &(name, buffer, expected) in named {
        if buffer.bytes != expected || buffer.ptr == DevicePtr::NULL {
            bail!("GLM KDA {name} buffer is null or has the wrong extent");
        }
        let end = buffer
            .ptr
            .0
            .checked_add(u64::try_from(buffer.bytes)?)
            .with_context(|| format!("GLM KDA {name} address overflow"))?;
        ranges.push((buffer.ptr.0, end));
    }
    for left in 0..ranges.len() {
        for right in left + 1..ranges.len() {
            if ranges[left].0 < ranges[right].1 && ranges[right].0 < ranges[left].1 {
                bail!("GLM KDA gated norm buffers overlap");
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

    fn reference(weight_bf16: bool) -> [bf16; 128] {
        let mut input = [bf16::ZERO; 128];
        let mut gate = [bf16::ZERO; 128];
        let mut weight = [0.0f32; 128];
        for index in 0..128 {
            input[index] = bf16::from_f32(0.5 + index as f32 * 0.01);
            gate[index] = bf16::from_f32(-0.75 + index as f32 * 0.0125);
            weight[index] = 1.001 + index as f32 * 0.0001;
        }
        let variance = input
            .iter()
            .map(|value| value.to_f32().powi(2))
            .sum::<f32>()
            / 128.0;
        let inverse_rms = (variance + NORM_EPS).sqrt().recip();
        std::array::from_fn(|index| {
            let admitted_weight = if weight_bf16 {
                bf16::from_f32(weight[index]).to_f32()
            } else {
                weight[index]
            };
            let normalized = input[index].to_f32() * inverse_rms;
            let weighted = admitted_weight * normalized;
            let sigmoid = 1.0 / (1.0 + (-gate[index].to_f32()).exp());
            bf16::from_f32(weighted * sigmoid)
        })
    }

    #[test]
    fn exact_kda_geometry_and_strict_f32_weight_are_pinned() {
        let plan = Glm53KdaNormPlan::new(2, 64, 128).unwrap();
        let forget = Glm53KdaForgetPlan::new(2, 64, 128).unwrap();
        let beta = Glm53KdaBetaPlan::new(2, 64).unwrap();
        assert_eq!(plan.groups, 128);
        assert_eq!(plan.values_bytes, 2 * 64 * 128 * 2);
        assert_eq!(plan.weight_bytes, 128 * 4);
        assert_eq!(forget.input_bytes, 2 * 64 * 128 * 2);
        assert_eq!(forget.output_bytes, 2 * 64 * 128 * 4);
        assert_eq!(forget.dt_bias_bytes, 64 * 128 * 4);
        assert_eq!(beta.values_bytes, 2 * 64 * 2);
        assert!(Glm53KdaNormPlan::new(0, 64, 128).is_err());
        assert!(Glm53KdaNormPlan::new(1, 64, 64).is_err());
        assert!(Glm53KdaForgetPlan::new(1, 63, 128).is_err());
        assert!(Glm53KdaBetaPlan::new(1, 63).is_err());
        assert_ne!(reference(false), reference(true));

        let projected = bf16::from_f32(-0.25).to_f32();
        let dt_bias = 0.125f32;
        let a_log = 0.5f32;
        let sigmoid = 1.0 / (1.0 + (-(a_log.exp() * (projected + dt_bias))).exp());
        let forget_gate = GATE_LOWER_BOUND * sigmoid;
        assert!(forget_gate < 0.0 && forget_gate > GATE_LOWER_BOUND);
        let beta_gate = bf16::from_f32(1.0 / (1.0 + (-projected).exp()));
        assert_eq!(beta_gate, bf16::from_f32(0.4378235));
    }

    #[test]
    fn launch_rejects_extent_and_overlap_before_enqueue() {
        let gpu = MockGpuBackend::new();
        let kernels = Glm53KdaKernels::load(&gpu).unwrap();
        let plan = Glm53KdaNormPlan::new(1, 64, 128).unwrap();
        let forget = Glm53KdaForgetPlan::new(1, 64, 128).unwrap();
        let beta = Glm53KdaBetaPlan::new(1, 64).unwrap();
        let at = |ptr, bytes| GgmlIqBuffer {
            ptr: DevicePtr(ptr),
            bytes,
        };
        let input = at(0x10_0000, plan.values_bytes);
        let weight = at(0x20_0000, plan.weight_bytes);
        let gate = at(0x30_0000, plan.values_bytes);
        let output = at(0x40_0000, plan.values_bytes);
        assert!(
            kernels
                .gated_norm(
                    &gpu,
                    plan,
                    Glm53KdaNormBuffers {
                        input_bf16: input,
                        weight_f32: weight,
                        gate_bf16: gate,
                        output_bf16: input,
                    },
                    0,
                )
                .is_err()
        );
        assert_eq!(gpu.launch_count(), 0);
        kernels
            .forget_gate(
                &gpu,
                forget,
                Glm53KdaForgetBuffers {
                    projected_bf16: at(0x50_0000, forget.input_bytes),
                    dt_bias_f32: at(0x60_0000, forget.dt_bias_bytes),
                    a_log_f32: at(0x70_0000, forget.a_log_bytes),
                    output_f32: at(0x80_0000, forget.output_bytes),
                },
                0,
            )
            .unwrap();
        kernels
            .beta(
                &gpu,
                beta,
                Glm53KdaBetaBuffers {
                    projected_bf16: at(0x90_0000, beta.values_bytes),
                    output_bf16: at(0x91_0000, beta.values_bytes),
                },
                0,
            )
            .unwrap();
        kernels
            .gated_norm(
                &gpu,
                plan,
                Glm53KdaNormBuffers {
                    input_bf16: input,
                    weight_f32: weight,
                    gate_bf16: gate,
                    output_bf16: output,
                },
                0,
            )
            .unwrap();
        assert_eq!(gpu.launch_count(), 3);
    }
}
