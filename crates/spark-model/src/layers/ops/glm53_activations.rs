// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::GgmlIqBuffer;

const HIDDEN: u32 = 4096;
const EXPERTS: u32 = 288;
const TOP_K: u32 = 8;
const EXPERT_INTERMEDIATE: u32 = 2048;
const SHARED_INTERMEDIATE: u32 = 12288;
const THREADS: u32 = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53SwigluPlan {
    pub rows: u32,
    pub width: u32,
    pub blocks: u32,
    pub value_bytes: usize,
}

impl Glm53SwigluPlan {
    pub fn new(rows: u32, width: u32) -> Result<Self> {
        if rows == 0 {
            bail!("GLM SwiGLU requires at least one row");
        }
        if width != EXPERT_INTERMEDIATE && width != SHARED_INTERMEDIATE {
            bail!("GLM SwiGLU requires exact width 2048 or 12288");
        }
        let values = u64::from(rows)
            .checked_mul(u64::from(width))
            .context("GLM SwiGLU element count overflow")?;
        let blocks = values
            .checked_add(u64::from(THREADS - 1))
            .context("GLM SwiGLU grid overflow")?
            / u64::from(THREADS);
        let value_bytes = usize::try_from(values)?
            .checked_mul(2)
            .context("GLM SwiGLU byte count overflow")?;
        Ok(Self {
            rows,
            width,
            blocks: u32::try_from(blocks).context("GLM SwiGLU CUDA grid overflow")?,
            value_bytes,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53ExpertReducePlan {
    pub tokens: u32,
    pub routed_bytes: usize,
    pub indices_bytes: usize,
    pub weights_bytes: usize,
    pub hidden_bytes: usize,
}

impl Glm53ExpertReducePlan {
    pub fn new(tokens: u32, hidden: u32, experts: u32, top_k: u32) -> Result<Self> {
        if tokens == 0 {
            bail!("GLM expert reduction requires at least one token");
        }
        if hidden != HIDDEN || experts != EXPERTS || top_k != TOP_K {
            bail!("GLM expert reduction requires exact H4096/E288/top8 geometry");
        }
        let bytes = |factors: &[u32], width: usize| -> Result<usize> {
            let values = factors.iter().try_fold(1usize, |values, &factor| {
                values
                    .checked_mul(usize::try_from(factor)?)
                    .context("GLM expert reduction element count overflow")
            })?;
            values
                .checked_mul(width)
                .context("GLM expert reduction byte count overflow")
        };
        Ok(Self {
            tokens,
            routed_bytes: bytes(&[tokens, top_k, hidden], 2)?,
            indices_bytes: bytes(&[tokens, top_k], 4)?,
            weights_bytes: bytes(&[tokens, top_k], 4)?,
            hidden_bytes: bytes(&[tokens, hidden], 2)?,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Glm53SwigluBuffers {
    pub gate_bf16: GgmlIqBuffer,
    pub up_bf16: GgmlIqBuffer,
    pub output_bf16: GgmlIqBuffer,
}

#[derive(Debug, Clone, Copy)]
pub struct Glm53ExpertReduceBuffers {
    pub routed_bf16: GgmlIqBuffer,
    pub indices_u32: GgmlIqBuffer,
    pub weights_f32: GgmlIqBuffer,
    pub shared_bf16: GgmlIqBuffer,
    pub output_bf16: GgmlIqBuffer,
}

pub struct Glm53ActivationKernels {
    swiglu: KernelHandle,
    reduce: KernelHandle,
}

impl Glm53ActivationKernels {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            swiglu: gpu.kernel("glm53_activations", "atlas_glm53_clamped_swiglu")?,
            reduce: gpu.kernel("glm53_activations", "atlas_glm53_ordered_expert_reduce")?,
        })
    }

    pub fn swiglu(
        &self,
        gpu: &dyn GpuBackend,
        plan: Glm53SwigluPlan,
        buffers: Glm53SwigluBuffers,
        stream: u64,
    ) -> Result<()> {
        validate_set(&[
            ("gate", buffers.gate_bf16, plan.value_bytes),
            ("up", buffers.up_bf16, plan.value_bytes),
            ("output", buffers.output_bf16, plan.value_bytes),
        ])?;
        KernelLaunch::new(gpu, self.swiglu)
            .grid([plan.blocks, 1, 1])
            .block([THREADS, 1, 1])
            .arg_ptr(buffers.gate_bf16.ptr)
            .arg_ptr(buffers.up_bf16.ptr)
            .arg_ptr(buffers.output_bf16.ptr)
            .arg_u32(plan.rows)
            .arg_u32(plan.width)
            .launch(stream)
    }

    pub fn reduce(
        &self,
        gpu: &dyn GpuBackend,
        plan: Glm53ExpertReducePlan,
        buffers: Glm53ExpertReduceBuffers,
        stream: u64,
    ) -> Result<()> {
        validate_set(&[
            ("routed", buffers.routed_bf16, plan.routed_bytes),
            ("indices", buffers.indices_u32, plan.indices_bytes),
            ("weights", buffers.weights_f32, plan.weights_bytes),
            ("shared", buffers.shared_bf16, plan.hidden_bytes),
            ("output", buffers.output_bf16, plan.hidden_bytes),
        ])?;
        KernelLaunch::new(gpu, self.reduce)
            .grid([plan.tokens, 1, 1])
            .block([THREADS, 1, 1])
            .arg_ptr(buffers.routed_bf16.ptr)
            .arg_ptr(buffers.indices_u32.ptr)
            .arg_ptr(buffers.weights_f32.ptr)
            .arg_ptr(buffers.shared_bf16.ptr)
            .arg_ptr(buffers.output_bf16.ptr)
            .arg_u32(plan.tokens)
            .arg_u32(HIDDEN)
            .arg_u32(EXPERTS)
            .arg_u32(TOP_K)
            .launch(stream)
    }
}

fn validate_set(named: &[(&str, GgmlIqBuffer, usize)]) -> Result<()> {
    let mut ranges = Vec::with_capacity(named.len());
    for &(name, buffer, expected) in named {
        if buffer.bytes != expected || buffer.ptr == DevicePtr::NULL {
            bail!("GLM activation {name} buffer is null or has the wrong extent");
        }
        let end = buffer
            .ptr
            .0
            .checked_add(u64::try_from(buffer.bytes)?)
            .with_context(|| format!("GLM activation {name} address overflow"))?;
        ranges.push((buffer.ptr.0, end));
    }
    for left in 0..ranges.len() {
        for right in left + 1..ranges.len() {
            if ranges[left].0 < ranges[right].1 && ranges[right].0 < ranges[left].1 {
                bail!("GLM activation device buffers overlap");
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

    fn round(value: f32) -> f32 {
        bf16::from_f32(value).to_f32()
    }

    fn swiglu(gate: f32, up: f32) -> f32 {
        let gate = round(gate).min(10.0);
        let up = round(up).clamp(-10.0, 10.0);
        let silu = round(gate / (1.0 + (-gate).exp()));
        round(silu * up)
    }

    fn ordered_sum(ids: &[u32; 8], values: &[f32; 8]) -> f32 {
        let mut slots = [0usize, 1, 2, 3, 4, 5, 6, 7];
        slots.sort_by_key(|&slot| ids[slot]);
        slots
            .into_iter()
            .fold(0.0, |sum, slot| round(sum + round(values[slot])))
    }

    #[test]
    fn exact_activation_geometries_and_bf16_boundaries_are_pinned() {
        let expert = Glm53SwigluPlan::new(16, 2048).unwrap();
        let shared = Glm53SwigluPlan::new(2, 12288).unwrap();
        let reduce = Glm53ExpertReducePlan::new(2, 4096, 288, 8).unwrap();
        assert_eq!(expert.value_bytes, 16 * 2048 * 2);
        assert_eq!(shared.value_bytes, 2 * 12288 * 2);
        assert_eq!(reduce.routed_bytes, 2 * 8 * 4096 * 2);
        assert_eq!(reduce.hidden_bytes, 2 * 4096 * 2);
        assert!(Glm53SwigluPlan::new(1, 4096).is_err());
        assert!(Glm53ExpertReducePlan::new(1, 4096, 288, 7).is_err());
        assert_eq!(swiglu(12.0, 12.0), swiglu(10.0, 10.0));
        assert_eq!(swiglu(-12.0, -12.0), swiglu(-12.0, -10.0));

        let ids = [0, 2, 1, 3, 4, 5, 6, 7];
        let values = [9984.0, 1.0, -9984.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let slot_order = values
            .into_iter()
            .fold(0.0, |sum, value| round(sum + round(value)));
        assert_eq!(ordered_sum(&ids, &values), 1.0);
        assert_eq!(slot_order, 0.0);
    }

    #[test]
    fn both_launches_reject_overlap_before_enqueue() {
        let gpu = MockGpuBackend::new();
        let kernels = Glm53ActivationKernels::load(&gpu).unwrap();
        let swiglu = Glm53SwigluPlan::new(8, 2048).unwrap();
        let reduce = Glm53ExpertReducePlan::new(1, 4096, 288, 8).unwrap();
        let at = |ptr, bytes| GgmlIqBuffer {
            ptr: DevicePtr(ptr),
            bytes,
        };
        assert!(
            kernels
                .swiglu(
                    &gpu,
                    swiglu,
                    Glm53SwigluBuffers {
                        gate_bf16: at(0x10_0000, swiglu.value_bytes),
                        up_bf16: at(0x20_0000, swiglu.value_bytes),
                        output_bf16: at(0x10_0000, swiglu.value_bytes),
                    },
                    0,
                )
                .is_err()
        );
        assert_eq!(gpu.launch_count(), 0);
        kernels
            .swiglu(
                &gpu,
                swiglu,
                Glm53SwigluBuffers {
                    gate_bf16: at(0x10_0000, swiglu.value_bytes),
                    up_bf16: at(0x20_0000, swiglu.value_bytes),
                    output_bf16: at(0x30_0000, swiglu.value_bytes),
                },
                0,
            )
            .unwrap();
        kernels
            .reduce(
                &gpu,
                reduce,
                Glm53ExpertReduceBuffers {
                    routed_bf16: at(0x40_0000, reduce.routed_bytes),
                    indices_u32: at(0x60_0000, reduce.indices_bytes),
                    weights_f32: at(0x61_0000, reduce.weights_bytes),
                    shared_bf16: at(0x62_0000, reduce.hidden_bytes),
                    output_bf16: at(0x63_0000, reduce.hidden_bytes),
                },
                0,
            )
            .unwrap();
        assert_eq!(gpu.launch_count(), 2);
    }
}
