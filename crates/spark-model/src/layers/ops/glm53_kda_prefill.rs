// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::GgmlIqBuffer;

const HEADS: u32 = 64;
const HEAD_DIM: u32 = 128;
const THREADS: u32 = 128;
const L2_EPS: f32 = 1.0e-6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53KdaPrefillPlan {
    pub batch: u32,
    pub tokens: u32,
    pub groups: u32,
    pub vector_bytes: usize,
    pub decay_bytes: usize,
    pub beta_bytes: usize,
    pub state_bytes: usize,
}

impl Glm53KdaPrefillPlan {
    pub fn new(batch: u32, tokens: u32, heads: u32, key_dim: u32, value_dim: u32) -> Result<Self> {
        if batch == 0 || tokens == 0 {
            bail!("GLM KDA prefill requires nonzero batch and token counts");
        }
        if heads != HEADS || key_dim != HEAD_DIM || value_dim != HEAD_DIM {
            bail!("GLM KDA prefill requires exact 64 heads x K128 x V128");
        }
        let groups = batch
            .checked_mul(heads)
            .context("GLM KDA prefill CUDA grid overflow")?;
        let token_groups = usize::try_from(batch)?
            .checked_mul(usize::try_from(tokens)?)
            .and_then(|count| count.checked_mul(usize::try_from(heads).ok()?))
            .context("GLM KDA prefill token-group overflow")?;
        let vectors = token_groups
            .checked_mul(usize::try_from(key_dim)?)
            .context("GLM KDA prefill vector element overflow")?;
        let states = usize::try_from(groups)?
            .checked_mul(usize::try_from(key_dim)?)
            .and_then(|count| count.checked_mul(usize::try_from(value_dim).ok()?))
            .context("GLM KDA prefill state element overflow")?;
        Ok(Self {
            batch,
            tokens,
            groups,
            vector_bytes: vectors
                .checked_mul(2)
                .context("GLM KDA prefill vector byte overflow")?,
            decay_bytes: vectors
                .checked_mul(4)
                .context("GLM KDA prefill decay byte overflow")?,
            beta_bytes: token_groups
                .checked_mul(2)
                .context("GLM KDA prefill beta byte overflow")?,
            state_bytes: states
                .checked_mul(4)
                .context("GLM KDA prefill state byte overflow")?,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Glm53KdaPrefillBuffers {
    pub state_f32: GgmlIqBuffer,
    pub query_bf16: GgmlIqBuffer,
    pub key_bf16: GgmlIqBuffer,
    pub value_bf16: GgmlIqBuffer,
    pub log_decay_f32: GgmlIqBuffer,
    pub beta_bf16: GgmlIqBuffer,
    pub output_bf16: GgmlIqBuffer,
}

pub struct Glm53KdaPrefillKernel {
    prefill: KernelHandle,
}

impl Glm53KdaPrefillKernel {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            prefill: gpu.kernel("glm53_kda", "atlas_glm53_kda_prefill")?,
        })
    }

    pub fn launch(
        &self,
        gpu: &dyn GpuBackend,
        plan: Glm53KdaPrefillPlan,
        buffers: Glm53KdaPrefillBuffers,
        stream: u64,
    ) -> Result<()> {
        validate_buffers(plan, buffers)?;
        KernelLaunch::new(gpu, self.prefill)
            .grid([plan.groups, 1, 1])
            .block([THREADS, 1, 1])
            .arg_ptr(buffers.state_f32.ptr)
            .arg_ptr(buffers.query_bf16.ptr)
            .arg_ptr(buffers.key_bf16.ptr)
            .arg_ptr(buffers.value_bf16.ptr)
            .arg_ptr(buffers.log_decay_f32.ptr)
            .arg_ptr(buffers.beta_bf16.ptr)
            .arg_ptr(buffers.output_bf16.ptr)
            .arg_u32(plan.batch)
            .arg_u32(plan.tokens)
            .arg_u32(HEADS)
            .arg_u32(HEAD_DIM)
            .arg_u32(HEAD_DIM)
            .arg_f32(L2_EPS)
            .launch(stream)
    }
}

fn validate_buffers(plan: Glm53KdaPrefillPlan, buffers: Glm53KdaPrefillBuffers) -> Result<()> {
    let named = [
        ("state", buffers.state_f32, plan.state_bytes),
        ("query", buffers.query_bf16, plan.vector_bytes),
        ("key", buffers.key_bf16, plan.vector_bytes),
        ("value", buffers.value_bf16, plan.vector_bytes),
        ("log decay", buffers.log_decay_f32, plan.decay_bytes),
        ("beta", buffers.beta_bf16, plan.beta_bytes),
        ("output", buffers.output_bf16, plan.vector_bytes),
    ];
    // Sized from the table it mirrors, never a literal: a hand-written
    // length silently goes out of bounds the moment the table grows.
    let mut ranges = Vec::with_capacity(named.len());
    for (name, buffer, expected) in named.iter().copied() {
        if buffer.bytes != expected || buffer.ptr == DevicePtr::NULL {
            bail!("GLM KDA prefill {name} buffer is null or has the wrong extent");
        }
        let end = buffer
            .ptr
            .0
            .checked_add(u64::try_from(buffer.bytes)?)
            .with_context(|| format!("GLM KDA prefill {name} address overflow"))?;
        ranges.push((buffer.ptr.0, end));
    }
    for left in 0..ranges.len() {
        for right in left + 1..ranges.len() {
            if ranges[left].0 < ranges[right].1 && ranges[right].0 < ranges[left].1 {
                bail!("GLM KDA prefill device buffers overlap");
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use spark_runtime::gpu::mock::MockGpuBackend;

    use super::*;

    fn step(mut state: [f32; 4], query: [f32; 2]) -> ([f32; 2], [f32; 4]) {
        let key = [1.5f32, 0.25];
        let value = [0.75f32, -0.5];
        let decay = [0.5f32, 0.25];
        let q_norm = (query.iter().map(|x| x * x).sum::<f32>() + L2_EPS).sqrt();
        let k_norm = (key.iter().map(|x| x * x).sum::<f32>() + L2_EPS).sqrt();
        let query = query.map(|x| x / q_norm * (2.0f32).sqrt().recip());
        let key = key.map(|x| x / k_norm);
        for row in 0..2 {
            for column in 0..2 {
                state[row * 2 + column] *= decay[row];
            }
        }
        let memory: [f32; 2] =
            std::array::from_fn(|column| state[column] * key[0] + state[2 + column] * key[1]);
        let delta: [f32; 2] =
            std::array::from_fn(|column| (value[column] - memory[column]) * 0.625);
        for row in 0..2 {
            for column in 0..2 {
                state[row * 2 + column] += key[row] * delta[column];
            }
        }
        let output: [f32; 2] =
            std::array::from_fn(|column| state[column] * query[0] + state[2 + column] * query[1]);
        (output, state)
    }

    #[test]
    fn exact_prefill_geometry_and_causal_state_are_pinned() {
        let plan = Glm53KdaPrefillPlan::new(2, 3, 64, 128, 128).unwrap();
        assert_eq!(plan.groups, 128);
        assert_eq!(plan.vector_bytes, 2 * 3 * 64 * 128 * 2);
        assert_eq!(plan.decay_bytes, 2 * 3 * 64 * 128 * 4);
        assert_eq!(plan.beta_bytes, 2 * 3 * 64 * 2);
        assert_eq!(plan.state_bytes, 2 * 64 * 128 * 128 * 4);
        assert!(Glm53KdaPrefillPlan::new(0, 3, 64, 128, 128).is_err());
        assert!(Glm53KdaPrefillPlan::new(1, 0, 64, 128, 128).is_err());
        assert!(Glm53KdaPrefillPlan::new(1, 3, 63, 128, 128).is_err());
        assert!(Glm53KdaPrefillPlan::new(u32::MAX, u32::MAX, 64, 128, 128).is_err());

        let initial = [1.0, 2.0, 3.0, 4.0];
        let (_, after_first) = step(initial, [0.5, -1.0]);
        let (causal_second, _) = step(after_first, [-0.25, 0.75]);
        let (reset_second, _) = step(initial, [-0.25, 0.75]);
        assert_ne!(causal_second, reset_second);
    }

    #[test]
    fn launch_rejects_alias_before_effect() {
        let gpu = MockGpuBackend::new();
        let kernel = Glm53KdaPrefillKernel::load(&gpu).unwrap();
        let plan = Glm53KdaPrefillPlan::new(1, 2, 64, 128, 128).unwrap();
        let at = |ptr, bytes| GgmlIqBuffer {
            ptr: DevicePtr(ptr),
            bytes,
        };
        let valid = Glm53KdaPrefillBuffers {
            state_f32: at(0x10_0000, plan.state_bytes),
            query_bf16: at(0x60_0000, plan.vector_bytes),
            key_bf16: at(0x70_0000, plan.vector_bytes),
            value_bf16: at(0x80_0000, plan.vector_bytes),
            log_decay_f32: at(0x90_0000, plan.decay_bytes),
            beta_bf16: at(0xa0_0000, plan.beta_bytes),
            output_bf16: at(0xb0_0000, plan.vector_bytes),
        };
        assert!(
            kernel
                .launch(
                    &gpu,
                    plan,
                    Glm53KdaPrefillBuffers {
                        output_bf16: valid.query_bf16,
                        ..valid
                    },
                    0,
                )
                .is_err()
        );
        assert_eq!(gpu.launch_count(), 0);
        kernel.launch(&gpu, plan, valid, 0).unwrap();
        assert_eq!(gpu.launch_count(), 1);
    }
}
