// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::GgmlIqBuffer;

const GLM53_HIDDEN: u32 = 4096;
const GLM53_EXPERTS: u32 = 288;
const GLM53_TOP_K: u32 = 8;
const THREADS: u32 = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53RouterPlan {
    pub tokens: u32,
    pub hidden: u32,
    pub experts: u32,
    pub top_k: u32,
    pub logits_blocks: u32,
    pub input_bytes: usize,
    pub router_bytes: usize,
    pub bias_bytes: usize,
    pub logits_bytes: usize,
    pub indices_bytes: usize,
    pub weights_bytes: usize,
    /// Per-expert diagnostic buffers: same extent as `logits_bytes`.
    pub scores_bytes: usize,
}

impl Glm53RouterPlan {
    pub fn new(tokens: u32, hidden: u32, experts: u32, top_k: u32) -> Result<Self> {
        if tokens == 0 {
            bail!("GLM router requires at least one token");
        }
        if hidden != GLM53_HIDDEN || experts != GLM53_EXPERTS || top_k != GLM53_TOP_K {
            bail!("GLM router requires exact H4096/E288/top8 geometry");
        }
        let logits_blocks = tokens
            .checked_mul(experts)
            .context("GLM router CUDA grid overflow")?;
        let elements = |rows: u32, columns: u32| -> Result<usize> {
            usize::try_from(rows)?
                .checked_mul(usize::try_from(columns)?)
                .context("GLM router element count overflow")
        };
        let bytes = |rows: u32, columns: u32, width: usize| -> Result<usize> {
            elements(rows, columns)?
                .checked_mul(width)
                .context("GLM router byte count overflow")
        };
        Ok(Self {
            tokens,
            hidden,
            experts,
            top_k,
            logits_blocks,
            input_bytes: bytes(tokens, hidden, 2)?,
            router_bytes: bytes(experts, hidden, 4)?,
            bias_bytes: bytes(1, experts, 4)?,
            logits_bytes: bytes(tokens, experts, 4)?,
            indices_bytes: bytes(tokens, top_k, 4)?,
            weights_bytes: bytes(tokens, top_k, 4)?,
            scores_bytes: bytes(tokens, experts, 4)?,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Glm53RouterBuffers {
    pub input_bf16: GgmlIqBuffer,
    pub router_f32: GgmlIqBuffer,
    pub bias_f32: GgmlIqBuffer,
    pub logits_f32: GgmlIqBuffer,
    pub indices_u32: GgmlIqBuffer,
    pub weights_f32: GgmlIqBuffer,
    /// sigmoid(logits), BEFORE `exp_probs_b`.
    pub probs_f32: GgmlIqBuffer,
    /// sigmoid(logits) + `exp_probs_b` — the score selection actually ranks on.
    pub biased_f32: GgmlIqBuffer,
}

pub struct Glm53RouterKernels {
    logits: KernelHandle,
    topk: KernelHandle,
}

impl Glm53RouterKernels {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            logits: gpu.kernel("glm53_router", "atlas_glm53_router_logits")?,
            topk: gpu.kernel("glm53_router", "atlas_glm53_topk_sigmoid_f32")?,
        })
    }

    pub fn launch(
        &self,
        gpu: &dyn GpuBackend,
        plan: Glm53RouterPlan,
        buffers: Glm53RouterBuffers,
        stream: u64,
    ) -> Result<()> {
        validate_buffers(plan, buffers)?;
        KernelLaunch::new(gpu, self.logits)
            .grid([plan.logits_blocks, 1, 1])
            .block([THREADS, 1, 1])
            .arg_ptr(buffers.input_bf16.ptr)
            .arg_ptr(buffers.router_f32.ptr)
            .arg_ptr(buffers.logits_f32.ptr)
            .arg_u32(plan.tokens)
            .arg_u32(plan.hidden)
            .arg_u32(plan.experts)
            .launch(stream)?;
        KernelLaunch::new(gpu, self.topk)
            .grid([plan.tokens, 1, 1])
            .block([THREADS, 1, 1])
            .arg_ptr(buffers.logits_f32.ptr)
            .arg_ptr(buffers.bias_f32.ptr)
            .arg_ptr(buffers.indices_u32.ptr)
            .arg_ptr(buffers.weights_f32.ptr)
            .arg_ptr(buffers.probs_f32.ptr)
            .arg_ptr(buffers.biased_f32.ptr)
            .arg_u32(plan.tokens)
            .arg_u32(plan.experts)
            .arg_u32(plan.top_k)
            .arg_f32(2.5)
            .launch(stream)
    }
}

fn validate_buffers(plan: Glm53RouterPlan, buffers: Glm53RouterBuffers) -> Result<()> {
    let named = [
        ("input", buffers.input_bf16, plan.input_bytes),
        ("router", buffers.router_f32, plan.router_bytes),
        ("bias", buffers.bias_f32, plan.bias_bytes),
        ("logits", buffers.logits_f32, plan.logits_bytes),
        ("indices", buffers.indices_u32, plan.indices_bytes),
        ("weights", buffers.weights_f32, plan.weights_bytes),
        // Appended, and `ranges` is now sized from `named` rather than by a
        // hand-written literal: the previous `[_; 6]` silently went out of
        // bounds the moment this table grew.
        ("router probs", buffers.probs_f32, plan.scores_bytes),
        ("router biased", buffers.biased_f32, plan.scores_bytes),
    ];
    let mut ranges = Vec::with_capacity(named.len());
    for (name, buffer, expected) in named.iter().copied() {
        if buffer.bytes != expected {
            bail!("GLM router {name} buffer extent mismatch");
        }
        if buffer.ptr == DevicePtr::NULL {
            bail!("GLM router {name} buffer is null");
        }
        let end = buffer
            .ptr
            .0
            .checked_add(u64::try_from(buffer.bytes)?)
            .with_context(|| format!("GLM router {name} address overflow"))?;
        ranges.push((buffer.ptr.0, end));
    }
    for left in 0..ranges.len() {
        for right in left + 1..ranges.len() {
            if ranges[left].0 < ranges[right].1 && ranges[right].0 < ranges[left].1 {
                bail!("GLM router device buffers overlap");
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::mock::MockGpuBackend;

    fn reference_route(logits: &[f32], bias: &[f32]) -> (Vec<usize>, Vec<f32>) {
        let sigmoid = logits
            .iter()
            .map(|value| 1.0 / (1.0 + (-value).exp()))
            .collect::<Vec<_>>();
        let mut ranked = sigmoid
            .iter()
            .zip(bias)
            .enumerate()
            .map(|(index, (score, bias))| (index, score + bias))
            .collect::<Vec<_>>();
        ranked.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        let indices = ranked[..8].iter().map(|entry| entry.0).collect::<Vec<_>>();
        let sum = indices.iter().map(|&index| sigmoid[index]).sum::<f32>();
        let weights = indices
            .iter()
            .map(|&index| sigmoid[index] / sum * 2.5)
            .collect();
        (indices, weights)
    }

    #[test]
    fn exact_target_geometry_and_extents_are_pinned() {
        let plan = Glm53RouterPlan::new(2, 4096, 288, 8).unwrap();
        assert_eq!(plan.logits_blocks, 576);
        assert_eq!(plan.input_bytes, 2 * 4096 * 2);
        assert_eq!(plan.router_bytes, 288 * 4096 * 4);
        assert_eq!(plan.bias_bytes, 288 * 4);
        assert_eq!(plan.logits_bytes, 2 * 288 * 4);
        assert_eq!(plan.indices_bytes, 2 * 8 * 4);
        assert_eq!(plan.weights_bytes, 2 * 8 * 4);
        assert!(Glm53RouterPlan::new(0, 4096, 288, 8).is_err());
        assert!(Glm53RouterPlan::new(1, 4095, 288, 8).is_err());
        assert!(Glm53RouterPlan::new(1, 4096, 289, 8).is_err());
        assert!(Glm53RouterPlan::new(1, 4096, 288, 7).is_err());
    }

    #[test]
    fn invalid_buffers_fail_before_both_launches() {
        let gpu = MockGpuBackend::new();
        let kernels = Glm53RouterKernels::load(&gpu).unwrap();
        let plan = Glm53RouterPlan::new(2, 4096, 288, 8).unwrap();
        let buffer = |ptr, bytes| GgmlIqBuffer {
            ptr: DevicePtr(ptr),
            bytes,
        };
        let valid = Glm53RouterBuffers {
            input_bf16: buffer(0x10_0000, plan.input_bytes),
            router_f32: buffer(0x20_0000, plan.router_bytes),
            bias_f32: buffer(0x70_0000, plan.bias_bytes),
            logits_f32: buffer(0x71_0000, plan.logits_bytes),
            indices_u32: buffer(0x72_0000, plan.indices_bytes),
            weights_f32: buffer(0x73_0000, plan.weights_bytes),
            probs_f32: buffer(0x74_0000, plan.scores_bytes),
            biased_f32: buffer(0x75_0000, plan.scores_bytes),
        };
        assert!(
            kernels
                .launch(
                    &gpu,
                    plan,
                    Glm53RouterBuffers {
                        logits_f32: buffer(valid.router_f32.ptr.0, plan.logits_bytes),
                        ..valid
                    },
                    0,
                )
                .is_err()
        );
        assert_eq!(gpu.launch_count(), 0);
        kernels.launch(&gpu, plan, valid, 0).unwrap();
        assert_eq!(gpu.launch_count(), 2);
    }

    #[test]
    fn reference_routing_uses_bias_only_for_selection() {
        let mut logits = vec![-8.0; 288];
        let mut bias = vec![0.0; 288];
        logits[3] = 2.0;
        logits[7] = 2.0;
        bias[200] = 10.0;
        let (indices, weights) = reference_route(&logits, &bias);
        assert_eq!(&indices[..3], &[200, 3, 7]);
        assert!(weights[0] < weights[1]);
        assert!((weights.iter().sum::<f32>() - 2.5).abs() < 1e-6);
    }
}
