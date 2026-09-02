// SPDX-License-Identifier: AGPL-3.0-only

//! Exact normalization and F32 index-projection boundaries for GLM-5.3 DSA.
//! The RMS scales are absolute model weights (never Atlas's legacy `1+w`
//! convention). The GGUF projection is physical output-major `[N=32][K=4096]`.

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::GgmlIqBuffer;

const MAX_ROWS: u32 = 65_520;
const RMS_EPS: f32 = 1.0e-5;
const LAYER_EPS: f32 = 1.0e-6;
const KV_RANK: u32 = 512;
const Q_RANK: u32 = 1536;
const INDEX_DIM: u32 = 128;
const HIDDEN: u32 = 4096;
const INDEX_HEADS: u32 = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Glm53DsaNormProjectionKind {
    AbsoluteRms512,
    AbsoluteRms1536,
    BiasedLayerNorm128,
    F32IndexProjection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53DsaNormProjectionPlan {
    pub kind: Glm53DsaNormProjectionKind,
    pub rows: u32,
    pub input_width: u32,
    pub output_width: u32,
    pub epsilon_bits: u32,
    pub threads: u32,
    pub input_bytes: usize,
    pub weight_bytes: usize,
    pub bias_bytes: usize,
    pub output_bytes: usize,
}

impl Glm53DsaNormProjectionPlan {
    pub fn new(
        kind: Glm53DsaNormProjectionKind,
        rows: u32,
        input_width: u32,
        output_width: u32,
        epsilon: f32,
    ) -> Result<Self> {
        if !(1..=MAX_ROWS).contains(&rows) {
            bail!("GLM DSA norm/projection requires rows in 1..=65,520");
        }
        let (exact_input, exact_output, exact_eps, threads, has_bias, projection) = match kind {
            Glm53DsaNormProjectionKind::AbsoluteRms512 => {
                (KV_RANK, KV_RANK, RMS_EPS, 256, false, false)
            }
            Glm53DsaNormProjectionKind::AbsoluteRms1536 => {
                (Q_RANK, Q_RANK, RMS_EPS, 256, false, false)
            }
            Glm53DsaNormProjectionKind::BiasedLayerNorm128 => {
                (INDEX_DIM, INDEX_DIM, LAYER_EPS, 128, true, false)
            }
            Glm53DsaNormProjectionKind::F32IndexProjection => {
                (HIDDEN, INDEX_HEADS, 0.0, 32, false, true)
            }
        };
        if input_width != exact_input
            || output_width != exact_output
            || epsilon.to_bits() != exact_eps.to_bits()
        {
            bail!("GLM DSA norm/projection geometry or epsilon mismatch");
        }
        let input_bytes = extent(rows, input_width, 2)?;
        let output_bytes = extent(rows, output_width, 2)?;
        let weight_bytes = if projection {
            extent(input_width, output_width, 4)?
        } else {
            extent(1, input_width, 4)?
        };
        let bias_bytes = if has_bias {
            extent(1, input_width, 4)?
        } else {
            0
        };
        Ok(Self {
            kind,
            rows,
            input_width,
            output_width,
            epsilon_bits: exact_eps.to_bits(),
            threads,
            input_bytes,
            weight_bytes,
            bias_bytes,
            output_bytes,
        })
    }

    pub fn validate(self) -> Result<()> {
        let rebuilt = Self::new(
            self.kind,
            self.rows,
            self.input_width,
            self.output_width,
            f32::from_bits(self.epsilon_bits),
        )?;
        if rebuilt != self {
            bail!("forged GLM DSA norm/projection plan");
        }
        Ok(())
    }
}

fn extent(rows: u32, columns: u32, element_bytes: u64) -> Result<usize> {
    usize::try_from(
        u64::from(rows)
            .checked_mul(u64::from(columns))
            .and_then(|values| values.checked_mul(element_bytes))
            .context("GLM DSA norm/projection byte-extent overflow")?,
    )
    .context("GLM DSA norm/projection extent does not fit usize")
}

#[derive(Debug, Clone, Copy)]
pub struct Glm53DsaNormProjectionBuffers {
    pub input_bf16: GgmlIqBuffer,
    pub weight_f32: GgmlIqBuffer,
    /// Must be exactly null/zero except for `BiasedLayerNorm128`.
    pub bias_f32: GgmlIqBuffer,
    pub output_bf16: GgmlIqBuffer,
}

pub struct Glm53DsaNormProjectionKernels {
    rms_norm: KernelHandle,
    layer_norm: KernelHandle,
    index_projection: KernelHandle,
}

impl Glm53DsaNormProjectionKernels {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            rms_norm: gpu.kernel(
                "glm53_dsa_norm_projection",
                "atlas_glm53_dsa_absolute_rms_norm_bf16",
            )?,
            layer_norm: gpu.kernel(
                "glm53_dsa_norm_projection",
                "atlas_glm53_dsa_biased_layer_norm_bf16",
            )?,
            index_projection: gpu.kernel(
                "glm53_dsa_norm_projection",
                "atlas_glm53_dsa_index_projection_f32_bf16",
            )?,
        })
    }

    pub fn launch(
        &self,
        gpu: &dyn GpuBackend,
        plan: Glm53DsaNormProjectionPlan,
        buffers: Glm53DsaNormProjectionBuffers,
        stream: u64,
    ) -> Result<()> {
        plan.validate()?;
        validate_buffers(plan, buffers)?;
        let launch = match plan.kind {
            Glm53DsaNormProjectionKind::AbsoluteRms512
            | Glm53DsaNormProjectionKind::AbsoluteRms1536 => KernelLaunch::new(gpu, self.rms_norm)
                .grid([plan.rows, 1, 1])
                .block([plan.threads, 1, 1])
                .arg_ptr(buffers.input_bf16.ptr)
                .arg_ptr(buffers.weight_f32.ptr)
                .arg_ptr(buffers.output_bf16.ptr)
                .arg_u32(plan.rows)
                .arg_u32(plan.input_width)
                .arg_f32(f32::from_bits(plan.epsilon_bits)),
            Glm53DsaNormProjectionKind::BiasedLayerNorm128 => {
                KernelLaunch::new(gpu, self.layer_norm)
                    .grid([plan.rows, 1, 1])
                    .block([plan.threads, 1, 1])
                    .arg_ptr(buffers.input_bf16.ptr)
                    .arg_ptr(buffers.weight_f32.ptr)
                    .arg_ptr(buffers.bias_f32.ptr)
                    .arg_ptr(buffers.output_bf16.ptr)
                    .arg_u32(plan.rows)
                    .arg_u32(plan.input_width)
                    .arg_f32(f32::from_bits(plan.epsilon_bits))
            }
            Glm53DsaNormProjectionKind::F32IndexProjection => {
                KernelLaunch::new(gpu, self.index_projection)
                    .grid([plan.rows, 1, 1])
                    .block([plan.threads, 1, 1])
                    .arg_ptr(buffers.input_bf16.ptr)
                    .arg_ptr(buffers.weight_f32.ptr)
                    .arg_ptr(buffers.output_bf16.ptr)
                    .arg_u32(plan.rows)
                    .arg_u32(plan.input_width)
                    .arg_u32(plan.output_width)
            }
        };
        launch.launch(stream)
    }
}

fn validate_buffers(
    plan: Glm53DsaNormProjectionPlan,
    buffers: Glm53DsaNormProjectionBuffers,
) -> Result<()> {
    let named = [
        ("input", buffers.input_bf16, plan.input_bytes, 2u64),
        ("weight", buffers.weight_f32, plan.weight_bytes, 4),
        ("bias", buffers.bias_f32, plan.bias_bytes, 4),
        ("output", buffers.output_bf16, plan.output_bytes, 2),
    ];
    let mut ranges = [(0u64, 0u64); 4];
    for (slot, (name, buffer, expected, alignment)) in named.iter().copied().enumerate() {
        if buffer.bytes != expected
            || (expected == 0 && buffer.ptr != DevicePtr::NULL)
            || (expected != 0 && (buffer.ptr == DevicePtr::NULL || buffer.ptr.0 % alignment != 0))
        {
            bail!("GLM DSA norm/projection {name} pointer, alignment, or extent mismatch");
        }
        if expected != 0 {
            let end = buffer
                .ptr
                .0
                .checked_add(u64::try_from(expected)?)
                .with_context(|| format!("GLM DSA norm/projection {name} address overflow"))?;
            ranges[slot] = (buffer.ptr.0, end);
        }
    }
    for left in 0..ranges.len() {
        if ranges[left].0 == ranges[left].1 {
            continue;
        }
        for right in left + 1..ranges.len() {
            if ranges[right].0 != ranges[right].1
                && ranges[left].0 < ranges[right].1
                && ranges[right].0 < ranges[left].1
            {
                bail!("GLM DSA norm/projection device buffers overlap");
            }
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "glm53_dsa_norm_projection_tests.rs"]
mod tests;
