// SPDX-License-Identifier: AGPL-3.0-only

//! Physical GLM-5.3-Flash EXL3 still-image tower.
//!
//! This stays separate from `VisionEncoder`: that type is Qwen-specific and
//! its LayerNorm/absolute-position/deepstack contract is not compatible with
//! GLM's RMSNorm/downsample/merger graph.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::glm53_vision_timing::{Glm53VisionStage, Glm53VisionTiming};
use super::ops::{
    GLM53_EXL3_LOCK_BYTES, GLM53_VISION_HIDDEN, GLM53_VISION_INTERMEDIATE,
    GLM53_VISION_MERGER_INTERMEDIATE, GLM53_VISION_OUT, Glm53Exl3Buffer, Glm53VisionKernels,
    Glm53VisionPlan,
};
use crate::weight_loader::{
    Glm53Exl3Bf16LinearBuffers, Glm53Exl3VisionCatalog, Glm53Exl3VisionRawTensor,
};

const BLOCKS: usize = 24;

struct Allocations<'a> {
    gpu: &'a dyn GpuBackend,
    pointers: Vec<DevicePtr>,
}

impl<'a> Allocations<'a> {
    fn new(gpu: &'a dyn GpuBackend) -> Self {
        Self {
            gpu,
            pointers: Vec::new(),
        }
    }

    fn alloc(&mut self, bytes: usize) -> Result<Glm53Exl3Buffer> {
        ensure!(bytes != 0, "GLM vision refuses zero-byte allocation");
        let ptr = self.gpu.alloc(bytes)?;
        self.pointers.push(ptr);
        Ok(Glm53Exl3Buffer { ptr, bytes })
    }

    fn zeroed(&mut self, bytes: usize) -> Result<Glm53Exl3Buffer> {
        let buffer = self.alloc(bytes)?;
        self.gpu.memset(buffer.ptr, 0, bytes)?;
        Ok(buffer)
    }

    fn release(mut self) -> Result<()> {
        let mut first = None;
        for ptr in self.pointers.drain(..).rev() {
            if let Err(error) = self.gpu.free(ptr) {
                first.get_or_insert(error);
            }
        }
        match first {
            Some(error) => Err(error).context("free GLM vision scratch"),
            None => Ok(()),
        }
    }
}

#[derive(Clone, Copy)]
struct ProjectionScratch {
    input_f16: Glm53Exl3Buffer,
    output_f16: Glm53Exl3Buffer,
    locks_i32: Glm53Exl3Buffer,
    input_hadamard_f16: Glm53Exl3Buffer,
}

/// Exact output row count and byte extent for one GLM image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Glm53VisionOutput {
    pub rows: u32,
    pub width: u32,
    pub bytes: usize,
}

/// Execute one still image and leave merged `[rows,4096]` BF16 embeddings in
/// caller-owned `output`. The method synchronizes before releasing temporary
/// buffers, so return means the output is safe for the language splice.
pub fn forward_glm53_exl3_image(
    gpu: &dyn GpuBackend,
    catalog: &Glm53Exl3VisionCatalog,
    pixels: &[f32],
    grid_height: usize,
    grid_width: usize,
    output: Glm53Exl3Buffer,
    stream: u64,
) -> Result<Glm53VisionOutput> {
    let plan = Glm53VisionPlan::new(grid_height, grid_width)?;
    let pixels_len = usize::try_from(u64::from(plan.rows) * 3 * 2 * 14 * 14)?;
    ensure!(
        pixels.len() == pixels_len,
        "GLM vision pixel extent does not match grid"
    );
    let output_bytes = plan.bytes(plan.merged_rows, GLM53_VISION_OUT, 2)?;
    ensure!(
        output.ptr != DevicePtr::NULL && output.bytes == output_bytes,
        "GLM vision output extent drift"
    );

    let mut allocations = Allocations::new(gpu);
    let result = (|| {
        let kernels = Glm53VisionKernels::load(gpu)?;
        let bytes = |rows, width| plan.bytes(rows, width, 2);
        let pixels_f32 = allocations.alloc(pixels_len * 4)?;
        let pixel_bytes =
            unsafe { std::slice::from_raw_parts(pixels.as_ptr().cast::<u8>(), pixels.len() * 4) };
        gpu.copy_h2d_group_on_stream(&[spark_runtime::gpu::HostToDeviceCopy::new(pixel_bytes, pixels_f32.ptr)], stream)?;

        let hidden = allocations.alloc(bytes(plan.rows, GLM53_VISION_HIDDEN)?)?;
        let norm = allocations.alloc(bytes(plan.rows, GLM53_VISION_HIDDEN)?)?;
        let q = allocations.alloc(bytes(plan.rows, GLM53_VISION_HIDDEN)?)?;
        let k = allocations.alloc(bytes(plan.rows, GLM53_VISION_HIDDEN)?)?;
        let v = allocations.alloc(bytes(plan.rows, GLM53_VISION_HIDDEN)?)?;
        let attention = allocations.alloc(bytes(plan.rows, GLM53_VISION_HIDDEN)?)?;
        let projected = allocations.alloc(bytes(plan.rows, GLM53_VISION_HIDDEN)?)?;
        let gate = allocations.alloc(bytes(plan.rows, GLM53_VISION_INTERMEDIATE)?)?;
        let up = allocations.alloc(bytes(plan.rows, GLM53_VISION_INTERMEDIATE)?)?;
        let activation = allocations.alloc(bytes(plan.rows, GLM53_VISION_INTERMEDIATE)?)?;
        let merged = allocations.alloc(bytes(plan.merged_rows, GLM53_VISION_OUT)?)?;
        let merger_projected = allocations.alloc(bytes(plan.merged_rows, GLM53_VISION_OUT)?)?;
        let merger_norm = allocations.alloc(bytes(plan.merged_rows, GLM53_VISION_OUT)?)?;
        let merger_gate =
            allocations.alloc(bytes(plan.merged_rows, GLM53_VISION_MERGER_INTERMEDIATE)?)?;
        let merger_up =
            allocations.alloc(bytes(plan.merged_rows, GLM53_VISION_MERGER_INTERMEDIATE)?)?;
        let merger_activation =
            allocations.alloc(bytes(plan.merged_rows, GLM53_VISION_MERGER_INTERMEDIATE)?)?;

        let max_input = bytes(plan.rows, GLM53_VISION_INTERMEDIATE)?
            .max(bytes(plan.merged_rows, GLM53_VISION_MERGER_INTERMEDIATE)?);
        let max_output = max_input.max(bytes(plan.rows, GLM53_VISION_INTERMEDIATE)?);
        let projection_scratch = ProjectionScratch {
            input_f16: allocations.alloc(max_input)?,
            output_f16: allocations.alloc(max_output)?,
            locks_i32: allocations.zeroed(GLM53_EXL3_LOCK_BYTES)?,
            input_hadamard_f16: allocations.alloc(max_input)?,
        };

        let mut timing = Glm53VisionTiming::begin(gpu, stream)?;
        kernels.patch_embed(
            gpu,
            plan,
            pixels_f32,
            raw(catalog, "model.visual.patch_embed.proj.weight")?,
            raw(catalog, "model.visual.patch_embed.proj.bias")?,
            hidden,
            stream,
        )?;
        timing.mark(gpu, stream, Glm53VisionStage::Patch)?;

        for block in 0..BLOCKS {
            let base = format!("model.visual.blocks.{block}");
            kernels.rmsnorm(
                gpu,
                hidden,
                raw(catalog, &format!("{base}.norm1.weight"))?,
                norm,
                plan.rows,
                GLM53_VISION_HIDDEN,
                1e-5,
                stream,
            )?;
            project(
                gpu,
                catalog,
                &format!("{base}.attn.q_proj"),
                norm,
                q,
                plan.rows,
                projection_scratch,
                stream,
            )?;
            project(
                gpu,
                catalog,
                &format!("{base}.attn.k_proj"),
                norm,
                k,
                plan.rows,
                projection_scratch,
                stream,
            )?;
            project(
                gpu,
                catalog,
                &format!("{base}.attn.v_proj"),
                norm,
                v,
                plan.rows,
                projection_scratch,
                stream,
            )?;
            timing.mark(gpu, stream, Glm53VisionStage::Qkv)?;
            kernels.qkv_prepare(
                gpu,
                plan,
                q,
                k,
                v,
                raw(catalog, &format!("{base}.attn.q_proj.bias"))?,
                raw(catalog, &format!("{base}.attn.k_proj.bias"))?,
                raw(catalog, &format!("{base}.attn.v_proj.bias"))?,
                raw(catalog, &format!("{base}.attn.q_norm.weight"))?,
                raw(catalog, &format!("{base}.attn.k_norm.weight"))?,
                q,
                k,
                v,
                stream,
            )?;
            kernels.attention(gpu, plan, q, k, v, attention, stream)?;
            timing.mark(gpu, stream, Glm53VisionStage::Attention)?;
            project(
                gpu,
                catalog,
                &format!("{base}.attn.proj"),
                attention,
                projected,
                plan.rows,
                projection_scratch,
                stream,
            )?;
            kernels.bias_residual(
                gpu,
                projected,
                raw(catalog, &format!("{base}.attn.proj.bias"))?,
                hidden,
                hidden,
                plan.rows,
                GLM53_VISION_HIDDEN,
                stream,
            )?;
            timing.mark(gpu, stream, Glm53VisionStage::AttentionOutput)?;
            kernels.rmsnorm(
                gpu,
                hidden,
                raw(catalog, &format!("{base}.norm2.weight"))?,
                norm,
                plan.rows,
                GLM53_VISION_HIDDEN,
                1e-5,
                stream,
            )?;
            project(
                gpu,
                catalog,
                &format!("{base}.mlp.gate_proj"),
                norm,
                gate,
                plan.rows,
                projection_scratch,
                stream,
            )?;
            project(
                gpu,
                catalog,
                &format!("{base}.mlp.up_proj"),
                norm,
                up,
                plan.rows,
                projection_scratch,
                stream,
            )?;
            timing.mark(gpu, stream, Glm53VisionStage::MlpInput)?;
            kernels.swiglu(
                gpu,
                gate,
                up,
                Some(raw(catalog, &format!("{base}.mlp.gate_proj.bias"))?),
                Some(raw(catalog, &format!("{base}.mlp.up_proj.bias"))?),
                activation,
                plan.rows,
                GLM53_VISION_INTERMEDIATE,
                stream,
            )?;
            project(
                gpu,
                catalog,
                &format!("{base}.mlp.down_proj"),
                activation,
                projected,
                plan.rows,
                projection_scratch,
                stream,
            )?;
            kernels.bias_residual(
                gpu,
                projected,
                raw(catalog, &format!("{base}.mlp.down_proj.bias"))?,
                hidden,
                hidden,
                plan.rows,
                GLM53_VISION_HIDDEN,
                stream,
            )?;
            timing.mark(gpu, stream, Glm53VisionStage::MlpOutput)?;
        }

        kernels.post_downsample(
            gpu,
            plan,
            hidden,
            raw(catalog, "model.visual.post_layernorm.weight")?,
            raw(catalog, "model.visual.downsample.weight")?,
            raw(catalog, "model.visual.downsample.bias")?,
            merged,
            stream,
        )?;
        project(
            gpu,
            catalog,
            "model.visual.merger.proj",
            merged,
            merger_projected,
            plan.merged_rows,
            projection_scratch,
            stream,
        )?;
        kernels.layernorm_gelu(
            gpu,
            merger_projected,
            raw(catalog, "model.visual.merger.post_projection_norm.weight")?,
            raw(catalog, "model.visual.merger.post_projection_norm.bias")?,
            merger_norm,
            plan.merged_rows,
            stream,
        )?;
        project(
            gpu,
            catalog,
            "model.visual.merger.gate_proj",
            merger_norm,
            merger_gate,
            plan.merged_rows,
            projection_scratch,
            stream,
        )?;
        project(
            gpu,
            catalog,
            "model.visual.merger.up_proj",
            merger_norm,
            merger_up,
            plan.merged_rows,
            projection_scratch,
            stream,
        )?;
        kernels.swiglu(
            gpu,
            merger_gate,
            merger_up,
            None,
            None,
            merger_activation,
            plan.merged_rows,
            GLM53_VISION_MERGER_INTERMEDIATE,
            stream,
        )?;
        project(
            gpu,
            catalog,
            "model.visual.merger.down_proj",
            merger_activation,
            output,
            plan.merged_rows,
            projection_scratch,
            stream,
        )?;
        timing.mark(gpu, stream, Glm53VisionStage::Merger)?;
        gpu.synchronize(stream)?;
        timing.report(plan.rows, plan.merged_rows);
        Ok(Glm53VisionOutput {
            rows: plan.merged_rows,
            width: GLM53_VISION_OUT,
            bytes: output_bytes,
        })
    })();

    // If a launch failed, synchronize best-effort before reclaiming buffers
    // that may still be referenced by earlier queued work.
    if result.is_err() {
        let _ = gpu.synchronize(stream);
    }
    let cleanup = allocations.release();
    match (result, cleanup) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(error), Ok(())) => Err(error),
        (Ok(_), Err(error)) => Err(error),
        (Err(error), Err(cleanup)) => {
            Err(error).context(format!("GLM vision cleanup also failed: {cleanup:#}"))
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn project(
    gpu: &dyn GpuBackend,
    catalog: &Glm53Exl3VisionCatalog,
    name: &str,
    input: Glm53Exl3Buffer,
    output: Glm53Exl3Buffer,
    rows: u32,
    scratch: ProjectionScratch,
    stream: u64,
) -> Result<()> {
    let linear = catalog
        .linear(name)
        .with_context(|| format!("missing GLM vision projection {name}"))?;
    let prepared = linear.prepare_bf16(gpu, rows)?;
    let plan = prepared.plan();
    ensure!(
        input.bytes == plan.input_bytes && output.bytes == plan.output_bytes,
        "GLM vision projection activation extent drift at {name}"
    );
    prepared.launch(
        gpu,
        Glm53Exl3Bf16LinearBuffers {
            input_bf16: input,
            output_bf16: output,
            input_f16: prefix(scratch.input_f16, plan.input_bytes)?,
            output_f16: prefix(scratch.output_f16, plan.output_bytes)?,
            locks_i32: scratch.locks_i32,
            input_hadamard_f16: prefix(scratch.input_hadamard_f16, plan.input_bytes)?,
        },
        stream,
    )
}

fn raw(catalog: &Glm53Exl3VisionCatalog, name: &str) -> Result<Glm53Exl3Buffer> {
    let tensor: &Glm53Exl3VisionRawTensor = catalog
        .raw(name)
        .with_context(|| format!("missing GLM vision tensor {name}"))?;
    Ok(Glm53Exl3Buffer {
        ptr: tensor.ptr(),
        bytes: tensor.bytes(),
    })
}

fn prefix(buffer: Glm53Exl3Buffer, bytes: usize) -> Result<Glm53Exl3Buffer> {
    ensure!(
        buffer.ptr != DevicePtr::NULL && buffer.bytes >= bytes,
        "GLM vision projection scratch is too small"
    );
    Ok(Glm53Exl3Buffer {
        ptr: buffer.ptr,
        bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_contract_is_merged_bf16_4096() {
        let plan = Glm53VisionPlan::new(8, 8).unwrap();
        assert_eq!(plan.merged_rows, 16);
        assert_eq!(
            plan.bytes(plan.merged_rows, GLM53_VISION_OUT, 2).unwrap(),
            131_072
        );
    }
}
