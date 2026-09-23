// SPDX-License-Identifier: AGPL-3.0-only

//! Exact host launch contract for GLM-5.3-Flash's distinct vision tower.

use std::ffi::OsStr;

use anyhow::{Context, Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::Glm53Exl3Buffer;

pub const GLM53_VISION_HIDDEN: u32 = 1024;
pub const GLM53_VISION_HEADS: u32 = 16;
pub const GLM53_VISION_HEAD_DIM: u32 = 64;
pub const GLM53_VISION_INTERMEDIATE: u32 = 4096;
pub const GLM53_VISION_OUT: u32 = 4096;
pub const GLM53_VISION_MERGER_INTERMEDIATE: u32 = 10240;
pub const GLM53_VISION_PATCH_DIM: u32 = 3 * 2 * 14 * 14;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Glm53VisionPlan {
    pub rows: u32,
    pub merged_rows: u32,
    pub grid_height: u32,
    pub grid_width: u32,
}

impl Glm53VisionPlan {
    pub fn new(grid_height: usize, grid_width: usize) -> Result<Self> {
        ensure!(
            grid_height >= 2 && grid_width >= 2 && grid_height % 2 == 0 && grid_width % 2 == 0,
            "GLM vision patch grid must be nonzero and divisible by merge size 2"
        );
        let rows = u32::try_from(
            grid_height
                .checked_mul(grid_width)
                .context("GLM vision patch count overflow")?,
        )?;
        ensure!(
            rows <= 32_000,
            "GLM vision patch count exceeds exact image cap"
        );
        Ok(Self {
            rows,
            merged_rows: rows / 4,
            grid_height: u32::try_from(grid_height)?,
            grid_width: u32::try_from(grid_width)?,
        })
    }

    pub fn bytes(self, rows: u32, width: u32, element_bytes: usize) -> Result<usize> {
        usize::try_from(u64::from(rows) * u64::from(width))?
            .checked_mul(element_bytes)
            .context("GLM vision buffer extent overflow")
    }
}

pub struct Glm53VisionKernels {
    patch_embed: KernelHandle,
    rmsnorm: KernelHandle,
    qkv_prepare: KernelHandle,
    attention: KernelHandle,
    flash_attention: Option<KernelHandle>,
    bias_residual: KernelHandle,
    swiglu_bias: KernelHandle,
    post_downsample: KernelHandle,
    layernorm_gelu: KernelHandle,
}

impl Glm53VisionKernels {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        let kernel = |name| gpu.kernel("glm53_vision", name);
        let flash_attention =
            if parse_flash_attention(std::env::var_os("ATLAS_GLM53_VISION_FLASH_ATTN").as_deref())?
            {
                Some(gpu.kernel("glm53_vision_flash_attention", "inferspark_prefill_64")?)
            } else {
                None
            };
        Ok(Self {
            patch_embed: kernel("atlas_glm53_vision_patch_embed")?,
            rmsnorm: kernel("atlas_glm53_vision_rmsnorm")?,
            qkv_prepare: kernel("atlas_glm53_vision_qkv_prepare")?,
            attention: kernel("atlas_glm53_vision_attention")?,
            flash_attention,
            bias_residual: kernel("atlas_glm53_vision_bias_residual")?,
            swiglu_bias: kernel("atlas_glm53_vision_swiglu_bias")?,
            post_downsample: kernel("atlas_glm53_vision_post_downsample")?,
            layernorm_gelu: kernel("atlas_glm53_vision_layernorm_gelu")?,
        })
    }

    pub fn patch_embed(
        &self,
        gpu: &dyn GpuBackend,
        plan: Glm53VisionPlan,
        pixels_f32: Glm53Exl3Buffer,
        weight_f16: Glm53Exl3Buffer,
        bias_f16: Glm53Exl3Buffer,
        output_bf16: Glm53Exl3Buffer,
        stream: u64,
    ) -> Result<()> {
        exact(
            pixels_f32,
            plan.bytes(plan.rows, GLM53_VISION_PATCH_DIM, 4)?,
            "pixels",
        )?;
        exact(
            weight_f16,
            usize::try_from(GLM53_VISION_HIDDEN * GLM53_VISION_PATCH_DIM)? * 2,
            "patch weight",
        )?;
        exact(
            bias_f16,
            usize::try_from(GLM53_VISION_HIDDEN)? * 2,
            "patch bias",
        )?;
        exact(
            output_bf16,
            plan.bytes(plan.rows, GLM53_VISION_HIDDEN, 2)?,
            "patch output",
        )?;
        KernelLaunch::new(gpu, self.patch_embed)
            .grid([plan.rows, 4, 1])
            .block([256, 1, 1])
            .arg_ptr(pixels_f32.ptr)
            .arg_ptr(weight_f16.ptr)
            .arg_ptr(bias_f16.ptr)
            .arg_ptr(output_bf16.ptr)
            .arg_u32(plan.rows)
            .launch(stream)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rmsnorm(
        &self,
        gpu: &dyn GpuBackend,
        input: Glm53Exl3Buffer,
        weight: Glm53Exl3Buffer,
        output: Glm53Exl3Buffer,
        rows: u32,
        width: u32,
        epsilon: f32,
        stream: u64,
    ) -> Result<()> {
        let bytes = usize::try_from(u64::from(rows) * u64::from(width))? * 2;
        exact(input, bytes, "RMS input")?;
        exact(weight, usize::try_from(width)? * 2, "RMS weight")?;
        exact(output, bytes, "RMS output")?;
        ensure!(
            epsilon.is_finite() && epsilon > 0.0,
            "invalid GLM vision RMS epsilon"
        );
        KernelLaunch::new(gpu, self.rmsnorm)
            .grid([rows, 1, 1])
            .block([256, 1, 1])
            .shared_mem(256 * 4)
            .arg_ptr(input.ptr)
            .arg_ptr(weight.ptr)
            .arg_ptr(output.ptr)
            .arg_u32(rows)
            .arg_u32(width)
            .arg_f32(epsilon)
            .launch(stream)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn qkv_prepare(
        &self,
        gpu: &dyn GpuBackend,
        plan: Glm53VisionPlan,
        q_in: Glm53Exl3Buffer,
        k_in: Glm53Exl3Buffer,
        v_in: Glm53Exl3Buffer,
        q_bias: Glm53Exl3Buffer,
        k_bias: Glm53Exl3Buffer,
        v_bias: Glm53Exl3Buffer,
        q_weight: Glm53Exl3Buffer,
        k_weight: Glm53Exl3Buffer,
        q_out: Glm53Exl3Buffer,
        k_out: Glm53Exl3Buffer,
        v_out: Glm53Exl3Buffer,
        stream: u64,
    ) -> Result<()> {
        let hidden = plan.bytes(plan.rows, GLM53_VISION_HIDDEN, 2)?;
        for (name, buffer) in [
            ("q input", q_in),
            ("k input", k_in),
            ("v input", v_in),
            ("q output", q_out),
            ("k output", k_out),
            ("v output", v_out),
        ] {
            exact(buffer, hidden, name)?;
        }
        for (name, buffer) in [("q bias", q_bias), ("k bias", k_bias), ("v bias", v_bias)] {
            exact(buffer, usize::try_from(GLM53_VISION_HIDDEN)? * 2, name)?;
        }
        for (name, buffer) in [("q norm", q_weight), ("k norm", k_weight)] {
            exact(buffer, usize::try_from(GLM53_VISION_HEAD_DIM)? * 2, name)?;
        }
        KernelLaunch::new(gpu, self.qkv_prepare)
            .grid([plan.rows, GLM53_VISION_HEADS, 1])
            .block([GLM53_VISION_HEAD_DIM, 1, 1])
            .arg_ptr(q_in.ptr)
            .arg_ptr(k_in.ptr)
            .arg_ptr(v_in.ptr)
            .arg_ptr(q_bias.ptr)
            .arg_ptr(k_bias.ptr)
            .arg_ptr(v_bias.ptr)
            .arg_ptr(q_weight.ptr)
            .arg_ptr(k_weight.ptr)
            .arg_ptr(q_out.ptr)
            .arg_ptr(k_out.ptr)
            .arg_ptr(v_out.ptr)
            .arg_u32(plan.rows)
            .arg_u32(plan.grid_width)
            .arg_f32(1e-5)
            .launch(stream)
    }

    pub fn attention(
        &self,
        gpu: &dyn GpuBackend,
        plan: Glm53VisionPlan,
        q: Glm53Exl3Buffer,
        k: Glm53Exl3Buffer,
        v: Glm53Exl3Buffer,
        output: Glm53Exl3Buffer,
        stream: u64,
    ) -> Result<()> {
        let bytes = plan.bytes(plan.rows, GLM53_VISION_HIDDEN, 2)?;
        for (name, buffer) in [("q", q), ("k", k), ("v", v), ("attention output", output)] {
            exact(buffer, bytes, name)?;
        }
        if let Some(kernel) = self.flash_attention {
            return KernelLaunch::new(gpu, kernel)
                .grid([GLM53_VISION_HEADS, div_ceil(plan.rows, 64), 1])
                .block([256, 1, 1])
                .arg_ptr(q.ptr)
                .arg_ptr(k.ptr)
                .arg_ptr(v.ptr)
                .arg_ptr(output.ptr)
                .arg_u32(plan.rows)
                .arg_u32(GLM53_VISION_HEADS)
                .arg_u32(GLM53_VISION_HEADS)
                .arg_u32(GLM53_VISION_HEAD_DIM)
                .arg_f32(0.125)
                .arg_u32(0)
                .arg_u32(0)
                .launch(stream);
        }
        KernelLaunch::new(gpu, self.attention)
            .grid([plan.rows, GLM53_VISION_HEADS, 1])
            .block([GLM53_VISION_HEAD_DIM, 1, 1])
            .arg_ptr(q.ptr)
            .arg_ptr(k.ptr)
            .arg_ptr(v.ptr)
            .arg_ptr(output.ptr)
            .arg_u32(plan.rows)
            .launch(stream)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn bias_residual(
        &self,
        gpu: &dyn GpuBackend,
        projected: Glm53Exl3Buffer,
        bias: Glm53Exl3Buffer,
        residual: Glm53Exl3Buffer,
        output: Glm53Exl3Buffer,
        rows: u32,
        width: u32,
        stream: u64,
    ) -> Result<()> {
        let bytes = usize::try_from(u64::from(rows) * u64::from(width))? * 2;
        for (name, buffer) in [
            ("projected", projected),
            ("residual", residual),
            ("sum", output),
        ] {
            exact(buffer, bytes, name)?;
        }
        exact(bias, usize::try_from(width)? * 2, "bias")?;
        KernelLaunch::new(gpu, self.bias_residual)
            .grid([div_ceil(rows * width, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(projected.ptr)
            .arg_ptr(bias.ptr)
            .arg_ptr(residual.ptr)
            .arg_ptr(output.ptr)
            .arg_u32(rows)
            .arg_u32(width)
            .launch(stream)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn swiglu(
        &self,
        gpu: &dyn GpuBackend,
        gate: Glm53Exl3Buffer,
        up: Glm53Exl3Buffer,
        gate_bias: Option<Glm53Exl3Buffer>,
        up_bias: Option<Glm53Exl3Buffer>,
        output: Glm53Exl3Buffer,
        rows: u32,
        width: u32,
        stream: u64,
    ) -> Result<()> {
        ensure!(
            matches!(
                width,
                GLM53_VISION_INTERMEDIATE | GLM53_VISION_MERGER_INTERMEDIATE
            ),
            "invalid GLM vision SwiGLU width"
        );
        let bytes = usize::try_from(u64::from(rows) * u64::from(width))? * 2;
        for (name, buffer) in [("gate", gate), ("up", up), ("SwiGLU output", output)] {
            exact(buffer, bytes, name)?;
        }
        ensure!(
            gate_bias.is_some() == up_bias.is_some(),
            "GLM vision SwiGLU biases must be paired"
        );
        let has_bias = u32::from(gate_bias.is_some());
        let null = Glm53Exl3Buffer {
            ptr: DevicePtr::NULL,
            bytes: 0,
        };
        let gate_bias = gate_bias.unwrap_or(null);
        let up_bias = up_bias.unwrap_or(null);
        if has_bias != 0 {
            exact(gate_bias, usize::try_from(width)? * 2, "gate bias")?;
            exact(up_bias, usize::try_from(width)? * 2, "up bias")?;
        }
        KernelLaunch::new(gpu, self.swiglu_bias)
            .grid([div_ceil(rows * width, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(gate.ptr)
            .arg_ptr(up.ptr)
            .arg_ptr(gate_bias.ptr)
            .arg_ptr(up_bias.ptr)
            .arg_ptr(output.ptr)
            .arg_u32(rows)
            .arg_u32(width)
            .arg_f32(10.0)
            .arg_u32(has_bias)
            .launch(stream)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn post_downsample(
        &self,
        gpu: &dyn GpuBackend,
        plan: Glm53VisionPlan,
        input: Glm53Exl3Buffer,
        norm_weight: Glm53Exl3Buffer,
        conv_weight: Glm53Exl3Buffer,
        conv_bias: Glm53Exl3Buffer,
        output: Glm53Exl3Buffer,
        stream: u64,
    ) -> Result<()> {
        exact(
            input,
            plan.bytes(plan.rows, GLM53_VISION_HIDDEN, 2)?,
            "post input",
        )?;
        exact(
            norm_weight,
            usize::try_from(GLM53_VISION_HIDDEN)? * 2,
            "post norm",
        )?;
        exact(
            conv_weight,
            usize::try_from(GLM53_VISION_OUT * GLM53_VISION_HIDDEN * 4)? * 2,
            "downsample weight",
        )?;
        exact(
            conv_bias,
            usize::try_from(GLM53_VISION_OUT)? * 2,
            "downsample bias",
        )?;
        exact(
            output,
            plan.bytes(plan.merged_rows, GLM53_VISION_OUT, 2)?,
            "downsample output",
        )?;
        KernelLaunch::new(gpu, self.post_downsample)
            .grid([plan.merged_rows, 16, 1])
            .block([256, 1, 1])
            .arg_ptr(input.ptr)
            .arg_ptr(norm_weight.ptr)
            .arg_ptr(conv_weight.ptr)
            .arg_ptr(conv_bias.ptr)
            .arg_ptr(output.ptr)
            .arg_u32(plan.merged_rows)
            .arg_f32(1e-5)
            .launch(stream)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn layernorm_gelu(
        &self,
        gpu: &dyn GpuBackend,
        input: Glm53Exl3Buffer,
        weight: Glm53Exl3Buffer,
        bias: Glm53Exl3Buffer,
        output: Glm53Exl3Buffer,
        rows: u32,
        stream: u64,
    ) -> Result<()> {
        let bytes = usize::try_from(u64::from(rows) * u64::from(GLM53_VISION_OUT))? * 2;
        exact(input, bytes, "merger norm input")?;
        exact(output, bytes, "merger norm output")?;
        exact(
            weight,
            usize::try_from(GLM53_VISION_OUT)? * 2,
            "merger norm weight",
        )?;
        exact(
            bias,
            usize::try_from(GLM53_VISION_OUT)? * 2,
            "merger norm bias",
        )?;
        KernelLaunch::new(gpu, self.layernorm_gelu)
            .grid([rows, 1, 1])
            .block([256, 1, 1])
            .shared_mem(256 * 4)
            .arg_ptr(input.ptr)
            .arg_ptr(weight.ptr)
            .arg_ptr(bias.ptr)
            .arg_ptr(output.ptr)
            .arg_u32(rows)
            .arg_u32(GLM53_VISION_OUT)
            .arg_f32(1e-5)
            .launch(stream)
    }
}

fn parse_flash_attention(value: Option<&OsStr>) -> Result<bool> {
    match value.and_then(OsStr::to_str) {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(_) => bail!("ATLAS_GLM53_VISION_FLASH_ATTN must be exactly 0 or 1"),
    }
}

fn exact(buffer: Glm53Exl3Buffer, bytes: usize, name: &str) -> Result<()> {
    if buffer.ptr == DevicePtr::NULL || buffer.bytes != bytes {
        bail!("GLM vision {name} buffer is null or has the wrong exact extent");
    }
    buffer
        .ptr
        .0
        .checked_add(u64::try_from(bytes)?)
        .with_context(|| format!("GLM vision {name} address overflow"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::ffi::OsStr;

    use super::*;

    #[test]
    fn exact_grid_contract() {
        let plan = Glm53VisionPlan::new(8, 8).unwrap();
        assert_eq!((plan.rows, plan.merged_rows), (64, 16));
        assert!(Glm53VisionPlan::new(7, 8).is_err());
        assert!(Glm53VisionPlan::new(0, 8).is_err());
    }

    #[test]
    fn glm53_vision_flash_attention_contract() {
        assert!(!parse_flash_attention(None).unwrap());
        assert!(!parse_flash_attention(Some(OsStr::new("0"))).unwrap());
        assert!(parse_flash_attention(Some(OsStr::new("1"))).unwrap());
        assert!(parse_flash_attention(Some(OsStr::new("yes"))).is_err());
        let source = include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../kernels/gb10/glm5.3-flash/exl3/glm53_vision_flash_attention.cu"
        ));
        assert!(source.contains("#define HDIM 64"));
        // Upstream's kernel signature, vendored GLM-local: this tree's common
        // inferspark_prefill takes two extra parameters the launcher never passes.
        assert!(source.contains("../iq3/glm53_inferspark_prefill_upstream.cuh"));
        assert!(!source.contains("../../common/inferspark_prefill.cu"));
    }
}
