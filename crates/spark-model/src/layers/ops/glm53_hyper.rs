// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::GgmlIqBuffer;
use crate::layers::glm53_target_schedule::GLM53_STREAM_ROUND_TO_BF16;

const HIDDEN: u32 = 4096;
const HC: u32 = 4;
const MIX: u32 = 24;
const SINKHORN_ITERS: u32 = 20;
const THREADS: u32 = 256;
const NORM_EPS: f32 = 1.0e-5;
const HC_EPS: f32 = 1.0e-6;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Glm53HyperPlan {
    pub tokens: u32,
    pub hidden_bytes: usize,
    pub streams_bytes: usize,
    pub function_bytes: usize,
    pub base_bytes: usize,
    pub scale_bytes: usize,
    pub post_bytes: usize,
    pub comb_bytes: usize,
    pub norm_weight_bytes: usize,
}

impl Glm53HyperPlan {
    pub fn new(tokens: u32, hidden: u32, hc: u32, sinkhorn_iters: u32) -> Result<Self> {
        if tokens == 0 {
            bail!("GLM mHC requires at least one token");
        }
        if hidden != HIDDEN || hc != HC || sinkhorn_iters != SINKHORN_ITERS {
            bail!("GLM mHC requires exact H4096/hc4/Sinkhorn20 geometry");
        }
        let bytes = |values: usize, width: usize| -> Result<usize> {
            values
                .checked_mul(width)
                .context("GLM mHC byte count overflow")
        };
        let tokens = usize::try_from(tokens)?;
        let hidden = usize::try_from(hidden)?;
        let hc = usize::try_from(hc)?;
        Ok(Self {
            tokens: u32::try_from(tokens)?,
            hidden_bytes: bytes(
                tokens.checked_mul(hidden).context("GLM hidden overflow")?,
                2,
            )?,
            // The mHC residual streams are F32, not BF16. They are the one
            // buffer re-read and rewritten once per layer across 45 layers, and
            // the fold that reads them amplifies its input error ~1.3x/layer,
            // so the two bf16 roundings a layer used to spend here dominated the
            // divergence. Everything else on this path stays BF16.
            streams_bytes: bytes(
                tokens
                    .checked_mul(hc)
                    .and_then(|values| values.checked_mul(hidden))
                    .context("GLM stream element overflow")?,
                4,
            )?,
            function_bytes: bytes(
                usize::try_from(MIX)?
                    .checked_mul(hc)
                    .and_then(|values| values.checked_mul(hidden))
                    .context("GLM mHC function element overflow")?,
                4,
            )?,
            base_bytes: bytes(usize::try_from(MIX)?, 4)?,
            scale_bytes: 3 * 4,
            post_bytes: bytes(tokens.checked_mul(hc).context("GLM post overflow")?, 2)?,
            comb_bytes: bytes(
                tokens
                    .checked_mul(hc)
                    .and_then(|values| values.checked_mul(hc))
                    .context("GLM comb overflow")?,
                2,
            )?,
            norm_weight_bytes: bytes(hidden, 4)?,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Glm53HyperPreBuffers {
    pub streams_f32: GgmlIqBuffer,
    pub function_f32: GgmlIqBuffer,
    pub base_f32: GgmlIqBuffer,
    pub scale_f32: GgmlIqBuffer,
    pub collapsed_bf16: GgmlIqBuffer,
    pub post_bf16: GgmlIqBuffer,
    pub comb_bf16: GgmlIqBuffer,
    /// Prompt-scope F32 staging for the hoisted mixing GEMM (NULL outside it).
    pub mixed_f32: GgmlIqBuffer,
}

#[derive(Debug, Clone, Copy)]
pub struct Glm53HyperPostBuffers {
    pub block_output_bf16: GgmlIqBuffer,
    pub residual_streams_f32: GgmlIqBuffer,
    pub post_bf16: GgmlIqBuffer,
    pub comb_bf16: GgmlIqBuffer,
    pub output_streams_f32: GgmlIqBuffer,
}

pub struct Glm53HyperKernels {
    expand: KernelHandle,
    pre: KernelHandle,
    post: KernelHandle,
    mean: KernelHandle,
    norm: KernelHandle,
}

impl Glm53HyperKernels {
    pub fn load(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            expand: gpu.kernel("glm53_hyper", "atlas_glm53_hc_expand")?,
            pre: gpu.kernel("glm53_hyper", "atlas_glm53_hc_pre")?,
            post: gpu.kernel("glm53_hyper", "atlas_glm53_hc_post")?,
            mean: gpu.kernel("glm53_hyper", "atlas_glm53_hc_mean")?,
            norm: gpu.kernel("glm53_hyper", "atlas_glm53_rms_norm")?,
        })
    }

    pub fn expand(
        &self,
        gpu: &dyn GpuBackend,
        plan: Glm53HyperPlan,
        hidden_bf16: GgmlIqBuffer,
        streams_f32: GgmlIqBuffer,
        stream: u64,
    ) -> Result<()> {
        validate_set(&[
            ("hidden", hidden_bf16, plan.hidden_bytes),
            ("streams", streams_f32, plan.streams_bytes),
        ])?;
        KernelLaunch::new(gpu, self.expand)
            .grid([plan.tokens, 1, 1])
            .block([THREADS, 1, 1])
            .arg_ptr(hidden_bf16.ptr)
            .arg_ptr(streams_f32.ptr)
            .arg_u32(HIDDEN)
            .arg_u32(HC)
            .arg_u32(GLM53_STREAM_ROUND_TO_BF16)
            .launch(stream)
    }

    pub fn pre(
        &self,
        gpu: &dyn GpuBackend,
        plan: Glm53HyperPlan,
        buffers: Glm53HyperPreBuffers,
        stream: u64,
    ) -> Result<()> {
        validate_set(&[
            ("streams", buffers.streams_f32, plan.streams_bytes),
            ("function", buffers.function_f32, plan.function_bytes),
            ("base", buffers.base_f32, plan.base_bytes),
            ("scale", buffers.scale_f32, plan.scale_bytes),
            ("collapsed", buffers.collapsed_bf16, plan.hidden_bytes),
            ("post", buffers.post_bf16, plan.post_bytes),
            ("comb", buffers.comb_bf16, plan.comb_bytes),
        ])?;
        if hc_pre_gemm_active(plan.tokens)? && !buffers.mixed_f32.ptr.is_null() {
            return self.pre_gemm(gpu, plan, buffers, stream);
        }
        KernelLaunch::new(gpu, self.pre)
            .grid([plan.tokens, 1, 1])
            .block([THREADS, 1, 1])
            .arg_ptr(buffers.streams_f32.ptr)
            .arg_ptr(buffers.function_f32.ptr)
            .arg_ptr(buffers.scale_f32.ptr)
            .arg_ptr(buffers.base_f32.ptr)
            .arg_ptr(buffers.collapsed_bf16.ptr)
            .arg_ptr(buffers.post_bf16.ptr)
            .arg_ptr(buffers.comb_bf16.ptr)
            .arg_u32(HIDDEN)
            .arg_u32(HC)
            .arg_u32(SINKHORN_ITERS)
            .arg_f32(NORM_EPS)
            .arg_f32(HC_EPS)
            .launch(stream)
    }

    pub fn post(
        &self,
        gpu: &dyn GpuBackend,
        plan: Glm53HyperPlan,
        buffers: Glm53HyperPostBuffers,
        stream: u64,
    ) -> Result<()> {
        let mut checked = vec![
            ("block output", buffers.block_output_bf16, plan.hidden_bytes),
            (
                "residual streams",
                buffers.residual_streams_f32,
                plan.streams_bytes,
            ),
            ("post", buffers.post_bf16, plan.post_bytes),
            ("comb", buffers.comb_bf16, plan.comb_bytes),
        ];
        if buffers.output_streams_f32.ptr != buffers.residual_streams_f32.ptr {
            checked.push((
                "output streams",
                buffers.output_streams_f32,
                plan.streams_bytes,
            ));
        } else if buffers.output_streams_f32.bytes != plan.streams_bytes {
            bail!("GLM mHC in-place output extent mismatch");
        }
        validate_set(&checked)?;
        if hc_post_prompt_active(plan.tokens)? {
            // Same fold, same fmaf order, same single rounding; 2D grid + float4
            // so the memory-bound fold is not pinned to one block per SM.
            let kernel = gpu.kernel("glm53_prompt_glue", "atlas_glm53_hc_post_prompt")?;
            return KernelLaunch::new(gpu, kernel)
                .grid([HIDDEN / (THREADS * 4), plan.tokens, 1])
                .block([THREADS, 1, 1])
                .arg_ptr(buffers.block_output_bf16.ptr)
                .arg_ptr(buffers.residual_streams_f32.ptr)
                .arg_ptr(buffers.post_bf16.ptr)
                .arg_ptr(buffers.comb_bf16.ptr)
                .arg_ptr(buffers.output_streams_f32.ptr)
                .arg_u32(HIDDEN)
                .arg_u32(HC)
                .arg_u32(plan.tokens)
                .arg_u32(GLM53_STREAM_ROUND_TO_BF16)
                .launch(stream);
        }
        KernelLaunch::new(gpu, self.post)
            .grid([plan.tokens, 1, 1])
            .block([THREADS, 1, 1])
            .arg_ptr(buffers.block_output_bf16.ptr)
            .arg_ptr(buffers.residual_streams_f32.ptr)
            .arg_ptr(buffers.post_bf16.ptr)
            .arg_ptr(buffers.comb_bf16.ptr)
            .arg_ptr(buffers.output_streams_f32.ptr)
            .arg_u32(HIDDEN)
            .arg_u32(HC)
            .arg_u32(GLM53_STREAM_ROUND_TO_BF16)
            .launch(stream)
    }

    pub fn mean(
        &self,
        gpu: &dyn GpuBackend,
        plan: Glm53HyperPlan,
        streams_f32: GgmlIqBuffer,
        hidden_bf16: GgmlIqBuffer,
        stream: u64,
    ) -> Result<()> {
        validate_set(&[
            ("streams", streams_f32, plan.streams_bytes),
            ("mean", hidden_bf16, plan.hidden_bytes),
        ])?;
        KernelLaunch::new(gpu, self.mean)
            .grid([plan.tokens, 1, 1])
            .block([THREADS, 1, 1])
            .arg_ptr(streams_f32.ptr)
            .arg_ptr(hidden_bf16.ptr)
            .arg_u32(HIDDEN)
            .arg_u32(HC)
            .launch(stream)
    }

    pub fn norm(
        &self,
        gpu: &dyn GpuBackend,
        plan: Glm53HyperPlan,
        input_bf16: GgmlIqBuffer,
        weight_f32: GgmlIqBuffer,
        output_bf16: GgmlIqBuffer,
        stream: u64,
    ) -> Result<()> {
        let mut checked = vec![
            ("norm input", input_bf16, plan.hidden_bytes),
            ("norm weight", weight_f32, plan.norm_weight_bytes),
        ];
        if output_bf16.ptr != input_bf16.ptr {
            checked.push(("norm output", output_bf16, plan.hidden_bytes));
        } else if output_bf16.bytes != plan.hidden_bytes {
            bail!("GLM in-place norm output extent mismatch");
        }
        validate_set(&checked)?;
        KernelLaunch::new(gpu, self.norm)
            .grid([plan.tokens, 1, 1])
            .block([1024, 1, 1])
            .arg_ptr(input_bf16.ptr)
            .arg_ptr(weight_f32.ptr)
            .arg_ptr(output_bf16.ptr)
            .arg_u32(HIDDEN)
            .arg_f32(NORM_EPS)
            .launch(stream)
    }
}

fn validate_set(named: &[(&str, GgmlIqBuffer, usize)]) -> Result<()> {
    let mut ranges = Vec::with_capacity(named.len());
    for &(name, buffer, expected) in named {
        if buffer.bytes != expected || buffer.ptr == DevicePtr::NULL {
            bail!("GLM mHC {name} buffer is null or has the wrong extent");
        }
        let end = buffer
            .ptr
            .0
            .checked_add(u64::try_from(buffer.bytes)?)
            .with_context(|| format!("GLM mHC {name} address overflow"))?;
        ranges.push((buffer.ptr.0, end));
    }
    for left in 0..ranges.len() {
        for right in left + 1..ranges.len() {
            if ranges[left].0 < ranges[right].1 && ranges[right].0 < ranges[left].1 {
                bail!("GLM mHC device buffers overlap");
            }
        }
    }
    Ok(())
}


/// `ATLAS_GLM53_HC_PRE_GEMM=1`: prompt-scope mHC pre with the 24 mixing dots
/// hoisted into one cuBLASLt f32 GEMM `[tokens,16384] x [24,16384]^T`.
/// 0 = off, 1 = cuBLASLt f32 GEMM, 2 = fused mix+rms kernel (4 tokens/block).
fn hc_pre_mode() -> Result<u8> {
    use std::sync::OnceLock;
    static MODE: OnceLock<std::result::Result<u8, String>> = OnceLock::new();
    MODE.get_or_init(|| match std::env::var("ATLAS_GLM53_HC_PRE_GEMM") {
        Ok(v) if v == "1" => Ok(1),
        Ok(v) if v == "2" => Ok(2),
        Ok(v) if v == "0" => Ok(0),
        Ok(other) => Err(format!("ATLAS_GLM53_HC_PRE_GEMM must be 0, 1 or 2, got {other:?}")),
        Err(std::env::VarError::NotPresent) => Ok(0),
        Err(e) => Err(format!("ATLAS_GLM53_HC_PRE_GEMM: {e}")),
    })
    .clone()
    .map_err(anyhow::Error::msg)
}

/// `ATLAS_GLM53_HC_POST_PROMPT=1`: prompt-scope mHC output fold on a 2D grid
/// with float4 column vectors. Arithmetic is identical to `atlas_glm53_hc_post`
/// (bit-exact); only the block decomposition and occupancy change.
fn hc_post_prompt_active(tokens: u32) -> Result<bool> {
    use std::sync::OnceLock;
    static ON: OnceLock<std::result::Result<bool, String>> = OnceLock::new();
    let on = ON
        .get_or_init(|| match std::env::var("ATLAS_GLM53_HC_POST_PROMPT") {
            Ok(v) if v == "1" => Ok(true),
            Ok(v) if v == "0" => Ok(false),
            Ok(other) => Err(format!(
                "ATLAS_GLM53_HC_POST_PROMPT must be 0 or 1, got {other:?}"
            )),
            Err(std::env::VarError::NotPresent) => Ok(false),
            Err(e) => Err(format!("ATLAS_GLM53_HC_POST_PROMPT: {e}")),
        })
        .clone()
        .map_err(anyhow::Error::msg)?;
    // HIDDEN must tile evenly into THREADS*4 columns per block.
    Ok(on
        && tokens >= 144
        && HIDDEN % (THREADS * 4) == 0
        && super::glm53_layer_major_prefill_active())
}

fn hc_pre_gemm_active(tokens: u32) -> Result<bool> {
    Ok(hc_pre_mode()? != 0 && tokens >= 144 && super::glm53_layer_major_prefill_active())
}

impl Glm53HyperKernels {
    fn pre_gemm(
        &self,
        gpu: &dyn GpuBackend,
        plan: Glm53HyperPlan,
        buffers: Glm53HyperPreBuffers,
        stream: u64,
    ) -> Result<()> {
        let mixed_bytes = usize::try_from(plan.tokens)? * MIX as usize * 4;
        anyhow::ensure!(
            buffers.mixed_f32.bytes >= mixed_bytes,
            "GLM hc_pre mixed scratch too small"
        );
        if hc_pre_mode()? == 2 {
            // mixed_raw [tokens,24] f32 then inv_rms [tokens] f32 in the same scratch.
            let inv_rms = buffers.mixed_f32.ptr.offset(mixed_bytes);
            anyhow::ensure!(
                buffers.mixed_f32.bytes >= mixed_bytes + usize::try_from(plan.tokens)? * 4,
                "GLM hc_mix scratch too small"
            );
            let mix = gpu.kernel("glm53_prompt_glue", "atlas_glm53_hc_mix_rms")?;
            KernelLaunch::new(gpu, mix)
                .grid([plan.tokens.div_ceil(4), 1, 1])
                .block([THREADS, 1, 1])
                .arg_ptr(buffers.streams_f32.ptr)
                .arg_ptr(buffers.function_f32.ptr)
                .arg_ptr(buffers.mixed_f32.ptr)
                .arg_ptr(inv_rms)
                .arg_u32(plan.tokens)
                .arg_u32(HIDDEN)
                .arg_u32(HC)
                .arg_f32(NORM_EPS)
                .launch(stream)?;
            let kernel = gpu.kernel("glm53_prompt_glue", "atlas_glm53_hc_pre_mixed2")?;
            return KernelLaunch::new(gpu, kernel)
                .grid([plan.tokens, 1, 1])
                .block([THREADS, 1, 1])
                .arg_ptr(buffers.streams_f32.ptr)
                .arg_ptr(buffers.mixed_f32.ptr)
                .arg_ptr(inv_rms)
                .arg_ptr(buffers.scale_f32.ptr)
                .arg_ptr(buffers.base_f32.ptr)
                .arg_ptr(buffers.collapsed_bf16.ptr)
                .arg_ptr(buffers.post_bf16.ptr)
                .arg_ptr(buffers.comb_bf16.ptr)
                .arg_u32(HIDDEN)
                .arg_u32(HC)
                .arg_u32(SINKHORN_ITERS)
                .arg_f32(NORM_EPS)
                .arg_f32(HC_EPS)
                .launch(stream);
        }
        for (first, rows) in super::glm53_gemm_row_slices(plan.tokens) {
            spark_runtime::cublaslt::f32_gemm_act_weight_t(
                buffers.streams_f32.ptr.0 + u64::from(first) * u64::from(HC * HIDDEN) * 4,
                buffers.function_f32.ptr.0,
                buffers.mixed_f32.ptr.0 + u64::from(first) * u64::from(MIX) * 4,
                rows,
                MIX,
                HC * HIDDEN,
                stream,
            )?;
        }
        let kernel = gpu.kernel("glm53_prompt_glue", "atlas_glm53_hc_pre_mixed")?;
        KernelLaunch::new(gpu, kernel)
            .grid([plan.tokens, 1, 1])
            .block([THREADS, 1, 1])
            .arg_ptr(buffers.streams_f32.ptr)
            .arg_ptr(buffers.mixed_f32.ptr)
            .arg_ptr(buffers.scale_f32.ptr)
            .arg_ptr(buffers.base_f32.ptr)
            .arg_ptr(buffers.collapsed_bf16.ptr)
            .arg_ptr(buffers.post_bf16.ptr)
            .arg_ptr(buffers.comb_bf16.ptr)
            .arg_u32(HIDDEN)
            .arg_u32(HC)
            .arg_u32(SINKHORN_ITERS)
            .arg_f32(NORM_EPS)
            .arg_f32(HC_EPS)
            .launch(stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::mock::MockGpuBackend;

    #[test]
    fn exact_glm_hyper_geometry_is_pinned() {
        let plan = Glm53HyperPlan::new(2, 4096, 4, 20).unwrap();
        assert_eq!(plan.hidden_bytes, 2 * 4096 * 2);
        assert_eq!(plan.streams_bytes, 2 * 4 * 4096 * 4);
        assert_eq!(plan.function_bytes, 24 * 4 * 4096 * 4);
        assert_eq!(plan.post_bytes, 2 * 4 * 2);
        assert_eq!(plan.comb_bytes, 2 * 4 * 4 * 2);
        assert!(Glm53HyperPlan::new(0, 4096, 4, 20).is_err());
        assert!(Glm53HyperPlan::new(1, 4096, 4, 19).is_err());
    }

    #[test]
    fn all_five_launches_validate_extents_and_exact_in_place_aliases() {
        let gpu = MockGpuBackend::new();
        let kernels = Glm53HyperKernels::load(&gpu).unwrap();
        let plan = Glm53HyperPlan::new(1, 4096, 4, 20).unwrap();
        let at = |ptr, bytes| GgmlIqBuffer {
            ptr: DevicePtr(ptr),
            bytes,
        };
        let hidden = at(0x10_0000, plan.hidden_bytes);
        let streams = at(0x20_0000, plan.streams_bytes);
        let function = at(0x30_0000, plan.function_bytes);
        let base = at(0x50_0000, plan.base_bytes);
        let scale = at(0x51_0000, plan.scale_bytes);
        let collapsed = at(0x52_0000, plan.hidden_bytes);
        let post = at(0x53_0000, plan.post_bytes);
        let comb = at(0x54_0000, plan.comb_bytes);
        let norm_weight = at(0x55_0000, plan.norm_weight_bytes);
        assert!(
            kernels
                .pre(
                    &gpu,
                    plan,
                    Glm53HyperPreBuffers {
                        mixed_f32: GgmlIqBuffer { ptr: DevicePtr::NULL, bytes: 0 },
                        streams_f32: streams,
                        function_f32: function,
                        base_f32: base,
                        scale_f32: scale,
                        collapsed_bf16: at(function.ptr.0, plan.hidden_bytes),
                        post_bf16: post,
                        comb_bf16: comb,
                    },
                    0,
                )
                .is_err()
        );
        assert_eq!(gpu.launch_count(), 0);
        kernels.expand(&gpu, plan, hidden, streams, 0).unwrap();
        kernels
            .pre(
                &gpu,
                plan,
                Glm53HyperPreBuffers {
                    mixed_f32: GgmlIqBuffer { ptr: DevicePtr::NULL, bytes: 0 },
                    streams_f32: streams,
                    function_f32: function,
                    base_f32: base,
                    scale_f32: scale,
                    collapsed_bf16: collapsed,
                    post_bf16: post,
                    comb_bf16: comb,
                },
                0,
            )
            .unwrap();
        kernels
            .post(
                &gpu,
                plan,
                Glm53HyperPostBuffers {
                    block_output_bf16: collapsed,
                    residual_streams_f32: streams,
                    post_bf16: post,
                    comb_bf16: comb,
                    output_streams_f32: streams,
                },
                0,
            )
            .unwrap();
        kernels.mean(&gpu, plan, streams, hidden, 0).unwrap();
        kernels
            .norm(&gpu, plan, hidden, norm_weight, hidden, 0)
            .unwrap();
        assert_eq!(gpu.launch_count(), 5);
    }
}
