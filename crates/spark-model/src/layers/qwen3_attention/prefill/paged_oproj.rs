// SPDX-License-Identifier: AGPL-3.0-only

//! Section 10 of `prefill_attention_paged`: O-projection GEMM
//! `[N, nq*hd] → [N, h]`. 6-way quantization dispatch (FP8 transposed,
//! FP8, FP8 col-scale, NVFP4 transposed, BF16 dense, NVFP4 default).
//! Extracted from `paged.rs` to keep that file under 500 LoC.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_attention_paged_oproj(
        &self,
        attn_out: DevicePtr,
        n: u32,
        h: u32,
        nq: u32,
        hd: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        if let Some(output) =
            self.try_flashinfer_projection_output(attn_out, n, h, nq, hd, ctx, stream)?
        {
            return Ok(output);
        }

        let o_out = ctx.buffers.norm_output();
        if let Some(ref fp8t) = self.o_fp8w_t {
            ops::w8a16_gemm_t(
                ctx.gpu,
                self.w8a16_gemm_t_k,
                attn_out,
                fp8t.weight_t,
                fp8t.scale_t,
                o_out,
                n,
                h,
                nq * hd,
                stream,
            )?;
        } else if self.o_weight.as_ref().and_then(|w| w.as_fp8()).is_some()
            && self.w8a16_gemm_k.0 != 0
        {
            let fp8w = self.o_weight.as_ref().and_then(|w| w.as_fp8()).unwrap();
            ops::w8a16_gemm(
                ctx.gpu,
                self.w8a16_gemm_k,
                attn_out,
                fp8w.weight,
                fp8w.row_scale,
                o_out,
                n,
                h,
                nq * hd,
                stream,
            )?;
        } else if let Some(fp8) = self.o_fp8 {
            use crate::layers::PrefillProjectionPipeRoute as Route;
            match crate::layers::prefill_fp8_w8_route(
                crate::layers::prefill_fp8_w8_enabled(), n, h, nq * hd, self.fp8_gemm_t_w8_k.0 != 0,
            ) {
                Route::Complete => {
                    static SEEN: std::sync::Once = std::sync::Once::new();
                    SEEN.call_once(|| tracing::info!("ENGAGED ATLAS_PREFILL_FP8_W8: attention_o fp8 M={n}"));
                    ops::fp8_gemm_t_w8(ctx.gpu, self.fp8_gemm_t_w8_k, attn_out, fp8, o_out, n, h, nq * hd, stream)?;
                    return Ok(o_out);
                }
                Route::Missing => anyhow::bail!("ATLAS_PREFILL_FP8_W8=1 requires fp8_gemm_t_m128n128_w8 (attention_o)"),
                Route::Disabled | Route::Ineligible => {}
            }
            if n > 128 {
                ops::fp8_gemm_n128_m128(
                    ctx.gpu,
                    self.fp8_gemm_t_m128_k,
                    attn_out,
                    fp8,
                    o_out,
                    n,
                    h,
                    nq * hd,
                    stream,
                )?;
            } else {
                ops::fp8_gemm_n128(
                    ctx.gpu,
                    self.fp8_gemm_k,
                    attn_out,
                    fp8,
                    o_out,
                    n,
                    h,
                    nq * hd,
                    stream,
                )?;
            }
        } else if let Some(nvfp4_t) = self
            .o_nvfp4_t
            .as_ref()
            .filter(|_| crate::layers::prefill_proj_fast_enabled())
        {
            // Correctness lever: when the transposed-TC prefill gate is off,
            // fall through to the non-transposed M64 `w4a16_gemm` below
            // (same class as ATLAS_PREFILL_FFN_FAST=0).
            if n > 128 {
                self.w4a16_gemm_m128_dispatch(
                    ctx.gpu,
                    attn_out,
                    nvfp4_t,
                    o_out,
                    n,
                    h,
                    nq * hd,
                    stream,
                )?;
            } else if n <= 32
                && self.w4a16_gemm_t_m16_k.0 != 0
                && crate::layers::tc_nvfp4_m16_enabled()
            {
                ops::w4a16_gemm_n128_m16(
                    ctx.gpu,
                    self.w4a16_gemm_t_m16_k,
                    attn_out,
                    nvfp4_t,
                    o_out,
                    n,
                    h,
                    nq * hd,
                    stream,
                )?;
            } else {
                ops::w4a16_gemm_n128(
                    ctx.gpu,
                    self.w4a16_gemm_t_k,
                    attn_out,
                    nvfp4_t,
                    o_out,
                    n,
                    h,
                    nq * hd,
                    stream,
                )?;
            }
        } else if let Some(o_bf16) = self.o_dense_bf16.as_ref() {
            // BF16 dense fallback (Gemma-4 dense per Nvidia ModelOpt's
            // ignore list — all self_attn projections must stay BF16).
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_k,
                attn_out,
                o_bf16,
                o_out,
                n,
                h,
                nq * hd,
                stream,
            )?;
        } else {
            self.exact_prefill_projection(
                ctx.gpu,
                "attention_o",
                attn_out,
                &self.attn.o_proj,
                o_out,
                n,
                h,
                nq * hd,
                stream,
            )?;
        }
        Ok(o_out)
    }
}
