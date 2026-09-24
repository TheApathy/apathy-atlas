// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_PREFILL_FFN_CUBLASLT_FP8=1`: the prefill FFN on cuBLASLt FP8.
//!
//! The FlashInfer route quantizes FFN activations to NVFP4 (W4A4), which
//! measured +2.7% teacher-forced NLL against BF16 activations. This route
//! keeps the `ATLAS_PREFILL_FFN_FAST` arithmetic class instead: weights are
//! the exact e4m3 bytes the W4A16 kernel dequantizes to (materialized once
//! per layer on first use), activations are cast to e4m3, and products
//! accumulate in FP32 with a BF16 result. The cuBLASLt algorithm is pinned
//! per (N, K) at a reference M, with split-K disabled, so a row's output does
//! not depend on how many rows share its chunk.

use anyhow::Result;
use spark_runtime::cublaslt::{GemmDtype, gemm_act_weight_t_typed_pinned};
use spark_runtime::gpu::DevicePtr;

use super::DenseFfnLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

/// M the pinned algorithm is chosen at; reused for every M.
const PIN_REF_M: u32 = 2048;

impl DenseFfnLayer {
    /// Returns Ok(false) when the route is off or unavailable, so the caller
    /// keeps its existing dispatch.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_forward_prefill_cublaslt_fp8(
        &self,
        input: DevicePtr,
        m: u32,
        h: u32,
        inter: u32,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        if !crate::layers::prefill_ffn_cublaslt_fp8_enabled()
            || self.fp8_cast_k.0 == 0
            || self.fp8_dequant_k.0 == 0
            || !h.is_multiple_of(128)
            || !inter.is_multiple_of(128)
        {
            return Ok(false);
        }
        let (Some(gt), Some(ut), Some(dt)) = (
            self.gate_proj_t.as_ref(),
            self.up_proj_t.as_ref(),
            self.down_proj_t.as_ref(),
        ) else {
            return Ok(false);
        };
        let Some([wg, wu, wd]) = self.fp8_weights(ctx, [gt, ut, dt], h, inter, stream)? else {
            return Ok(false);
        };
        let Some(act) = self.fp8_act_scratch(ctx, m as usize * inter.max(h) as usize)? else {
            return Ok(false);
        };
        let gemm = |a: DevicePtr, lda: u32, w: DevicePtr, out: DevicePtr, n: u32, k: u32| {
            gemm_act_weight_t_typed_pinned(
                a.0,
                lda,
                w.0,
                out.0,
                n,
                m,
                n,
                k,
                GemmDtype::E4m3,
                GemmDtype::Bf16,
                true,
                PIN_REF_M,
                stream,
            )
        };
        ops::cast_bf16_to_e4m3(ctx.gpu, self.fp8_cast_k, input, act, m as u64 * h as u64, stream)?;
        gemm(act, h, wg, gate_out, inter, h)?;
        gemm(act, h, wu, up_out, inter, h)?;
        ops::silu_mul(ctx.gpu, self.act_mul, gate_out, up_out, gate_out, m * inter, stream)?;
        ops::cast_bf16_to_e4m3(ctx.gpu, self.fp8_cast_k, gate_out, act, m as u64 * inter as u64, stream)?;
        gemm(act, inter, wd, ctx.buffers.moe_output(), h, inter)?;
        static SEEN: std::sync::Once = std::sync::Once::new();
        SEEN.call_once(|| {
            tracing::info!("ENGAGED ATLAS_PREFILL_FFN_CUBLASLT_FP8: M={m} H={h} inter={inter}")
        });
        Ok(true)
    }

    /// e4m3 copies of gate/up/down, materialized on first use.
    fn fp8_weights(
        &self,
        ctx: &ForwardContext,
        src: [&crate::weight_map::QuantizedWeight; 3],
        h: u32,
        inter: u32,
        stream: u64,
    ) -> Result<Option<[DevicePtr; 3]>> {
        let mut slot = self
            .fp8_weights
            .lock()
            .map_err(|_| anyhow::anyhow!("FFN FP8 weight slot poisoned"))?;
        if let Some(w) = *slot {
            return Ok(Some(w));
        }
        // [N, K] per projection: gate/up are [inter, h], down is [h, inter].
        let shapes = [(inter, h), (inter, h), (h, inter)];
        let mut out = [DevicePtr::NULL; 3];
        for (i, (q, (n, k))) in src.iter().zip(shapes).enumerate() {
            let ptr = ctx.gpu.alloc(n as usize * k as usize)?;
            ops::dequant_nvfp4_to_e4m3(
                ctx.gpu,
                self.fp8_dequant_k,
                q.weight,
                q.weight_scale,
                ptr,
                q.weight_scale_2,
                n,
                k,
                stream,
            )?;
            out[i] = ptr;
        }
        *slot = Some(out);
        Ok(Some(out))
    }

    /// Shared e4m3 activation scratch; grows, never shrinks (the old buffer is
    /// not freed: GB10 UVM frees can disturb neighbouring allocations).
    fn fp8_act_scratch(&self, ctx: &ForwardContext, bytes: usize) -> Result<Option<DevicePtr>> {
        let mut slot = self
            .fp8_act_scratch
            .lock()
            .map_err(|_| anyhow::anyhow!("FFN FP8 scratch slot poisoned"))?;
        if let Some((ptr, cap)) = *slot
            && cap >= bytes
        {
            return Ok(Some(ptr));
        }
        let ptr = ctx.gpu.alloc(bytes)?;
        *slot = Some((ptr, bytes));
        Ok(Some(ptr))
    }
}
