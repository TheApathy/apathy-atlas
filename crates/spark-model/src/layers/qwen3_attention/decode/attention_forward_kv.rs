// SPDX-License-Identifier: AGPL-3.0-only

//! K + V projection branch of `attention_forward` (decode path). Picks
//! one of: MLA-skip (K/V already produced), FP8 native dual GEMV, fused
//! NVFP4 dual `w4a16_gemv_dual`, or per-projection NVFP4/dense fallback.
//! Extracted from `attention_forward.rs` to keep that file under 500 LoC.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::QuantizedWeight;

/// Env-gate for the NVFP4×NVFP4 split-K dispatch on K/V projections.
/// Default OFF until A/B-validated against `w4a16_gemv_dual`. Cached on
/// first read to avoid per-call env lookup.
fn qkv_splitk_enabled() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("ATLAS_QKV_SPLITK")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// Minimum M for the split-K NVFP4 path to be considered. The split-K
/// kernel uses M_TILE=64 so M<8 wastes >87% of the accumulator tile
/// and the absmax+quantize+sync pipeline dominates. Single-seq decode
/// (this file's caller) is M=1 → the gate naturally falls through to
/// `w4a16_gemv_dual`. Multi-seq batched paths (n=4..17) can route
/// through here once they expose a `m` parameter.
const QKV_SPLITK_MIN_M: u32 = 8;

/// Upper threshold on K*N for split-K to win. Above this, the GMEM
/// bandwidth wall on the reduce kernel regresses vs the non-split kernel
/// (bench: K=N=4096 → 89µs split-K beats 111µs non-split, but
/// K=5120 N=17408 (89M) → 942µs split-K LOSES vs 827µs non-split).
const QKV_SPLITK_MAX_KN: u64 = 25_000_000;

/// Constant K-split factor. Bench data shows 2 is the sweet spot for
/// K=5120 K/V projections (K_proj K*N=5.2M): K_STEP=64 × K_SPLITS=2
/// divides K cleanly (5120/(64*2)=40 inner steps per split). Higher
/// splits add reduce overhead without wave-gain benefit at these
/// shapes.
const QKV_K_SPLITS: u32 = 2;

impl Qwen3AttentionLayer {
    /// True iff all five kernels required for the split-K path are loaded.
    fn has_qkv_splitk(&self) -> bool {
        self.nvfp4_gemm_k.0 != 0
            && self.nvfp4_gemm_splitk_k.0 != 0
            && self.nvfp4_splitk_reduce_k.0 != 0
            && self.nvfp4_absmax_k.0 != 0
            && self.nvfp4_quantize_k.0 != 0
    }

    /// Run the W4A4 (NVFP4×NVFP4) split-K projection for one K or V
    /// matrix. Mirrors `DenseFfnLayer::forward_e2m1_proj` but routes
    /// through the split-K partial+reduce pair instead of the
    /// monolithic `nvfp4_nvfp4_gemm_t_m64` kernel.
    ///
    /// Caller already absmax+quantized the activation into
    /// `(a_packed, a_scale, a_scale2)`. The two K/V projections share
    /// the same A so they reuse one prequant.
    #[allow(clippy::too_many_arguments)]
    fn forward_kv_proj_splitk(
        &self,
        ctx: &ForwardContext,
        a_packed: DevicePtr,
        a_scale: DevicePtr,
        a_scale2: f32,
        weight: &QuantizedWeight,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        let scale2_ab = a_scale2 * weight.weight_scale_2;
        let c_partial = ctx.buffers.splitk_workspace();
        // splitk_workspace is also used by paged-decode attention. The
        // attention kernel writes after this projection, so reuse is
        // safe within a single layer step. Size check: this projection
        // needs `K_SPLITS * M * N * 4` bytes; for M=17 N=1024 K_SPLITS=2
        // that's 136 KB — well under the extended-arena floor of
        // ~25 MB used by paged-decode-splitk.
        ops::nvfp4_nvfp4_gemm_splitk(
            ctx.gpu,
            self.nvfp4_gemm_splitk_k,
            self.nvfp4_splitk_reduce_k,
            a_packed,
            a_scale,
            weight.weight,
            weight.weight_scale,
            scale2_ab,
            c_partial,
            output,
            m,
            n,
            k,
            QKV_K_SPLITS,
            stream,
        )
    }

    /// Try the K + V split-K dispatch. Returns `Ok(true)` if both
    /// projections were handled (caller skips the legacy path). Returns
    /// `Ok(false)` if any gate fails — caller falls through to the
    /// existing dual-GEMV / per-proj GEMV / dense path.
    ///
    /// Gates:
    ///   1. `ATLAS_QKV_SPLITK=1`
    ///   2. All five PTX symbols loaded
    ///   3. Both K and V weights are NVFP4 (HuggingFace `[N, K/2]`)
    ///   4. `m >= QKV_SPLITK_MIN_M` (8)
    ///   5. `K * N <= QKV_SPLITK_MAX_KN` (25M)
    ///
    /// At the single-seq decode caller site (this file), `m` is always 1
    /// so gate (4) never passes — the dispatch falls through unchanged.
    /// The wiring is in place for multi-seq batched callers that may
    /// route here in the future.
    #[allow(clippy::too_many_arguments)]
    fn try_kv_splitk(
        &self,
        normed: DevicePtr,
        k_out: DevicePtr,
        v_out: DevicePtr,
        m: u32,
        nkv: u32,
        hd: u32,
        h: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        if !qkv_splitk_enabled() || !self.has_qkv_splitk() {
            return Ok(false);
        }
        let n = nkv * hd;
        let k = h;
        if m < QKV_SPLITK_MIN_M {
            return Ok(false);
        }
        if (k as u64) * (n as u64) > QKV_SPLITK_MAX_KN {
            return Ok(false);
        }
        let (k_fp4, v_fp4) = match (
            self.k_weight.as_ref().and_then(|w| w.as_nvfp4()),
            self.v_weight.as_ref().and_then(|w| w.as_nvfp4()),
        ) {
            (Some(k), Some(v)) => (k, v),
            _ => return Ok(false),
        };

        // Single absmax+quantize of A shared by both K and V projections.
        // Reuse the FFN `nvfp4_quantize` arena via ctx scratch.
        //
        // Activation layout: `normed` is [m, h] BF16. We write A_packed
        // [m, h/2] + A_scale [m, h/16] into the start of `ssm_qkvz`
        // (which is sized for the SSM in_proj — much larger than what
        // we need here on Qwen3.6-27B). `ssm_ba` holds the FP32 absmax
        // scalar (4 bytes).
        let scratch = ctx.buffers.ssm_qkvz();
        let a_packed = scratch;
        let a_scale = scratch.offset((m as usize) * (h as usize) / 2);
        let a_max = ctx.buffers.ssm_ba();

        // absmax → scale2_a
        ctx.gpu.memset_async(a_max, 0, 4, stream)?;
        ops::nvfp4_global_absmax(ctx.gpu, self.nvfp4_absmax_k, normed, a_max, m * h, stream)?;
        ctx.gpu.synchronize(stream)?;
        let mut bytes = [0u8; 4];
        ctx.gpu.copy_d2h(a_max, &mut bytes)?;
        let global_max = f32::from_le_bytes(bytes);
        let a_scale2 = if global_max > 0.0 {
            global_max / (6.0 * 448.0)
        } else {
            1.0
        };

        // BF16 [m, h] → NVFP4 packed [m, h/2] + scale [m, h/16]
        ops::quantize_bf16_to_nvfp4(
            ctx.gpu,
            self.nvfp4_quantize_k,
            normed,
            a_packed,
            a_scale,
            a_scale2,
            m,
            h,
            stream,
        )?;

        // K projection: split-K GEMM + reduce
        self.forward_kv_proj_splitk(
            ctx, a_packed, a_scale, a_scale2, k_fp4, k_out, m, n, h, stream,
        )?;
        // V projection: split-K GEMM + reduce (reuses A)
        self.forward_kv_proj_splitk(
            ctx, a_packed, a_scale, a_scale2, v_fp4, v_out, m, n, h, stream,
        )?;
        Ok(true)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn attention_forward_kv(
        &self,
        normed: DevicePtr,
        k_out: DevicePtr,
        v_out: DevicePtr,
        nkv: u32,
        hd: u32,
        h: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if self.mla.is_some() {
            // MLA branch already wrote K and V into k_out/v_out
            return Ok(());
        }

        // Split-K NVFP4×NVFP4 fast path (gated by ATLAS_QKV_SPLITK=1).
        // This file's caller is single-seq decode (M=1) so the M>=8
        // gate inside `try_kv_splitk` always falls through here; the
        // dispatch is wired for future multi-seq callers. Returns
        // Ok(true) only when both K and V were handled by the split-K
        // pipeline.
        if self.try_kv_splitk(normed, k_out, v_out, 1, nkv, hd, h, ctx, stream)? {
            return Ok(());
        }

        if let (Some(k_fp8), Some(v_fp8)) = (
            self.k_weight.as_ref().and_then(|w| w.as_fp8()),
            self.v_weight.as_ref().and_then(|w| w.as_fp8()),
        ) {
            // FP8 native: individual w8a16_gemv for K and V
            ops::w8a16_gemv(
                ctx.gpu,
                self.w8a16_gemv_k,
                normed,
                k_fp8.weight,
                k_fp8.row_scale,
                k_out,
                nkv * hd,
                h,
                stream,
            )?;
            ops::w8a16_gemv(
                ctx.gpu,
                self.w8a16_gemv_k,
                normed,
                v_fp8.weight,
                v_fp8.row_scale,
                v_out,
                nkv * hd,
                h,
                stream,
            )?;
            return Ok(());
        }

        // Fuse K+V projections into a single dual GEMV when both are NVFP4
        match (
            self.k_weight.as_ref().and_then(|w| w.as_nvfp4()),
            self.v_weight.as_ref().and_then(|w| w.as_nvfp4()),
        ) {
            (Some(k_fp4), Some(v_fp4)) => {
                ops::w4a16_gemv_dual(
                    ctx.gpu,
                    self.w4a16_gemv_dual_k,
                    normed,
                    k_fp4,
                    k_out,
                    v_fp4,
                    v_out,
                    nkv * hd,
                    h,
                    stream,
                )?;
            }
            _ => {
                if let Some(nvfp4) = self.k_weight.as_ref().and_then(|w| w.as_nvfp4()) {
                    ops::w4a16_gemv(
                        ctx.gpu,
                        self.w4a16_gemv_k,
                        normed,
                        nvfp4,
                        k_out,
                        nkv * hd,
                        h,
                        stream,
                    )?;
                } else {
                    ops::dense_gemv(
                        ctx.gpu,
                        self.dense_gemv_k,
                        normed,
                        &self.attn.k_proj,
                        k_out,
                        nkv * hd,
                        h,
                        stream,
                    )?;
                }
                if let Some(nvfp4) = self.v_weight.as_ref().and_then(|w| w.as_nvfp4()) {
                    ops::w4a16_gemv(
                        ctx.gpu,
                        self.w4a16_gemv_k,
                        normed,
                        nvfp4,
                        v_out,
                        nkv * hd,
                        h,
                        stream,
                    )?;
                } else {
                    ops::dense_gemv(
                        ctx.gpu,
                        self.dense_gemv_k,
                        normed,
                        &self.attn.v_proj,
                        v_out,
                        nkv * hd,
                        h,
                        stream,
                    )?;
                }
            }
        }
        Ok(())
    }
}
