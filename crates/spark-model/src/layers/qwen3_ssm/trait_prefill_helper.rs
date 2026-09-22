// SPDX-License-Identifier: AGPL-3.0-only

//! Output-projection GEMM dispatch for `Qwen3SsmLayer::prefill_inner`.
//!
//! Hoisted from `trait_prefill.rs` to keep that file under the 500 LoC cap.
//! The single helper `prefill_out_proj_dispatch` mirrors the original
//! Section 10 block 1:1: routes through dense / FP8 (with `n128_m128` fast
//! path for k>128) / NVFP4-transposed / NVFP4 paths based on which weight
//! variant is loaded.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::{
    QWEN38_FLASHINFER_HIDDEN, QWEN38_FLASHINFER_SSM_QKVZ, QWEN38_FLASHINFER_SSM_VALUE,
    Qwen3SsmLayer, SsmFlashinferPrefillRoute, qwen38_ssm_flashinfer_geometry,
    qwen38_ssm_flashinfer_tactics, ssm_flashinfer_prefill_route, ssm_flashinfer_scratch_layout,
};
use crate::layer::ForwardContext;
use crate::layers::ops;

/// Number of SSM prefill projections that actually executed on the cuBLASLt
/// route. Read from the server log as `CUBLASLT_PROJ_CALL n=<total>`; a run
/// whose total is short of `layers x requests` took the fail-safe fallback and
/// must not be scored as a null result.
static CUBLASLT_PROJ_CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl Qwen3SsmLayer {
    fn flashinfer_ssm_nvfp4_active(&self) -> bool {
        self.sequential_qkvz
            && self.qkvz_nvfp4.is_some()
            && self.out_proj_dense.is_none()
            && !self.ssm.out_proj.is_null()
    }

    /// Resolve the explicit selector and validate the whole two-projection
    /// family before RMS norm or any parent/candidate projection mutates a
    /// buffer. Exact but unprepared requests fail here.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn preflight_flashinfer_ssm_prefill(
        &self,
        ctx: &ForwardContext,
        rows: usize,
        hidden: usize,
        qkvz_size: usize,
        value_dim: usize,
        qkvz_input: DevicePtr,
        qkvz_output: DevicePtr,
        output_input: DevicePtr,
        output_output: DevicePtr,
        stream: u64,
    ) -> Result<bool> {
        let requested = crate::layers::prefill_ssm_flashinfer_enabled()?;
        let exact_geometry = qwen38_ssm_flashinfer_geometry(rows, hidden, qkvz_size, value_dim)
            && !ctx.graph_capture;
        #[cfg(all(feature = "cuda", target_os = "linux"))]
        let prepared = self.flashinfer_ssm_prefill.is_some();
        #[cfg(not(all(feature = "cuda", target_os = "linux")))]
        let prepared = false;

        match ssm_flashinfer_prefill_route(
            requested,
            exact_geometry,
            self.flashinfer_ssm_nvfp4_active(),
            prepared,
        ) {
            SsmFlashinferPrefillRoute::Disabled | SsmFlashinferPrefillRoute::Ineligible => {
                Ok(false)
            }
            SsmFlashinferPrefillRoute::Missing => anyhow::bail!(
                "ATLAS_PREFILL_SSM_FLASHINFER=1 requires prepared exact Qwen3.8 SSM QKVZ and output operands"
            ),
            SsmFlashinferPrefillRoute::Complete => {
                #[cfg(all(feature = "cuda", target_os = "linux"))]
                {
                    self.validate_flashinfer_ssm_prefill(
                        ctx,
                        rows,
                        qkvz_input,
                        qkvz_output,
                        output_input,
                        output_output,
                        stream,
                    )?;
                    Ok(true)
                }
                #[cfg(not(all(feature = "cuda", target_os = "linux")))]
                {
                    anyhow::bail!("ATLAS_PREFILL_SSM_FLASHINFER=1 requires Linux CUDA support")
                }
            }
        }
    }

    #[cfg(all(feature = "cuda", target_os = "linux"))]
    #[allow(clippy::too_many_arguments)]
    fn validate_flashinfer_ssm_prefill(
        &self,
        ctx: &ForwardContext,
        rows: usize,
        qkvz_input: DevicePtr,
        qkvz_output: DevicePtr,
        output_input: DevicePtr,
        output_output: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        use ops::flashinfer_sm121::FlashInferSm121Shape;
        use ops::nvfp4_dynamic_scale::{
            Nvfp4DynamicScalePlan, validate_qwen38_ssm_projection_dynamic_scale,
        };

        let route = self
            .flashinfer_ssm_prefill
            .as_ref()
            .context("FlashInfer SSM route disappeared after admission")?;
        let qkvz_weight = self
            .qkvz_nvfp4
            .as_ref()
            .context("FlashInfer SSM QKVZ weight disappeared after admission")?;
        ensure!(
            route.qkvz.weight == qkvz_weight.weight
                && route.qkvz.weight_scale_2.to_bits() == qkvz_weight.weight_scale_2.to_bits()
                && route.output.weight == self.ssm.out_proj.weight
                && route.output.weight_scale_2.to_bits()
                    == self.ssm.out_proj.weight_scale_2.to_bits(),
            "FlashInfer SSM retained operands no longer match active runtime-generated weights"
        );
        ensure!(
            (route.qkvz.n, route.qkvz.k) == (QWEN38_FLASHINFER_SSM_QKVZ, QWEN38_FLASHINFER_HIDDEN)
                && (route.output.n, route.output.k)
                    == (QWEN38_FLASHINFER_HIDDEN, QWEN38_FLASHINFER_SSM_VALUE),
            "FlashInfer SSM retained operand geometry changed"
        );
        for (label, projection) in [("qkvz", &route.qkvz), ("output", &route.output)] {
            ensure!(
                !projection.weight.is_null()
                    && projection.weight.0.is_multiple_of(16)
                    && !projection.weight_scales_128x4.is_null()
                    && projection.weight_scales_128x4.0.is_multiple_of(16)
                    && projection.weight != projection.weight_scales_128x4,
                "FlashInfer SSM {label} retained pointers are null, misaligned, or aliased"
            );
        }

        let (qkvz_tactic, output_tactic) = qwen38_ssm_flashinfer_tactics(rows)
            .context("missing qualified FlashInfer SSM runtime tactic")?;
        for (tactic, n, k) in [
            (
                qkvz_tactic,
                QWEN38_FLASHINFER_SSM_QKVZ,
                QWEN38_FLASHINFER_HIDDEN,
            ),
            (
                output_tactic,
                QWEN38_FLASHINFER_HIDDEN,
                QWEN38_FLASHINFER_SSM_VALUE,
            ),
        ] {
            let shape = FlashInferSm121Shape::new(tactic, rows, n, k, 1)?;
            let prepared = route
                .library
                .prepare_borrowed_zero_workspace(ctx.gpu, shape, stream)?;
            ensure!(
                prepared.shape() == shape && prepared.stream() == stream,
                "FlashInfer SSM runtime preparation changed shape or stream"
            );
        }

        let capacity = ctx.buffers.sizes().ssm_conv_out_f32;
        let scratch_base = ctx.buffers.ssm_conv_out_f32();
        for (label, cols, input, output, projection) in [
            (
                "qkvz",
                QWEN38_FLASHINFER_HIDDEN,
                qkvz_input,
                qkvz_output,
                &route.qkvz,
            ),
            (
                "output",
                QWEN38_FLASHINFER_SSM_VALUE,
                output_input,
                output_output,
                &route.output,
            ),
        ] {
            let plan = Nvfp4DynamicScalePlan::qwen38_ssm_projection(
                u32::try_from(rows)?,
                u32::try_from(cols)?,
            )?;
            let buffers =
                Self::flashinfer_ssm_dynamic_buffers(scratch_base, capacity, input, plan)?;
            validate_qwen38_ssm_projection_dynamic_scale(
                route.dynamic_scale_kernels,
                buffers,
                u32::try_from(rows)?,
                u32::try_from(cols)?,
                projection.weight_scale_2,
            )?;
            Self::validate_flashinfer_ssm_disjoint(
                label,
                input,
                plan.input_bytes,
                output,
                rows.checked_mul(projection.n)
                    .and_then(|elements| elements.checked_mul(2))
                    .context("FlashInfer SSM output-byte overflow")?,
                scratch_base,
                ssm_flashinfer_scratch_layout(plan, capacity)?.total_bytes,
            )?;
        }
        Ok(())
    }

    #[cfg(all(feature = "cuda", target_os = "linux"))]
    fn flashinfer_ssm_dynamic_buffers(
        scratch_base: DevicePtr,
        capacity: usize,
        input_bf16: DevicePtr,
        plan: ops::nvfp4_dynamic_scale::Nvfp4DynamicScalePlan,
    ) -> Result<ops::nvfp4_dynamic_scale::Nvfp4DynamicScaleBuffers> {
        let layout = ssm_flashinfer_scratch_layout(plan, capacity)?;
        ensure!(
            !scratch_base.is_null() && scratch_base.0.is_multiple_of(16),
            "FlashInfer SSM scratch base is null or misaligned"
        );
        let _ = scratch_base
            .0
            .checked_add(u64::try_from(layout.total_bytes)?)
            .context("FlashInfer SSM scratch address overflow")?;
        Ok(ops::nvfp4_dynamic_scale::Nvfp4DynamicScaleBuffers {
            input_bf16,
            packed_e2m1: scratch_base.offset(layout.packed_offset),
            scales_e4m3_128x4: scratch_base.offset(layout.scales_offset),
            global_max_f32: scratch_base.offset(layout.global_max_offset),
            scale2_a_f32: scratch_base.offset(layout.scale2_offset),
            status_u32: scratch_base.offset(layout.status_offset),
            combined_alpha_f32: scratch_base.offset(layout.alpha_offset),
        })
    }

    #[cfg(all(feature = "cuda", target_os = "linux"))]
    fn validate_flashinfer_ssm_disjoint(
        label: &str,
        input: DevicePtr,
        input_bytes: usize,
        output: DevicePtr,
        output_bytes: usize,
        scratch: DevicePtr,
        scratch_bytes: usize,
    ) -> Result<()> {
        let span = |name: &str, pointer: DevicePtr, bytes: usize| -> Result<(u64, u64)> {
            ensure!(!pointer.is_null(), "FlashInfer SSM {label} {name} is null");
            ensure!(
                pointer.0.is_multiple_of(16),
                "FlashInfer SSM {label} {name} is misaligned"
            );
            Ok((
                pointer.0,
                pointer
                    .0
                    .checked_add(u64::try_from(bytes)?)
                    .context("FlashInfer SSM device range overflow")?,
            ))
        };
        let input = span("input", input, input_bytes)?;
        let output = span("output", output, output_bytes)?;
        let scratch = span("scratch", scratch, scratch_bytes)?;
        let overlaps = |left: (u64, u64), right: (u64, u64)| left.0 < right.1 && right.0 < left.1;
        ensure!(
            !overlaps(input, output) && !overlaps(input, scratch) && !overlaps(output, scratch),
            "FlashInfer SSM {label} input/output/scratch ranges overlap"
        );
        Ok(())
    }

    #[cfg(all(feature = "cuda", target_os = "linux"))]
    fn launch_flashinfer_ssm_projection(
        &self,
        ctx: &ForwardContext,
        rows: u32,
        input: DevicePtr,
        output: DevicePtr,
        is_qkvz: bool,
        stream: u64,
    ) -> Result<()> {
        use ops::flashinfer_sm121::{FlashInferSm121Buffers, FlashInferSm121Shape};
        use ops::nvfp4_dynamic_scale::{
            Nvfp4DynamicScalePlan, enqueue_qwen38_ssm_projection_dynamic_scale,
        };

        let route = self
            .flashinfer_ssm_prefill
            .as_ref()
            .context("FlashInfer SSM launch without prepared operands")?;
        let (qkvz_tactic, output_tactic) = qwen38_ssm_flashinfer_tactics(rows as usize)
            .context("missing qualified FlashInfer SSM launch tactic")?;
        let (label, projection, tactic) = if is_qkvz {
            ("qkvz", &route.qkvz, qkvz_tactic)
        } else {
            ("output", &route.output, output_tactic)
        };
        let plan = Nvfp4DynamicScalePlan::qwen38_ssm_projection(rows, projection.k as u32)?;
        let buffers = Self::flashinfer_ssm_dynamic_buffers(
            ctx.buffers.ssm_conv_out_f32(),
            ctx.buffers.sizes().ssm_conv_out_f32,
            input,
            plan,
        )?;
        let dynamic = enqueue_qwen38_ssm_projection_dynamic_scale(
            ctx.gpu,
            route.dynamic_scale_kernels,
            buffers,
            rows,
            projection.k as u32,
            projection.weight_scale_2,
            stream,
        )?;
        let activation = dynamic.for_flashinfer(stream)?;
        let shape =
            FlashInferSm121Shape::new(tactic, rows as usize, projection.n, projection.k, 1)?;
        let mut prepared = route
            .library
            .prepare_borrowed_zero_workspace(ctx.gpu, shape, stream)?;
        prepared
            .launch_eager(
                FlashInferSm121Buffers {
                    output_bf16: output,
                    activation_fp4: activation.activation_fp4,
                    weight_fp4: projection.weight,
                    activation_scales: activation.activation_scales,
                    weight_scales: projection.weight_scales_128x4,
                    global_scale_f32: activation.global_scale_f32,
                },
                stream,
            )
            .with_context(|| {
                format!(
                    "FlashInfer SSM {label} projection failed (M={rows}, N={}, K={})",
                    projection.n, projection.k
                )
            })
    }

    #[cfg(all(feature = "cuda", target_os = "linux"))]
    pub(super) fn launch_flashinfer_ssm_qkvz(
        &self,
        ctx: &ForwardContext,
        rows: u32,
        input: DevicePtr,
        output: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        self.launch_flashinfer_ssm_projection(ctx, rows, input, output, true, stream)
    }

    #[cfg(all(feature = "cuda", target_os = "linux"))]
    pub(super) fn launch_flashinfer_ssm_output(
        &self,
        ctx: &ForwardContext,
        rows: u32,
        input: DevicePtr,
        output: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        self.launch_flashinfer_ssm_projection(ctx, rows, input, output, false, stream)?;
        let route = self
            .flashinfer_ssm_prefill
            .as_ref()
            .context("FlashInfer SSM route disappeared before receipt")?;
        let (qkvz_tactic, output_tactic) = qwen38_ssm_flashinfer_tactics(rows as usize)
            .context("missing FlashInfer SSM receipt tactics")?;
        static LOGGED_SHAPES: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);
        let shape_bit = if rows == 2_079 { 1 } else { 2 };
        if LOGGED_SHAPES.fetch_or(shape_bit, std::sync::atomic::Ordering::Relaxed) & shape_bit == 0
        {
            tracing::info!(
                layer = route.layer,
                m = rows,
                qkvz_n = route.qkvz.n,
                qkvz_k = route.qkvz.k,
                output_n = route.output.n,
                output_k = route.output.k,
                qkvz_tactic,
                output_tactic,
                "routed complete SSM prefill projection family through FlashInfer SM121"
            );
            eprintln!(
                "ATLAS_PREFILL_SSM_FLASHINFER ENGAGED family=ssm layer={} M={} qkvz={}/{}/{} output={}/{}/{}",
                route.layer,
                rows,
                route.qkvz.n,
                route.qkvz.k,
                qkvz_tactic,
                route.output.n,
                route.output.k,
                output_tactic,
            );
        }
        Ok(())
    }

    /// QKVZ large-M projection: the 8-warp shadow when requested/eligible,
    /// else the parent `w4a16_gemm_t_m128`. Bit-identical either way.
    #[allow(clippy::too_many_arguments)]
    /// e4m3 copy of this layer's NVFP4 QKVZ weight, materialised once on first
    /// use. Returns None if the dequant kernel is missing or the allocation
    /// fails, so the caller keeps the hand-written W4A16 kernel.
    fn qkvz_weight_e4m3(
        &self,
        ctx: &ForwardContext,
        nvfp4_t: &crate::weight_map::QuantizedWeight,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Option<DevicePtr> {
        let mut slot = self.ssm_qkvz_e4m3.lock().ok()?;
        if let Some(ptr) = *slot {
            return Some(ptr);
        }
        if self.dequant_nvfp4_to_e4m3_k.0 == 0 || n % 128 != 0 || k % 32 != 0 {
            return None;
        }
        let bytes = n as usize * k as usize;
        let ptr = ctx.gpu.alloc(bytes).ok()?;
        ops::dequant_nvfp4_to_e4m3(
            ctx.gpu,
            self.dequant_nvfp4_to_e4m3_k,
            nvfp4_t.weight,
            nvfp4_t.weight_scale,
            ptr,
            nvfp4_t.weight_scale_2,
            n,
            k,
            stream,
        )
        .ok()?;
        tracing::info!(
            "ATLAS_SSM_PROJ_CUBLASLT: materialised QKVZ e4m3 weight N={n} K={k} ({} MiB)",
            bytes / (1024 * 1024)
        );
        *slot = Some(ptr);
        Some(ptr)
    }

    pub(super) fn qkvz_prefill_m128_dispatch(
        &self,
        ctx: &ForwardContext,
        normed: DevicePtr,
        nvfp4_t: &crate::weight_map::QuantizedWeight,
        proj_dst: DevicePtr,
        k: u32,
        n: u32,
        h: u32,
        stream: u64,
    ) -> Result<()> {
        use crate::layers::PrefillProjectionPipeRoute as Route;
        // cuBLASLt FP8: needs the NVFP4 weight materialised as e4m3 once, then
        // multiplies exactly the bytes the W4A16 kernel would have built.
        if crate::layers::ssm_proj_cublaslt_enabled()
            && let Some(w) = self.qkvz_weight_e4m3(ctx, nvfp4_t, n, h, stream)
            && self.try_cublaslt_fp8_proj(ctx, normed, w, proj_dst, k, n, h, stream)?
        {
            return Ok(());
        }
        match crate::layers::prefill_fp8_w8_route(
            crate::layers::prefill_fp8_w8_enabled(), k, n, h, self.w4a16_gemm_t_w8_k.0 != 0,
        ) {
            Route::Complete => {
                static SEEN: std::sync::Once = std::sync::Once::new();
                SEEN.call_once(|| tracing::info!("ENGAGED ATLAS_PREFILL_FP8_W8: ssm_qkvz M={k} N={n} K={h}"));
                ops::w4a16_gemm_t_w8(ctx.gpu, self.w4a16_gemm_t_w8_k, normed, nvfp4_t, proj_dst, k, n, h, stream)
            }
            Route::Missing => anyhow::bail!("ATLAS_PREFILL_FP8_W8=1 requires w4a16_gemm_t_m128n128_w8 (ssm_qkvz)"),
            Route::Disabled | Route::Ineligible => ops::w4a16_gemm_n128_m128(
                ctx.gpu, self.w4a16_gemm_t_m128_k, normed, nvfp4_t, proj_dst, k, n, h, stream,
            ),
        }
    }

    /// e4m3 activation scratch for the cuBLASLt projection route. Returns None
    /// when the buffer cannot serve `bytes`, so callers fall back to the
    /// hand-written kernel rather than failing the request.
    fn act_e4m3_scratch(&self, ctx: &ForwardContext, bytes: usize) -> Option<DevicePtr> {
        let mut slot = self.ssm_act_e4m3_scratch.lock().ok()?;
        if let Some((ptr, cap)) = *slot
            && cap >= bytes
        {
            return Some(ptr);
        }
        // Grow rather than fall back. The two SSM projections have different K
        // (qkvz 5120, out_proj 6144) and share this buffer, so sizing it to
        // whichever ran first silently pushed the other onto the hand-written
        // kernel -- caught by the engagement receipt reading 288 of 576 calls.
        // Growth happens at most once per distinct K and the old allocation is
        // small (~10 MiB), so it is not retained.
        let ptr = ctx.gpu.alloc(bytes).ok()?;
        tracing::info!(
            "ATLAS_SSM_PROJ_CUBLASLT: e4m3 activation scratch {} MiB",
            bytes / (1024 * 1024)
        );
        *slot = Some((ptr, bytes));
        Some(ptr)
    }

    /// cuBLASLt FP8 projection: cast the BF16 activations to the same e4m3 bytes
    /// the MMA path produces, then run the library GEMM. Returns Ok(false) when
    /// the route is unavailable so the caller uses the existing kernel.
    #[allow(clippy::too_many_arguments)]
    fn try_cublaslt_fp8_proj(
        &self,
        ctx: &ForwardContext,
        act_bf16: DevicePtr,
        weight_fp8: DevicePtr,
        out: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<bool> {
        if !crate::layers::ssm_proj_cublaslt_enabled() || self.cast_bf16_to_e4m3_k.0 == 0 {
            return Ok(false);
        }
        // cuBLASLt FP8 constraint, and the cast kernel packs 4 elements per word.
        if k % 16 != 0 || n % 16 != 0 || (m as u64 * k as u64) % 4 != 0 {
            return Ok(false);
        }
        let bytes = m as usize * k as usize;
        let Some(act_e4m3) = self.act_e4m3_scratch(ctx, bytes) else {
            return Ok(false);
        };
        ops::cast_bf16_to_e4m3(
            ctx.gpu,
            self.cast_bf16_to_e4m3_k,
            act_bf16,
            act_e4m3,
            m as u64 * k as u64,
            stream,
        )?;
        spark_runtime::cublaslt::fp8_gemm_act_weight_t(
            act_e4m3.0,
            weight_fp8.0,
            out.0,
            m,
            n,
            k,
            stream,
        )?;
        // Engagement receipt. This route is deliberately fail-safe: a disabled
        // gate, a missing kernel, an ineligible shape or an undersized scratch
        // all fall back to the hand-written kernel and would otherwise produce
        // a silent "no change" that reads as a null result rather than a bug.
        // One line per call, with a running total, so a run can be scored only
        // after asserting the count matches layers x requests.
        let n_calls = CUBLASLT_PROJ_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        tracing::info!("CUBLASLT_PROJ_CALL n={n_calls} M={m} N={n} K={k}");
        Ok(true)
    }

    pub(super) fn prefill_out_proj_dispatch(
        &self,
        ctx: &ForwardContext,
        normed_out_buf: DevicePtr,
        out_proj_buf: DevicePtr,
        k: u32,
        h: usize,
        value_dim: usize,
        stream: u64,
    ) -> Result<()> {
        if let Some(ref dense_out) = self.out_proj_dense {
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_k,
                normed_out_buf,
                dense_out,
                out_proj_buf,
                k,
                h as u32,
                value_dim as u32,
                stream,
            )
        } else if let Some(fp8) = self.out_proj_fp8 {
            use crate::layers::PrefillProjectionPipeRoute as Route;
            // Weights here are already e4m3 in memory, so the library GEMM takes
            // the identical operands with no conversion pass.
            if self.try_cublaslt_fp8_proj(
                ctx, normed_out_buf, fp8, out_proj_buf, k, h as u32, value_dim as u32, stream,
            )? {
                return Ok(());
            }
            match crate::layers::prefill_fp8_w8_route(
                crate::layers::prefill_fp8_w8_enabled(), k, h as u32, value_dim as u32, self.fp8_gemm_t_w8_k.0 != 0,
            ) {
                Route::Complete => {
                    static SEEN: std::sync::Once = std::sync::Once::new();
                    SEEN.call_once(|| tracing::info!("ENGAGED ATLAS_PREFILL_FP8_W8: ssm_out M={k}"));
                    return ops::fp8_gemm_t_w8(ctx.gpu, self.fp8_gemm_t_w8_k, normed_out_buf, fp8, out_proj_buf, k, h as u32, value_dim as u32, stream)
                        .map_err(|e| anyhow::anyhow!("ssm prefill: out_proj w8 GEMM failed: {e}"));
                }
                Route::Missing => anyhow::bail!("ATLAS_PREFILL_FP8_W8=1 requires fp8_gemm_t_m128n128_w8 (ssm_out)"),
                Route::Disabled | Route::Ineligible => {}
            }
            if k > 128 {
                ops::fp8_gemm_n128_m128(
                    ctx.gpu,
                    self.fp8_gemm_t_m128_k,
                    normed_out_buf,
                    fp8,
                    out_proj_buf,
                    k,
                    h as u32,
                    value_dim as u32,
                    stream,
                )
            } else {
                ops::fp8_gemm_n128(
                    ctx.gpu,
                    self.fp8_gemm_k,
                    normed_out_buf,
                    fp8,
                    out_proj_buf,
                    k,
                    h as u32,
                    value_dim as u32,
                    stream,
                )
            }
        } else if let Some(ref nvfp4_t) = self.out_proj_nvfp4_t {
            // ATLAS_SSM_OUT_PREFILL_M128=1: w4a16_gemm_t_m128 is the same dequant
            // (FP4 -> e4m3) and the same m16n8k32 e4m3 MMA K-order per output
            // element as w4a16_gemm_t; it only halves weight re-reads (2 M chunks
            // per CTA). Bit-identical output, measured in bench/gemm_test.
            if k > 128 && crate::layers::ssm_out_prefill_m128_enabled() {
                static SEEN: std::sync::Once = std::sync::Once::new();
                SEEN.call_once(|| tracing::info!("ENGAGED ATLAS_SSM_OUT_PREFILL_M128: ssm_out M={k}"));
                return ops::w4a16_gemm_n128_m128(
                    ctx.gpu,
                    self.w4a16_gemm_t_m128_k,
                    normed_out_buf,
                    nvfp4_t,
                    out_proj_buf,
                    k,
                    h as u32,
                    value_dim as u32,
                    stream,
                )
                .map_err(|e| anyhow::anyhow!("ssm prefill: out_proj m128 GEMM failed: {e}"));
            }
            ops::w4a16_gemm_n128(
                ctx.gpu,
                self.w4a16_gemm_t_k,
                normed_out_buf,
                nvfp4_t,
                out_proj_buf,
                k,
                h as u32,
                value_dim as u32,
                stream,
            )
        } else {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm_k,
                normed_out_buf,
                &self.ssm.out_proj,
                out_proj_buf,
                k,
                h as u32,
                value_dim as u32,
                stream,
            )
        }
        .map_err(|e| anyhow::anyhow!("ssm prefill: out_proj GEMM failed: {e}"))
    }
}
