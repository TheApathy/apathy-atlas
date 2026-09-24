// SPDX-License-Identifier: AGPL-3.0-only

//! is_ssm_layer + prefill_phase1.

use super::*;

pub(super) fn use_packed_gdn_inputs(
    requested: bool,
    has_kernel: bool,
    num_tokens: usize,
    qkvz_size: usize,
    conv_dim: usize,
    value_dim: usize,
    gate_stride: usize,
) -> Result<bool, &'static str> {
    let compatible = (1..=65_535).contains(&num_tokens)
        && conv_dim > 0
        && value_dim > 0
        && gate_stride > 0
        && qkvz_size == conv_dim + value_dim
        && qkvz_size.is_multiple_of(8)
        && conv_dim.is_multiple_of(8)
        && value_dim.is_multiple_of(8)
        && gate_stride.is_multiple_of(4);
    let eligible = requested && compatible;
    if eligible && !has_kernel {
        return Err("ATLAS_SSM_PREFILL_PACK=1 requires causal_conv1d_update_prefill_zcopy");
    }
    Ok(eligible)
}

#[allow(clippy::too_many_arguments)]
pub(super) const fn fused_prefill_conv_l2_geometry(
    num_tokens: usize,
    conv_dim: usize,
    d_conv: usize,
    qk_channels: usize,
    head_dim: usize,
    z_dim: usize,
) -> bool {
    num_tokens > 0
        && d_conv == 4
        && head_dim == 128
        && conv_dim.is_multiple_of(head_dim)
        && qk_channels > 0
        && qk_channels <= conv_dim
        && qk_channels.is_multiple_of(head_dim)
        && z_dim > 0
        && z_dim <= conv_dim
}

pub(super) const fn packed_gdn_kernel_ready(
    has_zcopy: bool,
    fused_requested: bool,
    fused_geometry: bool,
    has_fused: bool,
) -> bool {
    has_zcopy || (fused_requested && fused_geometry && has_fused)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn use_fused_prefill_conv_l2(
    requested: bool,
    packed_inputs: bool,
    has_kernel: bool,
    num_tokens: usize,
    conv_dim: usize,
    d_conv: usize,
    qk_channels: usize,
    head_dim: usize,
    z_dim: usize,
) -> Result<bool, &'static str> {
    let exact_geometry =
        fused_prefill_conv_l2_geometry(num_tokens, conv_dim, d_conv, qk_channels, head_dim, z_dim);
    let eligible = requested && exact_geometry;
    if eligible && !packed_inputs {
        return Err("ATLAS_SSM_PREFILL_CONV_L2=1 requires ATLAS_SSM_PREFILL_PACK=1");
    }
    if eligible && !has_kernel {
        return Err(
            "ATLAS_SSM_PREFILL_CONV_L2=1 requires causal_conv1d_update_prefill_l2norm_zcopy",
        );
    }
    Ok(eligible)
}

impl Qwen3SsmLayer {
    pub(super) fn is_ssm_layer_inner(&self) -> bool {
        true
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_phase1_inner(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        _seq_len_start: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        _kv_write_start: usize,
        gdn_bufs: &GdnPrefillBuffers,
        token_offset: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let k = num_tokens as u32;
        let bf16 = 2usize;
        let fp32 = 4usize;

        let ssm_state = state
            .as_any_mut()
            .downcast_mut::<SsmLayerState>()
            .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState"))?;

        let nk = ctx.config.linear_num_key_heads;
        let kd = ctx.config.linear_key_head_dim;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let vpg = nv / nk;
        let key_dim = nk * kd;
        let value_dim = nv * vd;
        let conv_dim = key_dim * 2 + value_dim;
        let d_conv = ctx.config.linear_conv_kernel_dim;
        let qkvz_size = ctx.config.ssm_qkvz_size();
        let gate_stride = nv * 2;
        let normed = ctx.buffers.norm_output();
        let deinterleaved = ctx.buffers.ssm_deinterleaved();
        let proj_dst = if self.sequential_qkvz {
            deinterleaved
        } else {
            ctx.buffers.ssm_qkvz()
        };
        // In the single-sequence two-phase path, fail an exact requested GDN
        // route before RMS/residual or conv-state mutation. Batched phase1 has
        // no AttnMetadata and retains its existing batched parent GDN route.
        let _gdn_c143_preflight = if ctx.attn_metadata.is_some() && token_offset == 0 {
            self.preflight_gdn_c143_prefill(
                ctx,
                gdn_bufs.total_len,
                nk,
                nv,
                kd,
                vd,
                conv_dim,
                gate_stride,
                qkvz_size,
                ssm_state.h_state,
                gdn_bufs.qkv,
                gdn_bufs.gate_beta,
                gdn_bufs.output,
                stream,
            )?
        } else {
            None
        };
        let flashinfer_ssm = self.preflight_flashinfer_ssm_prefill(
            ctx,
            num_tokens,
            h,
            qkvz_size,
            value_dim,
            normed,
            proj_dst,
            ctx.buffers.ssm_qkvz(),
            ctx.buffers.moe_output(),
            stream,
        )?;
        let conv_l2_requested = crate::layers::ssm_prefill_conv_l2_enabled();
        let conv_l2_geometry = fused_prefill_conv_l2_geometry(
            num_tokens,
            conv_dim,
            d_conv,
            key_dim * 2,
            kd,
            value_dim,
        );
        let packed = use_packed_gdn_inputs(
            crate::layers::ssm_prefill_pack_enabled(),
            packed_gdn_kernel_ready(
                self.conv1d_prefill_zcopy_k.0 != 0,
                conv_l2_requested,
                conv_l2_geometry,
                self.conv1d_prefill_l2norm_zcopy_k.0 != 0,
            ),
            num_tokens,
            qkvz_size,
            conv_dim,
            value_dim,
            gate_stride,
        )
        .map_err(anyhow::Error::msg)?;
        let qkv_dst = gdn_bufs.qkv.offset(token_offset * conv_dim * bf16);
        let gb_dst = gdn_bufs.gate_beta.offset(token_offset * gate_stride * fp32);
        let z_dst_base = gdn_bufs.z.offset(token_offset * value_dim * bf16);
        let fused_conv_l2 = use_fused_prefill_conv_l2(
            conv_l2_requested,
            packed,
            self.conv1d_prefill_l2norm_zcopy_k.0 != 0,
            num_tokens,
            conv_dim,
            d_conv,
            key_dim * 2,
            kd,
            value_dim,
        )
        .map_err(anyhow::Error::msg)?;
        if packed {
            static PACK_SEEN: std::sync::Once = std::sync::Once::new();
            PACK_SEEN.call_once(|| {
                tracing::info!(
                    "ATLAS_SSM_PREFILL_PACK engaged: direct QKV/gates plus fused conv/Z replaces {} copies/chunk",
                    num_tokens + 2,
                );
            });
        }

        // ── 1. RMS norm + residual for N tokens ──
        ops::rms_norm_residual(
            ctx.gpu,
            self.rms_norm_residual_k,
            hidden,
            &self.input_norm,
            normed,
            residual,
            k,
            h as u32,
            eps,
            stream,
        )?;
        // ── 2+3. QKVZ GEMM (+ deinterleave if needed) ──
        if flashinfer_ssm {
            #[cfg(all(feature = "cuda", target_os = "linux"))]
            self.launch_flashinfer_ssm_qkvz(ctx, k, normed, proj_dst, stream)?;
            #[cfg(not(all(feature = "cuda", target_os = "linux")))]
            unreachable!("non-CUDA FlashInfer SSM route passed preflight");
        } else if let Some(fp8) = self.qkvz_fp8 {
            ops::fp8_gemm_n128(
                ctx.gpu,
                self.fp8_gemm_k,
                normed,
                fp8,
                proj_dst,
                k,
                qkvz_size as u32,
                h as u32,
                stream,
            )
            .map_err(|e| {
                anyhow::anyhow!("ssm phase1: QKVZ FP8 GEMM failed (M={k}, N={qkvz_size}): {e}")
            })?;
        } else if let Some(ref nvfp4_t) = self.qkvz_nvfp4_t {
            if k > 128 {
                self.qkvz_prefill_m128_dispatch(ctx, normed, nvfp4_t, proj_dst, k, qkvz_size as u32, h as u32, stream)
                .map_err(|e| {
                    anyhow::anyhow!("ssm phase1: QKVZ m128 GEMM failed (M={k}, N={qkvz_size}): {e}")
                })?;
            } else if k <= 32
                && self.w4a16_gemm_t_m16_k.0 != 0
                && super::super::tc_nvfp4_m16_enabled()
            {
                ops::w4a16_gemm_n128_m16(
                    ctx.gpu,
                    self.w4a16_gemm_t_m16_k,
                    normed,
                    nvfp4_t,
                    proj_dst,
                    k,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )
                .map_err(|e| {
                    anyhow::anyhow!("ssm phase1: QKVZ m16 GEMM failed (M={k}, N={qkvz_size}): {e}")
                })?;
            } else {
                ops::w4a16_gemm_n128(
                    ctx.gpu,
                    self.w4a16_gemm_t_k,
                    normed,
                    nvfp4_t,
                    proj_dst,
                    k,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )
                .map_err(|e| {
                    anyhow::anyhow!("ssm phase1: QKVZ GEMM failed (M={k}, N={qkvz_size}): {e}")
                })?;
            }
        } else if let Some(ref nvfp4) = self.qkvz_nvfp4 {
            match crate::layers::prefill_projection_pipe_route(
                crate::layers::prefill_proj_pipe_enabled(),
                h as u32,
                self.w4a16_gemm_pipe_k.0 != 0,
            ) {
                crate::layers::PrefillProjectionPipeRoute::Complete => {
                    static QKVZ_PIPE: std::sync::Once = std::sync::Once::new();
                    QKVZ_PIPE.call_once(|| {
                        tracing::info!("ENGAGED ATLAS_PREFILL_PROJ_PIPE: ssm_qkvz");
                    });
                    // Byte-exact pipelined shadow (see `prefill_proj_pipe_enabled`).
                    // K = h must be a multiple of the pipe's 64-row stage.
                    ops::w4a16_gemm_pipe(
                        ctx.gpu,
                        self.w4a16_gemm_pipe_k,
                        normed,
                        nvfp4,
                        proj_dst,
                        k,
                        qkvz_size as u32,
                        h as u32,
                        stream,
                    )
                    .map_err(|e| {
                        anyhow::anyhow!(
                            "ssm phase1: QKVZ PIPE GEMM failed (M={k}, N={qkvz_size}): {e}"
                        )
                    })?;
                }
                crate::layers::PrefillProjectionPipeRoute::Missing => {
                    anyhow::bail!(
                        "ATLAS_PREFILL_PROJ_PIPE=1 requires w4a16_gemm_pipe for an eligible SSM QKVZ projection"
                    );
                }
                crate::layers::PrefillProjectionPipeRoute::Disabled
                | crate::layers::PrefillProjectionPipeRoute::Ineligible => {
                    ops::w4a16_gemm(
                        ctx.gpu,
                        self.w4a16_gemm_k,
                        normed,
                        nvfp4,
                        proj_dst,
                        k,
                        qkvz_size as u32,
                        h as u32,
                        stream,
                    )
                    .map_err(|e| {
                        anyhow::anyhow!("ssm phase1: QKVZ GEMM failed (M={k}, N={qkvz_size}): {e}")
                    })?;
                }
            }
        } else {
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_k,
                normed,
                &self.ssm.in_proj_qkvz,
                proj_dst,
                k,
                qkvz_size as u32,
                h as u32,
                stream,
            )?;
        }
        if !self.sequential_qkvz {
            ops::deinterleave_qkvz(
                ctx.gpu,
                self.deinterleave_k,
                proj_dst,
                deinterleaved,
                k,
                nk as u32,
                kd as u32,
                vpg as u32,
                vd as u32,
                stream,
            )?;
        }

        // ── 4+5. Fused BA GEMM + GDN gates (token-parallel) ──
        let ba_size = ctx.config.ssm_ba_size();
        let gates_buf = if packed {
            gb_dst
        } else {
            ctx.buffers.ssm_gates()
        };
        ops::dense_gemm_ba_gates_prefill(
            ctx.gpu,
            self.ba_gates_prefill_k,
            self.ba_gates_prefill_rows_k,
            normed,
            &self.ssm.in_proj_ba,
            self.ssm.a_log.weight,
            self.ssm.dt_bias.weight,
            gates_buf,
            k,
            ba_size as u32,
            h as u32,
            h as u32,
            gate_stride as u32,
            nv as u32,
            vpg as u32,
            stream,
        )?;

        // ── 6. Batched conv1d for all N tokens ──
        let conv_out_buf = if packed {
            qkv_dst
        } else {
            ctx.buffers.ssm_qkvz()
        };
        let z_src_base = deinterleaved.offset((key_dim * 2 + value_dim) * bf16);
        if fused_conv_l2 {
            static FUSED_CONV_L2_SEEN: std::sync::Once = std::sync::Once::new();
            FUSED_CONV_L2_SEEN.call_once(|| {
                tracing::info!(
                    "ATLAS_SSM_PREFILL_CONV_L2 engaged: exact BF16 conv/L2/Z shadow for M={}",
                    num_tokens,
                );
            });
            ops::conv1d_update_prefill_l2norm_zcopy(
                ctx.gpu,
                self.conv1d_prefill_l2norm_zcopy_k,
                ssm_state.conv_state,
                deinterleaved,
                &self.ssm.conv1d,
                DevicePtr::NULL,
                conv_out_buf,
                z_src_base,
                z_dst_base,
                conv_dim as u32,
                d_conv as u32,
                k,
                qkvz_size as u32,
                conv_dim as u32,
                value_dim as u32,
                (key_dim * 2) as u32,
                kd as u32,
                1e-6,
                stream,
            )?;
        } else if packed {
            ops::conv1d_update_prefill_zcopy(
                ctx.gpu,
                self.conv1d_prefill_zcopy_k,
                ssm_state.conv_state,
                deinterleaved,
                &self.ssm.conv1d,
                DevicePtr::NULL,
                conv_out_buf,
                z_src_base,
                z_dst_base,
                conv_dim as u32,
                d_conv as u32,
                k,
                qkvz_size as u32,
                conv_dim as u32,
                value_dim as u32,
                stream,
            )?;
        } else {
            ops::conv1d_update_prefill(
                ctx.gpu,
                self.conv1d_prefill_k,
                ssm_state.conv_state,
                deinterleaved,
                &self.ssm.conv1d,
                DevicePtr::NULL,
                conv_out_buf,
                conv_dim as u32,
                d_conv as u32,
                k,
                qkvz_size as u32,
                conv_dim as u32,
                stream,
            )?;
        }
        // ── 7. Batched L2 norm on Q,K for all N tokens ──
        if !fused_conv_l2 {
            ops::l2_norm(
                ctx.gpu,
                self.l2_norm_k,
                conv_out_buf,
                (nk * 2) as u32,
                kd as u32,
                1e-6,
                k,
                conv_dim as u32,
                stream,
            )?;
        }

        // ── 8. Fallback persistence (direct route is already complete) ──
        if !packed {
            // Fallback preserves support for older/non-CUDA kernel bundles.
            ctx.gpu
                .copy_d2d_async(conv_out_buf, qkv_dst, num_tokens * conv_dim * bf16, stream)?;
            ctx.gpu
                .copy_d2d_async(gates_buf, gb_dst, num_tokens * gate_stride * fp32, stream)?;

            let z_elem_bytes = value_dim * bf16;
            for t in 0..num_tokens {
                let z_src = z_src_base.offset(t * qkvz_size * bf16);
                let z_dst = z_dst_base.offset(t * value_dim * bf16);
                ctx.gpu.copy_d2d_async(z_src, z_dst, z_elem_bytes, stream)?;
            }
        }

        Ok(())
    }
}
