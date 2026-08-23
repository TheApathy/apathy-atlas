// SPDX-License-Identifier: AGPL-3.0-only

//! TransformerLayer::decode_batched.

use super::*;

impl Qwen3SsmLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_batched_inner(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        _seq_len: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let k = num_tokens as u32;
        let bf16 = 2usize; // bytes per BF16
        let fp32 = 4usize; // bytes per FP32

        let ssm_state = state
            .as_any_mut()
            .downcast_mut::<SsmLayerState>()
            .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState"))?;

        let nk = ctx.config.linear_num_key_heads;
        let kd = ctx.config.linear_key_head_dim;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let vpg = nv / nk;
        let key_dim = nk * kd; // 2048
        let value_dim = nv * vd; // 4096
        let conv_dim = key_dim * 2 + value_dim; // 8192
        let qk_ch = (key_dim * 2) as u32; // Q+K channels for fused L2 norm
        let d_conv = ctx.config.linear_conv_kernel_dim;
        let qkvz_size = ctx.config.ssm_qkvz_size(); // 12288
        let serial = crate::model::env_diag::DflashSerialControls::current();

        // Validate every requested diagnostic route before launching the first
        // layer kernel. A requested route must select the same implementation
        // as ordinary K=1 decode; silently falling through would invalidate the
        // numerical bisection while leaving a misleading env snapshot.
        let k1_fused_qkvz = self.fused_rms_qkvz_k.0 != 0
            && self.sequential_qkvz
            && self.qkvz_nvfp4.is_some()
            && !ctx.config.use_fp32_residual()
            && std::env::var("ATLAS_FUSE_SSM_QKVZ").ok().as_deref() == Some("1");
        let force_serial_qkvz = serial_diag::require_serial_path(
            serial_diag::SerialPath::Qkvz,
            super::super::ssm_qkvz_serial_enabled(),
            num_tokens,
            self.sequential_qkvz
                && self.qkvz_nvfp4.is_some()
                && self.qkvz_fp8w.is_none()
                && !k1_fused_qkvz,
            "requires sequential NVFP4 QKVZ, no FP8 decode weight, and unfused K1 QKVZ",
        )?;
        let force_serial_ba = serial_diag::require_serial_path(
            serial_diag::SerialPath::Ba,
            super::super::ssm_ba_serial_enabled(),
            num_tokens,
            true,
            "requires the mandatory single-row BA/gate kernel",
        )?;
        let force_serial_recurrent = serial_diag::require_serial_path(
            serial_diag::SerialPath::Recurrent,
            super::super::ssm_recurrent_serial_enabled(),
            num_tokens,
            self.conv1d_l2norm_f32_k.0 != 0
                && self.gdn_f32_k.0 != 0
                && self.gated_rms_norm_f32_k.0 != 0,
            "requires non-null FP32 conv, GDN, and gated-norm kernels",
        )?;
        let force_serial_out = serial_diag::require_serial_path(
            serial_diag::SerialPath::Out,
            super::super::ssm_out_serial_enabled(),
            num_tokens,
            self.out_proj_fp8w.is_none()
                && self.out_proj_dense.is_none()
                && !self.ssm.out_proj.is_null(),
            "requires the ordinary K1 NVFP4 output-projection weight",
        )?;
        let force_serial_ffn = serial_diag::require_serial_path(
            serial_diag::SerialPath::Ffn,
            serial.ffn,
            num_tokens,
            ctx.ffn_defer.is_none(),
            "is incompatible with deferred cross-sequence FFN",
        )?;
        let force_serial_layer_norms = serial_diag::require_serial_path(
            serial_diag::SerialPath::LayerNorms,
            serial.layer_norms,
            num_tokens,
            !k1_fused_qkvz,
            "requires unfused K1 input RMS/QKVZ",
        )?;
        let exact_flat_route = exact_flat_ssm_route(
            num_tokens,
            ctx.ddtree_parent_ids_dev.is_some(),
            self.ba_gates_batchn_exact_k.0 != 0,
            self.conv1d_l2norm_f32_sequence_k.0 != 0,
            self.gdn_f32_sequence_k.0 != 0,
            self.gated_rms_norm_f32_multi_seq_k.0 != 0,
        );
        if exact_flat_route == ExactFlatSsmRoute::SerialK1 {
            anyhow::ensure!(
                self.conv1d_l2norm_f32_k.0 != 0
                    && self.gdn_f32_k.0 != 0
                    && self.gated_rms_norm_f32_k.0 != 0,
                "exact flat SSM fallback requires ordinary FP32 conv, GDN, and gated-norm kernels"
            );
        }
        crate::model::control_engagement::require(
            crate::model::control_engagement::ControlPath::SsmFfn,
            num_tokens > 1 && ctx.ffn_defer.is_none(),
            "requires a local SSM FFN and no deferred cross-sequence FFN",
        )?;
        crate::model::control_engagement::require(
            crate::model::control_engagement::ControlPath::SsmPreNorm,
            num_tokens > 1 && !k1_fused_qkvz,
            "requires unfused K1 input RMS/QKVZ",
        )?;
        crate::model::control_engagement::require(
            crate::model::control_engagement::ControlPath::SsmPostNorm,
            num_tokens > 1,
            "requires multi-row SSM post-normalization",
        )?;

        // ── 1. RMS norm + residual for K tokens ──
        let normed = ctx.buffers.norm_output();
        crate::kprof!(ctx.gpu, stream, "ssm_rms_norm_residual", {
            if force_serial_layer_norms {
                serial_diag::prove_serial_path(serial_diag::SerialPath::LayerNorms);
                let residual_elem = if ctx.config.use_fp32_residual() { 4 } else { 2 };
                for row in 0..num_tokens {
                    ops::rms_norm_residual(
                        ctx.gpu,
                        self.rms_norm_residual_k,
                        hidden.offset(row * h * residual_elem),
                        &self.input_norm,
                        normed.offset(row * h * bf16),
                        residual.offset(row * h * residual_elem),
                        1,
                        h as u32,
                        eps,
                        stream,
                    )?;
                }
                crate::model::control_engagement::engage(
                    crate::model::control_engagement::ControlPath::SsmPreNorm,
                )?;
            } else {
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
            }
            anyhow::Result::<()>::Ok(())
        })?;

        // ── 2+3. QKVZ projection (+ deinterleave if needed) ──
        // For sequential_qkvz (Qwen3.5): write directly to deinterleaved buffer.
        // For interleaved (80B): write to qkvz_out, then deinterleave per token.
        // NOTE: K=3 fused (rms_norm_residual + w4a16_gemv_batch3) kernel exists
        // in `w4a16_gemv_fused.cu` (`rms_norm_residual_w4a16_gemv_batch3`) and
        // is exposed via `ops::rms_norm_residual_w4a16_gemv_batch3` but the
        // 3*K*2 = 30 KiB smem footprint cuts occupancy on GB10 and the
        // per-token serial RMS pass loses the parallelism the original launch
        // gets from grid.x = 3, so wall-clock is ~7% worse on the AEON-27B
        // K=3 verify path. Kept off by default; re-enable when a parallel
        // per-token RMS variant is available.
        let deinterleaved = ctx.buffers.ssm_deinterleaved(); // [K, 12288] BF16
        let proj_dst = if self.sequential_qkvz {
            deinterleaved
        } else {
            ctx.buffers.ssm_qkvz()
        };
        // Losslessness diagnostic: run the exact single-token decode GEMV for
        // every verify row.  The normal K=gamma path intentionally uses a
        // different GEMM/split-K accumulation order, so disabling split-K is
        // not enough to determine whether QKVZ is where K=1 and K=gamma first
        // diverge.  Keep this opt-in and narrow to the sequential NVFP4 layout
        // used by Qwen3.5/3.6; it is far too slow for production serving.
        if force_serial_qkvz {
            serial_diag::prove_serial_path(serial_diag::SerialPath::Qkvz);
            let nvfp4 = self.qkvz_nvfp4.as_ref().unwrap();
            for t in 0..num_tokens {
                ops::w4a16_decode_gemv(
                    ctx.gpu,
                    self.w4a16_gemv_k,
                    self.w4a16_gemv_sw_k,
                    self.gemv_sw,
                    normed.offset(t * h * bf16),
                    nvfp4,
                    proj_dst.offset(t * qkvz_size * bf16),
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )?;
            }
        } else if self.sequential_qkvz
            && (2..=32).contains(&num_tokens)
            && !super::super::ssm_proj_tc_enabled()
            && let Some(nvfp4) = self.qkvz_nvfp4.as_ref()
        {
            crate::kprof!(ctx.gpu, stream, "ssm_qkvz_exact", {
                self.project_nvfp4_rows_exact_or_k1(
                    ctx.gpu,
                    normed,
                    nvfp4,
                    proj_dst,
                    k,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )?;
                anyhow::Result::<()>::Ok(())
            })?;
        } else if num_tokens == 3 {
            if let Some(ref nvfp4) = self.qkvz_nvfp4 {
                crate::kprof!(ctx.gpu, stream, "ssm_qkvz_w4a16_gemv_batch3", {
                    ops::w4a16_gemv_batch3(
                        ctx.gpu,
                        self.w4a16_gemv_batch3_k,
                        normed,
                        nvfp4,
                        proj_dst,
                        qkvz_size as u32,
                        h as u32,
                        stream,
                    )?;
                    anyhow::Result::<()>::Ok(())
                })?;
            } else {
                for t in 0..3u32 {
                    ops::dense_gemv(
                        ctx.gpu,
                        self.dense_gemv_k,
                        normed.offset(t as usize * h * bf16),
                        &self.ssm.in_proj_qkvz,
                        proj_dst.offset(t as usize * qkvz_size * bf16),
                        qkvz_size as u32,
                        h as u32,
                        stream,
                    )?;
                }
            }
        } else if num_tokens == 2 {
            if let Some(ref nvfp4) = self.qkvz_nvfp4 {
                ops::w4a16_gemv_batch2(
                    ctx.gpu,
                    self.w4a16_gemv_batch2_k,
                    normed,
                    nvfp4,
                    proj_dst,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )?;
            } else {
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    normed,
                    &self.ssm.in_proj_qkvz,
                    proj_dst,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )?;
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    normed.offset(h * bf16),
                    &self.ssm.in_proj_qkvz,
                    proj_dst.offset(qkvz_size * bf16),
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )?;
            }
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
            )?;
        } else if let Some(ref nvfp4_t) = self.qkvz_nvfp4_t {
            // m128 halves B re-reads for large M (prefill); m16 halves
            // discarded MMA work for small-M K=γ verify (MTP K=3 → M=3,
            // DFlash γ=16 → M=16-17); m64 default otherwise.
            if k > 128 {
                ops::w4a16_gemm_n128_m128(
                    ctx.gpu,
                    self.w4a16_gemm_t_m128_k,
                    normed,
                    nvfp4_t,
                    proj_dst,
                    k,
                    qkvz_size as u32,
                    h as u32,
                    stream,
                )?;
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
                )?;
            } else if k > 3
                && k <= 32
                && super::super::ssm_qkvz_splitk() > 0
                && self.w4a16_gemm_t_m32_n64_splitk_k.0 != 0
                && self.reduce_splitk_k.0 != 0
                && self.ssm_splitk_workspace.lock().unwrap().is_some()
            {
                // ATLAS_SSM_QKVZ_SPLITK: K=γ verify qkvz projection
                // [M=17, N=12288, K=5120] through the split-K m32_n64
                // kernel. Floor-map microbench (2026-07-05): the
                // w4a16_gemm_t default runs 220.5µs = 59% of the 132µs
                // DRAM floor; split-K×2 measures 167.3µs (85%, 232 GB/s)
                // + ~10µs reduce — ~2.2ms/step across 48 layers. Lossless
                // FP32 partials, mirrors the shipped ffn_down split-K.
                let ws = self.ssm_splitk_workspace.lock().unwrap().unwrap();
                ops::w4a16_gemm_n64_m32_splitk(
                    ctx.gpu,
                    self.w4a16_gemm_t_m32_n64_splitk_k,
                    self.reduce_splitk_k,
                    normed,
                    nvfp4_t,
                    proj_dst,
                    ws,
                    k,
                    qkvz_size as u32,
                    h as u32,
                    qkvz_size as u32, // ldb == N for tightly-packed T-weight
                    super::super::ssm_qkvz_splitk(),
                    stream,
                )?;
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
                )?;
            }
        } else if let Some(ref nvfp4) = self.qkvz_nvfp4 {
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
            )?;
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
            for t in 0..(num_tokens as u32) {
                let src = proj_dst.offset(t as usize * qkvz_size * bf16);
                let dst = deinterleaved.offset(t as usize * qkvz_size * bf16);
                ops::deinterleave_qkvz(
                    ctx.gpu,
                    self.deinterleave_k,
                    src,
                    dst,
                    1,
                    nk as u32,
                    kd as u32,
                    vpg as u32,
                    vd as u32,
                    stream,
                )?;
            }
        }

        // ── 4. BA projection + GDN gates per token ──
        // BA output: ssm_ba buffer; gates: ssm_gates buffer [K, nv*2] FP32
        // Layout per token: [gate(nv), beta(nv)] → stride = 2*nv FP32 elements.
        // Must match gdn_decode_chunk2's gb_stride parameter.
        let gates_buf = ctx.buffers.ssm_gates(); // [K, gate(nv) + beta(nv)] FP32
        let gate_beta_stride = nv * 2 * fp32; // bytes per token in gates buffer
        let ba_size = ctx.config.ssm_ba_size(); // 64
        if force_serial_ba || exact_flat_route == ExactFlatSsmRoute::SerialK1 {
            if force_serial_ba {
                serial_diag::prove_serial_path(serial_diag::SerialPath::Ba);
            }
            for t in 0..num_tokens {
                let gate_t = gates_buf.offset(t * gate_beta_stride);
                ops::dense_gemv_ba_gates(
                    ctx.gpu,
                    self.ba_gates_k,
                    normed.offset(t * h * bf16),
                    &self.ssm.in_proj_ba,
                    self.ssm.a_log.weight,
                    self.ssm.dt_bias.weight,
                    gate_t,
                    gate_t.offset(nv * fp32),
                    ba_size as u32,
                    h as u32,
                    vpg as u32,
                    stream,
                )?;
            }
        } else if exact_flat_route == ExactFlatSsmRoute::ExactSequence {
            crate::kprof!(ctx.gpu, stream, "ssm_ba_exact", {
                ops::dense_gemv_ba_gates_batchn(
                    ctx.gpu,
                    self.ba_gates_batchn_exact_k,
                    normed,
                    &self.ssm.in_proj_ba,
                    self.ssm.a_log.weight,
                    self.ssm.dt_bias.weight,
                    gates_buf,
                    gates_buf.offset(nv * fp32),
                    num_tokens as u32,
                    ba_size as u32,
                    h as u32,
                    vpg as u32,
                    (nv * 2) as u32,
                    stream,
                )?;
                anyhow::Result::<()>::Ok(())
            })?;
        } else {
            crate::kprof!(ctx.gpu, stream, "ssm_ba_proj_loop", {
                // K=3 verify fast path: collapse the 3-launch per-token loop into
                // a single `dense_gemv_bf16_batch3` launch. The per-token loop is
                // entirely launch-overhead bound (24 μs/launch × 3 × 48 SSM
                // layers ≈ 3.5 ms/verify wasted), while the underlying compute
                // (N=64 × K=hidden) is trivial. Gated by
                // ATLAS_SSM_BA_BATCHED=1 and requires the optional batch3 PTX
                // module to be present (NULL handle ⇒ fall through to the
                // per-token loop). `normed` is already a contiguous [K, h] BF16
                // buffer; `ssm_ba()` is already a contiguous [K, ba_size] BF16
                // buffer — both layouts match the kernel's [3, K]→[3, N]
                // contract exactly.
                if super::super::ssm_ba_batch_enabled() && self.dense_gemv_batchn_k.0 == 0 {
                    static WARNED: std::sync::Once = std::sync::Once::new();
                    super::super::warn_kernel_fallback(
                        &WARNED,
                        "ATLAS_SSM_BA_BATCH=1",
                        "gemv::dense_gemv_bf16_batchn",
                        "the K=γ verify BA projection stays on the per-token dense_gemv \
                     loop — 17 launches per SSM layer instead of 1, across 48 layers",
                    );
                }
                if num_tokens >= 2
                    && super::super::ssm_ba_batch_enabled()
                    && self.dense_gemv_batchn_k.0 != 0
                {
                    // General batched BA projection: ONE dense_gemv_bf16_batchn
                    // launch (grid.y = token) replaces the per-token dense_gemv
                    // loop. Each y-block runs the exact dense_gemv_bf16 body on
                    // its token's row, so output is BIT-IDENTICAL to the loop
                    // (md5-gated; a dense_gemm variant was NOT bit-exact — its
                    // 16×16-tile accumulation order differs → counting md5
                    // mismatch, measured 2026-07-02). `normed` is contiguous
                    // [num_tokens, h] BF16; `ssm_ba()` is contiguous
                    // [num_tokens, ba_size] BF16 — matching the kernel's A/C
                    // layout. Cuts 17 launches → 1 per SSM layer on the DFlash
                    // γ=16 verify. Covers K=17 which the batch3 path can't.
                    ops::dense_gemv_batchn(
                        ctx.gpu,
                        self.dense_gemv_batchn_k,
                        normed,
                        &self.ssm.in_proj_ba,
                        ctx.buffers.ssm_ba(),
                        num_tokens as u32,
                        ba_size as u32,
                        h as u32,
                        stream,
                    )?;
                } else if num_tokens == 3
                    && super::super::ssm_ba_batched_enabled()
                    && self.dense_gemv_batch3_k.0 != 0
                {
                    ops::dense_gemv_batch3(
                        ctx.gpu,
                        self.dense_gemv_batch3_k,
                        normed,
                        &self.ssm.in_proj_ba,
                        ctx.buffers.ssm_ba(),
                        ba_size as u32,
                        h as u32,
                        stream,
                    )?;
                } else {
                    for t in 0..(num_tokens as u32) {
                        let normed_t = normed.offset(t as usize * h * bf16);
                        let ba_out = ctx.buffers.ssm_ba().offset(t as usize * ba_size * bf16);
                        // Dense GEMV for BA projection (small: 64 outputs)
                        ops::dense_gemv(
                            ctx.gpu,
                            self.dense_gemv_k,
                            normed_t,
                            &self.ssm.in_proj_ba,
                            ba_out,
                            ba_size as u32,
                            h as u32,
                            stream,
                        )?;
                    }
                }
                anyhow::Result::<()>::Ok(())
            })?;
            // Batched compute_gdn_gates across K=γ tokens via the multi-seq
            // kernel (one CTA per (token, head)). Stride between successive
            // tokens in ba_out is `ba_size` BF16 elements; in gates/beta it's
            // `2*nv` FP32 elements (gate at offset 0, beta at offset nv).
            // Saves K-1 launches per SSM layer per K=γ verify step. Falls
            // back to per-token loop when the multi-seq kernel handle is null
            // (e.g. older PTX bundles).
            crate::kprof!(ctx.gpu, stream, "ssm_compute_gdn_gates_batched", {
                if self.compute_gdn_gates_multi_seq_k.0 != 0 && num_tokens >= 2 {
                    ops::compute_gdn_gates_multi_seq(
                        ctx.gpu,
                        self.compute_gdn_gates_multi_seq_k,
                        ctx.buffers.ssm_ba(),
                        self.ssm.a_log.weight,
                        self.ssm.dt_bias.weight,
                        gates_buf,
                        gates_buf.offset(nv * fp32),
                        num_tokens as u32,
                        nv as u32,
                        nk as u32,
                        vpg as u32,
                        ba_size as u32,
                        (nv * 2) as u32,
                        stream,
                    )?;
                } else {
                    for t in 0..(num_tokens as u32) {
                        let ba_out = ctx.buffers.ssm_ba().offset(t as usize * ba_size * bf16);
                        let gate_t = gates_buf.offset(t as usize * gate_beta_stride);
                        let beta_t = gates_buf.offset(t as usize * gate_beta_stride + nv * fp32);
                        ops::compute_gdn_gates(
                            ctx.gpu,
                            self.compute_gdn_gates_k,
                            ba_out,
                            self.ssm.a_log.weight,
                            self.ssm.dt_bias.weight,
                            gate_t,
                            beta_t,
                            1,
                            nv as u32,
                            nk as u32,
                            vpg as u32,
                            ba_size as u32,
                            stream,
                        )?;
                    }
                }
                anyhow::Result::<()>::Ok(())
            })?;
        }

        // ── 5-7. Conv1d + L2 norm + GDN per token (with intermediate checkpoints) ──
        // Reuse ssm_qkvz buffer for conv output (safe: deinterleave is done)
        let conv_out_buf = ctx.buffers.ssm_qkvz();
        let gdn_out_buf = ctx.buffers.attn_output();
        let h_bytes = self.h_state_bytes;
        let conv_bytes = self.conv_state_bytes;

        // Intermediates are pre-allocated from the pool (fixed GPU addresses for
        // CUDA graph stability). Verify they exist BEFORE we index into them — a
        // bare `debug_assert!` is a no-op in release and produces an opaque
        // out-of-bounds panic instead of an actionable error (see #bugs
        // m0t0chan EP=2 2026-04-05). Most-common cause: EP=2 worker started
        // without `--speculative --mtp-quantization` to mirror the head.
        if ssm_state.h_state_intermediates.len() < num_tokens
            || ssm_state.conv_state_intermediates.len() < num_tokens
        {
            anyhow::bail!(
                "SSM MTP intermediate buffers not allocated (h_state_intermediates.len()={}, \
                 conv_state_intermediates.len()={}, num_tokens={}). \
                 If this is an EP=2 worker, the head node is sending MTP verify commands \
                 but the worker was started without `--speculative` (and matching \
                 `--mtp-quantization`/`--num-drafts`). Add those flags to the worker invocation.",
                ssm_state.h_state_intermediates.len(),
                ssm_state.conv_state_intermediates.len(),
                num_tokens,
            );
        }

        let normed_out_buf = conv_out_buf;
        if force_serial_recurrent || exact_flat_route == ExactFlatSsmRoute::SerialK1 {
            if force_serial_recurrent {
                serial_diag::prove_serial_path(serial_diag::SerialPath::Recurrent);
            }
            // Ordinary decode deliberately retains conv and GDN outputs in
            // FP32 because BF16 truncation compounds through the recurrent
            // state.  The optimized K=gamma path currently uses BF16 WY
            // kernels, violating that contract.  This slow oracle executes
            // the exact three single-token kernels row-by-row and snapshots
            // every post-row state for the normal verify commit machinery.
            let conv_f32 = ctx.buffers.ssm_conv_out_f32();
            let gdn_f32 = conv_f32.offset(conv_dim * fp32);
            for t in 0..num_tokens {
                let qkv_t = deinterleaved.offset(t * qkvz_size * bf16);
                ops::conv1d_update_l2norm(
                    ctx.gpu,
                    self.conv1d_l2norm_f32_k,
                    ssm_state.conv_state,
                    qkv_t,
                    &self.ssm.conv1d,
                    conv_f32,
                    conv_dim as u32,
                    d_conv as u32,
                    1,
                    qk_ch,
                    kd as u32,
                    1e-6,
                    stream,
                )?;
                ctx.gpu.copy_d2d_async(
                    ssm_state.conv_state,
                    ssm_state.conv_state_intermediates[t],
                    conv_bytes,
                    stream,
                )?;

                let gate_t = gates_buf.offset(t * gate_beta_stride);
                ops::gdn_decode(
                    ctx.gpu,
                    self.gdn_f32_k,
                    ssm_state.h_state,
                    conv_f32,
                    conv_f32.offset(key_dim * fp32),
                    conv_f32.offset(key_dim * 2 * fp32),
                    gate_t,
                    gate_t.offset(nv * fp32),
                    gdn_f32,
                    1,
                    nk as u32,
                    nv as u32,
                    kd as u32,
                    vd as u32,
                    stream,
                )?;
                ctx.gpu.copy_d2d_async(
                    ssm_state.h_state,
                    ssm_state.h_state_intermediates[t],
                    h_bytes,
                    stream,
                )?;

                let z_t = qkv_t.offset((key_dim * 2 + value_dim) * bf16);
                ops::gated_rms_norm(
                    ctx.gpu,
                    self.gated_rms_norm_f32_k,
                    gdn_f32,
                    z_t,
                    &self.ssm.norm,
                    normed_out_buf.offset(t * value_dim * bf16),
                    nv as u32,
                    vd as u32,
                    vd as u32,
                    eps,
                    vd as u32,
                    stream,
                )?;
            }
        } else if exact_flat_route == ExactFlatSsmRoute::ExactSequence {
            let conv_f32 = ctx.buffers.ssm_conv_out_f32();
            let gdn_f32 = conv_f32.offset(conv_dim * fp32);
            let conv_inter = contiguous_intermediate_base(
                &ssm_state.conv_state_intermediates,
                num_tokens,
                conv_bytes,
                "conv",
            )?;
            let h_inter = contiguous_intermediate_base(
                &ssm_state.h_state_intermediates,
                num_tokens,
                h_bytes,
                "H",
            )?;
            crate::kprof!(ctx.gpu, stream, "ssm_conv_fp32_seq", {
                ops::conv1d_update_l2norm_f32_sequence(
                    ctx.gpu,
                    self.conv1d_l2norm_f32_sequence_k,
                    ssm_state.conv_state,
                    deinterleaved,
                    &self.ssm.conv1d,
                    conv_f32,
                    conv_inter,
                    num_tokens as u32,
                    conv_dim as u32,
                    d_conv as u32,
                    qk_ch,
                    kd as u32,
                    1e-6,
                    qkvz_size as u32,
                    qkvz_size as u32,
                    (conv_bytes / fp32) as u32,
                    stream,
                )?;
                anyhow::Result::<()>::Ok(())
            })?;
            crate::kprof!(ctx.gpu, stream, "ssm_gdn_fp32_seq", {
                // Register-resident H variant (identical single-accumulator FP32
                // order; H cached in registers across the token loop instead of
                // 3 global re-reads/token). Bit-exact by construction.
                // SPEED PROBE: ATLAS_SSM_GDN_NOSNAP=1 routes to a twin with the
                // per-token snapshot writes removed. Those are 89% of this
                // kernel's H traffic and impl_b2.rs reads exactly ONE of the 16
                // (index num_accepted). The probe measures the ceiling of
                // removing them; it INVALIDATES rollback, so output is wrong.
                // Measure only — the exact fix recomputes on commit.
                let nosnap = std::env::var("ATLAS_SSM_GDN_NOSNAP").ok().as_deref() == Some("1")
                    && self.gdn_f32_sequence_nosnap_k.0 != 0;
                // LAZY-COMMIT dispatch (ATLAS_SSM_GDN_LAZY=1): run the
                // lazyfinal kernel — no per-token snapshots (58.6% of this
                // kernel at k=16), final H into inter[num_tokens-1], h_state
                // left at step-initial for the commit replay. The async commit
                // mirrors this decision via gdn_seq_lazy_engaged_inner, so the
                // predicate MUST stay pure in num_tokens + env + handles.
                let lazy = self.gdn_seq_lazy_engaged_inner(num_tokens)
                    && ctx.ddtree_parent_ids_dev.is_none();
                if lazy {
                    // Retain this step's inputs BEFORE the GDN launch — at this
                    // point conv_f32/gates hold exactly what the kernel will
                    // read, so the copy is aliasing-proof by construction.
                    let max_k = crate::layers::qwen3_ssm::GDN_LAZY_MAX_K;
                    let retain = match self.gdn_lazy_retain.get() {
                        Some(pp) => *pp,
                        None => {
                            let bytes = max_k * qkvz_size * fp32 + max_k * nv * 2 * fp32;
                            let pp = ctx.gpu.alloc(bytes)?;
                            let _ = self.gdn_lazy_retain.set(pp);
                            pp
                        }
                    };
                    ctx.gpu.copy_d2d_async(
                        conv_f32,
                        retain,
                        num_tokens * qkvz_size * fp32,
                        stream,
                    )?;
                    ctx.gpu.copy_d2d_async(
                        gates_buf,
                        retain.offset(max_k * qkvz_size * fp32),
                        num_tokens * nv * 2 * fp32,
                        stream,
                    )?;
                }
                let persistent = !nosnap
                    && !lazy
                    && self.gdn_f32_sequence_persistent_k.0 != 0
                    && std::env::var("ATLAS_SSM_GDN_SEQ_PERSISTENT")
                        .ok()
                        .as_deref()
                        == Some("1");
                if nosnap {
                    // Same ABI as the plain sequence kernel — it simply skips
                    // the snapshot stores. h_inter is passed and left untouched.
                    ops::gdn_decode_f32_sequence(
                        ctx.gpu,
                        self.gdn_f32_sequence_nosnap_k,
                        ssm_state.h_state,
                        conv_f32,
                        conv_f32.offset(key_dim * fp32),
                        conv_f32.offset(key_dim * 2 * fp32),
                        gates_buf,
                        gates_buf.offset(nv * fp32),
                        gdn_f32,
                        h_inter,
                        num_tokens as u32,
                        nk as u32,
                        nv as u32,
                        kd as u32,
                        vd as u32,
                        qkvz_size as u32,
                        qkvz_size as u32,
                        (nv * 2) as u32,
                        qkvz_size as u32,
                        (h_bytes / fp32) as u32,
                        stream,
                    )?;
                } else if lazy {
                    // Same ABI + extern-shared layout as the persistent kernel.
                    ops::gdn_decode_f32_sequence_persistent(
                        ctx.gpu,
                        self.gdn_f32_sequence_lazyfinal_k,
                        ssm_state.h_state,
                        conv_f32,
                        conv_f32.offset(key_dim * fp32),
                        conv_f32.offset(key_dim * 2 * fp32),
                        gates_buf,
                        gates_buf.offset(nv * fp32),
                        gdn_f32,
                        h_inter,
                        num_tokens as u32,
                        nk as u32,
                        nv as u32,
                        kd as u32,
                        vd as u32,
                        qkvz_size as u32,
                        qkvz_size as u32,
                        (nv * 2) as u32,
                        qkvz_size as u32,
                        (h_bytes / fp32) as u32,
                        stream,
                    )?;
                } else if persistent {
                    ops::gdn_decode_f32_sequence_persistent(
                        ctx.gpu,
                        self.gdn_f32_sequence_persistent_k,
                        ssm_state.h_state,
                        conv_f32,
                        conv_f32.offset(key_dim * fp32),
                        conv_f32.offset(key_dim * 2 * fp32),
                        gates_buf,
                        gates_buf.offset(nv * fp32),
                        gdn_f32,
                        h_inter,
                        num_tokens as u32,
                        nk as u32,
                        nv as u32,
                        kd as u32,
                        vd as u32,
                        qkvz_size as u32,
                        qkvz_size as u32,
                        (nv * 2) as u32,
                        qkvz_size as u32,
                        (h_bytes / fp32) as u32,
                        stream,
                    )?;
                } else {
                    ops::gdn_decode_f32_sequence(
                        ctx.gpu,
                        self.gdn_f32_sequence_k,
                        ssm_state.h_state,
                        conv_f32,
                        conv_f32.offset(key_dim * fp32),
                        conv_f32.offset(key_dim * 2 * fp32),
                        gates_buf,
                        gates_buf.offset(nv * fp32),
                        gdn_f32,
                        h_inter,
                        num_tokens as u32,
                        nk as u32,
                        nv as u32,
                        kd as u32,
                        vd as u32,
                        qkvz_size as u32,
                        qkvz_size as u32,
                        (nv * 2) as u32,
                        qkvz_size as u32,
                        (h_bytes / fp32) as u32,
                        stream,
                    )?;
                }
                anyhow::Result::<()>::Ok(())
            })?;
            let z_base = deinterleaved.offset((key_dim * 2 + value_dim) * bf16);
            crate::kprof!(ctx.gpu, stream, "ssm_gated_norm_fp32_multi_seq", {
                ops::gated_rms_norm_f32_multi_seq(
                    ctx.gpu,
                    self.gated_rms_norm_f32_multi_seq_k,
                    gdn_f32,
                    z_base,
                    &self.ssm.norm,
                    normed_out_buf,
                    nv as u32,
                    num_tokens as u32,
                    vd as u32,
                    eps,
                    qkvz_size as u32,
                    qkvz_size as u32,
                    value_dim as u32,
                    stream,
                )?;
                anyhow::Result::<()>::Ok(())
            })?;
        } else {
            let args = super::trait_decode_batched_conv_gdn::ConvGdnArgs {
                num_tokens,
                deinterleaved,
                gates_buf,
                conv_out_buf,
                gdn_out_buf,
                h_bytes,
                conv_bytes,
                qkvz_size,
                conv_dim,
                key_dim,
                value_dim,
                d_conv,
                qk_ch,
                nk,
                nv,
                kd,
                vd,
                bf16,
                fp32,
                stream,
            };
            crate::kprof!(ctx.gpu, stream, "ssm_conv_gdn_combined", {
                self.decode_batched_conv_gdn(ssm_state, ctx, &args)?;
                anyhow::Result::<()>::Ok(())
            })?;

            // ── 8. Gated RMS norm — batched across K tokens ──
            // Z gate lives in `deinterleaved` at offset [Q + K + V] within
            // each token's qkvz slice; gdn output is BF16 [K, value_dim]
            // contig. Use the prefill kernel (grid = (heads_per_token,
            // num_actual_tokens, 1)) to collapse the per-token loop into a
            // single launch. The kernel reads stride parameters for
            // input/output and gate independently, so the value_dim/qkvz_size
            // stride mismatch is handled at the kernel.
            //
            // Saves num_tokens-1 launches per SSM layer per K=γ verify:
            //   K=3 × 48 SSM layers × ~20 μs/launch ≈ 2 ms/token.
            // The prefill kernel handle is non-null (init.rs:110 mandatory).
            crate::kprof!(ctx.gpu, stream, "ssm_gated_rms_norm_batched", {
                let z_base = deinterleaved.offset((key_dim * 2 + value_dim) * bf16);
                ops::gated_rms_norm_prefill(
                    ctx.gpu,
                    self.gated_rms_norm_prefill_k,
                    gdn_out_buf,
                    z_base,
                    &self.ssm.norm,
                    normed_out_buf,
                    nv as u32,
                    vd as u32,
                    eps,
                    num_tokens as u32,
                    value_dim as u32,
                    qkvz_size as u32,
                    stream,
                )?;
                anyhow::Result::<()>::Ok(())
            })?;
        }

        // ── 9. Output projection → [K, H] ──
        let out_proj_buf = ctx.buffers.moe_output(); // [K, H] BF16
        crate::kprof!(ctx.gpu, stream, "ssm_out_proj", {
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
                )?;
            } else if force_serial_out {
                serial_diag::prove_serial_path(serial_diag::SerialPath::Out);
                // Losslessness diagnostic matching `ssm_forward` exactly.
                // This deliberately streams the NVFP4 weight once per row;
                // use only to bisect K=1-vs-K=gamma numerical divergence.
                for t in 0..num_tokens {
                    ops::w4a16_decode_gemv(
                        ctx.gpu,
                        self.w4a16_gemv_k,
                        self.w4a16_gemv_sw_k,
                        self.gemv_sw,
                        normed_out_buf.offset(t * value_dim * bf16),
                        &self.ssm.out_proj,
                        out_proj_buf.offset(t * h * bf16),
                        h as u32,
                        value_dim as u32,
                        stream,
                    )?;
                }
            } else if (2..=32).contains(&num_tokens)
                && !self.ssm.out_proj.is_null()
                && !super::super::ssm_proj_tc_enabled()
            {
                crate::kprof!(ctx.gpu, stream, "ssm_out_exact", {
                    self.project_nvfp4_rows_exact_or_k1(
                        ctx.gpu,
                        normed_out_buf,
                        &self.ssm.out_proj,
                        out_proj_buf,
                        k,
                        h as u32,
                        value_dim as u32,
                        stream,
                    )?;
                    anyhow::Result::<()>::Ok(())
                })?;
            } else if num_tokens == 3
                && super::super::ssm_out_batch3_enabled()
                && !self.ssm.out_proj.is_null()
            {
                // ATLAS_SSM_OUT_BATCH3=1 fast path: triple-GEMV at M=3 avoids the
                // ~96% wasted MMA work of `w4a16_gemm` M_TILE=64. Requires a valid
                // NVFP4 non-transposed `ssm.out_proj` (Qwen3.5 NVFP4 + AEON-27B
                // path). Falls through to FP8 / NVFP4_T below when off or when the
                // model variant has `ssm.out_proj == null` (FP8-only loaders).
                ops::w4a16_gemv_batch3(
                    ctx.gpu,
                    self.w4a16_gemv_batch3_k,
                    normed_out_buf,
                    &self.ssm.out_proj,
                    out_proj_buf,
                    h as u32,
                    value_dim as u32,
                    stream,
                )?;
            } else if num_tokens == 2 {
                ops::w4a16_gemv_batch2(
                    ctx.gpu,
                    self.w4a16_gemv_batch2_k,
                    normed_out_buf,
                    &self.ssm.out_proj,
                    out_proj_buf,
                    h as u32,
                    value_dim as u32,
                    stream,
                )?;
            } else if k > 3
                && k <= 32
                && super::super::ssm_out_splitk() > 0
                && self.w4a16_gemm_t_m32_n64_splitk_k.0 != 0
                && self.reduce_splitk_k.0 != 0
                && let Some(proj_t) = self.out_proj_nvfp4_t.as_ref()
                && let Some(ws) = *self.ssm_splitk_workspace.lock().unwrap()
            {
                // ATLAS_SSM_OUT_SPLITK: K=γ verify out_proj
                // [M=17, N=H=5120, K=value_dim=6144] through the split-K
                // m32_n64 kernel. Placed BEFORE the fp8 branch: the
                // production route is `fp8_gemm_t` (out_proj_fp8 installed
                // by predequant_for_prefill), floor-map measured at
                // 191.6µs = 61% of its 117µs FP8 floor (2× weight bytes).
                // Split-K×4 on the NVFP4 T-weight slices K across
                // gridDim.z (320 CTAs) into FP32 partials and measures
                // 89.4µs (85% of the 76µs NVFP4 floor, 232 GB/s) + ~5µs
                // reduce — a 2.1× kernel win, ~4.6ms/step across 48
                // layers. Lossless FP32 partials +
                // reduce_splitk_f32_to_bf16, mirroring the shipped
                // ffn_down split-K route.
                ops::w4a16_gemm_n64_m32_splitk(
                    ctx.gpu,
                    self.w4a16_gemm_t_m32_n64_splitk_k,
                    self.reduce_splitk_k,
                    normed_out_buf,
                    proj_t,
                    out_proj_buf,
                    ws,
                    k,
                    h as u32,
                    value_dim as u32,
                    h as u32, // ldb == N for tightly-packed T-weight
                    super::super::ssm_out_splitk(),
                    stream,
                )?;
            } else if let Some(fp8) = self.out_proj_fp8 {
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
                    )?;
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
                    )?;
                }
            } else if let Some(ref nvfp4_t) = self.out_proj_nvfp4_t {
                if k <= 32 && self.w4a16_gemm_t_m16_k.0 != 0 && super::super::tc_nvfp4_m16_enabled()
                {
                    ops::w4a16_gemm_n128_m16(
                        ctx.gpu,
                        self.w4a16_gemm_t_m16_k,
                        normed_out_buf,
                        nvfp4_t,
                        out_proj_buf,
                        k,
                        h as u32,
                        value_dim as u32,
                        stream,
                    )?;
                } else if k > 3
                    && k <= 32
                    && self.w4a16_gemm_t_m32_n64_k.0 != 0
                    && super::super::ssm_out_proj_m32n64()
                {
                    // K=γ verify (DFlash γ=16 → M=17): route the SSM out_proj
                    // [M=17, N=H=5120, K=value_dim=6144] through the m32_n64
                    // transposed kernel instead of w4a16_gemm_n128 (N_TILE=128).
                    // At N=5120 the m128 kernel fields only ceil(5120/128)=40 CTAs
                    // — SM-starved at ~37 GB/s on a 15.7 MB weight, mirroring the
                    // ffn_down occupancy starve. N_TILE=64 doubles CTAs to
                    // ceil(5120/64)=80 and M_TILE=32 keeps a single B read for M≤32.
                    // Same proven T-weight + m32_n64 path as qkv/o/FFN (token-exact,
                    // single K-chain — bit-exact, no split-K rounding).
                    ops::w4a16_gemm_n64_m32(
                        ctx.gpu,
                        self.w4a16_gemm_t_m32_n64_k,
                        normed_out_buf,
                        nvfp4_t,
                        out_proj_buf,
                        k,
                        h as u32,
                        value_dim as u32,
                        stream,
                    )?;
                } else {
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
                    )?;
                }
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
                )?;
            }
            anyhow::Result::<()>::Ok(())
        })?;

        if force_serial_ffn {
            serial_diag::prove_serial_path(serial_diag::SerialPath::Ffn);
            let residual_elem = if ctx.config.use_fp32_residual() {
                4usize
            } else {
                bf16
            };
            for row in 0..num_tokens {
                let hidden_row = hidden.offset(row * h * residual_elem);
                let out_row = out_proj_buf.offset(row * h * bf16);
                let normed_row = ctx.buffers.norm_output().offset(row * h * bf16);
                let residual_row = residual.offset(row * h * residual_elem);
                ops::residual_add_rms_norm(
                    ctx.gpu,
                    self.residual_add_rms_norm_k,
                    hidden_row,
                    out_row,
                    &self.post_attn_norm,
                    normed_row,
                    residual_row,
                    1,
                    h as u32,
                    eps,
                    stream,
                )?;
                let moe_out = self.ffn.forward(normed_row, ctx, stream)?;
                ops::residual_add(
                    ctx.gpu,
                    self.residual_add_k,
                    hidden_row,
                    moe_out,
                    h as u32,
                    stream,
                )?;
            }
            crate::model::control_engagement::engage(
                crate::model::control_engagement::ControlPath::SsmFfn,
            )?;
            if force_serial_layer_norms {
                crate::model::control_engagement::engage(
                    crate::model::control_engagement::ControlPath::SsmPostNorm,
                )?;
            }
            return Ok(());
        }

        // ── 10. Batched residual + post-norm, then MoE + residual ──
        // residual_add_rms_norm supports multi-token (grid.x = num_tokens)
        let normed2_base = ctx.buffers.norm_output();
        crate::kprof!(ctx.gpu, stream, "ssm_post_attn_resid_norm", {
            if force_serial_layer_norms {
                let residual_elem = if ctx.config.use_fp32_residual() {
                    4
                } else {
                    bf16
                };
                for row in 0..num_tokens {
                    ops::residual_add_rms_norm(
                        ctx.gpu,
                        self.residual_add_rms_norm_k,
                        hidden.offset(row * h * residual_elem),
                        out_proj_buf.offset(row * h * bf16),
                        &self.post_attn_norm,
                        normed2_base.offset(row * h * bf16),
                        residual.offset(row * h * residual_elem),
                        1,
                        h as u32,
                        eps,
                        stream,
                    )?;
                }
            } else {
                ops::residual_add_rms_norm(
                    ctx.gpu,
                    self.residual_add_rms_norm_k,
                    hidden,
                    out_proj_buf,
                    &self.post_attn_norm,
                    normed2_base,
                    residual,
                    num_tokens as u32,
                    h as u32,
                    eps,
                    stream,
                )?;
            }
            anyhow::Result::<()>::Ok(())
        })?;
        if force_serial_layer_norms {
            crate::model::control_engagement::engage(
                crate::model::control_engagement::ControlPath::SsmPostNorm,
            )?;
        }

        // CROSS-SEQ BATCHED DFLASH VERIFY (#39): defer the FFN. The mixer +
        // post-attn residual RMS-norm above have written this seq's
        // `num_tokens` FFN-input rows into `normed2_base` and left the
        // post-mixer residual in `hidden` (= residual). Copy those rows into
        // the caller's external collection buffer at this seq's row offset and
        // return WITHOUT running the FFN — the model orchestrator batches the
        // FFN GEMM across all sequences' rows, reading FFN weights once, then
        // adds the batched FFN output back into each seq's `hidden`.
        if let Some(defer) = ctx.ffn_defer {
            let dst = defer.dst_base.offset(defer.row_offset * h * bf16);
            ctx.gpu
                .copy_d2d_async(normed2_base, dst, num_tokens * h * bf16, stream)?;
            return Ok(());
        }

        if num_tokens == 3 {
            // Fused K=3 MoE: 5 kernel launches instead of 15
            crate::kprof!(ctx.gpu, stream, "ssm_ffn_forward_k3", {
                self.ffn.forward_k3(normed2_base, ctx, stream)?;
                anyhow::Result::<()>::Ok(())
            })?;
            let moe_out = ctx.buffers.moe_output();
            crate::kprof!(ctx.gpu, stream, "ssm_ffn_residual_add", {
                ops::residual_add(
                    ctx.gpu,
                    self.residual_add_k,
                    hidden,
                    moe_out,
                    (3 * h) as u32,
                    stream,
                )?;
                anyhow::Result::<()>::Ok(())
            })?;
        } else if num_tokens == 2 {
            // Fused K=2 MoE: 5 kernel launches instead of 10
            self.ffn.forward_k2(normed2_base, ctx, stream)?;
            // Batched residual add for 2 tokens (flat element-wise, 2*h elements)
            let moe_out = ctx.buffers.moe_output();
            ops::residual_add(
                ctx.gpu,
                self.residual_add_k,
                hidden,
                moe_out,
                (2 * h) as u32,
                stream,
            )?;
        } else {
            // Per-token MoE fallback for K!=2.
            // CONCURRENT-DECODE BUG (sibling of decode_multi_seq fix at line 1102):
            // hardcoded `t * h * 4` over-strides for BF16 hidden (GB10 default).
            let residual_elem = if ctx.config.use_fp32_residual() {
                4usize
            } else {
                2usize
            };

            // Batched K=γ FFN path: replaces the per-token GEMV loop
            // (each call re-reads ~134 MB of NVFP4 FFN weights from
            // LPDDR5X) with 3 GEMMs at M=num_tokens that load each
            // weight once per layer. Gated by `ATLAS_FFN_KGAMMA_M16=1`;
            // only available for dense FFN. The `residual_add_rms_norm`
            // above this branch already wrote `num_tokens` rows into
            // `normed2_base` contiguously, so the batched FFN can read
            // it directly without the per-token offsetting.
            //
            // Threshold n > 3 (re-verified 2026-05-21): the K=2 / K=3
            // branches above own num_tokens in {2, 3} via their fused
            // batch kernels. For num_tokens >= 4 the w4a16_gemm_t_m16
            // (M_TILE=16) kernel path is the fast option and was
            // re-validated to produce coherent output across the full
            // adaptive-truncate range {4..16}. A prior defensive gate
            // `>= 16` was a workaround for a transient drafter/adaptive
            // interaction that has since been resolved upstream; keeping
            // it suppressed the fast kernel on truncated-γ verifies,
            // costing the prose path 15-20 tok/s.
            let try_kgamma = (num_tokens as u32) > 3
                && (super::super::ffn_kgamma_m16_enabled()
                    || self.ffn.exact_kgamma_applicable(num_tokens as u32));
            let used_kgamma = if try_kgamma {
                let serviced = crate::kprof!(
                    ctx.gpu,
                    stream,
                    "ssm_ffn_kgamma_dense",
                    self.ffn
                        .forward_kgamma(normed2_base, num_tokens as u32, ctx, stream)
                )?;
                if serviced {
                    let moe_out = ctx.buffers.moe_output();
                    crate::kprof!(ctx.gpu, stream, "ssm_ffn_kgamma_resid", {
                        ops::residual_add(
                            ctx.gpu,
                            self.residual_add_k,
                            hidden,
                            moe_out,
                            (num_tokens as u32) * h as u32,
                            stream,
                        )?;
                        anyhow::Result::<()>::Ok(())
                    })?;
                }
                serviced
            } else {
                false
            };

            if !used_kgamma {
                crate::kprof!(ctx.gpu, stream, "ssm_ffn_per_token_loop_n17", {
                    for t in 0..(num_tokens as u32) {
                        let normed2 = normed2_base.offset(t as usize * h * bf16);
                        let moe_out = self.ffn.forward(normed2, ctx, stream)?;
                        let hidden_t = hidden.offset(t as usize * h * residual_elem);
                        ops::residual_add(
                            ctx.gpu,
                            self.residual_add_k,
                            hidden_t,
                            moe_out,
                            h as u32,
                            stream,
                        )?;
                    }
                    anyhow::Result::<()>::Ok(())
                })?;
            }
        }

        Ok(())
    }

    /// CROSS-SEQ BATCHED DFLASH VERIFY (#39): run the dense FFN ONCE over
    /// `total_rows` (= c×K) rows collected from every sequence's deferred
    /// mixer output, then residual-add into the shared `[total_rows, H]`
    /// hidden. This is the weight-amortizing step — the FFN NVFP4 weights are
    /// streamed once for all sequences instead of once per sequence. Mirrors
    /// the `num_tokens > 3` batched-FFN branch of `decode_batched_inner`
    /// (forward_kgamma at M=total_rows → WIDE m128 window when 32 < M ≤ 256),
    /// falling back to the per-row `forward()` loop when the batched kernel
    /// isn't serviced (e.g. MoE FFN, kgamma disabled).
    pub(super) fn run_deferred_ffn_inner(
        &self,
        ffn_input: DevicePtr,
        hidden: DevicePtr,
        total_rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let bf16 = 2usize;
        let residual_elem = if ctx.config.use_fp32_residual() {
            4usize
        } else {
            2usize
        };

        let try_kgamma = (total_rows as u32) > 3
            && (super::super::ffn_kgamma_m16_enabled()
                || self.ffn.exact_kgamma_applicable(total_rows as u32));
        let used_kgamma = if try_kgamma {
            let serviced = self
                .ffn
                .forward_kgamma(ffn_input, total_rows as u32, ctx, stream)?;
            if serviced {
                let moe_out = ctx.buffers.moe_output();
                ops::residual_add(
                    ctx.gpu,
                    self.residual_add_k,
                    hidden,
                    moe_out,
                    (total_rows as u32) * h as u32,
                    stream,
                )?;
            }
            serviced
        } else {
            false
        };

        if !used_kgamma {
            for t in 0..total_rows {
                let in_t = ffn_input.offset(t * h * bf16);
                let moe_out = self.ffn.forward(in_t, ctx, stream)?;
                let hidden_t = hidden.offset(t * h * residual_elem);
                ops::residual_add(
                    ctx.gpu,
                    self.residual_add_k,
                    hidden_t,
                    moe_out,
                    h as u32,
                    stream,
                )?;
            }
        }
        Ok(())
    }
}
