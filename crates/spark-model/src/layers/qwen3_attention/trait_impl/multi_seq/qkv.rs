// SPDX-License-Identifier: AGPL-3.0-only

//! Phase 2: per-token Q/K/V projection. Three branches:
//! - n=3 + NVFP4 → batch3 GEMV path
//! - n=2 + NVFP4 → batch2 GEMV path
//! - else        → sequential per-token GEMV (FP8/NVFP4/BF16 fallback)
//!
//! Both batch paths read each weight once for N tokens and then scatter
//! into the per-seq QKV layout. The sequential path repeats the GEMV per
//! token but supports every weight encoding.

use anyhow::Result;

use super::ctx::MultiSeqCtx;
use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;

/// Cached `ATLAS_ATTN_QKV_FUSED` env-var lookup. When `1`/`true` the
/// batch3 QKV projection writes directly into the interleaved
/// `qkv_buf` layout and uses a single batched RMS norm launch instead
/// of 6 separate ones. Default off for A/B safety.
fn attn_qkv_fused_enabled() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("ATLAS_ATTN_QKV_FUSED")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

/// Separate gate for the n>3 batched-M16 QKV path in multi_seq decode.
///
/// Distinct from `ATLAS_TC_NVFP4_M16` (which enables the same kernel for
/// SSM in_proj / SSM out_proj / prefill QKV — all verified). The multi_seq
/// attention QKV path produces numerically different output from the
/// per-token sequential GEMV path on gated Qwen3.6-27B layers when
/// `w4a16_gemm_n128_m16` is used with NVFP4-T weights + `deinterleave_qg`
/// fixup; verify accept rate collapses to 0%. Root cause not yet
/// identified — the dispatch wiring is correct (set_prefill_weights now
/// populates q_nvfp4_t / k_nvfp4_t / v_nvfp4_t for the qwen35_dense
/// loader; m16 + deinterleave_qg matches the FP8/dense gated decode
/// shape; SSM uses the same kernel at the same M=17 without issue).
/// Keeping the dispatch behind its own flag so production sets
/// `ATLAS_TC_NVFP4_M16=1` (for SSM/prefill gains) without enabling the
/// broken attention path. Set `ATLAS_TC_NVFP4_M16_MS_ATTN=1` to opt in.
pub(super) fn tc_nvfp4_m16_ms_attn_enabled() -> bool {
    static GATE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *GATE
        .get_or_init(|| std::env::var("ATLAS_TC_NVFP4_M16_MS_ATTN").ok().as_deref() == Some("1"))
}

impl Qwen3AttentionLayer {
    pub(super) fn ms_phase_qkv(&self, c: &MultiSeqCtx<'_>) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            h,
            nq,
            nkv,
            hd,
            eps,
            bf16,
            q_dim,
            q_proj_dim,
            q_proj_bytes,
            per_seq_qkv,
            normed,
            qkv_buf,
            ..
        } = *c;

        if n == 3
            && self.q_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some()
            && self.k_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some()
            && self.v_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some()
        {
            self.ms_qkv_batch3(c)?;
        } else if n == 2
            && self.q_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some()
            && self.k_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some()
            && self.v_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some()
        {
            self.ms_qkv_batch2(c)?;
        } else if n > 3
            && n <= 32
            && self.w4a16_gemm_t_m16_k.0 != 0
            && crate::layers::tc_nvfp4_m16_enabled()
            && tc_nvfp4_m16_ms_attn_enabled()
            && self.q_nvfp4_t.is_some()
            && self.k_nvfp4_t.is_some()
            && self.v_nvfp4_t.is_some()
        {
            self.ms_qkv_batched_m16(c)?;
        } else {
            for i in 0..n {
                let normed_i = normed.offset(i * h * bf16);
                let q_out_i = qkv_buf.offset(i * per_seq_qkv);
                let k_out_i = q_out_i.offset(q_proj_bytes);
                let v_out_i = k_out_i.offset((nkv * hd) as usize * bf16);

                self.ms_qkv_seq_q(fwd, normed_i, q_out_i, q_proj_dim, q_dim, nq, hd, h, stream)?;
                self.ms_qkv_seq_kv(fwd, normed_i, k_out_i, v_out_i, nkv, hd, h, stream)?;

                if !self.attn.q_norm.weight.is_null() {
                    ops::rms_norm(
                        fwd.gpu,
                        self.rms_norm_k,
                        q_out_i,
                        &self.attn.q_norm,
                        q_out_i,
                        nq,
                        hd,
                        eps,
                        stream,
                    )?;
                }
                if !self.attn.k_norm.weight.is_null() {
                    ops::rms_norm(
                        fwd.gpu,
                        self.rms_norm_k,
                        k_out_i,
                        &self.attn.k_norm,
                        k_out_i,
                        nkv,
                        hd,
                        eps,
                        stream,
                    )?;
                }
            }
        }
        Ok(())
    }

    /// n=3 NVFP4 batched path.
    fn ms_qkv_batch3(&self, c: &MultiSeqCtx<'_>) -> Result<()> {
        // Fused path: GEMVs write directly into the interleaved qkv_buf
        // layout and q_norm/k_norm fuse into a single 6-block kernel.
        if attn_qkv_fused_enabled()
            && self.gated
            && self.w4a16_gemv_qg_batch3_strided_k.0 != 0
            && self.w4a16_gemv_dual_batch3_strided_k.0 != 0
            && self.rms_norm_qk_batch3_k.0 != 0
            && !self.attn.q_norm.weight.is_null()
            && !self.attn.k_norm.weight.is_null()
        {
            return self.ms_qkv_batch3_fused(c);
        }

        let MultiSeqCtx {
            fwd,
            stream,
            h,
            nq,
            nkv,
            hd,
            eps,
            bf16,
            q_proj_dim,
            q_proj_bytes,
            per_seq_qkv,
            normed,
            qkv_buf,
            ..
        } = *c;
        let q_nvfp4 = self.q_weight.as_ref().and_then(|w| w.as_nvfp4()).unwrap();
        let k_nvfp4 = self.k_weight.as_ref().and_then(|w| w.as_nvfp4()).unwrap();
        let v_nvfp4 = self.v_weight.as_ref().and_then(|w| w.as_nvfp4()).unwrap();

        let q_scratch = fwd.buffers.ssm_qkvz();
        if self.gated {
            ops::w4a16_gemv_qg_batch3(
                fwd.gpu,
                self.w4a16_gemv_qg_batch3_k,
                normed,
                q_nvfp4,
                q_scratch,
                q_proj_dim,
                h as u32,
                nq,
                hd,
                stream,
            )?;
        } else {
            ops::w4a16_gemv_batch3(
                fwd.gpu,
                self.w4a16_gemv_batch3_k,
                normed,
                q_nvfp4,
                q_scratch,
                q_proj_dim,
                h as u32,
                stream,
            )?;
        }

        let kv_dim = nkv * hd;
        let kv_bytes = kv_dim as usize * bf16;
        let k_scratch = fwd.buffers.attn_output();
        let v_scratch = k_scratch.offset(3 * kv_bytes);
        ops::w4a16_gemv_dual_batch3(
            fwd.gpu,
            self.w4a16_gemv_dual_batch3_k,
            normed,
            k_nvfp4,
            k_scratch,
            v_nvfp4,
            v_scratch,
            kv_dim,
            h as u32,
            stream,
        )?;

        for i in 0..3usize {
            let q_out_i = qkv_buf.offset(i * per_seq_qkv);
            let k_out_i = q_out_i.offset(q_proj_bytes);
            let v_out_i = k_out_i.offset(kv_bytes);
            fwd.gpu.copy_d2d_async(
                q_scratch.offset(i * q_proj_bytes),
                q_out_i,
                q_proj_bytes,
                stream,
            )?;
            fwd.gpu
                .copy_d2d_async(k_scratch.offset(i * kv_bytes), k_out_i, kv_bytes, stream)?;
            fwd.gpu
                .copy_d2d_async(v_scratch.offset(i * kv_bytes), v_out_i, kv_bytes, stream)?;
        }

        for i in 0..3usize {
            let q_out_i = qkv_buf.offset(i * per_seq_qkv);
            let k_out_i = q_out_i.offset(q_proj_bytes);
            if !self.attn.q_norm.weight.is_null() {
                ops::rms_norm(
                    fwd.gpu,
                    self.rms_norm_k,
                    q_out_i,
                    &self.attn.q_norm,
                    q_out_i,
                    nq,
                    hd,
                    eps,
                    stream,
                )?;
            }
            if !self.attn.k_norm.weight.is_null() {
                ops::rms_norm(
                    fwd.gpu,
                    self.rms_norm_k,
                    k_out_i,
                    &self.attn.k_norm,
                    k_out_i,
                    nkv,
                    hd,
                    eps,
                    stream,
                )?;
            }
        }
        Ok(())
    }

    /// Fused n=3 NVFP4 batched QKV path (ATLAS_ATTN_QKV_FUSED=1).
    ///
    /// Differences vs `ms_qkv_batch3`:
    ///   - GEMVs use the `_strided` variants so they write directly into
    ///     the per-seq-strided `qkv_buf` (no scratch + 9× d2d copies).
    ///   - The 6 q_norm/k_norm launches collapse into one
    ///     `rms_norm_qk_batch3` launch (grid=(6,1,1)).
    fn ms_qkv_batch3_fused(&self, c: &MultiSeqCtx<'_>) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            stream,
            h,
            nq,
            nkv,
            hd,
            eps,
            bf16,
            q_proj_dim,
            q_proj_bytes,
            per_seq_qkv,
            normed,
            qkv_buf,
            ..
        } = *c;
        let q_nvfp4 = self.q_weight.as_ref().and_then(|w| w.as_nvfp4()).unwrap();
        let k_nvfp4 = self.k_weight.as_ref().and_then(|w| w.as_nvfp4()).unwrap();
        let v_nvfp4 = self.v_weight.as_ref().and_then(|w| w.as_nvfp4()).unwrap();

        // Per-token stride within qkv_buf in BF16 elements.
        // qkv_buf layout for token i: [Q_i (q_proj_bytes)] [K_i (kv_bytes)] [V_i (kv_bytes)]
        debug_assert!(per_seq_qkv % bf16 == 0, "qkv stride must be BF16-aligned");
        let stride_bf16 = (per_seq_qkv / bf16) as u32;
        let kv_dim = nkv * hd;
        let kv_bytes = kv_dim as usize * bf16;

        // Q + gate writes [Q_all|Gate_all] for each token at qkv_buf + i*stride.
        // Q region is the first `q_proj_bytes` (= q_proj_dim BF16) of every token's slot.
        ops::w4a16_gemv_qg_batch3_strided(
            fwd.gpu,
            self.w4a16_gemv_qg_batch3_strided_k,
            normed,
            q_nvfp4,
            qkv_buf,
            q_proj_dim,
            h as u32,
            nq,
            hd,
            stride_bf16,
            stream,
        )?;

        // K writes at qkv_buf + i*stride + q_proj_bytes
        // V writes at qkv_buf + i*stride + q_proj_bytes + kv_bytes
        let k_base = qkv_buf.offset(q_proj_bytes);
        let v_base = k_base.offset(kv_bytes);
        ops::w4a16_gemv_dual_batch3_strided(
            fwd.gpu,
            self.w4a16_gemv_dual_batch3_strided_k,
            normed,
            k_nvfp4,
            k_base,
            v_nvfp4,
            v_base,
            kv_dim,
            h as u32,
            stride_bf16,
            stream,
        )?;

        // Batched per-head q_norm + k_norm for all 3 tokens.
        // `q_dim` for the norm is `nq * hd` (the actual Q region — gate is
        // not normed). `k_offset` is `q_proj_dim` BF16 elements: for gated
        // layers q_proj_dim == 2*nq*hd because of the Gate slab between Q
        // and K. For ungated, q_proj_dim == nq*hd == q_dim.
        ops::rms_norm_qk_batch3(
            fwd.gpu,
            self.rms_norm_qk_batch3_k,
            qkv_buf,
            &self.attn.q_norm,
            &self.attn.k_norm,
            stride_bf16,
            nq * hd,
            nkv * hd,
            q_proj_dim,
            hd,
            eps,
            stream,
        )?;
        Ok(())
    }

    /// n=2 NVFP4 batched path.
    fn ms_qkv_batch2(&self, c: &MultiSeqCtx<'_>) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            stream,
            h,
            nq,
            nkv,
            hd,
            eps,
            bf16,
            q_proj_dim,
            q_proj_bytes,
            per_seq_qkv,
            normed,
            qkv_buf,
            ..
        } = *c;
        let q_nvfp4 = self.q_weight.as_ref().and_then(|w| w.as_nvfp4()).unwrap();
        let k_nvfp4 = self.k_weight.as_ref().and_then(|w| w.as_nvfp4()).unwrap();
        let v_nvfp4 = self.v_weight.as_ref().and_then(|w| w.as_nvfp4()).unwrap();

        let q_scratch = fwd.buffers.ssm_qkvz();
        if self.gated {
            ops::w4a16_gemv_qg_batch2(
                fwd.gpu,
                self.w4a16_gemv_qg_batch2_k,
                normed,
                q_nvfp4,
                q_scratch,
                q_proj_dim,
                h as u32,
                nq,
                hd,
                stream,
            )?;
        } else {
            ops::w4a16_gemv_batch2(
                fwd.gpu,
                self.w4a16_gemv_batch2_k,
                normed,
                q_nvfp4,
                q_scratch,
                q_proj_dim,
                h as u32,
                stream,
            )?;
        }

        let kv_dim = nkv * hd;
        let kv_bytes = kv_dim as usize * bf16;
        let k_scratch = fwd.buffers.attn_output();
        let v_scratch = k_scratch.offset(2 * kv_bytes);
        ops::w4a16_gemv_dual_batch2(
            fwd.gpu,
            self.w4a16_gemv_dual_batch2_k,
            normed,
            k_nvfp4,
            k_scratch,
            v_nvfp4,
            v_scratch,
            kv_dim,
            h as u32,
            stream,
        )?;

        for i in 0..2usize {
            let q_out_i = qkv_buf.offset(i * per_seq_qkv);
            let k_out_i = q_out_i.offset(q_proj_bytes);
            let v_out_i = k_out_i.offset(kv_bytes);
            fwd.gpu.copy_d2d_async(
                q_scratch.offset(i * q_proj_bytes),
                q_out_i,
                q_proj_bytes,
                stream,
            )?;
            fwd.gpu
                .copy_d2d_async(k_scratch.offset(i * kv_bytes), k_out_i, kv_bytes, stream)?;
            fwd.gpu
                .copy_d2d_async(v_scratch.offset(i * kv_bytes), v_out_i, kv_bytes, stream)?;
        }

        for i in 0..2usize {
            let q_out_i = qkv_buf.offset(i * per_seq_qkv);
            let k_out_i = q_out_i.offset(q_proj_bytes);
            if !self.attn.q_norm.weight.is_null() {
                ops::rms_norm(
                    fwd.gpu,
                    self.rms_norm_k,
                    q_out_i,
                    &self.attn.q_norm,
                    q_out_i,
                    nq,
                    hd,
                    eps,
                    stream,
                )?;
            }
            if !self.attn.k_norm.weight.is_null() {
                ops::rms_norm(
                    fwd.gpu,
                    self.rms_norm_k,
                    k_out_i,
                    &self.attn.k_norm,
                    k_out_i,
                    nkv,
                    hd,
                    eps,
                    stream,
                )?;
            }
        }
        Ok(())
    }

    /// Batched M=16 NVFP4 QKV path for K=γ verify (DFlash γ>3, n=4..=32).
    ///
    /// Replaces the per-token sequential GEMV loop (3*n launches per layer)
    /// with three batched GEMMs (3 launches per layer) using the small-M
    /// specialization `w4a16_gemm_t_m16` (M_TILE=16, redesigned warp
    /// partitioning so all 4 warps share the 16 rows). Requires transposed
    /// NVFP4 weights (`q_nvfp4_t`/`k_nvfp4_t`/`v_nvfp4_t`).
    ///
    /// Output staging: Q goes to `ssm_qkvz` scratch [n, q_proj_dim], K/V go
    /// to `attn_output` scratch [2n, kv_dim] (K then V). Per-token d2d copies
    /// scatter into the strided `qkv_buf` layout. Per-token rms_norm calls
    /// follow (same as the n=3/n=2 batched paths — rms_norm is cheap, the
    /// QKV GEMVs are the bottleneck at n=17).
    fn ms_qkv_batched_m16(&self, c: &MultiSeqCtx<'_>) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            h,
            nq,
            nkv,
            hd,
            eps,
            bf16,
            q_proj_dim,
            q_proj_bytes,
            per_seq_qkv,
            normed,
            qkv_buf,
            ..
        } = *c;

        let q_nvfp4_t = self.q_nvfp4_t.as_ref().unwrap();
        let k_nvfp4_t = self.k_nvfp4_t.as_ref().unwrap();
        let v_nvfp4_t = self.v_nvfp4_t.as_ref().unwrap();

        let n_u32 = n as u32;
        let kv_dim = nkv * hd;
        let kv_bytes = kv_dim as usize * bf16;

        // Q scratch: [n, q_proj_dim] BF16 contiguous in ssm_qkvz.
        let q_scratch = fwd.buffers.ssm_qkvz();
        ops::w4a16_gemm_n128_m16(
            fwd.gpu,
            self.w4a16_gemm_t_m16_k,
            normed,
            q_nvfp4_t,
            q_scratch,
            n_u32,
            q_proj_dim,
            h as u32,
            stream,
        )?;

        // K scratch: [n, kv_dim] BF16 contiguous in attn_output.
        // V scratch: [n, kv_dim] BF16 contiguous at attn_output + n*kv_bytes.
        let k_scratch = fwd.buffers.attn_output();
        let v_scratch = k_scratch.offset(n * kv_bytes);
        ops::w4a16_gemm_n128_m16(
            fwd.gpu,
            self.w4a16_gemm_t_m16_k,
            normed,
            k_nvfp4_t,
            k_scratch,
            n_u32,
            kv_dim,
            h as u32,
            stream,
        )?;
        ops::w4a16_gemm_n128_m16(
            fwd.gpu,
            self.w4a16_gemm_t_m16_k,
            normed,
            v_nvfp4_t,
            v_scratch,
            n_u32,
            kv_dim,
            h as u32,
            stream,
        )?;

        // Scatter each token's (Q, K, V) into the per-seq-strided qkv_buf.
        // qkv_buf layout per token i: [Q (q_proj_bytes) | K (kv_bytes) | V (kv_bytes)].
        //
        // For gated layers the GEMM produces Q+Gate interleaved per head
        // ([Q_h0, G_h0, Q_h1, G_h1, ...]). The downstream multi_seq attention
        // path expects deinterleaved layout ([Q_h0..Q_hN-1, G_h0..G_hN-1])
        // since `ms_phase_o_proj` reads Gate from `q_dim*bf16` offset. We
        // apply `deinterleave_qg` per token after the scatter — same fixup
        // as `ms_qkv_seq_q` does for FP8/Dense gated.
        for i in 0..n {
            let q_out_i = qkv_buf.offset(i * per_seq_qkv);
            let k_out_i = q_out_i.offset(q_proj_bytes);
            let v_out_i = k_out_i.offset(kv_bytes);
            fwd.gpu.copy_d2d_async(
                q_scratch.offset(i * q_proj_bytes),
                q_out_i,
                q_proj_bytes,
                stream,
            )?;
            fwd.gpu
                .copy_d2d_async(k_scratch.offset(i * kv_bytes), k_out_i, kv_bytes, stream)?;
            fwd.gpu
                .copy_d2d_async(v_scratch.offset(i * kv_bytes), v_out_i, kv_bytes, stream)?;
            if self.gated {
                ops::deinterleave_qg(
                    fwd.gpu,
                    self.deinterleave_qg_k,
                    q_out_i,
                    1,
                    nq,
                    hd,
                    q_proj_dim,
                    stream,
                )?;
            }
        }

        // Per-token RMS norm (q_norm/k_norm). Each is a tiny per-head
        // single-block kernel — not the bottleneck. Keeping per-token here
        // preserves correctness for gated layers (Q + Gate share q_proj_dim)
        // without needing a strided rms_norm variant.
        for i in 0..n {
            let q_out_i = qkv_buf.offset(i * per_seq_qkv);
            let k_out_i = q_out_i.offset(q_proj_bytes);
            if !self.attn.q_norm.weight.is_null() {
                ops::rms_norm(
                    fwd.gpu,
                    self.rms_norm_k,
                    q_out_i,
                    &self.attn.q_norm,
                    q_out_i,
                    nq,
                    hd,
                    eps,
                    stream,
                )?;
            }
            if !self.attn.k_norm.weight.is_null() {
                ops::rms_norm(
                    fwd.gpu,
                    self.rms_norm_k,
                    k_out_i,
                    &self.attn.k_norm,
                    k_out_i,
                    nkv,
                    hd,
                    eps,
                    stream,
                )?;
            }
        }
        Ok(())
    }

    /// Sequential per-token Q projection (handles gated and ungated).
    #[allow(clippy::too_many_arguments)]
    fn ms_qkv_seq_q(
        &self,
        fwd: &crate::layer::ForwardContext<'_>,
        normed_i: spark_runtime::gpu::DevicePtr,
        q_out_i: spark_runtime::gpu::DevicePtr,
        q_proj_dim: u32,
        q_dim: u32,
        nq: u32,
        hd: u32,
        h: usize,
        stream: u64,
    ) -> Result<()> {
        if self.gated {
            if let Some(fp8) = self.q_weight.as_ref().and_then(|w| w.as_fp8()) {
                ops::w8a16_gemv(
                    fwd.gpu,
                    self.w8a16_gemv_k,
                    normed_i,
                    fp8.weight,
                    fp8.row_scale,
                    q_out_i,
                    q_proj_dim,
                    h as u32,
                    stream,
                )?;
                ops::deinterleave_qg(
                    fwd.gpu,
                    self.deinterleave_qg_k,
                    q_out_i,
                    1,
                    nq,
                    hd,
                    q_proj_dim,
                    stream,
                )?;
            } else if let Some(nvfp4) = self.q_weight.as_ref().and_then(|w| w.as_nvfp4()) {
                ops::w4a16_gemv_qg(
                    fwd.gpu,
                    self.w4a16_gemv_qg_k,
                    normed_i,
                    nvfp4,
                    q_out_i,
                    q_proj_dim,
                    h as u32,
                    nq,
                    hd,
                    stream,
                )?;
            } else {
                ops::dense_gemv(
                    fwd.gpu,
                    self.dense_gemv_k,
                    normed_i,
                    &self.attn.q_proj,
                    q_out_i,
                    q_proj_dim,
                    h as u32,
                    stream,
                )?;
                ops::deinterleave_qg(
                    fwd.gpu,
                    self.deinterleave_qg_k,
                    q_out_i,
                    1,
                    nq,
                    hd,
                    q_proj_dim,
                    stream,
                )?;
            }
        } else if let Some(fp8) = self.q_weight.as_ref().and_then(|w| w.as_fp8()) {
            ops::w8a16_gemv(
                fwd.gpu,
                self.w8a16_gemv_k,
                normed_i,
                fp8.weight,
                fp8.row_scale,
                q_out_i,
                q_dim,
                h as u32,
                stream,
            )?;
        } else if let Some(nvfp4) = self.q_weight.as_ref().and_then(|w| w.as_nvfp4()) {
            ops::w4a16_gemv(
                fwd.gpu,
                self.w4a16_gemv_k,
                normed_i,
                nvfp4,
                q_out_i,
                q_dim,
                h as u32,
                stream,
            )?;
        } else {
            ops::dense_gemv(
                fwd.gpu,
                self.dense_gemv_k,
                normed_i,
                &self.attn.q_proj,
                q_out_i,
                q_dim,
                h as u32,
                stream,
            )?;
        }
        Ok(())
    }

    /// Sequential per-token K + V projections.
    #[allow(clippy::too_many_arguments)]
    fn ms_qkv_seq_kv(
        &self,
        fwd: &crate::layer::ForwardContext<'_>,
        normed_i: spark_runtime::gpu::DevicePtr,
        k_out_i: spark_runtime::gpu::DevicePtr,
        v_out_i: spark_runtime::gpu::DevicePtr,
        nkv: u32,
        hd: u32,
        h: usize,
        stream: u64,
    ) -> Result<()> {
        if let (Some(k_fp8), Some(v_fp8)) = (
            self.k_weight.as_ref().and_then(|w| w.as_fp8()),
            self.v_weight.as_ref().and_then(|w| w.as_fp8()),
        ) {
            ops::w8a16_gemv(
                fwd.gpu,
                self.w8a16_gemv_k,
                normed_i,
                k_fp8.weight,
                k_fp8.row_scale,
                k_out_i,
                nkv * hd,
                h as u32,
                stream,
            )?;
            ops::w8a16_gemv(
                fwd.gpu,
                self.w8a16_gemv_k,
                normed_i,
                v_fp8.weight,
                v_fp8.row_scale,
                v_out_i,
                nkv * hd,
                h as u32,
                stream,
            )?;
        } else if let (Some(k_fp4), Some(v_fp4)) = (
            self.k_weight.as_ref().and_then(|w| w.as_nvfp4()),
            self.v_weight.as_ref().and_then(|w| w.as_nvfp4()),
        ) {
            ops::w4a16_gemv_dual(
                fwd.gpu,
                self.w4a16_gemv_dual_k,
                normed_i,
                k_fp4,
                k_out_i,
                v_fp4,
                v_out_i,
                nkv * hd,
                h as u32,
                stream,
            )?;
        } else {
            if let Some(nvfp4) = self.k_weight.as_ref().and_then(|w| w.as_nvfp4()) {
                ops::w4a16_gemv(
                    fwd.gpu,
                    self.w4a16_gemv_k,
                    normed_i,
                    nvfp4,
                    k_out_i,
                    nkv * hd,
                    h as u32,
                    stream,
                )?;
            } else {
                ops::dense_gemv(
                    fwd.gpu,
                    self.dense_gemv_k,
                    normed_i,
                    &self.attn.k_proj,
                    k_out_i,
                    nkv * hd,
                    h as u32,
                    stream,
                )?;
            }
            if let Some(nvfp4) = self.v_weight.as_ref().and_then(|w| w.as_nvfp4()) {
                ops::w4a16_gemv(
                    fwd.gpu,
                    self.w4a16_gemv_k,
                    normed_i,
                    nvfp4,
                    v_out_i,
                    nkv * hd,
                    h as u32,
                    stream,
                )?;
            } else {
                ops::dense_gemv(
                    fwd.gpu,
                    self.dense_gemv_k,
                    normed_i,
                    &self.attn.v_proj,
                    v_out_i,
                    nkv * hd,
                    h as u32,
                    stream,
                )?;
            }
        }
        Ok(())
    }
}
