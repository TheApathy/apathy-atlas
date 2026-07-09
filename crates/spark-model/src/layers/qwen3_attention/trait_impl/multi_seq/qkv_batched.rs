// SPDX-License-Identifier: AGPL-3.0-only

//! CROSS-SEQ BATCHED DFLASH VERIFY (#39 v2): batched QKV projection over
//! `c*K` rows.
//!
//! v1 ran the whole multi-seq attention (RMS-norm → QKV → RoPE → decode →
//! o_proj → deferred FFN) PER SEQUENCE, so the Q/K/V weights on every
//! FullAttention layer were read `c` times. This module hoists just the
//! WEIGHT-heavy Q/K/V projection out of the per-seq loop: one RMS-norm + one
//! batched GEMM per weight over ALL `c*K` rows, reading each Q/K/V weight
//! ONCE. The output lands in the shared `qkv_output` buffer in the per-seq
//! `per_seq_qkv`-strided layout that phases 3-7 already consume, so the
//! per-seq RoPE / cache-write / paged-decode / o_proj (which carry per-seq KV
//! state and cannot batch) read their `[s*K, (s+1)*K)` slice unchanged.
//!
//! ## Losslessness
//!
//! Bit-identical to the per-seq QKV: the projection GEMM is a deterministic
//! `C = A @ Wᵀ` whose per-row output does not depend on which other rows share
//! the launch (pure M-growth). The `w4a16_gemm_n128_m128` wide kernel is the
//! same one the FFN uses for `32 < M ≤ 256`, extended here past M=32 (its grid
//! tiles the M axis by 128, so M=c*K up to max_batch_tokens is handled). The
//! scatter + gated `deinterleave_qg` + per-head q/k norm are identical to
//! `ms_qkv_batched_plain` — only the row count grows.

use anyhow::{Result, bail};
use spark_runtime::gpu::DevicePtr;

use super::ctx::MultiSeqCtx;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;

impl Qwen3AttentionLayer {
    /// RMS-norm (`input_norm`) + batched Q/K/V projection over `num_rows`
    /// contiguous rows of `hidden`, writing into `qkv_out_base` in the
    /// `per_seq_qkv`-strided `[Q | K | V]` layout.
    ///
    /// `num_rows` may exceed 32 (unlike the `ms_qkv_batched_plain` path used
    /// inside `ms_phase_qkv`): the wide `w4a16_gemm_n128_m128` kernel tiles
    /// the M axis, so `c*K` rows project in one weight read per Q/K/V.
    ///
    /// Requires the transposed NVFP4 Q/K/V weights (`q_nvfp4_t` etc.), which
    /// are installed for the aeon-27b verify config. Bails otherwise so the
    /// caller can fall back to the per-seq path (v1).
    #[allow(clippy::too_many_arguments)]
    pub(in crate::layers::qwen3_attention) fn decode_multi_seq_qkv_batched_inner(
        &self,
        hidden_in: DevicePtr,
        num_rows: usize,
        qkv_out_base: DevicePtr,
        ctx: &ForwardContext<'_>,
        stream: u64,
    ) -> Result<()> {
        // The batched projection reads scalars/buffers from a MultiSeqCtx.
        // `residual`/`max_seq_len_host`/`bs` are unused by the projection (it
        // only norms `hidden` and writes `qkv_out_base`); pass placeholders.
        let c = MultiSeqCtx::new(self, ctx, hidden_in, hidden_in, num_rows, 1, stream, 0);
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
            hidden,
            ..
        } = c;

        // The wide batched path needs the transposed NVFP4 weights + the
        // m128 kernel. If either is missing (non-aeon config), refuse so the
        // caller keeps the validated per-seq route.
        let ok = self.w4a16_gemm_t_m128_k.0 != 0
            && self.q_nvfp4_t.is_some()
            && self.k_nvfp4_t.is_some()
            && self.v_nvfp4_t.is_some();
        if !ok {
            bail!("decode_multi_seq_qkv_batched: transposed QKV weights / m128 kernel unavailable");
        }
        // FP32-residual builds route phase-1 RMS-norm through the FP32-input
        // `rms_norm_residual_fp32` kernel (the hidden buffer is FP32-strided).
        // Our batched `rms_norm_k` is the BF16-input kernel, so it would
        // mis-stride an FP32 hidden buffer. Refuse and let the caller keep the
        // per-seq v1 path (which binds the correct FP32 kernel). Production
        // GB10 aeon-27b uses BF16 residual, so the fast path stays active.
        if fwd.config.use_fp32_residual() {
            bail!("decode_multi_seq_qkv_batched: FP32 residual — per-seq path owns the FP32 norm");
        }

        // ── RMS-norm over all `num_rows` rows into `norm_output` ──
        // `residual` is NOT updated here: phase 1 (`rms_norm_residual`) in the
        // per-seq attention path both norms AND seeds the residual buffer.
        // We only need the norm as the QKV input; the per-seq attention still
        // owns the residual seeding for its K rows (phases 3-7 read residual
        // from `hidden`, and the deferred-FFN phase re-derives the post-attn
        // residual). So mirror ONLY the norm half here.
        let normed = fwd.buffers.norm_output();
        ops::rms_norm(
            fwd.gpu,
            self.rms_norm_k,
            hidden,
            &self.input_norm,
            normed,
            num_rows as u32,
            h as u32,
            eps,
            stream,
        )?;

        let m = num_rows as u32;
        let kv_dim = nkv * hd;
        let kv_bytes = kv_dim as usize * bf16;

        let q_t = self.q_nvfp4_t.as_ref().unwrap();
        let k_t = self.k_nvfp4_t.as_ref().unwrap();
        let v_t = self.v_nvfp4_t.as_ref().unwrap();

        // Q scratch: [num_rows, q_proj_dim] BF16 contiguous in ssm_qkvz.
        // K scratch: [num_rows, kv_dim] BF16 contiguous in attn_output.
        // V scratch: [num_rows, kv_dim] BF16 contiguous at attn_output + m*kv_bytes.
        let q_scratch = fwd.buffers.ssm_qkvz();
        let k_scratch = fwd.buffers.attn_output();
        let v_scratch = k_scratch.offset(num_rows * kv_bytes);

        ops::w4a16_gemm_n128_m128(
            fwd.gpu,
            self.w4a16_gemm_t_m128_k,
            normed,
            q_t,
            q_scratch,
            m,
            q_proj_dim,
            h as u32,
            stream,
        )?;
        ops::w4a16_gemm_n128_m128(
            fwd.gpu,
            self.w4a16_gemm_t_m128_k,
            normed,
            k_t,
            k_scratch,
            m,
            kv_dim,
            h as u32,
            stream,
        )?;
        ops::w4a16_gemm_n128_m128(
            fwd.gpu,
            self.w4a16_gemm_t_m128_k,
            normed,
            v_t,
            v_scratch,
            m,
            kv_dim,
            h as u32,
            stream,
        )?;

        // Scatter each row's (Q, K, V) into the per-seq-strided qkv_out_base,
        // apply the gated Q/Gate deinterleave, then per-head q/k RMS norm.
        // Identical semantics to `ms_qkv_batched_plain`, one iteration per row.
        for i in 0..num_rows {
            let q_out_i = qkv_out_base.offset(i * per_seq_qkv);
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
}
