// SPDX-License-Identifier: AGPL-3.0-only

//! Phases 3-6: per-sequence RoPE, KV-cache write, batched paged
//! attention, gate multiply + O projection.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::{KvCacheDtype, PagedKvCache};

use super::ctx::MultiSeqCtx;
use crate::layer::AttnMetadataDev;
use crate::layers::ops;
use crate::layers::qwen3_attention::HeadGateActivation;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;

impl Qwen3AttentionLayer {
    /// ATLAS_FUSED_ELEMWISE=1 flat-verify epilogue eligibility. Every term is
    /// load-fixed (weights/kernels/dtypes) or capture-shape-fixed (`n`), so
    /// the decision is stable across a CUDA-graph capture and its replays.
    ///
    /// Requirements mirror exactly what the fused kernel implements:
    /// wide (n>3) contiguous-scratch QKV branch (batchn NVFP4 / FP8-mirror /
    /// BF16-dense), ungated, no LoRA, per-head q/k norms present, table-based
    /// yarn-scaled RoPE, BF16 paged KV cache, head_dim ≤ 256.
    pub(super) fn ms_fused_epilogue_eligible(&self, c: &MultiSeqCtx<'_>) -> bool {
        if !ops::fused_elemwise_enabled() || self.fused_qkv_norm_rope_cache_k.0 == 0 {
            return false;
        }
        if c.n <= 3 {
            return false; // batch2/batch3/seq branches scatter differently
        }
        if self.gated || self.lora.is_some() || self.mla.is_some() || self.hc.is_some() {
            return false;
        }
        if self.kv_dtype != KvCacheDtype::Bf16 {
            return false; // fused kernel writes a BF16 paged cache only
        }
        if self.yarn_inv_freq.is_null() {
            return false; // plain-theta rope path not implemented in the fused kernel
        }
        if self.attn.q_norm.weight.is_null() || self.attn.k_norm.weight.is_null() {
            return false;
        }
        if c.hd == 0 || !c.hd.is_multiple_of(2) || c.hd > 256 {
            return false;
        }
        let rot = self
            .rotary_dim_override
            .unwrap_or(c.fwd.config.rotary_dim() as u32);
        if rot < 2 || !rot.is_multiple_of(2) || rot > c.hd {
            return false;
        }
        // Must land on a wide contiguous-scratch projection branch of
        // `ms_phase_qkv` (they write [n, dim] GEMM outputs the fused kernel
        // consumes). Mirrors that dispatch chain for n > 3.
        let nvfp4 = self.q_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some()
            && self.k_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some()
            && self.v_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some();
        let bf16_dense = self.q_weight.is_none() && !self.attn.q_proj.weight.is_null();
        nvfp4 || self.attn_fp8_mirrors_present() || bf16_dense
    }

    /// Fused phases 3+4 (+ the deferred per-head norms, the scatter the QKV
    /// branch skipped, and the Q gather phase 5 will skip): ONE launch over
    /// the contiguous GEMM outputs. Bit-identical to the unfused chain — see
    /// kernels/gb10/common/fused_verify_elemwise.cu.
    pub(super) fn ms_fused_qk_epilogue(
        &self,
        c: &MultiSeqCtx<'_>,
        kv_cache: &PagedKvCache,
        meta: AttnMetadataDev,
    ) -> Result<()> {
        // Contiguous scratch layout shared by every wide (n>3) QKV branch:
        // Q at ssm_qkvz [n, q_dim], K at attn_output [n, kv_dim], V after K.
        let q_scratch = c.fwd.buffers.ssm_qkvz();
        let kv_bytes = (c.nkv * c.hd) as usize * c.bf16;
        let k_scratch = c.fwd.buffers.attn_output();
        let v_scratch = k_scratch.offset(c.n * kv_bytes);
        ops::fused_qkv_norm_rope_cache_write(
            c.fwd.gpu,
            self.fused_qkv_norm_rope_cache_k,
            q_scratch,
            k_scratch,
            v_scratch,
            &self.attn.q_norm,
            &self.attn.k_norm,
            meta.positions,
            self.yarn_inv_freq,
            kv_cache.k_pool_ptr(self.attn_layer_idx),
            kv_cache.v_pool_ptr(self.attn_layer_idx),
            meta.slot,
            c.n as u32,
            c.nq,
            c.nkv,
            c.hd,
            self.rotary_dim_override
                .unwrap_or(c.fwd.config.rotary_dim() as u32),
            c.bs,
            c.eps,
            self.yarn_attention_factor,
            u32::from(!self.norm_vanilla),
            c.stream,
        )
    }
    /// Phase 3: per-token RoPE (each sequence has its own position).
    pub(super) fn ms_phase_rope(&self, c: &MultiSeqCtx<'_>, meta: AttnMetadataDev) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            nq,
            nkv,
            hd,
            q_proj_bytes,
            per_seq_qkv,
            qkv_buf,
            ..
        } = *c;
        for i in 0..n {
            let q_out_i = qkv_buf.offset(i * per_seq_qkv);
            let k_out_i = q_out_i.offset(q_proj_bytes);
            let pos_i = meta.positions.offset(i * 4); // u32 per position
            if self.yarn_inv_freq.is_null() {
                ops::rope(
                    fwd.gpu,
                    self.rope_k,
                    q_out_i,
                    k_out_i,
                    pos_i,
                    1,
                    nq,
                    nkv,
                    hd,
                    self.rotary_dim_override
                        .unwrap_or(fwd.config.rotary_dim() as u32),
                    self.rope_theta_override
                        .unwrap_or(fwd.config.rope_theta as f32),
                    stream,
                )?;
            } else {
                ops::rope_yarn_scaled(
                    fwd.gpu,
                    self.rope_yarn_scaled_k,
                    q_out_i,
                    k_out_i,
                    pos_i,
                    1,
                    nq,
                    nkv,
                    hd,
                    self.rotary_dim_override
                        .unwrap_or(fwd.config.rotary_dim() as u32),
                    self.yarn_inv_freq,
                    self.yarn_attention_factor,
                    stream,
                )?;
            }
        }
        Ok(())
    }

    /// Phase 4: per-token KV cache write.
    pub(super) fn ms_phase_cache_write(
        &self,
        c: &MultiSeqCtx<'_>,
        kv_cache: &mut PagedKvCache,
        meta: AttnMetadataDev,
    ) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            nkv,
            hd,
            bs,
            bf16,
            q_proj_bytes,
            per_seq_qkv,
            qkv_buf,
            ..
        } = *c;
        let kv_stride = nkv * hd;
        for i in 0..n {
            let q_out_i = qkv_buf.offset(i * per_seq_qkv);
            let k_out_i = q_out_i.offset(q_proj_bytes);
            let v_out_i = k_out_i.offset((nkv * hd) as usize * bf16);
            let slot_i = meta.slot.offset(i * 8); // i64 per slot
            self.write_kv_cache(
                fwd.gpu,
                k_out_i,
                v_out_i,
                kv_cache,
                slot_i,
                1,
                nkv,
                hd,
                bs,
                kv_stride,
                kv_stride,
                stream,
                fwd.graph_capture,
            )?;
        }
        Ok(())
    }

    /// Phase 4 (tree-verify variant): KV cache write restricted to rows
    /// `[start_row, end_row)`.
    ///
    /// DDTree M2b: the tree verify writes bonus+spine rows into canonical
    /// blocks FIRST, then d2d-seeds each branch's scratch blocks, then
    /// writes the branch rows through their scratch slots. The QKV buffer
    /// and metadata are row-indexed, so this is the same per-row loop as
    /// [`Self::ms_phase_cache_write`] with explicit bounds. The full-range
    /// call used by the flat multiseq path is deliberately untouched.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn ms_phase_cache_write_range(
        &self,
        c: &MultiSeqCtx<'_>,
        kv_cache: &mut PagedKvCache,
        meta: AttnMetadataDev,
        start_row: usize,
        end_row: usize,
    ) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            stream,
            nkv,
            hd,
            bs,
            bf16,
            q_proj_bytes,
            per_seq_qkv,
            qkv_buf,
            ..
        } = *c;
        let kv_stride = nkv * hd;
        for i in start_row..end_row {
            let q_out_i = qkv_buf.offset(i * per_seq_qkv);
            let k_out_i = q_out_i.offset(q_proj_bytes);
            let v_out_i = k_out_i.offset((nkv * hd) as usize * bf16);
            let slot_i = meta.slot.offset(i * 8); // i64 per slot
            self.write_kv_cache(
                fwd.gpu,
                k_out_i,
                v_out_i,
                kv_cache,
                slot_i,
                1,
                nkv,
                hd,
                bs,
                kv_stride,
                kv_stride,
                stream,
                fwd.graph_capture,
            )?;
        }
        Ok(())
    }

    /// Phase 5: build contiguous Q buffer + run BATCHED paged decode.
    /// Returns the attn_out buffer pointer for downstream phases.
    pub(super) fn ms_phase_paged_decode(
        &self,
        c: &MultiSeqCtx<'_>,
        kv_cache: &mut PagedKvCache,
        meta: AttnMetadataDev,
    ) -> Result<DevicePtr> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            nq,
            nkv,
            hd,
            bs,
            bf16,
            q_dim,
            per_seq_qkv,
            qkv_buf,
            ..
        } = *c;
        // Build contiguous Q buffer [N, nq*hd] for batched attention.
        // FUSED epilogue (ATLAS_FUSED_ELEMWISE=1): ssm_qkvz ALREADY holds the
        // normed+roped Q contiguously (the fused kernel worked in place on the
        // Q GEMM output there) — the per-row gather is redundant.
        let q_contiguous = fwd.buffers.ssm_qkvz();
        if !c.fused_qk_epilogue {
            for i in 0..n {
                let q_out_i = qkv_buf.offset(i * per_seq_qkv);
                fwd.gpu.copy_d2d_async(
                    q_out_i,
                    q_contiguous.offset(i * q_dim as usize * bf16),
                    q_dim as usize * bf16,
                    stream,
                )?;
            }
        }
        let attn_out = fwd.buffers.attn_output();
        let inv_sqrt_d = self.effective_attn_scale(hd);

        // TurboQuant WHT bookends (mirrors decode/attention_forward.rs).
        // The cache holds WHT(K)/WHT(V) for turbo dtypes: rotate the batched
        // Q rows before the paged decode and rotate the output back after —
        // without these the multi-seq batched decode scores raw Q against
        // rotated K and returns output in the rotated-V basis.
        let (wht_k_dtype, wht_v_dtype) = self.kv_dtype.kv_pair();
        let k_is_turbo = wht_k_dtype.is_wht_rotated();
        let v_is_turbo = wht_v_dtype.is_wht_rotated();
        let weight_pre_rotated = std::env::var("TQ_PLUS_WEIGHT_ROTATION")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let wht_runtime_active = !weight_pre_rotated && (hd == 128 || hd == 256 || hd == 512);
        if k_is_turbo && self.innerq_apply_q_k.0 != 0 && hd == 128 {
            use spark_runtime::kernel_args::KernelLaunch;
            KernelLaunch::new(fwd.gpu, self.innerq_apply_q_k)
                .grid([n as u32 * nq, 1, 1])
                .block([32, 1, 1])
                .arg_ptr(q_contiguous)
                .arg_u32(hd)
                .launch(stream)?;
        }
        if k_is_turbo && wht_runtime_active && self.wht_bf16_k.0 != 0 {
            use spark_runtime::kernel_args::KernelLaunch;
            KernelLaunch::new(fwd.gpu, self.wht_bf16_k)
                .grid([n as u32 * nq, 1, 1]) // one warp per (seq, q_head)
                .block([32, 1, 1])
                .arg_ptr(q_contiguous)
                .arg_u32(hd)
                .launch(stream)?;
        }
        self.run_paged_decode(
            fwd.gpu,
            q_contiguous,
            kv_cache,
            attn_out,
            meta.block_table,
            meta.seq_len,
            meta.max_blocks_per_seq,
            n as u32,
            nq,
            nkv,
            hd,
            bs,
            inv_sqrt_d,
            nq * hd,
            fwd.buffers.splitk_workspace(),
            stream,
        )?;
        if v_is_turbo && wht_runtime_active && self.wht_bf16_k_inv.0 != 0 {
            use spark_runtime::kernel_args::KernelLaunch;
            KernelLaunch::new(fwd.gpu, self.wht_bf16_k_inv)
                .grid([n as u32 * nq, 1, 1])
                .block([32, 1, 1])
                .arg_ptr(attn_out)
                .arg_u32(hd)
                .launch(stream)?;
        }
        Ok(attn_out)
    }

    /// Phase 6: gate multiply (when gated) + O projection. Writes to
    /// `o_out`. Returns the o_out buffer pointer.
    pub(super) fn ms_phase_o_proj(
        &self,
        c: &MultiSeqCtx<'_>,
        attn_out: DevicePtr,
    ) -> Result<DevicePtr> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            h,
            nq,
            hd,
            bf16,
            q_dim,
            per_seq_qkv,
            qkv_buf,
            normed,
            ..
        } = *c;
        if self.gated {
            for i in 0..n {
                let gate_i = qkv_buf.offset(i * per_seq_qkv + q_dim as usize * bf16);
                let attn_out_i = attn_out.offset(i * q_dim as usize * bf16);
                ops::sigmoid_gate_mul(
                    fwd.gpu,
                    self.sigmoid_gate_mul_k,
                    attn_out_i,
                    gate_i,
                    attn_out_i,
                    q_dim,
                    stream,
                )?;
            }
        }

        if let Some(ref g_proj) = self.head_gate_weight {
            let gate_buf = qkv_buf;
            ops::dense_gemm_tc(
                fwd.gpu,
                self.dense_gemm_tc_k,
                normed,
                g_proj,
                gate_buf,
                n as u32,
                nq,
                h as u32,
                stream,
            )?;
            match self.head_gate_activation {
                HeadGateActivation::Sigmoid => ops::sigmoid_gate_mul_head_broadcast(
                    fwd.gpu,
                    self.sigmoid_gate_head_broadcast_k,
                    attn_out,
                    gate_buf,
                    attn_out,
                    nq,
                    hd,
                    n as u32,
                    stream,
                )?,
                HeadGateActivation::Softplus => ops::softplus_gate_mul_head_broadcast(
                    fwd.gpu,
                    self.softplus_gate_head_broadcast_k,
                    attn_out,
                    gate_buf,
                    attn_out,
                    nq,
                    hd,
                    n as u32,
                    stream,
                )?,
            }
        }

        let o_out = fwd.buffers.moe_output();
        if let Some(o_mirror) = self.o_fp8_mirror.as_ref()
            && self.fp8_gemm_row_scaled_k.0 != 0
        {
            // FP8 MIRROR batched O projection (ATLAS_TARGET_ATTN_FP8_MIRROR):
            // one row-scaled FP8 GEMM reads the o_proj mirror ONCE for all n
            // rows at half the BF16 weight bytes. Checked BEFORE the
            // o_dense_bf16 branch below. attn_out is contiguous [n, q_dim],
            // o_out contiguous [n, h] — same layout the BF16 GEMM uses. Both
            // the flat multiseq verify and the batched tree verify land here
            // (tree.rs reuses this phase fn).
            self.fp8_mirror_gemm(
                fwd.gpu,
                attn_out,
                o_mirror,
                o_out,
                n as u32,
                h as u32,
                nq * hd,
                stream,
            )?;
        } else if let Some(o_bf16) = self.o_dense_bf16.as_ref() {
            // ATLAS_FP8_DEQUANT_ATTN_TO_BF16: O-proj dequanted to BF16 at load.
            // attn_out is contiguous [n, q_dim] and o_out is [n, h], so a single
            // batched GEMM reads the BF16 o_proj weight ONCE for all n sequences
            // instead of once per sequence (per-seq dense_gemv re-read it N×).
            // Same-precision fast ladder (VERIFY_SPLIT measured the naive
            // dense_gemm at 44ms/step for the K=8 verify — 6.5× off the
            // bandwidth floor): cuBLASLt BF16 → pipelined → naive.
            if ops::cublas_gemm_enabled() && n > 1 {
                ops::cublas_bf16_proj_dense(
                    attn_out,
                    o_bf16.weight,
                    o_out,
                    n as u32,
                    h as u32,
                    nq * hd,
                    stream,
                )?;
            } else if self.dense_gemm_pipelined_k.0 != 0 {
                ops::dense_gemm_bf16_pipelined(
                    fwd.gpu,
                    self.dense_gemm_pipelined_k,
                    attn_out,
                    o_bf16,
                    o_out,
                    n as u32,
                    h as u32,
                    nq * hd,
                    stream,
                )?;
            } else {
                ops::dense_gemm(
                    fwd.gpu,
                    self.dense_gemm_k,
                    attn_out,
                    o_bf16,
                    o_out,
                    n as u32,
                    h as u32,
                    nq * hd,
                    stream,
                )?;
            }
        } else if let Some(o_fp8) = self.o_weight.as_ref().and_then(|w| w.as_fp8()) {
            // FP8 native: per-token w8a16_gemv for O projection.
            for i in 0..n {
                let attn_out_i = attn_out.offset(i * q_dim as usize * bf16);
                let o_out_i = o_out.offset(i * h * bf16);
                ops::w8a16_gemv(
                    fwd.gpu,
                    self.w8a16_gemv_k,
                    attn_out_i,
                    o_fp8.weight,
                    o_fp8.row_scale,
                    o_out_i,
                    h as u32,
                    nq * hd,
                    stream,
                )?;
            }
        } else if n == 3 && !self.attn.o_proj.is_null() {
            ops::w4a16_gemv_batch3(
                fwd.gpu,
                self.w4a16_gemv_batch3_k,
                attn_out,
                &self.attn.o_proj,
                o_out,
                h as u32,
                nq * hd,
                stream,
            )?;
        } else if n == 2 && !self.attn.o_proj.is_null() {
            ops::w4a16_gemv_batch2(
                fwd.gpu,
                self.w4a16_gemv_batch2_k,
                attn_out,
                &self.attn.o_proj,
                o_out,
                h as u32,
                nq * hd,
                stream,
            )?;
        } else if !self.attn.o_proj.is_null() {
            // WIDE-VERIFY BATCHED O-PROJ (DFlash γ=16, n>3). One GEMM reads
            // the o_proj weight ONCE for all n rows instead of the per-row
            // GEMV loop below. attn_out is contiguous [n, q_dim]; o_out is
            // contiguous [n, h]; both already laid out for a single M=n GEMM
            // (no scatter). Uses the pipelined m128_v2 kernel when the
            // transposed weight is present (base M64 GEMM is the slow path).
            self.wide_verify_gemm(
                c,
                attn_out,
                &self.attn.o_proj,
                self.o_nvfp4_t.as_ref(),
                o_out,
                n as u32,
                h as u32,
                nq * hd,
            )?;
        } else {
            for i in 0..n {
                let attn_out_i = attn_out.offset(i * q_dim as usize * bf16);
                let o_out_i = o_out.offset(i * h * bf16);
                ops::w4a16_gemv(
                    fwd.gpu,
                    self.w4a16_gemv_k,
                    attn_out_i,
                    &self.attn.o_proj,
                    o_out_i,
                    h as u32,
                    nq * hd,
                    stream,
                )?;
            }
        }

        // ── Per-request O LoRA delta (batched bgmv). x = attn_out (post-gate,
        // contiguous [n, q_dim]); base_out = o_out (contiguous [n, h]) folded in
        // place — matches the single-seq apply_lora_delta on o after o_proj.
        // No-op unless a routing table is installed AND seq_slot is non-null.
        if let Some(ref lw) = self.lora
            && c.seq_slot.0 != 0
            && let Some(ref route) = lw.o_route
        {
            ops::lora_delta::apply_lora_bgmv(
                fwd.gpu,
                &lw.kernels,
                route,
                attn_out,
                o_out,
                c.seq_slot,
                n as u32,
                q_dim,    // x row stride (elements): attn_out is [n, q_dim]
                h as u32, // out row stride (elements): o_out is [n, h] contiguous
                fwd.buffers.lora_xa(),
                stream,
            )?;
        }
        Ok(o_out)
    }
}
