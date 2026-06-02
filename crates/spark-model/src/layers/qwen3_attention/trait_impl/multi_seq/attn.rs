// SPDX-License-Identifier: AGPL-3.0-only

//! Phases 3-6: per-sequence RoPE, KV-cache write, batched paged
//! attention, gate multiply + O projection.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::{KvCacheDtype, PagedKvCache};

use super::ctx::MultiSeqCtx;
use crate::layer::AttnMetadataDev;
use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;

/// Cached `ATLAS_ATTN_QKV_MEGA` env-var lookup. When `1`/`true` the
/// multi-seq RoPE and KV-cache-write phases each collapse from N
/// sequential launches into a single batched launch. Default off for
/// A/B safety. Bit-identical to the sequential path when the
/// strided RoPE kernel is present.
fn attn_qkv_mega_enabled() -> bool {
    static CACHE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHE.get_or_init(|| {
        std::env::var("ATLAS_ATTN_QKV_MEGA")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false)
    })
}

impl Qwen3AttentionLayer {
    /// Phase 3: per-token RoPE (each sequence has its own position).
    ///
    /// When `ATLAS_ATTN_QKV_MEGA=1` and the strided kernel is present,
    /// all N tokens are RoPE'd in a single kernel launch via
    /// `rope_forward_strided_b3`. Otherwise the original per-token loop
    /// runs (1 launch per token).
    pub(super) fn ms_phase_rope(&self, c: &MultiSeqCtx<'_>, meta: AttnMetadataDev) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            nq,
            nkv,
            hd,
            bf16,
            q_proj_bytes,
            per_seq_qkv,
            qkv_buf,
            ..
        } = *c;

        // Mega path: batched RoPE in one launch.
        if attn_qkv_mega_enabled() && self.rope_strided_b3_k.0 != 0 {
            debug_assert!(per_seq_qkv % bf16 == 0, "qkv stride must be BF16-aligned");
            debug_assert!(q_proj_bytes % bf16 == 0, "q_proj_bytes must be BF16-aligned");
            let qkv_stride_bf16 = (per_seq_qkv / bf16) as u32;
            let k_offset_bf16 = (q_proj_bytes / bf16) as u32;
            return ops::rope_strided_b3(
                fwd.gpu,
                self.rope_strided_b3_k,
                qkv_buf,
                meta.positions,
                n as u32,
                qkv_stride_bf16,
                k_offset_bf16,
                nq,
                nkv,
                hd,
                self.rotary_dim_override
                    .unwrap_or(fwd.config.rotary_dim() as u32),
                self.rope_theta_override
                    .unwrap_or(fwd.config.rope_theta as f32),
                stream,
            );
        }

        // Fallback: per-token loop (N launches).
        for i in 0..n {
            let q_out_i = qkv_buf.offset(i * per_seq_qkv);
            let k_out_i = q_out_i.offset(q_proj_bytes);
            let pos_i = meta.positions.offset(i * 4); // u32 per position
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
        }
        Ok(())
    }

    /// Phase 4: per-token KV cache write.
    ///
    /// When `ATLAS_ATTN_QKV_MEGA=1` all N tokens write in a single
    /// kernel launch. The underlying `reshape_and_cache_flash*` kernels
    /// already accept a `key_stride`/`value_stride` argument and walk
    /// the token axis via `blockIdx.x`, so the only change is passing
    /// the per-token BF16 stride instead of the (contiguous) per-token
    /// element count — no kernel changes required.
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

        // Mega path: single batched launch covering all N tokens.
        // Skipped for Turbo3/4/8 because the WHT bookend in
        // `write_kv_cache` walks K/V as a contiguous `[num_tokens,
        // num_kv_heads, head_dim]` tensor — feeding it a strided
        // (per_seq_qkv) view would WHT the interleaved Q/gate slabs and
        // corrupt the cache.
        let turbo = matches!(
            self.kv_dtype,
            KvCacheDtype::Turbo3 | KvCacheDtype::Turbo4 | KvCacheDtype::Turbo8
        );
        if attn_qkv_mega_enabled() && !turbo {
            debug_assert!(per_seq_qkv % bf16 == 0, "qkv stride must be BF16-aligned");
            let per_token_stride_bf16 = (per_seq_qkv / bf16) as u32;
            let k_base = qkv_buf.offset(q_proj_bytes);
            let v_base = k_base.offset((nkv * hd) as usize * bf16);
            return self.write_kv_cache(
                fwd.gpu,
                k_base,
                v_base,
                kv_cache,
                meta.slot,
                n as u32,
                nkv,
                hd,
                bs,
                per_token_stride_bf16,
                per_token_stride_bf16,
                stream,
                fwd.graph_capture,
            );
        }

        // Fallback: N sequential launches.
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
            max_seq_len_host,
            ..
        } = *c;
        // Build contiguous Q buffer [N, nq*hd] for batched attention.
        let q_contiguous = fwd.buffers.ssm_qkvz();
        for i in 0..n {
            let q_out_i = qkv_buf.offset(i * per_seq_qkv);
            fwd.gpu.copy_d2d_async(
                q_out_i,
                q_contiguous.offset(i * q_dim as usize * bf16),
                q_dim as usize * bf16,
                stream,
            )?;
        }
        let attn_out = fwd.buffers.attn_output();
        let inv_sqrt_d = self.effective_attn_scale(hd);
        // ATLAS_TREE_AWARE_ATTN: pull optional KV indirection from the
        // forward context (populated by `verify_d.rs` when tree mode is
        // active). NULL keeps the kernel in legacy chain-mode.
        //
        // CUDA graph fix: `kv_indir_base_ptr` is a 1×i32 device buffer
        // (not the raw scalar) so a captured graph sees the fresh
        // `seq.seq_len` on each replay. Host writes the value into the
        // buffer before each step in `verify_d.rs`.
        let (kv_indir, kv_indir_base_ptr, kv_indir_stride) = match fwd.tree_aware_attn {
            Some(t) => (t.kv_indir, t.kv_indir_base_ptr, t.kv_indir_stride),
            None => (
                spark_runtime::gpu::DevicePtr::NULL,
                spark_runtime::gpu::DevicePtr::NULL,
                0u32,
            ),
        };

        // ATLAS_TREE_KV_PACK two-pool fast path:
        // 1. Scatter ancestor KV into per-layer scratch (one CTA per
        //    (seq, ancestor slot), each copies one token's K+V).
        // 2. Call the existing paged_decode_attn_fp8 kernel with BOTH the
        //    real cache (for prior-linear context, fast BC=4) AND the
        //    pack pool (for the tree window, fast BC=4). The kernel uses
        //    `pos < kv_indir_base` → real cache, `pos >= kv_indir_base`
        //    → pack pool. Indirection table is still passed but the
        //    pack-pool branch takes precedence inside the kernel.
        //
        // This keeps linear context attention identical to chain mode
        // (so outputs stay coherent) while restoring BC=4 batching for
        // the tree window. FP8 only at present (matches our aeon-27b
        // config); NVFP4 path falls through to the slow indirected fallback.
        if let Some(t) = fwd.tree_aware_attn
            && let Some(pack) = t.pack
            && (self.attn_layer_idx as u32) < pack.num_attn_layers
            && matches!(self.kv_dtype, KvCacheDtype::Fp8)
        {
            self.run_packed_kv_attn_fp8(
                fwd.gpu,
                q_contiguous,
                attn_out,
                n as u32,
                nq,
                nkv,
                hd,
                bs,
                inv_sqrt_d,
                kv_cache,
                meta,
                t,
                pack,
                stream,
            )?;
            return Ok(attn_out);
        }

        // FlashAttention-v2 inspired K=γ-fused path. Active when:
        //   - `ATLAS_FLASH_ATTN_KGAMMA=1`
        //   - num_seqs ≥ 8 (K=γ verify shape; chain/draft stay on legacy)
        //   - num_seqs ≤ 32 (kernel QTILE_MAX bound)
        //   - NVFP4 KV cache + HDIM=256 + kgamma kernel resolved
        //   - No tree-aware indirection (legacy kernel handles tree mode)
        //
        // Collapses 17×4=68 CTAs → 4 CTAs (one per q_head), loads each
        // KV vector once per warp and reuses across 2-3 owned queries.
        if crate::layers::flash_attn_kgamma_enabled()
            && (8..=32).contains(&n)
            && matches!(self.kv_dtype, KvCacheDtype::Nvfp4)
            && hd == 256
            && kv_indir == spark_runtime::gpu::DevicePtr::NULL
            && let Some(kgamma_k) = self.paged_decode_kgamma_k
        {
            // Split-K fork (task #96): when seq_len is long enough that
            // a single-CTA per q_head leaves SMs idle, partition the KV
            // history across `num_splits` CTAs per q_head and follow with
            // a reduce kernel. With num_q_heads=4 and num_splits=12 the
            // grid becomes 48 CTAs — one per SM on GB10 sm_120.
            //
            // Gates:
            //   - ATLAS_FLASH_ATTN_KGAMMA_SPLITK=1 (off by default)
            //   - max_seq_len_host >= 1024 (short context: single-CTA wins)
            //   - both splitk + reduce kernels resolved
            //
            // num_splits formula targets ~NUM_SMS / num_q_heads CTAs (12 for
            // aeon-27b 4-head), capped so each split processes at least
            // SPLIT_TILE positions to keep per-CTA work meaningful.
            const SPLIT_TILE: u32 = 512;
            const MAX_SPLITS_CAP: u32 = 64;
            use atlas_core::device::sm121::NUM_SMS;
            if crate::layers::flash_attn_kgamma_splitk_enabled()
                && max_seq_len_host >= 1024
                && let Some(splitk_k) = self.paged_decode_kgamma_splitk_k
                && let Some(reduce_k) = self.paged_decode_kgamma_reduce_k
            {
                // Target one CTA per SM. Also bound by ceil(seq_len/SPLIT_TILE)
                // so each split gets at least ~SPLIT_TILE positions.
                let occupancy_target = (NUM_SMS / nq).max(2);
                let seq_target = max_seq_len_host.div_ceil(SPLIT_TILE);
                let num_splits = occupancy_target
                    .min(seq_target.max(1))
                    .min(MAX_SPLITS_CAP);
                if num_splits >= 2 {
                    let workspace = fwd.buffers.splitk_workspace();
                    ops::paged_decode_attn_kgamma_nvfp4_splitk(
                        fwd.gpu,
                        splitk_k,
                        q_contiguous,
                        kv_cache.k_pool_ptr(self.attn_layer_idx),
                        kv_cache.v_pool_ptr(self.attn_layer_idx),
                        workspace,
                        meta.block_table,
                        meta.seq_len,
                        meta.max_blocks_per_seq,
                        nq,
                        nkv,
                        hd,
                        bs,
                        inv_sqrt_d,
                        num_splits,
                        nq * hd,
                        kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                        kv_cache.nvfp4_data_bytes() as u64,
                        n as u32,
                        stream,
                    )?;
                    ops::paged_decode_attn_kgamma_reduce_nvfp4(
                        fwd.gpu,
                        reduce_k,
                        workspace,
                        attn_out,
                        nq,
                        num_splits,
                        n as u32,
                        stream,
                    )?;
                    return Ok(attn_out);
                }
            }

            // FA2-grafted fast path: cp.async double-buffered KV tile
            // staging in shared memory. Same caller contract as the
            // baseline single-CTA kgamma kernel. Gated by
            // ATLAS_FA2_KGAMMA=1 (off by default). Takes precedence over
            // VEC when both are enabled — they share the same kernel
            // shape but FA2 stages KV through SMEM with a 2-stage
            // pipeline whereas VEC dequants direct-from-global in pairs.
            if crate::layers::flash_attn_kgamma_fa2_enabled()
                && let Some(fa2_k) = self.paged_decode_kgamma_fa2_k
            {
                // One-time INFO log to confirm the FA2 path is active.
                static FA2_FIRST_DISPATCH: std::sync::Once = std::sync::Once::new();
                FA2_FIRST_DISPATCH.call_once(|| {
                    tracing::info!(
                        "ATLAS_FA2_KGAMMA: dispatching paged_decode_attn_kgamma_nvfp4_fa2 (n={n})"
                    );
                });
                ops::paged_decode_attn_kgamma_nvfp4_fa2(
                    fwd.gpu,
                    fa2_k,
                    q_contiguous,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    attn_out,
                    meta.block_table,
                    meta.seq_len,
                    meta.max_blocks_per_seq,
                    nq,
                    nkv,
                    hd,
                    bs,
                    inv_sqrt_d,
                    nq * hd,
                    kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                    kv_cache.nvfp4_data_bytes() as u64,
                    n as u32,
                    stream,
                )?;
                return Ok(attn_out);
            }

            // VEC fast path: 16 warps/CTA + 2-position dequant batching.
            // Same caller contract as the baseline single-CTA kgamma kernel.
            // Gated by ATLAS_KGAMMA_VECDEQUANT=1 (off by default).
            if crate::layers::flash_attn_kgamma_vecdequant_enabled()
                && let Some(vec_k) = self.paged_decode_kgamma_vec_k
            {
                ops::paged_decode_attn_kgamma_nvfp4_vec(
                    fwd.gpu,
                    vec_k,
                    q_contiguous,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    attn_out,
                    meta.block_table,
                    meta.seq_len,
                    meta.max_blocks_per_seq,
                    nq,
                    nkv,
                    hd,
                    bs,
                    inv_sqrt_d,
                    nq * hd,
                    kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                    kv_cache.nvfp4_data_bytes() as u64,
                    n as u32,
                    stream,
                )?;
                return Ok(attn_out);
            }

            ops::paged_decode_attn_kgamma_nvfp4(
                fwd.gpu,
                kgamma_k,
                q_contiguous,
                kv_cache.k_pool_ptr(self.attn_layer_idx),
                kv_cache.v_pool_ptr(self.attn_layer_idx),
                attn_out,
                meta.block_table,
                meta.seq_len,
                meta.max_blocks_per_seq,
                nq,
                nkv,
                hd,
                bs,
                inv_sqrt_d,
                nq * hd,
                kv_cache.block_stride_bytes_for_layer(self.attn_layer_idx) as u64,
                kv_cache.nvfp4_data_bytes() as u64,
                n as u32,
                stream,
            )?;
            return Ok(attn_out);
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
            kv_indir,
            kv_indir_base_ptr,
            kv_indir_stride,
            max_seq_len_host,
            stream,
        )?;
        Ok(attn_out)
    }

    /// ATLAS_TREE_KV_PACK two-pool dispatcher (FP8). Scatters tree-window
    /// KV into per-layer scratch, then runs the existing FP8 attn kernel
    /// with the real cache as `K_cache/V_cache` (linear-context, fast
    /// BC=4) and the scratch as `K_pack_pool/V_pack_pool` (tree-window,
    /// fast BC=4 over contiguous slots). The kernel routes positions
    /// `< kv_indir_base` to the real cache and `>= kv_indir_base` to the
    /// pack pool — restores BC=4 batching for the entire query range.
    #[allow(clippy::too_many_arguments)]
    fn run_packed_kv_attn_fp8(
        &self,
        gpu: &dyn spark_runtime::gpu::GpuBackend,
        q_contiguous: DevicePtr,
        attn_out: DevicePtr,
        n: u32,
        nq: u32,
        nkv: u32,
        hd: u32,
        bs: u32,
        inv_sqrt_d: f32,
        kv_cache: &PagedKvCache,
        meta: crate::layer::AttnMetadataDev,
        tree: crate::layer::TreeAwareAttn,
        pack: crate::layer::TreeKvPack,
        stream: u64,
    ) -> Result<()> {
        use spark_runtime::kernel_args::KernelLaunch;

        let src_k_pool = kv_cache.k_pool_ptr(self.attn_layer_idx);
        let src_v_pool = kv_cache.v_pool_ptr(self.attn_layer_idx);
        // SAFETY: `scratch_k_ptrs` / `scratch_v_ptrs` reference a `Vec<DevicePtr>`
        // owned by `TransformerModel` (allocated at init, never resized).
        // Bounds-checked by the caller (`attn_layer_idx < num_attn_layers`).
        let dst_k_pool = unsafe { *pack.scratch_k_ptrs.add(self.attn_layer_idx) };
        let dst_v_pool = unsafe { *pack.scratch_v_ptrs.add(self.attn_layer_idx) };

        // Scatter: grid (n, max_chain_len). Each CTA copies one ancestor's
        // K+V from the real paged cache → scratch.
        let scatter_k = pack.scatter_fp8_kernel;
        if scatter_k.0 == 0 {
            anyhow::bail!("tree_kv_pack: scatter_fp8 kernel not loaded");
        }
        let max_chain_len = pack.block_size;
        KernelLaunch::new(gpu, scatter_k)
            .grid([n, max_chain_len, 1])
            .block([256, 1, 1])
            .arg_ptr(src_k_pool)
            .arg_ptr(src_v_pool)
            .arg_ptr(meta.block_table)
            .arg_ptr(tree.kv_indir)
            .arg_ptr(dst_k_pool)
            .arg_ptr(dst_v_pool)
            .arg_u32(n)
            .arg_u32(max_chain_len)
            .arg_u32(nkv)
            .arg_u32(hd)
            .arg_u32(pack.cache_block_size)
            .arg_u32(pack.cache_max_blocks_per_seq)
            .arg_u32(tree.kv_indir_stride)
            // CUDA graph fix: `abs_base` is now a device-buffer pointer so a
            // captured graph reads the fresh `seq.seq_len` on each replay.
            .arg_ptr(pack.abs_base_ptr)
            .launch(stream)?;

        // Now run the standard FP8 attn — same metadata as chain mode, but
        // hand the scratch in as the pack-pool. Kernel will:
        //   pos in [0..kv_indir_base)        → real cache, BC=4 batched
        //   pos in [kv_indir_base..seq_len)  → scratch, BC=4 batched
        // and feed both into the same online-softmax accumulator.
        use crate::layers::ops;
        let (k_scale, v_scale) = self.effective_fp8_scales();
        ops::paged_decode_attn_fp8(
            gpu,
            self.paged_decode_k,
            q_contiguous,
            src_k_pool,
            src_v_pool,
            attn_out,
            meta.block_table,
            meta.seq_len,
            meta.max_blocks_per_seq,
            n,
            nq,
            nkv,
            hd,
            bs,
            inv_sqrt_d,
            k_scale,
            v_scale,
            nq * hd,
            kv_cache.cache_stride() as u64,
            // Indirection still passed (legacy contract); the kernel's
            // pack-pool branch takes precedence when pack_pool != null.
            // CUDA graph fix: kv_indir_base is now a device-buffer ptr.
            tree.kv_indir,
            tree.kv_indir_base_ptr,
            tree.kv_indir_stride,
            // Pack pool: routes tree-window reads through the scratch.
            dst_k_pool,
            dst_v_pool,
            max_chain_len,
            stream,
        )?;
        Ok(())
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

        let o_out = fwd.buffers.moe_output();
        if let Some(o_fp8) = self.o_weight.as_ref().and_then(|w| w.as_fp8()) {
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
        } else if n > 3
            && n <= 32
            && self.w4a16_gemm_t_m16_k.0 != 0
            && crate::layers::tc_nvfp4_m16_enabled()
            && self.o_nvfp4_t.is_some()
            && std::env::var("ATLAS_TC_NVFP4_M16_MS_ATTN").ok().as_deref() == Some("1")
        {
            // K=γ verify path (DFlash γ>3): batch O-projection as a single
            // M=n GEMM using the small-M specialization, replacing n per-token
            // GEMV launches. attn_out is already contiguous [n, nq*hd] and
            // o_out is [n, h] — no scatter needed.
            //
            // Gated by the same ATLAS_TC_NVFP4_M16_MS_ATTN flag as
            // `ms_qkv_batched_m16` because the multi_seq attention m16
            // dispatch currently produces numerically different output
            // vs the per-token GEMV fallback (see qkv.rs comment for the
            // root-cause investigation status).
            // unwrap safe: gated by `self.o_nvfp4_t.is_some()` above.
            #[allow(clippy::unnecessary_unwrap)]
            let nvfp4_t = self.o_nvfp4_t.as_ref().unwrap();
            ops::w4a16_gemm_n128_m16(
                fwd.gpu,
                self.w4a16_gemm_t_m16_k,
                attn_out,
                nvfp4_t,
                o_out,
                n as u32,
                h as u32,
                nq * hd,
                stream,
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
        Ok(o_out)
    }
}
