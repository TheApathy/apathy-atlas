// SPDX-License-Identifier: AGPL-3.0-only

//! ATLAS_FUSED_ELEMWISE=1 — SERIAL (M=1) decode fused q/k epilogue.
//!
//! The single-token decode chain in `attention_forward.rs` launches, per
//! eligible layer:
//!   1. `rms_norm[_vanilla]` on Q  (per-head, grid = nq)
//!   2. `rms_norm[_vanilla]` on K  (per-head, grid = nkv)
//!   3. `rope_forward_yarn_scaled` (Q + K, one launch)
//!   4. `reshape_and_cache_flash`  (BF16 paged K/V write)
//! This module collapses those 4 launches into ONE
//! `fused_qkv_norm_rope_cache_write_bf16` launch (the same kernel the
//! multi-seq flat-verify epilogue landed — n = 1 row here; see
//! kernels/gb10/common/fused_verify_elemwise.cu for the bit-exactness
//! contract: identical FP32 expression order, BF16 rounding at every point
//! the unfused chain went through memory, `--fmad=false` inherited).
//!
//! Layout note (why this wiring is trivial at M=1): the serial GEMVs write
//! `qkv_output` directly — Q at offset 0 (`[1, nq*hd]`), K at
//! `q_proj_bytes`, V after K, each contiguous. With n = 1 those are exactly
//! the `[n, nq*hd]` / `[n, nkv*hd]` operands the fused kernel consumes, so
//! no scatter/gather is involved (unlike the multi-seq path, which also
//! deleted 4n copies). Q is normed+roped in place in `qkv_output` — the
//! very buffer the paged decode reads. K/V land in the paged BF16 cache;
//! nothing downstream reads `k_out`/`v_out` after the cache write.
//!
//! Graph-capture safe: pure device pointers + scalars, no host reads, and
//! the eligibility decision below is load-fixed (weights/kernels/dtypes),
//! so it is stable across a CUDA-graph capture and its replays.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::{KvCacheDtype, PagedKvCache};

use super::super::Qwen3AttentionLayer;
use crate::layer::{AttnMetadataDev, ForwardContext};
use crate::layers::ops;

impl Qwen3AttentionLayer {
    /// Serial (M=1) fused-epilogue eligibility. Mirrors
    /// `ms_fused_epilogue_eligible` (multi_seq/attn.rs) minus the n>3 /
    /// wide-QKV-branch terms, which don't exist at M=1 (every serial GEMV
    /// branch writes the contiguous `qkv_output` layout): ungated, no LoRA,
    /// no MLA/HC, per-head q/k norms present (no MiniMax full-width norm,
    /// no Gemma-4 v_norm), table-based yarn-scaled RoPE, BF16 paged KV
    /// cache, head_dim even ≤ 256, rotary_dim even in [2, head_dim].
    pub(super) fn serial_fused_epilogue_eligible(&self, ctx: &ForwardContext, hd: u32) -> bool {
        if !ops::fused_elemwise_enabled() || self.fused_qkv_norm_rope_cache_k.0 == 0 {
            return false;
        }
        if self.gated || self.lora.is_some() || self.mla.is_some() || self.hc.is_some() {
            return false;
        }
        if self.kv_dtype != KvCacheDtype::Bf16 {
            return false; // fused kernel writes a BF16 paged cache only
        }
        if self.yarn_inv_freq.is_null() {
            return false; // plain-theta / proportional / mrope not in the fused kernel
        }
        if self.attn.q_norm_full.is_some() || self.attn.k_norm_full.is_some() {
            return false; // MiniMax full-projected-hidden norm path
        }
        if self.attn.q_norm.weight.is_null() || self.attn.k_norm.weight.is_null() {
            return false; // Nemotron-H skip-norm path
        }
        if self.v_norm_weight.is_some() {
            return false; // Gemma-4 v_norm runs between the norms and RoPE
        }
        if hd == 0 || !hd.is_multiple_of(2) || hd > 256 {
            return false;
        }
        let rot = self
            .rotary_dim_override
            .unwrap_or(ctx.config.rotary_dim() as u32);
        rot >= 2 && rot.is_multiple_of(2) && rot <= hd
    }

    /// The ONE-launch serial epilogue: per-head q/k rms_norm → BF16 round →
    /// yarn-scaled RoPE → paged BF16 cache write (K) + verbatim V write.
    /// Bit-identical to the 4-launch chain it replaces (`rms_norm[_vanilla]`
    /// ×2 → `rope_forward_yarn_scaled` → `reshape_and_cache_flash`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn serial_fused_qk_epilogue(
        &self,
        q_out: DevicePtr,
        k_out: DevicePtr,
        v_out: DevicePtr,
        kv_cache: &PagedKvCache,
        meta: AttnMetadataDev,
        nq: u32,
        nkv: u32,
        hd: u32,
        bs: u32,
        eps: f32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        ops::fused_qkv_norm_rope_cache_write(
            ctx.gpu,
            self.fused_qkv_norm_rope_cache_k,
            q_out,
            k_out,
            v_out,
            &self.attn.q_norm,
            &self.attn.k_norm,
            meta.positions,
            self.yarn_inv_freq,
            kv_cache.k_pool_ptr(self.attn_layer_idx),
            kv_cache.v_pool_ptr(self.attn_layer_idx),
            meta.slot,
            1,
            nq,
            nkv,
            hd,
            self.rotary_dim_override
                .unwrap_or(ctx.config.rotary_dim() as u32),
            bs,
            eps,
            self.yarn_attention_factor,
            u32::from(!self.norm_vanilla),
            stream,
        )
    }
}
