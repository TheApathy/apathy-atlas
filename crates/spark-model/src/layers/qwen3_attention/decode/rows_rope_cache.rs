// SPDX-License-Identifier: AGPL-3.0-only

//! Batched rope + paged-cache write for the γ-row DFlash verify.
//!
//! `attention_forward_v4` runs steps 3–5.5 of the V4-Flash decode chain for ONE
//! row. The γ-row verify in `trait_impl::multi_seq::mla` used to call it once
//! per row, so rope extract, `rope_yarn`, rope writeback, cache assemble and the
//! paged cache write each ran γ times as γ separate 1-row launches. Measured
//! cost was 467 µs/layer for phase B, of which only ~120 µs was the attention
//! kernel itself — the other ~347 µs was γ-fold launch and setup overhead on
//! kernels that were already capable of doing all γ rows at once.
//!
//! Every kernel in that chain already takes a `num_tokens` argument and is
//! exercised at N≫1 by prefill: `prefill/cache_skip_v4.rs:297-375` is the same
//! call sequence with `n` where the decode path hard-codes `1`. So this is a
//! dispatch change, not a new kernel. The verify rows are already contiguous
//! `[n, *]` buffers (`mla.rs` carves `q_batch`/`kv_batch`/`v_batch` out of one
//! scratch allocation), and the same scratch buffers are used by prefill at
//! n≤1024, so n=γ needs no extra capacity.
//!
//! **Causality.** The γ rows are consecutive positions of ONE sequence, so
//! writing all γ KV entries before any attention read looks like it would let
//! row `i` attend to row `i+1`'s key. It does not: `verify_d.rs:220` fills
//! `seq_lens[i] = seq.seq_len + i + 1`, and the paged decode kernel stops at
//! `seq_lens[seq_idx]`, so each row still sees only up to its own position.
//! This is the same masking chunked prefill relies on. If that seq_lens fill
//! ever changes to a flat `seq_len + k`, this batching becomes WRONG and the
//! per-row loop must come back.
//!
//! What is NOT batched here: the paged attention itself. `mla_paged_decode_fp8`
//! documents `Q`/`O` as `[1, nq * q_dim]` and indexes only `seq_lens` and
//! `block_tables` by `blockIdx.y`, never Q or O — so raising `num_seqs` would
//! make all rows read the same Q and race on the same output. Batching that
//! needs a kernel change (stride Q/O by `seq_idx`), tracked separately.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::Qwen3AttentionLayer;
use crate::layer::{AttnMetadataDev, ForwardContext};
use crate::layers::ops;

/// Buffers and geometry for one layer's γ-row rope + cache write.
///
/// `k_rope_tmp` is caller-provided scratch of at least `n * rope` BF16: the
/// single-row path reuses `q_latent` for this after `wq_b` has consumed it, and
/// the batched caller passes the equivalent `ql_batch` slot.
pub(in crate::layers::qwen3_attention) struct RowsRopeArgs {
    pub q_batch: DevicePtr,
    pub k_batch: DevicePtr,
    pub v_batch: DevicePtr,
    pub k_rope_tmp: DevicePtr,
    pub n: u32,
    pub nq: u32,
    pub hd: u32,
    pub bs: u32,
    pub stream: u64,
}

impl Qwen3AttentionLayer {
    /// Steps 3 → 4 for all `n` verify rows: rope Q and K, assemble the
    /// `[latent|rope]` cache entries, and write them to the paged cache.
    ///
    /// Mirrors `attention_forward_v4` steps 3/3.5/4 argument for argument; the
    /// only difference is `n` in place of the literal `1`.
    pub(in crate::layers::qwen3_attention) fn mla_rows_rope_and_cache(
        &self,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        meta: &AttnMetadataDev,
        args: &RowsRopeArgs,
    ) -> Result<()> {
        let RowsRopeArgs {
            q_batch,
            k_batch,
            v_batch,
            k_rope_tmp,
            n,
            nq,
            hd,
            bs,
            stream,
        } = *args;
        let mla = self
            .mla
            .as_ref()
            .expect("mla_rows_rope_and_cache called without MLA config");
        let mla_rope = mla.rope as u32;
        let nope = mla.nope as u32;

        // Sliding layers (compressor==None) = reference "main" rope: plain
        // θ=10000, mscale=1 (no yarn). CSA/HCA keep the θ=160000 yarn table.
        let inv_freq = if mla.compressor.is_none() {
            mla.main_inv_freq
        } else {
            mla.yarn_inv_freq
        };
        let mscale = if mla.compressor.is_none() {
            1.0f32
        } else {
            super::super::helpers::yarn_rope_mscale(ctx.config)
        };

        // ── Step 3: RoPE for Q and K ──
        let q_rope_tmp = ctx.buffers.ssm_conv_out_f32();
        ops::mla_q_rope_extract_batched(
            ctx.gpu,
            self.mla_q_rope_extract_batched_k,
            q_batch,
            q_rope_tmp,
            n,
            nq,
            hd,
            nope,
            mla_rope,
            nq * hd,
            stream,
        )?;
        // K's rope channels. `kv_dim == nkv * hd` and V4-Flash is MQA (nkv=1),
        // so the row stride is `hd` — the same value the single-row path passes.
        ops::mla_q_rope_extract_batched(
            ctx.gpu,
            self.mla_q_rope_extract_batched_k,
            k_batch,
            k_rope_tmp,
            n,
            1,
            hd,
            nope,
            mla_rope,
            hd,
            stream,
        )?;
        // DeepSeek-V4 uses INTERLEAVED RoPE (rope_interleave=True). `seq_len=n`
        // walks the n contiguous rows against meta.positions[0..n], exactly as
        // prefill does for its n tokens.
        ops::rope_yarn(
            ctx.gpu,
            self.rope_yarn_interleaved_k,
            q_rope_tmp,
            k_rope_tmp,
            meta.positions,
            n,
            nq,
            1,
            mla_rope,
            mla_rope,
            inv_freq,
            mscale,
            stream,
        )?;
        ops::mla_q_rope_writeback_batched(
            ctx.gpu,
            self.mla_q_rope_writeback_batched_k,
            q_rope_tmp,
            q_batch,
            n,
            nq,
            hd,
            nope,
            mla_rope,
            nq * hd,
            stream,
        )?;
        ops::mla_q_rope_writeback_batched(
            ctx.gpu,
            self.mla_q_rope_writeback_batched_k,
            k_rope_tmp,
            k_batch,
            n,
            1,
            hd,
            nope,
            mla_rope,
            hd,
            stream,
        )?;

        // ── Step 3.5: assemble the [latent|rope] cache entries ──
        let k_cache_assembled = ctx.buffers.ssm_deinterleaved();
        let v_cache_assembled = ctx.buffers.ssm_qkvz();
        let kv_lora = mla.kv_lora_rank as u32;
        let mla_cache_dim = kv_lora + mla_rope;
        ops::mla_cache_assemble_batched(
            ctx.gpu,
            self.mla_cache_assemble_batched_k,
            v_batch,     // latent K (pre-writeback copy)
            k_rope_tmp,  // rotated K rope
            k_cache_assembled,
            v_cache_assembled,
            n,
            kv_lora,
            mla_rope,
            mla_cache_dim,
            stream,
        )?;

        // ── Step 4: write all n assembled entries to the paged cache ──
        // meta.slot holds n slots; each row lands in its own slot, so the n
        // writes are independent and the ordering the per-row loop used to
        // provide is not needed (see the causality note at the top of the file).
        self.write_kv_cache(
            ctx.gpu,
            k_cache_assembled,
            v_cache_assembled,
            kv_cache,
            meta.slot,
            n,
            1,
            mla_cache_dim,
            bs,
            mla_cache_dim,
            mla_cache_dim,
            stream,
            ctx.graph_capture,
        )
    }

    /// Step 5.5 for all `n` rows: de-rotate the attention output by the query
    /// position (DeepSeek-V4 eq.26) so each value's contribution is
    /// relative-distance.
    ///
    /// `attn_batch` is the contiguous `[n, nq * hd]` block the per-row paged
    /// attention wrote into.
    pub(in crate::layers::qwen3_attention) fn mla_rows_derotate(
        &self,
        ctx: &ForwardContext,
        meta: &AttnMetadataDev,
        attn_batch: DevicePtr,
        n: u32,
        nq: u32,
        hd: u32,
        stream: u64,
    ) -> Result<()> {
        let mla = self
            .mla
            .as_ref()
            .expect("mla_rows_derotate called without MLA config");
        let mla_rope = mla.rope as u32;
        let nope = mla.nope as u32;

        // MUST match the Q/K rope inv_freq for this layer type (rope-in ==
        // de-rotate-out), else the output is scrambled.
        let inv_freq = if mla.compressor.is_none() {
            mla.main_inv_freq
        } else {
            mla.yarn_inv_freq
        };
        let mscale = if mla.compressor.is_none() {
            1.0f32
        } else {
            super::super::helpers::yarn_rope_mscale(ctx.config)
        };

        let o_rope_tmp = ctx.buffers.ssm_conv_out_f32();
        ops::mla_q_rope_extract_batched(
            ctx.gpu,
            self.mla_q_rope_extract_batched_k,
            attn_batch,
            o_rope_tmp,
            n,
            nq,
            hd,
            nope,
            mla_rope,
            nq * hd,
            stream,
        )?;
        ops::rope_yarn(
            ctx.gpu,
            self.rope_yarn_interleaved_inv_k,
            o_rope_tmp,
            o_rope_tmp,
            meta.positions,
            n,
            nq,
            0, // no KV heads — de-rotate the query/output heads only
            mla_rope,
            mla_rope,
            inv_freq,
            mscale,
            stream,
        )?;
        ops::mla_q_rope_writeback_batched(
            ctx.gpu,
            self.mla_q_rope_writeback_batched_k,
            o_rope_tmp,
            attn_batch,
            n,
            nq,
            hd,
            nope,
            mla_rope,
            nq * hd,
            stream,
        )
    }
}
