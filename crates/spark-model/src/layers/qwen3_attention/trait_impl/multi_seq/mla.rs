// SPDX-License-Identifier: AGPL-3.0-only

//! MLA (Multi-head Latent Attention) branch of multi-sequence batched
//! decode — the batched analogue of `decode::attention_forward_mla`.
//!
//! GitHub issue #84: the standard `ms_phase_qkv` path unconditionally
//! reads `attn.q_proj` / `q_weight`, which the Mistral MLA loader leaves
//! as a NULL `DevicePtr` stub (the real projections live in `self.mla`).
//! Routing an MLA model through the non-MLA `decode_multi_seq` body
//! launched `dense_gemv` against a NULL pointer → illegal address.
//! Commit 9e68dc2 stopped the crash with an `is_mla_dispatch()` per-seq
//! `decode()` fallback, but that fallback shares one `logits` buffer
//! across the loop and `decode()`'s `zero_all` wipes it — cross-seq
//! contamination. This module is the proper fix.
//!
//! ## Design
//!
//! The MLA decode chain (Q latent → norm → expand → absorbed-Q → Q_rope
//! → K latent → K_rope+RoPE → cache assemble+write → paged decode → V
//! extract → O proj) is run **once per sequence**, each iteration using
//!
//!   * a distinct per-sequence slice of the `normed` input and the
//!     `o_out` output buffer (stride `h` elements), and
//!   * per-sequence attention metadata — `positions[i]` (u32, +4 bytes),
//!     `slot[i]` (i64, +8 bytes), `seq_len[i]` (i32, +4 bytes) and
//!     `block_table` row `i` (`max_blocks_per_seq` i32 entries).
//!
//! Every sequence therefore reads and writes ONLY its own compressed
//! latent-KV history — no cross-contamination. The transient scratch
//! buffers (`ssm_ba`, `ssm_deinterleaved`, `expert_up_out`, …) are
//! reused across iterations: each iteration fully overwrites them before
//! reading, and all work is serialized on a single CUDA stream, so the
//! reuse is sound. Unlike the per-seq `decode()` fallback this stays in
//! ONE forward pass — no `Buffers::zero_all`, no host round-trip — so
//! the assembled `[n, h]` `o_out` is handed straight to `ms_phase_ffn`.
//!
//! The paged-decode attention kernel (`paged_decode_mla_k`) is itself
//! multi-seq capable (`grid[num_q_heads, num_seqs, 1]`); we still invoke
//! it per-sequence here so each sequence's absorbed-Q (built in shared
//! head-strided scratch) is consumed before the next iteration reuses
//! that scratch. N ≤ 8, the chain is GEMV-bound, so the per-seq launch
//! overhead is negligible.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use super::ctx::MultiSeqCtx;
use super::mla_gemv::MlaDims;
use crate::layer::AttnMetadataDev;
use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;

/// The batched-GEMV entry points that serve one verify width — FP8 packed and
/// strided, plus their NVFP4 mirrors. Selected once per call and threaded
/// through both projection phases so a width can never mix kernel families.
#[derive(Clone, Copy)]
struct BatchGemv {
    w8: KernelHandle,
    w8_ld: KernelHandle,
    w4: KernelHandle,
    w4_ld: KernelHandle,
}

impl Qwen3AttentionLayer {
    /// Narrowest compiled batched-GEMV pair covering `n` rows.
    ///
    /// The `batch4` pair is the common path and must keep serving n<=4: its
    /// accumulator array, unroll and cross-warp reduction smem are all half
    /// the `batch8` pair's. The `batch8` pair exists for the DSpark block
    /// verify (γ=6), where the alternative isn't a slightly wider kernel but
    /// the per-row fallback — six full re-reads of every attention projection.
    fn batch_gemv_for(&self, n: usize) -> Option<BatchGemv> {
        let candidates = [
            (
                4usize,
                self.w8a16_gemv_batch4_k,
                self.w8a16_gemv_batch4_ld_k,
                self.w4a16_gemv_batch4_k,
                self.w4a16_gemv_batch4_ld_k,
            ),
            (
                8,
                self.w8a16_gemv_batch8_k,
                self.w8a16_gemv_batch8_ld_k,
                self.w4a16_gemv_batch8_k,
                self.w4a16_gemv_batch8_ld_k,
            ),
        ];
        for (max_m, w8, w8_ld, w4, w4_ld) in candidates {
            // The FP8 pair is the floor — the NVFP4 mirrors are an
            // optimization the caller gates on separately.
            if n <= max_m && w8.0 != 0 && w8_ld.0 != 0 {
                return Some(BatchGemv { w8, w8_ld, w4, w4_ld });
            }
        }
        None
    }

    /// Whether the V4-Flash batched attention pipeline will fire for `c.n`
    /// rows, and how many scratch bytes it needs from `expert_up_out`.
    ///
    /// Single source of truth for `batch_ok`: `ms_mla_decode_v4_flash` asks it
    /// to pick its route, and `decode_multi_seq_tree_inner` asks it BEFORE
    /// launching anything so it can honour the trait's "`Ok(false)` means no
    /// work queued" contract.
    fn ms_mla_v4_batch_ok(&self, c: &MultiSeqCtx<'_>) -> (bool, usize) {
        let Some(mla) = self.mla.as_ref() else {
            return (false, 0);
        };
        let n = c.n;
        let q_dim = c.nq * c.hd;
        let kv_dim = c.nkv * c.hd;
        let q_lora = mla.q_lora_rank as u32;
        let o_groups = c.fwd.config.o_groups.max(1) as u32;
        let latent_dim = o_groups * mla.o_lora_rank as u32;
        let row = |elems: u32| n * elems as usize * c.bf16;
        let need = row(q_dim) * 2 + row(kv_dim) * 2 + row(q_lora) + row(latent_dim);
        // ATLAS_MLA_NO_BATCH=1: force the per-row fallback, for A/B testing
        // whether the batched-GEMV projections perturb verify argmax vs the
        // single-row decode path (greedy-losslessness check).
        let no_batch = {
            static NB: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *NB.get_or_init(|| std::env::var("ATLAS_MLA_NO_BATCH").as_deref() == Ok("1"))
        };
        let ok = !no_batch
            && n >= 2
            && self.batch_gemv_for(n).is_some()
            && mla.wq_a_fp8.is_some()
            && mla.wq_b_fp8.is_some()
            && mla.wkv_a_fp8.is_some()
            && mla.wo_a_fp8.is_some()
            && mla.wo_b_fp8.is_some()
            && c.fwd.buffers.expert_up_out_bytes() >= need;
        (ok, need)
    }

    /// Whether this layer can serve a DDTree tree-verify row batch.
    ///
    /// The split cache write lives in `mla_rows_rope_and_cache`, which only
    /// runs on the V4-Flash batched route with `ATLAS_MLA_ROWS_BATCH` left on.
    /// Every other route (legacy per-row `attention_forward_v4`, the
    /// `rows_batched=0` A/B leg) writes the cache through paths that cannot be
    /// range-split, so the caller must take the per-row sequential fallback.
    pub(super) fn ms_mla_v4_tree_capable(&self, c: &MultiSeqCtx<'_>) -> bool {
        self.mla.as_ref().is_some_and(|m| m.o_lora_rank > 0)
            && Self::ms_mla_rows_batched()
            && self.ms_mla_v4_batch_ok(c).0
    }

    /// `ATLAS_MLA_ROWS_BATCH` (default on): batched rope + cache write for all
    /// verify rows, vs the per-row `attention_forward_v4` chain.
    fn ms_mla_rows_batched() -> bool {
        static RB: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *RB.get_or_init(|| std::env::var("ATLAS_MLA_ROWS_BATCH").as_deref() != Ok("0"))
    }

    /// Batched MLA decode for `c.n` sequences. Writes each sequence's
    /// O-projection output into `moe_output[i*h .. (i+1)*h]` and returns
    /// the `moe_output` base pointer for `ms_phase_ffn`.
    ///
    /// `c.normed` already holds the RMS-normed hidden state for all `n`
    /// tokens (phase 1 ran before dispatch).
    pub(super) fn ms_mla_decode(
        &self,
        c: &MultiSeqCtx<'_>,
        kv_cache: &mut PagedKvCache,
        meta: AttnMetadataDev,
    ) -> Result<DevicePtr> {
        let mla = self
            .mla
            .as_ref()
            .expect("ms_mla_decode called without MLA config");

        let h = c.h as u32;
        let nq = c.nq;
        let hd = c.hd;
        let eps = c.eps;
        let bf16 = c.bf16;
        let stream = c.stream;
        let bs = c.bs as usize;

        let q_lora = mla.q_lora_rank as u32;
        let kv_lora = mla.kv_lora_rank as u32;
        let mla_nope = mla.nope as u32;
        let mla_v_dim = mla.v_dim as u32;
        let mla_rope = mla.rope as u32;
        let mla_cache_dim = kv_lora + mla_rope;
        let q_dim = nq * hd;
        let inv_sqrt_d = self.effective_attn_scale(hd);

        // 4b inc-3 γ-verify catch-up capture: snapshot this layer's compressor
        // input (`c.normed`, the layer-input RMSNorm output — the SAME tensor the
        // decode append feeds `wkv`/`wgate`) for all `n` verify rows BEFORE the
        // MLA chain overwrites the shared scratch. The post-accept
        // `dspark_compress_catchup` replays `v4_compress_append` over the
        // committed rows from here, advancing the compressed pool that this
        // batched (`pos:None`) path otherwise freezes — the decode/verify
        // asymmetry fix. Armed (non-NULL) only on compressor layers; the ring
        // holds MAX_VERIFY_ROWS=8, and multi-seq verify never exceeds that.
        if !self.verify_comp_normed.is_null() {
            let hb = c.h * bf16; // BF16 bytes per token row
            let rows = c.n.min(crate::layers::qwen3_attention::MAX_VERIFY_ROWS);
            c.fwd
                .gpu
                .copy_d2d_async(c.normed, self.verify_comp_normed, rows * hb, stream)?;

            // …and immediately run the compressor FORWARD over all `rows`
            // draft positions, before the attention below reads the pool.
            //
            // Without this the pool sits at `pre_len/ratio` for the whole
            // verify while row `r` needs `(pre_len+r+1)/ratio` — every row
            // attends a shorter compressed history than plain decode would, so
            // its logits diverge and prefix-accept truncates. The capture above
            // is what makes it possible: all `rows` compressor inputs are in
            // hand here, before the MLA chain overwrites the shared scratch.
            //
            // Speculative state is undone by `v4_compress_restore` on the
            // post-accept path; see `v4_compress_speculate` for the full
            // argument and the ds4 references.
            if let Some(base) = c.verify_base_pos {
                // The per-row fallback (batch_ok=false) interleaves the
                // appends itself (args.pos above) — the speculate must then
                // only SNAPSHOT the frontiers so the post-accept restore can
                // rewind; appending here too would double-append.
                let appends = self.ms_mla_v4_batch_ok(c).0;
                self.v4_compress_speculate(c.fwd, base, rows, eps, appends, stream)?;
            }
        }

        // O-projection output destination. `ms_phase_o_proj` (the non-MLA
        // sibling) returns `moe_output`; match it so `ms_phase_ffn`
        // consumes the same buffer for both paths.
        let o_out = c.fwd.buffers.moe_output();

        // DeepSeek-V4-Flash (o_lora_rank > 0) takes the dedicated multi-seq
        // path: a weight-amortized batched Q/KV/O pipeline around the per-row
        // rope/cache/attention middle (or the legacy per-row loop when the
        // batched preconditions are unmet).
        if mla.o_lora_rank > 0 {
            return self.ms_mla_decode_v4_flash(c, kv_cache, &meta, o_out);
        }

        for i in 0..c.n {
            let normed_i = c.normed.offset(i * c.h * bf16);
            // Per-sequence metadata views. The batched metadata packs
            // positions as `[n]` u32, slot as `[n]` i64, seq_len as `[n]`
            // i32 and block_table as `[n * max_blocks_per_seq]` i32 —
            // identical to the layout `ms_phase_rope` / `ms_phase_cache_write`
            // index for the non-MLA path.
            let meta_i = AttnMetadataDev {
                positions: meta.positions.offset(i * 4),
                positions_h: meta.positions_h.offset(i * 4),
                positions_w: meta.positions_w.offset(i * 4),
                slot: meta.slot.offset(i * 8),
                seq_len: meta.seq_len.offset(i * 4),
                block_table: meta
                    .block_table
                    .offset(i * meta.max_blocks_per_seq as usize * 4),
                max_blocks_per_seq: meta.max_blocks_per_seq,
                num_seqs: 1,
                seq_slot: spark_runtime::gpu::DevicePtr(0),
            };
            let o_out_i = o_out.offset(i * c.h * bf16);

            self.ms_mla_decode_one(
                c,
                kv_cache,
                &meta_i,
                normed_i,
                o_out_i,
                mla,
                MlaDims {
                    h,
                    nq,
                    hd,
                    q_dim,
                    q_lora,
                    kv_lora,
                    mla_nope,
                    mla_v_dim,
                    mla_rope,
                    mla_cache_dim,
                    eps,
                    bs,
                    inv_sqrt_d,
                    o_lora_rank: mla.o_lora_rank as u32,
                },
                stream,
            )?;
        }

        // ATLAS_MLA_HSD: per-seq diagnostic — scans each sequence's full
        // `o_out` row for NaN/Inf and reports magnitude, to localize
        // cross-sequence corruption in the batched MLA decode.
        if std::env::var("ATLAS_MLA_HSD").is_ok_and(|v| v == "1") && self.attn_layer_idx == 0 {
            c.fwd.gpu.synchronize(stream)?;
            for i in 0..c.n {
                let mut row = vec![0u8; c.h * bf16];
                let _ = c.fwd.gpu.copy_d2h(o_out.offset(i * c.h * bf16), &mut row);
                let vals: Vec<f32> = row
                    .chunks_exact(2)
                    .map(|x| f32::from_bits((u16::from_le_bytes([x[0], x[1]]) as u32) << 16))
                    .collect();
                let bad = vals.iter().filter(|v| !v.is_finite()).count();
                let absmax = vals.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                tracing::info!(
                    "MLA_HSD L0 s{i}: o_out non-finite={bad}/{} absmax={absmax:.4}",
                    vals.len(),
                );
            }
        }
        Ok(o_out)
    }

    /// DeepSeek-V4-Flash (o_lora_rank > 0) multi-seq decode.
    ///
    /// V4-Flash uses the DIRECT-KV attention algorithm, NOT the absorbed-MLA
    /// chain (its V3-style absorption weights are NULL stubs), so this drives
    /// the same `attention_forward_v4` chain as the n=1 path — but with the
    /// weight-heavy stages batched across the n rows:
    ///
    ///   Phase A (batched): wq_a → q_a_norm → wq_b → q_b_norm, wkv → kv_norm
    ///     → V copy — one pass over each FP8 weight serves all n rows
    ///     (`w8a16_gemv_batch4`).
    ///   Phase B (per row): rope → cache write → paged attention →
    ///     de-rotation via `attention_forward_v4` with `skip_qkv` +
    ///     `attn_dest` (the per-row parts are position/slot-dependent and
    ///     read no large weights).
    ///   Phase C (batched): block-diagonal wo_a (strided
    ///     `w8a16_gemv_batch4_ld` per group) → wo_b straight into the
    ///     per-row `o_out` slots.
    ///
    /// Without this, the K=2 MTP verify re-read all ~125 MB/layer of
    /// attention projections once PER ROW (~20 ms/step at n=2). Falls back
    /// to the legacy per-row loop when the FP8 projections / batch kernels
    /// are absent, n is out of batch range, or the borrowed MoE scratch
    /// (`expert_up_out`, idle during attention) is too small.
    fn ms_mla_decode_v4_flash(
        &self,
        c: &MultiSeqCtx<'_>,
        kv_cache: &mut PagedKvCache,
        meta: &AttnMetadataDev,
        o_out: DevicePtr,
    ) -> Result<DevicePtr> {
        let mla = self.mla.as_ref().unwrap();
        let h = c.h as u32;
        let nq = c.nq;
        let hd = c.hd;
        let eps = c.eps;
        let bf16 = c.bf16;
        let stream = c.stream;
        let bs = c.bs as usize;
        let n = c.n;
        let q_lora = mla.q_lora_rank as u32;
        let q_dim = nq * hd;
        let kv_dim = c.nkv * hd;
        let o_groups = c.fwd.config.o_groups.max(1) as u32;
        let group_in = q_dim / o_groups;
        let o_lora = mla.o_lora_rank as u32;
        let latent_dim = o_groups * o_lora;
        let gpu = c.fwd.gpu;

        // Borrowed scratch layout (bytes, in `expert_up_out` — MoE runs after
        // attention within the layer, so it is idle here; same precedent as
        // the compressor ring in `attention_forward_v4`).
        let row = |elems: u32| n * elems as usize * bf16;
        let gemv = self.batch_gemv_for(n);
        let (batch_ok, need) = self.ms_mla_v4_batch_ok(c);

        {
            static ROUTE_ONCE: std::sync::Once = std::sync::Once::new();
            ROUTE_ONCE.call_once(|| {
                tracing::info!(
                    "V4-msdecode route: {} (n={n} batch4={} batch8={} fp8 q_a/q_b/kv/o_a/o_b={}/{}/{}/{}/{} scratch={}B need={need}B)",
                    if batch_ok { "BATCHED" } else { "per-row fallback" },
                    self.w8a16_gemv_batch4_k.0 != 0,
                    self.w8a16_gemv_batch8_k.0 != 0,
                    mla.wq_a_fp8.is_some(),
                    mla.wq_b_fp8.is_some(),
                    mla.wkv_a_fp8.is_some(),
                    mla.wo_a_fp8.is_some(),
                    mla.wo_b_fp8.is_some(),
                    c.fwd.buffers.expert_up_out_bytes(),
                );
            });
        }
        // The DDTree split cache write exists only on the batched rope/cache
        // route below. `decode_multi_seq_tree_inner` gates on exactly the same
        // predicate, so this is unreachable — it exists so a future route
        // change surfaces as a loud error instead of a silently flat verify
        // whose branch rows read unseeded scratch blocks.
        if c.tree.is_some() && !self.ms_mla_v4_tree_capable(c) {
            anyhow::bail!(
                "V4-msdecode L{}: tree split armed but the batched rope/cache route is unavailable \
                 (n={n} batch_ok={batch_ok} rows_batch={})",
                self.attn_layer_idx,
                Self::ms_mla_rows_batched(),
            );
        }

        if !batch_ok {
            // Legacy per-row loop: full `attention_forward_v4` per token, O
            // row copied out of the shared `qkv_output` before reuse.
            for i in 0..n {
                let meta_i = Self::meta_row(meta, i);
                let ctx_i = crate::layer::ForwardContext {
                    attn_metadata: Some(meta_i),
                    midchunk_capture: None,
                    ..*c.fwd
                };
                let qkv = c.fwd.buffers.qkv_output();
                let k_out = qkv.offset(q_dim as usize * bf16);
                let v_out = k_out.offset(kv_dim as usize * bf16);
                let args = super::super::super::decode::attention_forward_mla::DecodeMlaArgs {
                    normed: c.normed.offset(i * c.h * bf16),
                    q_out: qkv,
                    k_out,
                    v_out,
                    q_dim,
                    h,
                    nq,
                    hd,
                    eps,
                    bs,
                    stream,
                    // γ-verify (task #45): interleave the compressed-pool
                    // append per row, exactly like plain decode. The up-front
                    // speculate advanced the WHOLE pool before any row's
                    // attention — but plain REWRITES the CSA overlap block
                    // (w-1) as part of each boundary append, so early rows
                    // saw post-rewrite pool bytes where plain's equivalent
                    // decode saw pre-rewrite ones (measured: pool-b2 differed
                    // at the aligned launches; the last byte divergence from
                    // plain greedy). Per-row `pos` makes the append↔attention
                    // ordering identical to plain; the post-accept
                    // restore+catchup still rewinds rejected rows' appends.
                    pos: c.verify_base_pos.map(|b| (b + i) as u32),
                    skip_qkv: false,
                    attn_dest: None,
                };
                let o_v4 = self.attention_forward_v4(kv_cache, &ctx_i, &args)?;
                gpu.copy_d2d_async(o_v4, o_out.offset(i * c.h * bf16), c.h * bf16, stream)?;
            }
            return Ok(o_out);
        }

        // `batch_ok` above already proved a pair covers `n`.
        let gemv = gemv.expect("batch_ok implies a batched-GEMV pair for n");

        let scratch = c.fwd.buffers.expert_up_out();
        let q_batch = scratch; //                                  [n, q_dim]
        let attn_batch = q_batch.offset(row(q_dim)); //            [n, q_dim]
        let kv_batch = attn_batch.offset(row(q_dim)); //           [n, kv_dim]
        let v_batch = kv_batch.offset(row(kv_dim)); //             [n, kv_dim]
        let ql_batch = v_batch.offset(row(kv_dim)); //             [n, q_lora]
        let ol_batch = ql_batch.offset(row(q_lora)); //            [n, latent_dim]

        // ATLAS_PROFILE_VERIFY: A/B/C split of the batched attention. Phase A
        // and C are batched GEMVs (weights read ONCE for all n rows, so they
        // should barely grow with n); phase B is the per-row rope/cache/paged
        // attention loop (grows linearly with n by construction). Which of the
        // three dominates decides whether more batching is worth anything.
        let phase_prof = c.fwd.profile && !c.fwd.graph_capture;
        let mark = |label: &str, t: &mut std::time::Instant| -> Result<()> {
            if phase_prof {
                gpu.synchronize(stream)?;
                tracing::info!(
                    "MLAPROF L{} {label}={}µs",
                    self.attn_layer_idx,
                    t.elapsed().as_micros()
                );
                *t = std::time::Instant::now();
            }
            Ok(())
        };
        let mut t_phase = std::time::Instant::now();
        if phase_prof {
            gpu.synchronize(stream)?;
            t_phase = std::time::Instant::now();
        }
        // Layer-diff harness (task #45): row-0 probes at the batched phase
        // seams, labels matching attention_forward_v4's plain-side probes so
        // the two paths diff line-for-line. Norms, not first4, decide — the
        // L0 divergence hid behind four coincidentally-equal leading elems.
        let diag_this = {
            static D: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *D.get_or_init(|| {
                std::env::var("ATLAS_DIAG_V4_ALL_LAYERS").is_ok_and(|v| v == "1" || v == "true")
            })
        } && !c.fwd.graph_capture;

        // ── ATLAS_VERIFY_EXACT_GEMV (default ON): every batched GEMV
        // projection in this verify (Phase A wq_a/wq_b/wkv AND Phase C
        // wo_a/wo_b) runs a kernel whose per-row accumulation is
        // byte-identical to the single-row kernels plain decode uses. Both
        // stock batch families measurably drift (w4: chunk order + scale
        // regroup; w8: pair-sum fusion — see the kernel headers).
        //
        // MEASURED 2026-08-09, γ=5 serve A/B vs the _ld control: acceptance
        // 2.92–3.01 tok/step vs 2.83, zero-accept 17.7–18.8% vs 20.3%, prose
        // +1 tok/s, repeat parity; tool-eval-bench 90/100 (12/3/0) — the
        // quality bar exactly. A PARTIALLY exact chain is WORSE than either
        // extreme (o-proj-only: 2.54) — flip all legs together or none.
        // `=0` restores the drifted _ld/batchm kernels for A/B.
        let verify_exact_gemv = {
            static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *E.get_or_init(|| {
                std::env::var("ATLAS_VERIFY_EXACT_GEMV").as_deref() != Ok("0")
            })
        } && self.w8a16_gemv_batchm_exact_k.0 != 0
            && self.w4a16_gemv_grouped_batchm_k.0 != 0
            && n <= 8;

        // ── Phase A: batched Q + KV projections (weights read once) ──
        let wqa = mla.wq_a_fp8.as_ref().unwrap();
        if verify_exact_gemv {
            ops::w8a16_gemv_batchm_exact(
                gpu,
                self.w8a16_gemv_batchm_exact_k,
                c.normed,
                wqa.weight,
                wqa.row_scale,
                ql_batch,
                n as u32,
                q_lora,
                h,
                h,
                q_lora,
                stream,
            )?;
        } else {
        ops::w8a16_gemv_batch4(
            gpu,
            gemv.w8,
            c.normed,
            wqa.weight,
            wqa.row_scale,
            ql_batch,
            n as u32,
            q_lora,
            h,
            stream,
        )?;
        }
        ops::rms_norm(
            gpu,
            self.rms_norm_w_k,
            ql_batch,
            &mla.q_a_norm,
            ql_batch,
            n as u32,
            q_lora,
            eps,
            stream,
        )?;
        // Prefer the NVFP4 mirrors (ATLAS_V4_ATTN_NVFP4) for the fat
        // projections — half the FP8 traffic; same weights the single-token
        // decode argmaxes with (precision-consistency matters for the MTP
        // accept rate).
        let nv4_ok = gemv.w4.0 != 0 && gemv.w4_ld.0 != 0;
        if verify_exact_gemv && let Some(ref wqb4) = mla.wq_b_nvfp4 {
            // rows_per_group = N ⇒ single group ⇒ batched plain GEMV in
            // single-row K order.
            ops::w4a16_gemv_grouped_batchm(
                gpu,
                self.w4a16_gemv_grouped_batchm_k,
                ql_batch,
                wqb4,
                q_batch,
                n as u32,
                q_dim,
                q_lora,
                q_lora,
                q_dim,
                q_dim,
                stream,
            )?;
        } else if nv4_ok && let Some(ref wqb4) = mla.wq_b_nvfp4 {
            ops::w4a16_gemv_batchm(
                gpu,
                gemv.w4,
                ql_batch,
                wqb4,
                q_batch,
                n as u32,
                q_dim,
                q_lora,
                stream,
            )?;
        } else {
            let wqb = mla.wq_b_fp8.as_ref().unwrap();
            ops::w8a16_gemv_batch4(
                gpu,
                gemv.w8,
                ql_batch,
                wqb.weight,
                wqb.row_scale,
                q_batch,
                n as u32,
                q_dim,
                q_lora,
                stream,
            )?;
        }
        // q_b_norm: per-head unweighted RMSNorm (see attention_forward_v4).
        ops::rms_norm(
            gpu,
            self.rms_norm_k,
            q_batch,
            &crate::weight_map::DenseWeight {
                weight: c.fwd.buffers.norm_unit_w(),
            },
            q_batch,
            n as u32 * nq,
            hd,
            eps,
            stream,
        )?;
        if diag_this {
            super::super::diag_norm(
                gpu,
                q_batch,
                q_dim as usize,
                stream,
                &format!("V4-msdecode L{} Q after q_b_norm", self.attn_layer_idx),
            );
        }
        let wkv = mla.wkv_a_fp8.as_ref().unwrap();
        if verify_exact_gemv {
            ops::w8a16_gemv_batchm_exact(
                gpu,
                self.w8a16_gemv_batchm_exact_k,
                c.normed,
                wkv.weight,
                wkv.row_scale,
                kv_batch,
                n as u32,
                kv_dim,
                h,
                h,
                kv_dim,
                stream,
            )?;
        } else {
        ops::w8a16_gemv_batch4(
            gpu,
            gemv.w8,
            c.normed,
            wkv.weight,
            wkv.row_scale,
            kv_batch,
            n as u32,
            kv_dim,
            h,
            stream,
        )?;
        }
        ops::rms_norm(
            gpu,
            self.rms_norm_w_k,
            kv_batch,
            &mla.kv_a_norm,
            kv_batch,
            n as u32 * c.nkv as u32,
            kv_dim / c.nkv as u32,
            eps,
            stream,
        )?;
        // K=V for V4-Flash direct KV projection — all n rows in one copy.
        gpu.copy_d2d_async(kv_batch, v_batch, row(kv_dim), stream)?;
        if diag_this {
            super::super::diag_norm(
                gpu,
                kv_batch,
                kv_dim as usize,
                stream,
                &format!("V4-msdecode L{} K after proj", self.attn_layer_idx),
            );
        }

        mark("A_proj", &mut t_phase)?;

        // ── Phase B: batched rope / cache write + batched paged attention ──
        // Every kernel in the rope + cache-write half already takes a row
        // count and is exercised at n≫1 by prefill, so it runs once for all n
        // instead of n times at 1 row (see decode/rows_rope_cache.rs). That was
        // ~347 µs of the measured 467 µs/layer. The remaining ~120 µs of paged
        // attention used to be genuinely per-row — `mla_paged_decode_fp8`
        // indexed only seq_lens/block_tables by blockIdx.y and documented Q/O
        // as `[1, nq * q_dim]`, so raising num_seqs aliased every row onto row
        // 0 — until the kernel gained the `seq_idx * nq * q_head_dim` Q/O
        // stride; it is now launched once at num_seqs=n too.
        //
        // ATLAS_MLA_ROWS_BATCH=0 restores the per-row chain for A/B.
        let rows_batched = Self::ms_mla_rows_batched();
        if rows_batched {
            let rargs = super::super::super::decode::rows_rope_cache::RowsRopeArgs {
                q_batch,
                k_batch: kv_batch,
                v_batch,
                // `ql_batch` is dead after wq_b consumed it above — the
                // single-row chain reuses q_latent for the K rope the same way.
                k_rope_tmp: ql_batch,
                n: n as u32,
                nq,
                hd,
                bs: bs as u32,
                stream,
            };
            // DDTree: `c.tree` splits the cache write into
            // spine → branch-scratch re-seed → branch. `None` on every flat
            // path, where this stays the single whole-batch write.
            self.mla_rows_rope_and_cache(
                kv_cache,
                c.fwd,
                meta,
                &rargs,
                c.tree.map(|t| (t.spine_rows, t.reseed)),
            )?;
            let inv_sqrt_d = self.effective_attn_scale(hd);
            // One launch for all n rows. `mla_paged_decode_fp8` now strides Q/O
            // by `seq_idx * nq * q_head_dim`, and it already indexed
            // `seq_lens[seq_idx]` / `block_tables + seq_idx * max_blocks_per_seq`
            // — which is exactly the row-i offsetting the per-row loop below
            // does by hand, so the two are launch-for-launch equivalent.
            //
            // The occupancy argument: one row is gridDim=[nq=64, 1, 1] = 64
            // CTAs on 48 SMs, so the tail wave runs 16/48 SMs busy and n of
            // those waves run back to back. n rows in one launch is 384 CTAs,
            // which fills 8 full waves instead of n ragged ones.
            //
            // ATLAS_MLA_ATTN_BATCH=0 restores the per-row launches (the kernel
            // change is a no-op at num_seqs=1, so that leg is unaffected).
            let attn_batched = {
                static AB: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
                *AB.get_or_init(|| std::env::var("ATLAS_MLA_ATTN_BATCH").as_deref() != Ok("0"))
            };
            if attn_batched {
                self.run_paged_decode(
                    gpu,
                    q_batch,
                    kv_cache,
                    attn_batch,
                    meta.block_table,
                    meta.seq_len,
                    meta.max_blocks_per_seq,
                    n as u32,
                    nq,
                    c.nkv,
                    hd,
                    bs as u32,
                    inv_sqrt_d,
                    q_dim,
                    c.fwd.buffers.splitk_workspace(),
                    stream,
                )?;
            } else {
                for i in 0..n {
                    self.run_paged_decode(
                        gpu,
                        q_batch.offset(i * q_dim as usize * bf16),
                        kv_cache,
                        attn_batch.offset(i * q_dim as usize * bf16),
                        meta.block_table
                            .offset(i * meta.max_blocks_per_seq as usize * 4),
                        meta.seq_len.offset(i * 4),
                        meta.max_blocks_per_seq,
                        1,
                        nq,
                        c.nkv,
                        hd,
                        bs as u32,
                        inv_sqrt_d,
                        q_dim,
                        c.fwd.buffers.splitk_workspace(),
                        stream,
                    )?;
                }
            }
            if diag_this {
                super::super::diag_norm(
                    gpu,
                    attn_batch,
                    q_dim as usize,
                    stream,
                    &format!("V4-msdecode L{} attn_out", self.attn_layer_idx),
                );
            }
            self.mla_rows_derotate(c.fwd, meta, attn_batch, n as u32, nq, hd, stream)?;
            if diag_this {
                super::super::diag_norm(
                    gpu,
                    attn_batch,
                    q_dim as usize,
                    stream,
                    &format!("V4-msdecode L{} attn_out derot", self.attn_layer_idx),
                );
            }
        } else {
            for i in 0..n {
                let meta_i = Self::meta_row(meta, i);
                let ctx_i = crate::layer::ForwardContext {
                    attn_metadata: Some(meta_i),
                    midchunk_capture: None,
                    ..*c.fwd
                };
                let args = super::super::super::decode::attention_forward_mla::DecodeMlaArgs {
                    normed: c.normed.offset(i * c.h * bf16),
                    q_out: q_batch.offset(i * q_dim as usize * bf16),
                    k_out: kv_batch.offset(i * kv_dim as usize * bf16),
                    v_out: v_batch.offset(i * kv_dim as usize * bf16),
                    q_dim,
                    h,
                    nq,
                    hd,
                    eps,
                    bs,
                    stream,
                    pos: None,
                    skip_qkv: true,
                    attn_dest: Some(attn_batch.offset(i * q_dim as usize * bf16)),
                };
                self.attention_forward_v4(kv_cache, &ctx_i, &args)?;
            }
        }

        mark("B_attn", &mut t_phase)?;

        // ── Phase C: batched O projection ──
        // wo_a is block-diagonal: group g reads attn cols [g*group_in ..) with
        // row stride q_dim and writes latent cols [g*o_lora ..) with row
        // stride latent_dim — the strided batch kernel expresses that
        // directly (offsets mirror attention_forward_v4 Step 6).
        //
        // ATLAS_OPROJ_EXACT=1: run Step 6 PER ROW with the SAME single-row
        // kernels plain decode uses. The layer-diff harness (task #45) proved
        // this phase is the verify's FIRST divergence from plain: with every
        // upstream tensor norm-exact at L0, `attn_out derot` matches to 6
        // digits and `o_out` does not (95.3864 vs 95.3873). The batch4_ld /
        // batchm kernels reduce K in a different order than the single-row
        // GEMVs, the hyper-connection streams amplify the ulps 43 layers deep
        // into the 2-3% capture drift that collapses drafter acceptance
        // (1.06 online vs 3.69 offline). Per-row cost: wo weights re-read per
        // row (~n× the C_oproj 303µs/layer weight traffic) — the price of
        // bit-exactness until the batch kernels adopt the single-row K order.
        let oproj_exact = {
            static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *E.get_or_init(|| std::env::var("ATLAS_OPROJ_EXACT").as_deref() == Ok("1"))
        };
        if oproj_exact {
            {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    tracing::info!(
                        "OPROJ_EXACT: per-row Step-6 O projection active (n={n} nvfp4_woa={} nvfp4_wob={})",
                        mla.wo_a_nvfp4.is_some(),
                        mla.wo_b_nvfp4.is_some(),
                    );
                });
            }
            // Stage every row through the SAME buffers plain's Step 6 uses
            // (attn_output → o_latent → qkv_output). The GEMV kernels pick
            // their vectorized-load path from pointer alignment, so identical
            // VALUES at a different base address can still accumulate in a
            // different order — running at plain's addresses removes the last
            // free variable. Sequential rows on one stream, so reuse is safe.
            let stage_in = c.fwd.buffers.attn_output();
            let stage_lat = c.fwd.buffers.o_latent();
            let stage_out = c.fwd.buffers.qkv_output();
            for i in 0..n {
                gpu.copy_d2d_async(
                    attn_batch.offset(i * q_dim as usize * bf16),
                    stage_in,
                    q_dim as usize * bf16,
                    stream,
                )?;
                let attn_i = stage_in;
                let lat_i = stage_lat;
                for g in 0..o_groups {
                    let in_g = attn_i.offset((g * group_in) as usize * 2);
                    let out_g = lat_i.offset((g * o_lora) as usize * 2);
                    if let Some(ref woa4) = mla.wo_a_nvfp4 {
                        let sub = crate::weight_map::QuantizedWeight {
                            weight: woa4
                                .weight
                                .offset((g as usize) * (o_lora as usize) * (group_in as usize) / 2),
                            weight_scale: woa4
                                .weight_scale
                                .offset((g as usize) * (o_lora as usize) * (group_in as usize / 16)),
                            weight_scale_2: woa4.weight_scale_2,
                            input_scale: woa4.input_scale,
                            weight_scale_2_vec: if woa4.weight_scale_2_vec.is_null() {
                                woa4.weight_scale_2_vec
                            } else {
                                woa4.weight_scale_2_vec.offset((g as usize) * (o_lora as usize) * 4)
                            },
                        };
                        ops::w4a16_gemv(
                            gpu,
                            self.w4a16_gemv_k,
                            in_g,
                            &sub,
                            out_g,
                            o_lora,
                            group_in,
                            stream,
                        )?;
                    } else {
                        let woa = mla.wo_a_fp8.as_ref().unwrap();
                        let w_off = (g as usize) * (o_lora as usize) * (group_in as usize);
                        let s_off =
                            (g as usize) * (o_lora as usize / 128) * (group_in as usize / 128) * 4;
                        ops::w8a16_gemv(
                            gpu,
                            self.w8a16_gemv_k,
                            in_g,
                            woa.weight.offset(w_off),
                            woa.row_scale.offset(s_off),
                            out_g,
                            o_lora,
                            group_in,
                            stream,
                        )?;
                    }
                }
                if let Some(ref wob4) = mla.wo_b_nvfp4 {
                    ops::w4a16_gemv(
                        gpu,
                        self.w4a16_gemv_k,
                        lat_i,
                        wob4,
                        stage_out,
                        h,
                        latent_dim,
                        stream,
                    )?;
                } else {
                    let wob = mla.wo_b_fp8.as_ref().unwrap();
                    ops::w8a16_gemv(
                        gpu,
                        self.w8a16_gemv_k,
                        lat_i,
                        wob.weight,
                        wob.row_scale,
                        stage_out,
                        h,
                        latent_dim,
                        stream,
                    )?;
                }
                gpu.copy_d2d_async(stage_out, o_out.offset(i * c.h * bf16), c.h * bf16, stream)?;
            }
            mark("C_oproj", &mut t_phase)?;
            return Ok(o_out);
        }
        // ── Bit-exact batched O projection (ATLAS_OPROJ_BATCH_EXACT=1,
        // OPT-IN). `w4a16_gemv_grouped_batchm` keeps the single-row K order
        // per row, so the verify's o_out matches plain decode BYTE-FOR-BYTE
        // at 3.07x the per-row OPROJ_EXACT cost (grouped microtest).
        //
        // MEASURED 2026-08-09 serve A/B: o-proj exactness ALONE does not
        // recover acceptance (2.54 vs 2.83 tok/step control) — the residual
        // capture drift lives in the other reordered verify kernels (m=6
        // dedup MoE, batched attention), and the _ld kernels below were the
        // faster end-to-end config (repeat 31.2 vs 28.4). Default therefore
        // stays _ld; this path is the cheap bit-exactness instrument for
        // capture-chain experiments (pair it with a bit-exact MoE leg).
        let oproj_batch_exact = {
            static E: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *E.get_or_init(|| {
                std::env::var("ATLAS_OPROJ_BATCH_EXACT").as_deref() == Ok("1")
            })
        } || verify_exact_gemv;
        if oproj_batch_exact
            && self.w4a16_gemv_grouped_batchm_k.0 != 0
            && n <= 8
            && let (Some(woa4), Some(wob4)) = (&mla.wo_a_nvfp4, &mla.wo_b_nvfp4)
        {
            // wo_a: block-diagonal, rows_per_group = o_lora.
            ops::w4a16_gemv_grouped_batchm(
                gpu,
                self.w4a16_gemv_grouped_batchm_k,
                attn_batch,
                woa4,
                ol_batch,
                n as u32,
                latent_dim,
                group_in,
                q_dim,
                latent_dim,
                o_lora,
                stream,
            )?;
            // wo_b: plain batched GEMV with single-row K order
            // (rows_per_group = N ⇒ every row reads A cols [0..K)).
            ops::w4a16_gemv_grouped_batchm(
                gpu,
                self.w4a16_gemv_grouped_batchm_k,
                ol_batch,
                wob4,
                o_out,
                n as u32,
                h,
                latent_dim,
                latent_dim,
                h,
                h,
                stream,
            )?;
            mark("C_oproj", &mut t_phase)?;
            return Ok(o_out);
        }
        if nv4_ok && let Some(ref woa4) = mla.wo_a_nvfp4 {
            // NVFP4 groups: packed 0.5 B/elem, scales [N, K/16] 1 B row-major,
            // shared per-tensor scale2 (quantized as one tensor).
            for g in 0..o_groups {
                let w_off = (g as usize) * (o_lora as usize) * (group_in as usize) / 2;
                let s_off = (g as usize) * (o_lora as usize) * (group_in as usize / 16);
                ops::w4a16_gemv_batch4_ld(
                    gpu,
                    gemv.w4_ld,
                    attn_batch.offset(g as usize * group_in as usize * bf16),
                    woa4.weight.offset(w_off),
                    woa4.weight_scale.offset(s_off),
                    woa4.weight_scale_2,
                    ol_batch.offset(g as usize * o_lora as usize * bf16),
                    n as u32,
                    o_lora,
                    group_in,
                    q_dim,
                    latent_dim,
                    stream,
                )?;
            }
        } else {
            let woa = mla.wo_a_fp8.as_ref().unwrap();
            for g in 0..o_groups {
                let w_off = (g as usize) * (o_lora as usize) * (group_in as usize);
                let s_off = (g as usize) * (o_lora as usize / 128) * (group_in as usize / 128) * 4;
                ops::w8a16_gemv_batch4_ld(
                    gpu,
                    gemv.w8_ld,
                    attn_batch.offset(g as usize * group_in as usize * bf16),
                    woa.weight.offset(w_off),
                    woa.row_scale.offset(s_off),
                    ol_batch.offset(g as usize * o_lora as usize * bf16),
                    n as u32,
                    o_lora,
                    group_in,
                    q_dim,
                    latent_dim,
                    stream,
                )?;
            }
        }
        if nv4_ok && let Some(ref wob4) = mla.wo_b_nvfp4 {
            ops::w4a16_gemv_batchm(
                gpu,
                gemv.w4,
                ol_batch,
                wob4,
                o_out,
                n as u32,
                h,
                latent_dim,
                stream,
            )?;
        } else {
            let wob = mla.wo_b_fp8.as_ref().unwrap();
            ops::w8a16_gemv_batch4(
                gpu,
                gemv.w8,
                ol_batch,
                wob.weight,
                wob.row_scale,
                o_out,
                n as u32,
                h,
                latent_dim,
                stream,
            )?;
        }
        mark("C_oproj", &mut t_phase)?;
        Ok(o_out)
    }

    /// Per-row view of the batched attention metadata (positions `[n]` u32,
    /// slot `[n]` i64, seq_len `[n]` i32, block_table `[n, max_blocks]` i32).
    fn meta_row(meta: &AttnMetadataDev, i: usize) -> AttnMetadataDev {
        AttnMetadataDev {
            positions: meta.positions.offset(i * 4),
            positions_h: meta.positions_h.offset(i * 4),
            positions_w: meta.positions_w.offset(i * 4),
            slot: meta.slot.offset(i * 8),
            seq_len: meta.seq_len.offset(i * 4),
            block_table: meta
                .block_table
                .offset(i * meta.max_blocks_per_seq as usize * 4),
            max_blocks_per_seq: meta.max_blocks_per_seq,
            num_seqs: 1,
            seq_slot: spark_runtime::gpu::DevicePtr(0),
        }
    }

    /// Single-sequence absorbed-MLA decode chain. Mirrors
    /// `decode::attention_forward_mla` 1:1 but takes an explicit
    /// per-sequence `normed` input and `o_out` destination so the caller
    /// can drive it once per sequence in a batched decode step.
    #[allow(clippy::too_many_arguments)]
    fn ms_mla_decode_one(
        &self,
        c: &MultiSeqCtx<'_>,
        kv_cache: &mut PagedKvCache,
        meta: &AttnMetadataDev,
        normed: DevicePtr,
        o_out: DevicePtr,
        mla: &crate::layers::qwen3_attention::types::MlaWeights,
        d: MlaDims,
        stream: u64,
    ) -> Result<()> {
        let gpu = c.fwd.gpu;
        let buffers = c.fwd.buffers;

        // ── Step 1: Q latent → norm → expand ──
        let q_latent = buffers.ssm_ba();
        if let Some(ref wqa_nvfp4) = mla.wq_a_nvfp4 {
            ops::w4a16_gemv(
                gpu,
                self.w4a16_gemv_k,
                normed,
                wqa_nvfp4,
                q_latent,
                d.q_lora,
                d.h,
                stream,
            )?;
        } else {
            ops::dense_gemv(
                gpu,
                self.dense_gemv_k,
                normed,
                &mla.wq_a,
                q_latent,
                d.q_lora,
                d.h,
                stream,
            )?;
        }
        ops::rms_norm(
            gpu,
            self.rms_norm_w_k,
            q_latent,
            &mla.q_a_norm,
            q_latent,
            1,
            d.q_lora,
            d.eps,
            stream,
        )?;
        let q_full = buffers.ssm_deinterleaved();
        if let Some(ref wqb_nvfp4) = mla.wq_b_nvfp4 {
            ops::w4a16_gemv(
                gpu,
                self.w4a16_gemv_k,
                q_latent,
                wqb_nvfp4,
                q_full,
                d.q_dim,
                d.q_lora,
                stream,
            )?;
        } else {
            ops::dense_gemv(
                gpu,
                self.dense_gemv_k,
                q_latent,
                &mla.wq_b,
                q_full,
                d.q_dim,
                d.q_lora,
                stream,
            )?;
        }

        // ── Step 2: Q_absorbed (Q_nope @ W_UK_T) ──
        let q_absorbed_buf = buffers.expert_up_out();
        self.ms_mla_q_absorb(c, mla, &d, q_full, q_absorbed_buf, stream)?;

        // Q_rope scatter (rope half of q_full → strided absorbed layout).
        let q_rope_direct = buffers.ssm_conv_out_f32();
        if self.mla_q_rope_scatter_k.0 != 0 {
            ops::mla_q_rope_scatter(
                gpu,
                self.mla_q_rope_scatter_k,
                q_full,
                q_absorbed_buf,
                q_rope_direct,
                d.nq,
                d.hd,
                d.mla_nope,
                d.mla_rope,
                d.kv_lora,
                d.mla_cache_dim,
                stream,
            )?;
        } else {
            for head_idx in 0..d.nq as usize {
                let src = q_full.offset((head_idx * d.hd as usize + mla.nope) * 2);
                gpu.copy_d2d_async(
                    src,
                    q_rope_direct.offset(head_idx * mla.rope * 2),
                    mla.rope * 2,
                    stream,
                )?;
                gpu.copy_d2d_async(
                    src,
                    q_absorbed_buf
                        .offset((head_idx * d.mla_cache_dim as usize + mla.kv_lora_rank) * 2),
                    mla.rope * 2,
                    stream,
                )?;
            }
        }

        // ── Step 3: KV latent → norm ──
        let kv_latent = buffers.expert_gate_out();
        if let Some(ref wkva_nvfp4) = mla.wkv_a_nvfp4 {
            ops::w4a16_gemv(
                gpu,
                self.w4a16_gemv_k,
                normed,
                wkva_nvfp4,
                kv_latent,
                d.kv_lora,
                d.h,
                stream,
            )?;
        } else {
            ops::dense_gemv(
                gpu,
                self.dense_gemv_k,
                normed,
                &mla.wkv_a,
                kv_latent,
                d.kv_lora,
                d.h,
                stream,
            )?;
        }
        ops::rms_norm(
            gpu,
            self.rms_norm_w_k,
            kv_latent,
            &mla.kv_a_norm,
            kv_latent,
            1,
            d.kv_lora,
            d.eps,
            stream,
        )?;

        // ── Step 4: K_rope + RoPE + writeback ──
        // `k_rope_single` reuses `ssm_ba` — safe: `q_latent` (the prior
        // `ssm_ba` user) was fully consumed by the `wq_b` GEMV above.
        let k_rope_single = buffers.ssm_ba();
        ops::dense_gemv(
            gpu,
            self.dense_gemv_k,
            normed,
            &mla.wkv_a_rope,
            k_rope_single,
            d.mla_rope,
            d.h,
            stream,
        )?;
        ops::rope_yarn(
            gpu,
            self.rope_yarn_k,
            q_rope_direct,
            k_rope_single,
            meta.positions,
            1,
            d.nq,
            1,
            d.mla_rope,
            d.mla_rope,
            mla.yarn_inv_freq,
            c.fwd.config.rope_theta as f32,
            stream,
        )?;
        if self.mla_q_rope_writeback_k.0 != 0 {
            ops::mla_q_rope_writeback(
                gpu,
                self.mla_q_rope_writeback_k,
                q_rope_direct,
                q_absorbed_buf,
                d.nq,
                d.mla_rope,
                d.kv_lora,
                d.mla_cache_dim,
                stream,
            )?;
        } else {
            for head_idx in 0..d.nq as usize {
                let src = q_rope_direct.offset(head_idx * mla.rope * 2);
                let dst = q_absorbed_buf
                    .offset((head_idx * d.mla_cache_dim as usize + mla.kv_lora_rank) * 2);
                gpu.copy_d2d_async(src, dst, mla.rope * 2, stream)?;
            }
        }

        // ── Step 5: cache assemble + write (this seq's slot) ──
        // `k_out`/`v_out` use this layer's private QKV scratch region.
        let k_cache_entry = buffers.qkv_output();
        let v_cache_entry = k_cache_entry.offset(d.mla_cache_dim as usize * 2);
        if self.mla_cache_assemble_k.0 != 0 {
            ops::mla_cache_assemble(
                gpu,
                self.mla_cache_assemble_k,
                kv_latent,
                k_rope_single,
                k_cache_entry,
                v_cache_entry,
                d.kv_lora,
                d.mla_rope,
                d.mla_cache_dim,
                stream,
            )?;
        } else {
            gpu.copy_d2d_async(kv_latent, k_cache_entry, mla.kv_lora_rank * 2, stream)?;
            gpu.copy_d2d_async(
                k_rope_single,
                k_cache_entry.offset(mla.kv_lora_rank * 2),
                mla.rope * 2,
                stream,
            )?;
            gpu.copy_d2d_async(kv_latent, v_cache_entry, mla.kv_lora_rank * 2, stream)?;
            gpu.memset_async(
                v_cache_entry.offset(mla.kv_lora_rank * 2),
                0,
                mla.rope * 2,
                stream,
            )?;
        }
        self.write_kv_cache(
            gpu,
            k_cache_entry,
            v_cache_entry,
            kv_cache,
            meta.slot,
            1,
            1,
            d.mla_cache_dim,
            d.bs as u32,
            d.mla_cache_dim,
            d.mla_cache_dim,
            stream,
            c.fwd.graph_capture,
        )?;

        // ── Step 6: paged decode attention (this seq only) ──
        let attn_out = buffers.attn_output();
        ops::paged_decode_attn_bf16(
            gpu,
            self.paged_decode_mla_k,
            q_absorbed_buf,
            kv_cache.k_pool_ptr(self.attn_layer_idx),
            kv_cache.v_pool_ptr(self.attn_layer_idx),
            attn_out,
            meta.block_table,
            meta.seq_len,
            meta.max_blocks_per_seq,
            1,
            d.nq,
            1,
            d.mla_cache_dim,
            d.bs as u32,
            d.inv_sqrt_d,
            d.nq * d.mla_cache_dim,
            0,
            stream,
        )?;

        // ── Step 7: V extraction (attn_latent @ W_UV) ──
        // `ssm_qkvz` (not `norm_output`) — `norm_output` holds the `n`
        // per-sequence `normed` inputs that later loop iterations still
        // need; writing `v_extracted` there would clobber them.
        let v_extracted = buffers.ssm_qkvz();
        self.ms_mla_v_extract(c, mla, &d, attn_out, v_extracted, stream)?;

        // ── Step 8: O projection → this seq's o_out slot ──
        if d.o_lora_rank > 0 {
            // DeepSeek-V4-Flash: low-rank O projection (wo_a → wo_b)
            let o_latent = buffers.attn_output();
            if let Some(ref woa_nvfp4) = mla.wo_a_nvfp4 {
                ops::w4a16_gemv(
                    gpu,
                    self.w4a16_gemv_k,
                    v_extracted,
                    woa_nvfp4,
                    o_latent,
                    d.o_lora_rank,
                    d.nq * d.mla_v_dim,
                    stream,
                )?;
            } else {
                ops::dense_gemv(
                    gpu,
                    self.dense_gemv_k,
                    v_extracted,
                    &mla.wo_a,
                    o_latent,
                    d.o_lora_rank,
                    d.nq * d.mla_v_dim,
                    stream,
                )?;
            }
            if let Some(ref wob_nvfp4) = mla.wo_b_nvfp4 {
                ops::w4a16_gemv(
                    gpu,
                    self.w4a16_gemv_k,
                    o_latent,
                    wob_nvfp4,
                    o_out,
                    d.h,
                    d.o_lora_rank,
                    stream,
                )?;
            } else {
                ops::dense_gemv(
                    gpu,
                    self.dense_gemv_k,
                    o_latent,
                    &mla.wo_b,
                    o_out,
                    d.h,
                    d.o_lora_rank,
                    stream,
                )?;
            }
        } else if let Some(ref wo_nvfp4) = mla.wo_nvfp4 {
            ops::w4a16_gemv(
                gpu,
                self.w4a16_gemv_k,
                v_extracted,
                wo_nvfp4,
                o_out,
                d.h,
                d.nq * d.mla_v_dim,
                stream,
            )?;
        } else {
            ops::dense_gemv(
                gpu,
                self.dense_gemv_k,
                v_extracted,
                &mla.wo,
                o_out,
                d.h,
                d.nq * d.mla_v_dim,
                stream,
            )?;
        }
        Ok(())
    }
}
