// SPDX-License-Identifier: AGPL-3.0-only

//! Multi-sequence batched-decode body for [`super::super::Qwen3AttentionLayer`].
//!
//! Split into phase modules under the `_inner` delegation pattern:
//! - `ctx`  — `MultiSeqCtx` shared scalars + buffer pointers
//! - `qkv`  — phase 2: per-token Q/K/V projections (batch3/batch2/seq)
//! - `attn` — phases 3-6: RoPE → cache write → paged decode → O proj
//! - `ffn`  — phase 7: residual + post-norm + MoE/dense FFN
//!
//! The trait impl in `super::trait_impl` calls
//! [`Qwen3AttentionLayer::decode_multi_seq_inner`] which simply builds
//! the ctx, runs phase 1 inline (RMS norm), and dispatches the rest.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::Qwen3AttentionLayer;
use crate::layer::{ForwardContext, LayerState};
use crate::layers::ops;

mod attn;
mod ctx;
mod ffn;
mod mla;
mod mla_gemv;
mod qkv;
mod tree;

impl Qwen3AttentionLayer {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::layers::qwen3_attention) fn decode_multi_seq_inner<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        kv_cache: &mut PagedKvCache,
        seq_lens: &[usize],
        _block_tables: &[Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let _ = states; // Attention layers use EmptyLayerState — no per-seq state.
        let bs = kv_cache.block_size() as u32;
        let mut c = ctx::MultiSeqCtx::new(self, ctx, hidden, residual, num_seqs, bs, stream);
        // V4 compressed-arm speculation needs row 0's ABSOLUTE position on the
        // host (see `MultiSeqCtx::verify_base_pos`). `seq_lens` is the only
        // host-side view of it that reaches this path — everything else the
        // attention consumes is the pre-uploaded device `attn_metadata`.
        c.verify_base_pos = if crate::layers::qwen3_attention::DFLASH_VERIFY_ACTIVE.load(std::sync::atomic::Ordering::Relaxed)
        {
            ctx::verify_base_pos_of(seq_lens, num_seqs)
        } else {
            // Not the γ-verify: other multi-row forwards (drafter-adjacent
            // per-step forwards, warmups) also present contiguous seq_lens,
            // and running the compressor speculation there poisons prev_win
            // with non-committed rows (task #45's last divergence).
            None
        };
        // Per-request LoRA routing slot buffer for this step (from metadata).
        if let Some(m) = ctx.attn_metadata.as_ref() {
            c.seq_slot = m.seq_slot;
        }

        // DeepSeek-V4: Manifold-Constrained Hyper-Connections (mHC).
        if self.hc.is_some() {
            return self.decode_multi_seq_inner_hc(c, kv_cache, ctx, stream);
        }

        // ATLAS_FUSED_ELEMWISE=1: fold the post-QKV-GEMM elementwise swarm
        // (3n scatter D2Ds + 2n per-head norms + n rope + n cache writes +
        // n Q-gather D2Ds = 8n launches/layer) into ONE bit-identical kernel.
        // Load/shape-stable predicate → graph-capture safe.
        c.fused_qk_epilogue = self.ms_fused_epilogue_eligible(&c);

        // ── Phase 1: RMS norm + residual for N tokens ──
        ops::rms_norm_residual(
            ctx.gpu,
            self.rms_norm_residual_k,
            c.hidden,
            &self.input_norm,
            c.normed,
            c.residual,
            c.n as u32,
            c.h as u32,
            c.eps,
            c.stream,
        )?;

        let meta = ctx
            .attn_metadata
            .expect("attention layer requires metadata");

        // ATLAS_VERIFY_PROFILE=1 (eager): accumulate attention-vs-FFN split
        // across all layer calls; report every 48 calls (= one verify pass).
        static VP2_ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let vp2 = *VP2_ENV
            .get_or_init(|| std::env::var("ATLAS_VERIFY_PROFILE").ok().as_deref() == Some("1"))
            && !ctx.graph_capture;
        static ATTN_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        static FFN_US: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        static CALLS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let vp2_t0 = std::time::Instant::now();

        // ── Phases 2-6: attention ──
        // MLA models (Mistral-Small-4) take the dedicated absorbed-MLA
        // batched path (issue #84). The standard `ms_phase_qkv` reads
        // `attn.q_proj`, a NULL stub for MLA loaders — see `mla.rs`.
        static P_QKV: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        static P_ROPE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        static P_CACHE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        static P_PAGED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        static P_OPROJ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        macro_rules! ph {
            ($acc:ident, $body:expr) => {{
                if vp2 {
                    let t = std::time::Instant::now();
                    let r = $body;
                    ctx.gpu.synchronize(stream)?;
                    $acc.fetch_add(
                        t.elapsed().as_micros() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                    r
                } else {
                    $body
                }
            }};
        }
        let o_out = if let Some(ref _mla) = self.mla {
            self.ms_mla_decode(&c, kv_cache, meta)?
        } else {
            // ── Phase 2: QKV projections (batch3 / batch2 / sequential) ──
            ph!(P_QKV, self.ms_phase_qkv(&c)?);

            if c.fused_qk_epilogue {
                // ── Phases 3+4 fused (+ the norms/scatter/gather the other
                // phases skipped): one launch, bit-identical chain. ──
                ph!(P_ROPE, self.ms_fused_qk_epilogue(&c, kv_cache, meta)?);
            } else {
                // ── Phase 3: RoPE per-sequence ──
                ph!(P_ROPE, self.ms_phase_rope(&c, meta)?);

                // ── Phase 4: KV cache write ──
                ph!(P_CACHE, self.ms_phase_cache_write(&c, kv_cache, meta)?);
            }

            // ── Phase 5: paged decode attention (batched) ──
            let attn_out = ph!(P_PAGED, self.ms_phase_paged_decode(&c, kv_cache, meta)?);

            // ── Phase 6: gate multiply + O projection ──
            ph!(P_OPROJ, self.ms_phase_o_proj(&c, attn_out)?)
        };

        // TP all-reduce on o_out after o_proj (Megatron row-parallel
        // pattern). Mirrors decode_inner.rs and prefill_inner.rs. Without
        // this, multi-token decode (K=2 / K=3 / K=γ verify) under
        // tp_world_size>1 reads a partial attention output from each
        // rank, corrupting the FFN/MoE input and producing degenerate
        // logits — observed as `/`/`,` repetition spirals on
        // Qwen3.6 FP8 + TP=2 + MTP for HTML/code prompts.
        if c.fwd.config.tp_world_size > 1
            && let Some(comm) = c.fwd.comm
        {
            let bytes = c.n * c.h * c.bf16;
            comm.all_reduce_async(o_out.0, bytes, c.stream)?;
        }

        if vp2 {
            c.fwd.gpu.synchronize(c.stream)?;
            ATTN_US.fetch_add(
                vp2_t0.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }
        let vp2_t1 = std::time::Instant::now();

        // ── Phase 7: residual + post-norm + MoE ──
        self.ms_phase_ffn(&c, o_out)?;

        if vp2 {
            c.fwd.gpu.synchronize(c.stream)?;
            FFN_US.fetch_add(
                vp2_t1.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            let calls = CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if calls % 48 == 0 {
                let a = ATTN_US.swap(0, std::sync::atomic::Ordering::Relaxed);
                let f = FFN_US.swap(0, std::sync::atomic::Ordering::Relaxed);
                use std::sync::atomic::Ordering::Relaxed;
                tracing::info!(
                    "VERIFY_SPLIT (48 layer-calls): attn={:.1}ms ffn={:.1}ms | qkv={:.1} rope={:.1} cache={:.1} paged={:.1} oproj={:.1}",
                    a as f64 / 1000.0,
                    f as f64 / 1000.0,
                    P_QKV.swap(0, Relaxed) as f64 / 1000.0,
                    P_ROPE.swap(0, Relaxed) as f64 / 1000.0,
                    P_CACHE.swap(0, Relaxed) as f64 / 1000.0,
                    P_PAGED.swap(0, Relaxed) as f64 / 1000.0,
                    P_OPROJ.swap(0, Relaxed) as f64 / 1000.0,
                );
            }
        }

        Ok(())
    }

    /// HC-enabled batched multi-sequence decode.  Only the sequential
    /// per-token FFN branch is implemented (DeepSeek-V4 MLA always
    /// takes this path).
    fn decode_multi_seq_inner_hc(
        &self,
        c: ctx::MultiSeqCtx<'_>,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let n = c.n;
        let hc = self.hc.as_ref().unwrap();
        let hc_mult = hc.hc_mult as u32;
        let is_first_layer = self.attn_layer_idx == 0;
        let is_last_layer = self.attn_layer_idx + 1 == ctx.config.num_hidden_layers;
        let hc_streams = ctx.buffers.hc_streams();
        let post = ctx.buffers.hc_post();
        let comb = ctx.buffers.hc_comb();
        // One block per 256 hidden lanes — 16 at H=4096 — the same shard count
        // the plain path computes at `decode_inner.rs:521`. `hc_post` is a
        // grid-stride loop over the hidden dim
        // (`d = blockIdx.y*HC_BLOCK + tid; d < H; d += HC_BLOCK*gridDim.y`,
        // `hyper_connection.cu:637`) whose iterations are independent: no
        // __syncthreads, no atomics, no shared-memory reduction, and `out` is
        // only ever written at the `d` each lane owns. So extra blocks purely
        // partition the same lanes and every output element keeps identical
        // arithmetic. Constant at graph-capture time (H is fixed), so this is
        // graph-safe. All three verify-path sites below used the unsharded
        // `ops::hc_post`, i.e. grid.y == 1: at n=6 that put 6 rows on 6 of 48
        // SMs, and the per-row loop sites put ONE row on ONE SM.
        let post_shards = (h as u32).div_ceil(256);
        // diag_norm syncs the stream — illegal under CUDA-graph capture (see
        // decode_inner.rs); never probe while capturing.
        let diag_this = std::env::var("ATLAS_DIAG_V4_ALL_LAYERS")
            .is_ok_and(|v| v == "1" || v == "true")
            && !ctx.graph_capture;
        // ATLAS_PROFILE_VERIFY phase split: attention block vs mHC+FFN block.
        let t_phase = if ctx.profile && !ctx.graph_capture {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        if is_first_layer {
            // Task #45: raw layer-entry hidden per row (= the embedding at
            // L0). Row 1 collapsed to a DIFFERENT token's collapse than its
            // host-side draft — this probe, hashed against embedding-table
            // rows read straight from the safetensors, decides whether the
            // wrong token was embedded or the row was clobbered afterwards.
            if diag_this {
                for r in 0..n {
                    super::diag_norm(
                        ctx.gpu,
                        c.hidden.offset(r * c.h * c.bf16),
                        h,
                        stream,
                        &format!("V4-msdecode L{} pre r{r}", self.attn_layer_idx),
                    );
                }
            }
            ops::hc_expand(
                ctx.gpu,
                self.hc_expand_k,
                c.hidden,
                hc_streams,
                n as u32,
                h as u32,
                hc_mult,
                stream,
            )?;
        }

        // Fused `hc_pre` streams the whole ~1.5 MiB `hc_fn` matrix in ONE
        // block per token — at n=2 that is 2 of the GB10's SMs doing all the
        // mHC work, twice per layer (~the largest single cost of the K=2
        // verify before this). Use the sharded mix+finish split whenever its
        // kernels are present, exactly like the single-token decode
        // (decode_inner.rs); `ATLAS_HC_SPLIT=0` restores fused for A/B.
        let hc_split = self.hc_pre_mix_k.0 != 0
            && self.hc_pre_finish_k.0 != 0
            && !std::env::var("ATLAS_HC_SPLIT").is_ok_and(|v| v == "0");
        let hc_pre_dispatch = |site: &super::super::types_weights::HcSiteWeights| -> Result<()> {
            if hc_split {
                ops::hc_pre_split(
                    ctx.gpu,
                    self.hc_pre_mix_k,
                    self.hc_pre_finish_k,
                    hc_streams,
                    site.hc_fn,
                    site.hc_scale,
                    site.hc_base,
                    c.hidden,
                    post,
                    comb,
                    ctx.buffers.hc_mix(),
                    n as u32,
                    h as u32,
                    hc_mult,
                    hc.sinkhorn_iters as u32,
                    eps,
                    hc.hc_eps,
                    stream,
                )
            } else {
                ops::hc_pre(
                    ctx.gpu,
                    self.hc_pre_k,
                    hc_streams,
                    site.hc_fn,
                    site.hc_scale,
                    site.hc_base,
                    c.hidden,
                    post,
                    comb,
                    n as u32,
                    h as u32,
                    hc_mult,
                    hc.sinkhorn_iters as u32,
                    eps,
                    hc.hc_eps,
                    stream,
                )
            }
        };

        // ── Phase 1: collapse + norm for N tokens ──
        hc_pre_dispatch(&hc.attn)?;
        if diag_this {
            super::diag_norm(
                ctx.gpu,
                c.hidden,
                n * h,
                stream,
                &format!("V4-msdecode L{} hc_pre-attn", self.attn_layer_idx),
            );
            super::diag_norm_f32(
                ctx.gpu,
                post,
                n * (hc_mult as usize),
                stream,
                &format!("V4-msdecode L{} post-attn", self.attn_layer_idx),
            );
            super::diag_norm_f32(
                ctx.gpu,
                comb,
                n * (hc_mult as usize) * (hc_mult as usize),
                stream,
                &format!("V4-msdecode L{} comb-attn", self.attn_layer_idx),
            );
        }
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_w_k,
            c.hidden,
            &self.input_norm,
            c.normed,
            n as u32,
            h as u32,
            eps,
            stream,
        )?;
        // Task #45 rows>0 localization: per-row collapsed hidden and normed —
        // rows>0 diverge from plain at the MLA's Q input, so the layer-front
        // (hc collapse → rms) is under byte-level suspicion per row.
        if diag_this {
            for r in 0..n {
                super::diag_norm(
                    ctx.gpu,
                    c.hidden.offset(r * c.h * c.bf16),
                    h,
                    stream,
                    &format!("V4-msdecode L{} collapsed r{r}", self.attn_layer_idx),
                );
                super::diag_norm(
                    ctx.gpu,
                    c.normed.offset(r * c.h * c.bf16),
                    h,
                    stream,
                    &format!("V4-msdecode L{} normed r{r}", self.attn_layer_idx),
                );
            }
        }

        let meta = ctx
            .attn_metadata
            .expect("attention layer requires metadata");

        // ── Phases 2-6: attention ──
        let o_out = if let Some(ref _mla) = self.mla {
            self.ms_mla_decode(&c, kv_cache, meta)?
        } else {
            self.ms_phase_qkv(&c)?;
            self.ms_phase_rope(&c, meta)?;
            self.ms_phase_cache_write(&c, kv_cache, meta)?;
            let attn_out = self.ms_phase_paged_decode(&c, kv_cache, meta)?;
            self.ms_phase_o_proj(&c, attn_out)?
        };

        if c.fwd.config.tp_world_size > 1
            && let Some(comm) = c.fwd.comm
        {
            let bytes = c.n * c.h * c.bf16;
            comm.all_reduce_async(o_out.0, bytes, c.stream)?;
        }

        // Row-0 attention output, pre-hc_post — pairs with the plain path's
        // "V4-decode L{} o_out" probe (attention_forward_v4). Layer-diff
        // harness: this is the only tensor in the L0 divergence window
        // [attention epilogue → hc_post(attn) → hc_pre(ffn)] that had no
        // spec-side probe.
        if diag_this {
            super::diag_norm(
                ctx.gpu,
                o_out,
                h,
                stream,
                &format!("V4-msdecode L{} o_out", self.attn_layer_idx),
            );
        }
        // Expand attention output back into multi-stream state.
        // Sharded over the hidden dim — see the `post_shards` note above; the
        // grid goes (n,1,1) -> (n,16,1) with bit-identical output.
        ops::hc_post_sharded(
            ctx.gpu,
            self.hc_post_k,
            o_out,
            hc_streams,
            post,
            comb,
            hc_streams,
            n as u32,
            h as u32,
            hc_mult,
            post_shards,
            stream,
        )?;
        if diag_this {
            super::diag_norm(
                ctx.gpu,
                hc_streams,
                h,
                stream,
                &format!("V4-msdecode L{} hc_post-attn", self.attn_layer_idx),
            );
            super::diag_norm(
                ctx.gpu,
                hc_streams,
                n * (hc_mult as usize) * h,
                stream,
                &format!(
                    "V4-msdecode L{} hc_post-attn ALL_STREAMS",
                    self.attn_layer_idx
                ),
            );
        }

        let attn_us = match t_phase {
            Some(t0) => {
                ctx.gpu.synchronize(stream)?;
                Some(t0.elapsed().as_micros())
            }
            None => None,
        };
        let t_ffn = attn_us.map(|_| std::time::Instant::now());

        // Standalone attention (no FFN)
        if self.ffn.is_none() {
            if is_last_layer && let Some(ref head) = hc.head {
                ops::hc_head(
                    ctx.gpu,
                    self.hc_head_k,
                    hc_streams,
                    head.hc_fn,
                    head.hc_scale,
                    head.hc_base,
                    c.hidden,
                    n as u32,
                    h as u32,
                    hc_mult,
                    eps,
                    hc.hc_eps,
                    stream,
                )?;
                if diag_this {
                    super::diag_norm(
                        ctx.gpu,
                        c.hidden,
                        n * h,
                        stream,
                        &format!("V4-msdecode L{} hc_head", self.attn_layer_idx),
                    );
                }
            } else if is_last_layer {
                tracing::warn!(
                    "V4-msdecode L{}: hc_head SKIPPED (no head weights)",
                    self.attn_layer_idx
                );
            }
            return Ok(());
        }

        // ── Phase 7: FFN + hc_post (per-token sequential only) ──
        hc_pre_dispatch(&hc.ffn)?;
        if diag_this {
            super::diag_norm(
                ctx.gpu,
                c.hidden,
                n * h,
                stream,
                &format!("V4-msdecode L{} hc_pre-ffn", self.attn_layer_idx),
            );
            super::diag_norm_f32(
                ctx.gpu,
                post,
                n * (hc_mult as usize),
                stream,
                &format!("V4-msdecode L{} post-ffn", self.attn_layer_idx),
            );
            super::diag_norm_f32(
                ctx.gpu,
                comb,
                n * (hc_mult as usize) * (hc_mult as usize),
                stream,
                &format!("V4-msdecode L{} comb-ffn", self.attn_layer_idx),
            );
        }
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_w_k,
            c.hidden,
            &self.post_attn_norm,
            c.normed,
            n as u32,
            h as u32,
            eps,
            stream,
        )?;

        // n == 2: fused K=2 MoE — gate batch2 + batched topK + the MROW=2
        // dedup'd `_t` expert kernels, so the shared expert, the gate, and
        // every expert both candidate rows happen to share are read ONCE.
        //
        // Default-on only when `k2_verify_ffn_is_batched` says the fast path
        // will fire. Batching is NOT unconditionally a win: the older
        // `batch2_t` fallback is the pre-split-K kernel shape and measured 17.0
        // tok/s against 19.8 for this per-token loop. The MROW=2 rewrite
        // measured 21.0. `ATLAS_MSHC_FFN_K2=0` forces the per-token loop back.
        let ffn_k2 = n == 2
            && std::env::var("ATLAS_MSHC_FFN_K2").as_deref() != Ok("0")
            && self.ffn.k2_verify_ffn_is_batched(ctx);
        if ffn_k2 {
            self.ffn.forward_k2(c.normed, ctx, stream)?;
            let moe_out_base = ctx.buffers.moe_output();
            for i in 0..n {
                let moe_out = moe_out_base.offset(i * c.h * c.bf16);
                let hc_streams_i = hc_streams.offset(i * hc.hc_mult * c.h * 4);
                let post_i = post.offset(i * hc.hc_mult * 4);
                let comb_i = comb.offset(i * hc.hc_mult * hc.hc_mult * 4);
                // Sharded over the hidden dim — see `post_shards` above. This
                // is a per-ROW loop, so the unsharded form was grid (1,1,1):
                // one CTA of 48 SMs, n times per layer. Bit-identical.
                ops::hc_post_sharded(
                    ctx.gpu,
                    self.hc_post_k,
                    moe_out,
                    hc_streams_i,
                    post_i,
                    comb_i,
                    hc_streams_i,
                    1,
                    h as u32,
                    hc_mult,
                    post_shards,
                    stream,
                )?;
            }
        } else {
            // WIDE VERIFY (DSpark γ=6 → n=6). One dedup'd dispatch for all n
            // rows, ahead of the per-token loop, for two reasons:
            //
            //  1. Bandwidth. `forward` reads every routed expert's ~94 MB layer
            //     once PER ROW. At n=6 that is 72% of the verify step
            //     (measured: MoE 6×544µs vs attention 1.29ms per layer). The
            //     dedup'd `_t` split-K kernels read each expert once for every
            //     row that selected it.
            //  2. Correctness. `forward` is a single-token entry point — its
            //     hash-MoE routing reads `token_ids[0]` unconditionally
            //     (moe/forward.rs: "decode: single token at offset 0"), so rows
            //     1.. were routed with row 0's experts. On DeepSeek-V4's hash
            //     layers that silently corrupts the verify logits, which costs
            //     acceptance directly.
            //
            // `forward_verify_rows` falls back internally (and returns false
            // for dense/no FFN), so the per-token loop below still covers every
            // layer and width it declines.
            let batched = n > 1 && self.ffn.forward_verify_rows(c.normed, n, ctx, stream)?;
            // Per-row FFN output — pairs with plain's "V4-decode L{} ffn-out"
            // for the task-#45 byte-level layer diff. Row r of the verify at
            // position p+r computes the SAME function plain's step r+1 does
            // (when the draft matches plain's token), so occurrence r here
            // byte-pairs with plain-log occurrence r+1. Row 0 exactness is
            // proven; rows>0 are where the bonus token still leaves the plain
            // stream.
            if batched && diag_this {
                for r in 0..n {
                    super::diag_norm(
                        ctx.gpu,
                        ctx.buffers.moe_output().offset(r * c.h * c.bf16),
                        h,
                        stream,
                        &format!("V4-msdecode L{} ffn-out r{r}", self.attn_layer_idx),
                    );
                }
            }
            // ATLAS_MOE_OVERLAP=1 (no-op otherwise): open a row group so the
            // per-row MoE fires below can be scored for expert-set overlap.
            // The batched dispatch routes all rows in one launch — nothing to
            // score, and `forward_km` samples the union itself.
            if !batched {
                crate::layers::moe::dump::route_group_begin(n);
            }
            let moe_out_base = ctx.buffers.moe_output();
            for i in 0..n {
                let moe_out = if batched {
                    moe_out_base.offset(i * c.h * c.bf16)
                } else {
                    let normed2_i = c.normed.offset(i * c.h * c.bf16);
                    self.ffn.forward(normed2_i, ctx, stream)?
                };
                // hc_streams is the FP32 mHC highway (4 bytes/elem), not BF16.
                let hc_streams_i = hc_streams.offset(i * hc.hc_mult * c.h * 4);
                let post_i = post.offset(i * hc.hc_mult * 4);
                let comb_i = comb.offset(i * hc.hc_mult * hc.hc_mult * 4);
                // Sharded over the hidden dim — see `post_shards` above. This
                // is a per-ROW loop, so the unsharded form was grid (1,1,1):
                // one CTA of 48 SMs, n times per layer. Bit-identical.
                ops::hc_post_sharded(
                    ctx.gpu,
                    self.hc_post_k,
                    moe_out,
                    hc_streams_i,
                    post_i,
                    comb_i,
                    hc_streams_i,
                    1,
                    h as u32,
                    hc_mult,
                    post_shards,
                    stream,
                )?;
            }
        } // end ffn_k2 / per-token dispatch
        if diag_this {
            super::diag_norm(
                ctx.gpu,
                hc_streams,
                h,
                stream,
                &format!("V4-msdecode L{} hc_post-ffn", self.attn_layer_idx),
            );
            super::diag_norm(
                ctx.gpu,
                hc_streams,
                n * (hc_mult as usize) * h,
                stream,
                &format!(
                    "V4-msdecode L{} hc_post-ffn ALL_STREAMS",
                    self.attn_layer_idx
                ),
            );
        }

        if is_last_layer && let Some(ref head) = hc.head {
            ops::hc_head(
                ctx.gpu,
                self.hc_head_k,
                hc_streams,
                head.hc_fn,
                head.hc_scale,
                head.hc_base,
                c.hidden,
                n as u32,
                h as u32,
                hc_mult,
                eps,
                hc.hc_eps,
                stream,
            )?;
            if diag_this {
                super::diag_norm(
                    ctx.gpu,
                    c.hidden,
                    n * h,
                    stream,
                    &format!("V4-msdecode L{} hc_head", self.attn_layer_idx),
                );
            }
        } else if is_last_layer {
            tracing::warn!(
                "V4-msdecode L{}: hc_head SKIPPED (no head weights)",
                self.attn_layer_idx
            );
        }

        if let (Some(a), Some(f0)) = (attn_us, t_ffn) {
            ctx.gpu.synchronize(stream)?;
            tracing::info!(
                "MSPROF L{}: attn={a}µs ffn={}µs",
                self.attn_layer_idx,
                f0.elapsed().as_micros(),
            );
        }

        Ok(())
    }
}
