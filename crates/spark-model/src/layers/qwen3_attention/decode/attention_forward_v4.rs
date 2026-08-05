// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4-Flash decode path. Reuses low-rank Q projection (wq_a→norm→wq_b)
//! from the MLA path, but uses direct KV projection (no absorption) and
//! grouped low-rank O projection (wo_a→wo_b).

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    /// Run the DeepSeek-V4-Flash decode chain. Returns the O-projection
    /// output (`ctx.buffers.qkv_output()`).
    ///
    /// Visibility is widened to the whole `qwen3_attention` module so the
    /// multi-sequence batched-decode path (`trait_impl::multi_seq::mla`)
    /// can drive this exact single-token chain once per verify token —
    /// the V4-Flash direct-KV algorithm is the SSOT here, NOT the absorbed
    /// MLA chain used by Mistral-Small-4.
    pub(in crate::layers::qwen3_attention) fn attention_forward_v4(
        &self,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        args: &super::attention_forward_mla::DecodeMlaArgs,
    ) -> Result<DevicePtr> {
        let super::attention_forward_mla::DecodeMlaArgs {
            normed,
            q_out,
            k_out,
            v_out,
            q_dim,
            h,
            nq,
            hd,
            eps,
            bs,
            stream,
            pos,
            skip_qkv,
            attn_dest,
        } = *args;
        let mla = self
            .mla
            .as_ref()
            .expect("attention_forward_v4 called without MLA config");
        let meta = ctx
            .attn_metadata
            .expect("V4-Flash decode requires pre-uploaded metadata");

        let q_lora = mla.q_lora_rank as u32;
        let mla_rope = mla.rope as u32;
        let o_lora = mla.o_lora_rank as u32;
        let nkv = ctx.config.num_key_value_heads as u32;
        let profile = ctx.profile;
        let diag_all =
            std::env::var("ATLAS_DIAG_V4_ALL_LAYERS").is_ok_and(|v| v == "1" || v == "true");
        let diag_this = diag_all; // probing is opt-in (ATLAS_DIAG_V4_ALL_LAYERS=1): each probe syncs + reads D2H, real tok/s + TTFT cost
        macro_rules! prof {
            ($label:expr, $body:expr) => {{
                if profile {
                    let _t = std::time::Instant::now();
                    let _r = $body;
                    ctx.gpu.synchronize(stream)?;
                    tracing::info!("    V4 {}: {:.0}µs", $label, _t.elapsed().as_micros());
                    _r
                } else {
                    $body
                }
            }};
        }

        // ── 4b inc-3: decode-time compressed-block append ──
        // Capture this token's compressor input (`normed`, the layer-input
        // RMSNorm output prefill's `cache_skip_v4` feeds `wkv`/`wgate`) into a
        // per-layer BF16 ring, and at each window boundary rerun prefill's
        // compress pipeline over the ring to append ONE FP8 pool block —
        // restoring the double-representation (raw sliding window + compressed
        // history) that inc-2 froze at the prefill count. Runs BEFORE the Q/K/V
        // compute so the MoE scratch buffers (expert_up_out/…) are free, exactly
        // as prefill uses them. Single-sequence eager decode only: `pos` is None
        // on the batched/MTP path, and a captured graph can't re-run host logic.
        //
        // The pipeline body is factored into `v4_compress_append` so the γ-verify
        // catch-up (`v4_compress_catchup`) can replay the exact same append for
        // each committed row post-accept — the decode/verify pool asymmetry fix:
        // plain decode advances this pool via `pos:Some` here, but the batched
        // verify path passes `pos:None`, so without the catch-up the compressed
        // arm would freeze during speculative decode and diverge from greedy.
        if let Some(pos) = pos
            && mla.compressor.is_some()
            && meta.num_seqs == 1
            && !ctx.graph_capture
        {
            self.v4_compress_append(ctx, normed, pos, eps, stream)?;
        }

        // ── Step 1: Q latent → norm → expand ──
        let q_latent = ctx.buffers.ssm_ba();
        let kv_dim = nkv * hd;
        // Batched-verify seam: the caller precomputed q_out (post-q_b_norm),
        // k_out (post-kv_norm) and v_out for this row in one weight-amortized
        // pass over all verify rows — skip straight to RoPE (Step 3).
        // `q_latent` (ssm_ba) stays bound: Step 3 reuses it as k_rope_tmp.
        if !skip_qkv {
            prof!("wq_a", {
                if let Some(ref wqa_nvfp4) = mla.wq_a_nvfp4 {
                    ops::w4a16_gemv(
                        ctx.gpu,
                        self.w4a16_gemv_k,
                        normed,
                        wqa_nvfp4,
                        q_latent,
                        q_lora,
                        h,
                        stream,
                    )
                } else if let Some(ref wqa_fp8) = mla.wq_a_fp8 {
                    // Native block-scaled FP8 GEMV — half the weight traffic of BF16,
                    // lossless (in-kernel F32 dequant).
                    ops::w8a16_gemv(
                        ctx.gpu,
                        self.w8a16_gemv_k,
                        normed,
                        wqa_fp8.weight,
                        wqa_fp8.row_scale,
                        q_latent,
                        q_lora,
                        h,
                        stream,
                    )
                } else {
                    ops::dense_gemv(
                        ctx.gpu,
                        self.dense_gemv_k,
                        normed,
                        &mla.wq_a,
                        q_latent,
                        q_lora,
                        h,
                        stream,
                    )
                }
            })?;
            prof!("q_norm", {
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_w_k,
                    q_latent,
                    &mla.q_a_norm,
                    q_latent,
                    1,
                    q_lora,
                    eps,
                    stream,
                )
            })?;
            prof!("wq_b", {
                if let Some(ref wqb_nvfp4) = mla.wq_b_nvfp4 {
                    ops::w4a16_gemv(
                        ctx.gpu,
                        self.w4a16_gemv_k,
                        q_latent,
                        wqb_nvfp4,
                        q_out,
                        q_dim,
                        q_lora,
                        stream,
                    )
                } else if let Some(ref wqb_fp8) = mla.wq_b_fp8 {
                    ops::w8a16_gemv(
                        ctx.gpu,
                        self.w8a16_gemv_k,
                        q_latent,
                        wqb_fp8.weight,
                        wqb_fp8.row_scale,
                        q_out,
                        q_dim,
                        q_lora,
                        stream,
                    )
                } else {
                    ops::dense_gemv(
                        ctx.gpu,
                        self.dense_gemv_k,
                        q_latent,
                        &mla.wq_b,
                        q_out,
                        q_dim,
                        q_lora,
                        stream,
                    )
                }
            })?;
            // q_b_norm: per-head unweighted RMSNorm over head_dim (DeepSeek-V4).
            // Reference (DeepseekV4UnweightedRMSNorm) renormalizes each of nq heads'
            // hd-dim Q vector to unit RMS BEFORE rope. Missing this makes Q ~sqrt(hd)x
            // too small → near-flat softmax → incoherent output. Weight = all-ones.
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_k,
                q_out,
                &crate::weight_map::DenseWeight {
                    weight: ctx.buffers.norm_unit_w(),
                },
                q_out,
                nq,
                hd,
                eps,
                stream,
            )?;
            if diag_this {
                super::super::trait_impl::diag_norm(
                    ctx.gpu,
                    q_out,
                    q_dim as usize,
                    stream,
                    &format!("V4-decode L{} Q after q_b_norm", self.attn_layer_idx),
                );
            }

            // ── Step 2: Direct KV projection ──
            prof!("wkv", {
                if let Some(ref wkva_nvfp4) = mla.wkv_a_nvfp4 {
                    ops::w4a16_gemv(
                        ctx.gpu,
                        self.w4a16_gemv_k,
                        normed,
                        wkva_nvfp4,
                        k_out,
                        kv_dim,
                        h,
                        stream,
                    )
                } else if let Some(ref wkva_fp8) = mla.wkv_a_fp8 {
                    ops::w8a16_gemv(
                        ctx.gpu,
                        self.w8a16_gemv_k,
                        normed,
                        wkva_fp8.weight,
                        wkva_fp8.row_scale,
                        k_out,
                        kv_dim,
                        h,
                        stream,
                    )
                } else {
                    ops::dense_gemv(
                        ctx.gpu,
                        self.dense_gemv_k,
                        normed,
                        &mla.wkv_a,
                        k_out,
                        kv_dim,
                        h,
                        stream,
                    )
                }
            })?;
            // kv_norm: weighted RMSNorm over the kv latent BEFORE rope (DeepSeek-V4
            // reference: kv = kv_norm(kv_proj(h))). Missing this left K ~8x too large
            // → attention score overflow → NaN. nkv heads × (kv_dim/nkv) each.
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                k_out,
                &mla.kv_a_norm,
                k_out,
                nkv,
                kv_dim / nkv,
                eps,
                stream,
            )?;
            // K=V for V4-Flash direct KV projection
            ctx.gpu
                .copy_d2d_async(k_out, v_out, (kv_dim as usize) * 2, stream)?;
            if diag_this {
                super::super::trait_impl::diag_norm(
                    ctx.gpu,
                    k_out,
                    kv_dim as usize,
                    stream,
                    &format!("V4-decode L{} K after proj", self.attn_layer_idx),
                );
                super::super::trait_impl::diag_norm(
                    ctx.gpu,
                    v_out,
                    kv_dim as usize,
                    stream,
                    &format!("V4-decode L{} V after copy", self.attn_layer_idx),
                );
            }
        } // end !skip_qkv (Steps 1-2)

        // ── Step 3: RoPE for Q and K ──
        // V4-Flash: rope dims are at offset `nope` per head (matching MLA layout),
        // not at the beginning. Extract → RoPE → writeback.
        let q_rope_tmp = ctx.buffers.ssm_conv_out_f32();
        let k_rope_tmp = q_latent; // reuse after wq_b is done
        prof!("rope_extract", {
            ops::mla_q_rope_extract_batched(
                ctx.gpu,
                self.mla_q_rope_extract_batched_k,
                q_out,
                q_rope_tmp,
                1,
                nq,
                hd,
                mla.nope as u32,
                mla_rope,
                nq * hd,
                stream,
            )
        })?;
        // Extract K's rope channels too (MQA: 1 kv head, stride hd). The decode
        // path previously skipped this — `k_rope_tmp` (= reused q_latent) held
        // stale data, so `rope_yarn` rotated garbage and the cached keys got
        // near-zero positional signal → attention degenerates after a few decode
        // tokens. Mirrors the prefill K extract (cache_skip_v4.rs:304).
        prof!("k_rope_extract", {
            ops::mla_q_rope_extract_batched(
                ctx.gpu,
                self.mla_q_rope_extract_batched_k,
                k_out,
                k_rope_tmp,
                1,
                1,
                hd,
                mla.nope as u32,
                mla_rope,
                hd,
                stream,
            )
        })?;
        prof!("rope", {
            ops::rope_yarn(
                ctx.gpu,
                // DeepSeek-V4 uses INTERLEAVED RoPE (rope_interleave=True): adjacent
                // channel pairs (2i, 2i+1), matching the HF reference's rotate_half
                // over cos.repeat_interleave(2). The non-interleaved (NeoX, i/i+half)
                // kernel scrambles positions -> incoherent output.
                self.rope_yarn_interleaved_k,
                q_rope_tmp,
                k_rope_tmp,
                meta.positions,
                1,
                nq,
                1,
                mla_rope,
                mla_rope,
                // Sliding layers (compressor==None) = reference "main" rope:
                // plain θ=10000, mscale=1 (no yarn). CSA/HCA keep θ=160000 yarn.
                if mla.compressor.is_none() {
                    mla.main_inv_freq
                } else {
                    mla.yarn_inv_freq
                },
                if mla.compressor.is_none() {
                    1.0f32
                } else {
                    super::super::helpers::yarn_rope_mscale(ctx.config)
                },
                stream,
            )
        })?;
        prof!("rope_writeback", {
            ops::mla_q_rope_writeback_batched(
                ctx.gpu,
                self.mla_q_rope_writeback_batched_k,
                q_rope_tmp,
                q_out,
                1,
                nq,
                hd,
                mla.nope as u32,
                mla_rope,
                nq * hd,
                stream,
            )
        })?;
        prof!("k_rope_writeback", {
            ops::mla_q_rope_writeback_batched(
                ctx.gpu,
                self.mla_q_rope_writeback_batched_k,
                k_rope_tmp,
                k_out,
                1,
                1,
                hd,
                mla.nope as u32,
                mla_rope,
                hd,
                stream,
            )
        })?;
        if diag_this {
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                k_out,
                kv_dim as usize,
                stream,
                &format!("V4-decode L{} K after RoPE", self.attn_layer_idx),
            );
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                k_out.offset(mla.nope * 2),
                (kv_dim - mla.nope as u32) as usize,
                stream,
                &format!("V4-decode L{} K rope after RoPE", self.attn_layer_idx),
            );
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                q_out.offset(mla.nope * 2),
                (hd - mla.nope as u32) as usize,
                stream,
                &format!("V4-decode L{} Q rope after RoPE", self.attn_layer_idx),
            );
        }

        // ── Step 3.5: Assemble KV cache (V4-Flash: requires latent+rope assembly) ──
        // Cache needs 576-dim (512 latent + 64 rope), but k_out/v_out are 512-dim.
        // Extract RoPE from Q (which has correct [nope|rope] structure) and reuse for K cache.
        let k_cache_assembled = ctx.buffers.ssm_deinterleaved();
        let v_cache_assembled = ctx.buffers.ssm_qkvz();
        let kv_lora = mla.kv_lora_rank as u32;
        let mla_cache_dim = kv_lora + mla_rope;
        prof!("cache_assemble", {
            ops::mla_cache_assemble_batched(
                ctx.gpu,
                self.mla_cache_assemble_batched_k,
                v_out,      // 512-dim latent K (unmodified copy before RoPE writeback)
                k_rope_tmp, // 64-dim RoPE from K
                k_cache_assembled,
                v_cache_assembled,
                1,
                kv_lora,
                mla_rope,
                mla_cache_dim,
                stream,
            )
        })?;

        // ── Step 4: Write assembled K/V to paged cache ──
        prof!("write_kv_cache", {
            self.write_kv_cache(
                ctx.gpu,
                k_cache_assembled,
                v_cache_assembled,
                kv_cache,
                meta.slot,
                1,
                1,
                mla_cache_dim,
                bs as u32,
                mla_cache_dim,
                mla_cache_dim,
                stream,
                ctx.graph_capture,
            )
        })?;

        // ── Step 5: Paged decode attention ──
        // Batched-verify seam: land the attention output directly in the
        // caller's row slot so the batched wo pass reads contiguous rows.
        let attn_out = attn_dest.unwrap_or_else(|| ctx.buffers.attn_output());
        let inv_sqrt_d = self.effective_attn_scale(hd);
        prof!("paged_attn", {
            self.run_paged_decode(
                ctx.gpu,
                q_out,
                kv_cache,
                attn_out,
                meta.block_table,
                meta.seq_len,
                meta.max_blocks_per_seq,
                1,
                nq,
                nkv,
                hd,
                bs as u32,
                inv_sqrt_d,
                nq * hd,
                ctx.buffers.splitk_workspace(),
                stream,
            )
        })?;
        if diag_this {
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                attn_out,
                (nq * hd) as usize,
                stream,
                &format!("V4-decode L{} attn_out", self.attn_layer_idx),
            );
        }

        // ── Step 5.5: Attention-output de-rotation (DeepSeek-V4 eq.26) ──
        // The reference de-rotates the attention output by the query position
        // (apply_rotary(attn_out, cos, -sin)) so each value's contribution is
        // relative-distance. Since V==K carries the rotated rope in its trailing
        // `mla_rope` dims, undo that rotation on the output before o_proj. Reuse
        // the Q rope extract/writeback with the conjugate (negated-sin) kernel.
        {
            let o_rope_tmp = ctx.buffers.ssm_conv_out_f32();
            ops::mla_q_rope_extract_batched(
                ctx.gpu,
                self.mla_q_rope_extract_batched_k,
                attn_out,
                o_rope_tmp,
                1,
                nq,
                hd,
                mla.nope as u32,
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
                1,
                nq,
                0, // no KV heads — de-rotate the query/output heads only
                mla_rope,
                mla_rope,
                // MUST match the Q/K rope inv_freq for this layer type (rope-in
                // == de-rotate-out), else the output is scrambled. Sliding =
                // main θ=10000; CSA/HCA = θ=160000 yarn.
                if mla.compressor.is_none() {
                    mla.main_inv_freq
                } else {
                    mla.yarn_inv_freq
                },
                if mla.compressor.is_none() {
                    1.0f32
                } else {
                    super::super::helpers::yarn_rope_mscale(ctx.config)
                },
                stream,
            )?;
            ops::mla_q_rope_writeback_batched(
                ctx.gpu,
                self.mla_q_rope_writeback_batched_k,
                o_rope_tmp,
                attn_out,
                1,
                nq,
                hd,
                mla.nope as u32,
                mla_rope,
                nq * hd,
                stream,
            )?;
        }
        if diag_this {
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                attn_out,
                (nq * hd) as usize,
                stream,
                &format!("V4-decode L{} attn_out derot", self.attn_layer_idx),
            );
        }

        // Batched-verify seam: the caller runs the O projection itself as a
        // weight-amortized batched pass over all rows' `attn_dest` slots.
        if attn_dest.is_some() {
            return Ok(attn_out);
        }

        // ── Step 6: Grouped low-rank O projection (wo_a → wo_b) ──
        // wo_a is BLOCK-DIAGONAL (DeepseekV4GroupedLinear): the n_heads*head_dim
        // attention output is split into `o_groups` independent groups, each
        // projected group_in -> o_lora. Weight layout [o_groups*o_lora, group_in].
        // A single dense GEMV would mix across groups and (with o_lora<latent_dim)
        // read only 1/o_groups of wo_b — producing garbage every layer.
        let o_groups = ctx.config.o_groups.max(1) as u32;
        let group_in = (nq * hd) / o_groups; // 4096 = (64*512)/8
        let latent_dim = o_groups * o_lora; // 8192 = 8*1024
        let o_latent = ctx.buffers.o_latent();
        let o_out = ctx.buffers.qkv_output();
        prof!("wo_a_grouped", {
            for g in 0..o_groups {
                let in_g = attn_out.offset((g * group_in) as usize * 2);
                let out_g = o_latent.offset((g * o_lora) as usize * 2);
                if let Some(ref woa4) = mla.wo_a_nvfp4 {
                    // NVFP4 per group: packed rows [g*o_lora ..) at 0.5 B/elem,
                    // block scales [N, K/16] row-major (1 B/scale), shared
                    // per-tensor scale2 (quantized as ONE tensor).
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
                            woa4.weight_scale_2_vec
                                .offset((g as usize) * (o_lora as usize) * 4)
                        },
                    };
                    ops::w4a16_gemv(
                        ctx.gpu,
                        self.w4a16_gemv_k,
                        in_g,
                        &sub,
                        out_g,
                        o_lora,
                        group_in,
                        stream,
                    )?;
                } else if let Some(ref woa_fp8) = mla.wo_a_fp8 {
                    // Native block-scaled FP8 per group (block-diagonal):
                    // weight rows [g*o_lora:(g+1)*o_lora] (fp8, 1 byte/elem) and the
                    // matching [o_lora/128, group_in/128] block-scale sub-tile.
                    let w_off = (g as usize) * (o_lora as usize) * (group_in as usize); // fp8 bytes
                    let s_off =
                        (g as usize) * (o_lora as usize / 128) * (group_in as usize / 128) * 4; // FP32 block-scale bytes
                    ops::w8a16_gemv(
                        ctx.gpu,
                        self.w8a16_gemv_k,
                        in_g,
                        woa_fp8.weight.offset(w_off),
                        woa_fp8.row_scale.offset(s_off),
                        out_g,
                        o_lora,
                        group_in,
                        stream,
                    )?;
                } else {
                    let w_g = crate::weight_map::DenseWeight {
                        weight: mla
                            .wo_a
                            .weight
                            .offset((g as usize) * (o_lora as usize) * (group_in as usize) * 2),
                    };
                    ops::dense_gemv(
                        ctx.gpu,
                        self.dense_gemv_k,
                        in_g,
                        &w_g,
                        out_g,
                        o_lora,
                        group_in,
                        stream,
                    )?;
                }
            }
            Ok::<(), anyhow::Error>(())
        })?;
        prof!("wo_b", {
            if let Some(ref wob4) = mla.wo_b_nvfp4 {
                ops::w4a16_gemv(
                    ctx.gpu,
                    self.w4a16_gemv_k,
                    o_latent,
                    wob4,
                    o_out,
                    h,
                    latent_dim,
                    stream,
                )
            } else if let Some(ref wob_fp8) = mla.wo_b_fp8 {
                ops::w8a16_gemv(
                    ctx.gpu,
                    self.w8a16_gemv_k,
                    o_latent,
                    wob_fp8.weight,
                    wob_fp8.row_scale,
                    o_out,
                    h,
                    latent_dim,
                    stream,
                )
            } else {
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    o_latent,
                    &mla.wo_b,
                    o_out,
                    h,
                    latent_dim,
                    stream,
                )
            }
        })?;
        if diag_this {
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                o_out,
                h as usize,
                stream,
                &format!("V4-decode L{} o_out", self.attn_layer_idx),
            );
        }

        Ok(o_out)
    }

    /// Append ONE compressed-KV pool block for the window that closes at
    /// absolute position `pos`, replaying prefill's compress pipeline over the
    /// per-layer normed-x ring. Extracted VERBATIM from the decode-time append
    /// so the γ-verify catch-up (`v4_compress_catchup`) can advance the pool
    /// with byte-identical math for each accepted verify row — closing the
    /// decode/verify compressed-pool asymmetry (see `ms_mla_decode_v4_flash`,
    /// which passes `pos:None` and therefore froze this pool during spec decode).
    ///
    /// `normed` is this row's compressor input (the layer-input RMSNorm output).
    /// No-op when the window at `pos` does not close, is already filled, or the
    /// layer has no compressor. Uses the MoE scratch buffers (idle post-forward).
    pub(in crate::layers::qwen3_attention) fn v4_compress_append(
        &self,
        ctx: &ForwardContext,
        normed: DevicePtr,
        pos: u32,
        eps: f32,
        stream: u64,
    ) -> Result<()> {
        let mla = self.mla.as_ref();
        let mla = match mla {
            Some(m) => m,
            None => return Ok(()),
        };
        let comp = match mla.compressor.as_ref() {
            Some(c) => c,
            None => return Ok(()),
        };
        let mla_rope = mla.rope as u32;
        let h = ctx.config.hidden_size as u32;
        {
            use std::sync::atomic::Ordering::Relaxed;
            let ratio = comp.ratio as u32;
            let proj_dim = comp.proj_dim as u32;
            let nope = mla.nope as u32;
            let rope_d = mla_rope;
            let hd_mla = nope + rope_d; // compressed block width (= q head_dim)
            let hb = h as usize * 2; // BF16 bytes per token row

            // Capture normed → ring slot (pos % ratio). At a boundary the
            // ring then holds the completed window's `ratio` tokens in order.
            let slot = (pos % ratio) as usize;
            ctx.gpu
                .copy_d2d_async(normed, comp.ring.offset(slot * hb), hb, stream)?;

            if (pos + 1) % ratio == 0 {
                let w = (pos + 1) / ratio - 1;
                let filled = self.v4_comp_pool_filled.load(Relaxed);
                // Append the next window we don't already hold. Straddle
                // windows are handled by the prefill→decode ring seed
                // (cache_skip_v4), so the ring is always complete here — no
                // coverage guard needed. prev_win is seeded (CSA) so the very
                // first decode block gets its real Ca (not masked).
                if w >= filled {
                    use spark_runtime::kernel_args::KernelLaunch;
                    let prev_valid = self.v4_comp_prev_valid.load(Relaxed);
                    // CSA with a real previous window → 2×ratio overlap
                    // (grid[2], take block 1). HCA, or the first CSA window
                    // (Ca masked = window-0 semantics), → ring only (grid[1],
                    // block 0). See csa_compress.cu for the Ca/Cb layout.
                    let (comp_in, t_rows, launch_win, tgt) = if comp.is_csa && prev_valid {
                        ctx.gpu.copy_d2d_async(
                            comp.prev_win,
                            comp.stage,
                            ratio as usize * hb,
                            stream,
                        )?;
                        ctx.gpu.copy_d2d_async(
                            comp.ring,
                            comp.stage.offset(ratio as usize * hb),
                            ratio as usize * hb,
                            stream,
                        )?;
                        (comp.stage, 2 * ratio, 2u32, 1u32)
                    } else {
                        (comp.ring, ratio, 1u32, 0u32)
                    };

                    // compressor projections kv/gate = W·comp_in [T, proj_dim]
                    let kv_comp = ctx.buffers.expert_up_out();
                    let gate_comp = ctx.buffers.expert_down_out();
                    ops::dense_gemm(
                        ctx.gpu,
                        self.dense_gemm_k,
                        comp_in,
                        &comp.wkv,
                        kv_comp,
                        t_rows,
                        proj_dim,
                        h,
                        stream,
                    )?;
                    ops::dense_gemm(
                        ctx.gpu,
                        self.dense_gemm_k,
                        comp_in,
                        &comp.wgate,
                        gate_comp,
                        t_rows,
                        proj_dim,
                        h,
                        stream,
                    )?;
                    // window softmax-gated compression → [launch_win, hd_mla]
                    let compressed = ctx.buffers.moe_output();
                    KernelLaunch::new(ctx.gpu, self.csa_compress_k)
                        .grid([launch_win, 1, 1])
                        .block([256, 1, 1])
                        .arg_ptr(kv_comp)
                        .arg_ptr(gate_comp)
                        .arg_ptr(comp.ape)
                        .arg_ptr(compressed)
                        .arg_u32(t_rows)
                        .arg_u32(ratio)
                        .arg_u32(hd_mla)
                        .arg_u32(proj_dim)
                        .arg_u32(if comp.is_csa { 1 } else { 0 })
                        .launch(stream)?;
                    // rms_norm the target block in place (matches prefill).
                    let block = compressed.offset(tgt as usize * hd_mla as usize * 2);
                    ops::rms_norm(
                        ctx.gpu,
                        self.rms_norm_w_k,
                        block,
                        &comp.norm,
                        block,
                        1,
                        hd_mla,
                        eps,
                        stream,
                    )?;
                    // comp_k = rope(block): copy → extract tail → yarn @ w*ratio
                    // → writeback. Uses the window's compress position w*ratio,
                    // theta = yarn_inv_freq, interleaved — mirrors prefill.
                    let comp_k = compressed.offset(launch_win as usize * hd_mla as usize * 2);
                    ctx.gpu
                        .copy_d2d_async(block, comp_k, hd_mla as usize * 2, stream)?;
                    let pos_bytes = (w * ratio).to_le_bytes();
                    let comp_positions = ctx.buffers.ssm_ba();
                    ctx.gpu.copy_h2d_async(&pos_bytes, comp_positions, stream)?;
                    let comp_rope_tmp = ctx.buffers.ssm_conv_out_f32();
                    ops::mla_q_rope_extract_batched(
                        ctx.gpu,
                        self.mla_q_rope_extract_batched_k,
                        comp_k,
                        comp_rope_tmp,
                        1,
                        1,
                        hd_mla,
                        nope,
                        rope_d,
                        hd_mla,
                        stream,
                    )?;
                    ops::rope_yarn(
                        ctx.gpu,
                        self.rope_yarn_interleaved_k,
                        comp_rope_tmp,
                        comp_rope_tmp,
                        comp_positions,
                        1,
                        0,
                        1,
                        rope_d,
                        rope_d,
                        mla.yarn_inv_freq,
                        super::super::helpers::yarn_rope_mscale(ctx.config),
                        stream,
                    )?;
                    ops::mla_q_rope_writeback_batched(
                        ctx.gpu,
                        self.mla_q_rope_writeback_batched_k,
                        comp_rope_tmp,
                        comp_k,
                        1,
                        1,
                        hd_mla,
                        nope,
                        rope_d,
                        hd_mla,
                        stream,
                    )?;
                    // Quantize the rope'd block into pool[w] (FP8, 1 byte/elem,
                    // k_scale=1.0 → plain e4m3 cast, matches the raw KV arm).
                    ops::bf16_to_fp8(
                        ctx.gpu,
                        self.bf16_to_fp8_k,
                        comp_k,
                        comp.pool.offset(w as usize * hd_mla as usize),
                        hd_mla,
                        stream,
                    )?;
                    // Publish: decode's compressed arm now attends [0, w+1).
                    self.v4_comp_pool_filled.store(w + 1, Relaxed);
                    // Mirror to the device word the graphed kernels read at replay,
                    // on THIS stream so it is ordered before the next kernel that
                    // reads it (the fix for the graph-baked count freeze).
                    if !self.v4_comp_count_dev.is_null() {
                        ctx.gpu
                            .memset_u32_async(self.v4_comp_count_dev, w + 1, 1, stream)?;
                    }
                    if self.attn_layer_idx == 0
                        && std::env::var("ATLAS_DSPARK_CATCHUP_DIAG").is_ok()
                    {
                        tracing::info!(
                            "DSPARK APPEND: layer0 pos={pos} → pool_filled={}",
                            w + 1
                        );
                    }
                    // CSA: this window becomes the next window's Ca source.
                    if comp.is_csa {
                        ctx.gpu.copy_d2d_async(
                            comp.ring,
                            comp.prev_win,
                            ratio as usize * hb,
                            stream,
                        )?;
                        self.v4_comp_prev_valid.store(true, Relaxed);
                    }
                }
            }
        }
        Ok(())
    }

    /// γ-verify catch-up: replay `v4_compress_append` for each committed verify
    /// row, in absolute-position order, from the per-layer `verify_comp_normed`
    /// capture (armed only when this layer owns a compressor). Row `r` sits at
    /// absolute position `pre_len + r`. Called AFTER the accept walk so the pool
    /// advances for exactly the committed positions — restoring the double
    /// representation that the `pos:None` batched verify path skips. No-op when
    /// unarmed or the layer has no compressor. Eager only (never under capture).
    pub(crate) fn v4_compress_catchup(
        &self,
        ctx: &ForwardContext,
        pre_len: usize,
        num_committed: usize,
        eps: f32,
        stream: u64,
    ) -> Result<()> {
        if self.verify_comp_normed.is_null()
            || self
                .mla
                .as_ref()
                .and_then(|m| m.compressor.as_ref())
                .is_none()
        {
            return Ok(());
        }
        let hb = ctx.config.hidden_size * 2;
        let diag = std::env::var("ATLAS_DSPARK_CATCHUP_DIAG").is_ok();
        if diag && self.mla.as_ref().and_then(|m| m.compressor.as_ref()).is_some() {
            // Log the first compressor layer only, every step, to watch the pool grow.
            static FIRST: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(usize::MAX);
            let _ = FIRST.compare_exchange(usize::MAX, self.attn_layer_idx, std::sync::atomic::Ordering::Relaxed, std::sync::atomic::Ordering::Relaxed);
            if FIRST.load(std::sync::atomic::Ordering::Relaxed) == self.attn_layer_idx {
                tracing::info!(
                    "DSPARK CATCHUP L{}: pre_len={} committed={} pool_before={}",
                    self.attn_layer_idx, pre_len, num_committed,
                    self.v4_comp_pool_filled.load(std::sync::atomic::Ordering::Relaxed),
                );
            }
        }
        for r in 0..num_committed {
            let normed = self.verify_comp_normed.offset(r * hb);
            self.v4_compress_append(ctx, normed, (pre_len + r) as u32, eps, stream)?;
        }
        Ok(())
    }

    /// Report — once per distinct reason for the whole process — why
    /// [`Self::v4_compress_speculate`] declined to advance the compressor.
    ///
    /// Every bail path leaves the engine bit-identical to the pre-fix
    /// behaviour, so a skipped speculation is invisible in throughput: it looks
    /// exactly like "the fix ran and bought nothing". Warn at `warn!` (not
    /// `debug!`) because a skip is a silent correctness regression in the γ
    /// verify, and dedupe so a 43-layer × 40-step run does not emit thousands
    /// of identical lines.
    fn spec_skip_reason(&self, reason: &str) {
        use std::collections::HashSet;
        use std::sync::{Mutex, OnceLock};
        static SEEN: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
        let seen = SEEN.get_or_init(|| Mutex::new(HashSet::new()));
        let first = match seen.lock() {
            Ok(mut g) => g.insert(reason.to_string()),
            // A poisoned lock means another thread panicked mid-insert; log
            // rather than swallow, since dropping the message is the failure
            // mode this helper exists to prevent.
            Err(_) => true,
        };
        if first {
            tracing::warn!(
                "[v4-spec] compressor speculation SKIPPED (layer {} filled={}): {}",
                self.attn_layer_idx,
                self.v4_comp_pool_filled
                    .load(std::sync::atomic::Ordering::Relaxed),
                reason,
            );
        }
    }

    /// γ-verify SPECULATIVE compressor advance — run BEFORE the verify
    /// attention, over ALL `n_rows` draft rows.
    ///
    /// # Why this exists
    ///
    /// The compressed arm's causality rule is that a query at absolute position
    /// `p` may attend `(p+1)/ratio` compressed blocks, and every writer of
    /// `v4_comp_pool_filled` upholds `filled == seq_len/ratio` (prefill stores
    /// `n/ratio`; `v4_compress_append` stores `(pos+1)/ratio` and runs BEFORE
    /// attention on the plain-decode path). The batched verify path took
    /// `pos: None` and never appended, so the pool sat at `pre_len/ratio` while
    /// row `r` needed `(pre_len+r+1)/ratio` — at ratio 4 with `pre_len=15`, the
    /// six rows want 4,4,4,4,5,5 blocks and the pool holds **3**. Every row
    /// therefore attended a different (smaller) compressed history than plain
    /// decode would, its logits diverged, and prefix-accept truncated — the
    /// measured `accepted=1.33` against `2.18` with the arm disabled entirely.
    ///
    /// Advancing the pool over the drafts closes that gap, at the cost of
    /// putting speculative state into an append-only structure — hence the
    /// frontier snapshot here and [`Self::v4_compress_restore`] afterwards.
    /// This is ds4's `spec_frontier_snapshot` (ds4.c:30585) in Atlas terms.
    ///
    /// Cheap-exit: if no window boundary falls inside `[base_pos, base_pos+n)`
    /// nothing can be emitted, so neither the snapshot nor the appends run.
    /// That is the common case on the ratio-128 HCA layers (a 6-wide window
    /// crosses a boundary ~5% of steps) and never the case on ratio-4 CSA.
    pub(crate) fn v4_compress_speculate(
        &self,
        ctx: &ForwardContext,
        base_pos: usize,
        n_rows: usize,
        eps: f32,
        stream: u64,
    ) -> Result<()> {
        use std::sync::atomic::Ordering::Relaxed;
        // Self-heal: a previous speculation that never got rolled back would
        // otherwise be snapshotted AS the baseline here, making the corruption
        // permanent instead of one-step. The post-accept path always calls
        // `v4_compress_restore`, so this should never fire — but the failure
        // mode is silent divergence, which is exactly what we are trying to
        // eliminate, so it is not left to trust.
        if self.spec_rows.load(Relaxed) != 0 {
            self.v4_compress_restore(ctx, stream)?;
        }
        // Every early return below leaves behaviour bit-identical to the
        // pre-fix engine, so "the fix ran and did not help" and "the fix never
        // ran" are indistinguishable from throughput alone. Name the reason.
        macro_rules! bail_spec {
            ($reason:expr) => {{
                self.spec_skip_reason($reason);
                return Ok(());
            }};
        }
        if self.verify_comp_normed.is_null() {
            bail_spec!("verify_comp_normed unarmed (non-compressor layer)")
        }
        if n_rows == 0 {
            bail_spec!("n_rows == 0")
        }
        let comp = match self.mla.as_ref().and_then(|m| m.compressor.as_ref()) {
            Some(c) => c,
            None => bail_spec!("layer has no compressor"),
        };
        // A captured graph cannot re-run this host-side logic, so speculation
        // would silently not happen and the asymmetry would return.
        if ctx.graph_capture {
            bail_spec!(
                "under graph capture — host-side compressor logic cannot run inside a \
                 captured verify, so the compressed arm keeps the pre-fix shared block count"
            )
        }
        let ratio = comp.ratio;
        let n_rows = n_rows.min(super::super::MAX_VERIFY_ROWS);
        // No boundary inside the window → no emit → no state change to undo.
        if (base_pos + n_rows) / ratio == base_pos / ratio {
            bail_spec!("no window boundary inside the γ span (nothing would be emitted)")
        }

        let hb = ctx.config.hidden_size * 2;
        // Snapshot the frontiers. Only the ring slots this window can clobber
        // are saved: slot `(base_pos+j) % ratio` for j in [0, min(n_rows,
        // ratio)). Beyond `ratio` slots the map repeats, and since every save
        // happens before any write the repeats would copy identical bytes.
        let snap_slots = n_rows.min(ratio);
        for j in 0..snap_slots {
            let slot = (base_pos + j) % ratio;
            ctx.gpu.copy_d2d_async(
                comp.ring.offset(slot * hb),
                comp.ring_snap.offset(j * hb),
                hb,
                stream,
            )?;
        }
        if comp.is_csa && !comp.prev_win_snap.is_null() {
            ctx.gpu
                .copy_d2d_async(comp.prev_win, comp.prev_win_snap, ratio * hb, stream)?;
        }
        self.spec_saved_filled
            .store(self.v4_comp_pool_filled.load(Relaxed), Relaxed);
        self.spec_saved_prev_valid
            .store(self.v4_comp_prev_valid.load(Relaxed), Relaxed);
        self.spec_base_pos.store(base_pos as u32, Relaxed);
        self.spec_rows.store(n_rows as u32, Relaxed);

        // Advance over every draft row, in absolute-position order — the same
        // call plain decode makes, so the blocks are byte-identical to the ones
        // a non-speculative decode would have produced at those positions.
        for r in 0..n_rows {
            let normed = self.verify_comp_normed.offset(r * hb);
            self.v4_compress_append(ctx, normed, (base_pos + r) as u32, eps, stream)?;
        }
        if std::env::var("ATLAS_DSPARK_CATCHUP_DIAG").is_ok() {
            tracing::info!(
                "[v4-spec] layer {} base={} rows={} ratio={} filled {} -> {}",
                self.attn_layer_idx,
                base_pos,
                n_rows,
                ratio,
                self.spec_saved_filled.load(Relaxed),
                self.v4_comp_pool_filled.load(Relaxed),
            );
        }
        Ok(())
    }

    /// γ-verify frontier rollback (ds4's `spec_frontier_restore`, ds4.c:30620).
    /// Rewinds every frontier [`Self::v4_compress_speculate`] moved back to its
    /// pre-speculation value.
    ///
    /// This does NOT advance to the accepted prefix — the caller runs
    /// [`Self::v4_compress_catchup`] straight after, which replays the append
    /// for exactly the committed rows. Rewinding fully and replaying (rather
    /// than trying to "un-append" down to the committed count) leaves the
    /// committed path byte-identical to the pre-speculation catch-up, which is
    /// the behaviour prefix-accept losslessness is defined against, and keeps
    /// one replay implementation instead of two.
    ///
    /// Idempotent: `spec_rows` is swapped to 0, so a second call is a no-op.
    pub(crate) fn v4_compress_restore(
        &self,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        use std::sync::atomic::Ordering::Relaxed;
        let rows = self.spec_rows.swap(0, Relaxed) as usize;
        if rows == 0 {
            return Ok(());
        }
        let comp = match self.mla.as_ref().and_then(|m| m.compressor.as_ref()) {
            Some(c) => c,
            None => return Ok(()),
        };
        let ratio = comp.ratio;
        let base_pos = self.spec_base_pos.load(Relaxed) as usize;
        let hb = ctx.config.hidden_size * 2;

        for j in 0..rows.min(ratio) {
            let slot = (base_pos + j) % ratio;
            ctx.gpu.copy_d2d_async(
                comp.ring_snap.offset(j * hb),
                comp.ring.offset(slot * hb),
                hb,
                stream,
            )?;
        }
        if comp.is_csa && !comp.prev_win_snap.is_null() {
            ctx.gpu
                .copy_d2d_async(comp.prev_win_snap, comp.prev_win, ratio * hb, stream)?;
        }
        let saved = self.spec_saved_filled.load(Relaxed);
        self.v4_comp_pool_filled.store(saved, Relaxed);
        self.v4_comp_prev_valid
            .store(self.spec_saved_prev_valid.load(Relaxed), Relaxed);
        if !self.v4_comp_count_dev.is_null() {
            ctx.gpu
                .memset_u32_async(self.v4_comp_count_dev, saved, 1, stream)?;
        }
        Ok(())
    }
}
