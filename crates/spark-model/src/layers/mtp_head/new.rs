// SPDX-License-Identifier: AGPL-3.0-only

//! MTP head constructor.

use anyhow::Result;
use parking_lot::Mutex;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};

use super::{
    MTP_FP8_WARMUP_TOKENS, MtpHead, MtpQuantization, ProjectionWeight, mtp_fp8_calib_enabled,
};
use crate::layers::MoeLayer;
use crate::layers::fp8_calibration::Fp8KvCalibration;
use crate::weight_map::{DenseWeight, MoeWeights, MtpWeights, QuantizedWeight, quantize_to_nvfp4};

impl MtpHead {
    pub fn new(
        weights: MtpWeights,
        embed_tokens: DenseWeight,
        lm_head_nvfp4: QuantizedWeight,
        config: &atlas_core::config::ModelConfig,
        gpu: &dyn GpuBackend,
        quant: MtpQuantization,
        mtp_vocab_size: u32,
        max_seq_len: usize,
    ) -> Result<Self> {
        let stream = gpu.default_stream();
        let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
        let nvfp4_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
        let fp8_k = gpu.kernel("gemv_fp8w", "quantize_bf16_to_fp8")?;

        let h = config.hidden_size;
        let nq = config.num_attention_heads;
        let nkv = config.num_key_value_heads;
        let hd = config.head_dim;
        let inter = config.moe_intermediate_size;

        let q = |bf16: &DenseWeight, n: usize, k: usize| -> Result<ProjectionWeight> {
            Self::quantize_proj(bf16, n, k, quant, gpu, absmax_k, nvfp4_k, fp8_k, stream)
        };

        // Quantize projections
        let fc = q(&weights.fc, h, h * 2)?;
        let q_proj = q(&weights.q_proj, nq * hd * 2, h)?;
        let k_proj = q(&weights.k_proj, nkv * hd, h)?;
        let v_proj = q(&weights.v_proj, nkv * hd, h)?;
        let o_proj = q(&weights.o_proj, h, nq * hd)?;

        // MoE: NVFP4 uses fused MoeLayer; FP8/BF16 stores per-expert weights
        let (moe_nvfp4, moe_experts_generic, moe_shared_generic) = match quant {
            MtpQuantization::Nvfp4 => {
                let gate_nvfp4 = quantize_to_nvfp4(
                    &weights.moe_gate,
                    config.num_experts,
                    h,
                    gpu,
                    absmax_k,
                    nvfp4_k,
                    stream,
                )?;
                let mut experts = Vec::with_capacity(weights.experts.len());
                for (i, de) in weights.experts.iter().enumerate() {
                    let gate_proj =
                        quantize_to_nvfp4(&de.gate_proj, inter, h, gpu, absmax_k, nvfp4_k, stream)?;
                    let up_proj =
                        quantize_to_nvfp4(&de.up_proj, inter, h, gpu, absmax_k, nvfp4_k, stream)?;
                    let down_proj =
                        quantize_to_nvfp4(&de.down_proj, h, inter, gpu, absmax_k, nvfp4_k, stream)?;
                    experts.push(crate::weight_map::ExpertWeight {
                        gate_proj,
                        up_proj,
                        down_proj,
                    });
                    if (i + 1) % 128 == 0 {
                        tracing::info!(
                            "  MTP experts quantized: {}/{}",
                            i + 1,
                            weights.experts.len()
                        );
                    }
                }
                let shared_gate = quantize_to_nvfp4(
                    &weights.shared_expert.gate_proj,
                    inter,
                    h,
                    gpu,
                    absmax_k,
                    nvfp4_k,
                    stream,
                )?;
                let shared_up = quantize_to_nvfp4(
                    &weights.shared_expert.up_proj,
                    inter,
                    h,
                    gpu,
                    absmax_k,
                    nvfp4_k,
                    stream,
                )?;
                let shared_down = quantize_to_nvfp4(
                    &weights.shared_expert.down_proj,
                    h,
                    inter,
                    gpu,
                    absmax_k,
                    nvfp4_k,
                    stream,
                )?;
                let moe_weights = MoeWeights {
                    gate: weights.moe_gate,
                    shared_expert: crate::weight_map::ExpertWeight {
                        gate_proj: shared_gate,
                        up_proj: shared_up,
                        down_proj: shared_down,
                    },
                    shared_expert_gate: weights.shared_expert_gate,
                    experts,
                    router_pre_norm: None,
                    correction_bias: None,
                };
                let moe = MoeLayer::new(
                    moe_weights,
                    config.num_experts,
                    Some(gate_nvfp4),
                    gpu,
                    config,
                )?;
                (Some(moe), None, None)
            }
            MtpQuantization::Fp8 | MtpQuantization::Bf16 => {
                let mut experts_g = Vec::with_capacity(weights.experts.len());
                for (i, de) in weights.experts.iter().enumerate() {
                    let gate_proj = q(&de.gate_proj, inter, h)?;
                    let up_proj = q(&de.up_proj, inter, h)?;
                    let down_proj = q(&de.down_proj, h, inter)?;
                    experts_g.push((gate_proj, up_proj, down_proj));
                    if (i + 1) % 128 == 0 {
                        tracing::info!(
                            "  MTP experts quantized: {}/{}",
                            i + 1,
                            weights.experts.len()
                        );
                    }
                }
                let shared = (
                    q(&weights.shared_expert.gate_proj, inter, h)?,
                    q(&weights.shared_expert.up_proj, inter, h)?,
                    q(&weights.shared_expert.down_proj, h, inter)?,
                );
                (None, Some(experts_g), Some(shared))
            }
        };

        // MTP KV cache: 1 attention layer
        let kv_config = KvCacheConfig {
            block_size: 16,
            num_kv_heads: nkv,
            head_dim: hd,
            num_layers: 1,
            dtype: KvCacheDtype::Fp8, // MTP always uses FP8 for now
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
        };
        let mtp_num_blocks = max_seq_len / kv_config.block_size + 1;
        let kv_cache = PagedKvCache::new(kv_config, mtp_num_blocks, gpu)?;

        // Extra kernel handles for BF16/FP8 paths
        let (
            dense_gemv_k,
            dense_gemv_fp8w_k,
            deinterleave_qg_k,
            moe_topk_k,
            moe_silu_mul_k,
            moe_weighted_sum_blend_k,
        ) = match quant {
            MtpQuantization::Nvfp4 => (None, None, None, None, None, None),
            MtpQuantization::Fp8 => (
                // BF16 GEMV needed for gate (always BF16) + generic MoE dispatch
                Some(gpu.kernel("gemv", "dense_gemv_bf16")?),
                Some(gpu.kernel("gemv_fp8w", "dense_gemv_fp8w")?),
                Some(gpu.kernel("ssm_preprocess", "deinterleave_qg")?),
                Some(gpu.kernel("moe_topk", "moe_topk_softmax")?),
                Some(gpu.kernel("moe_silu_mul", "moe_silu_mul")?),
                Some(gpu.kernel("moe_expert_gemv", "moe_weighted_sum_blend")?),
            ),
            MtpQuantization::Bf16 => (
                Some(gpu.kernel("gemv", "dense_gemv_bf16")?),
                None,
                Some(gpu.kernel("ssm_preprocess", "deinterleave_qg")?),
                Some(gpu.kernel("moe_topk", "moe_topk_softmax")?),
                Some(gpu.kernel("moe_silu_mul", "moe_silu_mul")?),
                Some(gpu.kernel("moe_expert_gemv", "moe_weighted_sum_blend")?),
            ),
        };

        let effective_vocab = if mtp_vocab_size > 0 {
            (mtp_vocab_size as usize).min(config.vocab_size)
        } else {
            config.vocab_size
        };
        tracing::info!(
            "MTP head: quant={:?}, fc=[{h},{h2}], attn Q=[{qd},{h}], {ne} experts, \
             vocab={ev}/{fv} (LM head {lm:.1} MB)",
            quant,
            h2 = h * 2,
            qd = nq * hd * 2,
            ne = config.num_experts,
            ev = effective_vocab,
            fv = config.vocab_size,
            lm = (effective_vocab * h / 2) as f64 / (1024.0 * 1024.0),
        );

        Ok(Self {
            pre_fc_norm_embedding: weights.pre_fc_norm_embedding,
            pre_fc_norm_hidden: weights.pre_fc_norm_hidden,
            input_layernorm: weights.input_layernorm,
            post_attn_layernorm: weights.post_attn_layernorm,
            norm: weights.norm,
            fc,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm: weights.q_norm,
            k_norm: weights.k_norm,
            moe_nvfp4,
            moe_experts_generic,
            moe_shared_generic,
            moe_gate: weights.moe_gate,
            shared_expert_gate: weights.shared_expert_gate,
            // MoE constructor — dense MLP fields are unused.
            is_dense_mlp: false,
            dense_mlp_gate: None,
            dense_mlp_up: None,
            dense_mlp_down: None,
            dense_mlp_intermediate: 0,
            quant,
            mtp_vocab_size,
            embed_tokens,
            lm_head_nvfp4,
            kv_cache: Mutex::new(kv_cache),
            attn_layer_idx: 0,
            fp8_calibration: if mtp_fp8_calib_enabled() {
                let cal = Fp8KvCalibration::new(MTP_FP8_WARMUP_TOKENS, gpu)?;
                tracing::info!(
                    "MTP FP8 KV calibration: ENABLED (warmup={} tokens)",
                    MTP_FP8_WARMUP_TOKENS
                );
                Some(cal)
            } else {
                tracing::info!("MTP FP8 KV calibration: disabled (ATLAS_MTP_FP8_CALIB=0)");
                None
            },
            rms_norm_k: gpu.kernel("norm", "rms_norm")?,
            rms_norm_residual_k: if config.use_fp32_residual() {
                gpu.kernel("norm", "rms_norm_residual_f32")
                    .or_else(|_| gpu.kernel("norm", "rms_norm_residual"))?
            } else {
                gpu.kernel("norm", "rms_norm_residual")?
            },
            w4a16_gemv_k: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
            w4a16_gemv_qg_k: gpu.kernel("w4a16_gemv", "w4a16_gemv_qg")?,
            w4a16_gemv_dual_k: gpu.kernel("w4a16_gemv_fused", "w4a16_gemv_dual")?,
            rope_k: gpu.kernel("rope", "rope_forward")?,
            reshape_cache_k: gpu.kernel("reshape_and_cache", "reshape_and_cache_flash_fp8")?,
            paged_decode_k: gpu.kernel("paged_decode_fp8", "paged_decode_attn_fp8")?,
            residual_add_k: if config.use_fp32_residual() {
                gpu.kernel("norm", "f32_residual_add")
                    .or_else(|_| gpu.kernel("residual_add", "bf16_residual_add"))?
            } else {
                gpu.kernel("residual_add", "bf16_residual_add")?
            },
            residual_add_rms_norm_k: if config.use_fp32_residual() {
                gpu.kernel("norm", "residual_add_rms_norm_f32")
                    .or_else(|_| gpu.kernel("norm", "residual_add_rms_norm"))?
            } else {
                gpu.kernel("norm", "residual_add_rms_norm")?
            },
            sigmoid_gate_mul_k: gpu.kernel("residual_add", "sigmoid_gate_mul")?,
            bf16_concat_k: gpu.kernel("residual_add", "bf16_concat")?,
            argmax_k: gpu.kernel("argmax", "argmax_bf16")?,
            embed_from_argmax_k: gpu.kernel("embed_from_argmax", "embed_from_argmax")?,
            draft_token_id_dev: gpu.alloc(4)?,
            dense_gemv_k,
            dense_gemv_fp8w_k,
            w8a16_gemv_k: gpu.kernel("w8a16_gemv", "w8a16_gemv").ok(),
            deinterleave_qg_k,
            moe_topk_k,
            moe_silu_mul_k,
            moe_weighted_sum_blend_k,
        })
    }

    /// Construct an MtpHead for *dense-MLP* MTP (Qwen3.5/3.6 27B-class).
    ///
    /// Mirrors `new()` minus the MoE quantization. Weight precision follows
    /// `quant` (--mtp-quantization) for the attention projections AND the
    /// dense MLP: grafted checkpoint heads (e.g. AEON-Ultimate-MTP) ship
    /// BF16 `mtp.*` tensors whose acceptance rate collapses if re-quantized
    /// to NVFP4 — BF16 preserves them verbatim. Forward pass branches on
    /// `is_dense_mlp` and runs a 4-op MLP (gate → up → silu+mul → down)
    /// in place of the MoE block.
    #[allow(clippy::too_many_arguments)]
    pub fn new_dense(
        weights: crate::weight_map::MtpDenseWeights,
        embed_tokens: DenseWeight,
        lm_head_nvfp4: QuantizedWeight,
        config: &atlas_core::config::ModelConfig,
        gpu: &dyn GpuBackend,
        quant: MtpQuantization,
        mtp_vocab_size: u32,
        max_seq_len: usize,
    ) -> Result<Self> {
        let stream = gpu.default_stream();
        let absmax_k = gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax")?;
        let nvfp4_k = gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4")?;
        let fp8_k = gpu.kernel("gemv_fp8w", "quantize_bf16_to_fp8")?;

        let h = config.hidden_size;
        let nq = config.num_attention_heads;
        let nkv = config.num_key_value_heads;
        let hd = config.head_dim;
        // Dense MTP intermediate is the dense MLP intermediate, NOT the MoE one.
        // Read from the actual weight shape (gate_proj is [intermediate, hidden]).
        let inter = config.intermediate_size;

        let q = |bf16: &DenseWeight, n: usize, k: usize| -> Result<ProjectionWeight> {
            Self::quantize_proj(bf16, n, k, quant, gpu, absmax_k, nvfp4_k, fp8_k, stream)
        };

        // Quantize attention projections (same shapes as MoE constructor).
        let fc = q(&weights.fc, h, h * 2)?;
        let q_proj = q(&weights.q_proj, nq * hd * 2, h)?;
        let k_proj = q(&weights.k_proj, nkv * hd, h)?;
        let v_proj = q(&weights.v_proj, nkv * hd, h)?;
        let o_proj = q(&weights.o_proj, h, nq * hd)?;

        // Dense MLP weights follow the same precision as the projections.
        let mlp_gate = q(&weights.mlp_gate, inter, h)?;
        let mlp_up = q(&weights.mlp_up, inter, h)?;
        let mlp_down = q(&weights.mlp_down, h, inter)?;

        // KV cache for the single MTP attention layer (FP8, same as MoE path).
        let kv_config = spark_runtime::kv_cache::KvCacheConfig {
            block_size: 16,
            num_kv_heads: nkv,
            head_dim: hd,
            num_layers: 1,
            dtype: spark_runtime::kv_cache::KvCacheDtype::Fp8,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
        };
        let mtp_num_blocks = max_seq_len / kv_config.block_size + 1;
        let kv_cache = PagedKvCache::new(kv_config, mtp_num_blocks, gpu)?;

        // Dummy moe_gate / shared_expert_gate to keep the MoE-tagged fields
        // non-null. They are never read when is_dense_mlp == true.
        let null_dense = DenseWeight {
            weight: spark_runtime::gpu::DevicePtr::NULL,
        };

        let effective_vocab = if mtp_vocab_size > 0 {
            (mtp_vocab_size as usize).min(config.vocab_size)
        } else {
            config.vocab_size
        };
        tracing::info!(
            "Dense MTP head: {quant:?}, fc=[{h},{h2}], attn Q=[{qd},{h}], MLP I={inter}, \
             vocab={ev}/{fv} (LM head {lm:.1} MB)",
            h2 = h * 2,
            qd = nq * hd * 2,
            ev = effective_vocab,
            fv = config.vocab_size,
            lm = (effective_vocab * h / 2) as f64 / (1024.0 * 1024.0),
        );

        Ok(Self {
            pre_fc_norm_embedding: weights.pre_fc_norm_embedding,
            pre_fc_norm_hidden: weights.pre_fc_norm_hidden,
            input_layernorm: weights.input_layernorm,
            post_attn_layernorm: weights.post_attn_layernorm,
            norm: weights.norm,
            fc,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm: weights.q_norm,
            k_norm: weights.k_norm,
            // MoE fields are unused in dense mode.
            moe_nvfp4: None,
            moe_experts_generic: None,
            moe_shared_generic: None,
            moe_gate: null_dense,
            shared_expert_gate: null_dense,
            // Dense MLP fields.
            is_dense_mlp: true,
            dense_mlp_gate: Some(mlp_gate),
            dense_mlp_up: Some(mlp_up),
            dense_mlp_down: Some(mlp_down),
            dense_mlp_intermediate: inter,
            quant,
            mtp_vocab_size,
            embed_tokens,
            lm_head_nvfp4,
            kv_cache: Mutex::new(kv_cache),
            attn_layer_idx: 0,
            fp8_calibration: if mtp_fp8_calib_enabled() {
                let cal = Fp8KvCalibration::new(MTP_FP8_WARMUP_TOKENS, gpu)?;
                tracing::info!(
                    "Dense MTP FP8 KV calibration: ENABLED (warmup={} tokens)",
                    MTP_FP8_WARMUP_TOKENS
                );
                Some(cal)
            } else {
                tracing::info!("Dense MTP FP8 KV calibration: disabled (ATLAS_MTP_FP8_CALIB=0)");
                None
            },
            rms_norm_k: gpu.kernel("norm", "rms_norm")?,
            rms_norm_residual_k: if config.use_fp32_residual() {
                gpu.kernel("norm", "rms_norm_residual_f32")
                    .or_else(|_| gpu.kernel("norm", "rms_norm_residual"))?
            } else {
                gpu.kernel("norm", "rms_norm_residual")?
            },
            w4a16_gemv_k: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
            w4a16_gemv_qg_k: gpu.kernel("w4a16_gemv", "w4a16_gemv_qg")?,
            w4a16_gemv_dual_k: gpu.kernel("w4a16_gemv_fused", "w4a16_gemv_dual")?,
            rope_k: gpu.kernel("rope", "rope_forward")?,
            reshape_cache_k: gpu.kernel("reshape_and_cache", "reshape_and_cache_flash_fp8")?,
            paged_decode_k: gpu.kernel("paged_decode_fp8", "paged_decode_attn_fp8")?,
            residual_add_k: if config.use_fp32_residual() {
                gpu.kernel("norm", "f32_residual_add")
                    .or_else(|_| gpu.kernel("residual_add", "bf16_residual_add"))?
            } else {
                gpu.kernel("residual_add", "bf16_residual_add")?
            },
            residual_add_rms_norm_k: if config.use_fp32_residual() {
                gpu.kernel("norm", "residual_add_rms_norm_f32")
                    .or_else(|_| gpu.kernel("norm", "residual_add_rms_norm"))?
            } else {
                gpu.kernel("norm", "residual_add_rms_norm")?
            },
            sigmoid_gate_mul_k: gpu.kernel("residual_add", "sigmoid_gate_mul")?,
            bf16_concat_k: gpu.kernel("residual_add", "bf16_concat")?,
            argmax_k: gpu.kernel("argmax", "argmax_bf16")?,
            embed_from_argmax_k: gpu.kernel("embed_from_argmax", "embed_from_argmax")?,
            draft_token_id_dev: gpu.alloc(4)?,
            // moe_silu_mul is the same kernel for dense fused silu*up — it
            // does not require expert routing.
            moe_silu_mul_k: Some(gpu.kernel("moe_silu_mul", "moe_silu_mul")?),
            // BF16/FP8 GEMV kernels back the non-NVFP4 arms of gemv().
            dense_gemv_k: gpu.kernel("gemv", "dense_gemv_bf16").ok(),
            dense_gemv_fp8w_k: gpu.kernel("gemv_fp8w", "dense_gemv_fp8w").ok(),
            deinterleave_qg_k: gpu.kernel("ssm_preprocess", "deinterleave_qg").ok(),
            w8a16_gemv_k: None,
            moe_topk_k: None,
            moe_weighted_sum_blend_k: None,
        })
    }

    /// Forward through the dense MLP block (replaces MoE step 10 in
    /// `forward_one`). Inputs: post-attn-norm activation `[1, hidden]`.
    /// Output written to a fresh buffer scratch slot, returned.
    ///
    /// Layout: gate_out [I] · up_out [I] → silu_act [I] → mlp_out [hidden]
    pub(crate) fn dense_mlp_forward(
        &self,
        normed: spark_runtime::gpu::DevicePtr,
        ctx: &crate::layer::ForwardContext,
        stream: u64,
    ) -> Result<spark_runtime::gpu::DevicePtr> {
        let h = ctx.config.hidden_size as u32;
        let inter = self.dense_mlp_intermediate as u32;
        let bf16 = 2usize;

        // Reuse moe_intermediate buffer slots: gate_buf at offset 0, up_buf at offset I*bf16
        // moe_output is the final [hidden] output buffer.
        let scratch = ctx.buffers.expert_up_out();
        let gate_buf = scratch;
        let up_buf = scratch.offset(self.dense_mlp_intermediate * bf16);
        let mlp_out = ctx.buffers.moe_output();

        let gate_w = self.dense_mlp_gate.as_ref().unwrap();
        let up_w = self.dense_mlp_up.as_ref().unwrap();
        let down_w = self.dense_mlp_down.as_ref().unwrap();

        // 1. gate = gemv(normed, gate_w) — dispatches on MtpQuantization.
        self.gemv(ctx.gpu, normed, gate_w, gate_buf, inter, h, stream)?;
        // 2. up = gemv(normed, up_w)
        self.gemv(ctx.gpu, normed, up_w, up_buf, inter, h, stream)?;
        // 3. silu_act = silu(gate) * up — moe_silu_mul does it elementwise.
        crate::layers::ops::moe_silu_mul(
            ctx.gpu,
            self.moe_silu_mul_k.unwrap(),
            gate_buf,
            up_buf,
            gate_buf,
            inter,
            stream,
        )?;
        // 4. mlp_out = gemv(silu_act, down_w)
        self.gemv(ctx.gpu, gate_buf, down_w, mlp_out, h, inter, stream)?;
        Ok(mlp_out)
    }
}
